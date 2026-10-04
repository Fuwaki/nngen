"""
Efinix-oriented example: tiny encoder-decoder ("style transfer"-shaped) network.

  input : RGBX, int8 NHWC (1, H, W, 4), q = pixel >> 1  (x = pixel/256, scale factor 128),
          4th channel = 0  (exactly what the camera pre-processing writes to DDR)
  output: (1, H, W, 3) int8 NHWC, aligned to (1, H, W, 4) in memory -> RGBX with X = pad

  encoder : conv3x3 s2 (MobileNetV2 stem) + inverted residual blocks down to H/8
  decoder : nearest Upsample x2 + conv3x3 (+ additive skip from the encoder), 3 times
  head    : conv3x3 -> 3 channels, no activation

Flow: PyTorch (brief training on synthetic images, optional) -> ONNX (opset 13, legacy
exporter) -> nngen -> int8 quantization -> Verilog/IP-XACT -> C header + driver + params
image -> optional full RTL simulation (Verilator) compared bit-exactly with ng.eval.

Usage:
  python style_net_nngen.py --size 32 --par 1 --sim verilator
  python style_net_nngen.py --size 128 --par 1 --sim none        # IP + header only
  python style_net_nngen.py --size 128 --axi 16 --sim none
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import math
import time
import json
import argparse
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng  # noqa: E402
from veriloggen import *  # noqa: E402,F401,F403
import veriloggen.thread as vthread  # noqa: E402
import veriloggen.types.axi as axi  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--size', type=int, default=32, help='input/output H = W')
p.add_argument('--par', type=int, default=1, help='parallelism of conv/depthwise/upsample/add')
p.add_argument('--axi', type=int, default=32, help='AXI master data width (32 or 16)')
p.add_argument('--sim', default='verilator', help='verilator / iverilog / none')
p.add_argument('--steps', type=int, default=300, help='training steps (0 = random weights)')
p.add_argument('--out_ch', type=int, default=3)
p.add_argument('--outdir', default=None)
p.add_argument('--name', default='style_net')
p.add_argument('--seed', type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
H = W = args.size
name = args.name
chunk_size = 64
axi_datawidth = args.axi
act_dtype = ng.int8
weight_dtype = ng.int8
outdir = args.outdir or 'build_%d_par%d_axi%d' % (args.size, args.par, args.axi)
os.makedirs(outdir, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ model
def conv_bn(i, o, k, s, groups=1, act=True):
    layers = [nn.Conv2d(i, o, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(o)]
    if act:
        layers.append(nn.ReLU6())
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, i, o, s, t):
        super().__init__()
        h = i * t
        self.use_res = (s == 1 and i == o)
        self.conv = nn.Sequential(conv_bn(i, h, 1, 1), conv_bn(h, h, 3, s, groups=h),
                                  conv_bn(h, o, 1, 1, act=False))

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class StyleNet(nn.Module):
    def __init__(self, out_ch=3):
        super().__init__()
        self.stem = conv_bn(4, 16, 3, 2)                                      # H/2, 16
        self.enc2 = nn.Sequential(InvertedResidual(16, 24, 2, 4),
                                  InvertedResidual(24, 24, 1, 4))             # H/4, 24
        self.enc3 = nn.Sequential(InvertedResidual(24, 32, 2, 4),
                                  InvertedResidual(32, 32, 1, 4))             # H/8, 32
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec3 = conv_bn(32, 24, 3, 1)                                     # H/4
        self.dec2 = conv_bn(24, 16, 3, 1)                                     # H/2
        self.dec1 = conv_bn(16, 16, 3, 1)                                     # H
        self.head = nn.Conv2d(16, out_ch, 3, 1, 1)

    def forward(self, x):
        e1 = self.stem(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d = self.dec3(self.up(e3)) + e2
        d = self.dec2(self.up(d)) + e1
        d = self.dec1(self.up(d))
        return self.head(d)


def synth_images(n, h, w, rng):
    """smooth random 'natural-ish' RGB images in [0,1) + X=0 channel, NCHW."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    imgs = np.zeros((n, 4, h, w), np.float32)
    for k in range(n):
        img = np.zeros((3, h, w), np.float32) + rng.uniform(0, 1, (3, 1, 1))
        for _ in range(6):
            cy, cx = rng.uniform(0, h), rng.uniform(0, w)
            r = rng.uniform(0.05, 0.4) * h
            col = rng.uniform(-0.6, 0.6, (3, 1, 1))
            img += col * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
        img += rng.normal(0, 0.03, img.shape)
        imgs[k, :3] = np.clip(img, 0, 255.0 / 256)
    return imgs


def style_target(x):
    """a fixed 'style': sepia tone + unsharp mask, on RGB of NCHW input."""
    rgb = x[:, :3]
    m = torch.tensor([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
    sep = torch.einsum('oc,nchw->nohw', m, rgb)
    blur = F.avg_pool2d(sep, 5, 1, 2, count_include_pad=False)
    return torch.clamp(sep + 1.0 * (sep - blur), 0, 1)


model = StyleNet(args.out_ch)
print('# params: %d' % sum(q.numel() for q in model.parameters()))
rng = np.random.RandomState(args.seed)
ckpt = os.path.join(here, 'style_net_%d.pt' % args.out_ch)
if args.steps > 0:
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        print('# loaded', ckpt)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        t0 = time.time()
        for step in range(args.steps):
            x = torch.from_numpy(synth_images(16, 32, 32, rng))
            y = style_target(x)
            loss = F.mse_loss(model(x)[:, :3], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 == 0 or step == args.steps - 1:
                print('# train step %d loss %.5f' % (step, loss.item()))
        print('# training %.1fs' % (time.time() - t0))
        torch.save(model.state_dict(), ckpt)
model.eval()

# ------------------------------------------------------------------ ONNX -> nngen
onnx_filename = os.path.join(outdir, name + '.onnx')
torch.onnx.export(model, torch.zeros(1, 4, H, W), onnx_filename, input_names=['act'],
                  output_names=['out'], opset_version=13, dynamo=False)

(outputs, placeholders, variables, constants, operators) = ng.from_onnx(
    onnx_filename, value_dtypes={},
    default_placeholder_dtype=act_dtype, default_variable_dtype=weight_dtype,
    default_constant_dtype=weight_dtype, default_operator_dtype=act_dtype,
    default_scale_dtype=ng.int32, default_bias_dtype=ng.int32)
hist = collections.Counter(type(v).__name__ for v in {id(v): v for v in operators.values()}.values())
print('# op histogram:', dict(hist))

act_scale_factor = 128   # q = round(x * 128) with x = pixel / 256  ->  q = pixel >> 1


def calib_gen(node, num_samples):
    imgs = synth_images(node.shape[0] * num_samples, node.shape[1], node.shape[2], rng)
    q = np.clip(np.round(np.transpose(imgs, (0, 2, 3, 1)) * act_scale_factor), 0, 127)
    return q.astype(np.int64)


# image output: use ~90% of the int8 range for the last layer (default heuristic uses 33%)
outputs['out'].quant_range_rate = 0.9
ng.quantize(outputs, {'act': act_scale_factor}, input_generators={'act': calib_gen},
            num_samples=4)

for op in operators.values():
    if isinstance(op, ng.conv2d):
        op.attribute(par_ich=args.par, par_och=args.par)
    elif isinstance(op, (ng.depthwise_conv2d, ng.upsampling2d, ng.add, ng.scaled_add)):
        op.attribute(par=args.par)

act = placeholders['act']
out = outputs['out']
print('# act', act.shape, 'scale', act.scale_factor, '| out', out.shape, 'scale', out.scale_factor)

# ------------------------------------------------------------------ SW check
test_imgs = synth_images(4, H, W, np.random.RandomState(123))
corrs = []
for k in range(len(test_imgs)):
    mo = model(torch.from_numpy(test_imgs[k:k + 1])).detach().numpy()
    mo = np.transpose(mo, (0, 2, 3, 1))
    vin = np.clip(np.round(np.transpose(test_imgs[k:k + 1], (0, 2, 3, 1)) * act_scale_factor), 0, 127)
    vo = ng.eval([out], act=vin.astype(np.int64))[0]
    corrs.append(np.corrcoef(mo.reshape(-1), vo.reshape(-1) / out.scale_factor)[0, 1])
print('# float vs int8 (ng.eval) corrcoef over %d images: %s' % (len(corrs), ['%.4f' % c for c in corrs]))
vact = np.clip(np.round(np.transpose(test_imgs[0:1], (0, 2, 3, 1)) * act_scale_factor), 0, 127).astype(np.int64)
vout = ng.eval([out], act=vact)[0]

# ------------------------------------------------------------------ HDL + driver files
config = {'maxi_datawidth': axi_datawidth, 'offchipram_chunk_bytes': chunk_size}
t0 = time.time()
# generate ONCE (a second to_* call on the same graph sees already-assigned addresses);
# the IP-XACT package lands in <outdir>/<name>_v1_0, the plain Verilog is copied next to it
cwd = os.getcwd()
os.chdir(outdir)
import contextlib
import io
log = io.StringIO()
with contextlib.redirect_stdout(log):
    targ = ng.to_ipxact([out], name, silent=False, config=config)
os.chdir(cwd)
open(os.path.join(outdir, name + '_nngen_report.txt'), 'w').write(log.getvalue())
import shutil
shutil.copy(os.path.join(outdir, name + '_v1_0', 'hdl', name + '.v'), os.path.join(outdir, name + '.v'))
rtl = open(os.path.join(outdir, name + '.v')).read()
gen_time = time.time() - t0
info = ng.export_driver_files([out], name, outdir, config=config, params_c_array=True)
vsize = os.path.getsize(os.path.join(outdir, name + '.v'))
print('# generated %s.v: %d lines, %d bytes (%.1fs); params %d B; address space %d B'
      % (name, rtl.count('\n'), vsize, gen_time, info['param_size'], info['address_space_amount']))

summary = collections.OrderedDict(
    size=H, par=args.par, axi=axi_datawidth, verilog_lines=rtl.count('\n'), verilog_bytes=vsize,
    param_bytes=info['param_size'], address_space=info['address_space_amount'],
    out_scale_factor=float(out.scale_factor), corrcoef=[float(c) for c in corrs])
try:
    sys.path.insert(0, here)
    from ramest_efinix import estimate
    summary['ram'] = estimate(os.path.join(outdir, name + '.v'))
    print('# RAM estimate:', summary['ram'])
except Exception as e:  # pragma: no cover
    print('# RAM estimate failed:', e)

if args.sim == 'none':
    json.dump(summary, open(os.path.join(outdir, 'summary.json'), 'w'), indent=1)
    sys.exit(0)

# ------------------------------------------------------------------ RTL simulation
param_data = ng.export_ndarray([out], chunk_size)
regions = {r['kind'] if r['kind'] in ('temporal', 'params') else r['kind'] + ':' + r['name']: r
           for r in info['regions']}
act_addr = act.addr
out_addr = out.addr
param_addr = regions['params']['default_offset']
tmp_addr = regions['temporal']['default_offset']
check_addr = int(math.ceil((tmp_addr + regions['temporal']['size']) / chunk_size)) * chunk_size

memimg_datawidth = 32
mem_bytes = max(1 << 20, 1 << int(math.ceil(math.log2(check_addr + out.memory_size + 4096))))
mem = np.zeros([mem_bytes // (memimg_datawidth // 8)], dtype=np.int64) + [100]
align = max(int(math.ceil(axi_datawidth / act_dtype.width)), args.par)
axi.set_memory(mem, vact, memimg_datawidth, act_dtype.width, act_addr,
               max(int(math.ceil(axi_datawidth / act_dtype.width)), 1))
axi.set_memory(mem, param_data, memimg_datawidth, 8, param_addr)
axi.set_memory(mem, vout, memimg_datawidth, act_dtype.width, check_addr,
               max(int(math.ceil(axi_datawidth / act_dtype.width)), 1))

m = Module('test')
params = m.copy_params(targ)
ports = m.copy_sim_ports(targ)
clk = ports['CLK']
resetn = ports['RESETN']
rst = m.Wire('RST')
rst.assign(Not(resetn))
outputfile = os.path.join(outdir, '%s_%s.out' % (name, args.sim))
memory = axi.AxiMemoryModel(m, 'memory', clk, rst, datawidth=axi_datawidth, memimg=mem,
                            memimg_name=os.path.join(outdir, 'memimg_%s_%s.out' % (name, args.sim)),
                            memimg_datawidth=memimg_datawidth)
memory.connect(ports, 'maxi')
_saxi = vthread.AXIMLite(m, '_saxi', clk, rst, noio=True)
_saxi.connect(ports, 'saxi')
time_counter = m.Reg('time_counter', 32, initval=0)
seq = Seq(m, 'seq', clk, rst)
seq(time_counter.inc())
OH, OW, OC = out.shape[1], out.shape[2], out.shape[3]
AC = out.aligned_shape[3]


def ctrl():
    for i in range(100):
        pass
    # same sequence as nngen_driver.c: IER, start, wait busy, ack
    _saxi.write(ng.control_reg_interrupt_ier * 4, 1)
    start_time = time_counter.value
    ng.sim.start(_saxi)
    print('# start')
    ng.sim.wait(_saxi)
    end_time = time_counter.value
    print('# end')
    print('# execution cycles: %d' % (end_time - start_time))
    isr = _saxi.read(ng.control_reg_interrupt_isr * 4)
    print('# ISR after run: %d' % isr)
    _saxi.write(ng.control_reg_interrupt_iar * 4, 1)
    ng_count = 0
    for y in range(OH):
        for x in range(OW):
            for ch in range(OC):
                orig = memory.read_word((y * OW + x) * AC + ch, out_addr, act_dtype.width)
                check = memory.read_word((y * OW + x) * AC + ch, check_addr, act_dtype.width)
                if vthread.verilog.NotEql(orig, check):
                    if ng_count < 16:
                        print('NG (', y, x, ch, ') orig: ', orig, ' check: ', check)
                    ng_count += 1
    print('# mismatches: %d' % ng_count)
    if ng_count == 0:
        print('# verify: PASSED')
    else:
        print('# verify: FAILED')
    vthread.finish()


th = vthread.Thread(m, 'th_ctrl', clk, rst, ctrl)
fsm = th.start()
uut = m.Instance(targ, 'uut', params=m.connect_params(targ), ports=m.connect_ports(targ))
simulation.setup_clock(m, clk, hperiod=5)
init = simulation.setup_reset(m, resetn, m.make_reset(), period=100, polarity='low')
init.add(Delay(2000000000), Systask('finish'))
t0 = time.time()
sim = simulation.Simulator(m, sim=args.sim)
rslt = sim.run(outputfile=outputfile)
lines = [ln for ln in rslt.splitlines() if ln.startswith('#') or ln.startswith('NG')]
print('\n'.join(lines))
wall = time.time() - t0
print('# RTL simulation (%s) wall time: %.1fs' % (args.sim, wall))
for ln in lines:
    if ln.startswith('# execution cycles:'):
        summary['cycles'] = int(ln.split(':')[1])
    if ln.startswith('# verify:'):
        summary['verify'] = ln.split(':')[1].strip()
summary['sim_wall_s'] = wall
json.dump(summary, open(os.path.join(outdir, 'summary.json'), 'w'), indent=1)
