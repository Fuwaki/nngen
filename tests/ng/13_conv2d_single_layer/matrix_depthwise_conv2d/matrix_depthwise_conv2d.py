from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import functools
import math
import numpy as np

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng

from veriloggen import *
import veriloggen.thread as vthread
import veriloggen.types.axi as axi


def run(act_shape=(1, 7, 7, 15), ksize=(3, 3),
        act_dtype=ng.int16, weight_dtype=ng.int16,
        bias_dtype=ng.int32, scale_dtype=ng.int16,
        out_dtype=ng.int16,
        stride=(1, 1, 1, 1), padding=(1, 1, 1, 1),
        with_bias=True, with_scale=True,
        rshift_out=4, act_func=None, act_scale_factor=1.0,
        par=1,
        axi_datawidth=32, silent=False,
        filename=None, simtype='iverilog', outputfile=None):

    # create target hardware
    act = ng.placeholder(act_dtype, shape=act_shape, name='act')
    weight = ng.variable(weight_dtype, shape=(ksize[0], ksize[1], act_shape[-1]), name='weight')
    bias = ng.variable(bias_dtype, (act_shape[-1],), name='bias') if with_bias else None
    scale = ng.variable(scale_dtype, (act_shape[-1],), name='scale') if with_scale else None

    out = ng.depthwise_conv2d(act, weight, stride, bias, scale, rshift_out,
                              act_func, padding, dtype=out_dtype, sum_dtype=ng.int32,
                              name='depthwise_conv2d', par=par)
    out.scale_factor = act_scale_factor  # used by relu6 (clamp = round(6 * scale_factor))

    targ = ng.to_veriloggen([out], 'matrix_depthwise_conv2d', silent=silent,
                            config={'maxi_datawidth': axi_datawidth})

    # verification data (signed values to exercise negative paths)
    vact = np.arange(act.length, dtype=np.int64).reshape(act.shape) % [11] - [3]
    vweight = np.arange(weight.length, dtype=np.int64).reshape(weight.shape) % [7] - [3]
    vbias = (np.arange(bias.length, dtype=np.int64).reshape(bias.shape) % [5] - [2]) * 8 if bias is not None else None
    vscale = np.arange(scale.length, dtype=np.int64).reshape(scale.shape) % [6] + [1] if scale is not None else None

    eval_outs = ng.eval([out], act=vact, weight=vweight, bias=vbias, scale=vscale)
    vout = eval_outs[0]

    # independent float reference (no rounding issues when values are exact)
    ref = naive_depthwise(vact, vweight, vbias, vscale, stride, padding, rshift_out,
                          act_func, out_dtype, out.get_act_max())
    if not np.array_equal(ref, vout):
        raise ValueError('ng.eval mismatch with naive reference')

    # to memory image
    size_max = int(math.ceil(max(act.memory_size, weight.memory_size,
                                 bias.memory_size if bias is not None else 0,
                                 scale.memory_size if scale is not None else 0,
                                 out.memory_size) / 4096)) * 4096
    check_addr = max(act.addr, weight.addr,
                     bias.addr if bias is not None else -1,
                     scale.addr if scale is not None else -1,
                     out.addr) + size_max
    size_check = size_max
    tmp_addr = check_addr + size_check

    memimg_datawidth = 32
    #mem = np.zeros([1024 * 1024 * 8 // (memimg_datawidth // 8)], dtype=np.int64)
    mem = np.zeros([1024 * 1024 * 128 // (memimg_datawidth // 8)], dtype=np.int64)
    mem = mem + [100]

    axi.set_memory(mem, vact, memimg_datawidth,
                   act_dtype.width, act.addr,
                   max(int(math.ceil(axi_datawidth / act_dtype.width)), par))

    axi.set_memory(mem, vweight, memimg_datawidth,
                   weight_dtype.width, weight.addr,
                   max(int(math.ceil(axi_datawidth / weight_dtype.width)), par))

    if bias is not None:
        axi.set_memory(mem, vbias, memimg_datawidth,
                       bias_dtype.width, bias.addr,
                       max(int(math.ceil(axi_datawidth / bias_dtype.width)), par))

    if scale is not None:
        axi.set_memory(mem, vscale, memimg_datawidth,
                       scale_dtype.width, scale.addr,
                       max(int(math.ceil(axi_datawidth / scale_dtype.width)), par))

    axi.set_memory(mem, vout, memimg_datawidth,
                   out_dtype.width, check_addr,
                   max(int(math.ceil(axi_datawidth / out_dtype.width)), par))

    # test controller
    m = Module('test')
    params = m.copy_params(targ)
    ports = m.copy_sim_ports(targ)
    clk = ports['CLK']
    resetn = ports['RESETN']
    rst = m.Wire('RST')
    rst.assign(Not(resetn))

    # AXI memory model
    if outputfile is None:
        outputfile = os.path.splitext(os.path.basename(__file__))[0] + '.out'

    memimg_name = 'memimg_' + outputfile

    memory = axi.AxiMemoryModel(m, 'memory', clk, rst,
                                datawidth=axi_datawidth,
                                memimg=mem, memimg_name=memimg_name,
                                memimg_datawidth=memimg_datawidth)
    memory.connect(ports, 'maxi')

    # AXI-Slave controller
    _saxi = vthread.AXIMLite(m, '_saxi', clk, rst, noio=True)
    _saxi.connect(ports, 'saxi')

    # timer
    time_counter = m.Reg('time_counter', 32, initval=0)
    seq = Seq(m, 'seq', clk, rst)
    seq(
        time_counter.inc()
    )

    def ctrl():
        for i in range(100):
            pass

        ng.sim.set_global_addrs(_saxi, tmp_addr)

        start_time = time_counter.value
        ng.sim.start(_saxi)

        print('# start')

        ng.sim.wait(_saxi)
        end_time = time_counter.value

        print('# end')
        print('# execution cycles: %d' % (end_time - start_time))

        # verify
        ok = True
        for bat in range(out.shape[0]):
            for y in range(out.shape[1]):
                for x in range(out.shape[2]):
                    for ch in range(out.shape[3]):
                        orig = memory.read_word(
                            bat * out.aligned_shape[1] * out.aligned_shape[2] * out.aligned_shape[3] +
                            y * out.aligned_shape[2] * out.aligned_shape[3] +
                            x * out.aligned_shape[3] + ch,
                            out.addr, out_dtype.width)
                        check = memory.read_word(
                            bat * out.aligned_shape[1] * out.aligned_shape[2] * out.aligned_shape[3] +
                            y * out.aligned_shape[2] * out.aligned_shape[3] +
                            x * out.aligned_shape[3] + ch,
                            check_addr, out_dtype.width)

                        if vthread.verilog.NotEql(orig, check):
                            print('NG (', bat, y, x, ch,
                                  ') orig: ', orig, ' check: ', check)
                            ok = False
                        # else:
                        #    print('OK (', bat, y, x, ch,
                        #          ') orig: ', orig, ' check: ', check)

        if ok:
            print('# verify: PASSED')
        else:
            print('# verify: FAILED')

        vthread.finish()

    th = vthread.Thread(m, 'th_ctrl', clk, rst, ctrl)
    fsm = th.start()

    uut = m.Instance(targ, 'uut',
                     params=m.connect_params(targ),
                     ports=m.connect_ports(targ))

    # simulation.setup_waveform(m, uut)
    simulation.setup_clock(m, clk, hperiod=5)
    init = simulation.setup_reset(m, resetn, m.make_reset(), period=100, polarity='low')

    init.add(
        Delay(10000000),
        Systask('finish'),
    )

    # output source code
    if filename is not None:
        m.to_verilog(filename)

    # run simulation
    sim = simulation.Simulator(m, sim=simtype)
    rslt = sim.run(outputfile=outputfile)

    return rslt




def naive_depthwise(x, w, b, s, stride, padding, rshift, act_func, out_dtype, act_max):
    import math as _m
    n, h, wd, c = x.shape
    kh, kw, _ = w.shape
    pt, pb, pl, pr = padding
    xp = np.pad(x, [(0, 0), (pt, pb), (pl, pr), (0, 0)])
    oh = (h + pt + pb - kh) // stride[1] + 1
    ow = (wd + pl + pr - kw) // stride[2] + 1
    out = np.zeros([n, oh, ow, c], dtype=np.int64)
    p_th = (1 << (out_dtype.width - 1)) - 1
    for bi in range(n):
        for oy in range(oh):
            for ox in range(ow):
                for ch in range(c):
                    acc = 0
                    for ky in range(kh):
                        for kx in range(kw):
                            acc += int(xp[bi, oy * stride[1] + ky, ox * stride[2] + kx, ch]) * int(w[ky, kx, ch])
                    if b is not None:
                        acc += int(b[ch])
                    if s is not None:
                        acc *= int(s[ch])
                    if rshift > 0:
                        # round half away from zero
                        acc = int(_m.floor(abs(acc) / (1 << rshift) + 0.5)) * (1 if acc >= 0 else -1)
                    acc = max(min(acc, p_th), -p_th)
                    if act_func is ng.relu:
                        acc = max(acc, 0)
                    elif act_func is ng.relu6:
                        acc = min(max(acc, 0), act_max)
                    out[bi, oy, ox, ch] = acc
    return out


if __name__ == '__main__':
    rslt = run(silent=False, filename='tmp.v')
    print(rslt)
