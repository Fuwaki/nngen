"""
Bare-metal C driver support for NNgen-generated accelerators.

After a model has been converted with ``ng.to_veriloggen`` / ``ng.to_verilog`` /
``ng.to_ipxact`` (which assigns DDR addresses to all tensors), call

    ng.export_c_header([out], 'mymodel', filename='mymodel_nngen.h', config=cfg)

to emit a model-specific header describing

  * the AXI-Lite control register map (byte offsets),
  * every address register (temporal / outputs / inputs / unified parameters),
    with default offsets, sizes, shapes, aligned shapes and scale factors,
  * the parameter image layout produced by ``ng.export_ndarray``.

``ng.export_driver_files`` additionally writes the model-agnostic driver
(``nngen_driver.h`` / ``nngen_driver.c``) and the parameter image
(``<name>_params.bin`` and optionally ``<name>_params.h``).

Address semantics (see verilog.make_controls): the DMA address of a region is
``GLOBAL_OFFSET register + region address register``.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import re
import math
import datetime

import nngen.basic_types as bt
import nngen.storage as st
from . import verilog
from . import util
from .version import __version__

__all__ = ['export_c_header', 'export_driver_files', 'collect_address_map']


def _c_ident(s):
    s = re.sub(r'[^0-9A-Za-z_]', '_', str(s))
    s = re.sub(r'_+', '_', s).strip('_')
    if not s or s[0].isdigit():
        s = 'T_' + s
    return s.upper()


def _resolve(obj):
    while bt.is_view(obj) or bt.is_removable_reshape(obj):
        obj = obj.args[0]
    return obj


def _space(obj, chunk_size):
    return verilog.align_space(obj.dtype.width, obj.get_aligned_length(), chunk_size)


def _tensor_info(orig, obj, chunk_size):
    shape = tuple(orig.shape) if isinstance(orig.shape, (tuple, list)) else ()
    aligned = (tuple(orig.get_aligned_shape())
               if isinstance(orig.shape, (tuple, list)) else ())
    layout = orig.get_layout() if hasattr(orig, 'get_layout') else None
    return {
        'name': orig.name,
        'dtype_width': orig.dtype.width,
        'dtype_signed': orig.dtype.signed,
        'dtype_str': orig.dtype.to_str(),
        'shape': shape,
        'aligned_shape': aligned,
        'layout': tuple(layout) if layout is not None else None,
        'word_alignment': orig.get_word_alignment(),
        'scale_factor': float(getattr(orig, 'scale_factor', 1.0) or 1.0),
        'default_offset': obj.default_global_addr,
        'size': _space(obj, chunk_size),
        'global_index': obj.global_index,
    }


def collect_address_map(objs, config=None, chunk_size=None):
    """Collect the address map of already-generated objects as a dict."""

    if not isinstance(objs, (list, tuple)):
        objs = [objs]

    config = verilog.load_default_config(config)
    if chunk_size is None:
        chunk_size = config['offchipram_chunk_bytes']

    if config['use_map_ram']:
        raise NotImplementedError("use_map_ram=True is not supported by the C header generator.")

    numerics = util._collect_numerics(objs)

    outputs = []
    for o in objs:
        r = _resolve(o)
        if r.global_index is None:
            raise ValueError("'%s' has no address. Run ng.to_veriloggen/to_verilog/to_ipxact first."
                             % str(o.name))
        outputs.append(_tensor_info(o, r, chunk_size))

    output_ids = set(id(_resolve(o)) for o in objs)

    inputs = []
    params = []
    temporal_end = 0
    temporal_base = None
    for obj in numerics:
        if obj.global_index is None or id(obj) in output_ids:
            continue
        if bt.is_view(obj) or bt.is_removable_reshape(obj):
            continue
        if bt.is_input_storage(obj):
            inputs.append(_tensor_info(obj, obj, chunk_size))
        elif bt.is_storage(obj):
            info = _tensor_info(obj, obj, chunk_size)
            info['image_offset'] = obj.default_local_addr
            info['kind'] = 'variable' if isinstance(obj, st.variable) else 'constant'
            params.append(info)
        elif obj.global_index == 0:
            temporal_base = obj.default_global_addr
            temporal_end = max(temporal_end,
                               obj.default_local_addr + _space(obj, chunk_size))

    params.sort(key=lambda x: x['image_offset'])
    param_index = params[0]['global_index'] if params else None
    param_default = params[0]['default_offset'] if params else None
    param_size = len(util.export_ndarray(objs, chunk_size)) if params else 0

    # temporal: register index 0; default address follows all storages
    if temporal_base is None:
        ends = [t['default_offset'] + t['size'] for t in outputs + inputs]
        if params:
            ends.append(param_default + param_size)
        temporal_base = max(ends) if ends else 0

    regions = [{'kind': 'temporal', 'name': 'temporal', 'global_index': 0,
                'default_offset': temporal_base, 'size': temporal_end}]
    for t in outputs:
        regions.append(dict(t, kind='output'))
    for t in inputs:
        regions.append(dict(t, kind='input'))
    if params:
        regions.append({'kind': 'params', 'name': 'params', 'global_index': param_index,
                        'default_offset': param_default, 'size': param_size})

    for r in regions:
        r['reg'] = verilog.index_to_bytes(verilog.control_reg_global_addr + r['global_index'])

    amount = max([r['default_offset'] + r['size'] for r in regions] + [0])

    return {
        'config': config,
        'chunk_size': chunk_size,
        'regions': sorted(regions, key=lambda r: r['global_index']),
        'outputs': outputs,
        'inputs': inputs,
        'params': params,
        'param_size': param_size,
        'address_space_amount': amount,
    }


def _hex(v):
    return '0x%08xu' % v


def export_c_header(objs, name, filename=None, config=None, chunk_size=None,
                    prefix=None):
    """Generate a C header for an NNgen accelerator. Returns the header text."""

    info = collect_address_map(objs, config, chunk_size)
    config = info['config']
    P = _c_ident(prefix if prefix is not None else name)
    g = 'NNGEN_%s_H' % P
    b = verilog.index_to_bytes

    L = []
    w = L.append
    w('/*')
    w(' * %s_nngen.h -- generated by NNgen %s (nngen/c_header.py), %s' %
      (name, __version__, datetime.date.today().isoformat()))
    w(' * DO NOT EDIT: regenerate together with the Verilog of the same model.')
    w(' *')
    w(' * AXI master : data %d bit, addr %d bit, no ID, max burst %d beats, INCR' %
      (config['maxi_datawidth'], config['maxi_addrwidth'], verilog.max_burst_length))
    w(' * AXI-Lite   : data %d bit (32-bit word registers, byte offsets below)' %
      config['saxi_datawidth'])
    w(' * IRQ        : %s' % ('level-high "%s", ISR bit0 = busy falling edge' %
                             config['interrupt_name'] if config['interrupt_enable'] else 'disabled'))
    w(' * DMA address = REG_GLOBAL_OFFSET + region address register.')
    w(' */')
    w('#ifndef %s' % g)
    w('#define %s' % g)
    w('')
    w('#include <stdint.h>')
    w('')
    w('/* ---- control registers (byte offsets from the AXI-Lite base) ---- */')
    regs = [
        ('HEADER0', b(0)), ('HEADER1', b(1)), ('HEADER2', b(2)), ('HEADER3', b(3)),
        ('START', b(verilog.control_reg_start)),
        ('BUSY', b(verilog.control_reg_busy)),
        ('RESET', b(verilog.control_reg_reset)),
        ('EXTERN_SEND', b(verilog.control_reg_extern_send)),
        ('EXTERN_RECV', b(verilog.control_reg_extern_recv)),
    ]
    if config['interrupt_enable']:
        regs += [('ISR', b(verilog.control_reg_interrupt_isr)),
                 ('IER', b(verilog.control_reg_interrupt_ier)),
                 ('IAR', b(verilog.control_reg_interrupt_iar))]
    if config['measurable_main_fsm']:
        regs += [('COUNT', b(verilog.control_reg_count)),
                 ('COUNT_STATE', b(verilog.control_reg_count_state)),
                 ('COUNT_DIV', b(verilog.control_reg_count_div))]
    regs += [('ADDRESS_AMOUNT', b(verilog.control_reg_address_amount)),
             ('GLOBAL_OFFSET', b(verilog.control_reg_global_offset))]
    for n, v in regs:
        w('#define %s_REG_%-16s 0x%03xu' % (P, n, v))
    w('#define %s_HAS_IRQ               %d' % (P, 1 if config['interrupt_enable'] else 0))
    w('#define %s_ISR_BUSY_BIT          %d' % (P, verilog.control_reg_interrupt_isr_busy))
    w('')
    w('/* ---- build configuration ---- */')
    w('#define %s_MAXI_DATAWIDTH        %d' % (P, config['maxi_datawidth']))
    w('#define %s_CHUNK_BYTES           %d' % (P, info['chunk_size']))
    w('#define %s_DEFAULT_GLOBAL_OFFSET %s' % (P, _hex(config['default_global_addr_offset'])))
    w('#define %s_ADDRESS_SPACE_AMOUNT  %du  /* bytes, from default memory map */' %
      (P, info['address_space_amount']))
    w('#define %s_NUM_ADDR_REGS         %d' % (P, len(info['regions'])))
    w('')
    w('/* ---- address registers: one per region (offsets relative to GLOBAL_OFFSET) ---- */')
    used = set()
    for r in info['regions']:
        if r['kind'] in ('temporal', 'params'):
            tag = r['kind'].upper()
        else:
            base = '%s_%s' % ('OUT' if r['kind'] == 'output' else 'IN', _c_ident(r['name']))
            tag = base
            k = 1
            while tag in used:
                tag = '%s_%d' % (base, k)
                k += 1
        used.add(tag)
        r['tag'] = tag
        w('/* %s: %s */' % (r['kind'], r['name']))
        w('#define %s_%s_REG            0x%03xu' % (P, tag, r['reg']))
        w('#define %s_%s_DEFAULT_OFFSET %s' % (P, tag, _hex(r['default_offset'])))
        w('#define %s_%s_SIZE           %du' % (P, tag, r['size']))
        if r['kind'] in ('output', 'input'):
            shp = list(r['shape'])
            ash = list(r['aligned_shape'])
            dims = ['N', 'H', 'W', 'C'] if len(shp) == 4 else ['D%d' % i for i in range(len(shp))]
            for d, s, a in zip(dims, shp, ash):
                w('#define %s_%s_%s              %d' % (P, tag, d, s))
            w('#define %s_%s_ALIGNED_%s      %d  /* innermost dim padded to word alignment */' %
              (P, tag, dims[-1], ash[-1]))
            w('#define %s_%s_DTYPE_BITS     %d  /* %s */' % (P, tag, r['dtype_width'], r['dtype_str']))
            w('#define %s_%s_SCALE_FACTOR   %.9gf  /* real = int / SCALE_FACTOR */' %
              (P, tag, r['scale_factor']))
            if r['layout'] is not None:
                w('/* layout: %s, element (n,h,w,c) at byte offset ((n*H+h)*W+w)*ALIGNED_C+c (int8) */'
                  % ''.join(r['layout']))
        w('')
    w('/* ---- parameter image (ng.export_ndarray, chunk %d B): %d bytes, %d tensors ---- */' %
      (info['chunk_size'], info['param_size'], len(info['params'])))
    w('#define %s_PARAM_IMAGE_SIZE      %du' % (P, info['param_size']))
    for p in info['params']:
        w('/*   +0x%06x %6d B %-6s %-24s %s */' %
          (p['image_offset'], p['size'], p['dtype_str'], str(p['shape']), p['name']))
    w('')
    w('/* ---- region table for nngen_driver.c ---- */')
    w('#ifdef NNGEN_DRIVER_H')
    w('static const nngen_region_t %s_regions[] = {' % P.lower())
    for r in info['regions']:
        w('  { 0x%03xu, 0x%08xu, %du, "%s" },' % (r['reg'], r['default_offset'], r['size'],
                                                r['tag'].lower()))
    w('};')
    w('static const nngen_model_t %s_model = {' % P.lower())
    w('  "%s", %s_regions, %d, %du, %du' % (name, P.lower(), len(info['regions']),
                                           info['param_size'], info['address_space_amount']))
    w('};')
    w('#endif')
    w('')
    w('#endif /* %s */' % g)

    text = '\n'.join(L) + '\n'
    if filename is not None:
        with open(filename, 'w') as f:
            f.write(text)
    return text


DRIVER_H = r'''/*
 * nngen_driver.h -- minimal model-agnostic bare-metal driver for NNgen accelerators.
 * Generated by NNgen (nngen/c_header.py). Include this BEFORE <model>_nngen.h to get
 * the region table (<model>_model).
 *
 * The accelerator is an AXI-Lite slave (32-bit registers) plus an AXI4 master that
 * reads/writes DDR. DMA address of every region = GLOBAL_OFFSET + region register.
 * The driver never touches DDR itself: loading the parameter image and the input
 * tensor, and reading the output tensor, is done by the caller (CPU DDR window,
 * DMA, or a mailbox). With a data cache, flush/invalidate around nngen_start().
 */
#ifndef NNGEN_DRIVER_H
#define NNGEN_DRIVER_H

#include <stdint.h>

#define NNGEN_REG_START          0x010u
#define NNGEN_REG_BUSY           0x014u
#define NNGEN_REG_RESET          0x018u
#define NNGEN_REG_ISR            0x024u
#define NNGEN_REG_IER            0x028u
#define NNGEN_REG_IAR            0x02cu
#define NNGEN_REG_COUNT          0x030u
#define NNGEN_REG_COUNT_STATE    0x034u
#define NNGEN_REG_COUNT_DIV      0x038u
#define NNGEN_REG_ADDRESS_AMOUNT 0x07cu
#define NNGEN_REG_GLOBAL_OFFSET  0x080u
#define NNGEN_ISR_BUSY           0x1u

typedef struct {
  uint32_t reg;            /* byte offset of the address register */
  uint32_t default_offset; /* default region offset (relative to GLOBAL_OFFSET) */
  uint32_t size;           /* region size in bytes */
  const char *name;        /* "temporal", "params", "out_<name>", "in_<name>" */
} nngen_region_t;

typedef struct {
  const char *name;
  const nngen_region_t *regions;
  uint32_t num_regions;
  uint32_t param_image_size;
  uint32_t address_space_amount;
} nngen_model_t;

typedef struct {
  uintptr_t base;          /* CPU address of the AXI-Lite register window */
  uint32_t global_offset;  /* DDR base of the model's default memory map */
} nngen_dev_t;

static inline void nngen_write(const nngen_dev_t *d, uint32_t off, uint32_t v)
{ *(volatile uint32_t *)(d->base + off) = v; }
static inline uint32_t nngen_read(const nngen_dev_t *d, uint32_t off)
{ return *(volatile uint32_t *)(d->base + off); }

/* Init: soft reset, program GLOBAL_OFFSET and all address registers with defaults. */
int  nngen_init(nngen_dev_t *d, uintptr_t reg_base, uint32_t ddr_global_offset,
                const nngen_model_t *model);
/* Region address as a DDR address (absolute); must be >= global_offset. */
int  nngen_set_region_addr(const nngen_dev_t *d, uint32_t addr_reg, uint32_t ddr_addr);
uint32_t nngen_get_region_addr(const nngen_dev_t *d, uint32_t addr_reg);
/* Default absolute DDR address of a region of the model. */
uint32_t nngen_default_region_addr(const nngen_dev_t *d, const nngen_region_t *r);
/* Start one inference; waits until the start request has been accepted. */
void nngen_start(const nngen_dev_t *d);
static inline int nngen_busy(const nngen_dev_t *d) { return nngen_read(d, NNGEN_REG_BUSY) != 0; }
/* Poll BUSY until idle. Returns 0, or -1 on timeout (timeout=0: wait forever). */
int  nngen_wait(const nngen_dev_t *d, uint32_t timeout_loops);
/* Interrupt helpers (irq output is level-high while ISR & IER != 0). */
static inline void nngen_irq_enable(const nngen_dev_t *d, int en)
{ nngen_write(d, NNGEN_REG_IER, en ? NNGEN_ISR_BUSY : 0); }
static inline uint32_t nngen_irq_status(const nngen_dev_t *d)
{ return nngen_read(d, NNGEN_REG_ISR); }
static inline void nngen_irq_ack(const nngen_dev_t *d, uint32_t bits)
{ nngen_write(d, NNGEN_REG_IAR, bits); }
/* Soft reset of the internal logic (waits until AXI master is idle). */
void nngen_soft_reset(const nngen_dev_t *d);
/* Cycle counter of main-FSM state `state` (see nngen "measurable_main_fsm"). */
static inline void nngen_count_setup(const nngen_dev_t *d, uint32_t state, uint32_t div)
{ nngen_write(d, NNGEN_REG_COUNT_STATE, state); nngen_write(d, NNGEN_REG_COUNT_DIV, div); }
static inline uint32_t nngen_count(const nngen_dev_t *d) { return nngen_read(d, NNGEN_REG_COUNT); }

#endif /* NNGEN_DRIVER_H */
'''

DRIVER_C = r'''/* nngen_driver.c -- see nngen_driver.h. Generated by NNgen (nngen/c_header.py). */
#include "nngen_driver.h"

void nngen_soft_reset(const nngen_dev_t *d)
{
  nngen_write(d, NNGEN_REG_RESET, 1);
  while (nngen_read(d, NNGEN_REG_BUSY) != 0) { }
}

int nngen_init(nngen_dev_t *d, uintptr_t reg_base, uint32_t ddr_global_offset,
               const nngen_model_t *model)
{
  uint32_t i;
  d->base = reg_base;
  d->global_offset = ddr_global_offset;
  nngen_soft_reset(d);
  nngen_write(d, NNGEN_REG_GLOBAL_OFFSET, ddr_global_offset);
  if (model) {
    for (i = 0; i < model->num_regions; i++)
      nngen_write(d, model->regions[i].reg, model->regions[i].default_offset);
    if (nngen_read(d, NNGEN_REG_ADDRESS_AMOUNT) != model->address_space_amount)
      return -1; /* header and bitstream do not match */
  }
  nngen_irq_ack(d, 0xffffffffu);
  return 0;
}

int nngen_set_region_addr(const nngen_dev_t *d, uint32_t addr_reg, uint32_t ddr_addr)
{
  if (ddr_addr < d->global_offset) return -1;
  nngen_write(d, addr_reg, ddr_addr - d->global_offset);
  return 0;
}

uint32_t nngen_get_region_addr(const nngen_dev_t *d, uint32_t addr_reg)
{
  return d->global_offset + nngen_read(d, addr_reg);
}

uint32_t nngen_default_region_addr(const nngen_dev_t *d, const nngen_region_t *r)
{
  return d->global_offset + r->default_offset;
}

void nngen_start(const nngen_dev_t *d)
{
  nngen_write(d, NNGEN_REG_START, 1);
  /* START reads back non-zero until the main FSM has accepted the request */
  while (nngen_read(d, NNGEN_REG_START) != 0) { }
}

int nngen_wait(const nngen_dev_t *d, uint32_t timeout_loops)
{
  while (nngen_read(d, NNGEN_REG_BUSY) != 0) {
    if (timeout_loops) {
      if (--timeout_loops == 0) return -1;
    }
  }
  return 0;
}
'''


def export_driver_files(objs, name, outdir='.', config=None, chunk_size=None,
                        params_c_array=False):
    """Write <name>_nngen.h, nngen_driver.h/.c and <name>_params.bin into outdir."""

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    info = collect_address_map(objs, config, chunk_size)
    header = export_c_header(objs, name, os.path.join(outdir, '%s_nngen.h' % name),
                             config, info['chunk_size'])
    with open(os.path.join(outdir, 'nngen_driver.h'), 'w') as f:
        f.write(DRIVER_H)
    with open(os.path.join(outdir, 'nngen_driver.c'), 'w') as f:
        f.write(DRIVER_C)
    param = util.export_ndarray(objs, info['chunk_size'])
    param.astype('uint8').tofile(os.path.join(outdir, '%s_params.bin' % name))
    if params_c_array:
        P = _c_ident(name).lower()
        with open(os.path.join(outdir, '%s_params.h' % name), 'w') as f:
            f.write('/* parameter image of %s (%d bytes), load to GLOBAL_OFFSET + PARAMS offset */\n'
                    % (name, len(param)))
            f.write('#include <stdint.h>\n')
            f.write('static const uint8_t %s_params[%d] __attribute__((aligned(16))) = {\n'
                    % (P, max(len(param), 1)))
            for i in range(0, len(param), 16):
                f.write('  ' + ', '.join('0x%02x' % int(v) for v in param[i:i + 16]) + ',\n')
            f.write('};\n')
    return info
