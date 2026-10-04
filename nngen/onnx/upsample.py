from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import collections
import numpy as np

import nngen.operator as operator

from . import util


def Upsample(visitor, node):

    mode = 'nearest'
    scale_value = 1.0

    for attribute in node.attribute:
        if attribute.name == 'mode':
            mode = attribute.s.decode()

        # deprecated attribute since Upsample-9
        if attribute.name == 'scale':
            scale_value = attribute.f

    if mode != 'nearest':
        raise ValueError("Upsampling mode must be 'nearest', not '%s'." % mode)

    if round(scale_value) != scale_value:
        raise ValueError("Upsampling factor must be a multiple of integer, not %f." %
                         scale_value)

    scale_value = round(scale_value)

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    srcs = [util.optimize_to_raw_value(src) for src in srcs]

    input = srcs[0]

    if len(input.shape) != 4:
        raise ValueError("not supported shape: %s" % str(tuple(input.shape)))

    # transpose data layout to nngen-compatible format
    input = util.transpose_layout(input, visitor.nngen_input_layout, visitor.onnx_input_layout)

    name = util.get_name(node)

    if len(srcs) > 1:
        factors = srcs[1]
        if not isinstance(factors, (np.ndarray, np.floating, np.integer, float, int)):
            raise TypeError("Upsampling factor must be constant, not %s." % str(type(factors)))

        if not isinstance(factors, np.ndarray):
            factors = np.array(factors)

        if np.not_equal(np.round(factors), factors).any():
            raise ValueError("Upsampling factor must be a multiple of integer, not %s." %
                             str(factors))

        if len(factors) == 4:
            factors = [int(round(factors[visitor.onnx_input_layout.index(l)]))
                       for l in visitor.nngen_input_layout]
        else:
            factors = np.array([int(round(factors.reshape([-1])[0]))] * 4)

    else:
        factors = np.array([scale_value] * 4)

    if name in visitor.value_dtypes:
        dtype = visitor.value_dtypes[name]
    else:
        dtype = visitor.default_operator_dtype

    kwargs = collections.OrderedDict()
    kwargs['factors'] = factors
    kwargs['dtype'] = dtype
    kwargs['name'] = name

    c = operator.upsampling2d(input, **kwargs)
    c.layout = visitor.nngen_input_layout
    c.onnx_layout = visitor.onnx_input_layout

    return c


def Resize(visitor, node):
    """ONNX Resize (opset 10/11/13/18+) restricted to nearest-neighbor with
    integer scale factors, which is mapped to nngen.upsampling2d.

    Supported input forms:
      opset 10:  (X, scales)
      opset 11+: (X, roi, scales[, sizes]); empty names ('') are skipped.
    """

    mode = 'nearest'
    coord_mode = 'half_pixel'
    nearest_mode = 'round_prefer_floor'

    for attribute in node.attribute:
        if attribute.name == 'mode':
            mode = attribute.s.decode()
        if attribute.name == 'coordinate_transformation_mode':
            coord_mode = attribute.s.decode()
        if attribute.name == 'nearest_mode':
            nearest_mode = attribute.s.decode()
        if attribute.name in ('antialias', 'exclude_outside') and attribute.i != 0:
            raise ValueError("Resize: '%s' is not supported." % attribute.name)
        if attribute.name == 'axes':
            raise ValueError("Resize: 'axes' attribute is not supported.")

    if mode != 'nearest':
        raise ValueError("Resize mode must be 'nearest', not '%s'." % mode)

    # For integer up-scaling, these combinations all reduce to pixel replication
    # (out[y] = in[y // s]).
    ok_modes = {
        'asymmetric': ('floor', 'round_prefer_floor', 'round_prefer_ceil'),
        'half_pixel': ('floor', 'round_prefer_floor', 'round_prefer_ceil'),
        'pytorch_half_pixel': ('floor', 'round_prefer_floor', 'round_prefer_ceil'),
        'tf_half_pixel_for_nn': ('floor', 'round_prefer_floor', 'round_prefer_ceil'),
    }
    if coord_mode not in ok_modes or nearest_mode not in ok_modes[coord_mode]:
        raise ValueError("Resize: unsupported combination coordinate_transformation_mode='%s', "
                         "nearest_mode='%s'." % (coord_mode, nearest_mode))

    input = util.optimize_to_raw_value(visitor.visit(node.input[0]))

    if len(input.shape) != 4:
        raise ValueError("not supported shape: %s" % str(tuple(input.shape)))

    def get_const(index):
        if len(node.input) <= index or node.input[index] == '':
            return None
        v = util.optimize_to_raw_value(visitor.visit(node.input[index]))
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        if v.size == 0:
            return None
        return v

    if len(node.input) == 2:  # opset 10: (X, scales)
        scales = get_const(1)
        sizes = None
    else:
        scales = get_const(2)
        sizes = get_const(3)

    # shape of input in ONNX layout
    if input.get_layout() is not None and input.get_onnx_layout() is not None:
        layout = input.get_layout()
        onnx_layout = input.get_onnx_layout()
        onnx_shape = [input.shape[layout.index(l)] for l in onnx_layout]
    else:
        onnx_shape = list(input.shape)

    if scales is not None:
        factors_onnx = [float(f) for f in scales.reshape([-1])]
    elif sizes is not None:
        factors_onnx = [float(o) / float(i) for o, i in zip(sizes.reshape([-1]), onnx_shape)]
    else:
        raise ValueError("Resize: either 'scales' or 'sizes' must be given as a constant.")

    if len(factors_onnx) != 4:
        raise ValueError("Resize: number of scale factors must be 4, not %d." % len(factors_onnx))

    for f in factors_onnx:
        if f < 1.0 or round(f) != f:
            raise ValueError("Resize: scale factors must be positive integers (up-sampling), "
                             "not %s." % str(factors_onnx))

    # transpose data layout to nngen-compatible format
    input = util.transpose_layout(input, visitor.nngen_input_layout, visitor.onnx_input_layout)

    factors = [int(round(factors_onnx[visitor.onnx_input_layout.index(l)]))
               for l in visitor.nngen_input_layout]

    name = util.get_name(node)

    if name in visitor.value_dtypes:
        dtype = visitor.value_dtypes[name]
    else:
        dtype = visitor.default_operator_dtype

    kwargs = collections.OrderedDict()
    kwargs['factors'] = factors
    kwargs['dtype'] = dtype
    kwargs['name'] = name

    c = operator.upsampling2d(input, **kwargs)
    c.layout = visitor.nngen_input_layout
    c.onnx_layout = visitor.onnx_input_layout

    return c
