from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import numpy as np

from . import util


def depthwise_conv2d(visitor, node):
    """
    Quantizer for nngen.depthwise_conv2d.

    Depthwise filters have very different ranges per channel, so when a
    per-channel scale exists, every channel of the filter is normalized by its
    max-abs value and the factor is moved into the (per-channel) scale.
    Then filter / bias / scale are linearly quantized and rshift_out is searched
    so that the (pre-activation) output does not overflow.
    """
    from nngen.operator.relu import relu, relu6

    threshold_norm_filter = 1e-3

    input = node.args[0]
    filter = node.args[1]
    bias = node.args[2] if node.has_bias else None
    scale = node.args[node.scale_index] if node.has_scale else None

    visitor.visit(input)
    visitor.visit(filter)
    if bias is not None:
        visitor.visit(bias)
    if scale is not None:
        visitor.visit(scale)

    def as_array(v):
        if isinstance(v, (tuple, list)):
            return np.array(v)
        return v

    # per-channel normalization (filter: (Kh, Kw, C))
    if scale is not None:
        ch_max = np.abs(filter.value).reshape([-1, filter.shape[-1]]).max(axis=0)
        ch_max = np.clip(ch_max, threshold_norm_filter, None)
        filter.value = filter.value / ch_max
        scale.value = as_array(scale.value) * ch_max
        if bias is not None:
            bias.value = as_array(bias.value) / ch_max

    q_filter_value, filter_scale_factor = util.quantize_linear(filter.value, filter.dtype.width)
    filter.set_value(q_filter_value)
    filter.scale_factor = filter_scale_factor

    if bias is not None:
        bias_sf = input.scale_factor * filter_scale_factor
        q_bias_value = util.quantize_linear_by_scale_factor(
            as_array(bias.value), bias.dtype.width, bias_sf)
        bias.set_value(q_bias_value)
        bias.scale_factor = bias_sf
    else:
        q_bias_value = None

    if scale is not None:
        q_scale_value, scale_scale_factor = util.quantize_linear(
            as_array(scale.value), scale.dtype.width)
        scale.set_value(q_scale_value)
        scale.scale_factor = scale_scale_factor
    else:
        q_scale_value = None
        scale_scale_factor = 1.0

    if node.cshamt_out is None:
        init_rshift = max(math.ceil(math.log(max(np.mean(np.abs(q_filter_value)), 1) * 2.0, 2)), 0)
        if scale is not None:
            init_rshift += max(math.ceil(
                math.log(max(np.mean(np.abs(q_scale_value)), 1) * 2.0, 2)), 0)
        rshift_out = find_optimal_rshift(visitor, node, q_filter_value, q_bias_value,
                                         q_scale_value, init_rshift)
        if node.act_func is relu6:
            # values above 6.0 are clamped anyway: it is enough that 6.0 fits in the range
            base_sf = input.scale_factor * filter_scale_factor * scale_scale_factor
            p_th = 2 ** (node.dtype.width - 1) - 1 if node.dtype.signed else 2 ** node.dtype.width - 1
            r_six = max(int(math.ceil(math.log(max(6.0 * base_sf / p_th, 1e-30), 2))), 0)
            rshift_out = min(rshift_out, r_six)
        node.cshamt_out = rshift_out if rshift_out > 0 else None
    total_rshift = node.cshamt_out if node.cshamt_out is not None else 0

    node.scale_factor = (input.scale_factor * filter_scale_factor *
                         scale_scale_factor / (2 ** total_rshift))

    # final result with the real activation (relu6 clamp depends on scale_factor)
    visitor.memo.pop(id(node), None)
    visitor.memo[id(node)] = node.eval(visitor.memo, visitor.input_dict)


def find_optimal_rshift(visitor, node, filter, bias, scale, init_rshift,
                        allowed_rate=0.0, range_rate=0.33):
    import nngen.verify as verify
    from nngen.operator.relu import relu6

    input = node.args[0].eval(visitor.memo, visitor.input_dict)

    if node.dtype.signed:
        _range = round((2 ** (node.dtype.width - 1)) * range_rate)
    else:
        _range = round((2 ** node.dtype.width) * range_rate)

    # relu6 clamp is not known yet: search on relu output
    act_func = node.act_func
    if act_func is relu6:
        from nngen.operator.relu import relu
        act_func = relu

    rshift = init_rshift
    while True:
        rslt = verify.depthwise_conv2d(input, filter, node.strides, bias, scale, rshift,
                                       act_func, node.padding, node.asymmetric_clip,
                                       node.dtype, node.sum_dtype, node.name, node.par)
        num_overflow = np.sum((rslt <= -_range) | (rslt >= _range))
        if num_overflow / rslt.size <= allowed_rate:
            break
        rshift += 1

    return rshift
