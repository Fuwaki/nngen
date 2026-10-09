from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import numpy as np

from . import util


def conv2d(visitor, node):

    if node.act_func is not None and not hasattr(node, 'visited_before_act_func'):
        node.visited_before_act_func = True
        visitor.visit(node.act_func)
        node.scale_factor = node.act_func.scale_factor
        return

    threshold_norm_filter = 0.2

    input = node.args[0]
    filter = node.args[1]

    bias = node.args[node.args_dict['bias']] if node.has_bias else None
    scale = node.args[node.args_dict['scale']] if node.has_scale else None

    rshift_mul = (node.args[node.args_dict['vshamt_mul']]
                  if node.has_vshamt_mul else node.cshamt_mul)
    rshift_sum = (node.args[node.args_dict['vshamt_sum']]
                  if node.has_vshamt_sum else node.cshamt_sum)
    rshift_out = (node.args[node.args_dict['vshamt_out']]
                  if node.has_vshamt_out else node.cshamt_out)

    visitor.visit(input)
    visitor.visit(filter)

    if bias is not None:
        visitor.visit(bias)

    if scale is not None:
        visitor.visit(scale)

    if rshift_mul is not None:
        visitor.visit(rshift_mul)

    if rshift_sum is not None:
        visitor.visit(rshift_sum)

    if rshift_out is not None:
        visitor.visit(rshift_out)

    # normalize filter vaules for each output channel
    if (scale is not None and scale.shape[-1] == node.shape[-1] and
        scale.dtype.width >= filter.dtype.width * 4 and
            (bias is None or bias.shape[-1] == node.shape[-1])):

        out_scale_value = np.abs(filter.value)
        for _ in range(len(filter.value.shape) - 1):
            out_scale_value = out_scale_value.max(axis=1)

        out_scale_value = np.clip(out_scale_value, threshold_norm_filter, None)
        filter.value = (filter.value /
                        out_scale_value.reshape([-1] + [1] * (len(filter.value.shape) - 1)))

        scale_value = scale.value
        if isinstance(scale_value, (tuple, list)):
            scale_value = np.array(scale_value)
        scale.value = scale_value * out_scale_value

        if bias is not None:
            bias_value = bias.value
            if isinstance(bias_value, (tuple, list)):
                bias_value = np.array(bias_value)
            bias.value = bias_value / out_scale_value

    q_filter_value, filter_scale_factor = util.quantize_linear(filter.value, filter.dtype.width)
    filter.set_value(q_filter_value)
    filter.scale_factor = filter_scale_factor

    if bias is not None:
        bias_value = bias.value
        if isinstance(bias_value, (tuple, list)):
            bias_value = np.array(bias_value)

        q_bias_value = util.quantize_linear_by_scale_factor(
            bias_value, bias.dtype.width, input.scale_factor * filter_scale_factor)
        bias.set_value(q_bias_value)
        bias.scale_factor = input.scale_factor * filter_scale_factor
    else:
        q_bias_value = None

    if scale is not None:
        scale_value = scale.value
        if isinstance(scale_value, (tuple, list)):
            scale_value = np.array(scale_value)

        q_scale_value, scale_scale_factor = util.quantize_linear(scale_value, scale.dtype.width)
        scale.set_value(q_scale_value)
        scale.scale_factor = scale_scale_factor
    else:
        scale_scale_factor = 1.0
        q_scale_value = None

    if ((rshift_mul is None or isinstance(rshift_mul, int)) and
        (rshift_sum is None or isinstance(rshift_sum, int)) and
            (rshift_out is None or isinstance(rshift_out, int))):

        init_rshift_out = max(math.ceil(math.log(np.mean(np.abs(q_filter_value)) * 2.0, 2)), 0)
        if scale is not None:
            init_rshift_out += max(math.ceil(math.log(np.mean(np.abs(q_scale_value)) * 2.0, 2)), 0)

        q_rshift_mul, q_rshift_sum, q_rshift_out = find_optimal_rshift(
            visitor, node, q_filter_value, q_bias_value, q_scale_value,
            init_rshift_mul=0,
            init_rshift_sum=0,
            init_rshift_out=init_rshift_out)

        if _is_relu6(node.act_func):
            # values above 6.0 are clamped anyway: it is enough that 6.0 fits in the range
            base_sf = input.scale_factor * filter_scale_factor * scale_scale_factor
            for v in (node.cshamt_mul, node.cshamt_sum, node.cshamt_out):
                if v is not None:
                    base_sf /= 2 ** v
            base_sf /= 2 ** (q_rshift_mul + q_rshift_sum)
            p_th = (2 ** (node.dtype.width - 1) - 1 if node.dtype.signed
                    else 2 ** node.dtype.width - 1)
            r_six = max(int(math.ceil(math.log(max(6.0 * base_sf / p_th, 1e-30), 2))), 0)
            q_rshift_out = min(q_rshift_out, r_six)

        total_rshift = 0

        if node.cshamt_mul is not None:
            node.cshamt_mul += q_rshift_mul
            total_rshift += node.cshamt_mul
        elif q_rshift_mul > 0:
            node.cshamt_mul = q_rshift_mul
            total_rshift += node.cshamt_mul

        if node.cshamt_sum is not None:
            node.cshamt_sum += q_rshift_sum
            total_rshift += node.cshamt_sum
        elif q_rshift_sum > 0:
            node.cshamt_sum = q_rshift_sum
            total_rshift += node.cshamt_sum

        if node.cshamt_out is not None:
            node.cshamt_out += q_rshift_out
            total_rshift += node.cshamt_out
        elif q_rshift_out > 0:
            node.cshamt_out = q_rshift_out
            total_rshift += node.cshamt_out

        node.scale_factor = (input.scale_factor * filter_scale_factor *
                             scale_scale_factor / (2 ** total_rshift))

        if _is_relu6(node.act_func):
            # final result with the real relu6 clamp (depends on node.scale_factor)
            visitor.memo[id(node)] = try_rshift(node, input.eval(visitor.memo, visitor.input_dict),
                                                q_filter_value, q_bias_value, q_scale_value,
                                                node.cshamt_mul or 0, node.cshamt_sum or 0,
                                                node.cshamt_out or 0)

    else:
        node.scale_factor = (input.scale_factor * filter_scale_factor *
                             scale_scale_factor)


def find_optimal_rshift(visitor, node, filter, bias, scale,
                        allowed_rate=0.0, range_rate=0.33,
                        init_rshift_mul=0, init_rshift_sum=0, init_rshift_out=0):

    rshift_mul = init_rshift_mul
    rshift_sum = init_rshift_sum
    rshift_out = init_rshift_out

    input = node.args[0].eval(visitor.memo, visitor.input_dict)

    # per-operator override, e.g. op.quant_range_rate = 0.9 for an image output layer
    range_rate = getattr(node, 'quant_range_rate', None) or range_rate
    allowed_rate = getattr(node, 'quant_allowed_rate', None) or allowed_rate

    if node.dtype.signed:
        _range = round((2 ** (node.dtype.width - 1)) * range_rate)
    else:
        _range = round((2 ** node.dtype.width) * range_rate)

    # The relu6 clamp (round(6 * scale_factor)) depends on the scale factor that is being
    # searched here; node.scale_factor is stale at this point. Search on plain relu output.
    act_func_override = _ReluProxy() if _is_relu6(node.act_func) else None

    while True:
        rslt = try_rshift(node, input, filter, bias, scale,
                          rshift_mul, rshift_sum, rshift_out,
                          act_func_override=act_func_override)
        neg_overflow = np.where(rslt <= - _range,
                                np.ones_like(rslt), np.zeros_like(rslt))
        pos_overflow = np.where(rslt >= _range,
                                np.ones_like(rslt), np.zeros_like(rslt))
        num_overflow = np.sum(neg_overflow + pos_overflow)

        rate = num_overflow / rslt.size
        if rate <= allowed_rate:
            break

        rshift_out += 1

    visitor.memo[id(node)] = rslt

    return rshift_mul, rshift_sum, rshift_out


class _ReluProxy(object):
    def get_act_func(self):
        return lambda x: np.maximum(x, 0)


def _is_relu6(act_func):
    from nngen.operator.relu import relu6
    return act_func is not None and isinstance(act_func, relu6)


def try_rshift(node, input, filter, bias, scale,
               rshift_mul, rshift_sum, rshift_out, act_func_override=None):

    import nngen.verify as verify

    name = node.__class__.__name__
    method = getattr(verify, name, None)

    strides = node.strides

    kwargs = {}
    kwargs['strides'] = strides
    kwargs['bias'] = bias
    kwargs['scale'] = scale
    kwargs['rshift_mul'] = rshift_mul
    kwargs['rshift_sum'] = rshift_sum
    kwargs['rshift_out'] = rshift_out
    kwargs['act_func'] = node.act_func if act_func_override is None else act_func_override
    kwargs['padding'] = node.padding
    kwargs['dtype'] = node.dtype
    kwargs['mul_dtype'] = node.mul_dtype
    kwargs['sum_dtype'] = node.sum_dtype
    kwargs['name'] = node.name
    kwargs['par_ich'] = node.par_ich
    kwargs['par_och'] = node.par_och
    kwargs['par_col'] = node.par_col
    kwargs['par_row'] = node.par_row
    kwargs['concur_och'] = node.concur_och
    kwargs['stationary'] = node.stationary

    if 'matmul' in method.__name__:
        del kwargs['strides']
        del kwargs['padding']
        del kwargs['par_ich']
        del kwargs['par_och']
        del kwargs['par_col']
        del kwargs['par_row']
        del kwargs['concur_och']

    return method(input, filter, **kwargs)
