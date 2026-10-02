from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import collections

import nngen.storage as storage
import nngen.operator as operator
import nngen.dtype_list as dtype_list

from . import util


def Conv(visitor, node,
         batchnorm_scale=None, batchnorm_bias=None, act_func=None):

    # input, filter
    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    input = srcs[0]
    filter = srcs[1]

    group = 1
    for attribute in node.attribute:
        if attribute.name == 'group':
            group = attribute.i

    if group != 1:
        num_ch = filter.shape[0]
        if group == num_ch and filter.shape[1] == 1:
            return _DepthwiseConv(visitor, node, srcs,
                                  batchnorm_scale, batchnorm_bias, act_func)
        raise NotImplementedError(
            "Conv with group=%d (filter shape %s) is not supported: "
            "only group=1 and depthwise (group == channels, multiplier 1) are supported." %
            (group, str(tuple(filter.shape))))

    # transpose data layout to nngen-compatible format
    input = util.transpose_layout(input, visitor.nngen_input_layout, visitor.onnx_input_layout)
    filter = util.transpose_layout(filter, visitor.nngen_filter_layout, visitor.onnx_filter_layout)

    bias = srcs[2] if len(srcs) > 2 else None

    name = util.get_name(node)

    scale_name = '_'.join(['onnx', name, 'conv.scale'])
    scale_dtype = visitor.default_scale_dtype
    scale_shape = batchnorm_scale.shape if batchnorm_scale is not None else (1,)
    scale = storage.variable(dtype=scale_dtype, shape=scale_shape, name=scale_name)
    scale_value = batchnorm_scale if batchnorm_scale is not None else [1]
    scale.set_value(scale_value)
    visitor.variables[scale_name] = scale

    if bias is None and batchnorm_bias is not None:
        bias_name = '_'.join(['onnx', name, 'conv.bias'])
        bias_dtype = visitor.default_bias_dtype
        bias_shape = batchnorm_bias.shape
        bias = storage.variable(dtype=bias_dtype, shape=bias_shape, name=bias_name)
        bias_value = batchnorm_bias / batchnorm_scale
        bias.set_value(bias_value)
        visitor.variables[bias_name] = bias

    elif bias is not None and batchnorm_bias is not None:
        bias.dtype = visitor.default_bias_dtype
        bias_value = batchnorm_bias / batchnorm_scale + bias.value
        bias.set_value(bias_value)

    elif bias is not None:
        bias.dtype = visitor.default_bias_dtype

    #rshift_out_name = '_'.join(['onnx', name, 'conv.rshift_out'])
    #rshift_out_width = filter.dtype.width
    #rshift_out_dtype = dtype_list.dtype_int(rshift_out_width, signed=False)
    #rshift_out_shape = (1,)
    #rshift_out = storage.variable(dtype=scale_dtype, shape=scale_shape, name=rshift_out_name)
    #visitor.variables[rshift_out_name] = rshift_out
    rshift_out = 0

    if name in visitor.value_dtypes:
        dtype = visitor.value_dtypes[name]
    else:
        dtype = visitor.default_operator_dtype

    if dtype.width >= 16:
        sum_dtype = dtype_list.dtype_int(dtype.width * 4)
    else:
        sum_dtype = dtype_list.int32

    strides = [1, 1, 1, 1]  # B, H, W, C
    padding = [0, 0, 0, 0]  # Top, Bottom, Left, Right

    for attribute in node.attribute:
        if attribute.name == 'auto_pad':
            padding = 'SAME'

        elif attribute.name == 'pads':
            padding[0] = attribute.ints[0]
            padding[1] = attribute.ints[1]
            padding[2] = attribute.ints[2]
            padding[3] = attribute.ints[3]
            padding = tuple(padding)

        elif attribute.name == 'strides':
            strides[1] = attribute.ints[0]
            strides[2] = attribute.ints[1]
            strides = tuple(strides)

    args = [input, filter]

    kwargs = collections.OrderedDict()
    kwargs['strides'] = strides
    kwargs['bias'] = bias
    kwargs['scale'] = scale
    kwargs['rshift_out'] = rshift_out
    kwargs['act_func'] = act_func
    kwargs['padding'] = padding
    kwargs['dtype'] = dtype
    kwargs['sum_dtype'] = sum_dtype
    kwargs['name'] = name

    c = operator.conv2d(*args, **kwargs)
    c.layout = visitor.nngen_input_layout
    c.onnx_layout = visitor.onnx_input_layout

    return c


def _DepthwiseConv(visitor, node, srcs,
                   batchnorm_scale=None, batchnorm_bias=None, act_func=None):
    """ Conv(group == C, filter (C, 1, Kh, Kw)) -> nngen.depthwise_conv2d """

    import numpy as np

    input = srcs[0]
    filter = srcs[1]
    bias = srcs[2] if len(srcs) > 2 else None

    if filter.value is None:
        raise NotImplementedError('depthwise Conv requires a constant filter.')

    if act_func is not None and act_func not in (operator.relu, operator.relu6):
        raise NotImplementedError('depthwise Conv supports only relu/relu6 fusion.')

    input = util.transpose_layout(input, visitor.nngen_input_layout, visitor.onnx_input_layout)

    name = util.get_name(node)
    num_ch = filter.shape[0]
    kh, kw = filter.shape[2], filter.shape[3]

    # (C, 1, Kh, Kw) -> (Kh, Kw, C)
    fvalue = np.transpose(np.reshape(np.array(filter.value), (num_ch, kh, kw)), (1, 2, 0))
    filter_name = '_'.join(['onnx', name, 'dwconv.filter'])
    dw_filter = storage.variable(dtype=filter.dtype, shape=fvalue.shape, name=filter_name)
    dw_filter.set_value(fvalue)
    visitor.variables[filter_name] = dw_filter

    scale_name = '_'.join(['onnx', name, 'dwconv.scale'])
    scale = storage.variable(dtype=visitor.default_scale_dtype, shape=(num_ch,), name=scale_name)
    if batchnorm_scale is not None:
        scale_value = np.broadcast_to(np.array(batchnorm_scale, dtype=np.float64),
                                      (num_ch,)).copy()
    else:
        scale_value = np.ones([num_ch], dtype=np.float64)
    scale.set_value(scale_value)
    visitor.variables[scale_name] = scale

    bias_value = None
    if bias is not None:
        bias_value = np.reshape(np.array(bias.value, dtype=np.float64), (-1,))
    if batchnorm_bias is not None:
        bn_bias = np.array(batchnorm_bias, dtype=np.float64) / scale_value
        bias_value = bn_bias if bias_value is None else bias_value + bn_bias

    dw_bias = None
    if bias_value is not None:
        bias_name = '_'.join(['onnx', name, 'dwconv.bias'])
        dw_bias = storage.variable(dtype=visitor.default_bias_dtype, shape=(num_ch,),
                                   name=bias_name)
        dw_bias.set_value(bias_value)
        visitor.variables[bias_name] = dw_bias

    if name in visitor.value_dtypes:
        dtype = visitor.value_dtypes[name]
    else:
        dtype = visitor.default_operator_dtype

    if dtype.width >= 16:
        sum_dtype = dtype_list.dtype_int(dtype.width * 4)
    else:
        sum_dtype = dtype_list.int32

    strides = [1, 1, 1, 1]
    padding = (0, 0, 0, 0)  # Top, Bottom, Left, Right
    for attribute in node.attribute:
        if attribute.name == 'auto_pad':
            v = attribute.s.decode() if isinstance(attribute.s, bytes) else attribute.s
            if v in ('SAME_UPPER', 'SAME_LOWER'):
                padding = 'SAME'
            elif v == 'VALID':
                padding = 'VALID'
        elif attribute.name == 'pads':
            # ONNX: [top, left, bottom, right]
            p = attribute.ints
            padding = (p[0], p[2], p[1], p[3])
        elif attribute.name == 'strides':
            strides[1] = attribute.ints[0]
            strides[2] = attribute.ints[1]
        elif attribute.name == 'dilations':
            if any(d != 1 for d in attribute.ints):
                raise NotImplementedError('dilated depthwise Conv is not supported.')
    strides = tuple(strides)

    c = operator.depthwise_conv2d(input, dw_filter, strides,
                                  bias=dw_bias, scale=scale, rshift_out=None,
                                  act_func=act_func, padding=padding,
                                  dtype=dtype, sum_dtype=sum_dtype, name=name)
    c.layout = visitor.nngen_input_layout
    c.onnx_layout = visitor.onnx_input_layout

    return c
