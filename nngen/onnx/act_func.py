from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import functools

import nngen.operator as operator

from . import util
from . import basic
from . import batchnormalization
from . import conv
from . import gemm


def _act_func(method, visitor, node, num_inputs=None):

    node_name = util.get_name(node)

    src_name = node.input[0]
    src_node = util.search_node_from_model(visitor.model, src_name)

    if (not visitor.disable_fusion and
        src_node is not None and
        src_node.op_type == 'BatchNormalization' and
            len(visitor.consumers[src_name]) == 1):

        src_op = batchnormalization.BatchNormalization(visitor, src_node,
                                                       act_func=method)
        visitor.operators[node_name] = src_op
        return src_op

    if (not visitor.disable_fusion and
        src_node is not None and
            src_node.op_type == 'Conv' and len(visitor.consumers[src_name]) == 1):

        src_op = conv.Conv(visitor, src_node, act_func=method)
        visitor.operators[node_name] = src_op
        return src_op

    if (not visitor.disable_fusion and
        src_node is not None and
            src_node.op_type == 'Gemm' and len(visitor.consumers[src_name]) == 1):

        src_op = gemm.Gemm(visitor, src_node, act_func=method)
        visitor.operators[node_name] = src_op
        return src_op

    if num_inputs is not None:
        args = [visitor.visit(src) for src in list(node.input)[:num_inputs]]
        kwargs = {'name': node_name}
        if node_name in visitor.value_dtypes:
            kwargs['dtype'] = visitor.value_dtypes[node_name]
        return method(*args, **kwargs)

    return basic._elementwise(method, visitor, node)


def Relu(visitor, node):
    return _act_func(operator.relu, visitor, node)


def Clip(visitor, node):
    """ Clip(x, 0, 6) -> relu6, Clip(x, 0, +inf) -> relu """
    import numpy as np

    min_v = None
    max_v = None
    for attribute in node.attribute:
        if attribute.name == 'min':
            min_v = attribute.f
        elif attribute.name == 'max':
            max_v = attribute.f

    # opset >= 11: min/max are (optional) inputs
    for i, key in ((1, 'min'), (2, 'max')):
        if len(node.input) > i and node.input[i]:
            obj = visitor.visit(node.input[i])
            value = getattr(obj, 'value', None)
            if value is None:
                raise NotImplementedError('Clip with non-constant %s is not supported.' % key)
            value = float(np.array(value).reshape([-1])[0])
            if key == 'min':
                min_v = value
            else:
                max_v = value

    if min_v is None:
        min_v = -np.inf
    if max_v is None:
        max_v = np.inf

    if min_v == 0.0 and max_v == 6.0:
        return _act_func(operator.relu6, visitor, node, num_inputs=1)

    if min_v == 0.0 and np.isinf(max_v):
        return _act_func(operator.relu, visitor, node, num_inputs=1)

    raise NotImplementedError('Clip(min=%s, max=%s) is not supported: '
                              'only (0, 6) as ReLU6 and (0, inf) as ReLU.' % (min_v, max_v))


def LeakyRelu(visitor, node):
    alpha = 0.01
    for attribute in node.attribute:
        if attribute.name == 'alpha':
            alpha = attribute.f

    rshift = 31
    slope = round(alpha * (2 ** 31))
    op = operator.get_leaky_relu_op(slope, rshift)
    return _act_func(op, visitor, node)


def Sigmoid(visitor, node):
    return _act_func(operator.sigmoid, visitor, node)
