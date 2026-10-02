from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import functools
from collections import OrderedDict

import nngen.basic_types as bt


class relu(bt._ActFuncOperator):
    """
    Applies the rectified linear unit function element-wise:

    .. math::
        {ReLU}(x) = \max(0, x)

    Args:
        features: A Tensor.
        dtype: Output data type (optional).
        name: A name for the operation (optional).
        par: The number of parallel operations (optional).
    """

    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.Mux(args[0] > strm.Int(0), args[0], strm.Int(0))

    def __init__(self, features, dtype=None, name=None, par=1):
        shape = None
        bt._ActFuncOperator.__init__(self, features,
                                     dtype=dtype, shape=shape, name=name, par=par)

    def get_eval_method(self):
        import nngen.verify as verify

        name = self.__class__.__name__
        method = getattr(verify, name, None)
        method = functools.partial(method,
                                   features_dtype=self.args[0].dtype)
        return method


class relu6(relu):
    """
    Applies the element-wise function:

    .. math::
        {ReLU6}(x) = \min(\max(0, x), 6)

    Args:
        features: A Tensor.
        dtype: Output data type (optional).
        name: A name for the operation (optional).
        par: The number of parallel operations (optional).
    """

    def get_max_val(self):
        # 6.0 in the integer domain of the input (round(6 * scale_factor)),
        # clamped to the maximum value of the input data type
        max_val = int(round(6.0 * self.args[0].scale_factor))
        dtype = self.args[0].dtype
        if dtype is not None:
            limit = (2 ** (dtype.width - 1) - 1) if dtype.signed else (2 ** dtype.width - 1)
            max_val = min(max_val, limit)
        return max_val

    def get_local_control_param_values(self):
        return OrderedDict([('max_val', self.get_max_val())])

    def op(self, strm, *args, **kwargs):
        return strm.Mux(args[0] > strm.Int(0),
                        strm.Mux(args[0] > self.max_val, self.max_val, args[0]),
                        strm.Int(0))

    def get_eval_method(self):
        import nngen.verify as verify

        name = self.__class__.__name__
        method = getattr(verify, name, None)
        method = functools.partial(method,
                                   features_dtype=self.args[0].dtype,
                                   features_scale_factor=self.get_max_val() / 6.0)
        return method
