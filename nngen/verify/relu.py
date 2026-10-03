from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def relu(features, dtype=None, name=None, par=1,
         features_dtype=None):

    features_point = 0 if features_dtype is None else features_dtype.point
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - features_point

    zeros = np.zeros_like(features, dtype=np.int64)
    comp = features >= 0

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(np.where(comp, features, zeros))

    return ret


def relu6(features, dtype=None, name=None, par=1,
          features_dtype=None, features_scale_factor=None):

    if features_scale_factor is None:
        raise ValueError('relu6 requires features_scale_factor to adjust the clipping range.')

    features_point = 0 if features_dtype is None else features_dtype.point
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - features_point

    zeros = np.zeros_like(features, dtype=np.int64)
    comp0 = features >= 0
    max_val = int(round(features_scale_factor * 6))
    if features_dtype is not None:
        limit = ((2 ** (features_dtype.width - 1) - 1) if features_dtype.signed else
                 (2 ** features_dtype.width - 1))
        max_val = min(max_val, limit)
    sixs = np.zeros_like(features, dtype=np.int64) + [max_val]
    comp6 = features > max_val

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(np.where(comp0, np.where(comp6, sixs, features), zeros))

    return ret
