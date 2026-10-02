from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.util as util


def depthwise_conv2d(input, filter, strides,
                     bias=None, scale=None, rshift_out=None,
                     act_func=None, padding='SAME', asymmetric_clip=False,
                     dtype=None, sum_dtype=None, name=None, par=1,
                     value_ram_size=None, out_ram_size=None,
                     act_max=None):
    """
    Integer reference model of nngen.depthwise_conv2d (bit-exact with the hardware).
    input: (N, H, W, C), filter: (Kh, Kw, C), bias/scale: (C,) or (1,) or None.
    Padding follows the pooling operators' convention (same control logic in hardware).
    """
    from nngen.operator.relu import relu, relu6

    input = np.array(input, dtype=np.int64)
    filter = np.array(filter, dtype=np.int64)
    kh, kw, ch = filter.shape
    sh, sw = strides[1], strides[2]
    in_h, in_w = input.shape[1], input.shape[2]

    if isinstance(padding, str) and padding == 'SAME':
        _, pt, pb = util.pad_size_split(in_h, kh, sh)
        _, pl, pr = util.pad_size_split(in_w, kw, sw)
    elif isinstance(padding, str) and padding == 'VALID':
        pt = pb = pl = pr = 0
    elif isinstance(padding, int):
        pt = pb = pl = pr = padding
    elif isinstance(padding, (tuple, list)):
        pt, pb, pl, pr = padding
    else:
        raise ValueError("padding options must be 'SAME', 'VALID', int, tuple, or list.")

    out_h = util.pix_size(in_h + pt + pb, kh, sh, 'VALID')
    out_w = util.pix_size(in_w + pl + pr, kw, sw, 'VALID')

    x = np.pad(input, [(0, 0), (pt, pb), (pl, pr), (0, 0)], 'constant')

    acc = np.zeros([input.shape[0], out_h, out_w, ch], dtype=np.int64)
    for ky in range(kh):
        for kx in range(kw):
            win = x[:, ky: ky + sh * (out_h - 1) + 1: sh, kx: kx + sw * (out_w - 1) + 1: sw, :]
            acc += win * filter[ky, kx, :]

    sum_width = 32 if sum_dtype is None else sum_dtype.width

    def wrap(v, width):
        if width >= 64:
            return v
        mask = (1 << width) - 1
        v = np.bitwise_and(v, mask)
        return np.where(v >= (1 << (width - 1)), v - (1 << width), v)

    acc = wrap(acc, sum_width)

    if bias is not None:
        acc = wrap(acc + np.array(bias, dtype=np.int64).reshape([-1]), sum_width)

    if scale is not None:
        acc = acc * np.array(scale, dtype=np.int64).reshape([-1])

    r = 0 if rshift_out is None else int(rshift_out)
    if r > 0:
        rnd = 1 << (r - 1)
        acc = np.where(acc >= 0, acc + rnd, acc + rnd - 1)
        acc = np.right_shift(acc, r)

    out_width = 32 if dtype is None else dtype.width
    out_signed = True if dtype is None else dtype.signed
    p_th, n_th = util.clip_threshold(out_width, out_signed, asymmetric_clip)
    acc = np.where(acc > p_th, p_th, np.where(acc < n_th, n_th, acc))

    if act_func is relu:
        acc = np.where(acc > 0, acc, 0)
    elif act_func is relu6:
        if act_max is None:
            raise ValueError('act_max is required for relu6.')
        acc = np.where(acc > 0, np.where(acc > act_max, act_max, acc), 0)

    return acc.astype(np.int64)
