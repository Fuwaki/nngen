from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
from collections import OrderedDict

import veriloggen as vg

import nngen.basic_types as bt
import nngen.util as util

from . import pool
from .relu import relu, relu6


class depthwise_conv2d(pool._pool):
    """
    Depthwise 2D convolution (= grouped convolution with groups == channels).

    out[n, y, x, c] = act( clip( round_shift( (sum_{ky,kx} in[n, y*sy+ky, x*sx+kx, c] * filter[ky, kx, c]
                                              + bias[c]) * scale[c], rshift_out ) ) )

    The hardware reuses the sliding-window datapath/control of the pooling operators
    (each channel is processed independently, channels are streamed with "par" lanes),
    and adds per-tap filter RAMs plus bias/scale RAMs that are loaded once by DMA.

    Args:
        input: 4D tensor (N, H, W, C)
        filter: 3D tensor (Kh, Kw, C)
        strides: (1, stride_h, stride_w, 1)
        bias: (C,) tensor (optional)
        scale: (C,) tensor (optional)
        rshift_out: int, right shift amount after scaling (rounded)
        act_func: None, nngen.relu or nngen.relu6
        padding: 'SAME', 'VALID', int or (top, bottom, left, right)
        sum_dtype: dtype of the accumulator (default int32)
        par: number of parallel channel lanes (power of 2)
    """

    input_chainable = False
    output_chainable = False

    def __sub_str__(self):
        base = pool._pool.__sub_str__(self)
        bias = ' bias:%s' % str(self.args[2].shape) if self.has_bias else ''
        scale = ' scale:%s' % str(self.args[self.scale_index].shape) if self.has_scale else ''
        cshamt_out = ' cshamt_out:%d' % self.cshamt_out if self.cshamt_out is not None else ''
        act = (' act_func:%s' % self.act_func.__name__) if self.act_func is not None else ''
        sum_dtype = ' sum_dtype:%s' % self.sum_dtype.to_str() if self.sum_dtype is not None else ''
        return ''.join([base, bias, scale, cshamt_out, act, sum_dtype])

    def __init__(self, input, filter, strides,
                 bias=None, scale=None, rshift_out=None,
                 act_func=None, padding='SAME', asymmetric_clip=False,
                 dtype=None, sum_dtype=None, name=None, par=1,
                 value_ram_size=None, out_ram_size=None):

        if bt.get_rank(input.shape) != 4:
            raise ValueError('rank of input must be 4.')
        if bt.get_rank(filter.shape) != 3:
            raise ValueError('rank of filter must be 3: (Kh, Kw, C)')
        if filter.shape[-1] != input.shape[-1]:
            raise ValueError('filter.shape[-1] (%d) must be same as input channels (%d)' %
                             (filter.shape[-1], input.shape[-1]))
        num_ch = input.shape[-1]
        if bias is not None and tuple(bias.shape) != (num_ch,):
            raise ValueError('shape of bias must be (C,)')
        if scale is not None and tuple(scale.shape) != (num_ch,):
            raise ValueError('shape of scale must be (C,)')
        if rshift_out is not None and not isinstance(rshift_out, int):
            raise TypeError('rshift_out must be int or None.')
        if act_func is not None and act_func not in (relu, relu6):
            raise ValueError('act_func must be None, relu, or relu6.')

        ksize = (1, filter.shape[0], filter.shape[1], 1)

        pool._pool.__init__(self, input, ksize, strides, padding,
                            dtype, name, par, value_ram_size, out_ram_size)

        # append parameter arguments after the activation
        self.has_bias = bias is not None
        self.has_scale = scale is not None
        args = [input, filter]
        if self.has_bias:
            args.append(bias)
        self.scale_index = len(args)
        if self.has_scale:
            args.append(scale)
        self.args = tuple(args)
        for arg in self.args[1:]:
            arg.add_consumer(self)
            arg.output_chainable = False
            arg.chain_head = False
        for arg in self.args:
            arg.add_alignment_request(self.par)

        self.cshamt_out = rshift_out
        self.act_func = act_func
        self.asymmetric_clip = asymmetric_clip
        self.sum_dtype = sum_dtype

    def attribute(self, par=None, value_ram_size=None, out_ram_size=None):
        pool._pool.attribute(self, par, value_ram_size, out_ram_size)

    # ------------------------------------------------------------
    def get_num_taps(self):
        return self.ksize[-2] * self.ksize[-3]

    def get_sum_dtype(self):
        import nngen.dtype_list as dtype_list
        if self.sum_dtype is not None:
            return self.sum_dtype
        return dtype_list.int32

    def get_act_max(self):
        """ upper clamp value for relu6 in the output integer domain """
        p_th, _ = util.clip_threshold(self.dtype.width, self.dtype.signed, self.asymmetric_clip)
        if self.act_func is relu6:
            return int(min(round(6.0 * self.scale_factor), p_th))
        return p_th

    def get_required_rams(self):
        inputs, outputs, temps = pool._pool.get_required_rams(self)

        num_taps = self.get_num_taps()
        num_ch = self.args[0].get_aligned_shape()[-1]
        param_len = int(math.ceil(num_ch / self.par))

        filter = self.args[1]
        inputs.extend([(filter.get_ram_width() * self.par, param_len)] * num_taps)
        if self.has_bias:
            inputs.append((self.args[2].get_ram_width() * self.par, param_len))
        if self.has_scale:
            inputs.append((self.args[self.scale_index].get_ram_width() * self.par, param_len))

        return inputs, outputs, temps

    def get_stream_hash(self):
        base = pool._pool.get_stream_hash(self)
        return (base, self.has_bias, self.has_scale, self.act_func,
                self.asymmetric_clip, self.get_sum_dtype())

    def get_required_substreams(self):
        sum_dtype = self.get_sum_dtype()
        sum_width = sum_dtype.width
        if self.has_scale:
            scale = self.args[self.scale_index]
            scale_width = scale.get_op_width()
            scale_point = scale.get_op_point()
            scale_signed = scale.get_signed()
        else:
            scale_width, scale_point, scale_signed = 2, 0, True
        args = (sum_width, 0, True,
                scale_width, scale_point, scale_signed,
                sum_width + scale_width, 0, True,
                self.get_op_width(), self.get_op_point(), self.get_signed(),
                self.asymmetric_clip)
        return [('mul_rshift_round_clip', args)] * self.par

    def get_stream_func(self):

        def func(strm):
            num_taps = self.get_num_taps()

            mask = strm.parameter(datawidth=num_taps, signed=False)
            rshift = strm.parameter(datawidth=8, signed=False)
            act_max = strm.parameter(datawidth=self.get_op_width() + 1, signed=True)
            kh = self.ksize[-3]
            kw = self.ksize[-2]
            # rotation of the circular line buffer (row_select * kw + col_select)
            rot = strm.parameter(datawidth=max((kh * kw).bit_length(), 1), signed=False)

            def vec_source(arg):
                width = arg.get_op_width()
                vec = strm.source(datawidth=width * self.par, signed=False)
                return vec

            def split(vec, arg):
                width = arg.get_op_width()
                point = arg.get_op_point()
                signed = arg.get_signed()
                if self.par == 1:
                    return [strm.ReinterpretCast(vec, width, point, signed)]
                return strm.Split(vec, width, point, signed, reverse=True)

            act = self.args[0]
            filter = self.args[1]

            # source order: act taps, filter taps, bias, scale
            act_vecs = [vec_source(act) for _ in range(num_taps)]
            filter_vecs = [vec_source(filter) for _ in range(num_taps)]
            bias_vec = vec_source(self.args[2]) if self.has_bias else None
            scale_vec = vec_source(self.args[self.scale_index]) if self.has_scale else None

            act_lanes = [split(v, act) for v in act_vecs]
            filter_lanes = [split(v, filter) for v in filter_vecs]
            bias_lanes = split(bias_vec, self.args[2]) if self.has_bias else None
            scale_lanes = (split(scale_vec, self.args[self.scale_index])
                           if self.has_scale else None)

            sum_width = self.get_sum_dtype().width
            mul_width = act.get_op_width() + filter.get_op_width()

            def rotated_filter(t, lane):
                # physical slot t = (py, px) holds logical tap ((py - rs) % kh, (px - cs) % kw)
                py, px = t // kw, t % kw
                v = None
                for rs in range(kh):
                    for cs in range(kw):
                        logical = ((py - rs) % kh) * kw + ((px - cs) % kw)
                        cand = filter_lanes[logical][lane]
                        if v is None:
                            v = cand
                        else:
                            v = strm.Mux(rot == rs * kw + cs, cand, v)
                return v

            out_vars = []
            for i in range(self.par):
                prods = []
                for t in range(num_taps):
                    a = strm.Mux(mask[t], strm.Int(0), act_lanes[t][i])
                    p = strm.Times(a, rotated_filter(t, i))
                    # use the accumulator width already for the products,
                    # so that no partial sum in the adder tree can overflow
                    p.width = max(mul_width, sum_width)
                    p.signed = True
                    prods.append(p)

                s = strm.AddTree(*prods) if len(prods) > 1 else prods[0]
                s.width = sum_width
                s.signed = True

                if self.has_bias:
                    s = s + bias_lanes[i]
                    s.width = sum_width
                    s.signed = True

                scl = scale_lanes[i] if self.has_scale else strm.Int(1)

                sub = strm.substream(self.substreams[i])
                sub.to_source('x', s)
                sub.to_source('y', scl)
                sub.to_source('rshift', rshift)
                v = sub.from_sink('z')

                if self.act_func is relu:
                    v = strm.Mux(v > strm.Int(0), v, strm.Int(0))
                elif self.act_func is relu6:
                    v = strm.Mux(v > strm.Int(0), strm.Mux(v > act_max, act_max, v), strm.Int(0))

                v = bt.out_rcast(strm, v, self.get_op_width(), self.get_op_point(),
                                 self.get_signed())
                out_vars.append(v)

            vec_out = out_vars[0] if self.par == 1 else strm.Cat(*reversed(out_vars))
            strm.sink(vec_out)

        return func

    def get_control_param_values(self):
        ret = pool._pool.get_control_param_values(self)

        filter = self.args[1]
        num_ch = self.args[0].get_aligned_shape()[-1]
        param_len = int(math.ceil(num_ch / self.par))

        aligned_filter_ch = bt.align_word(filter.shape[-1], filter.get_word_alignment())
        filter_tap_step = bt.to_byte(aligned_filter_ch * filter.get_ram_width())

        ret['dw_param_read_size'] = param_len
        ret['dw_filter_tap_step'] = filter_tap_step
        ret['dw_cshamt_out'] = self.cshamt_out if self.cshamt_out is not None else 0
        ret['dw_act_max'] = self.get_act_max()
        return ret

    def control_init_hook(self, fsm):
        # load filter (one RAM per tap), bias, and scale RAMs once per invocation
        num_taps = self.get_num_taps()
        param_rams = self.input_rams[num_taps:]
        filter_rams = param_rams[:num_taps]
        rest = param_rams[num_taps:]

        bt.bus_lock(self.maxi, fsm)
        for t, ram in enumerate(filter_rams):
            gaddr = self.arg_objaddrs[1] + self.dw_filter_tap_step * t
            bt.dma_read(self.maxi, fsm, ram, 0, gaddr, self.dw_param_read_size, port=1)
        idx = 0
        if self.has_bias:
            bt.dma_read(self.maxi, fsm, rest[idx], 0, self.arg_objaddrs[2],
                        self.dw_param_read_size, port=1)
            idx += 1
        if self.has_scale:
            bt.dma_read(self.maxi, fsm, rest[idx], 0, self.arg_objaddrs[self.scale_index],
                        self.dw_param_read_size, port=1)
        bt.bus_unlock(self.maxi, fsm)

    def control_comp_hook(self, comp_fsm):
        num_taps = self.get_num_taps()
        param_rams = self.input_rams[num_taps:]

        names = list(self.stream.parameters.keys())
        self.stream.set_parameter(comp_fsm, names[1], self.dw_cshamt_out)
        comp_fsm.set_index(comp_fsm.current - 1)
        self.stream.set_parameter(comp_fsm, names[2], self.dw_act_max)
        comp_fsm.set_index(comp_fsm.current - 1)
        kw = self.ksize[-2]
        rot = self.window_row_select * kw + self.window_col_select
        self.stream.set_parameter(comp_fsm, names[3], rot)
        comp_fsm.set_index(comp_fsm.current - 1)

        src_names = list(self.stream.sources.keys())[num_taps:]
        for name, ram in zip(src_names, param_rams):
            self.stream.set_source(comp_fsm, name, ram, 0, self.stream_size)
            comp_fsm.set_index(comp_fsm.current - 1)

    def eval(self, memo, input_dict, **kwargs):
        if id(self) in memo:
            return memo[id(self)]

        import nngen.verify as verify

        args = [arg.eval(memo, input_dict) for arg in self.args]
        input = args[0]
        filter = args[1]
        bias = args[2] if self.has_bias else None
        scale = args[self.scale_index] if self.has_scale else None

        act_max = self.get_act_max() if self.act_func is relu6 else None
        ret = verify.depthwise_conv2d(input, filter, self.strides,
                                      bias, scale, self.cshamt_out,
                                      self.act_func, self.padding, self.asymmetric_clip,
                                      self.dtype, self.sum_dtype, self.name, self.par,
                                      act_max=act_max)
        memo[id(self)] = ret
        return ret
