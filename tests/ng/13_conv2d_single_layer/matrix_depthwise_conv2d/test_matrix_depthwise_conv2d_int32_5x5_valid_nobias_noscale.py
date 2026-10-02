from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import nngen as ng
import veriloggen

import matrix_depthwise_conv2d


act_shape = (1, 9, 9, 5)
ksize = (5, 5)
act_dtype = ng.int32
weight_dtype = ng.int32
bias_dtype = ng.int32
scale_dtype = ng.int32
out_dtype = ng.int32
stride = (1, 1, 1, 1)
padding = (0, 0, 0, 0)
with_bias = False
with_scale = False
rshift_out = 0
act_func = None
act_scale_factor = 1.0
par = 1
axi_datawidth = 32


def run(simtype, silent, outputfile, filename=None):
    return matrix_depthwise_conv2d.run(act_shape, ksize,
                                       act_dtype, weight_dtype, bias_dtype, scale_dtype, out_dtype,
                                       stride, padding, with_bias, with_scale,
                                       rshift_out, act_func, act_scale_factor, par,
                                       axi_datawidth, silent,
                                       filename=filename, simtype=simtype,
                                       outputfile=outputfile)


def test(request, silent=True):
    veriloggen.reset()

    simtype = request.config.getoption('--sim')

    rslt = run(simtype, silent, os.path.splitext(os.path.basename(__file__))[0] + '.out')

    verify_rslt = [line for line in rslt.splitlines() if line.startswith('# verify:')][0]
    assert(verify_rslt == '# verify: PASSED')


if __name__ == '__main__':
    rslt = run('iverilog', False, os.path.splitext(os.path.basename(__file__))[0] + '.out', 'tmp.v')
    print(rslt)
