from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import nngen as ng
import veriloggen

import onnx_matrix_depthwise_conv2d


act_shape = (1, 7, 7, 6)
ksize = 3
act_dtype = ng.int16
weight_dtype = ng.int16
stride = 1
padding = 0
with_batchnorm = False
act_func = 'ReLU'
disable_fusion = False
par = 1
chunk_size = 64
axi_datawidth = 32


def run(simtype, silent, outputfile, filename=None):
    return onnx_matrix_depthwise_conv2d.run(act_shape, ksize,
                                            act_dtype, weight_dtype,
                                            stride, padding,
                                            with_batchnorm, act_func, disable_fusion,
                                            par, chunk_size,
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
