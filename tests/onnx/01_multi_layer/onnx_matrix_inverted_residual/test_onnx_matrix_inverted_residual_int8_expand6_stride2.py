from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import nngen as ng
import veriloggen

import onnx_matrix_inverted_residual


act_shape = (1, 8, 8, 8)
expand_ratio = 6
stride = 2
act_dtype = ng.int8
weight_dtype = ng.int8
par_ich = 1
par_och = 1
par_dw = 1
chunk_size = 64
axi_datawidth = 32


def run(simtype, silent, outputfile, filename=None):
    return onnx_matrix_inverted_residual.run(act_shape, expand_ratio, stride,
                                             act_dtype, weight_dtype,
                                             par_ich, par_och, par_dw, chunk_size,
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
