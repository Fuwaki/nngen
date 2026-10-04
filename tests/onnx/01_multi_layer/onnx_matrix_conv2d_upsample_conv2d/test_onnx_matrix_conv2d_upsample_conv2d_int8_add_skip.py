from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import nngen as ng
import veriloggen

import onnx_matrix_conv2d_upsample_conv2d

kwargs = dict(skip='add', opset_version=13)


def run(simtype, silent, outputfile, filename=None):
    return onnx_matrix_conv2d_upsample_conv2d.run(silent=silent, filename=filename,
                                                  simtype=simtype, outputfile=outputfile,
                                                  **kwargs)


def test(request, silent=True):
    veriloggen.reset()

    simtype = request.config.getoption('--sim')

    rslt = run(simtype, silent, os.path.splitext(os.path.basename(__file__))[0] + '.out')

    verify_rslt = [line for line in rslt.splitlines() if line.startswith('# verify:')][0]
    assert(verify_rslt == '# verify: PASSED')


if __name__ == '__main__':
    rslt = run('iverilog', False, os.path.splitext(os.path.basename(__file__))[0] + '.out', 'tmp.v')
    print(rslt)
