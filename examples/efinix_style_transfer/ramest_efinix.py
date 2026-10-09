"""
Rough Efinix Titanium RAM10K (10 Kbit block) estimate for NNgen-generated Verilog.

Every NNgen on-chip memory is a leaf module with `reg [W-1:0] mem [0:D-1];` and two
read/write ports (true dual port). Titanium RAM10K true-dual-port shapes used here:
1024x10, 1024x8, 2048x5, 2048x4, 4096x2, 8192x1 (cascading in width/depth).
Small memories (depth <= SMALL_DEPTH) are reported separately: Efinity usually
implements them in registers/LUTs instead of RAM blocks.

This is an estimate; confirm with the Efinity place report.

usage: python ramest_efinix.py design.v
"""
import re
import sys
import math
import collections

TDP_SHAPES = [(10, 1024), (8, 1024), (5, 2048), (4, 2048), (2, 4096), (1, 8192)]
SMALL_DEPTH = 32


def blocks(width, depth):
    return min(math.ceil(width / cw) * math.ceil(depth / cd) for cw, cd in TDP_SHAPES)


def estimate(filename, small_depth=SMALL_DEPTH, verbose=False):
    src = open(filename).read()
    mems = [(int(w), int(d)) for w, d in re.findall(r'reg \[(\d+)-1:0\] mem \[0:(\d+)-1\];', src)]
    c = collections.Counter(mems)
    total = 0
    small_bits = 0
    small_cnt = 0
    bits = 0
    shapes = []
    for (w, d), k in sorted(c.items()):
        if d <= small_depth:
            small_bits += w * d * k
            small_cnt += k
            shapes.append('%dx%d x%d (logic)' % (w, d, k))
            continue
        b = blocks(w, d)
        total += b * k
        bits += w * d * k
        shapes.append('%dx%d x%d -> %d blk each' % (w, d, k, b))
    util = bits / (total * 10240.0) if total else 0.0
    res = collections.OrderedDict(
        ram10k=total, ram_leafs=sum(k for (w, d), k in c.items() if d > small_depth),
        ram_kbit=round(bits / 1024.0, 1), block_fill=round(util, 3),
        small_mems=small_cnt, small_mem_bits=small_bits, shapes=shapes)
    if verbose:
        for s in shapes:
            print('  ' + s)
    return res


if __name__ == '__main__':
    for f in sys.argv[1:]:
        r = estimate(f, verbose=True)
        print(f, dict((k, v) for k, v in r.items() if k != 'shapes'))
