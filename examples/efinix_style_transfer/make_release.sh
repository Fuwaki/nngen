#!/bin/sh
# Regenerate the reference Efinix deliverable in ip_128x128/ :
#   style_net.v (single-file Verilog IP), style_net_nngen.h (register/memory map),
#   nngen_driver.[ch] (bare-metal driver), style_net_params.bin/.h (parameter image),
#   style_net_nngen_report.txt (nngen memory/register map dump), summary.json
# Config: 128x128 RGBX in/out, par=1, AXI master 16 bit, min_onchip_ram_capacity=1024.
# Usage: ./make_release.sh [--sim verilator]   (RTL sim of the 128x128 build takes ~2 min)
set -e
cd "$(dirname "$0")"
SIM=none
[ "$1" = "--sim" ] && SIM="$2"
python style_net_nngen.py --size 128 --par 1 --axi 16 --sim "$SIM" --outdir build_release
mkdir -p ip_128x128
for f in style_net.v style_net_nngen.h nngen_driver.h nngen_driver.c \
         style_net_params.bin style_net_params.h style_net_nngen_report.txt summary.json; do
    cp build_release/$f ip_128x128/
done
( cd ip_128x128 && sha256sum style_net.v style_net_params.bin > SHA256SUMS )
echo "ip_128x128/ refreshed"
