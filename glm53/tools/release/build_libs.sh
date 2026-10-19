#!/bin/sh
# build_libs.sh OUT: the shared library the Linux aarch64 archive carries in lib/, built on a GB10 host.
#   libroce_proxy.so  zig/native/roce_proxy.c (from b12x, Apache-2.0) with glibc 2.28 as the floor, stripped
# NCCL is not carried: each node installs NVIDIA's NCCL 2.30.7 (RUNTIME.md).
# Builds under /tmp so that no home directory path ends up in the library. Pass OUT to `zig build dist -Ddist-lib=`.
set -eu
mkdir -p "$1"
out=$(cd "$1" && pwd)
here=$(cd "$(dirname "$0")/../../.." && pwd)
zig=${ZIG:-zig}
cp "$here/zig/native/roce_proxy.c" /tmp/roce_proxy.c
"$zig" cc -target aarch64-linux-gnu.2.28 -O2 -shared -fPIC -o "$out/libroce_proxy.so" /tmp/roce_proxy.c \
    -isystem /usr/include -L/usr/lib/aarch64-linux-gnu -libverbs -lpthread
strip --strip-unneeded "$out/libroce_proxy.so"
if strings "$out/libroce_proxy.so" | grep -q -E '/home/|/Users/'; then echo "libroce_proxy.so carries a home path" >&2; exit 1; fi
ls -l "$out"
