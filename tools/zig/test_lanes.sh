#!/usr/bin/env bash
# Modified by Fiore-lab from TensorFold 1.0.2 (NOTICE lists the changed files).
# The lane core's tests, plus record_lanes.py's vectors and traces where TF_LANES_FIXTURES names them; ZIG picks the compiler.
set -euo pipefail
zig="${ZIG:-zig}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
lanes="$root/zig/src/core/lanes/lanes.zig"
# the keyed sampler in better C, with zig/build/lanes.zig's flags (a module's C files come before its -M)
sampling=(-cflags -std=c++20 -nostdinc++ -fno-exceptions -fno-rtti -ffp-contract=off -fno-fast-math -Wall -Wextra -- "$root/zig/src/core/lanes/sampling.cpp")
"$zig" test -lc "$lanes" "${sampling[@]}"
for name in lanes_units lanes_replay; do
  "$zig" test -lc --dep lanes "-Mroot=$root/zig/tests/$name.zig" "${sampling[@]}" "-Mlanes=$lanes"
done
