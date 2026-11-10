#!/usr/bin/env bash
# Modified by Fiore-lab from TensorFold 1.0.2 (NOTICE lists the changed files).
# The server's unit tests (its module with the engine seam, tokenizer and template), then the engine seam's own. ZIG overrides the compiler.
set -euo pipefail
zig="${ZIG:-zig}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
# the keyed sampler in better C, with zig/build/lanes.zig's flags (a module's C files come before its -M)
sampling=(-cflags -std=c++20 -nostdinc++ -fno-exceptions -fno-rtti -ffp-contract=off -fno-fast-math -Wall -Wextra -- "$root/zig/src/core/lanes/sampling.cpp")
"$zig" test -lc --dep engine_api --dep tokenizer --dep template \
  "-Mroot=$root/zig/src/server/root.zig" \
  --dep lanes "-Mengine_api=$root/zig/src/core/engine_api.zig" \
  "${sampling[@]}" "-Mlanes=$root/zig/src/core/lanes/lanes.zig" \
  "-Mtokenizer=$root/zig/src/core/tokenizer/tokenizer.zig" \
  "-Mtemplate=$root/zig/src/core/template/template.zig" "$@"
"$zig" test -lc --dep lanes "-Mroot=$root/zig/src/core/engine_api.zig" "${sampling[@]}" "-Mlanes=$root/zig/src/core/lanes/lanes.zig"
