#!/bin/sh
# build_libs.sh OUT: the shared libraries the Linux aarch64 archive carries in lib/, built on a GB10 host.
#   libroce_proxy.so  zig/native/roce_proxy.c (from b12x, Apache-2.0) with glibc 2.28 as the floor, stripped
#   libllguidance.so  third_party/llguidance (llguidance v1.9.1, MIT, its crates vendored): rank 0's grammar for
#                     structured output. Rust 1.95.0 (rustup; CARGO, default ~/.cargo/bin/cargo) builds it offline
#                     from the tree with zig cc as the linker, for the same glibc floor; stripped
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
# llguidance: crates from vendor/ only (third_party/llguidance/.cargo/config.toml, an empty CARGO_HOME), no rayon (no
# thread of its own); its panic messages' paths remapped, the tree after $HOME since the last matching prefix wins
cargo=${CARGO:-$HOME/.cargo/bin/cargo}
[ -x "$cargo" ] || { echo "no cargo at $cargo: libllguidance.so needs Rust 1.95.0 (rustup)" >&2; exit 1; }
# rustup's cargo finds rustc on PATH and adds only $CARGO_HOME/bin to it, which is the empty one below
PATH=$(dirname "$cargo"):$PATH
printf '#!/bin/sh\nexec "%s" cc -target aarch64-linux-gnu.2.28 "$@"\n' "$(command -v "$zig")" > /tmp/tf-zigcc.sh
chmod 755 /tmp/tf-zigcc.sh
mkdir -p /tmp/tf-cargo-home
(cd "$here/third_party/llguidance" && CARGO_HOME=/tmp/tf-cargo-home CARGO_TARGET_DIR=/tmp/tf-llg-target \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=/tmp/tf-zigcc.sh \
    RUSTFLAGS="--remap-path-prefix=$HOME=home --remap-path-prefix=$here=tensorfold" \
    "$cargo" +1.95.0 build --release --locked --offline -p llguidance --target aarch64-unknown-linux-gnu \
    --no-default-features --features lark,referencing,ahash)
cp /tmp/tf-llg-target/aarch64-unknown-linux-gnu/release/libllguidance.so "$out/"
strip --strip-unneeded "$out/libllguidance.so"
if strings "$out/libllguidance.so" | grep -q -E '/home/|/Users/'; then echo "libllguidance.so carries a home path" >&2; exit 1; fi
strings "$out/libllguidance.so" | grep -q 'llguidance@1\.9\.1' || { echo "libllguidance.so is not llguidance 1.9.1" >&2; exit 1; }
nm -D --defined-only "$out/libllguidance.so" | grep -q ' llg_new_constraint_json$' || { echo "libllguidance.so exports no llg_new_constraint_json" >&2; exit 1; }
ls -l "$out"
