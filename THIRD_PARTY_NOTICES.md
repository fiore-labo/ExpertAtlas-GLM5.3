# Third-party notices

TensorFold uses [MLX](https://github.com/ml-explore/mlx) and
[mlx-lm](https://github.com/ml-explore/mlx-lm), MIT License, Copyright © 2023 Apple Inc.
They are installed as dependencies.

## MLX and mlx-lm adaptations

The DeltaNet implementations in `src/tensorfold/kernels/qwen/dense/v1/lane_gdn.py` and
`lane_tree.py` adapt mlx-lm's `qwen3_5` and `gated_delta` model math and kernels under its MIT License.

## Qwen Flash Next

The n-gram ID helpers in `src/tensorfold/families/qwen4_exp/model.py` and `cuda/ngram.py` translate
Hugging Face transformers' `models/qwen4_exp/modeling_qwen4_exp.py` into MLX and NumPy with renamed
identifiers. Copyright 2026 The Qwen Team and The HuggingFace Inc. team, Apache License 2.0.
See [the license text](LICENSES/Apache-2.0.txt). The same helpers appear in mlx-vlm's
`models/qwen4_exp/language.py`, MIT License, Copyright © 2025 Prince Canuma.

## GLM-5.3-Flash on Apple Silicon

The MLX engine of `glm5_next` (`src/tensorfold/families/glm5_next/`: the forward pass in `model.py`, `kda.py`,
`mla.py` and `mlp.py`, the draft head in `mtp.py`, `runtime.py`) and its Metal kernels
(`src/tensorfold/kernels/glm/flash/v1/`) are written for TensorFold. What they follow or port:

- The forward pass follows, op for op on its prefill path, the GLM-5.3-Flash (`glm5_next`)
  implementation added to [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) by PR #2030 (by Lazarus-931; MIT License,
  Copyright (c) 2025 Prince Canuma), as vendored by [oMLX](https://github.com/jundot/omlx) (Apache-2.0). Nothing is
  imported from either at runtime.
- `kernels/glm/flash/v1/kda.py` is ported from mlx-vlm PR #2105 ("glm5_next: fuse the KDA decode chain into one
  Metal kernel", by avlp12; `mlx_vlm/models/glm5_next/fused_kda.py`; closed without merging, MIT License,
  Copyright (c) 2025 Prince Canuma): the whole KDA decode step in one Metal kernel. TensorFold runs a window of
  rows in order inside the launch, folds the 4-bit `f_b` / `g_b` projections in with MLX's one-row `qmv_quad`
  arithmetic, and keeps its own rounding points. Its precision rules (precise exp, uncontracted sums of squares)
  are also used in `fused.py`, `moe.py` and `hc.py`.
- `kernels/glm/flash/v1/sparse_attention.py` is mlx-vlm's `indexed_sparse_attention` kernel
  (`mlx_vlm/models/sparse_attention.py`) as extended by mlx-vlm PR #2245 ("Fix GLM-5.3 cached decode batch
  invariance", by raullenchai; closed without merging, MIT License, Copyright (c) 2025 Prince Canuma), adapted to
  TensorFold's single latent cache.
- mlx-vlm PR #2107 (the sparse indexer's incremental decode and a stale-pool fix, by avlp12) needed no code:
  TensorFold's cache already pools once per completed block. Its stale-pool case is pinned by
  `tests/test_glm5_ported_kernels.py`.
- The hyper-connection kernel `_HC_SPLIT` in `kernels/glm/flash/v1/kernels.py`, and the sinkhorn and collapse in
  `hc.py`, repeat the `hc_sinkhorn_collapse` kernel of mlx-vlm's `mlx_vlm/models/deepseek_v4/hyper_connection.py`
  (MIT License, Copyright (c) 2026 Apple Inc.), with its output type set to the input's.
- The 4-bit matvec `_QMV_ROWS` in `kernels.py` is Flash Next's `qmv_rows` with MLX's group-64 scale indexing, and
  the expert kernels (`_EXPERT_GROUP`, `_EXPERT_QMV`) follow Flash Next's `expert_group` / `grouped_gateup`. The
  row kernels in `kernels.py`, `moe.py` and `hc.py` repeat the arithmetic and partitions of MLX 0.32's own kernels (MIT
  License, Copyright © 2023 Apple Inc.): `qmv_fast`, `qmv_quad` and `gather_qmv_fast` (`quantized.h`), `GEMVKernel`
  and `GEMVTKernel` (`gemv.h`) and the `rms_norm` kernels, one row per grid slice with the tiling MLX picks for one
  row, so each row keeps MLX's one-row bits.

## CUDA

CUDA backends use [PyTorch](https://github.com/pytorch/pytorch), BSD-3-Clause, and
[Triton](https://github.com/triton-lang/triton), MIT, supplied by NVIDIA's container rather than bundled.
Dense Qwen implements mlx-lm's model math. CUDA DFlash2 implementations port z-lab's architecture under
the MIT License, Copyright © 2026 Z Lab.

Flash Next implements transformers' model math under the attribution above. Its new DeltaNet kernel
follows flash-linear-attention's numerics, MIT; its NCCL wrapper follows vLLM's stream convention,
Apache-2.0, without copying either implementation.

GLM's CUDA engine implements transformers' `models/glm5_next/modular_glm5_next.py` math, Apache-2.0,
without including that source. Its draft inputs and thinking-off rendering follow the public GLM recipe
from Mia-AiLab without including recipe code.

GLM EXL3 (`families/glm5_next/cuda/exl3.py`, `exl3.cu`, `exl3_mm.py`), the shared EXL3 module
(`src/tensorfold/cuda/exl3/`) and the EXL3 loaders of Qwen3.8-27B and Qwen3.8 Flash Next
(`families/qwen3_5/cuda/exl3_load.py`, `families/qwen4_exp/cuda/exl3.py`) read
[ExLlamaV3](https://github.com/turboderp-org/exllamav3)'s EXL3 format: its trellis layout and bitstream, its
"3inst", "mcg" and "mul1" codebooks, its half-integer bit widths and its tensor-core fragment order. Flash
Next's packs also carry ExLlamaV3's n-gram row codec, read as its `ngram_dequant` reads it. ExLlamaV3 uses the
MIT License, Copyright © 2025 Turboderp. TensorFold's decoders and kernels are separate implementations,
checked bit for bit against ExLlamaV3's dequantization.

Flash Next's optional int8 and int4 KV caches (`families/qwen4_exp/cuda/kvcache.py`) follow the cache quantization scheme of [ExLlamaV3](https://github.com/turboderp-org/exllamav3) `-cq 8` and `-cq 4` (MIT License, Copyright (c) 2025 Turboderp, text below): groups of 32, one fp16 absmax scale per group, the group rotated by a 32-point Hadamard, midpoint-grid codes, `compand_a == 0`. 8-bit stores each code as a signed int8 (`q - 128`). 4-bit stores two unsigned codes per byte, low nibble first (the same bits as ExLlamaV3's little-endian packing, a uint8 tensor rather than their uint32 words). Their dequantizer folds another `1/sqrt(32)` into the scale and applies the unnormalized butterfly on the way out; this cache applies the normalized H32 to the query and to the merged output instead, and leaves the stored codes rotated. Scales match their quantizer bit for bit. Reconstructed values agree within fp16/bf16 rounding (under 0.01 on random groups), not bit for bit. The quantizer and the attention dequant are written for TensorFold and checked against an independent reference of that arithmetic.

## Full GLM-5.3 in Zig

The Zig engine for full GLM-5.3 (`zig/src/families/glm_moe_dsa/`) and its kernels `native_ops.cu`, `glue.cu` and
`roce.cu` (`zig/kernels/cuda/glm53/`) are new code written with reference to the work below, and we thank its
authors.

- TensorFold's Python engine for full GLM-5.3, designed by drowzeys
  ([ashhart/TensorFold#159](https://github.com/ashhart/TensorFold/pull/159)), Apache-2.0: the engine's structure, its
  Triton kernels (which `native_ops.cu` rewrites as plain CUDA), and its MTP and DFlash2 drafter paths.
- The GLM-5.3-Flash prompt kernels of
  [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold),
  Apache-2.0 (its LICENSE at commit 3fd60dfc; the text is `LICENSES/Apache-2.0.txt`), Copyright 2026 MiaAI-Lab, whose
  NOTICE reads: "GLM-5.3-Flash EXL3 on 2x DGX Spark with TensorFold / Copyright 2026 MiaAI-Lab".
  `LICENSES/MiaAI-Lab-MIT.txt` is the license of MiaAI-Lab's Flash Next patch below, not of this project.
- The 4-bit latent KV cache (`TF_GLM_KV=fp4`, `patches/0115-glm-full-kv-fp4.patch`) of
  [MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold),
  Apache-2.0 (its LICENSE at commit eb6414b4; the text is `LICENSES/Apache-2.0.txt`), Copyright 2026 Mia's AI Lab
  (MiaAI-Lab), whose NOTICE reads: "GLM-5.3 EXL3 on 3x DGX Spark with TensorFold / Copyright 2026 Mia's AI Lab
  (MiaAI-Lab)". It gives the layout of `TF_GLM53_KV=fp4` (432-byte rows: 512 e2m1 codes, an e4m3 scale for each 16, a
  power-of-two row scale, the bf16 rope), which `native_ops.cu` implements with its own code; the sm_121a build stores
  the codes of each 16-byte piece in another order.
- The RoCE one-shot collectives of [b12x](https://github.com/local-inference-lab/b12x) (which `roce.cu` rewrites as
  plain CUDA), Apache-2.0, Copyright (c) 2025 by the b12x authors.
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3)'s EXL3 format, MIT License, Copyright © 2025 Turboderp.
- [z-lab/dflash](https://github.com/z-lab/dflash)'s DFlash2 architecture, MIT License, Copyright © 2026 Z Lab.
- The decode speed-ups of the gr8 recipe,
  [knapcio/GLM-5.3-gr8-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-gr8-4x-DGX-Spark-TP4), MIT License,
  Copyright (c) 2026 knapcio, and of its engine, [kindlingai/gr8](https://github.com/kindlingai/gr8), MIT License,
  Copyright (c) 2026 gr8 engine contributors: `TF_GLM53_SPLIT_QA`, `TF_GLM53_KVB`, `TF_GLM53_L2PF`,
  `TF_GLM53_DFLASH_HEAD4` and `TF_GLM53_DFLASH_CONF` take their ideas, implemented with this engine's own code.
- [ollama/ollama#12375](https://github.com/ollama/ollama/issues/12375), the report on KV caches and token ids: Byte
  Resume (`TF_GLM53_BYTE_RESUME`) takes its idea of matching a prompt to the cache by its text past the first id that
  differs, implemented with this engine's own code.

It also includes code of other projects, as each file's first lines say:

- From TensorFold's Python engine (Apache-2.0, Copyright 2026 TensorFold contributors): `decode.cuh` and
  `qmm_frag.cuh`, unmodified; `experts_grouped.cuh`, `qmm_group.cu`, `prompt_experts.cu`, and `exl3_experts.cu` /
  `exl3_linear.cu` (its `experts.cu` / `linear.cu`), modified by Fiore-lab: host code compiled out for the Zig build,
  and in `prompt_experts.cu` the expert-order table built only when ordering by count.
- From b12x (Apache-2.0, Copyright (c) 2025 by the b12x authors): `zig/native/roce_proxy.c`, its
  `b12x/comm/roce/_roce_proxy.c` at commit 1a7e3ec2, modified by Fiore-lab to pick a GID index for each RDMA device.

The Linux aarch64 release archive carries `lib/libroce_proxy.so`, built from `zig/native/roce_proxy.c` above. The
engine opens [NCCL](https://github.com/NVIDIA/nccl) at run time; neither the repository nor the archive carries it.

`lib/libroce_proxy.so` is compiled against rdma-core's `<infiniband/verbs.h>` (the build host's `libibverbs-dev`),
and the inline wrappers it calls (`ibv_post_send`, `ibv_poll_cq` and others) are compiled into the library. That
header is available under the GNU General Public License version 2 or the OpenIB.org BSD license; it is used here
under the latter:

```
Copyright (c) 2004, 2005 Topspin Communications.  All rights reserved.
Copyright (c) 2004, 2011-2012 Intel Corporation.  All rights reserved.
Copyright (c) 2005, 2006, 2007 Cisco Systems, Inc.  All rights reserved.
Copyright (c) 2005 PathScale, Inc.  All rights reserved.
Copyright (c) 2020 Intel Corporation.  All rights reserved.

    Redistribution and use in source and binary forms, with or
    without modification, are permitted provided that the following
    conditions are met:

     - Redistributions of source code must retain the above
       copyright notice, this list of conditions and the following
       disclaimer.

     - Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials
       provided with the distribution.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The control panel's page (`panel.html`, which `bin/tensorfold-native` also carries) holds a subset of the font
M PLUS 1 Code, Copyright 2021 The M+ FONTS Project Authors (https://github.com/coz-m/MPLUS_FONTS), under the SIL Open
Font License, Version 1.1 (`LICENSES/OFL-1.1.txt`); the subset keeps the font's copyright and license fields.

The container image built from the archive's `docker/Dockerfile` adds to the archive's files: the Ubuntu 24.04 base
system and the packages it installs, each under its own license (listed in the image's `/usr/share/doc`); NVIDIA's
cuBLAS 13 (`libcublas-13-0` from NVIDIA's CUDA repository), under the NVIDIA CUDA Toolkit End User License Agreement
(its text in the image's `/usr/share/doc/libcublas-13-0/`); rdma-core's `libibverbs1` and `ibverbs-providers`, under
the GNU General Public License version 2 or the OpenIB.org BSD license; and NCCL 2.30.7 at `/opt/nccl-fenced`,
built by Fiore-lab from NCCL's source at tag v2.30.7-1, for sm_121 only, with commit `05d026f4`
(NVIDIA/nccl#2393, an acquire fence in `ncclIbIsend`) added; mostly Apache-2.0, with parts under their original BSD
license and NVTX under Apache-2.0 with LLVM exceptions (its `LICENSE.txt` and `ThirdPartyNotices.txt` beside the
library). The image carries no model weights.

The archive's binaries are built with Zig 0.17.0, whose standard library and compiler runtime are linked in (MIT
License, Copyright (c) Zig contributors, text below). Their GPU code is compiled with NVIDIA's CUDA 13.0 compiler and
includes its device math routines, as CUDA programs do (NVIDIA CUDA Toolkit End User License Agreement).

The engine drafts by default with the MTP layer of the GLM-5.3 checkpoint, which is part of the weights and keeps
their license (the GLM-5.3 License, Copyright (c) 2026 Z.AI: a permissive grant whose copyright and permission notice
must go with every copy, and under which a licensee running a Model-as-a-Service business with group revenue above
US$10 billion in 12 months must pass Z.AI's security review before commercial use); neither the repository nor the
archive carries weights. The optional
`incoai/GLM-5.3-DFlash2` model card states CC BY-NC-ND 4.0 (non-commercial use without derivatives), for research and
evaluation, with commercial licensing from Inco AI; the engine runs without it, and neither the repository nor the
archive carries it.

### llguidance (structured output)

The Linux aarch64 release archive also carries `lib/libllguidance.so` (and so does the container image built from it),
the grammar engine of the structured output, which the engine opens at run time;
`zig/src/families/glm_moe_dsa/structure/llg.zig` declares in Zig the structs and functions of its C API
(`parser/llguidance.h`). The library is built from `third_party/llguidance/`:
[llguidance](https://github.com/guidance-ai/llguidance) v1.9.1 (tag v1.9.1, commit f0971424), MIT License, Copyright (c)
Microsoft Corporation (its `LICENSE`, text below). Its members `parser` (the crate `llguidance`, the C library) and
`toktrie` keep upstream's code unmodified; Fiore-lab changed the build files only: the workspace `Cargo.toml` (those two
members and a release profile), `parser/Cargo.toml` (without the dev-dependencies, the bench, and the optional
dependencies and features the build does not use), `Cargo.lock` (trimmed to the two members, every version and checksum
as upstream's), `.cargo/config.toml` and `.gitignore`.

`third_party/llguidance/vendor/` is `cargo vendor`'s output for that lock: 62 crates from crates.io, each in its own
directory with its manifest and license files. The library is built from them offline with Rust 1.95.0,
`cargo build --release --locked --offline -p llguidance --target aarch64-unknown-linux-gnu --no-default-features
--features lark,referencing,ahash` (upstream's default features without `rayon`), and linked by Zig 0.17.0
(`zig cc -target aarch64-linux-gnu.2.28`). Rust's standard library is linked into it, as into any Rust library: the
Rust Project's code under MIT OR Apache-2.0 (Copyright The Rust Project Developers) and the crates.io crates the
standard library is built from, each under its own permissive license (the toolchain's
`share/doc/rust/COPYRIGHT-library.html`). Of those, the library's panic and backtrace support shows addr2line 0.25.1,
gimli 0.32.3, hashbrown 0.16.1, miniz_oxide 0.8.9 and rustc-demangle 0.1.27, each offering the MIT License (miniz_oxide:
MIT OR Zlib OR Apache-2.0; the others MIT OR Apache-2.0) and used here under it.

Where a crate offers the MIT License among its licenses, it is used here under the MIT License (text below); `ryu`,
which offers Apache-2.0 or the Boost Software License 1.0, is used under Apache-2.0 (`LICENSES/Apache-2.0.txt`). Each
crate below is named with its version and its manifest's license expression, then the copyright lines of its license
files, and its manifest's authors where those lines name no holder.

Compiled into `lib/libllguidance.so` (`cargo tree --offline --locked -e normal,no-proc-macro --target
aarch64-unknown-linux-gnu -p llguidance --no-default-features --features lark,referencing,ahash`), besides `llguidance`
and `toktrie` 1.9.1 (MIT, Copyright (c) Microsoft Corporation), 30 crates:

- `ahash` 0.8.12, MIT OR Apache-2.0: Copyright (c) 2018 Tom Kaitchuck
- `anyhow` 1.0.98, MIT OR Apache-2.0: no copyright line; authors: David Tolnay
- `borrow-or-share` 0.2.2, MIT-0: Copyright 2024 Scallop Ye (its `LICENSE` is `LICENSES/MIT-0.txt`)
- `bytemuck` 1.23.1, Zlib OR Apache-2.0 OR MIT: Copyright (c) 2019 Daniel "Lokathor" Gee. (its MIT text has
  "(including the next paragraph)" after "this permission notice")
- `cfg-if` 1.0.1, MIT OR Apache-2.0: Copyright (c) 2014 Alex Crichton
- `derivre` 0.3.13, MIT: Copyright (c) Microsoft Corporation.
- `equivalent` 1.0.2, Apache-2.0 OR MIT: Copyright (c) 2016--2023; no authors in its manifest (repository
  indexmap-rs/equivalent)
- `fluent-uri` 0.3.2, MIT: Copyright (c) 2024 Scallop Ye
- `getrandom` 0.3.3, MIT OR Apache-2.0: Copyright (c) 2018-2025 The rust-random Project Developers; Copyright (c) 2014
  The Rust Project Developers
- `hashbrown` 0.15.4 and 0.17.1, MIT OR Apache-2.0: Copyright (c) 2016 Amanieu d'Antras
- `indexmap` 2.10.0, Apache-2.0 OR MIT: Copyright (c) 2016--2017; no authors in its manifest (repository
  indexmap-rs/indexmap)
- `itoa` 1.0.15, MIT OR Apache-2.0: no copyright line; authors: David Tolnay
- `libc` 0.2.174, MIT OR Apache-2.0: Copyright (c) 2014-2020 The Rust Project Developers
- `lock_api` 0.4.13, `parking_lot` 0.12.4 and `parking_lot_core` 0.9.11, MIT OR Apache-2.0: Copyright (c) 2016 The Rust
  Project Developers
- `memchr` 2.7.5, Unlicense OR MIT: Copyright (c) 2015 Andrew Gallant
- `once_cell` 1.21.3, MIT OR Apache-2.0: no copyright line; authors: Aleksey Kladov
- `percent-encoding` 2.3.1, MIT OR Apache-2.0: Copyright (c) 2013-2022 The rust-url developers
- `ref-cast` 1.0.24, MIT OR Apache-2.0: no copyright line; authors: David Tolnay
- `referencing` 0.29.1, MIT: Copyright (c) 2020-2025 Dmitry Dygalo
- `regex-syntax` 0.8.5, MIT OR Apache-2.0: Copyright (c) 2014 The Rust Project Developers. Its Unicode tables
  (`src/unicode_tables/`, generated from the Unicode Character Database 16.0.0) are compiled in, under the Unicode, Inc.
  License Agreement in its `src/unicode_tables/LICENSE-UNICODE`, Copyright © 1991-2018 Unicode, Inc.
  (`LICENSES/Unicode-DFS-2016.txt`)
- `ryu` 1.0.20, Apache-2.0 OR BSL-1.0: no copyright line; authors: David Tolnay
- `scopeguard` 1.2.0, MIT OR Apache-2.0: Copyright (c) 2016-2019 Ulrik Sverdrup "bluss" and scopeguard developers
- `serde` 1.0.219 and `serde_json` 1.0.140, MIT OR Apache-2.0: no copyright line; authors: Erick Tryzelaar, David Tolnay
- `smallvec` 1.15.1, MIT OR Apache-2.0: Copyright (c) 2018 The Servo Project Developers
- `strum` 0.28.0, MIT: Copyright (c) 2019 Peter Glotfelty
- `zerocopy` 0.8.26, BSD-2-Clause OR Apache-2.0 OR MIT: Copyright 2023 The Fuchsia Authors

Run by the compiler while the library is built, and not linked into it: the procedural macros `bytemuck_derive`,
`ref-cast-impl`, `serde_derive` and `strum_macros`, which generate code into the crates above, and their dependencies:

- `bytemuck_derive` 1.9.3, Zlib OR Apache-2.0 OR MIT: Copyright (c) 2019 Daniel "Lokathor" Gee.
- `heck` 0.5.0, MIT OR Apache-2.0: Copyright (c) 2015 The Rust Project Developers
- `proc-macro2` 1.0.95, MIT OR Apache-2.0: no copyright line; authors: David Tolnay, Alex Crichton
- `quote` 1.0.40, MIT OR Apache-2.0: no copyright line; authors: David Tolnay
- `ref-cast-impl` 1.0.24, MIT OR Apache-2.0: no copyright line; authors: David Tolnay
- `serde_derive` 1.0.219, MIT OR Apache-2.0: no copyright line; authors: Erick Tryzelaar, David Tolnay
- `strum_macros` 0.28.0, MIT: Copyright (c) 2019 Peter Glotfelty
- `syn` 2.0.104, MIT OR Apache-2.0: no copyright line; authors: David Tolnay
- `unicode-ident` 1.0.18, (MIT OR Apache-2.0) AND Unicode-3.0: no copyright line in its MIT text; authors: David
  Tolnay. Its Unicode tables: Copyright © 1991-2023 Unicode, Inc., under the Unicode License V3 in its
  `LICENSE-UNICODE` (`LICENSES/Unicode-3.0.txt`)

Build scripts only (build-dependencies): `autocfg` 1.5.0, Apache-2.0 OR MIT: Copyright (c) 2018 Josh Stone;
`version_check` 0.9.5, MIT/Apache-2.0: Copyright (c) 2017-2018 Sergio Benitez.

In `vendor/` but not built (the optional feature `rayon` is off): `rayon` 1.10.0 and `rayon-core` 1.12.1, MIT OR
Apache-2.0: Copyright (c) 2010 The Rust Project Developers; `crossbeam-deque` 0.8.6, `crossbeam-epoch` 0.9.18 and
`crossbeam-utils` 0.8.21, MIT OR Apache-2.0: Copyright (c) 2019 The Crossbeam Project Developers; `either` 1.15.0, MIT
OR Apache-2.0: Copyright (c) 2015; authors: bluss.

Fifteen crates in `vendor/` are stubs: cargo needs their manifests to resolve the lock, but they are reached only on
platforms the release does not build for (Windows, WASI, Redox, UEFI) or, for `zerocopy-derive`, through an optional
dependency that is off. Each keeps upstream's manifests and license files; every source file its manifest names is a
placeholder of at most two lines by Fiore-lab, so they contain none of upstream's code:

- `bitflags` 2.9.1, MIT OR Apache-2.0: Copyright (c) 2014 The Rust Project Developers
- `r-efi` 5.3.0, MIT OR Apache-2.0 OR LGPL-2.1-or-later: Copyright (C) 2017-2023 Red Hat, Inc.; Copyright (C)
  2019-2023 Microsoft Corporation; Copyright (C) 2022-2023 David Rheinsberg (from upstream's `AUTHORS`, which holds its
  license statement and which the stub leaves out)
- `redox_syscall` 0.5.13, MIT: Copyright (c) 2017 Redox OS Developers
- `wasi` 0.14.2+wasi-0.2.4, Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT: no copyright line; authors: The
  Cranelift Project Developers
- `wit-bindgen-rt` 0.39.0, Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT: upstream's crate has no license file,
  and its manifest no authors
- `windows-targets`, `windows_aarch64_gnullvm`, `windows_aarch64_msvc`, `windows_i686_gnu`, `windows_i686_gnullvm`,
  `windows_i686_msvc`, `windows_x86_64_gnu`, `windows_x86_64_gnullvm` and `windows_x86_64_msvc` 0.52.6, MIT OR
  Apache-2.0: Copyright (c) Microsoft Corporation.
- `zerocopy-derive` 0.8.26, BSD-2-Clause OR Apache-2.0 OR MIT: Copyright 2023 The Fuchsia Authors

## Unicode data

The tokenizer and the template engine embed tables generated from the Unicode Character Database 16.0.0
(`zig/src/core/tokenizer/unicode_data.zig`, `zig/src/core/template/unicode_data.zig`), under the Unicode License V3:

```
UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 1991-2024 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.
```

## Vendored code and weights

`src/tensorfold/drafters/vendor/z_lab_dflash/model_mlx.py` is the unmodified `dflash/model_mlx.py` from
[z-lab/dflash](https://github.com/z-lab/dflash), MIT License, Copyright © 2026 Z Lab.

`src/tensorfold/families/deepseek_v4/vendor/encoding_dsv4.py` is the unmodified `encoding/encoding_dsv4.py` of
[deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (revision 60d8d70), and
`tests/fixtures/deepseek_v4/` holds two of its test cases, MIT License, Copyright (c) 2023 DeepSeek.
The MTP layer TensorFold drafts with comes from that checkpoint's last shard (MIT), converted by
`families/deepseek_v4/convert.py`.

TensorFold ships no model weights. The `z-lab/Qwen3.8-27B-DFlash2` model card states Apache-2.0.
The optional `incoai/GLM-5.3-Flash-DFlash2` model card states CC BY-NC-ND 4.0, for non-commercial use
without derivatives. Each checkpoint keeps its own license.

## MIT License text

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Flash Next CUDA image integration

The multimodal rotary and image-feature integration is adapted from MiaAI-Lab's
[Flash Next vision patch 0008](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/a3aa89835022c55ca8e55008c37785954834e04f/patches/0008-flash-next-vision.patch),
MIT License, Copyright (c) 2026 MiaAI-Lab. The license is included in `LICENSES/MiaAI-Lab-MIT.txt`.
The port preserves the v0.5 CUDA execution APIs and adds an offline EXL3 vision adapter.
