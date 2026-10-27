English | [日本語](README.ja.md) | [简体中文](README.zh-CN.md)

# Making the weights yourself

Every weight file this recipe loads can be made from Z.AI's official GLM-5.3 checkpoint with the tools in this
directory: the eight-node and four-node checkpoints, and the 2, 3, 4 and 5-bit expert banks of Expert Atlas. You do not
need any file Fiore-lab published on Hugging Face.

Why this matters: weights cannot be read like code. Weights altered to behave differently in some cases look like any
other weights. Fiore-lab makes its weights carefully, but you have no way to check that from the files, and you may not
want to rely on weights whose making you cannot check. If you make them yourself, the only weights you rely on are
Z.AI's own, and this page shows how.

## What you rely on, and what you do not need

| | |
|---|---|
| **Z.AI's official checkpoint** | [zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16) at revision `9d2398f478cab2de883137db3a36ad2c96205e24`, the revision Fiore-lab's weights were made from. `check_official.py` checks your download against the list Hugging Face gives for that revision: every file's size and hash. |
| **The quantizer** | [ExLlamaV3](https://github.com/turboderp-org/exllamav3) 1.6.1 (MIT license) and PyTorch, installed from their own releases by `convert.sh setup`. |
| **The tools here** | Short Python and shell scripts, Apache-2.0 like the rest of this repository. Read them before you run them; the table at the end says what each does. |
| **Not needed** | Any weight file of Fiore-lab's. Optionally, one small data file of Fiore-lab's for the four-node checkpoint (`q3-selection.json`: a list of expert numbers, see step 6), which you can also make yourself. |

The files themselves cannot carry code: safetensors files hold a JSON header and raw numbers, and this recipe's engine
reads them with its own code (Zig, no Python at run time, no pickle, no code taken from a model repository). The chat
template and the tokenizer are the official files, copied unchanged. The risk in a weight file is altered numbers, and
making the weights yourself removes that dependency.

## What you make

| What | Files | Size | Used for |
|---|---|---|---|
| 4-bit bank | `experts-layer-003` … `077.safetensors` | 368.5 GB | the routed experts of the eight-node checkpoint; Expert Atlas's 4 bits |
| Eight-node checkpoint | the 4-bit bank's files, `nonexperts-5bit.safetensors`, `unquantized-bf16.safetensors`, three `mtp-layer-78-*` files, the official small files, `config.json`, the index, `SHA256SUMS` | 389.9 GB | `./tf-glm53 up` on eight nodes; the rest of every Expert Atlas start |
| 3-bit bank | `experts-layer-*.safetensors` | 277.9 GB | Expert Atlas's 3 bits; the four-node checkpoint |
| 2-bit / 5-bit banks | `experts-layer-*.safetensors` | 187.3 / 459.1 GB | Expert Atlas's 2 and 5 bits (optional) |
| Four-node checkpoint | the eight-node checkpoint with half of the routed experts at 3 bits | 344.6 GB | `./tf-glm53 up` on four nodes |

What you need depends on how you run the recipe: eight nodes without the control panel need only the eight-node
checkpoint; four nodes without it need the four-node checkpoint (made from the eight-node one and the 3-bit bank); the
control panel needs a checkpoint and the banks of the widths you want (on four nodes the 3- and 4-bit banks, and the
eight-node checkpoint's directory can serve as the 4-bit bank).

## How your files compare with Fiore-lab's

- **The same layout.** The same tensors with the same names, shapes, widths and codebooks, each expert cut into the
  same eight tensor-parallel shares, laid out the same way in the files. The engine loads them exactly as it loads
  Fiore-lab's, and `tf-glm53 verify` and `verify-banks` check them against your own `SHA256SUMS` and manifests.
- **Not the same bytes.** Quantizing runs on GPUs, and Fiore-lab has not tested whether a second conversion, on other
  GPUs or with other library versions, gives the same bytes. The 2- and 5-bit banks also differ by design: Fiore-lab
  calibrated those with ExLlamaV3's default data plus Japanese text and agent / coding sessions, while this page uses
  ExLlamaV3's default calibration data for every width, as Fiore-lab did for the 3- and 4-bit banks and both
  checkpoints. The metadata text in the files differs too (Fiore-lab's begins "Fiore-lab:").
- **Accuracy.** The KL divergence measurements in the README were made with Fiore-lab's files. Yours are made the same
  way and should be close, but they have not been measured.

### What has been checked

The assembly (`assemble.py`, `bank_manifest.py`) was run on the conversion outputs Fiore-lab kept, and every result was
compared with the published file: the 2-, 3- and 5-bit banks layer by layer, the non-expert linears, the MTP layer's
files, the tensors kept unquantized, the index and `config.json`, the four-node mix, and the bank manifests. The tensors
and their bytes are identical (the files differ only in the metadata text above).

The conversion steps (`convert.sh` and the two patches) are the ExLlamaV3 commands and patches Fiore-lab ran, with the
paths made into settings. Fiore-lab has not run `convert.sh` itself from start to end. The patches were checked to
apply to ExLlamaV3 1.6.1's released source, and every option `convert.sh` passes exists in that version.

## Hardware and time

### Rented GPUs (what Fiore-lab used)

Fiore-lab converted on rented machines with NVIDIA RTX PRO 6000 GPUs (96 GB each), ExLlamaV3 1.6.1:

| Step | GPUs | Time |
|---|---|---|
| `full4` (every linear at 4 bits) | 4 | 5.3 hours |
| `experts 3` (the routed experts at 3 bits) | 2 | 8.1 hours |
| `nonexperts` (the other linears at 5 bits) | 1 | 2.6 hours |
| `mtp` | 1 | a few minutes (the quantizing itself takes seconds) |
| `experts 2` / `experts 5` | | not measured with this method; the time a layer depends little on the width, so expect about the same as 3 bits for the same GPUs |
| `assemble.py`, `bank_manifest.py` | none | minutes to an hour a step, depending on the disk |

Disk: the official checkpoint takes 1.51 TB. Each conversion step writes its quantized tensors to its work directory
(about the size of what it quantized) and ExLlamaV3 writes a complete model of its own to `<step>-out`, which this
recipe does not use: delete it after the step if space is short. Plan for about 4 TB in all for the eight-node
checkpoint, the 3-bit bank and the four-node checkpoint, deleting each `<step>-out` after its step. The download of 1.51 TB depends on your connection.

### DGX Spark (not yet tried)

Fiore-lab has not run these conversions on a DGX Spark, so this part is what is known, not a tested procedure:

- A DGX Spark is aarch64 with CUDA 13 and a GB10 GPU sharing 128 GB of memory with the CPU. ExLlamaV3 1.6.1 has no
  prebuilt package for it: install a PyTorch build with CUDA for aarch64 that supports the GB10, then ExLlamaV3 1.6.1
  from its source (`pip install git+https://github.com/turboderp-org/exllamav3@v1.6.1`), which builds its CUDA
  extension. Whether 1.6.1 builds and runs there has not been tested. `convert.sh setup` installs the x86_64 wheel
  only; on a DGX Spark make the venv yourself and run the other steps with `VENV` pointing at it.
- One GB10 has far less GPU compute than the rented GPUs above, so expect each step to take many times longer; it has
  not been measured. The widths are independent after `full4`: with several nodes, run one width on each (each node
  needs its own copy of the official checkpoint and of `full4`'s work directory).
- Watch the memory. When a DGX Spark runs out of memory it can stop responding until it is powered off and on.
  `assemble.py` stops by itself when less than 20 GiB is available.

## Steps

The commands below use `/data` as an example place; use any directory with the space above.

### 1. Get the tools and the official checkpoint

```sh
git clone https://github.com/fiore-labo/ExpertAtlas-GLM5.3 && cd ExpertAtlas-GLM5.3/glm53/weights
pip install -U "huggingface_hub[hf_xet]"
hf download zai-org/GLM-5.3-BF16 --revision 9d2398f478cab2de883137db3a36ad2c96205e24 --local-dir /data/GLM-5.3-BF16
python3 check_official.py /data/GLM-5.3-BF16          # must end with "OK: ... 0 problems"
```

`check_official.py` reads the file list of that revision from huggingface.co when you run it and compares every file:
the sha256 of each weight file and the Git blob hash of each small file. It reads 1.51 TB, so it takes a while.

### 2. Install the quantizer

```sh
./convert.sh setup                                     # a venv in ./venv: PyTorch 2.8 (CUDA 12.8), ExLlamaV3 1.6.1
```

### 3. Convert

```sh
export BF16=/data/GLM-5.3-BF16 WORK=/data/work
./convert.sh full4            # first: the 4-bit experts, and what the next steps calibrate through
./convert.sh nonexperts       # the other linears at 5 bits
./convert.sh mtp              # the MTP layer's linears
./convert.sh experts 3        # the 3-bit experts (for the four-node checkpoint and Expert Atlas)
./convert.sh experts 2        # optional
./convert.sh experts 5        # optional
```

Each step logs to `$WORK/<step>.log`, runs in the foreground (use `tmux` or `nohup` for the long ones) and, started
again after an interruption, resumes from its last checkpoint. `GPUS=0,1` limits a step to those GPUs.

### 4. Assemble the banks

```sh
python3 -m venv /data/tools && /data/tools/bin/pip install numpy xxhash   # or use ./venv
P=/data/tools/bin/python
$P assemble.py bank $WORK/full4/qtensors    /data/banks/4bit
$P assemble.py bank $WORK/experts3/qtensors /data/banks/3bit
$P assemble.py bank $WORK/experts2/qtensors /data/banks/2bit     # if made
$P assemble.py bank $WORK/experts5/qtensors /data/banks/5bit     # if made
```

Each layer's experts are cut into the eight tensor-parallel shares the engine loads; nothing is quantized again.

### 5. Assemble the eight-node checkpoint

```sh
$P assemble.py checkpoint --official $BF16 --experts /data/banks/4bit --mtp-qtensors $WORK/full4/qtensors \
   --nonexperts $WORK/nonexperts/qtensors --mtp-linears $WORK/mtp-linears-5bit.safetensors --out /data/GLM-5.3-8x
```

The expert files are hard links to the 4-bit bank's (add `--copy` to copy them, or when the two directories are on
different file systems). The tensors kept unquantized (embeddings, `lm_head`, norms, the router, `kv_b_proj`, parts
of the indexer, the MTP layer's `eh_proj`) are copied from the official files as stored. The command ends with a check
of the index against the files and writes `SHA256SUMS`.

### 6. Assemble the four-node checkpoint (four nodes only)

The four-node checkpoint stores half of the routed experts at 3 bits. Which half is a list,
`q3-selection.json` (`{"layer": [expert, ...]}` for layers 3 to 77). Either take Fiore-lab's list (it holds only
expert numbers, the least used half across agent work, coding, Linux operations and server security in Japanese and
English; it cannot change what an expert computes), or write your own:

```sh
hf download fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks q3-selection.json --local-dir /data
$P assemble.py mix --base /data/GLM-5.3-8x --bank3 /data/banks/3bit --selection /data/q3-selection.json --out /data/GLM-5.3-4x
```

### 7. The bank manifests and the bank list

The engine loads a bank only when its manifest's sha256 is the one `expert-plan/banks.json` gives. Make a manifest for
each bank and a bank list for them:

```sh
CAL="exllamav3 1.6.1 default calibration data (250 x 2,048 tokens)"
$P bank_manifest.py make /data/banks/4bit --id own-experts-4bit --bits 4 --calibration "$CAL"
$P bank_manifest.py make /data/banks/3bit --id own-experts-3bit --bits 3 --calibration "$CAL"
$P bank_manifest.py banks banks.json /data/banks/3bit /data/banks/4bit      # and the 2- and 5-bit banks if made
```

Each manifest is written into its bank's directory as `<id>.json`. Put the `banks.json` this writes in place of the
release's `expert-plan/banks.json`; the bank directories must be at the same path on every node. Allocations saved
with other banks (Fiore-lab's) do not open with yours (the panel says the bank differs); start new ones.

### 8. Use them

Copy the checkpoint to every node at the same path and set `WEIGHTS` in `glm53.conf` to it, copy the banks to the
directories `banks.json` names, then:

```sh
./tf-glm53 deploy
./tf-glm53 verify            # the weights against your SHA256SUMS on every node
./tf-glm53 verify-banks      # the banks against your manifests on every node
./tf-glm53 up
```

## The files in this directory

| File | What it does |
|---|---|
| `check_official.py` | checks a download of zai-org/GLM-5.3-BF16 against the file list Hugging Face gives for the revision |
| `convert.sh` | installs ExLlamaV3 1.6.1 and runs the conversions (`full4`, `experts BITS`, `nonexperts`, `mtp`) |
| `patch_experts.py` | patches ExLlamaV3's converter to quantize only the routed experts, taking every other linear from `full4` |
| `patch_nonexperts.py` | patches it to quantize only the other linears, taking the routed experts from `full4` |
| `convert_mtp_linears.py` | the MTP layer's linears at 5 bits (mul1), without calibration data |
| `assemble.py` | the banks, the eight- and four-node checkpoints, and `check` for a checkpoint directory |
| `bank_manifest.py` | a bank's manifest (`make`, `check`) and the bank list (`banks`) |

The weights you make are derived from GLM-5.3 and come under the GLM-5.3 License (the official `LICENSE`, which the
checkpoint carries).
