English | [日本語](RUNTIME.ja.md) | [简体中文](RUNTIME.zh-CN.md)

# Running full GLM-5.3 from this archive

First time? `QUICKSTART.md` takes you from the image to a running engine in one line. The checkpoint and the 3-bit bank
are always needed, and a start loads every expert at 3 bits unless you choose otherwise.

This archive serves full GLM-5.3 (`glm_moe_dsa`, 755B) on four or eight NVIDIA DGX Spark systems, one rank per GB10,
with tensor parallelism 4 or 8. The server needs no Python, compiler or source checkout; the control panel's launcher
and the log keeper run on the system `python3` with its standard library only. It is Fiore-lab's unofficial
derivative of TensorFold 1.0.2, not a release of the TensorFold project. This page covers both node counts; `T4.md`
gives what differs with four nodes, with its measurements, and `RING.md` four nodes cabled as a ring without a switch.

## What the archive holds

- `bin/tensorfold-native`: the server, with its CUDA kernels built for GB10 (sm_121) inside. Linux aarch64, glibc 2.28
  or later.
- `lib/libroce_proxy.so`: the RDMA proxy of the decode collectives (needs `libibverbs.so.1`).
- `tf-glm53`, `glm53.conf.example` (eight nodes) and `glm53.t4.conf.example` (four nodes): start, stop and check the
  ranks. `CONFIG.md` lists every setting of `glm53.conf`.
- `tools/quickstart.py`: a fresh install to a running engine (`docker run ... IMAGE up` and `./tf-glm53 quickstart` run
  it). `tools/topology.py`: judges how the nodes are cabled. `tools/logkeep.py`: the log keeper.
- `claude-glm53`: Claude Code on the server ("Claude Code").
- The control panel (four or eight nodes): `launcher/tf_launcher.py` (the launcher), `panel.html` (its page; the
  server holds the same page), `tools/quant/expert_plan.py` (writes the expert plans) and `expert-plan/banks.json` (the
  expert banks and where they are).
- `docker/Dockerfile` and `docker/entry.sh`: the release as a container image ("Running in containers").
- The documents, each in English, Japanese (`.ja.md`) and Simplified Chinese (`.zh-CN.md`): `QUICKSTART.md` (the first
  start), `BEFORE-START.md` (what to decide before it), `RUNTIME.md` (this page), `CONFIG.md`, `T4.md` (four nodes),
  `RING.md` (a ring), `PANEL-GUIDE.md` (the panel's screens), `AGENT-API.md` (the control API for an AI agent),
  `AI-GUIDE.md` (the whole system on one page, for an AI assistant) and `ATLAS-TESTS.md` (the expert re-sort's
  measurements).
- `LICENSE`, `NOTICE`, `LICENSES/`, `THIRD_PARTY_NOTICES.md`, `VERSION`, `SHA256SUMS`.

It holds no model weights, no draft model and no NCCL. The image holds NCCL with the AArch64 fence ("Running in
containers").

## What each node needs

- DGX OS (Ubuntu 24.04) with the NVIDIA driver 580 or later, which supplies `libcuda.so.1`.
- cuBLAS 13 (`libcublas.so.13`, `libcublasLt.so.13`) on the loader path: DGX OS installs it with CUDA 13
  (`libcublas-13-0`).
- NCCL 2.30.7 at the same path on every node (below).
- `libibverbs.so.1` and one RoCE v2 network between the systems (for four nodes, one 100 Gb/s port a node is enough), or
  four nodes cabled as a ring (`RING.md`); every node, the first included, reachable from the first over ssh without a
  password at its address in `NODES`.
- Disk, at the same path on every node (below): the checkpoint, 390 GB (363 GiB) for eight nodes, 344.6 GB (321 GiB)
  for four; and the 3-bit expert bank, 277.9 GB, which every start needs.
- Eight nodes: 112 GiB of memory available on every node before the start (115 with DFlash2), as `free -g` shows it:
  the ranks use about 108 GiB (111 with DFlash2) with `CONTEXT_LENGTH` 819,200 and `KV_SIZE` 1,638,400. Each
  100,000 positions of `KV_SIZE` fewer need 2.7 GiB less. `tf-glm53 check` refuses a node with less. With the 4-bit KV
  cache (`KV=fp4`) eight nodes take `CONTEXT_LENGTH` 1,048,576 and `KV_SIZE` 2,097,152: the ranks used about 94 GiB
  (measured, with a prompt of 1,018,993 tokens); check's estimate is for the 8-bit cache, so set `NEED_GB=100` there.
- Four nodes: memory sized for this server alone (`T4.md`); with DFlash2 the lowest free memory measured was about
  3.2 GiB a node, at 512K.
- `python3` on every node, for the log keeper (DGX OS has it), with the archive and with the image alike.
- Other widths: disk for the expert banks you use ("The control panel", below).
- In containers ("Running in containers"): Docker and the NVIDIA container toolkit on every node instead of cuBLAS,
  libibverbs and NCCL.

## Before the first start

NCCL: NVIDIA publishes NCCL 2.30.7 as a Python wheel. pip (package `python3-pip`) only unpacks the library, and
nothing runs Python. On every node:

```sh
pip3 install --no-deps --target ~/nccl-2.30.7 nvidia-nccl-cu13==2.30.7
```

`NCCL_LIB` in `glm53.conf` names `~/nccl-2.30.7/nvidia/nccl/lib/libnccl.so.2` by default. NCCL 2.30.7 built from
source (NVIDIA's DGX Spark playbook "NCCL for Two Sparks", https://build.nvidia.com/spark/nccl) works too: set
`NCCL_LIB` to it. On a switch the wheel runs, but the build with the AArch64 fence is recommended: without the fence a
proxy thread on Arm can hang with no error (`RING.md`, "The fenced NCCL"). A ring requires that build.

The checkpoint and the 3-bit bank are both needed; the bank is under "The control panel", below. The checkpoint: Fiore-lab's EXL3 checkpoint for the number of nodes on Hugging Face,
`fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks` for eight nodes or `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks` for four. Download it
once with the Hugging Face CLI (`python3 -m venv` needs the `python3-venv` package), then copy it to the other nodes
over the RoCE network. For eight nodes:

```sh
python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install -U huggingface_hub
D=~/models/GLM-5.3-EXL3-8x-DGX-Sparks
~/hf-cli/bin/hf download fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks --local-dir $D
for n in NODE2 NODE3 NODE4 NODE5 NODE6 NODE7 NODE8; do    # the other seven addresses in NODES
  rsync -a --mkpath -e "ssh -c aes128-gcm@openssh.com" $D/ $n:$D/ &
done; wait
```

For four nodes, `T4.md` gives the same commands with the four-node checkpoint and the other three nodes.

To check the copies, run `cd ~/models/GLM-5.3-EXL3-8x-DGX-Sparks && sha256sum -c --quiet SHA256SUMS` on every
node: a few minutes each, silent when every file matches (or, once this directory is deployed, `./tf-glm53 verify`
does it on every node at once). `WEIGHTS` in `glm53.conf` names the directory
(`~/models/GLM-5.3-EXL3-8x-DGX-Sparks` here).

Every weight file, the checkpoints and the expert banks, can also be made from Z.AI's official BF16 checkpoint
without any of Fiore-lab's files, on GPU hosts: `glm53/weights/README.md` in the source repository
(<https://github.com/fiore-labo/ExpertAtlas-GLM5.3/tree/main/glm53/weights>; the archive does not carry that directory).

## Start

```sh
tar xzf tensorfold-VERSION-linux-aarch64.tar.gz
cd tensorfold-VERSION-linux-aarch64
sha256sum -c SHA256SUMS
cp glm53.conf.example glm53.conf    # nodes, network names, NCCL, weights, drafter, port
./tf-glm53 deploy                   # the same directory on the other nodes
./tf-glm53 up                       # the control panel: open the address it prints, start the engine there
./tf-glm53 down                     # stops the engine, as the panel's Stop does
./tf-glm53 start                    # starts it again with the last start's plan, as the panel's Start does
./tf-glm53 -d dflash2 start         # another start with the DFlash2 drafter (Drafters, below)
```

`up` starts the control panel's launcher on the first node and prints the panel's address,
`http://FIRST-NODE:LAUNCHER_PORT/panel`. The first time, the panel opens on the node check: it finds how the nodes
are cabled (a ring, or a switch whose servers you choose), checks them and registers them in `glm53.nodes.conf`;
from then on it opens on the launch settings, where the weights' locations can be set too. A start from the panel loads the routed experts from the expert banks at the
widths of the plan chosen there ("The control panel", below); `start` takes the last start's plan (none yet: every expert at 3 bits) and waits until the
engine runs. `./tf-glm53 engine-up` by hand does the same as `start`. With the launcher's plan it checks every node, starts the ranks, waits for the
server and runs the smoke test (`./tf-glm53 engine-down` stops every rank); a start from the panel runs it so.

For four nodes, copy `glm53.t4.conf.example` instead of `glm53.conf.example` (next section). `tf-glm53` neither uses
`sudo` nor changes system settings. Its smoke test checks that the server answers and reuses a cached prompt; it is
not a measure of accuracy.

## Settings for four and eight nodes

`tf-glm53` runs tensor parallelism 4 or 8 by the number of addresses in `NODES`. The two settings examples differ in
these lines; the network names, NCCL, the drafter and the server are set the same way in both:

| Setting | `glm53.conf.example` (eight nodes) | `glm53.t4.conf.example` (four nodes) |
|---|---|---|
| `NODES` | eight addresses, rank 0 first | four addresses, rank 0 first |
| `WEIGHTS` | `~/models/GLM-5.3-EXL3-8x-DGX-Sparks` | `~/models/GLM-5.3-EXL3-4x-DGX-Sparks` |
| `KV` | `fp8` (the default) | `fp4` |
| `CONTEXT_LENGTH` | 819200 | 737280 |
| `KV_SIZE` | 1638400 | 837280 |
| `DCP` | 2 | 2 |
| `ROWS` / `SP_PARTS` | 8192 / 4 (the defaults) | 16384 / 4 |

`KV` is the format of the KV cache: FP8, or 4 bits. `DCP` is the decode context parallelism, and `ROWS` and `SP_PARTS`
are the rows of a prompt chunk and the sequence-parallel parts a chunk is split into. `CONTEXT_LENGTH` and
`KV_SIZE` are under "Conversations kept between requests" below. These are the examples' values; `T4.md` explains the
four-node ones.

`glm53.nodes.conf`, which the node check writes, is read after `glm53.conf` and wins: once it exists, `TOPOLOGY`,
`NODES`, `SOCKET_IFNAME` and `IB_HCA` come from it. Every setting of `glm53.conf`, its default, who reads it and when a
change takes effect: `CONFIG.md`. A `glm53.conf` written for an earlier release is still read: its `CONTEXT` is read
as `CONTEXT_LENGTH`, and `CONTEXT` + `STORE` as `KV_SIZE` (`CONFIG.md`).

## A first request

The server speaks the OpenAI chat API at `http://HOST:PORT/v1` with the model named by `NAME` (`glm-5.3`, port 8000 in
`glm53.conf.example`). On the first node:

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3", "max_tokens": 300,
  "messages": [{"role": "user", "content": "Say hello in one short sentence."}]}'
```

The reply's `message` holds the model's reasoning in `reasoning_content`, then the answer in `content`; `max_tokens`
counts both. From another machine, send the request to `http://FIRST-NODE:PORT` (with the default `HOST=0.0.0.0`).

## Drafters

The engine drafts a few tokens a round and the model checks them; the reply is the model's own choice in every case.
`DRAFTER_KIND` in `glm53.conf` picks what drafts. The panel's launch screen picks it for each start, and `-d KIND`
before `start` or `engine-up` for one start without editing the file (`./tf-glm53 -d dflash2 start`,
`./tf-glm53 -d mtp engine-up`); `./tf-glm53 status` shows what the last start used:

- `mtp` (default): the checkpoint's own MTP layer, part of the GLM-5.3 weights and under their license. It needs a
  checkpoint that has the layer; `tf-glm53 check` refuses one without it. At most `MTP_K` drafts a round, 3 by default
  (four were no faster than three); a round's drafts stop where the product of their probabilities falls under
  `TF_GLM53_MTP_CONF` (0.7 by default, 0 turns it off).
- `dflash2`: incoai/GLM-5.3-DFlash2, in the directory `DRAFTER`. **It is not included in this archive or the
  repository because of its license**: its model card states CC BY-NC-ND 4.0 (non-commercial use, no derivatives),
  for research and evaluation, with commercial licensing from Inco AI. If your use is covered by that license, "Using
  DFlash2" below sets it up; decode then runs faster still (Speed).
- `none`: no drafts.

### Using DFlash2

1. Read the license on the model card, https://huggingface.co/incoai/GLM-5.3-DFlash2, and check that your use is
   covered. The engine reads the published files as they are: at the start every rank takes its share of the
   drafter's weights and quantizes it to 4 bits in memory. Nothing is converted beforehand and nothing is written to
   disk.
2. Download it once on the first node, at the revision checked for this archive, with the Hugging Face CLI set up for
   the checkpoint. The repository needs no account or token; its weights are one file, `model.safetensors`
   (4,918,859,112 bytes). Use a directory of its own (`--local-dir`), not the Hugging Face cache: the directory in
   `DRAFTER` must hold `config.json` and `model.safetensors` themselves.

   ```sh
   R=~/models/GLM-5.3-DFlash2
   ~/hf-cli/bin/hf download incoai/GLM-5.3-DFlash2 --revision 425aa615ce320caac34400208b30808c8f14f76c --local-dir $R
   ```

3. Copy it to the same path on the other nodes. `tf-glm53 deploy` copies this archive's directory, not the drafter:

   ```sh
   for n in NODE2 NODE3 NODE4 NODE5 NODE6 NODE7 NODE8; do    # the other addresses in NODES (three on four nodes)
     rsync -a --mkpath -e "ssh -c aes128-gcm@openssh.com" $R/ $n:$R/ &
   done; wait
   ```

4. Check the files on every node, the first included. Every rank reads its own node's copy, and `tf-glm53 check` only
   looks for `config.json` on the node it runs on; it does not compare the drafter between nodes.

   ```sh
   for n in NODE1 NODE2 NODE3 NODE4 NODE5 NODE6 NODE7 NODE8; do    # every address in NODES
     printf '%s  %s\n' \
       f59e1da17d41d24a1aba588aecee1607788adb34a03805f2c883add8ca954e9b config.json \
       3105f14043bef642baa49a7d533fdf0b8b2895737ec84b6305601da662656161 model.safetensors |
       ssh $n "cd $R && sha256sum -c --quiet" && echo "ok $n" || echo "FAIL $n"
   done
   ```

   Every node must print `ok`.
5. In `glm53.conf`, with the same path:

   ```sh
   DRAFTER_KIND=dflash2
   DRAFTER=~/models/GLM-5.3-DFlash2
   ```

   With `DRAFTER` set and `DRAFTER_KIND` left at `mtp`, DFlash2 is picked for one start on the launch screen or with
   `./tf-glm53 -d dflash2 start`. While the engine is stopped the launcher reads the edit by itself.
   `MTP_K` is not used with DFlash2. On eight nodes every node needs 115 GiB available before the start instead of
   112 (What each node needs); `tf-glm53 check` counts it. The drafter's state of recent conversations
   is kept too (`DRAFT_SAVES`, Conversations kept between requests).
6. Start as usual: `./tf-glm53 down` first if it runs, then start it from the panel (or `./tf-glm53 start`).
   `tf-glm53 check` prints `drafter: dflash2 (PATH)`, and so does a start, in `tf-glm53 logs launcher`; the engine's
   own start-up line says `drafts: on` whatever the drafter.
7. To see that it drafts, read `completion_tokens_total` and `rounds_total` from `curl -s http://127.0.0.1:8000/health`
   (no key needed) before and after a request. The growth of the first over the growth of the second is the tokens
   the request kept a round: 1 means no draft was kept. For coding prompts like those under Speed, DFlash2 kept 4.3
   to 5.0.

To go back to the MTP layer, set `DRAFTER_KIND=mtp`, then `./tf-glm53 down` and start again; for one start, pick MTP
on the launch screen or `./tf-glm53 -d mtp start`.

## Conversations kept between requests

The context length (`CONTEXT_LENGTH` in `glm53.conf`) is the number of positions one request may reach (prompt and
reply), at most the model's 1,048,576. The KV size (`KV_SIZE`) is the whole KV cache in positions, shared by every
conversation: the context length for the request, the rest for earlier conversations' KV.

The KV cache is one pool of `KV_SIZE` positions in blocks of 256 positions (`KV_SIZE` at least `CONTEXT_LENGTH` + 512);
every conversation's KV is a table of those blocks. Earlier conversations stay in whichever blocks are free; nothing
is reserved for them separately. A request resumes from the longest cached prefix of any conversation, not only of
the one just before it. A conversation's next turn continues its own blocks. A new conversation that begins like an
earlier one (the same system prompt, such as the one of about 27K tokens some agent harnesses send with every
request, or a sub-agent) shares the earlier conversation's full blocks below the point where they differ, so they
are held once, and computes only from that block on. When the pool is full, the least recently used conversations
give up blocks from their ends first.

Each cached block is checked against the request's tokens (hashes of sampled rows) before it is reused. On any doubt
every conversation is dropped and the request is computed from the start ("cold"); a block is never reused wrongly.
With DFlash2 the pool also keeps the drafter's state of up to `DRAFT_SAVES` conversations (8 by default), so their
next turns draft at full speed. The reply is the model's own; a resumed request can differ in its last bits
from one computed from position 0 (the rounding order differs).

Byte Resume (`TF_GLM53_BYTE_RESUME`, `on` by default) keeps the cache when a model cut a text differently from the
tokenizer: the next turn's prompt then holds the same bytes under other token ids, and the match by ids would stop
there. Past the first id that differs, the prompt is compared with the kept conversations in bytes, and where the
bytes agree it takes their ids, so the cache resumes past the place. The text does not change; the reply may differ
slightly from one computed from the tokenizer's ids. `log` only reports the places, `off` turns it off.

`tf-glm53 status` prints the pool, and so does a start, in `tf-glm53 logs launcher`; with the four-node settings:

```text
conversation cache (pool): 837280 positions in 3271 blocks of 256, shared by up to 64 conversations (context 737280)
```

The server's `/health` then holds a `pool` object: `blocks`, `free_blocks`, `slots` (the conversations kept),
`shared_blocks` (blocks held by more than one conversation), `evicted_total` (conversations dropped to make room) and
`cold_total` (requests computed cold).

## Several conversations at once

By default one request runs at a time. `MULTI` (1 to 4 in `glm53.conf`, or on the launch screen) keeps up to that many
conversations live and decodes them together, with the MTP layer, DFlash2 or no drafter, at a cost in node memory
(`BEFORE-START.md`). A request that finds no room in the conversation cache beside the others waits up to `MULTI_WAIT`
seconds, then is refused as busy.

While they run, the panel or `POST /v1/schedule` sets how the GPU is shared when a prompt is computed beside
conversations that are decoding (`MULTI_DECODE_SHARE`), and a short prompt can go before the longer ones
(`MULTI_SHORT_PROMPT`, `MULTI_SHORT_PREEMPT`). `CONFIG.md` gives each one.

## The thinking: its limit and organizing

`max_tokens` counts the thinking too. By default the thinking may keep up to 90 % (`THINKING_SHARE`) of what the prompt
leaves of the context length in the context, or of `max_tokens` when that is smaller, and at that point the engine closes
the thinking at a line break, so the rest stays for the answer. A request that names no `max_tokens` gets 70 % of the
context length (`MAX_TOKENS_SHARE`).

The limit is on what the thinking keeps in the context, not on what it writes. Near the limit (10 % before it,
`THINKING_FOLD_MARGIN`) the engine asks the model to rewrite its thinking and replaces the thinking in the context with
the result (`THINKING_FOLD_AT_BUDGET`, on by default); the model then decides whether to go on or close. So a reply can
write more than the limit. Organizing can also run midway (`THINKING_FOLD`, off by default) or when the model
closes its thinking (`THINKING_FOLD_AT_CLOSE_SHARE`, off by default). The client still gets all of the reasoning.
`THINKING_NOTICE` (off by default) tells the model how much it has used.

The settings marked **live** in `CONFIG.md` change while the engine runs, from the panel's [ REASONING CONTROL ] or
`POST /v1/thinking`, and are written to `glm53.conf`. A request's own fields come first: `thinking_budget`,
`thinking_fold`, `thinking_fold_at_budget`, `thinking_fold_margin` and `thinking_fold_at_close` (in tokens).
`AGENT-API.md` has the details and measurements.

## The control panel (Expert Atlas)

On four or eight nodes, the routed experts of layers 3 to 77 (19,200, 256 a layer) can each be loaded at 2, 3, 4 or 5
bits, expert by expert, from four banks; the control panel chooses the widths and changes them while the engine runs.
The first time, the panel's node check registers the nodes (the head and three or seven workers) in
`glm53.nodes.conf`. The 3-bit bank must be on every node whatever a start uses, and a start that names no widths loads
every expert at 3 bits.
On eight nodes each node holds half of the routed experts, so wider widths fit. `T4.md`, "The control panel", gives
which banks four nodes need and what fits there.

**The banks.** Each holds the routed experts at one width, with its manifest (`<id>.json`: every file's size, header
and sha256): `fiore-lab/GLM-5.3-EXL3-Experts-2bit`, `-3bit`, `-4bit` and `-5bit` on Hugging Face (187.3, 277.9,
368.5 and 459.1 GB). `expert-plan/banks.json` names each bank's directory, the same on every node
(`~/models/GLM-5.3-EXL3-Experts-<N>bit`; change `dir` there for another place, then `./tf-glm53 deploy`). The 3- and
4-bit banks for four nodes:

```sh
for w in 3 4; do
  D=~/models/GLM-5.3-EXL3-Experts-${w}bit
  ~/hf-cli/bin/hf download fiore-lab/GLM-5.3-EXL3-Experts-${w}bit --local-dir $D
  for n in NODE2 NODE3 NODE4; do    # the other three addresses in NODES
    rsync -a --mkpath -e "ssh -c aes128-gcm@openssh.com" $D/ $n:$D/ &
  done; wait
done
./tf-glm53 verify-banks 3 4    # every file against its manifest's sha256 on every node, at low priority (minutes)
```

The eight-node checkpoint's routed experts are the 4-bit bank file for file: with that checkpoint as `WEIGHTS`, put the
4-bit manifest into its directory on every node
(`hf download fiore-lab/GLM-5.3-EXL3-Experts-4bit glm53-exl3-experts-4bit.json --local-dir ~/models/GLM-5.3-EXL3-8x-DGX-Sparks`)
and set the 4-bit bank's `dir` to `~/models/GLM-5.3-EXL3-8x-DGX-Sparks`. The launcher checks every width's bank on
every node (the manifest, and every layer file at its size) when it starts, at most once a minute while the page is
open, and before every start; the panel
offers only the widths every node holds, and a start or a change that needs another is refused before anything
changes. A start checks each layer file's size and header against the manifest; `verify-banks` reads the files whole.

**The launcher.** It runs on the first node and stays on, whether the engine runs or not:

```sh
./tf-glm53 up                 # start it and print the panel's address (./tf-glm53 launcher starts it too)
./tf-glm53 launcher status    # also ./tf-glm53 launcher stop
```

While the engine is stopped the launcher reads `glm53.conf` again by itself after an edit. `LAUNCHER_HOST` and
`LAUNCHER_PORT` take effect when it starts again (`./tf-glm53 launcher stop`, then `./tf-glm53 up`; the engine may keep
running). `CONFIG.md` lists what it reads.

Open `http://FIRST-NODE:LAUNCHER_PORT/panel`, the address `up` prints: port 8010 in `glm53.t4.conf.example`
(without `LAUNCHER_PORT`, `PORT` + 10). The launcher listens on every address by default (`LAUNCHER_HOST=0.0.0.0`)
and has no key, so anyone who reaches it can start, stop and swap the engine: keep the nodes on a closed network, or
set `LAUNCHER_HOST=127.0.0.1` and open the panel from another machine through an ssh tunnel
(`ssh -L 8010:127.0.0.1:8010 FIRST-NODE`). The panel's other calls go to the engine through the launcher, so the page
has one address whether the engine runs or not; an engine started while the launcher runs (on every address, the
default) sends a browser that opens its own `/panel`, on `PORT`, to the launcher's. Start and stop the engine from the
panel while you use it. `PANEL-GUIDE.md` describes the screens; the page is in English or Japanese (by the browser's
language, or Display > Language).

- **A start** from the panel: the launcher checks every node (reachable, no engine or other GPU process, the plan
  fits its memory), writes the widths as an expert plan (`launch/plan-<time>.json`), copies it to every node and runs
  `tf-glm53 engine-up` with it (`tf-glm53 start` makes the same start with the last start's plan); the page shows
  each layer as it loads. Every routed expert then comes from the banks; the checkpoint gives the other weights.
- **The drafter** is chosen on the launch screen for each start, beside the KV format: MTP (the checkpoint's layer),
  DFlash2 (offered only when `DRAFTER` in `glm53.conf` holds its model on every node, "Using DFlash2" above) or none;
  `DRAFTER_KIND` is the default. Each has its own memory: DFlash2 leaves about 4 GiB less a node than MTP, so the
  plan is made again for the drafter picked.
- **The context length** of a start (Context length on the launch screen) is chosen there too: 128K, 256K, 512K,
  `CONTEXT_LENGTH` or the longest that fits (at most 1,048,576 tokens); a length the plan's memory cannot hold is cut
  to what fits.
- **What each node keeps for the OS** (`RESERVE_GB`, 4 GiB by default, 2 to 32) is chosen there too. The panel does not
  ask for a KV size: the KV cache takes all the room the plan leaves (the context length for one request, the rest for
  earlier conversations), and the launch screen's KV rows (KV FP4 / KV FP8) show it in tokens. `tf-glm53 start` takes
  the context length and the KV size from `glm53.conf`, and `-e KV_SIZE=N` on the `docker run` line sets it for that start. The plan sizes the KV
  cache with what each node keeps; the engine's own memory check at the start keeps the same amount. The KV cache is
  set aside at the start and does not grow while the engine runs.
- **The nodes.** While the engine is stopped the workers' addresses can be changed on the launch screen; the first
  node stays. Check copies this directory to them (`tf-glm53 deploy`) and runs `tf-glm53 check` on the list; Save writes
  `glm53.nodes.conf` once that list passed, keeping the previous file beside it. `glm53.conf` is not touched.
- **The weights.** The launch screen's [ WEIGHTS ] names the checkpoint, the banks and DFlash2's model; Check and save
  writes them once every node has what they name (`WEIGHTS` and `DRAFTER` in `glm53.conf`, the banks' directories in
  `expert-plan/banks.json`).
- **The engine settings screen** (the switch at the top right) changes the settings that need a restart, and offers
  Save and restart (`PANEL-GUIDE.md`).
- **Allocations and usage records.** The engine counts the experts each prompt row and each decoded token picks
  (`EXPERT_LIVE=1` in `glm53.conf`, the default; a start from the panel always counts). An engine started without the
  counts (`EXPERT_LIVE=0`) still shows the live panel with Stop, but no picks and no expert optimisation. A
  plan's widths are saved as an allocation in `ATLAS_DIR/allocs/user/<name>.json` (`atlas/` in this directory by
  default) with the usage record of the runs it was saved from: the record the run started with plus that run's
  picks. Nothing is saved by itself: Save and stop at a stop, or Save while the engine runs, names it. The name
  offered is the allocation the run started from, saved over (its old version goes to `ATLAS_DIR/allocs/trash/`).
  The launch screen opens with a new plan at the last run's shares (before any run: all 3 bits); Open brings back the
  last run's allocation or a saved one bit for bit, and a start from it uses its bits and the banks it was saved
  with. Plans rank the experts by the usage record of the allocation they come from, the most used getting the
  widest bits; a plan without a record ranks them by the counts the page ships with, measured by Fiore-lab over 12
  kinds of work (coding, agents, Japanese and English text, Linux operations, security).
- **Expert optimisation**, while the engine runs, re-sorts the loaded experts by the run's usage record, each width
  keeping its number of experts, so the mean width of the experts picked rises at the same memory. The panel's Count
  picks the counts: decode, prefill, or both (an expert's decode share weighed 0.7 and its prefill share 0.3: the
  weights that kept the most of the text the model writes, `ATLAS-TESTS.md`). The change starts once no request runs
  or waits (the page tries again until then); the engine changes the layers in place on every rank, drops the
  conversations it keeps (their KV was computed with the old experts) and serves again, and requests that arrive
  meanwhile wait; a full change of 75 layers took 83 to 90 s on four nodes. The layers whose widths shrink go first,
  then those that grow, and every node needs the largest layer's new copy on top of what the layers before it grew,
  plus 2 GiB (`TF_GLM53_SWAP_MARGIN_GIB`): the estimate says whether each node has it, and the page refuses the change
  with the node and the amounts when one does not (more kept for the OS at the next start leaves room, since the
  KV takes all the room the plan leaves; a start from glm53.conf without the panel, a smaller `KV_SIZE`). A rank stuck past
  `TF_GLM53_SWAP_DEADLINE_S` (600 s by default) ends its process, and `tf-glm53`'s watcher stops every rank. The
  result is not saved by itself: Save keeps it. A start loads the widths it is given and never re-sorts.
- With `MULTI` set (`glm53.conf`) the engine changes no widths while it runs: expert optimisation saves the
  estimate's widths under a name (`opt-<time>`), for a stop and a start with it.
- The page's settings (colours, the circle's order and the rest) are one file for every browser:
  `launch/panel-settings.json` (`TF_PANEL_SETTINGS` in `EXTRA_ENV` moves it).
- `LAUNCH_SNAPSHOTS=1` in `glm53.conf` writes each node's GPU, memory and log tail before and after each start and
  stop to `snapshots/`; `LAUNCH_DROP_CACHES=1` drops the page cache on every node before a start (it needs `sudo`
  without a password). Both are off by default.

## When a rank stops

Every start (`tf-glm53 engine-up`, which a start from the panel and `tf-glm53 start` run) leaves a watcher that
checks the ranks every 5 seconds. If some ranks are gone while others still run (twice in a row), it stops them all,
since the others would wait forever, and names the ranks that were gone in its log (`tf-glm53 logs watch`). Start
again from the panel or with `tf-glm53 start`. Start and stop the ranks with `tf-glm53` only: ranks started by hand
have no watcher, and if one fails while starting, the others wait in a collective until they are stopped.

A start that fails (a rank stops, the server is not ready within `READY_TIMEOUT`, the smoke test fails, or Ctrl-C)
stops every rank before `tf-glm53 engine-up` returns (a start from the panel shows why on the page, `tf-glm53 start`
prints it). Each node keeps its rank's log, and the start before's; `tf-glm53 logs R` follows rank R's (`R.prev`: the
start before's).

The logs are kept in memory, nothing on disk: `tf-glm53 engine-up` starts a log keeper on every node, and `up` one
on the first node for the launcher (`tools/logkeep.py`, `python3` with its standard library; it reads and answers on
an abstract unix socket, so it has no file either), and each rank's output, the watcher's, the launcher's and
`tf-glm53`'s own go into it. Each log keeps its last `LOG_KEEP_MB` (8 by default); older lines go, and a reboot or a
stopped keeper loses them. The keeper stays on after a stop, so a failed start's logs can still be read; the next
start keeps one start before. It answers the user it runs as only (an abstract socket has no file permissions, so it
checks the peer's user). A change of `LOG_LISTEN` or `LOG_KEEP_MB` takes effect at the engine's next start, which
starts the keepers anew (the lines they held are lost).
`tf-glm53 logs R | R.prev | watch | launcher` follows one. To keep the logs anywhere for longer, set `LOG_LISTEN=HOST:PORT`: each node's keeper
then also answers, read-only and with no key, `GET /streams` (the logs it holds) and `GET /logs?stream=rank0&after=N`
(the lines after line N, numbered), for a monitor on another host that polls them and stores them ("Serving other
machines"). The server leaves out the access line of the requests a monitor or the panel repeats (`/health`,
`/metrics`, `/v1/experts/*`, `/v1/memory`), so the logs stay small.

Each rank runs with `oom_score_adj` 1000. A node that runs out of memory then loses its rank to the kernel's OOM
killer and the watcher stops the others; the node itself stays up. (A shell started over ssh inherits sshd's -1000,
which the OOM killer never picks: a rank left at that value lets the node hang instead.)

## Running in containers

The shortest way is one line, `docker run ... IMAGE up`: `QUICKSTART.md`. This section is the same by hand.

The release is also a container image: the archive's files in `/opt/glm53` with what the server needs at run time
(cuBLAS 13, libibverbs and its providers, NCCL 2.30.7 with the AArch64 fence). With `RUNTIME=docker` and `IMAGE` in `glm53.conf`, `tf-glm53`
starts each rank as a container of `IMAGE` on its node over ssh. The nodes then need Docker and the NVIDIA container
toolkit (DGX OS has both) instead of cuBLAS, libibverbs and NCCL, and the user running `tf-glm53` must be allowed to
run `docker` on every node (the `docker` group). `tf-glm53`, the control panel's launcher and the settings stay on the
first node's host, as with the archive; the weights and the expert banks stay on the hosts.

```sh
I=ghcr.io/fiore-labo/expertatlas-glm5.3:r3.3-102
docker pull $I                                                # on every node
mkdir glm53 && docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/glm53":/out $I install /out
cd glm53 && cp glm53.conf.example glm53.conf                  # or glm53.t4.conf.example; set RUNTIME=docker, IMAGE=$I
./tf-glm53 deploy && ./tf-glm53 check && ./tf-glm53 up        # the control panel: start the engine there
```

- `install` writes the release's files (the same as the unpacked archive) to the directory. `./tf-glm53 pull` runs
  `docker pull IMAGE` on every node.
- The image can be built from the archive instead: in a freshly unpacked archive,
  `docker build --build-context fenced=DIR -f docker/Dockerfile -t glm53-tensorfold:r3.3-102 .`, with the fenced NCCL in
  `DIR` (`RING.md`; the build fails without it). The build refuses any file the archive's
  `SHA256SUMS` does not list, so nothing of an installation goes into the image. Copy it to the other nodes with
  `docker save` and `docker load`.
- `tf-glm53 check` checks, on every node, that `IMAGE` is there with the first node's layers, that a container reaches
  the GPU (`--gpus all`), and that the image holds this directory's release (`NG: image-not-this-release` otherwise).
- Each rank's container has the GPU, `/dev/infiniband`, the host network and IPC,
  `--ulimit memlock=-1 --ulimit nofile=1048576:1048576`, `--oom-score-adj 1000`, the host user and an init. Mounted
  at the same paths: `WEIGHTS`, `DRAFTER` (DFlash2) and every bank of `expert-plan/banks.json` the node has, read-only;
  `launch/`, writable. Docker keeps no log of them (`--log-driver none`): their output goes to the node's log keeper.
  Containers are named `glm53-p<PORT>-r<rank>` and removed when they stop.
  `tf-glm53 down`, `status`, the watcher and the control panel work on them as on ranks run from the host.
- The engine reads the NCCL at `/opt/nccl-fenced`, on a switch and on a ring. NVIDIA's wheel stays at `/opt/nccl` for
  comparison and is not read by default.
- Measured on four nodes (the eight-node checkpoint, a context length of 131,072 tokens, MTP): the start took 4
  minutes, as on the host; the smoke test passed; `down` stopped every rank in 4 s; with one rank's container killed,
  the watcher stopped the others and every container was gone in 43 s.
- A `systemctl daemon-reload` on a host can make running GPU containers lose the GPU (`nvidia-smi` fails in them):
  `tf-glm53 down` and start again (the panel, or `tf-glm53 start`).

## Serving other machines

The server listens on every address by default (`HOST=0.0.0.0`) and speaks the OpenAI chat API, and the Messages API
("Claude Code"), so other machines on the network reach it at `http://FIRST-NODE:PORT`. The API has no key: anyone who can reach it can use it (`tf-glm53
check` says so), and `HOST=127.0.0.1` keeps it to the first node.
The server speaks plain HTTP as well, so keep it on a closed network, or put TLS and access control (a reverse proxy)
in front. The control panel's launcher listens on every address too, with no key ("The control panel").
The server's own page (`http://FIRST-NODE:PORT/panel`) gets the experts' counts and the memory (`/v1/experts/*`,
`/v1/memory`) only on the first node itself, so from another machine it shows a message and nothing more: open the
control panel's page (`http://FIRST-NODE:LAUNCHER_PORT/panel`), whose calls the launcher makes from the first node.
`/v1/thinking` and `/v1/schedule` answer the same way, on the first node only: from another machine, send them to the
launcher's port. `AGENT-API.md` gives the launcher's control API.

A monitor on another host can read the server without reaching the API: set `METRICS_LISTEN=HOST:PORT` (for example
`0.0.0.0:9894`). The first node then also answers `GET /metrics` (Prometheus text), `GET /health` and
`GET /v1/experts/usage` (the experts' use counts, no text) on that address, and nothing else (any other request gets
404), while the API stays on `HOST`. With `LOG_LISTEN` every node's log keeper answers its logs too (above, "When a rank
stops"): a monitor that polls both keeps the record, and the nodes write nothing to disk. `tf-glm53 status` prints
the address. Keep the port on a closed network: it shows no prompt or reply, but it shows the load and the token counts.
The `tensorfold_health:*_total` counters include the running request and never fall back, so a speed is the difference
of two reads: prefill tokens/s = Δ`prefill_tokens_total` / Δ`prefill_seconds_total`, decode = Δ`completion_tokens_total`
/ Δ`decode_seconds_total`, cache hits = Δ`cached_tokens_total` / Δ`prompt_tokens_total`, draft acceptance =
Δ`accepted_total` / Δ`drafted_total`. `tensorfold:requests_prefilling` and `tensorfold:requests_decoding` give the
running request's phase, `tensorfold:prefill_tokens_remaining` what its prefill has left, `tensorfold:pool_*` the
conversation cache and `tensorfold:engine_info` the settings.

## Claude Code

The server also answers the Anthropic Messages API (`POST /v1/messages` and `/v1/messages/count_tokens`), so
Claude Code can use it in place of Anthropic's service. `claude-glm53` in this archive starts Claude Code against a server:

```sh
ENDPOINT=http://FIRST-NODE:8000 ./claude-glm53      # Claude Code's own arguments may follow
```

It reads the server's model and context length from `/health`, sends one short request, and starts Claude Code with
every model alias (the small fast model too) on the server's model, a context window of the server's context length
less `MAX_OUT` (40,000) for a reply, and a 30-minute stream timeout. Run it where Claude Code is installed (it needs
`curl` and `python3` as well); on macOS Claude Code needs the Local Network permission to reach an address on the LAN.
Anthropic does not support pointing Claude Code at another model through a gateway: use it at your own risk.

What the server does for it, with no setting:

- Each request is rebuilt from its blocks, so Claude Code's cache markers and its two forms of content (a string or
  text blocks) render the same prompt, and the attribution line at the head of Claude Code's system prompt
  (`x-anthropic-billing-header`), whose value changes with each conversation, is dropped. Measured: a conversation's
  later requests read 99 % or more of their prompt from the conversation cache, a new conversation's first request
  about 84 %.
- While a long prompt is read nothing else is sent, so the server sends a keep-alive event after each 10 s of silence;
  Claude Code drops a stream after 300 s without one.
- An image becomes the note `[image omitted: this model reads text only]` (`tf-glm53` sets
  `TF_MESSAGES_OMIT_IMAGES=1`): the model reads text only, and a refused image would refuse every later request of the
  conversation that holds it (Claude Code sends the image a Read of an image file returns).

Limits:

- By default requests run one at a time (`MULTI`: "Several conversations at once"). Claude Code's background requests (without thinking, a few tokens long, for example to
  name a conversation) wait for the main one, and they begin unlike the main conversation's requests, so they share
  no cache with them; they share it with each other.
- Structured output (`output_config.format`) is refused (400); Claude Code carries on without it.
- Effort: GLM-5.3 has low, high and max; minimal runs as low, medium as high and xhigh as max (Claude Code sends xhigh).
  Claude Code asks for at most 32,000 tokens a reply (`CLAUDE_CODE_MAX_OUTPUT_TOKENS` raises it). The thinking's limit
  keeps the rest of it for the answer; with `THINKING_SHARE=0`, a reply that reaches the limit while still thinking ends
  with no text.

## Speed

**Measured with release r2-102, with the GPU clock capped at 2.1 GHz (2,100 MHz) on every node.** The engine has changed
since, and these numbers have not been measured again for this release. They load each checkpoint's own widths, not the
default start's 3 bits for every expert. A coding prompt not yet cached, one request at a time (reasoning effort low,
up to 4,000 reply tokens). Eight nodes: the settings of `glm53.conf.example` (FP8 KV cache, `CONTEXT_LENGTH` 819,200) and `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks`
(routed experts 4 bits, other linear layers 5 bits). Four nodes: the settings of `glm53.t4.conf.example` (4-bit KV
cache, `CONTEXT_LENGTH` 737,280) and `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks`. MTP: the median of two runs; DFlash2: one
run on eight nodes, the median of two on four. The model advanced 3.4 to 3.5 tokens a round with MTP and 4.6 to 4.8
with DFlash2. Speeds differ with the environment (the checkpoint, the prompt, the clocks): take these numbers as a guide.

| Nodes | Prompt | MTP (k=3): time to first token / decode | DFlash2: time to first token / decode |
|---|---|---|---|
| 8 | 128K | 82.8 s / 47.5 tokens/s | 85.2 s / 53.2 |
| 8 | 256K | 172.9 s / 45.7 | 177.6 s / 52.6 |
| 8 | 512K | 376.5 s / 42.8 | 386.0 s / 47.1 |
| 4 | 128K | 120.1 s / 35.3 | 121.9 s / 42.9 |
| 4 | 256K | 253.3 s / 33.2 | 256.9 s / 40.7 |
| 4 | 512K | 565.4 s / 32.2 | 571.5 s / 37.8 |

## When something goes wrong

`tf-glm53 check`, also the first step of every start (`engine-up`), lists each node's problems after `NG:`. A node
with every item NG is one the first node cannot reach: `ssh -o BatchMode=yes ADDRESS true` must work from the first
node for every address in `NODES`.

| `NG:` | What to do |
|---|---|
| `files` | `tf-glm53 deploy` again |
| `weights` | the checkpoint is missing at `WEIGHTS`, or its file names, sizes or `config.json` differ from the first's; `tf-glm53 verify` checks the contents |
| `NCCL(...)` | NCCL is not at `NCCL_LIB` on that node (Before the first start) |
| `cuBLAS-13` | install CUDA 13's cuBLAS (`libcublas-13-0`) |
| `libibverbs` | install `libibverbs1` and `ibverbs-providers` |
| the RDMA device | `IB_HCA` is not an RDMA device there: `rdma link` lists them, `ibdev2netdev` names their ports |
| `RoCE-v2-GID` | no RoCE v2 entry with an IPv4 address on that port: give the port its address, or set `GID_INDEX` |
| `GPU-busy` | another process holds the GPU (`nvidia-smi`) |
| `memory(N<M GiB)` | free memory on that node, or lower `KV_SIZE` |
| `GPU-unreadable` | `nvidia-smi` fails on that node: check the driver, and restart the node if it stays so |
| `python3(the log keeper)` | install `python3` on that node |
| `rendezvous(...)` | `SOCKET_IFNAME` has no IPv4 address on that node |
| `lane-down(...)` | on a ring: a link's half is down on that node: check the cable (`RING.md`) |
| `ring: NG: ...` | the cabling is not a ring: the message says what to fix (`RING.md`) |
| `NCCL-not-the-fenced-build(...)` | a ring needs the fenced NCCL, and its sha256 in `NCCL_SHA256` (`RING.md`) |
| `image-missing`, `image-differs-from-the-first-node` | `tf-glm53 pull`, or copy the image with `docker save` and `docker load` |
| `image-not-this-release` | this directory and `IMAGE` are different releases: `install` again from `IMAGE`, or set `IMAGE` to this release |
| `image-without-...` | this image has no fenced NCCL: use the release's image, or build it with `--build-context fenced=DIR` (`RING.md`) |
| `docker-GPU(...)` | `--gpus all` fails: check the NVIDIA container toolkit on that node |
| `bank 3 bit: nothing at ...` | put the 3-bit bank there, on every node: every start needs it |

When a start fails it stops every rank and says why (a start from the panel or `tf-glm53 start`: also in
`tf-glm53 logs launcher`); each node keeps its rank's log in memory (`tf-glm53 logs R`):

- `a rank stopped (N/M running)`: the lines above it name the rank and its log's last errors. `NCCL WARN` lines about
  connections: on a switch, `NODES` must be the addresses on the RoCE network, `SOCKET_IFNAME` and `IB_HCA` that network's
  names on every node, and the nodes must reach each other there on any port (a ring: `RING.md`). `cannot listen on`: `PORT` is in use on that
  node. `PoolValuesDiffer`: the ranks were started with different pool settings and refuse to start; start them all
  with `tf-glm53` from one `glm53.conf`.
- `not ready after N s`: a start takes about two and a half minutes with the checkpoint on a local NVMe drive; slower
  disks or network storage take longer: raise `READY_TIMEOUT`. A rendezvous that never completes waits too: check
  that `MASTER_PORT` is free on the first node.
- `the smoke test failed`: the failed question shows the reply it got; rank 0's log shows the request.

While serving, `tf-glm53 status` shows every rank, its free memory, the drafter of the last start, the conversation
cache and the server's `/health`. If the server stops answering while every rank still runs, `tf-glm53 down` and
`tf-glm53 start`: the watcher stops the ranks only when some are gone. If a rank refuses a write to the pool outside
the request's own blocks (`PoolWriteRefused` in the logs), every rank stops together at the end of that request; start
again from the panel or with `tf-glm53 start`.

## Limits

- Four or eight GB10 systems (tensor parallelism 4 or 8), on a switch or, for four, a ring. Two- and three-node forms are
  being tested. Other GPU models and other rank counts are not supported.
- By default one request runs at a time (`MULTI` above). Up to 32 others wait in arrival order, chat-title requests (and those sent with
  `"priority": "background"`) behind the rest. Past 32 a request is refused with "the engine is busy; retry shortly":
  a 503, or an error event in a streamed reply.
- With the MTP drafter, the rows accepted in a reply's last round are not yet written into the MTP layer's cache when
  the conversation is kept. The pool keeps the conversation only up to where the MTP layer's cache is complete, a few
  positions earlier, and computes those positions again. The reply is the model's own.
- The engine's start-up line says `drafts: on` whatever the drafter: it means requests may use drafts. `tf-glm53`
  prints the drafter in use after it.
- A container needs the GPU, `/dev/infiniband`, the host network, an unlimited
  locked-memory limit and a high open-file limit (`--ulimit memlock=-1 --ulimit nofile=1048576:1048576`).
