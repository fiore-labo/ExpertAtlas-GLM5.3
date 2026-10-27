English | [日本語](RUNTIME.ja.md) | [简体中文](RUNTIME.zh-CN.md)

# Running full GLM-5.3 from this archive

This archive serves full GLM-5.3 (`glm_moe_dsa`, 755B) on four or eight NVIDIA DGX Spark systems, one rank per GB10,
with tensor parallelism 4 or 8. The server needs no Python, compiler or source checkout; the control panel's launcher
runs on the system `python3` with its standard library only. It is Fiore-lab's unofficial
derivative of TensorFold 1.0.2, not a release of the TensorFold project. This page covers both node counts; `T4.md`
gives what differs with four nodes, with its measurements.

## What the archive holds

- `bin/tensorfold-native`: the server, with its CUDA kernels built for GB10 (sm_121) inside. Linux aarch64, glibc 2.28
  or later.
- `lib/libroce_proxy.so`: the RDMA proxy of the decode collectives (needs `libibverbs.so.1`).
- `tf-glm53`, `glm53.conf.example` (eight nodes) and `glm53.t4.conf.example` (four nodes): start, stop and check the
  ranks.
- The control panel (four or eight nodes): `launcher/tf_launcher.py` (the launcher), `panel.html` (its page; the
  server holds the same page), `tools/quant/expert_plan.py` (writes the expert plans), `expert-plan/banks.json` (the
  expert banks and where they are) and `PANEL-GUIDE.md` (the panel's screens; also in Japanese).
- `docker/Dockerfile` and `docker/entry.sh`: the release as a container image ("Running in containers").
- `RUNTIME.md` (this page) and `T4.md` (four nodes), each also in Japanese (`.ja.md`) and Simplified Chinese
  (`.zh-CN.md`); `ATLAS-TESTS.md` (the expert re-sort's measurements; also in Japanese).
- `AI-GUIDE.md`: the whole system on one page for an AI assistant that helps you set it up, use it or find a fault
  (also in Japanese).
- `LICENSE`, `NOTICE`, `LICENSES/`, `THIRD_PARTY_NOTICES.md`, `VERSION`, `SHA256SUMS`.

It holds no model weights, no draft model and no NCCL.

## What each node needs

- DGX OS (Ubuntu 24.04) with the NVIDIA driver 580 or later, which supplies `libcuda.so.1`.
- cuBLAS 13 (`libcublas.so.13`, `libcublasLt.so.13`) on the loader path: DGX OS installs it with CUDA 13
  (`libcublas-13-0`).
- NCCL 2.30.7 at the same path on every node (below).
- `libibverbs.so.1` and one RoCE v2 network between the systems (for four nodes, one 100 Gb/s port a node is enough);
  every node, the first included, reachable from the first over ssh without a password at its address in `NODES`.
- Disk for the checkpoint, at the same path on every node (below): 390 GB (363 GiB) for eight nodes, 344.6 GB
  (321 GiB) for four.
- Eight nodes: 112 GiB of memory available on every node before the start (115 with DFlash2), as `free -g` shows it:
  the ranks use about 108 GiB (111 with DFlash2) with `CONTEXT` and `STORE` at 819,200; with DFlash2 the pool (the
  default conversation cache) needs about 0.5 GiB more a rank than the slot cache (`CACHE=slots`). Each 100,000 positions of `STORE`
  fewer need 2.7 GiB less. `tf-glm53 check` refuses a node with less.
- Four nodes: memory sized for this server alone (`T4.md`); with DFlash2 the lowest free memory measured was about
  3.2 GiB a node, at 512K.
- The control panel (four or eight nodes): disk for the expert banks it uses (below) and `python3` on every node (DGX
  OS has it).
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
`NCCL_LIB` to it.

The checkpoint: Fiore-lab's EXL3 checkpoint for the number of nodes on Hugging Face,
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
./tf-glm53 up                       # checks, starts the ranks, waits, runs the smoke test
./tf-glm53 -d dflash2 up            # another start with the DFlash2 drafter (Drafters, below)
./tf-glm53 down
```

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
| `CONTEXT` | 819200 | 737280 |
| `STORE` | 819200 | 100000 |
| `CACHE` | `pool` (the default) | `pool` |
| `DCP` | 2 (the default) | 2 |
| `ROWS` / `SP_PARTS` | 8192 / 4 (the defaults) | 16384 / 4 |

`KV` is the format of the KV cache: FP8, or 4 bits. `DCP` is the decode context parallelism, and `ROWS` and `SP_PARTS`
are the rows of a prompt chunk and the sequence-parallel parts a chunk is split into. `CONTEXT`, `STORE` and `CACHE`
are under "Conversations kept between requests" below. Four nodes need the four-node checkpoint, `KV=fp4` and the
values of `glm53.t4.conf.example`; `T4.md` explains them.

## A first request

The server speaks the OpenAI chat API at `http://HOST:PORT/v1` with the model named by `NAME` (`glm-5.3`, port 8000 in
`glm53.conf.example`). On the first node:

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3", "max_tokens": 300,
  "messages": [{"role": "user", "content": "Say hello in one short sentence."}]}'
```

The reply's `message` holds the model's reasoning in `reasoning_content`, then the answer in `content`; `max_tokens`
counts both. From another machine, with `HOST` set, send the request to `http://HOST:PORT`.

## Drafters

The engine drafts a few tokens a round and the model checks them; the reply is the model's own choice in every case.
`DRAFTER_KIND` in `glm53.conf` picks what drafts. `-d KIND` before the command picks it for one start without
editing the file (`./tf-glm53 -d dflash2 up`, `./tf-glm53 -d mtp up`), and `./tf-glm53 status` shows what the last
start used:

- `mtp` (default): the checkpoint's own MTP layer, part of the GLM-5.3 weights and under their license. It needs a
  checkpoint that has the layer; `tf-glm53 check` refuses one without it. `MTP_K` drafts a round, 3 by default (four
  were no faster than three).
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

   With `DRAFTER` set and `DRAFTER_KIND` left at `mtp`, `./tf-glm53 -d dflash2 up` uses DFlash2 for one start.
   `MTP_K` is not used with DFlash2. On eight nodes every node needs 115 GiB available before the start instead of
   112 (What each node needs); `tf-glm53 check` counts it. With the pool, the drafter's state of recent conversations
   is kept too (`DRAFT_SAVES`, Conversations kept between requests).
6. Start as usual: `./tf-glm53 down` first if it runs, then `./tf-glm53 up`. `tf-glm53 check` and `up` print
   `drafter: dflash2 (PATH)`; the engine's own start-up line says `drafts: on` whatever the drafter.
7. To see that it drafts, read `completion_tokens_total` and `rounds_total` from `curl -s http://127.0.0.1:8000/health`
   (no key needed) before and after a request. The growth of the first over the growth of the second is the tokens
   the request kept a round: 1 means no draft was kept. For coding prompts like those under Speed, DFlash2 kept 4.3
   to 5.0.

To go back to the MTP layer, set `DRAFTER_KIND=mtp`, then `./tf-glm53 down` and `./tf-glm53 up`.

## Conversations kept between requests

`CONTEXT` is the number of positions one request may reach (prompt and reply). `STORE` is the number of positions
kept beyond them for earlier conversations. `CACHE` in `glm53.conf` picks how they are kept:

- `pool` (the default): one KV pool of `CONTEXT` + `STORE` positions in blocks of 256 positions (`STORE` 512 at
  least); every conversation's
  KV is a table of those blocks. A request resumes from the longest cached prefix of any conversation, not only of the
  one just before it. A conversation's next turn continues its own blocks. A new conversation that begins like an
  earlier one (the same system prompt, such as the one of about 27K tokens some agent harnesses send with every
  request, or a sub-agent) shares the earlier conversation's full blocks below the point where they differ, so they
  are held once, and computes only from that block on. When the pool is full, the least recently used conversations
  give up blocks from their ends first.
- `slots`: one store per conversation, `STORE` positions beside the window for earlier conversations.

Each cached block is checked against the request's tokens (hashes of sampled rows) before it is reused. On any doubt
every conversation is dropped and the request is computed from the start ("cold"); a block is never reused wrongly.
With DFlash2 the pool also keeps the drafter's state of up to `DRAFT_SAVES` conversations (8 by default), so their
next turns draft at full speed. The reply is the model's own either way; a resumed request can differ in its last bits
from one computed from position 0 (the rounding order differs), as with the slot cache.

`tf-glm53 up` and `status` print the pool; with the four-node settings:

```text
conversation cache (pool): 837280 positions in 3271 blocks of 256, shared by up to 64 conversations (context 737280)
```

The server's `/health` then holds a `pool` object: `blocks`, `free_blocks`, `slots` (the conversations kept),
`shared_blocks` (blocks held by more than one conversation), `evicted_total` (conversations dropped to make room) and
`cold_total` (requests computed cold).

## The control panel (Expert Atlas)

On four or eight nodes, the routed experts of layers 3 to 77 (19,200, 256 a layer) can each be loaded at 2, 3, 4 or 5
bits, expert by expert, from four banks; the control panel chooses the widths and changes them while the engine runs.
The node count is set on the launch screen's Nodes (the head and three or seven workers, checked, then saved to
`NODES`); the launcher refuses other counts, and `./tf-glm53 up` still starts the nodes with the checkpoint's widths.
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
./tf-glm53 launcher           # start it; also ./tf-glm53 launcher status, ./tf-glm53 launcher stop
```

Open `http://127.0.0.1:LAUNCHER_PORT/panel`: `127.0.0.1:8010` in `glm53.t4.conf.example` (without `LAUNCHER_PORT`,
`PORT` + 10). From another machine, forward the port (`ssh -L 8010:127.0.0.1:8010 FIRST-NODE`, then
`http://127.0.0.1:8010/panel`): the launcher listens on loopback only and refuses another `LAUNCHER_HOST`, since
anyone who reached it could start, stop and swap the engine. The panel's other calls go to the engine through the
launcher, so the page has one address whether the engine runs or not. Start and stop the engine from the panel while
you use it. `PANEL-GUIDE.md` describes the screens; the page is in English or Japanese (by the browser's language, or
Display > Language).

- **A start** from the panel: the launcher checks every node (reachable, no engine or other GPU process, the plan
  fits its memory), writes the widths as an expert plan (`launch/plan-<time>.json`), copies it to every node and runs
  `tf-glm53 up` with it; the page shows each layer as it loads. Every routed expert then comes from the banks; the
  checkpoint gives the other weights.
- **The drafter** is chosen on the launch screen for each start, beside the KV format: MTP (the checkpoint's layer),
  DFlash2 (offered only when `DRAFTER` in `glm53.conf` holds its model on every node, "Using DFlash2" above) or none;
  `DRAFTER_KIND` is the default. Each has its own memory: DFlash2 leaves about 4 GiB less a node than MTP, so the
  plan is made again for the drafter picked.
- **The context** of a start is chosen there too: 128K, 256K, 512K, `CONTEXT` or the longest that fits (at most
  1,048,576 tokens); a length the plan's memory cannot hold is cut to what fits.
- **The store** (`STORE`: the tokens kept beside the context for earlier conversations) is chosen there too: `STORE`,
  a size, or all the memory the context leaves; and **what each node keeps for the OS** (`RESERVE_GB`, 4 GiB by
  default, 2 to 32). The plan sizes the KV cache with both; the engine's own memory check at the start keeps the same
  amount. The KV cache is set aside at the start and does not grow while the engine runs.
- **The nodes.** While the engine is stopped the workers' addresses (`NODES` after the first) can be changed on the
  launch screen; the first node stays. Check copies this directory to them (`tf-glm53 deploy`) and runs
  `tf-glm53 check` on the list; Save writes `NODES` in `glm53.conf` once that list passed, keeping the previous file
  beside it (`glm53.conf.bak-<time>`).
- **Allocations and usage records.** The engine counts the experts each prompt row and each decoded token picks. A
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
  meanwhile wait; a full change of 75 layers took 83 to 90 s on four nodes. A rank stuck past
  `TF_GLM53_SWAP_DEADLINE_S` (600 s by default) ends its process, and `tf-glm53`'s watcher stops every rank. The
  result is not saved by itself: Save keeps it. A start loads the widths it is given and never re-sorts.
- With `TF_GLM53_MULTI` in `EXTRA_ENV` the engine changes no widths while it runs: expert optimisation saves the
  estimate's widths under a name (`opt-<time>`), for a stop and a start with it.
- The page's settings (colours, the circle's order and the rest) are one file for every browser:
  `launch/panel-settings.json` (`TF_PANEL_SETTINGS` in `EXTRA_ENV` moves it).
- `LAUNCH_SNAPSHOTS=1` in `glm53.conf` writes each node's GPU, memory and log tail before and after each start and
  stop to `snapshots/`; `LAUNCH_DROP_CACHES=1` drops the page cache on every node before a start (it needs `sudo`
  without a password). Both are off by default.

## When a rank stops

`tf-glm53 up` leaves a watcher that checks the ranks every 5 seconds. If some ranks are gone while others still run
(twice in a row), it stops them all, since the others would wait forever, and names the ranks that were gone in
`logs/watch.log`. Start again with `tf-glm53 up`. Start and stop the ranks with `tf-glm53` only: ranks started by hand
have no watcher, and if one fails while starting, the others wait in a collective until they are stopped.

A start that fails (a rank stops, the server is not ready within `READY_TIMEOUT`, the smoke test fails, or Ctrl-C)
stops every rank before `tf-glm53 up` returns. Each node keeps its rank's log in `logs/rankR.log`, and the start before
in `logs/rankR.log.prev`; `tf-glm53 logs R` follows rank R's.

Each rank runs with `oom_score_adj` 1000. A node that runs out of memory then loses its rank to the kernel's OOM
killer and the watcher stops the others; the node itself stays up. (A shell started over ssh inherits sshd's -1000,
which the OOM killer never picks: a rank left at that value lets the node hang instead.)

## Running in containers

The release is also a container image: the archive's files in `/opt/glm53` with what the server needs at run time
(cuBLAS 13, libibverbs and its providers, NCCL 2.30.7). With `RUNTIME=docker` and `IMAGE` in `glm53.conf`, `tf-glm53`
starts each rank as a container of `IMAGE` on its node over ssh. The nodes then need Docker and the NVIDIA container
toolkit (DGX OS has both) instead of cuBLAS, libibverbs and NCCL, and the user running `tf-glm53` must be allowed to
run `docker` on every node (the `docker` group). `tf-glm53`, the control panel's launcher and the settings stay on the
first node's host, as with the archive; the weights and the expert banks stay on the hosts.

```sh
I=ghcr.io/fiore-labo/expertatlas-glm5.3:r3-102
docker pull $I                                                # on every node
mkdir glm53 && docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/glm53":/out $I install /out
cd glm53 && cp glm53.conf.example glm53.conf                  # or glm53.t4.conf.example; set RUNTIME=docker, IMAGE=$I
./tf-glm53 deploy && ./tf-glm53 check && ./tf-glm53 up
```

- `install` writes the release's files (the same as the unpacked archive) to the directory. `./tf-glm53 pull` runs
  `docker pull IMAGE` on every node.
- The image can be built from the archive instead: in a freshly unpacked archive,
  `docker build -f docker/Dockerfile -t glm53-tensorfold:r3-102 .`. The build refuses any file the archive's
  `SHA256SUMS` does not list, so nothing of an installation goes into the image. Copy it to the other nodes with
  `docker save` and `docker load`.
- `tf-glm53 check` checks, on every node, that `IMAGE` is there with the first node's layers, that a container reaches
  the GPU (`--gpus all`), and that the image holds this directory's release (`NG: image-not-this-release` otherwise).
- Each rank's container has the GPU, `/dev/infiniband`, the host network and IPC,
  `--ulimit memlock=-1 --ulimit nofile=1048576:1048576`, `--oom-score-adj 1000`, the host user and an init. Mounted
  at the same paths: `WEIGHTS`, `DRAFTER` (DFlash2) and every bank of `expert-plan/banks.json` the node has, read-only;
  `logs/` and `launch/`, writable. Containers are named `glm53-p<PORT>-r<rank>` and removed when they stop.
  `tf-glm53 down`, `status`, the watcher and the control panel work on them as on ranks run from the host.
- Measured on four nodes (the eight-node checkpoint, 131,072 tokens of context, MTP): the start took 4 minutes, as
  on the host; the smoke test passed; `down` stopped every rank in 4 s; with one rank's container killed, the watcher
  stopped the others and every container was gone in 43 s.
- A `systemctl daemon-reload` on a host can make running GPU containers lose the GPU (`nvidia-smi` fails in them):
  `tf-glm53 down` and `up` again.

## Serving other machines

The server listens on `127.0.0.1` by default and speaks the OpenAI chat API. To serve other machines, set `HOST`. The
API has no key: on a non-loopback address `tf-glm53` warns and starts, and the API is open to anyone who can reach it.
The server speaks plain HTTP as well, so keep it on a closed network, or put TLS and access control (a reverse proxy)
in front. The control panel's launcher stays on loopback either way; another machine reaches the panel through an ssh
tunnel ("The control panel").

A monitor on another host can read the server without reaching the API: set `METRICS_LISTEN=HOST:PORT` (for example
`0.0.0.0:9894`). The first node then also answers `GET /metrics` (Prometheus text) and `GET /health` on that address,
and nothing else (any other request gets 404), while the API stays on `HOST`. `tf-glm53 status` prints
the address. Keep the port on a closed network: it shows no prompt or reply, but it shows the load and the token counts.
The `tensorfold_health:*_total` counters include the running request and never fall back, so a speed is the difference
of two reads: prefill tokens/s = Δ`prefill_tokens_total` / Δ`prefill_seconds_total`, decode = Δ`completion_tokens_total`
/ Δ`decode_seconds_total`, cache hits = Δ`cached_tokens_total` / Δ`prompt_tokens_total`, draft acceptance =
Δ`accepted_total` / Δ`drafted_total`. `tensorfold:requests_prefilling` and `tensorfold:requests_decoding` give the
running request's phase, `tensorfold:prefill_tokens_remaining` what its prefill has left, `tensorfold:pool_*` the
conversation cache and `tensorfold:engine_info` the settings.

## Speed

**Measured with the GPU clock capped at 2.1 GHz (2,100 MHz) on every node.** A coding prompt not yet cached, one
request at a time (reasoning effort low, up to 4,000 reply tokens), this archive with `CACHE=pool`. Eight nodes: the
settings of `glm53.conf.example` (FP8 KV cache, `CONTEXT` 819,200) and `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks` (routed
experts 4 bits, other linear layers 5 bits). Four nodes: the settings of `glm53.t4.conf.example` (4-bit KV cache,
`CONTEXT` 737,280) and `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks`. MTP: the median of two runs; DFlash2: one run on eight
nodes, the median of two on four. The model advanced 3.4 to 3.5 tokens a round with MTP and 4.6 to 4.8 with DFlash2. Your
numbers depend on the checkpoint, the prompt and the clocks.

| Nodes | Prompt | MTP (k=3): time to first token / decode | DFlash2: time to first token / decode |
|---|---|---|---|
| 8 | 128K | 82.8 s / 47.5 tokens/s | 85.2 s / 53.2 |
| 8 | 256K | 172.9 s / 45.7 | 177.6 s / 52.6 |
| 8 | 512K | 376.5 s / 42.8 | 386.0 s / 47.1 |
| 4 | 128K | 120.1 s / 35.3 | 121.9 s / 42.9 |
| 4 | 256K | 253.3 s / 33.2 | 256.9 s / 40.7 |
| 4 | 512K | 565.4 s / 32.2 | 571.5 s / 37.8 |

## When something goes wrong

`tf-glm53 check`, also the first step of `up`, lists each node's problems after `NG:`. A node with every item NG is one
the first node cannot reach: `ssh -o BatchMode=yes ADDRESS true` must work from the first node for every address in
`NODES`.

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
| `memory(N<M GiB)` | free memory on that node, or lower `STORE` |

When `up` fails it stops every rank and says why; each node keeps its rank's log (`logs/rankR.log`, `tf-glm53 logs R`):

- `a rank stopped (N/M running)`: the lines above it name the rank and its log's last errors. `NCCL WARN` lines about
  connections: `NODES` must be the addresses on the RoCE network, `SOCKET_IFNAME` and `IB_HCA` that network's names on
  every node, and the nodes must reach each other there on any port. `cannot listen on`: `PORT` is in use on that
  node. `PoolValuesDiffer`: the ranks were started with different pool settings and refuse to start; start them all
  with `tf-glm53` from one `glm53.conf`.
- `not ready after N s`: a start takes about two and a half minutes with the checkpoint on a local NVMe drive; slower
  disks or network storage take longer: raise `READY_TIMEOUT`. A rendezvous that never completes waits too: check
  that `MASTER_PORT` is free on the first node.
- `the smoke test failed`: the failed question shows the reply it got; rank 0's log shows the request.

While serving, `tf-glm53 status` shows every rank, its free memory, the drafter of the last start, the conversation
cache and the server's `/health`. If the server stops answering while every rank still runs, `tf-glm53 down` and
`tf-glm53 up`: the watcher stops the ranks only when some are gone. If a rank refuses a write to the pool outside the
request's own blocks (`PoolWriteRefused` in the logs), every rank stops together at the end of that request; start
again with `tf-glm53 up`. A start with `DCP=1` warns in the logs that the prompt chunks run without sequence-parallel
parts (`SP_PARTS` is not used then).

With `CACHE=slots`, `up` and `status` print `slot cache: WARNING: no store` when a rank lacked the memory for `STORE`
positions at the start: the server answers, but a request resumes only the conversation just before it. Free memory
on the nodes or lower `STORE`, then `tf-glm53 down` and `tf-glm53 up`.

## Limits

- Four or eight GB10 systems only (tensor parallelism 4 or 8); other GPU models and other rank counts are not
  supported. Four nodes need the four-node checkpoint and `KV=fp4` (with the control panel, either checkpoint and the
  banks, `T4.md`).
- One request runs at a time. Up to 32 others wait in arrival order, chat-title requests (and those sent with
  `"priority": "background"`) behind the rest. Past 32 a request is refused with "the engine is busy; retry shortly":
  a 503, or an error event in a streamed reply.
- With the MTP drafter, the rows accepted in a reply's last round are not yet written into the MTP layer's cache when
  the conversation is kept. With `CACHE=slots`, a later request that resumes there drafts a little less well at those
  few positions; the pool keeps the conversation only up to where the MTP layer's cache is complete, a few positions
  earlier, and computes those positions again. Either way the reply is the model's own.
- The engine's start-up line says `drafts: on` whatever the drafter: it means requests may use drafts. `tf-glm53`
  prints the drafter in use after it.
- In containers, `tf-glm53 check` cannot see GPU processes of the host or of other containers; the free-memory check
  still stops a start on a busy node. A container needs the GPU, `/dev/infiniband`, the host network, an unlimited
  locked-memory limit and a high open-file limit (`--ulimit memlock=-1 --ulimit nofile=1048576:1048576`).
