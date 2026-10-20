English | [日本語](RUNTIME.ja.md) | [简体中文](RUNTIME.zh-CN.md)

# Running full GLM-5.3 from this archive

This archive serves full GLM-5.3 (`glm_moe_dsa`, 755B) on four or eight NVIDIA DGX Spark systems, one rank per GB10,
with tensor parallelism 4 or 8. It needs no Python, compiler or source checkout. It is Fiore-lab's unofficial
derivative of TensorFold 1.0.2, not a release of the TensorFold project. This page covers both node counts; `T4.md`
gives what differs with four nodes, with its measurements.

## What the archive holds

- `bin/tensorfold-native`: the server, with its CUDA kernels built for GB10 (sm_121) inside. Linux aarch64, glibc 2.28
  or later.
- `lib/libroce_proxy.so`: the RDMA proxy of the decode collectives (needs `libibverbs.so.1`).
- `tf-glm53`, `glm53.conf.example` (eight nodes) and `glm53.t4.conf.example` (four nodes): start, stop and check the
  ranks.
- `RUNTIME.md` (this page) and `T4.md` (four nodes), each also in Japanese (`.ja.md`) and Simplified Chinese
  (`.zh-CN.md`).
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
counts both. From another machine, with `HOST` and `API_KEY_FILE` set, send the request to `http://HOST:PORT` and add
`-H "Authorization: Bearer KEY"` (see [Serving other machines](#serving-other-machines) for the key file).

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

- `pool` (the default): one KV pool of `CONTEXT` + `STORE` positions in blocks of 256 positions; every conversation's
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

## When a rank stops

`tf-glm53 up` leaves a watcher that checks the ranks every 5 seconds. If some ranks are gone while others still run
(twice in a row), it stops them all, since the others would wait forever, and names the ranks that were gone in
`logs/watch.log`. Start again with `tf-glm53 up`. Start and stop the ranks with `tf-glm53` only: ranks started by hand
have no watcher, and if one fails while starting, the others wait in a collective until they are stopped.

A start that fails (a rank stops, the server is not ready within `READY_TIMEOUT`, the smoke test fails, or Ctrl-C)
stops every rank before `tf-glm53 up` returns. Each node keeps its rank's log in `logs/rankR.log`, and the start before
in `logs/rankR.log.prev`; `tf-glm53 logs R` follows rank R's.

## Serving other machines

The server listens on `127.0.0.1` by default and speaks the OpenAI chat API. To serve other machines, set `HOST` and
`API_KEY_FILE`: **an API key file is required**, and `tf-glm53` refuses a non-loopback address without one. The server
speaks plain HTTP: put TLS in front or keep it on a closed network, or the key travels in the clear.

On the first node:

1. Make the key file, readable by you alone:

   ```sh
   umask 077; mkdir -p ~/.config/tf-glm53
   echo "lan: $(openssl rand -hex 32)" > ~/.config/tf-glm53/api-keys
   ```

   The file holds one key a line, `KEY` or `LABEL: KEY` (`#` starts a comment). A key is text without whitespace; a
   label is 1 to 64 letters, digits, dots, underscores or hyphens. Give each client a line of its own to tell them
   apart, or to take one key away without touching the others.
2. In `glm53.conf`, set `HOST=0.0.0.0` (every address; or the first node's address on your network) and
   `API_KEY_FILE` to the file's full path, for example `API_KEY_FILE=/home/YOU/.config/tf-glm53/api-keys`.
3. Start with `./tf-glm53 up` (stop a running server with `./tf-glm53 down` first). Its check tests the key file
   before anything starts.

On a client, the base URL is `http://FIRST_NODE:PORT/v1` and the API key is the key without its label:

```sh
curl -s http://FIRST_NODE:8000/v1/chat/completions -H "Authorization: Bearer KEY" \
  -H 'Content-Type: application/json' -d '{"model": "glm-5.3", "max_tokens": 300,
  "messages": [{"role": "user", "content": "Say hello in one short sentence."}]}'
```

The key file must be a regular file that other users cannot read (`chmod 600`), or the server refuses to start;
`tf-glm53 check` tests both. It is read on the first node only: `tf-glm53 deploy` does not copy it to the other nodes.
A request without a valid key is refused ("A valid API key is required"); `/health` answers without one.

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
| `API_KEY_FILE ...` | as the line says |

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
  supported. Four nodes need the four-node checkpoint and `KV=fp4`.
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
