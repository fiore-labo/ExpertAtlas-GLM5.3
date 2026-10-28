# A guide for AI assistants: full GLM-5.3 on DGX Spark (release r3.1-102)

This page describes the whole system in one place: what it is, what each part does, how it is used, what it can do
and what it cannot. It is written for AI assistants that help someone understand, install, run or troubleshoot this
release, and reads the same for a person. Its numbers come from the other pages, which hold the full details and say
more where this page is short: `RUNTIME.md` (running it, both node counts), `T4.md` (four nodes), `PANEL-GUIDE.md`
(the control panel's screens), `ATLAS-TESTS.md` (the expert re-sort's measurements). In the source repository they
are `packaging/RUNTIME.md`, `glm53/release/T4.md`, `glm53/release/PANEL-GUIDE.md` and `glm53/release/ATLAS-TESTS.md`.
Japanese: `AI-GUIDE.ja.md`.

## 1. What it is

A server for the full GLM-5.3 model (755B parameters, architecture `glm_moe_dsa`) on four or eight NVIDIA DGX Spark
systems (one GB10 each), one rank per GB10, with tensor parallelism 4 or 8. It is Fiore-lab's unofficial derivative of
TensorFold 1.0.2 (an engine written in Zig with plain CUDA kernels), not a release of the TensorFold project. The
server needs no Python at run time.

| Fact | Value |
|---|---|
| Model | GLM-5.3, full size, EXL3-quantized checkpoints published by Fiore-lab on Hugging Face |
| Nodes | 4 or 8 DGX Spark (GB10) on one RoCE v2 network; nothing else is supported |
| Context | up to 1,048,576 tokens on both node counts. Eight nodes: `STORE` 1,048,576 beside it on the 4-bit KV cache (`KV=fp4`, `NEED_GB=100`), 589,824 on the 8-bit one. Four nodes: from the control panel with 70 % of the routed experts at 3 bits and 30 % at 4 (4-bit KV, `STORE` 512); the four-node checkpoint alone 819,200 (`STORE` 18,080). Measured with one prompt of about 1,019,000 tokens each. The 4-bit KV cache costs some accuracy (KL 0.0190 against 0.0156, `T4.md`). The examples ship 819,200 and 737,280 |
| API | OpenAI-compatible chat API (`/v1/chat/completions`) on every address (`0.0.0.0:8000`) by default |
| Requests | one runs at a time; up to 32 more wait in a queue |
| Drafters | MTP (the checkpoint's own layer, default), DFlash2 (optional, not included, separate licence), or none |
| Control panel | "Expert Atlas": per-expert bit widths (2/3/4/5 bits), a web page; four or eight nodes |
| Distribution | a container image (recommended) and an archive with the same files; both for Linux aarch64 |
| Licence | code Apache-2.0; the weights keep the GLM-5.3 License; DFlash2 keeps its own (CC BY-NC-ND 4.0) |

## 2. The parts

| File or directory (in the archive) | What it is |
|---|---|
| `bin/tensorfold-native` | the server: the GLM-5.3 engine and the HTTP API, CUDA kernels for GB10 (sm_121) built in |
| `lib/libroce_proxy.so` | the RDMA proxy the decode collectives use (needs `libibverbs.so.1`) |
| `tf-glm53` | the script that deploys, checks, starts, stops and watches the ranks; run it on the first node |
| `claude-glm53` | starts Claude Code against the server's Messages API (RUNTIME.md, "Claude Code") |
| `glm53.conf.example` / `glm53.t4.conf.example` | settings for eight / four nodes; one is copied to `glm53.conf` |
| `CONFIG.md` | every setting of `glm53.conf`: its default, who reads it and when a change takes effect |
| `launcher/tf_launcher.py` | the control panel's launcher (system `python3`, standard library only) |
| `panel.html` | the control panel's page (the server holds the same page) |
| `tools/quant/expert_plan.py` | writes the expert plans (which width each expert loads at) |
| `expert-plan/banks.json` | where each expert bank (one width) lives on the nodes |
| `RUNTIME.md`, `T4.md`, `PANEL-GUIDE.md`, `ATLAS-TESTS.md` | the documents (several also in Japanese and Simplified Chinese) |
| `AI-GUIDE.md`, `AI-GUIDE.ja.md` | this page (English, Japanese) |
| `docker/Dockerfile`, `docker/entry.sh` | the release as a container image (`install`, `serve`, `version`) |
| `VERSION`, `SHA256SUMS`, `LICENSE`, `NOTICE`, `LICENSES/`, `THIRD_PARTY_NOTICES.md` | release metadata and licences |

Not in the archive: the model weights, the expert banks, the DFlash2 drafter and NCCL. They are downloaded separately
(section 4).
The weights and the banks can also be made from Z.AI's official BF16 checkpoint without any of Fiore-lab's files, on
GPU hosts: `glm53/weights/README.md` in the source repository
(<https://github.com/fiore-labo/ExpertAtlas-GLM5.3/tree/main/glm53/weights>), not in the archive.

## 3. How it works, in short

- **Ranks.** One process (rank) on each GB10. Rank 0 runs on the first node in `NODES` and serves the HTTP API.
  Tensor parallelism splits every layer across the ranks; the KV cache is split across rank pairs (decode context
  parallelism, `DCP=2`).
- **Prefill and decode.** A prompt is read in chunks (prefill), then the reply is generated token by token (decode).
  Prefill speed is compute-bound; decode speed is bound by memory bandwidth.
- **Drafts.** Each decode round a drafter proposes a few tokens and the model checks them in one step; only tokens the
  model itself would choose are kept, so the reply is the model's own either way.
- **Conversation cache.** With `CACHE=pool` (default) one KV pool of `CONTEXT` + `STORE` positions, in blocks of 256,
  is shared by all conversations: a request resumes from the longest cached prefix of any conversation, and
  conversations that start alike (one system prompt) hold their common blocks once. Cached blocks are checked against
  the request's tokens before reuse; on any doubt every conversation is dropped and the request is computed from the
  start ("cold"), so a block is never reused wrongly.
- **Experts.** GLM-5.3 has 256 routed experts in each of layers 3 to 77 (19,200 in all) and picks 8 per layer for
  every token. The checkpoints store them quantized: the eight-node checkpoint all at 4 bits, the four-node checkpoint
  half at 3 bits (the least used) and half at 4 bits. The control panel can load each expert at its own width.

## 4. Two ways to run it

| | `./tf-glm53 up` | The control panel (`./tf-glm53 launcher`) |
|---|---|---|
| Nodes | 4 or 8 | 4 or 8 (chosen on the launch screen) |
| Expert widths | as stored in the checkpoint | chosen per expert from the banks (2, 3, 4, 5 bits) |
| Needs | the checkpoint | the checkpoint (its non-expert weights) and the banks of the widths used |
| Started from | the command line | a web page (start, stop, plan, live view, expert optimisation) |

Both use the same `glm53.conf`, the same server and the same API.

### In containers (the recommended way)

With `RUNTIME=docker` and `IMAGE` in `glm53.conf`, `tf-glm53` starts each rank as a container of the release image on
its node (one container a rank, over ssh from the first node). The nodes need Docker and the NVIDIA container toolkit
(DGX OS has both) instead of cuBLAS, libibverbs and NCCL; steps 1 and 2 below are not needed. `docker pull` the image
on every node (or `./tf-glm53 pull`), write the release's files to a directory on the first node with
`docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/glm53":/out IMAGE install /out`, then go on from step 4. The
weights and the banks stay on the hosts and are mounted read-only. `RUNTIME.md`, "Running in containers", has the
details; the image can also be built from the archive (`docker/Dockerfile`).

### Setting up (both)

1. Every node: DGX OS (Ubuntu 24.04), NVIDIA driver 580 or later, cuBLAS 13 (`libcublas-13-0`), `libibverbs`, the
   nodes on one RoCE v2 network, the first node reaching every node over ssh without a password.
2. NCCL 2.30.7 on every node: `pip3 install --no-deps --target ~/nccl-2.30.7 nvidia-nccl-cu13==2.30.7` (pip only
   unpacks the library; nothing runs Python).
3. The checkpoint, at the same path on every node: `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks` (390 GB, eight nodes) or
   `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks` (344.6 GB, four nodes). Download once with the Hugging Face CLI, copy with
   `rsync` to the other nodes, check with `./tf-glm53 verify`.
   The checkpoint can also be made from the official BF16 checkpoint (section 2, `glm53/weights/`).
4. Unpack the archive on the first node, `sha256sum -c SHA256SUMS`, copy the right example to `glm53.conf`, set
   `NODES`, `SOCKET_IFNAME`, `IB_HCA`, `NCCL_LIB`, `WEIGHTS`.
5. `./tf-glm53 deploy` (copies the directory to the other nodes), then `./tf-glm53 up` (checks every node, starts the
   ranks, waits for the server, runs a smoke test). A start takes about two and a half minutes from a local NVMe drive.
6. `./tf-glm53 down` stops every rank.

### Setting up the control panel

1. The expert banks the plans will use, on every node in the directories `expert-plan/banks.json` names (by default
   `~/models/GLM-5.3-EXL3-Experts-<N>bit`): on four nodes `fiore-lab/GLM-5.3-EXL3-Experts-3bit` and `-4bit` are the
   usual pair (277.9 and 368.5 GB); `-2bit` (187.3 GB) and `-5bit` (459.1 GB) are optional. The eight-node checkpoint
   can serve as the 4-bit bank (`RUNTIME.md`, "The control panel"). Check them with `./tf-glm53 verify-banks 3 4`.
2. `./tf-glm53 launcher` on the first node; open `http://FIRST-NODE:8010/panel` (without `LAUNCHER_PORT`, the port is
   `PORT` + 10). The launcher listens on every address by default and has no key: anyone who reaches it can start,
   stop and swap the engine (`LAUNCHER_HOST=127.0.0.1` keeps it to the node; then an ssh tunnel,
   `ssh -L 8010:127.0.0.1:8010 FIRST-NODE`). It stays on whether the engine runs or not; the page reaches the engine
   through it. It reads `glm53.conf` when it starts, so an edit needs `./tf-glm53 launcher stop` and
   `./tf-glm53 launcher` again.
3. On the page: choose a plan, the KV format, the context, the store, what each node keeps for the OS and the drafter;
   check the memory verdict; press Start (then Confirm). Stop from the page too.

## 5. Settings (`glm53.conf`)

| Setting | Meaning | Eight-node example / four-node example |
|---|---|---|
| `NODES` | the nodes' addresses on the RoCE network, rank 0 first | 8 / 4 addresses |
| `SOCKET_IFNAME`, `IB_HCA`, `GID_INDEX` | that network's interface and RDMA device on every node; the GID index (empty: chosen) | the same |
| `NCCL_LIB` | NCCL 2.30.7's `libnccl.so.2`, the same path on every node | the same |
| `WEIGHTS` | the checkpoint's directory, the same path on every node | the 8x / 4x checkpoint |
| `DRAFTER_KIND`, `MTP_K`, `DRAFTER` | the drafter (`mtp`, `dflash2`, `none`), MTP's drafts a round (3), DFlash2's directory | `mtp`, 3, empty |
| `NAME`, `HOST`, `PORT` | the model name in the API, the listen address and port | `glm-5.3`, `0.0.0.0`, 8000 |
| `METRICS_LISTEN` | an extra read-only address answering `/metrics` and `/health` only | off |
| `CONTEXT` | positions one request may reach, prompt and reply together | 819,200 / 737,280 |
| `STORE` | positions kept beyond `CONTEXT` for earlier conversations (pool: 512 at least) | 819,200 / 100,000 |
| `CACHE` | `pool` (shared blocks) or `slots` (one store per conversation) | `pool` |
| `KV` | the KV cache format, `fp8` or `fp4` (four nodes need `fp4`) | `fp8` / `fp4` |
| `DCP`, `ROWS`, `SP_PARTS` | decode context parallelism, prompt chunk rows, sequence-parallel parts a chunk | 2, 8,192, 4 / 2, 16,384, 4 |
| `NEED_GB` | the memory a node must have available before a start (less 2.7 GiB for each 100,000 of `STORE` below 819,200, plus 3 with DFlash2) | 112 (default) / 131 (asks 112 GiB) |
| `MASTER_PORT`, `READY_TIMEOUT` | the ranks' rendezvous port; seconds to wait for the server | 29551, 900 |
| `RUNTIME`, `IMAGE` | how the ranks run: `host` (default, from this directory) or `docker` (each in a container of `IMAGE`) | `host`, empty |
| `EXTRA_ENV` | engine variables, for example `TF_GLM53_MULTI=4` (several conversations decoded together; off by default) | empty |
| `LAUNCHER_HOST`, `LAUNCHER_PORT`, `ATLAS_DIR`, `RESERVE_GB` | the control panel: its address, port, saved allocations, memory kept for the OS | four-node example: `0.0.0.0`, 8010, `atlas/`, 4 |

## 6. Commands (`./tf-glm53`, run on the first node)

| Command | What it does |
|---|---|
| `deploy` | copies this directory to every node (same path) and checks `SHA256SUMS` there |
| `check` | checks nodes, files, weights, libraries, RDMA devices, GPUs and memory; lists problems after `NG:` |
| `verify` | checks the weights against their `SHA256SUMS` on every node (a few minutes) |
| `verify-banks [BITS...]` | checks the expert banks' files against their manifests |
| `up` | `check`, starts the ranks (last to first), waits for the server, runs the smoke test |
| `down` | stops every rank (TERM, then KILL after 30 s) |
| `status` | the ranks, free memory, the drafter, the conversation cache, the server's `/health` |
| `smoke` | three requests whose answers are checked |
| `logs [R]` | follows rank R's log (default 0); logs are `logs/rankR.log` on each node |
| `launcher [stop\|status]` | starts, stops or reports the control panel's launcher |
| `-d KIND` (before the command) | the drafter for this start (`mtp`, `dflash2`, `none`) |
| `pull` | `RUNTIME=docker`: `docker pull IMAGE` on every node |
| `-c FILE` (before the command) | another settings file |

`tf-glm53` uses no `sudo` and changes no system settings. The ranks are meant to be started and stopped only with it:
it leaves a watcher that stops every rank when some are gone (ranks waiting on a lost peer would otherwise wait
forever).

## 7. The API

| Path | Method | Notes |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI chat: streaming (`stream`, `stream_options.include_usage`), tools (`tools`, `tool_choice` `auto` or `none`) |
| `/v1/completions` | POST | plain text completion of a prompt (no chat template is added) |
| `/v1/models` | GET | the served model name |
| `/v1/tokenize`, `/v1/detokenize` | POST | token counts and ids, for a prompt or for messages rendered by the chat template |
| `/health` | GET | status, memory, live speeds, the conversation pool, token and draft counters |
| `/metrics` | GET | the same counters in Prometheus text |
| `/v1/experts/...`, `/v1/memory` | GET / POST | expert usage, bits and swaps, used by the control panel |

- The reply's `message.reasoning_content` holds the model's reasoning and `message.content` the answer; `max_tokens`
  counts both. `chat_template_kwargs` takes `{"enable_thinking": false}` (no reasoning) and `reasoning_effort`.
- Refused with 400: `n` other than 1, `logprobs`, `response_format` `json_object`, `tool_choice` `required`, images
  (text only), a `max_tokens` beyond the context. A streamed request that cannot fit ends with an error event.
- A client that disconnects cancels its request.
- No key: the API has none. By default it listens on every address (`HOST=0.0.0.0`), so anyone who can reach it can
  use it, and the server speaks plain HTTP, so text crosses the network in the clear unless the server stays on a
  closed network or TLS and access control are put in front (`HOST=127.0.0.1` keeps the API to the first node).
- Speeds from the counters: decode tokens/s = Δ`completion_tokens_total` / Δ`decode_seconds_total`; prefill tokens/s
  = Δ`prefill_tokens_total` / Δ`prefill_seconds_total`; tokens kept per draft round = Δ`completion_tokens_total` /
  Δ`rounds_total` (1 means no draft was kept); cache hits = Δ`cached_tokens_total` / Δ`prompt_tokens_total`. The
  counters include the running request and never fall back.

The control panel's launcher has its own API under `/v1/launch/` (status, plan, start, stop, usage, resort, check,
config, nodes, and the saved allocations: allocs, alloc, allocs/save, allocs/rename, allocs/delete); the page uses it,
and its fields may change between releases.

## 8. Expert Atlas (the control panel)

- **Banks**: the routed experts at one width each (2, 3, 4 or 5 bits), downloaded separately, each with a manifest.
  The page offers only the widths every node holds.
- **Plan (allocation)**: the width of each of the 19,200 experts. The launch screen opens with a new plan at the last
  run's shares (before any run: all 3 bits); the shares are typed per width or dragged on a curve. Open brings back
  the last run's allocation or a saved one bit for bit, and a start from it uses its bits and the banks it was saved
  with. The page shows each node's memory and a verdict (fits, tight, does not fit) before a start.
- **Usage records**: the engine counts the experts every prompt row (prefill) and every generated token (decode)
  picks. Each saved allocation keeps the usage record of the runs it was saved from; there is no common pool of all
  runs. A new plan, a run, a save: that run's record only. An opened allocation, a run, a save: its record plus that
  run's. Plans rank experts by the record of the allocation they come from (a plan without one, by counts shipped
  with the page, measured over 12 kinds of work). The most used experts get the widest bits.
- **Saving**: nothing is saved by itself. At a stop, Save and stop saves the bits loaded and the usage record under a
  name (by default the allocation the run started from, saved over; the old version goes to
  `ATLAS_DIR/allocs/trash/`); Save on the live panel does the same at any time. Allocations are the files
  `ATLAS_DIR/allocs/user/<name>.json`; Open lists them, with rename and delete.
- **Count (the ranking basis)**: decode, prefill, or Both. Both weighs an expert's share of its layer's decode picks
  0.7 and of its prefill picks 0.3. Prefill and decode pick different experts (their top 64 a layer overlap 42 %, where
  chance gives 25 %; the deeper the layer, the less they overlap).
- **Expert optimisation (re-sort)**: only while the engine runs. It gives the loaded experts' widths again by the
  run's usage record (the record it started with plus this run's picks), each width keeping its number of experts, so
  the memory stays the same. It starts once no request runs or waits, swaps the layers in place on every rank, drops
  the conversations it keeps (their KV was computed with the old experts) and serves again; requests that arrive
  meanwhile wait (a full change of 75 layers took 83 to 90 s on four nodes). Every node needs room for the largest
  layer's new copy on top of what the layers before it grew, plus 2 GiB; the estimate says whether each has it, and
  the page refuses the change, naming the node, when one does not. The result is not saved by itself. A start
  loads the widths it is given and never re-sorts. With `TF_GLM53_MULTI` in `EXTRA_ENV` nothing changes while the
  engine runs: the estimate's widths are saved under a name (`opt-<time>`) for a stop and a start with it.
- **What the re-sort gains** (four nodes, half 3 / half 4 bits, Both): the KL divergence from the all-4-bit model fell
  24 % on the reply's tokens at 8,192 tokens and 32 to 34 % at 128K and 256K tokens; 15 to 22 % on the prompt's tokens
  (`ATLAS-TESTS.md`). On kinds of work not in the counts it ranks by the gain is smaller (about two thirds).
- **Memory, four nodes** (4-bit KV, `STORE` 100,000, MTP): all 3 bits fits 737,280 tokens of context; half 3 / half
  4 bits 737,280; a quarter 3 / three quarters 4 bits 622,592; all 4 bits 327,680; all 5 bits does not fit. With
  `CONTEXT` 1,048,576 and `STORE` 512, 70 % at 3 bits and 30 % at 4 bits fits with about 7.5 GiB a node to spare,
  enough for a live re-sort. DFlash2
  needs about 4 GiB more a node. Mixes with 2-bit experts were 13 to 30 % further from the reference at the same memory.

## 9. What it can do (measured)

- **Speed** (GPU clock capped at 2.1 GHz, one request, a coding prompt not yet cached, reasoning effort low):

| Nodes | Prompt | MTP: time to first token / decode | DFlash2: time to first token / decode |
|---|---|---|---|
| 8 | 128K | 82.8 s / 47.5 tokens/s | 85.2 s / 53.2 |
| 8 | 256K | 172.9 s / 45.7 | 177.6 s / 52.6 |
| 8 | 512K | 376.5 s / 42.8 | 386.0 s / 47.1 |
| 4 | 128K | 120.1 s / 35.3 | 121.9 s / 42.9 |
| 4 | 256K | 253.3 s / 33.2 | 256.9 s / 40.7 |
| 4 | 512K | 565.4 s / 32.2 | 571.5 s / 37.8 |

- **Cached prompts**: a conversation sent again resumes from the pool; a 703,916-token conversation gave its first
  token in under 3 s on four nodes.
- **Accuracy** (KL divergence from the official BF16 model run by the official implementation in FP32): eight nodes
  about 0.013 on 2,048-token texts (top-1 token agrees about 96 %); 0.012 to 0.019 over 8,192-token texts; four nodes
  0.019 to 0.032 over the same texts. Every routed expert at one width, positions 6,144 to 8,190: 2 bits 0.0517,
  3 bits 0.0203, 4 bits 0.0127, 5 bits 0.0114.
- **Tools and streaming**: OpenAI-style function calls and streamed replies with reasoning deltas.

## 10. What it cannot do

- Run on anything but four or eight GB10 systems (no other GPUs, no other rank counts, no single-node mode).
- Serve requests in parallel by default: one runs at a time, up to 32 wait (chat-title requests and
  `"priority": "background"` ones last), more are refused with 503 "the engine is busy; retry shortly". Agents that
  run sub-agents in parallel are better served by sending the sub-agents to another model. The option
  `TF_GLM53_MULTI=4` in `EXTRA_ENV` keeps up to 4 conversations live and decodes them together; it needs
  `CACHE=pool`, `DCP=2` and DFlash2 or no drafter (MTP is not batched), costs node memory, and is off by default.
- Change expert widths while a request runs (a re-sort waits for the request to end), or grow the KV cache after the
  start (it is set aside at the start).
- Accept images, `n` > 1, `logprobs`, JSON mode or forced tool calls (`tool_choice` `required`).
- Promise bit-identical replies at temperature 0 from run to run in the served configuration: the summation order
  of some kernels varies. (The accuracy measurements use a deterministic configuration made for scoring.)
- Include DFlash2: it is downloaded separately under its own licence (CC BY-NC-ND 4.0, non-commercial, research and
  evaluation); whether a use is covered is for its operator to judge.

## 11. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `check` prints `NG: files` / `weights` | the directory or the checkpoint differs on a node | `./tf-glm53 deploy`; `./tf-glm53 verify` |
| `NG: NCCL(...)`, `cuBLAS-13`, `libibverbs` | a library is missing on that node | install it (`RUNTIME.md`, "When something goes wrong") |
| `NG: RoCE-v2-GID` or the RDMA device | the port has no RoCE v2 IPv4 entry, or `IB_HCA` is wrong | give the port its address or set `GID_INDEX`; `rdma link`, `ibdev2netdev` |
| `NG: GPU-busy` | another process holds the GPU | stop it (`nvidia-smi`) |
| `NG: memory(N<M GiB)` | too little free memory | free memory on the node, or lower `STORE` |
| every item `NG` on a node | the first node cannot reach it over ssh | `ssh -o BatchMode=yes ADDRESS true` must work |
| `up`: `a rank stopped` with `NCCL WARN` | network names or addresses wrong, ports blocked | `NODES` on the RoCE network; `SOCKET_IFNAME`, `IB_HCA` right on every node |
| `up`: `cannot listen on` | `PORT` is in use on that node | free it or change `PORT` |
| `up`: `the smoke test failed` | a reply did not answer its question | the failed question shows the reply; rank 0's log shows the request |
| `up`: `not ready after N s` | slow disk or network storage | raise `READY_TIMEOUT`; check `MASTER_PORT` is free |
| `PoolValuesDiffer` | ranks started with different settings | start every rank with `tf-glm53` from one `glm53.conf` |
| the server stops answering, every rank still runs | a hang | `./tf-glm53 down`, `./tf-glm53 up` |
| a node ran out of memory | too large a context or store | each rank runs with `oom_score_adj` 1000, so the kernel stops the rank, the watcher stops the rest and the node stays up; lower `STORE`/`CONTEXT` |
| panel: a width shows "No data" | its bank is missing on some node | download it there; `./tf-glm53 verify-banks N` |
| `slot cache: WARNING: no store` (`CACHE=slots`) | a rank lacked the memory for `STORE` at the start | free memory or lower `STORE`, then `down` and `up` |
| `PoolWriteRefused` in the logs | a rank refused a write to the pool outside the request's blocks | every rank stops at the end of that request; `./tf-glm53 up` |

Logs: `logs/rankR.log` on each node (the start before in `.prev`), `logs/watch.log` (ranks the watcher found gone),
the launcher's log in `logs/`.

## 12. Easy to get wrong

- `max_tokens` counts reasoning tokens too; a reply can end in its reasoning if `max_tokens` is small.
- The engine's start-up line says `drafts: on` whatever the drafter; `tf-glm53 status` names the drafter in use.
- "Four nodes" needs the four-node checkpoint and `KV=fp4` with `./tf-glm53 up`; the control panel can use either
  checkpoint with the banks.
- The re-sort's measurements are against the all-4-bit model, not the BF16 model; `ATLAS-TESTS.md` says which.
- Inside a chat template's user turn the model predicts the user's own text poorly. This is about predicting the
  user's text, as a KL divergence over a prompt does, not about the replies. The measurements in `ATLAS-TESTS.md`
  score a prompt's tokens without the template's opening ("How it was measured").
- The pool cache needs `STORE` of at least 512.
- A container needs the GPU, `/dev/infiniband`, the host network and
  `--ulimit memlock=-1 --ulimit nofile=1048576:1048576`; inside one, `tf-glm53 check` cannot see other containers' GPU
  processes.
- The API listens on every address by default (`HOST=0.0.0.0`) and has no key: anyone who can reach it can use it
  (`HOST=127.0.0.1` keeps it to the first node). The control panel's launcher too (`LAUNCHER_HOST`).
