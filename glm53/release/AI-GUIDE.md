English | [日本語](AI-GUIDE.ja.md) | [简体中文](AI-GUIDE.zh-CN.md)

# A guide for AI assistants: full GLM-5.3 on DGX Spark (release r3.3-102)

This page describes the whole system in one place: what it is, what each part does, how it is used, what it can do
and what it cannot. It is written for AI assistants that help someone understand, install, run or troubleshoot this
release, and reads the same for a person. Its numbers come from the other pages, which hold the full details and say
more where this page is short: `QUICKSTART.md` (the first start), `BEFORE-START.md` (what to decide before it),
`RUNTIME.md` (running it, both node counts), `CONFIG.md` (every setting), `T4.md` (four nodes), `RING.md` (four nodes
as a ring), `PANEL-GUIDE.md` (the control panel's screens), `AGENT-API.md` (the control API with no dashboard) and
`ATLAS-TESTS.md` (the expert re-sort's measurements). In the source repository `RUNTIME.md` is `packaging/RUNTIME.md`
and the others are in `glm53/release/`. Each page is also in Japanese (`.ja.md`) and Simplified Chinese (`.zh-CN.md`).

## 1. What it is

A server for the full GLM-5.3 model (755B parameters, architecture `glm_moe_dsa`) on four or eight NVIDIA DGX Spark
systems (one GB10 each), one rank per GB10, with tensor parallelism 4 or 8. It is Fiore-lab's unofficial derivative of
TensorFold 1.0.2 (an engine written in Zig with plain CUDA kernels), not a release of the TensorFold project. The
server needs no Python at run time.

| Fact | Value |
|---|---|
| Model | GLM-5.3, full size, EXL3-quantized checkpoints published by Fiore-lab on Hugging Face |
| Nodes | 4 or 8 DGX Spark (GB10) on one RoCE v2 network, or 4 cabled as a ring without a switch; 2- and 3-node forms are being tested, and nothing else is supported |
| Context length | up to 1,048,576 tokens on both node counts. Eight nodes: `KV_SIZE` 2,097,152 with it on the 4-bit KV cache (`KV=fp4`, `NEED_GB=100`), 1,638,400 on the 8-bit one. Four nodes: from the control panel with 70 % of the routed experts at 3 bits and 30 % at 4 (4-bit KV, `KV_SIZE` 1,049,088); a default start, every routed expert at 3 bits, 737,280. Eight nodes on the 4-bit KV cache and four nodes from the control panel were each measured with one prompt of about 1,019,000 tokens. The 4-bit KV cache costs some accuracy (KL 0.0190 against 0.0156, `T4.md`). The examples ship 819,200 and 737,280 |
| API | OpenAI-compatible chat API (`/v1/chat/completions`) and Anthropic's Messages API (`/v1/messages`) on every address (`0.0.0.0:8000`) by default |
| Requests | one runs at a time by default, up to 32 more waiting in a queue; `MULTI` decodes up to 4 conversations together |
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
| `QUICKSTART.md`, `BEFORE-START.md`, `RUNTIME.md`, `T4.md`, `RING.md`, `PANEL-GUIDE.md`, `ATLAS-TESTS.md` | the documents, each also in Japanese (`.ja.md`) and Simplified Chinese (`.zh-CN.md`) |
| `AI-GUIDE.md` | this page (also `.ja.md`, `.zh-CN.md`) |
| `docker/Dockerfile`, `docker/entry.sh` | the release as a container image (`up`, `install`, `serve`, `version`) |
| `tools/quickstart.py` | a fresh install to a running engine (`up` and `./tf-glm53 quickstart` run it) |
| `tools/topology.py` | judges how the nodes are cabled (`./tf-glm53 topology`) |
| `tools/logkeep.py` | the log keeper on every node: the logs stay in memory |
| `AGENT-API.md` | the launcher's and the engine's control API for an AI agent, with no dashboard (also `.ja.md`, `.zh-CN.md`) |
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
- **Conversation cache.** The KV cache is one pool of `KV_SIZE` positions (the KV size),
  in blocks of 256, shared by all conversations: a request reaches at most `CONTEXT_LENGTH` positions (the context
  length), and earlier conversations stay in whatever blocks are free. A request resumes from the longest cached
  prefix of any conversation, and conversations that start alike (one system prompt) hold their common blocks once.
  Cached blocks are checked against the request's tokens before reuse; on any doubt every conversation is dropped and
  the request is computed from the start ("cold"), so a block is never reused wrongly.
- **Experts.** GLM-5.3 has 256 routed experts in each of layers 3 to 77 (19,200 in all) and picks 8 per layer for
  every token. The checkpoints store them quantized: the eight-node checkpoint all at 4 bits, the four-node checkpoint
  half at 3 bits (the least used) and half at 4 bits. A start can load each expert at its own width from the expert
  banks, and with no plan loads every expert at 3 bits.
- **Byte Resume.** A model may cut a text differently from the tokenizer, so the next turn's prompt holds the same bytes
  under other token ids. Past the first id that differs, the prompt is compared with the kept conversations in bytes,
  and the cache resumes past the place (`TF_GLM53_BYTE_RESUME`, on by default).
- **The thinking.** `max_tokens` counts it. By default the thinking may keep up to 90 % (`THINKING_SHARE`) of what the
  prompt leaves of the context length, or of `max_tokens` when that is smaller; near that limit the model rewrites its
  thinking and the rewrite replaces it in the context, and the engine closes the thinking when it reaches the limit.
  The limit is on what the context holds, so a reply can write more thinking than that. These settings change while the
  engine runs (`POST /v1/thinking`, the panel's [ REASONING CONTROL ]).
- **Several conversations.** With `MULTI` set, up to that many conversations (at most 4) are decoded together. While they
  run, `POST /v1/schedule` or the panel sets the GPU's split between prefill and decode, and a short prompt can go first.

## 4. Two ways to run it

| | `docker run ... up` or `./tf-glm53 start` | The control panel (`./tf-glm53 up`) |
|---|---|---|
| Nodes | 4 or 8, found by the node check | 4 or 8, registered by the node check (`glm53.nodes.conf`) |
| Expert widths | every expert at 3 bits (`BITS` for another; `start`: the last start's plan) | chosen per expert from the banks (2, 3, 4, 5 bits) |
| Needs | the checkpoint and the 3-bit bank, always | the checkpoint (its non-expert weights), the 3-bit bank, and the banks of the widths used |
| Started from | the command line | a web page (start, stop, plan, live view, expert optimisation); `./tf-glm53 start` repeats the last start's plan |

Both use the same `glm53.conf`, the same server and the same API. `./tf-glm53 up` starts the panel's launcher and
prints the panel's address; the engine is started there. `./tf-glm53 down` stops the engine either way.

An AI assistant that must start the engine with no person at the page uses the line below, or `./tf-glm53 start`: it
starts with the last start's plan (`launch/plan-*.bits`; none yet: every expert at 3 bits), as the panel's Start does,
and waits until the engine runs. `./tf-glm53 engine-up` by hand does the same. `-d KIND` before either picks the
drafter for that start.

### In one line, for an AI assistant (the recommended way)

On the first node, as the user who runs it (every node reaches every node over ssh without a password, their host keys
known (`QUICKSTART.md`, section 1), and the image is on them or can be pulled):

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host IMAGE up
```

It writes the release to `~/expertatlas` and its `glm53.conf`, starts the launcher on the host, finds the nodes (a ring
or a switch), checks them, looks for the weights, starts the ranks in containers with every expert at 3 bits and sends
one request.
The launcher and the engine stay on after this container exits. It prints one numbered line a step and ends in `READY`
with the addresses (the dashboard at `http://NODE:8010/panel`, the API at `http://NODE:8000/v1`, no key, both open to the
LAN), or in `STOP` (what is wrong, which directory and which repository on which nodes), `FIX` (exactly what to do)
and `THEN` (the line to run again). Every launch setting is an option, as `-e NAME=VALUE`: `WEIGHTS`, `BANK_2` to `BANK_5`,
`DRAFTER`, `BITS`, `ALLOC`, `NODES`, `PORT`, `LAUNCHER_PORT`, `EA_DIR`, `IMAGE`, and any key of the engine settings (not `TOPOLOGY`, `SOCKET_IFNAME` or `IB_HCA`, which the node check decides). `QUICKSTART.md` lists them and is the page for a first start.
After that: `AGENT-API.md`.
The release's image holds NCCL with the AArch64 fence, and reads it on a switch and on a ring (`RING.md`).

### In containers

With `RUNTIME=docker` and `IMAGE` in `glm53.conf`, `tf-glm53` starts each rank as a container of the release image on
its node (one container a rank, over ssh from the first node). The nodes need Docker and the NVIDIA container toolkit
(DGX OS has both) instead of cuBLAS, libibverbs and NCCL; steps 1 and 2 below are not needed. `docker pull` the image
on every node (or `./tf-glm53 pull`), write the release's files to a directory on the first node with
`mkdir glm53 && docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/glm53":/out IMAGE install /out` (made first: a
directory Docker makes for the mount is root's, and the install cannot write it), then go on from step 4. The
weights and the banks stay on the hosts and are mounted read-only. `RUNTIME.md`, "Running in containers", has the
details; the image can also be built from the archive with the fenced NCCL (`docker/Dockerfile`, `RING.md`).

### Setting up (both)

1. Every node: DGX OS (Ubuntu 24.04), NVIDIA driver 580 or later, cuBLAS 13 (`libcublas-13-0`), `libibverbs`, the
   nodes on one RoCE v2 network, the first node reaching every node over ssh without a password, their host keys known (`QUICKSTART.md`, section 1).
2. NCCL 2.30.7 on every node: `pip3 install --no-deps --target ~/nccl-2.30.7 nvidia-nccl-cu13==2.30.7` (pip only
   unpacks the library; nothing runs Python). On a switch this runs, and the build with the AArch64 fence
   is recommended; a ring requires that build (`RING.md`).
3. The checkpoint, at the same path on every node: `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks` (390 GB, eight nodes) or
   `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks` (344.6 GB, four nodes). Download once with the Hugging Face CLI, copy with
   `rsync` to the other nodes, check with `./tf-glm53 verify`.
   The checkpoint can also be made from the official BF16 checkpoint (section 2, `glm53/weights/`).
4. Unpack the archive on the first node, `sha256sum -c SHA256SUMS`, copy the right example to `glm53.conf`, set
   `NODES`, `SOCKET_IFNAME`, `IB_HCA`, `NCCL_LIB`, `WEIGHTS`.
5. `./tf-glm53 deploy` (copies the directory to the other nodes), then, without the page, `./tf-glm53 start`
   (every expert at 3 bits: checks every node, starts the ranks, waits for the server, runs a smoke test); with it, `./tf-glm53 up` (below).
   A start takes about two and a half minutes from a local NVMe drive.
6. `./tf-glm53 down` stops the engine (through the launcher when it runs) and every rank.

### Setting up the control panel

1. The expert banks the plans will use, on every node in the directories `expert-plan/banks.json` names (by default
   `~/models/GLM-5.3-EXL3-Experts-<N>bit`): on four nodes `fiore-lab/GLM-5.3-EXL3-Experts-3bit` and `-4bit` are the
   usual pair (277.9 and 368.5 GB); `-2bit` (187.3 GB) and `-5bit` (459.1 GB) are optional. The eight-node checkpoint
   can serve as the 4-bit bank (`RUNTIME.md`, "The control panel"). Check them with `./tf-glm53 verify-banks 3 4`.
2. `./tf-glm53 up` on the first node starts the launcher and prints the panel's address; open it,
   `http://FIRST-NODE:8010/panel` (without `LAUNCHER_PORT`, the port is `PORT` + 10). The launcher listens on every
   address by default and has no key: anyone who reaches it can start, stop and swap the engine
   (`LAUNCHER_HOST=127.0.0.1` keeps it to the node; then an ssh tunnel, `ssh -L 8010:127.0.0.1:8010 FIRST-NODE`). It
   stays on whether the engine runs or not; the page reaches the engine through it, and an engine started while it
   runs sends a browser that opens its own `/panel` to it. While the engine is stopped it reads `glm53.conf` again by
   itself after an edit; only `LAUNCHER_HOST` and `LAUNCHER_PORT` need `./tf-glm53 launcher stop` and `./tf-glm53 up`.
   The first time, the page opens on the node check, which finds a ring or a switch's nodes, checks them and writes
   `glm53.nodes.conf`. That file is read after `glm53.conf` and wins for `TOPOLOGY`, `NODES`, `SOCKET_IFNAME` and `IB_HCA`.
   The launch screen's [ WEIGHTS ] sets where the weights are, and the engine settings screen the settings that need a restart.
3. On the page: choose a plan, the KV format, the context length, what each node keeps for the OS, the drafter and the
   concurrent inferences (the KV size is all the room the plan leaves, shown on the KV rows); check the memory verdict;
   press Start (then Confirm). Stop from the page too (or `./tf-glm53 down`). Later starts with the same plan:
   `./tf-glm53 start`.

## 5. Settings (`glm53.conf`)

| Setting | Meaning | Eight-node example / four-node example |
|---|---|---|
| `NODES` | the nodes' addresses on the RoCE network, rank 0 first (a ring: on the management network) | 8 / 4 addresses |
| `TOPOLOGY` | `full` (a switch), `ring4` or `auto` (judged from the cabling at each check and `./tf-glm53 up`; a known bug in `CONFIG.md`) | `full` |
| `SOCKET_IFNAME`, `IB_HCA`, `GID_INDEX` | that network's interface and RDMA device on every node; the GID index (empty: chosen) | the same |
| `NCCL_LIB` | NCCL 2.30.7's `libnccl.so.2`, the same path on every node | the same |
| `WEIGHTS` | the checkpoint's directory, the same path on every node | the 8x / 4x checkpoint |
| `DRAFTER_KIND`, `MTP_K`, `DRAFTER` | the drafter (`mtp`, `dflash2`, `none`), MTP's drafts a round (3), DFlash2's directory | `mtp`, 3, empty |
| `NAME`, `HOST`, `PORT` | the model name in the API, the listen address and port | `glm-5.3`, `0.0.0.0`, 8000 |
| `METRICS_LISTEN` | an extra read-only address answering `/metrics`, `/health` and `/v1/experts/usage` only | off |
| `LOG_LISTEN` | every node's log keeper answers its logs on this address, read-only | off |
| `CONTEXT_LENGTH` | the context length: positions one request may reach, prompt and reply together (at most 1,048,576) | 819,200 / 737,280 |
| `KV_SIZE` | the KV size: the whole KV cache in positions, one pool every conversation shares; what a request does not use keeps earlier conversations (`CONTEXT_LENGTH` + 512 at least) | 1,638,400 / 837,280 |
| `KV` | the KV cache format, `fp8` or `fp4` | `fp8` / `fp4` |
| `DCP`, `ROWS`, `SP_PARTS` | decode context parallelism, prompt chunk rows, sequence-parallel parts a chunk | 2, 8,192, 4 |
| `NEED_GB` | the memory a node must have available before a start (it follows the KV size, plus 3 with DFlash2; `CONFIG.md` gives how) | 112 (default) / 133 (asks 112 GiB) |
| `MASTER_PORT`, `READY_TIMEOUT` | the ranks' rendezvous port; seconds to wait for the server | 29551, 900 |
| `RUNTIME`, `IMAGE` | how the ranks run: `host` (default, from this directory) or `docker` (each in a container of `IMAGE`) | `host`, empty |
| `MULTI`, `MULTI_WAIT` | conversations decoded together (1 .. 4); seconds a request may wait for room beside them | `0` (one at a time), 600 |
| `MULTI_DECODE_SHARE`, `MULTI_SHORT_PROMPT`, `MULTI_SHORT_PREEMPT` | with `MULTI`: the decode's percent of the GPU's time beside a prefill; a prompt this short goes first; whether it overtakes a running prefill | 30, 8192, 0 |
| `MAX_TOKENS_SHARE`, `MAX_TOKENS` | a reply's `max_tokens` when the request names none: a percent of `CONTEXT_LENGTH`, or a count that comes before it | 70, empty |
| `THINKING_SHARE`, `THINKING_BUDGET` | the thinking's limit: a percent, or a count that comes before it | 90, 0 |
| `THINKING_FOLD`, `THINKING_FOLD_SHARE`, `THINKING_FOLD_MAX` | organizing midway: how (`off`, `recent`, `checkpoint`, `branch`), how often, how many times | `off`, 100, 0 |
| `THINKING_FOLD_AT_BUDGET`, `THINKING_FOLD_MARGIN` | organizing near the limit, and how long before it | 1, 10 |
| `THINKING_FOLD_AT_CLOSE_SHARE` | organizing when the model closes its thinking: at least this percent of the limit; 0 is off | 0 |
| `THINKING_NOTICE` | every this many thinking tokens, a line tells the model what it used; 0 is off | 0 |
| `TF_GLM53_BYTE_RESUME` | Byte Resume: `on`, `log`, `off` | `on` |
| `ROCE_PROXY_CORES` | the cores the RoCE proxy threads poll on: `a725`, `os` | `a725` |
| `EXTRA_ENV` | engine variables (not `TF_GLM53_MULTI`: that is `MULTI`) | empty |
| `LAUNCHER_HOST`, `LAUNCHER_PORT`, `ATLAS_DIR`, `RESERVE_GB` | the control panel: its address, port, saved allocations, memory kept for the OS | four-node example: `0.0.0.0`, 8010, `atlas/`, 4 |

Older `glm53.conf` files are still read: `CONTEXT` as `CONTEXT_LENGTH`, `CONTEXT` + `STORE` as `KV_SIZE`. `CONFIG.md`
has every setting.

## 6. Commands (`./tf-glm53`, run on the first node)

| Command | What it does |
|---|---|
| `deploy` | copies this directory to every node (same path) and checks `SHA256SUMS` there |
| `check` | checks nodes, files, weights, libraries, RDMA devices, GPUs and memory; lists problems after `NG:` |
| `verify` | checks the weights against their `SHA256SUMS` on every node (a few minutes) |
| `verify-banks [BITS...]` | checks the expert banks' files against their manifests |
| `quickstart` | a fresh install to a running engine, as `docker run ... IMAGE up` runs it |
| `topology [--json]` | judges how the nodes are cabled |
| `up` | starts the control panel's launcher and prints the panel's address; the engine is started from the panel |
| `start` | starts the engine through the launcher, as the panel's Start does, with the last start's plan; waits until it runs |
| `down` | stops the engine: through the launcher when it runs, then every rank directly |
| `engine-up` | by hand, the same as `start`; with the launcher's plan: `check`, starts the ranks (last to first), waits for the server, runs the smoke test |
| `engine-down` | stops every rank directly (TERM, then KILL after 30 s) |
| `status` | the ranks, free memory, the drafter, the conversation cache, the server's `/health` |
| `smoke` | three requests whose answers are checked |
| `logs [R]` | follows rank R's log (default 0; `R.prev`, `watch`, `launcher`); the logs are in memory on each node |
| `launcher [stop\|status]` | starts (as `up` does), stops or reports the control panel's launcher |
| `-d KIND` (before the command) | the drafter for this start (`start` or `engine-up`: `mtp`, `dflash2`, `none`) |
| `pull` | `RUNTIME=docker`: `docker pull IMAGE` on every node |
| `-c FILE` (before the command) | another settings file |

`tf-glm53` uses no `sudo` and changes no system settings. The ranks are meant to be started and stopped only with it:
it leaves a watcher that stops every rank when some are gone (ranks waiting on a lost peer would otherwise wait
forever).

## 7. The API

| Path | Method | Notes |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI chat: streaming (`stream`, `stream_options.include_usage`), tools (`tools`, `tool_choice` `auto` or `none`) |
| `/v1/messages`, `/v1/messages/count_tokens` | POST | Anthropic's Messages API, for Claude Code (`RUNTIME.md`, "Claude Code") |
| `/v1/responses` | POST | OpenAI's Responses API |
| `/v1/completions` | POST | plain text completion of a prompt (no chat template is added) |
| `/v1/models` | GET | the served model name |
| `/v1/tokenize`, `/v1/detokenize` | POST | token counts and ids, for a prompt or for messages rendered by the chat template |
| `/health` | GET | status, memory, live speeds, the conversation pool, token and draft counters |
| `/metrics` | GET | the same counters in Prometheus text |
| `/v1/experts/...`, `/v1/memory` | GET / POST | expert usage, bits and swaps, used by the control panel |
| `/v1/thinking` | GET / POST | the thinking's settings while the engine runs |
| `/v1/schedule` | GET / POST | the GPU's split and short prompts, with `MULTI` (404 without it) |
| `/v1/pool` | GET | the conversations kept |

`/v1/experts/...`, `/v1/memory`, `/v1/thinking`, `/v1/schedule` and `/v1/pool` answer on the first node only (403 from elsewhere). The launcher's port serves them to the network.

- The reply's `message.reasoning_content` holds the model's reasoning and `message.content` the answer; `max_tokens`
  counts both. `chat_template_kwargs` takes `{"enable_thinking": false}` (no reasoning) and `reasoning_effort`. A request's
  own `thinking_budget`, `thinking_fold`, `thinking_fold_at_budget`, `thinking_fold_margin` and `thinking_fold_at_close`
  (in tokens) come before the settings. `reasoning_details` carries the thinking the context holds, for the next turn.
- Refused with 400: `n` other than 1, `logprobs`, `response_format` `json_object` and `json_schema`, `guided_*`,
  `tool_choice` `required` or a named tool, images on the chat API (on `/v1/messages` an image becomes a note; text only),
  and, with no summaries midway and no organising at the budget's end, a `max_tokens` beyond what the
  context has left (`CONFIG.md`, `MAX_TOKENS`, gives the rule with summaries midway). A streamed request that cannot fit ends with an error event.
- A client that disconnects cancels its request.
- No key: the API has none. By default it listens on every address (`HOST=0.0.0.0`), so anyone who can reach it can
  use it, and the server speaks plain HTTP, so text crosses the network in the clear unless the server stays on a
  closed network or TLS and access control are put in front (`HOST=127.0.0.1` keeps the API to the first node).
- Speeds from the counters: decode tokens/s = Δ`completion_tokens_total` / Δ`decode_seconds_total`; prefill tokens/s
  = Δ`prefill_tokens_total` / Δ`prefill_seconds_total`; tokens kept per draft round = Δ`completion_tokens_total` /
  Δ`rounds_total` (1 means no draft was kept); cache hits = Δ`cached_tokens_total` / Δ`prompt_tokens_total`. The
  counters include the running request and never fall back.

The control panel's launcher has its own API under `/v1/launch/` (status, plan, start, stop, restart, settings, weights,
discover, ring, check, config, usage, resort, and the saved allocations: allocs, alloc, allocs/save, allocs/rename,
allocs/delete). `AGENT-API.md` gives what an agent needs of it; its fields may change between releases.

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
  loads the widths it is given and never re-sorts. With `MULTI` set nothing changes while the
  engine runs: the estimate's widths are saved under a name (`opt-<time>`) for a stop and a start with it.
- **What the re-sort gains** (four nodes, half 3 / half 4 bits, Both): the KL divergence from the all-4-bit model fell
  24 % on the reply's tokens at 8,192 tokens and 32 to 34 % at 128K and 256K tokens; 15 to 22 % on the prompt's tokens
  (`ATLAS-TESTS.md`). On kinds of work not in the counts it ranks by the gain is smaller (about two thirds).
- **Memory, four nodes** (4-bit KV, a KV size of the context length + 100,000, MTP, `ROWS` 16,384; the default of 8,192 leaves more): all 3 bits fits a context length
  of 737,280 tokens; half 3 / half 4 bits 737,280; a quarter 3 / three quarters 4 bits 622,592; all 4 bits 327,680; all
  5 bits does not fit. With `CONTEXT_LENGTH` 1,048,576 and `KV_SIZE` 1,049,088, 70 % at 3 bits and 30 % at 4 bits
  fits with about 7.5 GiB a node to spare, enough for a live re-sort. DFlash2 needs about 4 GiB more a node. Mixes with 2-bit experts were 13 to 30 % further from the reference at the same memory.

## 9. What it can do (measured)

- **Speed** (measured with release r2-102 and each checkpoint's own widths, not measured again for this release; on four nodes `ROWS` was 16,384 then and is 8,192 now; GPU clock
  capped at 2.1 GHz, one request, a coding prompt not yet cached, reasoning effort low; speeds differ with the environment, so these are a guide):

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

- Run on anything but four or eight GB10 systems (no other GPUs, no other rank counts, no single-node mode). Two- and
  three-node forms are being tested.
- Serve requests in parallel by default: one runs at a time, up to 32 wait (chat-title requests and
  `"priority": "background"` ones last), more are refused with 503 "the engine is busy; retry shortly". Agents that
  run sub-agents in parallel are better served by sending the sub-agents to another model. The setting
  `MULTI=4` keeps up to 4 conversations live and decodes them together (the MTP layer, DFlash2 or no drafter; not on
  3 nodes yet); it costs node memory and is off by default.
- Change expert widths while a request runs (a re-sort waits for the request to end), or grow the KV cache after the
  start (it is set aside at the start).
- Read images (the chat API refuses them, `/v1/messages` turns them into a note), or accept `n` > 1, `logprobs`, JSON mode
  or forced tool calls (`tool_choice` `required` or a named tool).
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
| `NG: memory(N<M GiB)` | too little free memory | free memory on the node, or lower `KV_SIZE` |
| every item `NG` on a node | the first node cannot reach it over ssh | `ssh -o BatchMode=yes ADDRESS true` must work |
| a start: `a rank stopped` with `NCCL WARN` | network names or addresses wrong, ports blocked | `NODES` on the RoCE network; `SOCKET_IFNAME`, `IB_HCA` right on every node |
| a start: `cannot listen on` | `PORT` is in use on that node | free it or change `PORT` |
| a start: `the smoke test failed` | a reply did not answer its question | the failed question shows the reply; rank 0's log shows the request |
| a start: `not ready after N s` | slow disk or network storage | raise `READY_TIMEOUT`; check `MASTER_PORT` is free |
| `PoolValuesDiffer` | ranks started with different settings | start every rank with `tf-glm53` from one `glm53.conf` |
| the server stops answering, every rank still runs | a hang | `./tf-glm53 down`, `./tf-glm53 start` |
| a node ran out of memory | too large a KV size | each rank runs with `oom_score_adj` 1000, so the kernel stops the rank, the watcher stops the rest and the node stays up; lower `KV_SIZE` |
| panel: a width shows "No data" | its bank is missing on some node | download it there; `./tf-glm53 verify-banks N` |
| `PoolWriteRefused` in the logs | a rank refused a write to the pool outside the request's blocks | every rank stops at the end of that request; `./tf-glm53 start` |

Logs: in memory, in a log keeper on each node (`tools/logkeep.py`), nothing on disk: each rank's (and the start before's,
`R.prev`), the watcher's (ranks it found gone) and the launcher's, `tf-glm53 logs R | R.prev | watch | launcher`. With
`LOG_LISTEN` a monitor reads them over HTTP and keeps them.

## 12. Easy to get wrong

- `max_tokens` counts reasoning tokens too. A request that names none gets 70 % of `CONTEXT_LENGTH`. The thinking's limit
  keeps the rest for the answer, so a small `max_tokens` gives a short answer; with `THINKING_SHARE=0` a reply can end in
  its reasoning.
- The engine's start-up line says `drafts: on` whatever the drafter; `tf-glm53 status` names the drafter in use.
- Every start needs a checkpoint and the 3-bit bank; the control panel can use either checkpoint with the banks.
- The re-sort's measurements are against the all-4-bit model, not the BF16 model; `ATLAS-TESTS.md` says which.
- Inside a chat template's user turn the model predicts the user's own text poorly. This is about predicting the
  user's text, as a KL divergence over a prompt does, not about the replies. The measurements in `ATLAS-TESTS.md`
  score a prompt's tokens without the template's opening ("How it was measured").
- The pool cache needs a `KV_SIZE` of at least `CONTEXT_LENGTH` + 512.
- A container needs the GPU, `/dev/infiniband`, the host network and
  `--ulimit memlock=-1 --ulimit nofile=1048576:1048576`.
- `/v1/thinking` and `/v1/schedule` answer on the first node only; from another machine they go through the launcher's port.
- `glm53.nodes.conf` wins over `glm53.conf` for `TOPOLOGY`, `NODES`, `SOCKET_IFNAME` and `IB_HCA`.
- The API listens on every address by default (`HOST=0.0.0.0`) and has no key: anyone who can reach it can use it
  (`HOST=127.0.0.1` keeps it to the first node). The control panel's launcher too (`LAUNCHER_HOST`).
