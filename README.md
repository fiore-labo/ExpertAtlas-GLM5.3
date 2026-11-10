English | [日本語](README.ja.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md)

# Full GLM-5.3 on four or eight DGX Sparks

Full GLM-5.3 (755B, `glm_moe_dsa`) served by four or eight NVIDIA DGX Spark systems, one GB10 each, with tensor
parallelism 4 or 8, on one RoCE switch or, for four nodes, a ring of direct cables: fast, with a long context for
agents and coding. One request runs at a time by default; `MULTI` decodes up to four conversations together. The
engine is written in Zig with plain CUDA kernels and needs no Python at run time.

This is Fiore-lab's unofficial derivative of [TensorFold](https://github.com/ashhart/TensorFold) 1.0.2, not a release
of the TensorFold project. It carries TensorFold's code, adds the engine for full GLM-5.3
(`zig/src/families/glm_moe_dsa/`, `zig/kernels/cuda/glm53/`) and leaves out TensorFold's documents, which describe
other models, weights and machines. For TensorFold itself, see its repository.

## Documents

Each page is in English, Japanese (`.ja.md`) and Simplified Chinese (`.zh-CN.md`). In the release archive and the image they
sit together at the top level.

| Page | What it gives |
|---|---|
| [QUICKSTART](glm53/release/QUICKSTART.md) | The first start: the weights, one `docker run` line, every option, the common `STOP` messages |
| [BEFORE-START](glm53/release/BEFORE-START.md) | What to decide before the first start, and what can wait |
| [PANEL-GUIDE](glm53/release/PANEL-GUIDE.md) | The control panel's screens, item by item |
| [SCREENS](glm53/release/SCREENS.md) | Every screen of the control panel, with pictures |
| [FEATURES](glm53/release/FEATURES.md) | Each feature: what it does, why it is there, how to use it |
| [CONFIG](glm53/release/CONFIG.md) | Every setting of `glm53.conf`: its default, who reads it, when a change takes effect |
| [RUNTIME](packaging/RUNTIME.md) | Running it from the archive or by hand: every command, the cache, Claude Code, failures |
| [T4](glm53/release/T4.md) | Four nodes: the weights, the settings, the measurements |
| [RING](glm53/release/RING.md) | Four nodes as a ring without a switch: the cabling, the fenced NCCL, the measurements |
| [AGENT-API](glm53/release/AGENT-API.md) | The control API for an AI agent, with no dashboard |
| [AI-GUIDE](glm53/release/AI-GUIDE.md) | The whole system on one page, for an AI assistant that helps you understand or run it |
| [ATLAS-TESTS](glm53/release/ATLAS-TESTS.md) | The expert re-sort's measurements |
| [weights](glm53/weights/README.md) | Making every weight file yourself from Z.AI's official BF16 checkpoint |

## What it does

- Tensor parallelism 8 or 4 with the KV cache split across rank pairs (decode context parallelism 2). The engine and
  the model take a context length of up to 1M tokens (1,048,576), and **both node counts reach 1M**. Eight nodes:
  `CONTEXT_LENGTH=1048576` with `KV_SIZE=2097152` on the 4-bit KV cache (`KV=fp4`, `NEED_GB=100`; about 23 GiB a node
  to spare), or `KV_SIZE=1638400` on the 8-bit one. Four nodes: `CONTEXT_LENGTH=1048576` from the control panel with
  3 bits for 70 % of the routed experts and 4 bits for the rest (`KV=fp4`, `KV_SIZE=1049088`: the context length and
  512 more, the least the pool takes; about 7.5 GiB a node to spare at `ROWS` 16,384, enough for a live re-sort). A default
  start asks for `CONTEXT_LENGTH=1048576` on either node count, and one with less room cuts it and says so. Eight nodes on the 4-bit KV cache and four nodes from
  the control panel were each measured with one prompt of about 1,019,000 tokens. The 4-bit KV cache costs some accuracy (KL 0.0190 against 0.0156 with the 8-bit cache, same
  weights: [glm53/release/T4.md](glm53/release/T4.md)); the speeds below are with the 8-bit cache on eight nodes. The
  conf examples ship with context lengths of 819,200 and 1,048,576 and KV sizes of 1,638,400 and 1,049,088 (the KV size is
  the whole KV cache, one pool every conversation shares: a request takes up to the context length, and the rest keeps
  earlier conversations' KV in whatever blocks are free).
- Long prompts read in sequence-parallel parts; decode-step collectives over RoCE.
- One conversation cache for all conversations, in blocks of 256 positions: a request resumes from the longest cached
  prefix of any conversation, and conversations that begin alike (an agent's system prompt, its sub-agents) hold the
  blocks they share once.
- Drafts checked by the model in each round: by GLM-5.3's own MTP layer by default, or by the optional DFlash2 drafter.
  The reply is the model's own choice in every case.
- The thinking kept in bounds: a limit as a share of `max_tokens`, and organizing a long thinking midway or at its close, set
  per request or while the engine runs.
- Structured output: a reply's answer held to JSON or to a JSON Schema (`response_format`), with the thinking on or
  off.
- Same answers on request (`DETERMINISTIC=1`): the same requests in the same order give the same tokens after every
  start.
- `/metrics` (Prometheus text) and `/health` count the running request too, so a monitor reads prefill and decode
  speed, cache hits and draft acceptance while a request runs. `METRICS_LISTEN` serves the two read-only on an address
  of their own, for a monitor on another host.

## New in r3.3.7-102

- **Structured output with `MULTI`.** A structured request (`response_format`, `text.format`, `output_config.format`)
  no longer gets a 400 on an engine started with `MULTI`: it is one of the conversations decoded together, one row of
  the window a round under its grammar, beside the others' drafted rows. Plain requests decode as on r3.3.6-102.
  Checked on the ring's four nodes with DFlash2 at `MULTI=2` and `MULTI=4`: structured and plain requests mixed (greedy
  and sampled), four structured at once, with the thinking, beside a long prompt's prefill, a stream the client dropped,
  cut by `max_tokens`.

## New in r3.3.6-102

- **Built by Expert Atlas's own build.** The engine and the server come from Expert Atlas's own CUDA build
  (`zig/build/expertatlas.zig`): the seven GLM-5.3 kernels, the GLM-5.3 engine, and the server over Expert Atlas's own
  engine contract (`zig/src/expertatlas/`), with r3.3.5-102's sampler (the same draws). TensorFold 1.0.2's lane host and
  its parts for other models are no longer built. In the same answers mode the tokens and the KV cache are the same as
  r3.3.5-102's.
- **A smaller source tree:** the repository leaves out the parts of TensorFold 1.0.2 that the build does not use (the
  other model families, the Metal backend and the macOS build, the CUDA kernels other than GLM-5.3's, Nemotron's and the
  older CLI's parts, the Python server tests, and the code only those parts used), 699 files. `NOTICE` and `THIRD_PARTY_NOTICES.md` say what is left out.
- The archive and the image hold the same files as r3.3.5-102's but `LICENSES/MiaAI-Lab-MIT.txt`: the Flash Next code
  it covers is not in them.

## New in r3.3.5-102

- **Faster decode, the same tokens.** The Hadamard rotation before each rotated linear is done by the kernel that makes
  its input (the rmsnorm, the SwiGLU), and post-attention's residual add and rmsnorm run as one kernel; at a
  temperature above 0 each rank picks a row's candidates on its GPU and only those are gathered, instead of the whole
  vocabulary's logits. On the ring a decode round is 0.3 to 0.4 % shorter from the first and another 0.4 to 0.8 % from
  the second, with the same tokens and KV cache as before (compared in the same answers mode).
- **The sampler in Expert Atlas's own code:** the seed from a prompt, the random numbers, `top_k` / `top_p` / `min_p`
  and the draw are written in C, with TensorFold 1.0.2's algorithm and the same draws bit for bit.
- **The launcher's ssh:** the launcher's look round and `tf-glm53`'s watch reuse one ssh connection a node, so a node's
  `auth.log` and journal get one login a start instead of one every 5 s.
- **Fixed:** the launch screen no longer pulls a page scrolled down back to the top every 5 s (seen in Safari).

## New in r3.3.4-102

- **Same answers (`DETERMINISTIC=1`, off by default).** On the same release, nodes and settings, the same requests sent
  in the same order from a start give the same tokens after every start, at any temperature (a request without a
  `seed` takes one from its prompt). The EXL3 linears take fixed tiles instead of timed ones, the prompt experts add
  their slots in a fixed order, and the ranks compare their settings at the start. It runs one conversation at a time
  (`MULTI=0`) and takes the context length and the KV size as set; prefill is about 20 % slower, decode about the same
  ([FEATURES](glm53/release/FEATURES.md#14-same-answers-deterministic-mode)).
- **Fixed:** cuBLAS uses the work area set aside for it, which it did not before. Twelve short prompts gave the same
  tokens as on r3.3.3-102.

## New in r3.3.3-102

- **Structured output, on by default.** A reply's answer can be held to a JSON object or to a JSON Schema:
  `response_format` on `/v1/chat/completions` (vLLM's `guided_json` and `structured_outputs` too), `text.format` on
  `/v1/responses` and `output_config.format` on `/v1/messages`. At each token the model chooses only among the tokens
  the grammar allows, so the answer parses and follows the schema unless `max_tokens` cuts it. With the thinking on,
  the model thinks freely first. A schema the grammar would keep less strictly than it says is refused (400) rather
  than half kept, and so is a structured request beside tools. The grammar is llguidance 1.9.1 (MIT),
  in the archive and the image; `STRUCTURED=0` turns it off ([FEATURES](glm53/release/FEATURES.md#13-structured-output)).
- A structured request decodes without the drafter, so more slowly. Plain requests are as on r3.3.2-102: the same
  tokens and the same speed.
- A build from source also needs Rust 1.95.0 ("Build from source").

## New in r3.3.2-102

- **Fixed:** GLM tool calls with `</arg_value>` or `</tool_call>` in a value (code, or a text that shows a tool call)
  are read whole: one call runs and nothing leaks into the text, where before the value was cut and an example inside
  it could run. A tool-call block that opens with DSML's tag but is too short no longer makes the server panic and
  take the engine down. Rendering the chat template for a long conversation of tool results no longer takes memory
  growing with its square: 3,000 turns took 738 MB and now take 85 MB.
- **A faster start:** every rank starts at once and the engine's ready is seen within a second; on the switch's eight
  nodes the whole start went from 115 s to 86 s.
- **DFlash2's defaults:** with `MULTI`, a window of 4 rows for a conversation beside others, from two conversations
  (+12 % for four conversations writing prose on eight nodes, 2 % slower for code on four), and a confidence of 0.6.
  MTP, the default drafter, ran faster than either in the same measurement (73.1 tok/s).
  `TF_GLM53_MULTI_DRAFT_ROWS=8 TF_GLM53_DFLASH_CONF=0.4` in `EXTRA_ENV` brings the earlier behaviour back.

## New in r3.3.1-102

- **The engine rests while it has nothing to do.** After a second with nothing to do (`TF_GLM53_IDLE_PARK_MS` in
  `EXTRA_ENV`), every rank parks: the other ranks sleep on their link to rank 0 and every RoCE proxy thread stops. A
  request, a swap of the experts' bits or the stop wakes them in 0.6 to 2.0 ms, and a short request's first token comes
  as soon as before. On the ring, parked, each node's engine used under 1 % of a core and the GPUs 0 %, where the other
  ranks' GPUs read 96 % without parking; their SoC read 50 to 54 °C against 64 to 85 °C with `ENGINE_CORES=os`.
  `/health` `idle` shows the state, and the panel's RDMA cells read idle ([CONFIG](glm53/release/CONFIG.md)).
- **Fixed: a stop with requests queued or running** took 36 to 37 s in r3.3-102 and left the other ranks' GPUs
  busy (P0, about 11 W) with no process. Now it ends the requests and tells every rank: 6 s under load as when idle,
  and every GPU goes back to idle (P8).

## New in r3.3-102

**Changes that need something from you:**

- **The weights to download changed:** `fiore-lab/GLM-5.3-EXL3-Base` (21.2 GB) and the 3-bit bank (277.9 GB), on four nodes and on eight. A start reads every
  routed expert from the banks, so the full checkpoints are no longer needed; if you have one, name it with `WEIGHTS`. A start with no plan loads every expert at 3 bits.
  `BITS` or a saved allocation chooses other widths, whose banks are then needed too.
- **Settings renamed:** `CONTEXT_LENGTH` (one request's longest) and `KV_SIZE` (the whole KV cache) replace `CONTEXT` and
  `STORE`. An older `glm53.conf` is still read.
- **A reply's default limit** was 4,096 tokens and is now 70 % of the context length (`MAX_TOKENS_SHARE`). A client's
  `max_tokens` comes first.
- **Fewer choices in the settings:** `CACHE` is `pool`, `DCP` is 2 and `ROCE_PROXY_CORES` is `a725` or `os`.
- **`./tf-glm53 up` starts the control panel only;** the engine is started from the panel, or with `./tf-glm53 start`.
  `./tf-glm53 engine-up` starts it by hand.

**New:**

- **One line to a running engine:** `docker run ... IMAGE up` writes the release to the first node, finds the nodes,
  looks for the weights, starts the engine and sends one request. It ends in `READY` with the addresses, or in `STOP`,
  `FIX` and `THEN`, which say what to do next. Every setting of the engine settings screen is a `-e NAME=VALUE` option
  ([QUICKSTART](glm53/release/QUICKSTART.md)).
- **Four nodes as a ring**, with no switch ([RING](glm53/release/RING.md)).
- **NCCL with the AArch64 fence** in the image, on a switch and on a ring: NVIDIA/nccl#2393, which no NCCL release has
  yet. Without it a proxy thread on Arm can hang with no error ([RING](glm53/release/RING.md), "The fenced NCCL").
- **The node check** finds a ring or a switch's nodes from the cabling, checks them and registers them
  (`glm53.nodes.conf`), so `TOPOLOGY`, `SOCKET_IFNAME` and `IB_HCA` need no writing.
- **Reasoning Control:** the thinking's limit as a share of `max_tokens`, organizing midway and at the close, changed while
  the engine runs from the panel or `POST /v1/thinking` ([CONFIG](glm53/release/CONFIG.md)).
- **Several conversations at once:** `MULTI` (1 to 4) now works with the MTP layer too. While they run, the panel or
  `POST /v1/schedule` sets the GPU's split between prefill and decode, and a short prompt can go first.
- **Byte Resume** (on by default): when a model cuts a text differently from the tokenizer, the next turn's prompt holds
  the same bytes under other token ids; the cache now resumes past that place instead of computing the rest again.
- **The panel:** an engine settings screen, where the weights are and a check of them ([ WEIGHTS ]), the startup flow,
  the KV pool on the circle, each node's temperatures.
- **The control API for an AI agent:** start, stop and set the engine with no dashboard
  ([AGENT-API](glm53/release/AGENT-API.md)).
- **Faster and cooler:** the weights load in parallel (ready in 127 s instead of 366 s, measured on the ring). MTP stops a
  round's drafts when their confidence falls (`TF_GLM53_MTP_CONF`, 0.7 by default). The RoCE proxy threads stay on the
  Cortex-A725 cores (`ROCE_PROXY_CORES`), which ran the SoC 5 to 16 °C cooler at the same speed.

## New in r3.2-102

- Nothing is written to disk while the engine serves. Each rank's log, the watcher's and the launcher's are kept in
  memory, in a log keeper on every node (`tools/logkeep.py`; `python3`, which DGX OS has, on every node, with the
  archive and with the image alike), the last `LOG_KEEP_MB` (8) of each; `./tf-glm53 logs R` reads them. With
  `RUNTIME=docker` the containers keep no log either. r3.1-102 kept them in memory only when the user lingered; r3.2-102
  needs no linger.
- A monitor on another host keeps the records: with `LOG_LISTEN=HOST:PORT` every node's keeper answers its logs,
  read-only and with no key (`GET /streams`, `GET /logs`), and the metrics listener (`METRICS_LISTEN`) also answers the
  experts' use counts (`GET /v1/experts/usage`) ([packaging/RUNTIME.md](packaging/RUNTIME.md), "Serving other machines").

## New in r3.1-102

- The API and the control panel listen on every address by default (`HOST=0.0.0.0`, `LAUNCHER_HOST=0.0.0.0`), with
  no key: other machines on the network reach them directly. Keep the nodes on a closed network
  (`HOST=127.0.0.1` / `LAUNCHER_HOST=127.0.0.1` keep them to the first node).
- Claude Code on the Messages API: `claude-glm53` starts it against a server, and the server keeps its conversations
  in the cache, keeps long prompts' streams alive and turns images into a note
  ([packaging/RUNTIME.md](packaging/RUNTIME.md), "Claude Code").
- The control panel works with an engine started by `./tf-glm53 up` too (the engine counts the experts' use by
  default). A re-sort while the engine runs swaps the layers that do not grow first and checks every node's memory
  against the swap's peak before it starts; the panel names a node that is short.
- Every setting of `glm53.conf`: [glm53/release/CONFIG.md](glm53/release/CONFIG.md).

## Expert Atlas (new in r3-102)

GLM-5.3 picks 8 of 256 routed experts a layer for each token, and some experts are picked far more often than others.
Expert Atlas loads each of the 19,200 routed experts (layers 3 to 77) at its own width, 2, 3, 4 or 5 bits, so the
experts in most use get the most bits for the memory there is:

- **Banks**: the routed experts at each width, on Hugging Face: [fiore-lab/GLM-5.3-EXL3-Experts-2bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-2bit), [-3bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-3bit),
  [-4bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-4bit) and [-5bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-5bit). The panel offers the widths every node holds.
- **The control panel**: a page served by the launcher (`./tf-glm53 up`). Before a start it plans the widths
  and shows each node's memory, whether the plan fits and the KV size it leaves; it starts and stops the engine. While
  the engine runs it shows the experts each token picks, the mean width of the picked experts and the speeds.
- **Usage records**: the engine counts the experts every prompt row and decoded token picks. A saved allocation keeps
  the usage record of the runs it was saved from, so its data travels with it: opened and run again, its record grows
  by that run. Plans rank the experts by the record (a plan without one, by counts Fiore-lab measured and ships with
  the page). Nothing is saved by itself: the stop saves the allocation and its record under a name.
- **Expert optimisation**: re-sorts the loaded experts by the run's record with each width's number of experts kept:
  the same memory, a higher mean width for the experts the model actually picks. It runs on a live engine (a full
  change took 83 to 90 s on four nodes, between requests); the result is saved under a name when wanted and opened
  for later starts.

The control panel runs on four or eight nodes, the count chosen on its launch screen. Without it, `./tf-glm53 start`
and `./tf-glm53 engine-up` start them with the last start's plan, or every expert at 3 bits.
[PANEL-GUIDE](glm53/release/PANEL-GUIDE.md), `packaging/RUNTIME.md` ("The control panel") and `glm53/release/T4.md` give
the details.

## Weights on Hugging Face

| What | Repository | Size | Used for |
|---|---|---|---|
| Base weights: everything but the routed experts | [fiore-lab/GLM-5.3-EXL3-Base](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Base) | 21.2 GB | **Required** on every start, on four nodes and on eight |
| Expert bank, 2 bits | [fiore-lab/GLM-5.3-EXL3-Experts-2bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-2bit) | 187.3 GB | Optional: for experts at 2 bits |
| Expert bank, 3 bits | [fiore-lab/GLM-5.3-EXL3-Experts-3bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-3bit) | 277.9 GB | **Required** on every start |
| Expert bank, 4 bits | [fiore-lab/GLM-5.3-EXL3-Experts-4bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-4bit) | 368.5 GB | Optional: for experts at 4 bits (not needed with the eight-node checkpoint) |
| Expert bank, 5 bits | [fiore-lab/GLM-5.3-EXL3-Experts-5bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-5bit) | 459.1 GB | Optional: for experts at 5 bits |
| Eight-node checkpoint: every weight, the routed experts at 4 bits | [fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks) | 390 GB | Optional: not needed to start. Named by `WEIGHTS` it works as it is, and it serves as the 4-bit bank |
| Four-node checkpoint: every weight, the least used half of the routed experts at 3 bits | [fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks) | 344.6 GB | Optional: not needed to start. Named by `WEIGHTS` it works as it is |

A bank holds the 19,200 routed experts at one width; the engine loads each expert from the bank of the width its plan
gives it. **Every start needs the Base weights and the 3-bit bank, 299.1 GB together on four nodes and on eight; download the other banks only for the widths you
want to use**: the panel offers the widths every node holds. Every node needs the same files at the same paths. Download once and copy to the other
nodes, for example (`python3 -m venv` needs the `python3-venv` package):

```sh
python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install -U huggingface_hub
~/hf-cli/bin/hf download fiore-lab/GLM-5.3-EXL3-Base --local-dir ~/models/GLM-5.3-EXL3-Base
~/hf-cli/bin/hf download fiore-lab/GLM-5.3-EXL3-Experts-3bit --local-dir ~/models/GLM-5.3-EXL3-Experts-3bit
```

[QUICKSTART](glm53/release/QUICKSTART.md) has the copy to the other nodes, [packaging/RUNTIME.md](packaging/RUNTIME.md)
the checks (`./tf-glm53 verify`, `./tf-glm53 verify-banks`).

## Making the weights yourself

Weights cannot be read like code: weights altered to behave differently in some cases look like any others, so you
may not want to rely on weights whose making you cannot check. You do not have to rely on Fiore-lab's. Every weight file this recipe loads, the Base weights, the eight-
and four-node checkpoints and the 2-, 3-, 4- and 5-bit expert banks, can be made from Z.AI's official BF16 checkpoint
([zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16)) with the tools in
[glm53/weights](glm53/weights/README.md), which give the steps, the hardware and the measured times. The files you make
have the same layout as Fiore-lab's and load the same way.

## What you need

- Four or eight DGX Spark systems (GB10) on one RoCE v2 network, or four cabled as a ring. The first reaches every
  node, and itself, over ssh with a key that has no passphrase, their host keys already known. [QUICKSTART](glm53/release/QUICKSTART.md), section 1,
  has the ssh and ConnectX-7 network steps. For four nodes on a switch, one 100 Gb/s port a node is
  enough.
- With the image: the NVIDIA driver 580 or later (the image uses the host's), Docker and the NVIDIA container toolkit
  (DGX OS has both) and `python3` on every node, and about
  115 GiB free on each before the start.
- With the archive instead: DGX OS (Ubuntu 24.04) with the NVIDIA driver 580 or later and cuBLAS 13, and NCCL 2.30.7 on
  every node: NVIDIA's `nvidia-nccl-cu13` wheel, unpacked by pip (`packaging/RUNTIME.md`).
- With the archive, NCCL with an AArch64 fence, which NVIDIA's wheel lacks, is recommended on a switch and required on a
  ring ([RING](glm53/release/RING.md)). The image holds it.
- Fiore-lab's EXL3 Base weights, on Hugging Face, at the same path on every node: [fiore-lab/GLM-5.3-EXL3-Base](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Base) (21.2 GB), on four nodes and on eight.
  A full checkpoint works in their place, [fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks) (390 GB) or [fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks) (344.6 GB); a start does not read its routed experts.
- Every weight file can also be made yourself from Z.AI's official BF16 checkpoint, without any of Fiore-lab's files:
  [glm53/weights](glm53/weights/README.md).
- The 3-bit bank (277.9 GB), at the same path on every node: every start needs it. The other banks only for the
  widths you use (the eight-node checkpoint can serve as the 4-bit bank; with 4 bits alone the context length is shorter).
- `python3` on the nodes for the launcher and the log keeper.

## Run

The release comes as a container image and as an archive with the same files; the image is the way we recommend. With
the weights in place, one line on the first node starts the engine:

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host ghcr.io/fiore-labo/expertatlas-glm5.3:r3.3.7-102 up
```

It ends in `READY` with the addresses: the API at `http://FIRST-NODE:8000/v1` and the control panel at
`http://FIRST-NODE:8010/panel`. If something is missing it ends in `STOP`, with what to do and the line to run next.
[QUICKSTART](glm53/release/QUICKSTART.md) gives the steps and every option.

To the same end by hand, with the image: `docker pull` it on every node, run
`mkdir glm53 && docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/glm53":/out IMAGE install /out`, and set
`RUNTIME=docker` and `IMAGE` in `glm53.conf` ([packaging/RUNTIME.md](packaging/RUNTIME.md), "Running in containers").
Without Docker: download the release archive `tensorfold-r3.3.7-102-linux-aarch64.tar.gz` and follow its `RUNTIME.md`
([packaging/RUNTIME.md](packaging/RUNTIME.md) here). Either way, copy `glm53.conf.example` (eight nodes) or
`glm53.t4.conf.example` (four nodes, see [glm53/release/T4.md](glm53/release/T4.md)) to `glm53.conf`, fill in what it
asks for, then `./tf-glm53 deploy` and `./tf-glm53 up`, which starts the control panel, and start the engine there.
`./tf-glm53 start` starts it with the last start's plan, and `./tf-glm53 down` stops it.

The server speaks the OpenAI chat API, and Anthropic's Messages API for Claude Code, on every address (`HOST=0.0.0.0`)
by default. Neither the API nor the control panel has a key, so anyone who can reach them can use them: keep the nodes on
a closed network or put TLS and access control in front (`HOST=127.0.0.1` and `LAUNCHER_HOST=127.0.0.1` keep them to the
first node).

## Drafters

- `mtp` (default): the MTP layer in the weights, part of the GLM-5.3 weights. About twice the decode speed of no
  drafts, with the same prefill speed.
- `dflash2`: incoai/GLM-5.3-DFlash2. **Not included in this repository or the archive, because of its license**: its
  model card states CC BY-NC-ND 4.0 (non-commercial use, no derivatives), for research and evaluation, with
  commercial licensing from Inco AI. If your use is covered by that license, follow "Using DFlash2" in
  [packaging/RUNTIME.md](packaging/RUNTIME.md#using-dflash2): download it once at a fixed revision, copy it to every
  node at the same path, check the copies, then set `DRAFTER_KIND=dflash2` and `DRAFTER`. The engine quantizes it to
  4 bits in memory at the start and writes nothing to disk. Decode is faster still (Speed).
- `none`: no drafts.

`DRAFTER_KIND` in `glm53.conf` picks one; the launch screen writes its choice there, and `./tf-glm53 -d KIND start` (or `engine-up`)
picks one for a single start.

## Speed

**Measured with release r2-102, with the GPU clock capped at 2.1 GHz (2,100 MHz) on every node.** The engine has changed
since, and these numbers have not been measured again for this release. One request at a time with a coding prompt not
yet cached (reasoning effort low, up to 4,000 reply tokens), the settings of `glm53.conf.example` (eight nodes, FP8 KV
cache) and `glm53.t4.conf.example` (four nodes, 4-bit KV cache; `ROWS` was 16,384 then and is 8,192 now), and Fiore-lab's published checkpoints with their own widths: every routed expert at 4 bits on eight nodes, half at 3 and half at 4 bits on four. A default start loads every expert at 3 bits, which these rows did not measure. Speeds differ with the environment (the checkpoint, the prompt, the clocks): take these numbers as a guide. `packaging/RUNTIME.md` gives the conditions in full.

| Nodes | Prompt | MTP (k=3): time to first token / decode | DFlash2: time to first token / decode |
|---|---|---|---|
| 8 | 128K | 82.8 s / 47.5 tokens/s | 85.2 s / 53.2 |
| 8 | 256K | 172.9 s / 45.7 | 177.6 s / 52.6 |
| 8 | 512K | 376.5 s / 42.8 | 386.0 s / 47.1 |
| 4 | 128K | 120.1 s / 35.3 | 121.9 s / 42.9 |
| 4 | 256K | 253.3 s / 33.2 | 256.9 s / 40.7 |
| 4 | 512K | 565.4 s / 32.2 | 571.5 s / 37.8 |

Prefill at each `ROWS`, measured for this release on four nodes, one run each: 1,054 tokens/s at 8,192, 1,102 at 16,384, 1,110 at 32,768. A reference only; [glm53/release/T4.md](glm53/release/T4.md) gives the conditions.

## Accuracy

KL divergence of this recipe's output distributions from the original BF16 model's (the official BF16 weights run by
the official implementation, computed in FP32), with this recipe's serving configurations:

- Eight nodes (the eight-node checkpoint, every routed expert at 4 bits, FP8 KV cache, DCP 2; measured on the same weights with an earlier build of this engine): four 2,048-token texts (dialogue,
  legal, prose, reasoning), the last 1,023 positions each, five runs: mean 0.0133 (0.0128 to 0.0146); the top-1 token
  agrees at about 96%.
- Eight nodes, four 8,192-token texts (two English, two code), by position range: 0.012 (positions 1,024 to 2,047),
  0.018, 0.019 and 0.013 (6,144 to 8,190). For scale, the official BF16 weights computed in BF16 instead of FP32 differ
  from the same reference by 0.007 to 0.010 over the same ranges.
- Four nodes (the four-node checkpoint's own widths, half at 3 and half at 4 bits, 4-bit KV cache, DCP 2), four 8,192-token texts by the same position ranges:
  0.019 (positions 1,024 to 2,047), 0.032, 0.029 and 0.019 (6,144 to 8,190).
- Expert Atlas's widths, every routed expert at one width with the other weights as above, the same four texts,
  positions 6,144 to 8,190: 2 bits 0.0517, 3 bits 0.0203, 4 bits 0.0127 (four nodes), 5 bits 0.0114 (eight nodes).
  A default start loads every expert at 3 bits.
  At the same memory, a mix with 2-bit experts was 13 to 30 % further from the reference than one without, and 5-bit
  experts paid for by moving others down to 3 bits gained nothing measurable.

## Build from source

On a GB10 system with Zig 0.17.0 on the `PATH` (or `ZIG=` its path for `build_libs.sh`), CUDA 13 (`nvcc`), the
libibverbs headers (`libibverbs-dev`) and Rust 1.95.0 through rustup (`rustup toolchain install 1.95.0`;
`build_libs.sh` runs `~/.cargo/bin/cargo`, or the one `CARGO=` names):

```sh
zig build fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 --prefix /tmp/fatbins
sh glm53/tools/release/build_libs.sh /tmp/lib        # lib/libroce_proxy.so, lib/libllguidance.so
zig build dist -Dversion=r3.3.7-102 -Ddist-commit=$(git rev-parse HEAD) -Ddist-platforms=linux-aarch64 \
    -Ddist-fatbins=/tmp/fatbins/fatbin -Ddist-lib=/tmp/lib -Ddist-extra=glm53/release
```

The archive lands in `zig-out/dist/`. `build_libs.sh` builds llguidance offline, from `third_party/llguidance/` and the
crates vendored there. The image and the archive need no Rust.

## Limits

- Four or eight GB10 systems. Two- and three-node forms are being tested. Other GPUs and other rank counts are not
  supported.
- By default one request runs at a time; up to 32 others wait in arrival order, chat-title requests last, and past 32 the
  server refuses them (a 503, or an error event in a stream). `MULTI` decodes up to four together, at a cost in node
  memory. Agents that run sub-agents in parallel are otherwise better served by pointing the sub-agents at another model.
- Structured output takes a JSON object or a JSON Schema (no regex, choice or grammar yet), and not beside tools.
- The control panel's screen is in English and Japanese only.
- `packaging/RUNTIME.md` lists the rest.

## License and credits

The code is Apache-2.0 (`LICENSE`); `NOTICE` and `THIRD_PARTY_NOTICES.md` say what comes from where. The weights keep
their own license (GLM-5.3 License); the optional DFlash2 drafter keeps its own.

- [TensorFold](https://github.com/ashhart/TensorFold) by the TensorFold contributors: the server, the build and the
  CUDA engine this recipe extends.
- TensorFold's Python engine for full GLM-5.3 by drowzeys
  ([ashhart/TensorFold#159](https://github.com/ashhart/TensorFold/pull/159)): the GLM-5.3 engine rewrites its design in
  Zig and plain CUDA (the engine's structure, its Triton kernels and its MTP and DFlash2 drafter paths). Some of its
  CUDA files are included, a few of them modified.
- [b12x](https://github.com/local-inference-lab/b12x): its RoCE one-shot collectives are rewritten as plain CUDA
  (`roce.cu`), and its RoCE proxy is included (`zig/native/roce_proxy.c`), modified to pick a GID index for each RDMA
  device.
- [MiaAI-Lab's GLM-5.3-Flash on 2x DGX Spark](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold):
  the GLM-5.3 prompt kernels are written with reference to its prompt kernels.
- [MiaAI-Lab's GLM-5.3 EXL3 on 3x DGX Spark](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold): the
  4-bit KV cache takes its layout, implemented with this engine's own code.
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3): the checkpoints are quantized with it, and the engine reads
  its EXL3 format.
- The [gr8 recipe](https://github.com/knapcio/GLM-5.3-gr8-4x-DGX-Spark-TP4) and [engine](https://github.com/kindlingai/gr8):
  several decode speed-ups take their ideas, implemented with this engine's own code.
- [ollama/ollama#12375](https://github.com/ollama/ollama/issues/12375): Byte Resume takes its idea from this report.

## TensorFold's contributors

From TensorFold's README: thank you to everyone who has sent TensorFold a pull request, a measurement or a bug report
(the list as of TensorFold 1.0.2; see [the TensorFold repository](https://github.com/ashhart/TensorFold) for its
current list):

[@0mao0](https://github.com/0mao0),
[@321sssrt-bit](https://github.com/321sssrt-bit),
[@aditya1503](https://github.com/aditya1503),
[@AdrianBinDC](https://github.com/AdrianBinDC),
[@akol1](https://github.com/akol1),
[@Alexbob0](https://github.com/Alexbob0),
[@anvilsong](https://github.com/anvilsong),
[@Arminova](https://github.com/Arminova),
[@b-ostrov](https://github.com/b-ostrov),
[@barelyworkingcode](https://github.com/barelyworkingcode),
[@Benjamin-Wegener](https://github.com/Benjamin-Wegener),
[@benthecarman](https://github.com/benthecarman),
[@benwilson](https://github.com/benwilson),
[@BHCC2025](https://github.com/BHCC2025),
[@Bizuayeu](https://github.com/Bizuayeu),
[@BlivionIaG](https://github.com/BlivionIaG),
[@BobClawblaw](https://github.com/BobClawblaw),
[@borodach23](https://github.com/borodach23),
[@Boscoeuk](https://github.com/Boscoeuk),
[@boxabirds](https://github.com/boxabirds),
[@brandonmmusic-max](https://github.com/brandonmmusic-max),
[@bunnyfu](https://github.com/bunnyfu),
[@CerebralCoding](https://github.com/CerebralCoding),
[@cesarswong](https://github.com/cesarswong),
[@chadhurley25075-png](https://github.com/chadhurley25075-png),
[@chaog992](https://github.com/chaog992),
[@Charlie-Louis](https://github.com/Charlie-Louis),
[@Chedrian07](https://github.com/Chedrian07),
[@chris247474](https://github.com/chris247474),
[@christrade215](https://github.com/christrade215),
[@crescit](https://github.com/crescit),
[@cshintov](https://github.com/cshintov),
[@cwschroeder](https://github.com/cwschroeder),
[@DakotaTexas](https://github.com/DakotaTexas),
[@danyo1399](https://github.com/danyo1399),
[@Deesha08](https://github.com/Deesha08),
[@Defilan](https://github.com/Defilan),
[@DevRico003](https://github.com/DevRico003),
[@di37](https://github.com/di37),
[@drowzeys](https://github.com/drowzeys),
[@ecohash-co](https://github.com/ecohash-co),
[@edurdias](https://github.com/edurdias),
[@eleqtrizit](https://github.com/eleqtrizit),
[@ericlsimplifi](https://github.com/ericlsimplifi),
[@EugeneClaw](https://github.com/EugeneClaw),
[@feni6](https://github.com/feni6),
[@gbgbgbg](https://github.com/gbgbgbg),
[@GDACONSULT](https://github.com/GDACONSULT),
[@gecobattya](https://github.com/gecobattya),
[@gilby](https://github.com/gilby),
[@Gogo6969](https://github.com/Gogo6969),
[@gprot42](https://github.com/gprot42),
[@GraithSecurity](https://github.com/GraithSecurity),
[@grantoverton](https://github.com/grantoverton),
[@grearjake-star](https://github.com/grearjake-star),
[@greatyingzi](https://github.com/greatyingzi),
[@harrisonfriia](https://github.com/harrisonfriia),
[@haxudev](https://github.com/haxudev),
[@heitke](https://github.com/heitke),
[@hichaiuse](https://github.com/hichaiuse),
[@ivanfioravanti](https://github.com/ivanfioravanti),
[@jasontitus](https://github.com/jasontitus),
[@jayleaton](https://github.com/jayleaton),
[@jeffpeng3](https://github.com/jeffpeng3),
[@jeidbugs404](https://github.com/jeidbugs404),
[@jetnet](https://github.com/jetnet),
[@jkuepker](https://github.com/jkuepker),
[@johnymoo](https://github.com/johnymoo),
[@JordiPosthumus](https://github.com/JordiPosthumus),
[@JRaxworthy](https://github.com/JRaxworthy),
[@jregan-beasley-smc](https://github.com/jregan-beasley-smc),
[@jschmied](https://github.com/jschmied),
[@juliankang4](https://github.com/juliankang4),
[@kingjamez](https://github.com/kingjamez),
[@kky42](https://github.com/kky42),
[@lcgutierrez](https://github.com/lcgutierrez),
[@LECYWZA](https://github.com/LECYWZA),
[@lijian1999](https://github.com/lijian1999),
[@liumorrisclaw](https://github.com/liumorrisclaw),
[@LXD-8](https://github.com/LXD-8),
[@m-naoki-m](https://github.com/m-naoki-m),
[@mapamalu](https://github.com/mapamalu),
[@mcclanahanaman](https://github.com/mcclanahanaman),
[@MESevenJourney](https://github.com/MESevenJourney),
[@mgoldwasser](https://github.com/mgoldwasser),
[@MiaAI-Lab](https://github.com/MiaAI-Lab),
[@mikolaj92](https://github.com/mikolaj92),
[@millaguie](https://github.com/millaguie),
[@Mirrdhyn](https://github.com/Mirrdhyn),
[@Moutonc](https://github.com/Moutonc),
[@MovieMaker93](https://github.com/MovieMaker93),
[@mrpmorris](https://github.com/mrpmorris),
[@MV10](https://github.com/MV10),
[@NeoAiLabs](https://github.com/NeoAiLabs),
[@Nipale-ai](https://github.com/Nipale-ai),
[@nood-co1](https://github.com/nood-co1),
[@nullburn](https://github.com/nullburn),
[@olexale](https://github.com/olexale),
[@omar16100](https://github.com/omar16100),
[@optimisme](https://github.com/optimisme),
[@outcastofmusic](https://github.com/outcastofmusic),
[@paragontasx](https://github.com/paragontasx),
[@peacockesq](https://github.com/peacockesq),
[@philip-pentatonic](https://github.com/philip-pentatonic),
[@plotarmordev](https://github.com/plotarmordev),
[@pmeenan](https://github.com/pmeenan),
[@pulseandthread](https://github.com/pulseandthread),
[@quigles1977](https://github.com/quigles1977),
[@rafafortes](https://github.com/rafafortes),
[@raymondkpwong](https://github.com/raymondkpwong),
[@robertpitt](https://github.com/robertpitt),
[@RoscoeTT](https://github.com/RoscoeTT),
[@salmanarshad321](https://github.com/salmanarshad321),
[@samwang0041-star](https://github.com/samwang0041-star),
[@sanjaibalajee](https://github.com/sanjaibalajee),
[@satindergrewal](https://github.com/satindergrewal),
[@scottleimroth](https://github.com/scottleimroth),
[@sethforprivacy](https://github.com/sethforprivacy),
[@sfxnz](https://github.com/sfxnz),
[@shantanugoel](https://github.com/shantanugoel),
[@simon-lin88](https://github.com/simon-lin88),
[@simonmd](https://github.com/simonmd),
[@spenchey](https://github.com/spenchey),
[@squarrier](https://github.com/squarrier),
[@ss-cong](https://github.com/ss-cong),
[@styles01](https://github.com/styles01),
[@SxMShaDoW](https://github.com/SxMShaDoW),
[@sxuff](https://github.com/sxuff),
[@taussoe](https://github.com/taussoe),
[@tfolkman](https://github.com/tfolkman),
[@ThinkOffApp](https://github.com/ThinkOffApp),
[@Thotheris](https://github.com/Thotheris),
[@tinyapps](https://github.com/tinyapps),
[@tolewis](https://github.com/tolewis),
[@tomByrer](https://github.com/tomByrer),
[@tonydehnke](https://github.com/tonydehnke),
[@tournierjc](https://github.com/tournierjc),
[@tpischke](https://github.com/tpischke),
[@urtho](https://github.com/urtho),
[@vcruz305](https://github.com/vcruz305),
[@vinicius-symetrix](https://github.com/vinicius-symetrix),
[@wojo](https://github.com/wojo),
[@xjqx2z](https://github.com/xjqx2z),
[@Yuepixel](https://github.com/Yuepixel),
[@YvesLaRose](https://github.com/YvesLaRose).
