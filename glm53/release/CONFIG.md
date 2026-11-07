English | [日本語](CONFIG.ja.md) | [简体中文](CONFIG.zh-CN.md)

# glm53.conf: the settings

With `docker run ... IMAGE up`, the keys of the engine settings can be given as `-e KEY=value` (`QUICKSTART.md`). These stop the line
with `not an option`, so write them in `~/expertatlas/glm53.conf`: `GID_INDEX`, `RUNTIME`, `NCCL_LIB`, `EXPERT_LIVE`, `LOG_LISTEN`, `CONTEXT`, `STORE`, `THINKING_FOLD_KEEP`, `NEED_GB`, `LAUNCHER_HOST`, `ATLAS_DIR`, `LAUNCH_SNAPSHOTS`. The node check decides `TOPOLOGY`, `SOCKET_IFNAME` and `IB_HCA`.

The defaults below are `tf-glm53`'s. `docker run ... IMAGE up` writes `glm53.conf` from `glm53.t4.conf.example` on any node count, so it begins
with `CONTEXT_LENGTH` 1048576, `KV_SIZE` 1049088, `KV` fp4 and `NEED_GB` 115, and the start's KV size is `max`.

`glm53.conf` holds a lane's settings: the nodes, the network, the weights, the drafter, the server, the memory and
the control panel. It is a plain shell file that you write and edit by hand, next to `tf-glm53` on the first node.
Start from `glm53.t4.conf.example` (four nodes) or `glm53.conf.example` (eight nodes): copy it to `glm53.conf` and
fill in the nodes, the network devices and the paths. The examples carry the measured values for the rest.

## Who reads it, and when a change takes effect

- `tf-glm53` reads it at every command. Only the first node's file is read: `tf-glm53 deploy` copies this directory
  to the other nodes without `glm53.conf`, and each rank gets its settings from the first node when it starts.
- A running engine keeps the settings it started with. A change reaches the engine at its next start
  (Stop, then Start on the control panel; or `./tf-glm53 down`, then `./tf-glm53 start`).
- The control panel's launcher reads it when the launcher starts, and again by itself whenever the file changed
  while the engine is stopped (an edit made while the engine runs is read once it stops). `LAUNCHER_HOST` and
  `LAUNCHER_PORT` take effect when the launcher starts again: `./tf-glm53 launcher stop`, then `./tf-glm53 up`. The
  engine may keep running meanwhile: the launcher finds it running.
- Nothing else writes it, with two exceptions, both on the control panel. Each writes only the lines it changes (one
  line a setting, a comment after the value kept, appended when absent) and keeps the previous file beside it as
  `glm53.conf.bak-<time>`:
  - Save under the weights on the launch settings writes `WEIGHTS` and `DRAFTER` once every node has what they name
    (the expert banks' directories go to `expert-plan/banks.json` the same way, copied to every node), and the
    launcher reads them again at once.
  - The engine settings screen (switched to from the header) writes the settings that need a restart:
    `NAME`, `CONTEXT_LENGTH`, `KV_SIZE`, `KV`, `MAX_TOKENS`, `THINKING_*`, `MULTI*`, `DRAFTER_KIND`, `MTP_K`, `DRAFT_SAVES`, `DCP`, `MASTER_PORT`, `RESERVE_GB`, `READY_TIMEOUT`, `LAUNCH_DROP_CACHES`, `HOST`, `PORT`, `METRICS_LISTEN`, `LOG_KEEP_MB`, `STRUCTURED`, `LAUNCHER_PORT`, `ROWS`, `SP_PARTS`, `ROCE_PROXY_CORES`, `ENGINE_CORES`, `ATLAS_GENERIC`, `EXTRA_ENV` (the engine's `TF_GLM53_CANON_CLOSE` and `TF_GLM53_CANON_NEAR` as tokens of it); `TOPOLOGY`,
    `SOCKET_IFNAME` and `IB_HCA` too until the node check has registered the nodes. Each value is checked first as
    `tf-glm53` checks it; `PORT` only while the engine is stopped. A change is written as it is made and used from the next start (a running engine
    keeps its settings); the thinking's live keys (`THINKING_SHARE`, `THINKING_FOLD`, `THINKING_FOLD_AT_CLOSE_SHARE`,
    `THINKING_FOLD_SHARE`, `THINKING_FOLD_MAX`) also go to a running engine at once, and the inference screen's [ REASONING CONTROL ] writes its
    changes here, so both screens show one value. Restart to apply, while the engine runs: the stop first asks whether to keep this run's
    usage, as Stop does, then the engine starts again with the same experts and the saved settings.
- The nodes the control panel's node check registers go to their own file, `glm53.nodes.conf` beside this one
  (`TOPOLOGY`, `NODES`, `SOCKET_IFNAME`, `IB_HCA`, as the check judged the cabling; the previous file stays beside it
  as `glm53.nodes.conf.bak-<time>`). `tf-glm53` and the launcher read it after `glm53.conf`, so its lines win over
  those four here. While it is missing, the panel opens on the node check; once it is there, on the launch settings.
  `NODES_CONF=FILE` reads another file, `NODES_CONF=` none. `tf-glm53 deploy` does not copy it.
- What the launch screen chooses (the KV format, the context length, the drafter, the concurrent inferences, what each node
  keeps for the OS) is written into this file as it is chosen (`KV`, `CONTEXT_LENGTH`, `DRAFTER_KIND`, `MULTI`, `RESERVE_GB`), and a
  start writes the context length its plan took, so the next page opens on it. `KV_SIZE` only rises to what the context
  length needs. The experts' widths and the KV the plan takes are that start's: the launcher hands them to `tf-glm53` in
  its environment (`LAUNCH_*`); they are not settings of this file.
- The page's own settings (colours, the circle's order and the rest) are kept in `launch/panel-settings.json`.
- `glm53.conf` is not one of the release's files: `SHA256SUMS` does not list it, so `tf-glm53 check` and
  `tf-glm53 deploy` never compare it. Keep a copy of a working file before a large change.
- Another settings file: `tf-glm53 -c FILE COMMAND` (the launcher too: `./tf-glm53 -c FILE up`). Two lanes on
  the same nodes need their own `PORT`, `MASTER_PORT` and `LAUNCHER_PORT`.

## The settings

"Default" is what `tf-glm53` uses when a setting is missing or empty; "(needed)" has none. Settings marked
**panel** can also be chosen on the launch screen, which writes it here; a start without that screen (`tf-glm53 start`,
`engine-up`) takes the file's value.

### Nodes and network

| Setting | Default | Meaning |
|---|---|---|
| `NODES` | (needed) | The nodes in rank order, by their address on the RDMA network, as a shell array: `NODES=(192.0.2.11 192.0.2.12 192.0.2.13 192.0.2.14)`. Four or eight (two and three are being tested). Run `tf-glm53` on the first. Every node reaches the others over ssh without a password. The control panel's node check registers the nodes in `glm53.nodes.conf` instead, which wins over this and the next three. |
| `SOCKET_IFNAME` | (needed) | The network interface of that network on every node (`ip -br addr`), for example `enp1s0f1np1`. |
| `IB_HCA` | (needed) | The RDMA device of that interface (`ibv_devices`), for example `rocep1s0f1`. |
| `TOPOLOGY` | `full` | How the nodes are cabled. `full`: a switch, or one link. `ring4`: four nodes cabled as a ring without a switch, NODES in ring order (RING.md). `auto`: judged from each node's RDMA devices and LLDP neighbours at each `check` and `./tf-glm53 up` (`tf-glm53 topology` shows the judgment): a ring takes `ring4` with NODES put in ring order (rank 0 kept), a switch or one link `full`; `IB_HCA` and `SOCKET_IFNAME` are then the judgment's and may be left out (`IB_HCA` only picks a switch's half). A cabling that is neither is refused with what to fix; nothing falls back. **Known bug:** with `TOPOLOGY=auto`, `IB_HCA` and `SOCKET_IFNAME` left out and nothing judged yet, a first start with `./tf-glm53 start` or `engine-up` stops with `NG: auto RoCE-v2-GID(gid=)`: those commands do not judge, `check` judges without saving, and only `./tf-glm53 up` saves the judgment. Run `./tf-glm53 up` first. With those two written there is no such `NG`, but with nothing judged the start takes `full`, which does not work on a ring. `docker run ... IMAGE up` and `TOPOLOGY=full` or `ring4` are not affected. |
| `GID_INDEX` | empty | The RoCE v2 GID index. Empty: found on each node (the last RoCE v2 entry with an IPv4 address). |
| `MASTER_PORT` | `29551` | The first node's port for the NCCL rendezvous. |

### Where the ranks run

| Setting | Default | Meaning |
|---|---|---|
| `RUNTIME` | `host` | `host`: from this directory, with cuBLAS 13, libibverbs and NCCL on every node. `docker`: each rank in a container of `IMAGE` (RUNTIME.md, "Running in containers"). |
| `IMAGE` | empty | With `RUNTIME=docker`: this release's image. |
| `NCCL_LIB` | `~/nccl-2.30.7/nvidia/nccl/lib/libnccl.so.2` | NCCL 2.30.7 at this path on every node (`RUNTIME=host` only; RUNTIME.md, "Before the first start"). NVIDIA's wheel runs on a switch; the build with the AArch64 fence is recommended, and a ring requires it (RING.md). |

### Weights and drafter

| Setting | Default | Meaning |
|---|---|---|
| `WEIGHTS` | (needed) | The directory of the Base weights (`GLM-5.3-EXL3-Base`), the same path on every node, on four nodes and on eight. A full checkpoint (`GLM-5.3-EXL3-4x-DGX-Sparks`, `GLM-5.3-EXL3-8x-DGX-Sparks`) works in their place. The launch settings can set it, and the expert banks' directories (`expert-plan/banks.json`), once every node has them. |
| `DRAFTER_KIND` | `mtp` | **panel.** What drafts the tokens the model then checks: `mtp` (the MTP layer in `WEIGHTS`), `dflash2` (the model in `DRAFTER`) or `none`. `./tf-glm53 -d KIND start` (or `engine-up`) picks another for one start. |
| `MTP_K` | `3` | Drafts a round with MTP, 1 to 7. |
| `DRAFTER` | empty | The DFlash2 model's directory, the same path on every node. `DRAFTER_KIND=dflash2` needs it, and the launch screen offers DFlash2 only when it is set and every node has the model. Not part of the release (RUNTIME.md, "Using DFlash2"). The launch settings can set it. |
| `DRAFT_SAVES` | `8` | With DFlash2: the conversations whose drafter state is kept between requests. |

### The server

| Setting | Default | Meaning |
|---|---|---|
| `NAME` | `glm-5.3` | The model's name in the API: `/v1/models` lists it and replies carry it (a request may name any model). |
| `HOST` | `0.0.0.0` | The address the API listens on. `0.0.0.0`: every address. The API has no key, so anyone who reaches it can use it: keep the nodes on a closed network, or put TLS and access control in front. `127.0.0.1`: this node only; another machine then reaches it through an ssh tunnel. |
| `PORT` | `8000` | The API's port. |
| `METRICS_LISTEN` | empty | `HOST:PORT` where the first node also answers `GET /metrics` (Prometheus text), `GET /health` and `GET /v1/experts/usage` (the experts' use counts), read-only, for a monitor on another host. Empty: off. |
| `EXPERT_LIVE` | `1` | `1`: the first node counts the experts each prompt row and each decoded token picks, for the control panel (its live view, the usage records and expert optimisation). `0`: not counted; the panel then shows the running engine with Stop, but no picks and no optimisation. A start from the panel always counts. |
| `LOG_LISTEN` | empty | `HOST:PORT` where each node's log keeper also answers, read-only and with no key, `GET /streams` and `GET /logs?stream=S&after=N`, for a monitor on another host that keeps the logs. Empty: the logs stay in each node's memory only (`tf-glm53 logs`). A change takes effect at the engine's next start, which starts the keepers anew (the lines they held are lost). |
| `LOG_KEEP_MB` | `8` | The MB each log (each rank's, the watcher's, the launcher's) keeps in memory; older lines go. Nothing is written to disk. A change takes effect as `LOG_LISTEN`'s does. |
| `STRUCTURED` | `1` | `1`: structured output, a reply's answer held to JSON or to a JSON Schema (`response_format` and the like; FEATURES.md, "Structured output"). Rank 0 loads `lib/libllguidance.so` at the start; when it cannot, the engine starts without structured output. `0`: off, and requests that ask for it get 400. A change takes effect at the next start. |

### Context, cache and KV

| Setting | Default | Meaning |
|---|---|---|
| `CONTEXT_LENGTH` | `819200` | **panel.** The positions one request may reach (prompt + reply): 65,536 to 1,048,576 (the model's longest) in steps of 16,384. |
| `KV_SIZE` | `1638400` | **panel.** The whole KV in positions, shared by every conversation: `CONTEXT_LENGTH` + 512 at least. A start from the panel takes this value once it is set in the panel (here, or the launch screen's KV size); what a node keeps, typed on the launch screen, gives the KV all the room its plan leaves again. A start from `glm53.conf` alone takes this value. |
| `CONTEXT`, `STORE` | | The earlier names of `CONTEXT_LENGTH` and `KV_SIZE`, still read when those are absent: `CONTEXT_LENGTH` = `CONTEXT`, `KV_SIZE` = `CONTEXT` + `STORE` (`STORE` 819,200 when unset). A save from the engine settings screen writes the new names and drops these lines. |
| `KV` | `fp8` | **panel.** The KV cache's format: `fp8`, or `fp4` (less memory a position, a little less accurate; T4.md). |
| `DCP` | `2` | Decode context parallelism: `2` splits the KV between the ranks of a pair. |
| `MULTI` | `0` | Up to this many conversations (1 .. 4) decoded together in one window a round; `0` runs requests one at a time. Works with the MTP layer, DFlash2 (`DRAFT_SAVES` of 1 or more) or no drafter; not on 3 nodes yet (refused before the start). Expert optimisation works as with `0`: a change waits until no request runs or waits. |
| `MULTI_WAIT` | `600` | With `MULTI`: seconds a request may wait for room in the conversation cache beside the conversations in flight; past it the request is refused as busy so the client retries: HTTP 503 (`overloaded_error` on the Messages endpoint), or in a stream already begun an error event of type `overloaded_error` (Messages), `server_error` with code `overloaded` (Chat Completions), or a failed response with code `rate_limit_exceeded` (Responses). The wait counts from the request's first try that found no room, and is judged between decode rounds and prompt chunks, so a refusal may come a round late. A request cancelled while it waits ends as cancelled. `0`: no limit. |
| `MULTI_DECODE_SHARE` | `30` | With `MULTI`: the decode's percent of the GPU's time while a prompt waits or is computed beside conversations that are decoding. `0` puts the prompts first (no decode round inside a prefill, the queued requests admitted one after another); `100` puts the decode first (a prompt waits while any conversation decodes, a prefill already running pauses until they end; a short prompt still goes, see `MULTI_SHORT_PROMPT`); between, the engine measures both and keeps that share. With only one kind of work it takes the whole GPU. This is the value at start: `POST /v1/schedule {"decode_share": N}` (an API key, or a client on the head node) changes it while the engine runs, and a prefill already running takes it within a few layers; `GET /v1/schedule` and `/health` (`multi.decode_share`, `multi.measured`) show it. A wait for the share is not limited by `MULTI_WAIT` (that is a wait for room). |
| `MULTI_SHORT_PROMPT` | `8192` | With `MULTI`: a prompt with at most this many tokens to compute (past what the conversation cache reuses, so a long conversation's next turn counts only its new part) is short: it goes before the longer prompts waiting, is not held by `MULTI_DECODE_SHARE`, and its prefill runs to its end without decode rounds (whatever the share, the GPU is the prompt's while it runs: the conversations decoding wait for it, about 7 s for 8,192 tokens on four nodes). It waits for a prefill already running unless `MULTI_SHORT_PREEMPT=1`. `0`: no prompt is short (the queue keeps its order). This is the value at start: `POST /v1/schedule {"short_prompt": N}` changes it while the engine runs, from the next admission; `GET /v1/schedule` and `/health` (`multi.short_prompt`) show it. |
| `MULTI_SHORT_PREEMPT` | `0` | With `MULTI` and `MULTI_SHORT_PROMPT`: `1` makes a longer prefill already running give way to a short prompt waiting, at its next pause for the decode, or at its next chunk when nothing else decodes. The chunks it finished stay in the cache and it resumes from them afterwards, its stream open; the work of a chunk given up at a pause is computed again (up to one chunk: about +14 s for a long prompt at a share of 30 on four nodes), and after each resume it finishes a chunk before it gives way again. `0` (the default): a short prompt waits for the prefill running. `POST /v1/schedule {"short_preempt": true}` changes it while the engine runs; `GET /v1/schedule` and `/health` (`multi.short_preempt`) show it. |
| `ROWS` | `8192` | The rows of a prompt chunk: 256 to 65,536, in steps of 256. More rows use more memory, about 0.41 GiB a node for each 1,024 rows (measured on four nodes). A start's plan takes that from the KV, so the KV gets smaller. |
| `SP_PARTS` | `4` | The sequence-parallel parts a prompt chunk is split into: 2 to 4. |
| `ROCE_PROXY_CORES` | `a725` | The cores the RoCE proxy threads (two a rank: the nodes' one and the DCP pair's) poll on. They spin while the decode runs. `a725` keeps them on the GB10's Cortex-A725 cores 0 and 10 (one a cluster); `os` leaves them to the scheduler, which often puts them on X925 cores. On four nodes, A725 ran the SoC 5-16 °C cooler than the scheduler's choice, with the same decode speed. On a CPU without that kind of core they are left to the scheduler. The engine's log says where each one polls; `/health` `roce` shows every rank's (rank 0's read live). |
| `ENGINE_CORES` | `os` | The cores every rank runs on. `os`: the scheduler's, every core of the node. `a725`: each node's Cortex-A725 cores only. `ROCE_PROXY_CORES=x925` cannot go with `a725`: `tf-glm53` refuses it. The speed on A725 alone has not been measured. |
| `MAX_TOKENS_SHARE` | `70` | A reply's max_tokens when the request names none, as a percent of `CONTEXT_LENGTH` (1 to 100). A client's `max_tokens` comes first. |
| `MAX_TOKENS` | (empty) | The same as a token count. When set it comes before `MAX_TOKENS_SHARE`; a save of `MAX_TOKENS_SHARE` on the panel empties it. With no summaries midway and no organising at the budget's end, a request that asks for more than the context has left is refused (400). With summaries midway the engine fits its interval to the room; a request that names its own `thinking_fold_window` is refused (400) if that does not fit. |
| `THINKING_SHARE` | `90` | **live.** The percent of a reply's max_tokens, at most of what the prompt leaves of `CONTEXT_LENGTH`, its thinking may use (`0`: no limit); the rest is the answer's. At that many thinking tokens, at the next line break (at most 256 tokens later), the engine writes a closing sentence and `</think>`; the answer follows in the rest. With `THINKING_FOLD` the thinking goes on past it instead: it folds when it fills that percent of the context, and its budget is max_tokens less the answer's part. A request's `thinking_budget` (Anthropic: `thinking.budget_tokens`) comes first. |
| `THINKING_BUDGET` | `0` | A reply's thinking tokens at most, as a count. With `THINKING_FOLD` it is a virtual length that may pass `CONTEXT_LENGTH`, since the folded thinking leaves the context. At the budget the engine closes the thinking at a line break and the answer goes on. `0`: `THINKING_SHARE` decides. A request's `thinking_budget` comes first. |
| `THINKING_NOTICE` | `0` | Every this many thinking tokens (from `256`; `0`: none), at the next line break, the engine writes a line into the thinking that tells the model how many thinking tokens it used and how many are left (of the budget, or of max_tokens without one). The line is part of the reply's reasoning (`reasoning_content`). Each costs a short re-admission: the cache resumes all but the line. |
| `THINKING_FOLD` | `off` | **live.** Folds a long thinking in the model's context so its KV stays bounded; the client still gets all of the reasoning. `recent` keeps the thinking's first 100 tokens and its last `THINKING_FOLD_KEEP`. `checkpoint` asks the model for a checkpoint first (its conclusions so far, what was ruled out, what is still open and what to examine next, ended by `END CHECKPOINT`, at most `THINKING_FOLD_CAP` tokens), which then replaces the reasoning before it; the model goes on thinking only about what is open. `branch` sets the reply aside and asks the model, in an answer beside it (the block closed, a user turn, an answer without thinking), for a summary of the reasoning since its last summary; an answer ends at its EOS, where a checkpoint written in the thinking seldom writes `END CHECKPOINT`. The summaries stack in the thinking, each under "(My reasoning grew long, so I summarized it once ...)", and the client gets them in the reasoning. It folds when the thinking fills its `THINKING_SHARE` of what the prompt leaves of the context (all of it with `THINKING_SHARE=0`), less the summary's room, and then thinks on past the context. A request may set `thinking_fold`, `thinking_fold_window`, `thinking_fold_max` and `thinking_fold_keep`. |
| `THINKING_FOLD_SHARE` | `100` | **live.** With `THINKING_FOLD`: a fold no later than this percent of each reply's max_tokens (at least 1024 tokens), 10 to below `THINKING_SHARE`; `100`: no earlier than the limit. A request's `thinking_fold_window` comes first. |
| `THINKING_FOLD_MAX` | `0` | **live.** With `THINKING_FOLD`: a reply's folds at most (`0`: no limit). Past them, or when the summaries kept and one more would pass the thinking's share of the context (the summaries stack there), it folds no more: the thinking goes on until it fills that share and then closes as at its budget's end (with `THINKING_FOLD_AT_CLOSE_SHARE`, the summary takes the closing sentence's place). When the thinking budget left holds no summary midway and the summary at the close, the thinking closes at the fold instead. A request's `thinking_fold_max` comes first. |
| `THINKING_FOLD_KEEP` | `0` | With `THINKING_FOLD=recent`: the last thinking tokens kept (`0`: half the window). |
| `THINKING_FOLD_AT_CLOSE_SHARE` | `0` | **live.** The summary at the thinking's close: when the model ends a thinking with at least this percent of the thinking's limit in its context (past the last fold; `0`: never, 1 to 100; a request's `thinking_fold_at_close` names tokens and comes first), it first writes a checkpoint of what the reasoning learned (the facts and results it settled, intermediate results with their conditions, every idea it ruled out with the reason, the mistakes it corrected, the hypotheses still open, the plan of its answer and the points to keep right; the back-and-forth and the restatements are left out; at most `THINKING_FOLD_CAP` tokens unless the request's `thinking_fold_cap` says otherwise), which replaces the thinking, and the answer is written from it; the next turns get the summary too. When the thinking budget ends the thinking instead, the same summary takes the place of the budget's closing sentence, and it also records where the thinking stopped and, for each point left unsettled, its best answer so far, how sure it is and what the answer does about it. A summary is asked for only when it and an answer as long still fit in the reply's `max_tokens`. `0`: never. Works with `THINKING_FOLD=off` too. A request's `thinking_fold_at_close` comes first. |
| `THINKING_FOLD_AT_BUDGET` | `1` | **live.** Organises the thinking near the end of its budget: it replaces the thinking in the context with its rewrite, and the model then decides whether to close. `0`: off. A request's `thinking_fold_at_budget` overrides it. |
| `THINKING_FOLD_MARGIN` | `10` | **live.** The organising starts this percent before the end of the thinking's budget (0 to 50). |
| `THINKING_FOLD_PROMPT` | the org text | **live.** The instruction for that organising. Empty: back to the default. |
| `THINKING_CLOSE_PROMPT` | the org text | **live.** The instruction for the summary at the thinking's close. Empty: back to the default. |
| `TF_GLM53_BYTE_RESUME` | `on` | A model may cut a text differently from the tokenizer, and the next turn's prompt then holds the same bytes under other ids, so the cache stops there. `on`: past the first id that differs, the prompt is compared with the kept conversations in bytes, and where the bytes agree it takes their ids, so the cache resumes past the place. The text does not change; the reply may differ slightly from one computed without the cache. `off`: the cache stops at that place. `log`: as `off`, and compares and logs (`byteresume:` in `tf-glm53 logs 0`). |
| `THINKING_FOLD_CAP_SHARE` | `70` | A summary's tokens at most, as a percent of the thinking it organises (10 to 100, at least 128 tokens): the checkpoint of `THINKING_FOLD=checkpoint` and the summary of `THINKING_FOLD_AT_CLOSE_SHARE`. A request's `thinking_fold_cap` (tokens, 128 to 16384) comes first. |

### Memory and start

| Setting | Default | Meaning |
|---|---|---|
| `NEED_GB` | `112` | The GiB a node must have available before a start (`tf-glm53 check`): 3 more with DFlash2, and for each 100,000 positions `KV_SIZE` is below 1,638,400, 2.55 fewer with `KV=fp8` and 1.73 fewer with `KV=fp4`. The four-node example sets 115, which comes to 105 with its `KV_SIZE`. A start from the panel uses its plan's figure instead. |
| `READY_TIMEOUT` | `900` | Seconds a start (`tf-glm53 engine-up`, which the panel's start runs) waits for the server. |

### The control panel

| Setting | Default | Meaning |
|---|---|---|
| `LAUNCHER_HOST` | `0.0.0.0` | The address the panel listens on. It has no key: anyone who reaches it can start, stop and change the engine. `127.0.0.1`: this node only; another machine then opens it through an ssh tunnel (`ssh -L 8010:127.0.0.1:8010 FIRST-NODE`). |
| `LAUNCHER_PORT` | `PORT` + 10 | The panel's port (8010 in the four-node example): `http://FIRST-NODE:LAUNCHER_PORT/panel`. |
| `RESERVE_GB` | `4` | **panel.** The GiB each node keeps for the OS after a start from the panel, 2 to 32. |
| `ATLAS_DIR` | `atlas/` in this directory | Where the saved allocations are kept (`allocs/user/<name>.json`). |
| `ATLAS_GENERIC` | `1` | `1`: a start from the panel runs every layer on the generic grouped kernel, so expert optimisation may move any expert. |
| `LAUNCH_SNAPSHOTS` | `0` | `1`: each node's GPU, memory and log tail written to `snapshots/` before and after each start and stop. |
| `LAUNCH_DROP_CACHES` | `0` | `1`: the page cache dropped on every node before a start (needs `sudo` without a password). |

### More environment for the ranks

| Setting | Default | Meaning |
|---|---|---|
| `EXTRA_ENV` | empty | Environment words (`NAME=value`, separated by spaces, in quotes) given to every rank (the conversations decoded together are `MULTI`, not `TF_GLM53_MULTI` here: `tf-glm53` refuses that). The launcher reads a few of them for itself, among them `TF_PANEL_HTML` (another page to serve) and `TF_PANEL_SETTINGS` (another file for the page's settings). |
| `TF_GLM53_IDLE_PARK_MS` | `1000` | Set in `EXTRA_ENV`. With nothing to do for this many milliseconds, every rank parks: the other ranks sleep on their link to rank 0 instead of waiting inside a collective, every RoCE proxy thread stops, and the GPUs and cores rest. A request, a swap of the experts' bits or the stop wakes them first (about 1 to 2 ms). `0` parks as soon as nothing runs; `off` never parks (the earlier behaviour). One rank alone never parks. A rank that dies while the others are parked stops the engine at the next wake, or earlier through the watcher. `/health` `idle` shows the state. On a ring of four, parked, each node's engine used under 1 % of a core and the GPUs 0 %, where the other ranks' GPUs read 96 % without parking; their SoC read 50 to 54 °C against 64 to 85 °C, with `ENGINE_CORES=os`. |
| `TF_GLM53_IDLE_WAKE_TIMEOUT_S` | `30` | Set in `EXTRA_ENV`. When rank 0 wakes the other ranks, the seconds it waits for each one to say it is ready. Past it the engine stops (exit code 70), as when a rank fails. |
