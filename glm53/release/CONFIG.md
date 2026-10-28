English | [日本語](CONFIG.ja.md) | [简体中文](CONFIG.zh-CN.md)

# glm53.conf: the settings

`glm53.conf` holds a lane's settings: the nodes, the network, the weights, the drafter, the server, the memory and
the control panel. It is a plain shell file that you write and edit by hand, next to `tf-glm53` on the first node.
Start from `glm53.t4.conf.example` (four nodes) or `glm53.conf.example` (eight nodes): copy it to `glm53.conf` and
fill in the nodes, the network devices and the paths. The examples carry the measured values for the rest.

## Who reads it, and when a change takes effect

- `tf-glm53` reads it at every command. Only the first node's file is read: `tf-glm53 deploy` copies this directory
  to the other nodes without `glm53.conf`, and each rank gets its settings from the first node when it starts.
- A running engine keeps the settings it started with. A change reaches the engine at its next start
  (`./tf-glm53 down`, then `./tf-glm53 up`; or Stop, then Start on the control panel).
- The control panel's launcher reads it when the launcher starts. After an edit, start the launcher again:
  `./tf-glm53 launcher stop`, then `./tf-glm53 launcher`. The engine may keep running meanwhile: the launcher finds
  it running.
- Nothing else writes it, with one exception: Save on the launch screen's list of nodes writes `NODES` (the
  previous file stays beside it as `glm53.conf.bak-<time>`), and the launcher reads `NODES` again at once.
- What the launch screen chooses for a start (the experts' widths, the KV format, the context, the store, the
  drafter, what each node keeps for the OS) holds for that start only and is not written into `glm53.conf`. The
  launcher hands those choices to `tf-glm53` in its environment (`LAUNCH_*`); they are not settings of this file.
- The page's own settings (colours, the circle's order and the rest) are kept in `launch/panel-settings.json`.
- `glm53.conf` is not one of the release's files: `SHA256SUMS` does not list it, so `tf-glm53 check` and
  `tf-glm53 deploy` never compare it. Keep a copy of a working file before a large change.
- Another settings file: `tf-glm53 -c FILE COMMAND` (the launcher too: `./tf-glm53 -c FILE launcher`). Two lanes on
  the same nodes need their own `PORT`, `MASTER_PORT` and `LAUNCHER_PORT`.

## The settings

"Default" is what `tf-glm53` uses when a setting is missing or empty; "(needed)" has none. Settings marked
**panel** can be chosen again on the launch screen for one start; the file's value is what `tf-glm53 up` uses.

### Nodes and network

| Setting | Default | Meaning |
|---|---|---|
| `NODES` | (needed) | The nodes in rank order, by their address on the RDMA network, as a shell array: `NODES=(192.0.2.11 192.0.2.12 192.0.2.13 192.0.2.14)`. Four or eight. Run `tf-glm53` on the first. Every node reaches the others over ssh without a password. The launch screen can change the workers (all but the first) and save them here. |
| `SOCKET_IFNAME` | (needed) | The network interface of that network on every node (`ip -br addr`), for example `enp1s0f1np1`. |
| `IB_HCA` | (needed) | The RDMA device of that interface (`ibv_devices`), for example `rocep1s0f1`. |
| `GID_INDEX` | empty | The RoCE v2 GID index. Empty: found on each node (the last RoCE v2 entry with an IPv4 address). |
| `MASTER_PORT` | `29551` | The first node's port for the NCCL rendezvous. |

### Where the ranks run

| Setting | Default | Meaning |
|---|---|---|
| `RUNTIME` | `host` | `host`: from this directory, with cuBLAS 13, libibverbs and NCCL on every node. `docker`: each rank in a container of `IMAGE` (RUNTIME.md, "Running in containers"). |
| `IMAGE` | empty | With `RUNTIME=docker`: this release's image. |
| `NCCL_LIB` | `~/nccl-2.30.7/nvidia/nccl/lib/libnccl.so.2` | NCCL 2.30.7 at this path on every node (`RUNTIME=host` only; RUNTIME.md, "Before the first start"). |

### Weights and drafter

| Setting | Default | Meaning |
|---|---|---|
| `WEIGHTS` | (needed) | The checkpoint's directory, the same path on every node: `GLM-5.3-EXL3-4x-DGX-Sparks` for four nodes, `GLM-5.3-EXL3-8x-DGX-Sparks` for eight. |
| `DRAFTER_KIND` | `mtp` | **panel.** What drafts the tokens the model then checks: `mtp` (the checkpoint's MTP layer), `dflash2` (the model in `DRAFTER`) or `none`. `./tf-glm53 -d KIND up` picks another for one start. |
| `MTP_K` | `3` | Drafts a round with MTP, 1 to 7. |
| `DRAFTER` | empty | The DFlash2 model's directory, the same path on every node. `DRAFTER_KIND=dflash2` needs it, and the launch screen offers DFlash2 only when it is set and every node has the model. Not part of the release (RUNTIME.md, "Using DFlash2"). |
| `DRAFT_SAVES` | `8` | With `CACHE=pool` and DFlash2: the conversations whose drafter state is kept between requests. |

### The server

| Setting | Default | Meaning |
|---|---|---|
| `NAME` | `glm-5.3` | The model's name in the API: `/v1/models` lists it and replies carry it (a request may name any model). |
| `HOST` | `0.0.0.0` | The address the API listens on. `0.0.0.0`: every address. The API has no key, so anyone who reaches it can use it: keep the nodes on a closed network, or put TLS and access control in front. `127.0.0.1`: this node only; another machine then reaches it through an ssh tunnel. |
| `PORT` | `8000` | The API's port. |
| `METRICS_LISTEN` | empty | `HOST:PORT` where the first node also answers `GET /metrics` (Prometheus text) and `GET /health`, read-only, for a monitor on another host. Empty: off. |
| `EXPERT_LIVE` | `1` | `1`: the first node counts the experts each prompt row and each decoded token picks, for the control panel (its live view, the usage records and expert optimisation). `0`: not counted; the panel then shows the running engine with Stop, but no picks and no optimisation. A start from the panel always counts. |

### Context, cache and KV

| Setting | Default | Meaning |
|---|---|---|
| `CONTEXT` | `819200` | **panel.** The positions a request may reach (prompt + reply). |
| `STORE` | `819200` | **panel.** The positions kept beyond `CONTEXT` for earlier conversations; at least 512 with `CACHE=pool`. |
| `CACHE` | `pool` | `pool`: `CONTEXT` + `STORE` positions in blocks of 256 that every conversation shares. `slots`: one store of `STORE` positions for each conversation. |
| `KV` | `fp8` | **panel.** The KV cache's format: `fp8`, or `fp4` (less memory a position, a little less accurate; T4.md). |
| `DCP` | `2` | Decode context parallelism: `2` splits the KV between the ranks of a pair, `1` keeps all of it on each rank. `CACHE=pool` takes 1 or 2. |
| `ROWS` | `8192` | The rows of a prompt chunk (`16384` in the four-node example). |
| `SP_PARTS` | `4` | The sequence-parallel parts a prompt chunk is split into. |

### Memory and start

| Setting | Default | Meaning |
|---|---|---|
| `NEED_GB` | `112` | The GiB a node must have available before a start (`tf-glm53 check`): 3 more with DFlash2, and 2.7 fewer for each 100,000 positions of `STORE` below 819,200. The four-node example sets 131, which comes to 112 with its `STORE`. A start from the panel uses its plan's figure instead. |
| `READY_TIMEOUT` | `900` | Seconds `tf-glm53 up` waits for the server. |

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
| `EXTRA_ENV` | empty | Environment words (`NAME=value`, separated by spaces, in quotes) given to every rank, for example `EXTRA_ENV="TF_GLM53_MULTI=4"` (up to four conversations decoded together; `glm53.conf.example`). The launcher reads a few of them for itself, among them `TF_PANEL_HTML` (another page to serve) and `TF_PANEL_SETTINGS` (another file for the page's settings). |
