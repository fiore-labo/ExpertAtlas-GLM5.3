[English](CONFIG.md) | [日本語](CONFIG.ja.md) | 简体中文

# glm53.conf 的设置项

使用 `docker run ... IMAGE up` 时，引擎设置中的项目可以用 `-e 项目=值` 指定（`QUICKSTART.zh-CN.md`）。以下项目会以 `not an option` 停止，请写入 `~/expertatlas/glm53.conf`：`GID_INDEX`、`RUNTIME`、`NCCL_LIB`、`EXPERT_LIVE`、`LOG_LISTEN`、`CONTEXT`、`STORE`、`THINKING_FOLD_KEEP`、`NEED_GB`、`LAUNCHER_HOST`、`ATLAS_DIR`、`LAUNCH_SNAPSHOTS`。`TOPOLOGY`、`SOCKET_IFNAME` 和 `IB_HCA` 由节点检查决定。

下面的默认值是 `tf-glm53` 的默认值。无论节点数多少，`docker run ... IMAGE up` 都根据 `glm53.t4.conf.example` 写入 `glm53.conf`，所以初始值为 `CONTEXT_LENGTH` 1048576、`KV_SIZE` 1049088、`KV` fp4、`NEED_GB` 115，启动时的 KV 大小为 `max`。

`glm53.conf` 保存一条通道（lane）的设置：节点、网络、权重、草稿模型、服务器、内存和控制面板。它是一个普通的 shell 文件，由你手工编写和编辑，放在第一个节点上 `tf-glm53` 所在的目录中。请从 `glm53.t4.conf.example`（四节点）或 `glm53.conf.example`（八节点）开始：复制为 `glm53.conf`，填写节点、网络设备和路径；其余项目在示例中已是实测值。

## 由谁读取，修改何时生效

- `tf-glm53` 每执行一个命令都会读取它。只读取第一个节点上的文件：`tf-glm53 deploy` 将本目录复制到其他节点时不包含 `glm53.conf`，各 rank 在启动时从第一个节点获得设置。
- 正在运行的引擎保持启动时的设置。修改在引擎下一次启动时生效（在控制面板上先“停止”再“启动”；或 `./tf-glm53 down` 后 `./tf-glm53 start`）。
- 控制面板的启动器在启动器自身启动时读取它；引擎停止期间文件有变化时，启动器会自行重新读取（引擎运行期间的编辑在引擎停止后读取）。`LAUNCHER_HOST` 和 `LAUNCHER_PORT` 在启动器重新启动后生效：`./tf-glm53 launcher stop`，然后 `./tf-glm53 up`。期间引擎可以继续运行：启动器会发现它正在运行。
- 除两种情况外，没有其他程序会改写它，两者都在控制面板上。两者都只写改动的行（每个设置一行，值后面的注释保留，没有该行时追加在末尾），并把之前的文件以 `glm53.conf.bak-<时间>` 保留在旁边：
  - 启动设置中权重下的“保存”会写入 `WEIGHTS` 和 `DRAFTER`，前提是每个节点上都有它们所指的内容（专家库的目录以同样的方式写入 `expert-plan/banks.json`，并复制到每个节点），启动器会立即重新读取。
  - 引擎设置界面（从标题栏切换进入）写入需要重启的设置：`NAME`, `CONTEXT_LENGTH`, `KV_SIZE`, `KV`, `MAX_TOKENS`, `THINKING_*`, `MULTI*`, `DRAFTER_KIND`, `MTP_K`, `DRAFT_SAVES`, `DCP`, `MASTER_PORT`, `RESERVE_GB`, `READY_TIMEOUT`, `LAUNCH_DROP_CACHES`, `HOST`, `PORT`, `METRICS_LISTEN`, `LOG_KEEP_MB`, `STRUCTURED`, `LAUNCHER_PORT`, `ROWS`, `SP_PARTS`, `ROCE_PROXY_CORES`, `ENGINE_CORES`, `ATLAS_GENERIC`, `EXTRA_ENV`（引擎的 `TF_GLM53_CANON_CLOSE` 和 `TF_GLM53_CANON_NEAR` 作为其中的词）；在节点检查登记节点之前还包括 `TOPOLOGY`、`SOCKET_IFNAME` 和 `IB_HCA`。每个值先按 `tf-glm53` 的规则检查；`PORT` 只能在引擎停止时修改。更改在发生时写入，从下一次启动起使用（正在运行的引擎保持启动时的设置）；但思考中运行时生效的项（`THINKING_SHARE`、`THINKING_FOLD`、`THINKING_FOLD_AT_CLOSE_SHARE`、`THINKING_FOLD_SHARE`）也会立即交给运行中的引擎，推理界面 [ THINKING ] 中的修改也写入此文件，因此两个界面显示同一个值。“Restart to apply”（引擎运行时）：与“停止”一样先询问是否保留本次运行的使用记录，然后停止，再以相同的专家和保存的设置重新启动。
- 控制面板的节点检查所登记的节点写入单独的文件 `glm53.nodes.conf`（在本文件旁边）：`TOPOLOGY`、`NODES`、`SOCKET_IFNAME`、`IB_HCA`，取确认时对连接方式的判定（之前的文件以 `glm53.nodes.conf.bak-<时间>` 保留）。`tf-glm53` 和启动器在 `glm53.conf` 之后读取它，所以这四项以它为准。没有这个文件时，面板从节点检查开始；有了之后，从启动设置开始。`NODES_CONF=FILE` 读取另一个文件，`NODES_CONF=`（空）则不读取。`tf-glm53 deploy` 不复制它。
- 在启动界面选择的内容（KV 格式、上下文长度、草稿模型、并发推理数、每个节点为操作系统保留的内存）在选择时写入本文件（`KV`、`CONTEXT_LENGTH`、`DRAFTER_KIND`、`MULTI`、`RESERVE_GB`）。启动时写入计划采用的上下文长度，因此下次打开时会选中该值。`KV_SIZE` 只在不够上下文长度时才调高。专家的位宽和计划采用的 KV 只对该次启动有效，启动器通过环境变量（`LAUNCH_*`）把它们交给 `tf-glm53`；它们不是本文件的设置项。
- 页面自身的设置（颜色、圆的排列方式等）保存在 `launch/panel-settings.json` 中。
- `glm53.conf` 不属于发布包的文件：`SHA256SUMS` 中没有它，因此 `tf-glm53 check` 和 `tf-glm53 deploy` 都不会核对它。大幅修改前，请保留一份可用文件的副本。
- 使用另一个设置文件：`tf-glm53 -c FILE 命令`（启动器也一样：`./tf-glm53 -c FILE up`）。在同一组节点上运行两条通道时，`PORT`、`MASTER_PORT` 和 `LAUNCHER_PORT` 需要各不相同。

## 设置项

“默认值”是某项缺失或为空时 `tf-glm53` 使用的值；“（必需）”表示没有默认值。标有 **面板** 的项也可以在启动界面上选择，选择后写入这里；不经启动界面的启动（`tf-glm53 start`、`engine-up`）使用文件中的值。

### 节点与网络

| 设置项 | 默认值 | 含义 |
|---|---|---|
| `NODES` | （必需） | 按 rank 顺序、以 RDMA 网络地址列出的节点，写成 shell 数组：`NODES=(192.0.2.11 192.0.2.12 192.0.2.13 192.0.2.14)`。四个或八个（两个和三个正在测试中）。在第一个节点上运行 `tf-glm53`。每个节点都能通过无密码 ssh 登录其他节点。控制面板的节点检查把节点登记到 `glm53.nodes.conf` 而不是这里，它优先于本项和后面三项。 |
| `SOCKET_IFNAME` | （必需） | 每个节点上该网络的网络接口（`ip -br addr`），例如 `enp1s0f1np1`。 |
| `IB_HCA` | （必需） | 该接口的 RDMA 设备（`ibv_devices`），例如 `rocep1s0f1`。 |
| `TOPOLOGY` | `full` | 节点的连接方式。`full`：交换机，或一根线直连。`ring4`：不经交换机连成环的 4 个节点，NODES 按环的顺序（RING.md）。`auto`：每次 `check` 和 `./tf-glm53 up` 时，根据各节点的 RDMA 设备和 LLDP 邻居判定（`tf-glm53 topology` 显示判定结果）：环取 `ring4`，并把 NODES 排成环的顺序（rank 0 不变）；交换机或一根线取 `full`。此时 `IB_HCA` 和 `SOCKET_IFNAME` 使用判定的结果，可以不写（`IB_HCA` 只用于选择交换机上的哪一半）。两者都不是的连接方式会被拒绝，并指出要修正的地方，不会自动退回到别的方式。**已知缺陷：** 使用 `TOPOLOGY=auto`、不写 `IB_HCA` 和 `SOCKET_IFNAME` 且尚无判定结果时，如果首次启动使用 `./tf-glm53 start` 或 `engine-up`，会以 `NG: auto RoCE-v2-GID(gid=)` 停止：这两个命令不做判定，`check` 判定但不保存，只有 `./tf-glm53 up` 保存判定结果。请先运行 `./tf-glm53 up`。写了这两项就不会出现这个 `NG`，但没有判定结果时会按 `full` 启动，这在环上无法使用。`docker run ... IMAGE up` 以及 `TOPOLOGY=full`、`ring4` 不受影响。 |
| `GID_INDEX` | 空 | RoCE v2 的 GID 索引。为空时在每个节点上自动查找（带 IPv4 地址的最后一个 RoCE v2 条目）。 |
| `MASTER_PORT` | `29551` | 第一个节点用于 NCCL rendezvous 的端口。 |

### rank 的运行方式

| 设置项 | 默认值 | 含义 |
|---|---|---|
| `RUNTIME` | `host` | `host`：从本目录运行，每个节点需要 cuBLAS 13、libibverbs 和 NCCL。`docker`：每个 rank 在 `IMAGE` 的容器中运行（RUNTIME.zh-CN.md“在容器中运行”）。 |
| `IMAGE` | 空 | `RUNTIME=docker` 时：本发布的镜像。 |
| `NCCL_LIB` | `~/nccl-2.30.7/nvidia/nccl/lib/libnccl.so.2` | 每个节点上此路径下的 NCCL 2.30.7（仅 `RUNTIME=host`；RUNTIME.zh-CN.md“首次启动之前”）。在交换机上 NVIDIA 的 wheel 可以运行，但推荐使用带 AArch64 fence 的构建；环必须使用该构建（RING.zh-CN.md）。 |

### 权重与草稿模型

| 设置项 | 默认值 | 含义 |
|---|---|---|
| `WEIGHTS` | （必需） | Base 权重（`GLM-5.3-EXL3-Base`）的目录，所有节点上路径相同（四节点和八节点都一样）。也可以用完整检查点（`GLM-5.3-EXL3-4x-DGX-Sparks`、`GLM-5.3-EXL3-8x-DGX-Sparks`）代替。可以在启动设置中指定，专家库的目录（`expert-plan/banks.json`）也一样；都在确认每个节点上都有之后才写入。 |
| `DRAFTER_KIND` | `mtp` | **面板**。由什么起草随后由模型核对的 token：`mtp`（`WEIGHTS` 中的 MTP 层）、`dflash2`（`DRAFTER` 中的模型）或 `none`。`./tf-glm53 -d KIND start`（或 `engine-up`）可为一次启动选择另一种。 |
| `MTP_K` | `3` | MTP 每轮的草稿数，1 至 7。 |
| `DRAFTER` | 空 | DFlash2 模型的目录，所有节点上路径相同。`DRAFTER_KIND=dflash2` 需要它；只有在设置了它且每个节点都有该模型时，启动界面才提供 DFlash2。不包含在发布中（RUNTIME.zh-CN.md“使用 DFlash2 的步骤”）。可以在启动设置中指定。 |
| `DRAFT_SAVES` | `8` | 使用 DFlash2 时：在请求之间保留草稿模型状态的对话数。 |

### 服务器

| 设置项 | 默认值 | 含义 |
|---|---|---|
| `NAME` | `glm-5.3` | API 中的模型名称：`/v1/models` 列出它，回复中带有它（请求可以指定任何模型名）。 |
| `HOST` | `0.0.0.0` | API 监听的地址。`0.0.0.0`：所有地址。API 没有密钥，能访问到它的人都可以使用：请将节点留在封闭网络中，或在前面放置 TLS 和访问控制。`127.0.0.1`：仅本节点；其他机器通过 ssh 隧道访问。 |
| `PORT` | `8000` | API 的端口。 |
| `METRICS_LISTEN` | 空 | `HOST:PORT`：第一个节点也在该地址上以只读方式响应 `GET /metrics`（Prometheus 文本）、`GET /health` 和 `GET /v1/experts/usage`（专家的使用次数），供另一台主机上的监控使用。为空时关闭。 |
| `EXPERT_LIVE` | `1` | `1`：第一个节点为控制面板统计每个提示行和每个解码 token 选中的专家（实时显示、使用记录和专家优化）。`0`：不统计；面板仍显示正在运行的引擎和“停止”，但不显示选择，也没有专家优化。从面板启动时始终统计。 |
| `LOG_LISTEN` | 空 | `HOST:PORT`：每个节点的日志保管进程还在此以只读、无密钥的方式响应 `GET /streams` 和 `GET /logs?stream=S&after=N`，供另一台主机上保存日志的监控程序使用。为空时，日志只在每个节点的内存中（`tf-glm53 logs`）。修改在引擎下一次启动时生效：届时重新启动保管进程，之前保存的行会丢失。 |
| `LOG_KEEP_MB` | `8` | 每条日志（各 rank、监视进程、启动器）在内存中保留的 MB，较旧的行被丢弃。不向磁盘写入任何内容。修改的生效方式与 `LOG_LISTEN` 相同。 |
| `STRUCTURED` | `1` | `1`：结构化输出，把回复的答案限定为 JSON 或 JSON Schema（`response_format` 等；见 FEATURES.zh-CN.md 的“结构化输出”）。启动时 rank 0 加载 `lib/libllguidance.so`，无法加载时引擎在没有结构化输出的情况下启动。`0`：关闭，请求它的请求得到 400。修改在下一次启动时生效。 |

### 上下文、缓存与 KV

| 设置项 | 默认值 | 含义 |
|---|---|---|
| `CONTEXT_LENGTH` | `819200` | **面板**。一个请求可达到的位置数（提示 + 回复）：65,536 至 1,048,576（模型的上限），以 16,384 为步长。 |
| `KV_SIZE` | `1638400` | **面板**。整个 KV 的位置数，由所有对话共享；至少为 `CONTEXT_LENGTH` + 512。在面板中设置后（这里或启动界面的 KV 大小），从面板启动时使用此值；在启动界面输入每个节点保留的内存，KV 又会使用计划剩余的全部空间。仅凭 `glm53.conf` 的启动也使用此值。 |
| `CONTEXT`、`STORE` | | `CONTEXT_LENGTH` 和 `KV_SIZE` 之前的名称。没有新名称时仍会读取：`CONTEXT_LENGTH` = `CONTEXT`，`KV_SIZE` = `CONTEXT` + `STORE`（未设置 `STORE` 时为 819,200）。在引擎设置界面保存时写入新名称并删除这些行。 |
| `KV` | `fp8` | **面板**。KV 缓存的格式：`fp8`，或 `fp4`（每个位置占用更少内存，精度略低；T4.zh-CN.md）。 |
| `DCP` | `2` | 解码上下文并行：`2` 把 KV 分到一对 rank 上。 |
| `MULTI` | `0` | 每轮在一个窗口中一起解码的对话数上限（1 .. 4）；`0` 表示请求逐个处理。可与 MTP 层、DFlash2（`DRAFT_SAVES` 至少为 1）或不使用草稿模型一起使用；暂不支持 3 个节点（启动前即拒绝）。专家优化与 `0` 时相同：更改会等到没有正在运行或等待的请求。 |
| `MULTI_WAIT` | `600` | 设置了 `MULTI` 时：请求等待会话缓存在进行中的对话之外腾出空间的秒数；超过后以繁忙拒绝，客户端可重试：HTTP 503（Messages 端点为 `overloaded_error`）；已开始的流中则是类型为 `overloaded_error`（Messages）或 `server_error`、代码 `overloaded`（Chat Completions）的错误事件，或代码为 `rate_limit_exceeded` 的失败响应（Responses）。等待时间从请求第一次因没有空间而未能进入时算起，并在 decode 轮次与 prompt 分块之间判断，因此拒绝可能晚一轮。等待期间被取消的请求以取消结束。`0` 表示不限。 |
| `MULTI_DECODE_SHARE` | `30` | 设置了 `MULTI` 时：在正在 decode 的对话旁有 prompt 等待或正在计算期间，GPU 时间中分给 decode 的百分比。`0` 让 prompt 优先（prefill 中途不插入 decode 轮次，排队的请求连续接收）；`100` 让 decode 优先（只要有对话在 decode，prompt 就等待，已在运行的 prefill 也暂停到这些对话结束；短 prompt 仍可通过，见 `MULTI_SHORT_PROMPT`）；中间的值由引擎测量两者的时间并保持该比例。只有一种工作时，它使用整个 GPU。这是启动时的值：运行中可用 `POST /v1/schedule {"decode_share": N}`（需 API 密钥，或来自首节点的客户端）修改，正在运行的 prefill 也会在几层之内生效；`GET /v1/schedule` 与 `/health`（`multi.decode_share`、`multi.measured`）可查看。因比例而等待不受 `MULTI_WAIT` 限制（`MULTI_WAIT` 是等待空间的时间）。 |
| `MULTI_SHORT_PROMPT` | `8192` | 设置了 `MULTI` 时：需要计算的 token（会话缓存可复用部分之后的部分，因此长对话的下一轮只计新增部分）不超过此值的 prompt 视为短 prompt：它排在等待中的较长 prompt 之前，不受 `MULTI_DECODE_SHARE` 阻挡，其 prefill 不插入 decode 轮次一直运行到结束（无论比例如何，此期间 GPU 归该 prompt；正在 decode 的对话在此期间等待；四节点上 8,192 个 token 约 7 秒）。除非 `MULTI_SHORT_PREEMPT=1`，它会等待已在运行的 prefill 结束。`0` 表示没有短 prompt（队列保持原顺序）。这是启动时的值：运行中可用 `POST /v1/schedule {"short_prompt": N}` 修改，从下一次接收起生效；`GET /v1/schedule` 与 `/health`（`multi.short_prompt`）可查看。 |
| `MULTI_SHORT_PREEMPT` | `0` | 设置了 `MULTI` 与 `MULTI_SHORT_PROMPT` 时：`1` 让已在运行的较长 prefill 也让给等待中的短 prompt，在下一个为 decode 暂停之处（若没有其他对话在 decode，则在下一个分块）。已完成的分块留在缓存中，流保持打开，之后从那里继续；在暂停处放弃的分块需重新计算（最多一个分块：四节点、比例 30 时长 prompt 约慢 14 秒），每次继续后要先完成一个分块才会再次让出。`0`（默认）：短 prompt 等待正在运行的 prefill 结束。运行中可用 `POST /v1/schedule {"short_preempt": true}` 修改；`GET /v1/schedule` 与 `/health`（`multi.short_preempt`）可查看。 |
| `ROWS` | `8192` | 提示分块的行数：256 至 65,536，步长 256。增大它会占用更多内存（每 1,024 行每个节点约 0.41 GiB，在四个节点上测得）。启动计划会从 KV 中扣除这部分，因此 KV 会变小。 |
| `SP_PARTS` | `4` | 一个提示分块被拆分成的序列并行部分数：2 至 4。 |
| `ROCE_PROXY_CORES` | `a725` | RoCE 代理线程（每个 rank 两个：节点间的和 DCP 对的）轮询所在的核。它们在 decode 期间一直自旋。`a725` 固定在 GB10 的 Cortex-A725 核 0 和 10（每个簇一个）；`os` 交给调度器（常落在 X925 上）。四节点上 A725 比调度器的选择使 SoC 低 5-16 °C，decode 速度相同。没有该类核的 CPU 上交给调度器。引擎日志会写出每个代理在哪里轮询；`/health` 的 `roce` 显示每个 rank 的（rank 0 的为当前值）。 |
| `ENGINE_CORES` | `os` | 所有 rank 运行所用的核。`os`：交给调度器，即节点的所有核。`a725`：仅每个节点的 Cortex-A725 核。`a725` 不能与 `ROCE_PROXY_CORES=x925` 同时使用，`tf-glm53` 会拒绝。仅用 A725 运行时的速度尚未测量。 |
| `MAX_TOKENS_SHARE` | `70` | 请求未指定 max_tokens 时回复的 max_tokens，以 `CONTEXT_LENGTH` 的百分比（1 至 100）指定。客户端的 `max_tokens` 优先。 |
| `MAX_TOKENS` | （空） | 以 token 数指定同一项。有值时优先于 `MAX_TOKENS_SHARE`；在仪表板中保存 `MAX_TOKENS_SHARE` 时会清空。既无中途整理也无预算将尽时的整理时，请求超过上下文剩余量会被拒绝（400）。开启中途整理时，引擎会让间隔适应剩余空间；请求自行指定 `thinking_fold_window` 且放不下时会被拒绝（400）。 |
| `THINKING_SHARE` | `90` | **运行中也可改**。 回复的 max_tokens 中思考可以使用的百分比（但最多为 prompt 在 `CONTEXT_LENGTH` 中剩余部分的该百分比；`0`：不限），其余部分属于回答。思考达到该 token 数后，在下一个换行处（最多 256 个 token 之后），引擎写入一句收尾的话和 `</think>`，其余部分用于回答。使用 `THINKING_FOLD` 时不收尾而继续：思考填满上下文的该百分比时折叠，预算为 max_tokens 减去回答部分。请求的 `thinking_budget`（Anthropic：`thinking.budget_tokens`）优先。 |
| `THINKING_BUDGET` | `0` | 一次回复中模型可用于思考的 token 数。使用 `THINKING_FOLD` 时，折叠的思考会离开上下文，因此可大于 `CONTEXT_LENGTH`（虚拟长度）。达到预算时，引擎在换行处结束思考并进入回答。`0`：由 `THINKING_SHARE` 决定。请求中的 `thinking_budget` 优先。 |
| `THINKING_NOTICE` | `0` | 思考每前进这么多 token（`256` 起；`0`：不使用），在下一个换行处，引擎在思考中写入一行，告诉模型已用的思考 token 数和剩余数（预算的；没有预算时为 max_tokens 的）。该行是回复思考（`reasoning_content`）的一部分。每次需要一次短的重新接纳：缓存恢复除该行以外的全部。 |
| `THINKING_FOLD` | `off` | **运行中也可改**。 在模型的上下文中折叠过长的思考，使 KV 不再持续增长（客户端仍收到全部思考）。`recent` 保留思考的前 100 个 token 和最后 `THINKING_FOLD_KEEP` 个。`checkpoint` 先让模型写一个检查点（目前的结论、失败的方法、尚未确定的事项、下一步要检查的内容，以 `END CHECKPOINT` 结束，最多 `THINKING_FOLD_CAP` 个 token），由它取代之前的思考；模型只继续思考尚未确定的事项。`branch` 将回复暂放一边，在回复旁的回答中（结束思考、用户轮次、无思考的回答）让模型总结自上次总结以来的思考；回答必定以 EOS 结束（在思考中写的检查点很少写出 `END CHECKPOINT`）。摘要以 "(My reasoning grew long, so I summarized it once ...)" 为标题累积在思考中，客户端也会在思考中收到。当思考填满 prompt 剩余上下文的 `THINKING_SHARE`（`THINKING_SHARE=0` 时为全部）减去摘要所需部分时折叠，之后可超出上下文继续思考。请求也可以用 `thinking_fold`、`thinking_fold_window`、`thinking_fold_max`、`thinking_fold_keep` 指定。 |
| `THINKING_FOLD_SHARE` | `100` | **运行中也可改**。使用 `THINKING_FOLD` 时：达到每个请求的 max_tokens 的该百分比时，最迟在此折叠（至少 1024 token，10 至小于 `THINKING_SHARE`）；`100`：到上限前不整理。请求中的 `thinking_fold_window` 优先。 |
| `THINKING_FOLD_MAX` | `0` | **运行中也可改**。使用 `THINKING_FOLD` 时：一个回复最多折叠的次数（`0`：不限）。用完之后，或已累积的摘要再加一个将超过上下文中思考的百分比（摘要在其中累积）时，不再折叠：思考继续直到填满该百分比，然后按预算用尽的方式结束（有 `THINKING_FOLD_AT_CLOSE_SHARE` 时以摘要代替收尾的话）。思考预算的剩余容纳不下中途摘要和思考完成时摘要时，在折叠处结束思考。请求中的 `thinking_fold_max` 优先。 |
| `THINKING_FOLD_KEEP` | `0` | `THINKING_FOLD=recent` 时：保留的最后思考 token 数（`0`：窗口的一半）。 |
| `THINKING_FOLD_AT_CLOSE_SHARE` | `0` | **运行中也可改**。 思考完成时的摘要：模型结束思考时，若上下文中的思考（自上次折叠以来）不少于思考上限的该百分比（`0`：不做，1 至 100；请求的 `thinking_fold_at_close` 以 token 数指定并优先），则在回答前先让模型写一个记录推理所得的检查点（已确认的事实与结论、带条件的中间结果、被否定的思路及其理由、已纠正的错误、尚未确认的假设、回答的计划、写作时需注意的要点；反复试错与重复表述不保留；请求未指定 `thinking_fold_cap` 时最多 `THINKING_FOLD_CAP` 个 token），用它取代思考，再据此写出回答；之后的回合也收到该摘要。当思考预算由引擎结束思考时，也以同样的摘要代替预算的收尾句（此时还会写出思考停在何处，以及每个未能确定的问题当前的最佳答案、把握程度和回答中的处理方式）。仅当摘要及与其等长的回答仍能放入回复的 `max_tokens` 时才会要求摘要。`0`：不使用。`THINKING_FOLD=off` 时也可用。请求中的 `thinking_fold_at_close` 优先。 |
| `THINKING_FOLD_AT_BUDGET` | `1` | **运行中也可改**。在思考预算接近用完时整理思考：用改写替换上下文中的思考，之后由模型决定是否结束。`0`：关闭。请求的 `thinking_fold_at_budget` 优先。 |
| `THINKING_FOLD_MARGIN` | `10` | **运行中也可改**。在思考预算用完之前这个百分比处开始整理（0 至 50）。 |
| `THINKING_FOLD_PROMPT` | org 文本 | **运行中也可改**。该整理的指令。留空则恢复默认。 |
| `THINKING_CLOSE_PROMPT` | org 文本 | **运行中也可改**。思考结束时摘要的指令。留空则恢复默认。 |
| `TF_GLM53_BYTE_RESUME` | `on` | 模型的切分方式可能与分词器不同，这时下一轮的 prompt 以不同的 id 持有相同的字节，缓存就停在那里。`on`：从第一个不同的 id 起，把 prompt 与保存的对话按字节比较，字节一致之处使用它们的 id，缓存从那之后继续。文本不变；回复可能与不用缓存计算的结果略有不同。`off`：缓存停在那里。`log`：与 `off` 相同，并比较和记录（`tf-glm53 logs 0` 中的 `byteresume:`）。 |
| `THINKING_FOLD_CAP_SHARE` | `70` | 一次摘要的上限，以被整理的思考的百分比指定（10 至 100，至少 128 个 token）。适用于 `THINKING_FOLD=checkpoint` 的检查点和 `THINKING_FOLD_AT_CLOSE_SHARE` 的摘要。请求中的 `thinking_fold_cap`（token 数，128 至 16384）优先。 |

### 内存与启动

| 设置项 | 默认值 | 含义 |
|---|---|---|
| `NEED_GB` | `112` | 启动前每个节点必须可用的 GiB（`tf-glm53 check`）：使用 DFlash2 时多 3；`KV_SIZE` 每比 1,638,400 少 100,000 个位置，`KV=fp8` 时少 2.55，`KV=fp4` 时少 1.73。四节点示例设为 115，按其 `KV_SIZE` 计算后为 105。从面板启动时改用计划中的数值。 |
| `READY_TIMEOUT` | `900` | 启动（`tf-glm53 engine-up`；从面板启动时也执行它）等待服务器的秒数。 |

### 控制面板

| 设置项 | 默认值 | 含义 |
|---|---|---|
| `LAUNCHER_HOST` | `0.0.0.0` | 面板监听的地址。面板没有密钥：能访问到它的人都可以启动、停止和更改引擎。`127.0.0.1`：仅本节点；其他机器通过 ssh 隧道打开（`ssh -L 8010:127.0.0.1:8010 FIRST-NODE`）。 |
| `LAUNCHER_PORT` | `PORT` + 10 | 面板的端口（四节点示例中为 8010）：`http://FIRST-NODE:LAUNCHER_PORT/panel`。 |
| `RESERVE_GB` | `4` | **面板**。从面板启动后每个节点为操作系统保留的 GiB，2 至 32。 |
| `ATLAS_DIR` | 本目录下的 `atlas/` | 已保存配置的位置（`allocs/user/<名称>.json`）。 |
| `ATLAS_GENERIC` | `1` | `1`：从面板启动时所有层都使用通用的 grouped 内核，使专家优化可以移动任何专家。 |
| `LAUNCH_SNAPSHOTS` | `0` | `1`：每次启动和停止前后，把每个节点的 GPU、内存和日志末尾写入 `snapshots/`。 |
| `LAUNCH_DROP_CACHES` | `0` | `1`：启动前在所有节点上丢弃页缓存（需要无密码的 `sudo`）。 |

### 传给 rank 的更多环境变量

| 设置项 | 默认值 | 含义 |
|---|---|---|
| `EXTRA_ENV` | 空 | 传给每个 rank 的环境变量（`名称=值`，以空格分隔，放在引号中），一起解码的对话数用 `MULTI` 设置，不要在这里写 `TF_GLM53_MULTI`（`tf-glm53` 会拒绝）。启动器也会自己读取其中几项，例如 `TF_PANEL_HTML`（提供另一个页面）和 `TF_PANEL_SETTINGS`（页面设置的另一个文件）。 |
| `TF_GLM53_IDLE_PARK_MS` | `1000` | 写在 `EXTRA_ENV` 中。连续这么多毫秒无事可做时，所有 rank 进入休眠：其他 rank 不在集合通信中等待，而是在与 rank 0 的连接上休眠，所有 RoCE 代理线程停止，GPU 和 CPU 核心得以休息。请求、专家位宽的切换或停止到来时，先唤醒所有 rank（约 1 至 2 ms）。`0` 表示一旦无事可做就立即休眠；`off` 表示从不休眠（以前的行为）。只有一个 rank 时从不休眠。休眠期间若有 rank 退出，引擎会在下次唤醒时停止，或更早由监视进程停止。状态见 `/health` 的 `idle`。在四节点环上测得：休眠时每个节点的引擎占用不到一个核心的 1 %，GPU 为 0 %；不休眠时其他 rank 的 GPU 为 96 %。使用 `ENGINE_CORES=os` 时，这些节点的 SoC 温度从 64 至 85 °C 降到 50 至 54 °C。 |
| `TF_GLM53_IDLE_WAKE_TIMEOUT_S` | `30` | 写在 `EXTRA_ENV` 中。rank 0 唤醒其他 rank 时，等待每个 rank 回报就绪的秒数。超时则引擎停止（退出码 70），与 rank 失败时相同。 |
