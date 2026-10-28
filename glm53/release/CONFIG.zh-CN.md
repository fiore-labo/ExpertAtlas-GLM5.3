[English](CONFIG.md) | [日本語](CONFIG.ja.md) | 简体中文

# glm53.conf 的设置项

`glm53.conf` 保存一条通道（lane）的设置：节点、网络、权重、草稿模型、服务器、内存和控制面板。它是一个普通的 shell 文件，由你手工编写和编辑，放在第一个节点上 `tf-glm53` 所在的目录中。请从 `glm53.t4.conf.example`（四节点）或 `glm53.conf.example`（八节点）开始：复制为 `glm53.conf`，填写节点、网络设备和路径；其余项目在示例中已是实测值。

## 由谁读取，修改何时生效

- `tf-glm53` 每执行一个命令都会读取它。只读取第一个节点上的文件：`tf-glm53 deploy` 将本目录复制到其他节点时不包含 `glm53.conf`，各 rank 在启动时从第一个节点获得设置。
- 正在运行的引擎保持启动时的设置。修改在引擎下一次启动时生效（`./tf-glm53 down` 后 `./tf-glm53 up`；或在控制面板上先“停止”再“启动”）。
- 控制面板的启动器在启动器自身启动时读取它。编辑后请重新启动启动器：`./tf-glm53 launcher stop`，然后 `./tf-glm53 launcher`。期间引擎可以继续运行：启动器会发现它正在运行。
- 除一种情况外，没有其他程序会改写它：在启动界面的节点列表中“保存”会写入 `NODES`（之前的文件以 `glm53.conf.bak-<时间>` 保留在旁边），启动器会立即重新读取 `NODES`。
- 启动界面为一次启动所做的选择（专家的位宽、KV 格式、上下文、存储、草稿模型、每个节点为操作系统保留的内存）只对该次启动有效，不会写入 `glm53.conf`。启动器通过环境变量（`LAUNCH_*`）把这些选择交给 `tf-glm53`；它们不是本文件的设置项。
- 页面自身的设置（颜色、圆的排列方式等）保存在 `launch/panel-settings.json` 中。
- `glm53.conf` 不属于发布包的文件：`SHA256SUMS` 中没有它，因此 `tf-glm53 check` 和 `tf-glm53 deploy` 都不会核对它。大幅修改前，请保留一份可用文件的副本。
- 使用另一个设置文件：`tf-glm53 -c FILE 命令`（启动器也一样：`./tf-glm53 -c FILE launcher`）。在同一组节点上运行两条通道时，`PORT`、`MASTER_PORT` 和 `LAUNCHER_PORT` 需要各不相同。

## 设置项

“默认值”是某项缺失或为空时 `tf-glm53` 使用的值；“（必需）”表示没有默认值。标有 **面板** 的项可以在启动界面上为一次启动重新选择；文件中的值供 `tf-glm53 up` 使用。

### 节点与网络

| 设置项 | 默认值 | 含义 |
|---|---|---|
| `NODES` | （必需） | 按 rank 顺序、以 RDMA 网络地址列出的节点，写成 shell 数组：`NODES=(192.0.2.11 192.0.2.12 192.0.2.13 192.0.2.14)`。四个或八个。在第一个节点上运行 `tf-glm53`。每个节点都能通过无密码 ssh 登录其他节点。启动界面可以更改 worker（第一个节点以外）并保存到这里。 |
| `SOCKET_IFNAME` | （必需） | 每个节点上该网络的网络接口（`ip -br addr`），例如 `enp1s0f1np1`。 |
| `IB_HCA` | （必需） | 该接口的 RDMA 设备（`ibv_devices`），例如 `rocep1s0f1`。 |
| `GID_INDEX` | 空 | RoCE v2 的 GID 索引。为空时在每个节点上自动查找（带 IPv4 地址的最后一个 RoCE v2 条目）。 |
| `MASTER_PORT` | `29551` | 第一个节点用于 NCCL rendezvous 的端口。 |

### rank 的运行方式

| 设置项 | 默认值 | 含义 |
|---|---|---|
| `RUNTIME` | `host` | `host`：从本目录运行，每个节点需要 cuBLAS 13、libibverbs 和 NCCL。`docker`：每个 rank 在 `IMAGE` 的容器中运行（RUNTIME.zh-CN.md“在容器中运行”）。 |
| `IMAGE` | 空 | `RUNTIME=docker` 时：本发布的镜像。 |
| `NCCL_LIB` | `~/nccl-2.30.7/nvidia/nccl/lib/libnccl.so.2` | 每个节点上此路径下的 NCCL 2.30.7（仅 `RUNTIME=host`；RUNTIME.zh-CN.md“首次启动之前”）。 |

### 权重与草稿模型

| 设置项 | 默认值 | 含义 |
|---|---|---|
| `WEIGHTS` | （必需） | 检查点目录，所有节点上路径相同：四节点用 `GLM-5.3-EXL3-4x-DGX-Sparks`，八节点用 `GLM-5.3-EXL3-8x-DGX-Sparks`。 |
| `DRAFTER_KIND` | `mtp` | **面板**。由什么起草随后由模型核对的 token：`mtp`（检查点的 MTP 层）、`dflash2`（`DRAFTER` 中的模型）或 `none`。`./tf-glm53 -d KIND up` 可为一次启动选择另一种。 |
| `MTP_K` | `3` | MTP 每轮的草稿数，1 至 7。 |
| `DRAFTER` | 空 | DFlash2 模型的目录，所有节点上路径相同。`DRAFTER_KIND=dflash2` 需要它；只有在设置了它且每个节点都有该模型时，启动界面才提供 DFlash2。不包含在发布中（RUNTIME.zh-CN.md“使用 DFlash2 的步骤”）。 |
| `DRAFT_SAVES` | `8` | `CACHE=pool` 且使用 DFlash2 时：在请求之间保留草稿模型状态的对话数。 |

### 服务器

| 设置项 | 默认值 | 含义 |
|---|---|---|
| `NAME` | `glm-5.3` | API 中的模型名称：`/v1/models` 列出它，回复中带有它（请求可以指定任何模型名）。 |
| `HOST` | `0.0.0.0` | API 监听的地址。`0.0.0.0`：所有地址。API 没有密钥，能访问到它的人都可以使用：请将节点留在封闭网络中，或在前面放置 TLS 和访问控制。`127.0.0.1`：仅本节点；其他机器通过 ssh 隧道访问。 |
| `PORT` | `8000` | API 的端口。 |
| `METRICS_LISTEN` | 空 | `HOST:PORT`：第一个节点也在该地址上以只读方式响应 `GET /metrics`（Prometheus 文本）和 `GET /health`，供另一台主机上的监控使用。为空时关闭。 |
| `EXPERT_LIVE` | `1` | `1`：第一个节点为控制面板统计每个提示行和每个解码 token 选中的专家（实时显示、使用记录和专家优化）。`0`：不统计；面板仍显示正在运行的引擎和“停止”，但不显示选择，也没有专家优化。从面板启动时始终统计。 |
| `LOGS_IN_MEMORY` | `1` | `1`：当用户处于 lingering 状态时（在每个节点上 `sudo loginctl enable-linger USER`），日志（各 rank、监视进程、启动器）放在每个节点的内存中，即 `/dev/shm/tf-glm53-<uid>-p<PORT>`；否则放在本目录下的 `logs/`（磁盘）。`0`：始终为 `logs/`。内存中的日志在重启后消失。 |

### 上下文、缓存与 KV

| 设置项 | 默认值 | 含义 |
|---|---|---|
| `CONTEXT` | `819200` | **面板**。一个请求可达到的位置数（提示 + 回复）。 |
| `STORE` | `819200` | **面板**。在 `CONTEXT` 之外为之前的对话保留的位置数；`CACHE=pool` 时至少 512。 |
| `CACHE` | `pool` | `pool`：以所有对话共享的 256 位置块保存 `CONTEXT` + `STORE` 个位置。`slots`：每个对话一个 `STORE` 位置的存储。 |
| `KV` | `fp8` | **面板**。KV 缓存的格式：`fp8`，或 `fp4`（每个位置占用更少内存，精度略低；T4.zh-CN.md）。 |
| `DCP` | `2` | 解码上下文并行：`2` 把 KV 分到一对 rank 上，`1` 在每个 rank 上保存全部 KV。`CACHE=pool` 取 1 或 2。 |
| `ROWS` | `8192` | 提示分块的行数（四节点示例中为 `16384`）。 |
| `SP_PARTS` | `4` | 一个提示分块被拆分成的序列并行部分数。 |

### 内存与启动

| 设置项 | 默认值 | 含义 |
|---|---|---|
| `NEED_GB` | `112` | 启动前每个节点必须可用的 GiB（`tf-glm53 check`）：使用 DFlash2 时多 3；`STORE` 每比 819,200 少 100,000 个位置，少 2.7。四节点示例设为 131，按其 `STORE` 计算后为 112。从面板启动时改用计划中的数值。 |
| `READY_TIMEOUT` | `900` | `tf-glm53 up` 等待服务器的秒数。 |

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
| `EXTRA_ENV` | 空 | 传给每个 rank 的环境变量（`名称=值`，以空格分隔，放在引号中），例如 `EXTRA_ENV="TF_GLM53_MULTI=4"`（最多四个对话一起解码；`glm53.conf.example`）。启动器也会自己读取其中几项，例如 `TF_PANEL_HTML`（提供另一个页面）和 `TF_PANEL_SETTINGS`（页面设置的另一个文件）。 |
