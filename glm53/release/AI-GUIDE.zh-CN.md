[English](AI-GUIDE.md) | [日本語](AI-GUIDE.ja.md) | 简体中文

# 面向 AI 助手的指南：在 DGX Spark 上运行完整版 GLM-5.3（发布版本 r3.3.2-102）

本页在一处说明整个系统：它是什么、各部分做什么、如何使用、能做什么、不能做什么。本页是为帮助他人理解、安装、运行本发布版本或排查其故障的 AI 助手而写的，人来读也是同样的内容。其中的数字来自其他页面，那些页面有完整的细节，本页简略之处在那里有更多说明：`QUICKSTART.zh-CN.md`（首次启动）、`BEFORE-START.zh-CN.md`（启动前要决定的事项）、`RUNTIME.zh-CN.md`（运行，两种节点数）、`CONFIG.zh-CN.md`（所有设置）、`T4.zh-CN.md`（四个节点）、`RING.zh-CN.md`（连成环的四个节点）、`PANEL-GUIDE.zh-CN.md`（控制面板的各个界面）、`AGENT-API.zh-CN.md`（不用仪表板的控制 API）和 `ATLAS-TESTS.zh-CN.md`（专家重排的测量）。在源代码仓库中，`RUNTIME.md` 是 `packaging/RUNTIME.md`，其他页面在 `glm53/release/` 中。每个页面也有日语版（`.ja.md`）和简体中文版（`.zh-CN.md`）。

## 1. 它是什么

完整版 GLM-5.3 模型（755B 参数，架构 `glm_moe_dsa`）的服务器，运行在四台或八台 NVIDIA DGX Spark 系统（每台一个 GB10）上，每个 GB10 一个 rank，张量并行 4 或 8。它是 Fiore-lab 基于 TensorFold 1.0.2（一个用 Zig 编写、使用普通 CUDA 内核的引擎）制作的非官方衍生版本，并非 TensorFold 项目的发布版本。服务器运行时不需要 Python。

| 项目 | 内容 |
|---|---|
| 模型 | 完整大小的 GLM-5.3，由 Fiore-lab 在 Hugging Face 上发布的 EXL3 量化权重 |
| 节点 | 同一个 RoCE v2 网络上的 4 台或 8 台 DGX Spark（GB10），或不用交换机连成环的 4 台；2 台和 3 台的配置正在测试中，不支持其他配置 |
| 上下文长度 | 两种节点数都最多 1,048,576 个 token。八节点：在 4 bit KV 缓存上（`KV=fp4`、`NEED_GB=100`）`KV_SIZE` 为 2,097,152，在 8 bit 上为 1,638,400。四节点：通过控制面板，70 % 的路由专家为 3 bit、30 % 为 4 bit（4 bit KV，`KV_SIZE` 1,049,088）；默认启动不论节点数都请求 1,048,576，放不下时会缩短并说明。八个节点的 4 bit KV 缓存和四个节点通过控制面板的启动，各用一个约 1,019,000 个 token 的 prompt 测量。4 bit KV 缓存会损失一些精度（KL 0.0190 对 0.0156，`T4.zh-CN.md`）。示例附带的值是 819,200 和 1,048,576 |
| API | OpenAI 兼容的聊天 API（`/v1/chat/completions`）和 Anthropic 的 Messages API（`/v1/messages`），默认在所有地址上（`0.0.0.0:8000`） |
| 请求 | 默认一次运行一个，另有最多 32 个在队列中等待；`MULTI` 最多同时解码 4 个对话 |
| 草稿模型 | MTP（权重中的层，默认）、DFlash2（可选，不包含，另有许可证）或不使用 |
| 控制面板 | “Expert Atlas”：逐个专家的位宽（2/3/4/5 bit），一个网页；四个或八个节点 |
| 分发 | 容器镜像（推荐）和包含相同文件的归档；都用于 Linux aarch64 |
| 许可证 | 代码为 Apache-2.0；权重保留 GLM-5.3 License；DFlash2 保留其自身的许可证（CC BY-NC-ND 4.0） |

## 2. 组成部分

| 文件或目录（归档中） | 是什么 |
|---|---|
| `bin/tensorfold-native` | 服务器：GLM-5.3 引擎和 HTTP API，内置面向 GB10（sm_121）的 CUDA 内核 |
| `lib/libroce_proxy.so` | 解码集合通信使用的 RDMA 代理（需要 `libibverbs.so.1`） |
| `tf-glm53` | 部署、检查、启动、停止和监视各 rank 的脚本；在第一个节点上运行 |
| `claude-glm53` | 启动 Claude Code 并连接服务器的 Messages API（RUNTIME.zh-CN.md 的“Claude Code”一节） |
| `glm53.conf.example` / `glm53.t4.conf.example` | 八节点 / 四节点的设置；把其中一个复制为 `glm53.conf` |
| `CONFIG.md` | `glm53.conf` 的每项设置：默认值、由谁读取、修改何时生效 |
| `launcher/tf_launcher.py` | 控制面板的启动器（系统的 `python3`，只用标准库） |
| `panel.html` | 控制面板的页面（服务器持有相同的页面） |
| `tools/quant/expert_plan.py` | 写出专家计划（每个专家以哪种位宽载入） |
| `expert-plan/banks.json` | 每个专家 bank（一种位宽）在节点上的位置 |
| `QUICKSTART.md`、`BEFORE-START.md`、`RUNTIME.md`、`T4.md`、`RING.md`、`PANEL-GUIDE.md`、`ATLAS-TESTS.md` | 文档，每份也有日语版（`.ja.md`）和简体中文版（`.zh-CN.md`） |
| `AI-GUIDE.md` | 本页（也有 `.ja.md`、`.zh-CN.md`） |
| `docker/Dockerfile`、`docker/entry.sh` | 作为容器镜像的发布（`up`、`install`、`serve`、`version`） |
| `tools/quickstart.py` | 从全新安装到引擎运行（`up` 和 `./tf-glm53 quickstart` 会运行它） |
| `tools/topology.py` | 判定节点的连接方式（`./tf-glm53 topology`） |
| `tools/logkeep.py` | 每个节点上的日志保管进程：日志保留在内存中 |
| `AGENT-API.md` | 启动器和引擎面向 AI 代理的控制 API，不需要仪表板（也有 `.ja.md`、`.zh-CN.md`） |
| `VERSION`、`SHA256SUMS`、`LICENSE`、`NOTICE`、`LICENSES/`、`THIRD_PARTY_NOTICES.md` | 发布元数据和许可证 |

不在归档中：模型权重、专家 bank、DFlash2 草稿模型和 NCCL。它们需要另外下载（第 4 节）。权重和 bank 也可以在 GPU 主机上从 Z.AI 的官方 BF16 检查点制作，不需要 Fiore-lab 的任何文件：见源代码仓库中的 `glm53/weights/README.md`（<https://github.com/fiore-labo/ExpertAtlas-GLM5.3/tree/main/glm53/weights>），它不在归档中。

## 3. 工作原理简述

- **rank。** 每个 GB10 上一个进程（rank）。rank 0 运行在 `NODES` 中的第一个节点上，并提供 HTTP API。张量并行把每一层分到各 rank 上；KV 缓存分到成对的 rank 上（解码上下文并行，`DCP=2`）。
- **prefill 与解码。** prompt 分块读取（prefill），然后逐个 token 生成回复（解码）。prefill 速度受计算限制，解码速度受内存带宽限制。
- **草稿。** 每轮解码时，草稿模型提出几个 token，模型一步核对它们；只保留模型自己也会选择的 token，因此无论如何回复都是模型自己的。
- **对话缓存。** KV 缓存是一个有 `KV_SIZE` 个位置（KV 大小）的池，以 256 个为一块，由所有对话共享：一个请求最多达到 `CONTEXT_LENGTH` 个位置（上下文长度），先前的对话保留在任何空闲的块中。请求从任一对话中已缓存的最长前缀处继续，开头相同的对话（同一个系统提示词）只持有一份共同的块。已缓存的块在复用前会与请求的 token 核对；只要有疑问，所有对话都会被丢弃，请求从头计算（“cold”），因此块绝不会被错误复用。
- **专家。** GLM-5.3 在第 3 至 77 层每层有 256 个路由专家（共 19,200 个），每个 token 每层选 8 个。启动时从专家 bank 以各自的位宽载入每个专家；没有计划时以 3 bit 载入所有专家。其他权重在 `fiore-lab/GLM-5.3-EXL3-Base` 中。完整检查点也以量化形式存放专家（八节点检查点全部为 4 bit，四节点检查点一半为 3 bit（使用最少的）、一半为 4 bit），但启动时不读取它们。
- **Byte Resume。** 模型切分文本的方式可能与分词器不同，因此下一轮的 prompt 以不同的 token id 持有相同的字节。从第一个不同的 id 起，prompt 与保留的对话按字节比较，缓存越过该处继续（`TF_GLM53_BYTE_RESUME`，默认开启）。
- **思考。** `max_tokens` 把它计算在内。默认情况下，思考最多可保留 prompt 在上下文长度中剩余部分的 90 %（`THINKING_SHARE`），`max_tokens` 较小时则以它为基准；接近该上限时，模型改写自己的思考，改写的内容在上下文中取代原来的思考，达到上限时引擎结束思考。该上限针对的是上下文中保留的内容，因此一次回复写出的思考可以更多。这些设置可在引擎运行时更改（`POST /v1/thinking`，面板的 [ REASONING CONTROL ]）。
- **多个对话。** 设置 `MULTI` 后，最多这么多个对话（不超过 4 个）一起解码。它们运行期间，`POST /v1/schedule` 或面板设置 GPU 在 prefill 和解码之间的分配，短 prompt 可以优先。

## 4. 两种运行方式

| | `docker run ... up` 或 `./tf-glm53 start` | 控制面板（`./tf-glm53 up`） |
|---|---|---|
| 节点 | 4 个或 8 个，由节点检查找到 | 4 个或 8 个，由节点检查登记（`glm53.nodes.conf`） |
| 专家位宽 | 所有专家 3 bit（其他位宽用 `BITS`；`start`：上一次启动的计划） | 从 bank 中逐个专家选择（2、3、4、5 bit） |
| 需要 | 始终需要 Base 权重和 3 bit bank | Base 权重、3 bit bank，以及所用位宽的 bank |
| 启动位置 | 命令行 | 网页（启动、停止、计划、实时视图、专家优化）；`./tf-glm53 start` 重复上一次启动的计划 |

两者使用相同的 `glm53.conf`、相同的服务器和相同的 API。`./tf-glm53 up` 启动面板的启动器并输出面板的地址；引擎在那里启动。`./tf-glm53 down` 在两种方式下都停止引擎。

AI 助手在无人操作页面时启动引擎，使用下面这一行或 `./tf-glm53 start`：后者以上一次启动的计划（`launch/plan-*.bits`；还没有时：所有专家 3 bit）启动，与面板的 Start 相同，并等待引擎运行。手动运行 `./tf-glm53 engine-up` 也一样。在两者之前加 `-d KIND` 可为该次启动选择草稿模型。

### 一行命令，面向 AI 助手（推荐方式）

在第一个节点上，以运行它的用户执行（每个节点都能通过无密码 ssh 访问每个节点，并已登记它们的 host key（`QUICKSTART.zh-CN.md` 第 1 节），镜像已在节点上或可以拉取）：

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host IMAGE up
```

它把发布内容和 `glm53.conf` 写到 `~/expertatlas`，在主机上启动启动器，找到节点（环或交换机）并检查，查找权重，在容器中以所有专家 3 bit 启动各 rank，并发送一次请求。此容器退出后，启动器和引擎继续运行。它每一步输出一行带编号的信息，最后以 `READY` 结束并给出地址（仪表板 `http://NODE:8010/panel`、API `http://NODE:8000/v1`，没有密钥，都对局域网开放），或者以 `STOP`（出了什么问题，哪些节点上的哪个目录和哪个仓库）、`FIX`（具体要做什么）和 `THEN`（要再次运行的一行）结束。每项启动设置都是选项，写作 `-e NAME=VALUE`：`WEIGHTS`、`BANK_2` 至 `BANK_5`、`DRAFTER`、`BITS`、`ALLOC`、`NODES`、`PORT`、`LAUNCHER_PORT`、`EA_DIR`、`IMAGE`，以及引擎设置的任何项（`TOPOLOGY`、`SOCKET_IFNAME` 和 `IB_HCA` 除外，它们由节点检查决定）。`QUICKSTART.zh-CN.md` 列出了它们，也是首次启动时阅读的页面。之后见 `AGENT-API.zh-CN.md`。发布的镜像包含带 AArch64 fence 的 NCCL，在交换机和环上都读取它（`RING.zh-CN.md`）。

### 在容器中

在 `glm53.conf` 中设置 `RUNTIME=docker` 和 `IMAGE` 后，`tf-glm53` 把每个 rank 作为发布镜像的容器在其节点上启动（每个 rank 一个容器，从第一个节点通过 ssh）。节点需要 Docker 和 NVIDIA Container Toolkit（DGX OS 都带有），而不是 cuBLAS、libibverbs 和 NCCL；不需要下面的第 1 和第 2 步。在每个节点上 `docker pull` 镜像（或 `./tf-glm53 pull`），用 `mkdir glm53 && docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/glm53":/out IMAGE install /out` 把发布的文件写到第一个节点上的一个目录（要先创建：Docker 为挂载创建的目录属于 root，install 无法写入），然后从第 4 步继续。权重和 bank 留在主机上，以只读方式挂载。详情见 `RUNTIME.zh-CN.md` 的“在容器中运行”；镜像也可以用 fenced NCCL 从归档构建（`docker/Dockerfile`、`RING.zh-CN.md`）。

### 设置（两种方式都需要）

1. 每个节点：DGX OS（Ubuntu 24.04）、NVIDIA 驱动 580 或更高版本、cuBLAS 13（`libcublas-13-0`）、`libibverbs`，节点在同一个 RoCE v2 网络上，第一个节点能通过无密码 ssh 访问每个节点，并已登记它们的 host key（`QUICKSTART.zh-CN.md` 第 1 节）。
2. 每个节点上的 NCCL 2.30.7：`pip3 install --no-deps --target ~/nccl-2.30.7 nvidia-nccl-cu13==2.30.7`（pip 只解压库；不运行任何 Python）。在交换机上这样可以运行，但推荐使用带 AArch64 fence 的构建；环必须使用该构建（`RING.zh-CN.md`）。
3. Base 权重和 3 bit bank，在每个节点上放在相同路径下：`fiore-lab/GLM-5.3-EXL3-Base`（21.2 GB）和 `fiore-lab/GLM-5.3-EXL3-Experts-3bit`（277.9 GB），四节点和八节点都一样。用 Hugging Face CLI 下载一次，用 `rsync` 复制到其他节点，用 `./tf-glm53 verify` 校验。也可以用完整检查点（`fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks`，390 GB，或 `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks`，344.6 GB）代替 Base 权重。它们都可以从官方 BF16 检查点制作（第 2 节，`glm53/weights/`）。
4. 在第一个节点上解压归档，运行 `sha256sum -c SHA256SUMS`，把对应的示例复制为 `glm53.conf`，设置 `NODES`、`SOCKET_IFNAME`、`IB_HCA`、`NCCL_LIB`、`WEIGHTS`。
5. `./tf-glm53 deploy`（把目录复制到其他节点），然后，不用页面时运行 `./tf-glm53 start`（所有专家 3 bit：检查每个节点，启动各 rank，等待服务器，运行冒烟测试）；使用页面时运行 `./tf-glm53 up`（见下文）。从本地 NVMe 驱动器启动约需两分半钟。
6. `./tf-glm53 down` 停止引擎（启动器运行时经由启动器）和每个 rank。

### 设置控制面板

1. 计划要使用的专家 bank，放在每个节点上 `expert-plan/banks.json` 指定的目录中（默认 `~/models/GLM-5.3-EXL3-Experts-<N>bit`）：四个节点时，`fiore-lab/GLM-5.3-EXL3-Experts-3bit` 和 `-4bit` 是常用的一对（277.9 和 368.5 GB）；`-2bit`（187.3 GB）和 `-5bit`（459.1 GB）可选。八节点检查点可以作为 4 bit bank（`RUNTIME.zh-CN.md` 的“控制面板”）。用 `./tf-glm53 verify-banks 3 4` 校验。
2. 在第一个节点上运行 `./tf-glm53 up`，启动启动器并输出面板的地址；打开 `http://FIRST-NODE:8010/panel`（没有 `LAUNCHER_PORT` 时，端口为 `PORT` + 10）。启动器默认监听所有地址，且没有密钥：任何能访问到它的人都可以启动、停止和更改引擎（`LAUNCHER_HOST=127.0.0.1` 把它限制在本节点；这时用 ssh 隧道，`ssh -L 8010:127.0.0.1:8010 FIRST-NODE`）。无论引擎是否运行，它都保持运行；页面通过它访问引擎，在它运行期间启动的引擎会把打开引擎自身 `/panel` 的浏览器转到它那里。引擎停止期间，编辑后它会自行重新读取 `glm53.conf`；只有 `LAUNCHER_HOST` 和 `LAUNCHER_PORT` 需要 `./tf-glm53 launcher stop` 和 `./tf-glm53 up`。第一次时，页面从节点检查打开，它找到环或交换机上的节点，检查后写入 `glm53.nodes.conf`。该文件在 `glm53.conf` 之后读取，并在 `TOPOLOGY`、`NODES`、`SOCKET_IFNAME` 和 `IB_HCA` 上以它为准。启动界面的 [ WEIGHTS ] 设置权重的位置，引擎设置界面设置需要重启的设置。
3. 在页面上：选择计划、KV 格式、上下文长度、每个节点为操作系统保留的内存、草稿模型和并发推理数（除非输入 KV 大小，否则它是计划剩余的全部空间）；查看内存判定；按 Start（然后 Confirm）。也从页面停止（或 `./tf-glm53 down`）。之后使用相同计划启动：`./tf-glm53 start`。

## 5. 设置（`glm53.conf`）

| 设置 | 含义 | 八节点示例 / 四节点示例 |
|---|---|---|
| `NODES` | 节点在 RoCE 网络上的地址，rank 0 在最前（环：在管理网络上） | 8 个 / 4 个地址 |
| `TOPOLOGY` | `full`（交换机）、`ring4` 或 `auto`（每次 `check` 和 `./tf-glm53 up` 时根据连线判定；已知缺陷见 `CONFIG.zh-CN.md`） | `full` |
| `SOCKET_IFNAME`、`IB_HCA`、`GID_INDEX` | 每个节点上该网络的接口和 RDMA 设备；GID 索引（空：自动选择） | 相同 |
| `NCCL_LIB` | NCCL 2.30.7 的 `libnccl.so.2`，每个节点上路径相同 | 相同 |
| `WEIGHTS` | Base 权重或完整检查点的目录，每个节点上路径相同 | `~/models/GLM-5.3-EXL3-Base` |
| `DRAFTER_KIND`、`MTP_K`、`DRAFTER` | 草稿模型（`mtp`、`dflash2`、`none`）、MTP 每轮的草稿数（3）、DFlash2 的目录 | `mtp`、3、空 |
| `NAME`、`HOST`、`PORT` | API 中的模型名称、监听地址和端口 | `glm-5.3`、`0.0.0.0`、8000 |
| `METRICS_LISTEN` | 额外的只读地址，只响应 `/metrics`、`/health` 和 `/v1/experts/usage` | 关闭 |
| `LOG_LISTEN` | 每个节点的日志保管进程在此地址以只读方式提供其日志 | 关闭 |
| `CONTEXT_LENGTH` | 上下文长度：一个请求可达到的位置数，prompt 和回复合计（最多 1,048,576） | 819,200 / 1,048,576 |
| `KV_SIZE` | KV 大小：整个 KV 缓存的位置数，所有对话共享一个池；请求未使用的部分保留先前的对话（至少为 `CONTEXT_LENGTH` + 512） | 1,638,400 / 1,049,088 |
| `KV` | KV 缓存的格式，`fp8` 或 `fp4` | `fp8` / `fp4` |
| `DCP`、`ROWS`、`SP_PARTS` | 解码上下文并行、prompt 分块的行数、每块的序列并行部分数 | 2、8,192、4 |
| `NEED_GB` | 启动前节点必须可用的内存（随 KV 大小变化，使用 DFlash2 时多 3；计算方法见 `CONFIG.zh-CN.md`） | 112（默认）/ 115（要求 101 GiB） |
| `MASTER_PORT`、`READY_TIMEOUT` | 各 rank 的 rendezvous 端口；等待服务器的秒数 | 29551、900 |
| `RUNTIME`、`IMAGE` | rank 的运行方式：`host`（默认，从本目录）或 `docker`（各在 `IMAGE` 的容器中） | `host`、空 |
| `MULTI`、`MULTI_WAIT` | 一起解码的对话数（1 .. 4）；请求可等待在它们旁边腾出空间的秒数 | `0`（逐个处理）、600 |
| `MULTI_DECODE_SHARE`、`MULTI_SHORT_PROMPT`、`MULTI_SHORT_PREEMPT` | 使用 `MULTI` 时：prefill 进行期间解码所占 GPU 时间的百分比；不超过此长度的 prompt 优先；它是否抢在运行中的 prefill 之前 | 30、8192、0 |
| `MAX_TOKENS_SHARE`、`MAX_TOKENS` | 请求未指定时回复的 `max_tokens`：`CONTEXT_LENGTH` 的百分比，或优先于它的数量 | 70、空 |
| `THINKING_SHARE`、`THINKING_BUDGET` | 思考的上限：百分比，或优先于它的数量 | 90、0 |
| `THINKING_FOLD`、`THINKING_FOLD_SHARE`、`THINKING_FOLD_MAX` | 中途整理：方式（`off`、`recent`、`checkpoint`、`branch`）、频率、次数 | `off`、100、0 |
| `THINKING_FOLD_AT_BUDGET`、`THINKING_FOLD_MARGIN` | 接近上限时的整理，以及提前多少开始 | 1、10 |
| `THINKING_FOLD_AT_CLOSE_SHARE` | 模型结束思考时的整理：至少达到上限的这一百分比；0 为关闭 | 0 |
| `THINKING_NOTICE` | 每隔这么多思考 token，用一行告诉模型已用的量；0 为关闭 | 0 |
| `TF_GLM53_BYTE_RESUME` | Byte Resume：`on`、`log`、`off` | `on` |
| `ROCE_PROXY_CORES` | RoCE 代理线程轮询所在的核：`a725`、`os` | `a725` |
| `EXTRA_ENV` | 引擎变量（不包括 `TF_GLM53_MULTI`：那是 `MULTI`） | 空 |
| `LAUNCHER_HOST`、`LAUNCHER_PORT`、`ATLAS_DIR`、`RESERVE_GB` | 控制面板：地址、端口、已保存的分配、为操作系统保留的内存 | 四节点示例：`0.0.0.0`、8010、`atlas/`、4 |

旧的 `glm53.conf` 文件仍可读取：`CONTEXT` 读作 `CONTEXT_LENGTH`，`CONTEXT` + `STORE` 读作 `KV_SIZE`。`CONFIG.zh-CN.md` 包含所有设置。

## 6. 命令（`./tf-glm53`，在第一个节点上运行）

| 命令 | 作用 |
|---|---|
| `deploy` | 把本目录复制到每个节点（相同路径）并在那里检查 `SHA256SUMS` |
| `check` | 检查节点、文件、权重、库、RDMA 设备、GPU 和内存；在 `NG:` 之后列出问题 |
| `verify` | 在每个节点上按权重的 `SHA256SUMS` 校验权重（几分钟） |
| `verify-banks [BITS...]` | 按清单校验专家 bank 的文件 |
| `quickstart` | 从全新安装到引擎运行，与 `docker run ... IMAGE up` 所运行的相同 |
| `topology [--json]` | 判定节点的连接方式 |
| `up` | 启动控制面板的启动器并输出面板的地址；引擎从面板启动 |
| `start` | 通过启动器以上一次启动的计划启动引擎，与面板的 Start 相同；等待它运行 |
| `down` | 停止引擎：启动器运行时经由启动器，然后直接停止每个 rank |
| `engine-up` | 手动，与 `start` 相同；使用启动器的计划：`check`，启动各 rank（从最后一个到第一个），等待服务器，运行冒烟测试 |
| `engine-down` | 直接停止每个 rank（TERM，30 s 后 KILL） |
| `status` | 各 rank、可用内存、草稿模型、对话缓存、服务器的 `/health` |
| `smoke` | 三个请求，检查其回答 |
| `logs [R]` | 跟踪 rank R 的日志（默认 0；`R.prev`、`watch`、`launcher`）；日志在每个节点的内存中 |
| `launcher [stop\|status]` | 启动（与 `up` 相同）、停止或报告控制面板的启动器 |
| `-d KIND`（在命令之前） | 本次启动的草稿模型（`start` 或 `engine-up`：`mtp`、`dflash2`、`none`） |
| `pull` | `RUNTIME=docker`：在每个节点上 `docker pull IMAGE` |
| `-c FILE`（在命令之前） | 另一个设置文件 |

`tf-glm53` 不使用 `sudo`，也不更改系统设置。各 rank 只应通过它启动和停止：它会留下一个监视进程，在部分 rank 消失时停止所有 rank（否则等待已丢失对端的 rank 会永远等待）。

## 7. API

| 路径 | 方法 | 说明 |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI 聊天：流式（`stream`、`stream_options.include_usage`），工具（`tools`、`tool_choice` `auto` 或 `none`） |
| `/v1/messages`、`/v1/messages/count_tokens` | POST | Anthropic 的 Messages API，供 Claude Code 使用（`RUNTIME.zh-CN.md` 的“Claude Code”） |
| `/v1/responses` | POST | OpenAI 的 Responses API |
| `/v1/completions` | POST | 对 prompt 的纯文本补全（不添加聊天模板） |
| `/v1/models` | GET | 提供的模型名称 |
| `/v1/tokenize`、`/v1/detokenize` | POST | token 数和 id，针对 prompt 或经聊天模板渲染的消息 |
| `/health` | GET | 状态、内存、实时速度、对话池、token 和草稿计数器 |
| `/metrics` | GET | 相同的计数器，Prometheus 文本 |
| `/v1/experts/...`、`/v1/memory` | GET / POST | 专家的使用情况、位宽和交换，供控制面板使用 |
| `/v1/thinking` | GET / POST | 引擎运行时思考的设置 |
| `/v1/schedule` | GET / POST | GPU 的分配和短 prompt，使用 `MULTI` 时（没有它时返回 404） |
| `/v1/pool` | GET | 保留的对话 |

`/v1/experts/...`、`/v1/memory`、`/v1/thinking`、`/v1/schedule` 和 `/v1/pool` 只在第一个节点上响应（从其他地方访问返回 403）。启动器的端口把它们提供给网络。

- 回复的 `message.reasoning_content` 保存模型的思考，`message.content` 保存回答；`max_tokens` 把两者都计算在内。`chat_template_kwargs` 接受 `{"enable_thinking": false}`（无思考）和 `reasoning_effort`。请求自身的 `thinking_budget`、`thinking_fold`、`thinking_fold_at_budget`、`thinking_fold_margin` 和 `thinking_fold_at_close`（单位为 token）优先于设置。`reasoning_details` 携带上下文中保留的思考，供下一轮使用。
- 以 400 拒绝：`n` 不为 1、`logprobs`、`response_format` `json_object` 和 `json_schema`、`guided_*`、`tool_choice` `required` 或指定工具、聊天 API 上的图片（在 `/v1/messages` 上图片变为一条说明；仅文本），以及在既无中途整理也无预算将尽时的整理时，超过上下文剩余量的 `max_tokens`（`CONFIG.zh-CN.md` 的 `MAX_TOKENS` 给出中途整理时的规则）。放不下的流式请求以错误事件结束。
- 客户端断开连接会取消其请求。
- 没有密钥：API 不使用密钥。默认监听所有地址（`HOST=0.0.0.0`），因此任何能访问到它的人都可以使用；服务器使用明文 HTTP，因此除非服务器留在封闭网络中，或在前面放置 TLS 和访问控制，否则文本以明文穿过网络（`HOST=127.0.0.1` 把 API 限制在第一个节点）。
- 从计数器得到速度：解码 tokens/s = Δ`completion_tokens_total` / Δ`decode_seconds_total`；prefill tokens/s = Δ`prefill_tokens_total` / Δ`prefill_seconds_total`；每轮草稿保留的 token 数 = Δ`completion_tokens_total` / Δ`rounds_total`（1 表示没有保留草稿）；缓存命中 = Δ`cached_tokens_total` / Δ`prompt_tokens_total`。计数器包括正在运行的请求，且永不回退。

控制面板的启动器在 `/v1/launch/` 下有自己的 API（status、plan、start、stop、restart、settings、weights、discover、ring、check、config、usage、resort，以及已保存的分配：allocs、alloc、allocs/save、allocs/rename、allocs/delete）。`AGENT-API.zh-CN.md` 给出代理所需的部分；其字段在不同发布之间可能变化。

## 8. Expert Atlas（控制面板）

- **bank**：每种一个位宽（2、3、4 或 5 bit）的路由专家，另外下载，各自附带清单。页面只提供每个节点都具备的位宽。
- **计划（分配）**：19,200 个专家各自的位宽。启动界面打开时是一个新计划，比例取上一次运行的比例（还没有运行时：全部 3 bit）；比例可按位宽输入，或在曲线上拖动。Open 按位原样恢复上一次运行的分配或已保存的分配，从它启动时使用它的位宽和保存时所用的 bank。页面在启动前显示每个节点的内存和判定（Fits、Tight、Does not fit）。
- **使用记录**：引擎统计每个 prompt 行（prefill）和每个生成的 token（解码）选中的专家。每个已保存的分配保留保存它的那些运行的使用记录；没有所有运行共用的池。新计划、运行、保存：只有那次运行的记录。打开的分配、运行、保存：它的记录加上那次运行的记录。计划按其来源分配的记录对专家排序（没有记录的计划，按页面自带的次数排序，测量自 12 类工作）。使用最多的专家得到最宽的位宽。
- **保存**：不会自动保存任何内容。停止时，Save and stop 把已载入的位宽和使用记录以一个名称保存（默认是本次运行所基于的分配，覆盖保存；旧版本移到 `ATLAS_DIR/allocs/trash/`）；运行面板上的 Save 可随时做同样的事。分配是文件 `ATLAS_DIR/allocs/user/<name>.json`；Open 列出它们，并可重命名和删除。
- **Count（排序依据）**：decode、prefill 或 Both。Both 对专家在其所在层解码选择中的占比加权 0.7，对 prefill 选择中的占比加权 0.3。prefill 和解码选择不同的专家（它们每层前 64 名的重叠为 42 %，随机情况为 25 %；层越深，重叠越少）。
- **专家优化（重排）**：仅在引擎运行时。它按本次运行的使用记录（开始时带有的记录加上本次运行的选择）重新分配已载入专家的位宽，每种位宽的专家数量保持不变，因此内存不变。它在没有请求运行或等待时开始，在每个 rank 上就地交换各层，丢弃保留的对话（它们的 KV 是用旧专家计算的），然后再次提供服务；其间到达的请求等待（四个节点上，完整更改 75 层用了 83 至 90 s）。每个节点需要在之前各层增长量之上，再为最大一层的新副本留出空间，外加 2 GiB；估算会说明每个节点是否具备，某个节点不具备时，页面拒绝更改并指出该节点。结果不会自动保存。启动时载入给定的位宽，绝不重排。设置了 `MULTI` 时也一样：更改会等到没有正在运行或等待的请求。
- **重排的收益**（四个节点，一半 3 bit / 一半 4 bit，Both）：相对全 4 bit 模型的 KL 散度，在回复的 token 上，8,192 个 token 时下降 24 %，128K 和 256K 个 token 时下降 32 至 34 %；在 prompt 的 token 上下降 15 至 22 %（`ATLAS-TESTS.zh-CN.md`）。在其排序所用次数未涵盖的工作类型上，收益较小（约三分之二）。
- **内存，四个节点**（4 bit KV，上下文长度最多 737,280 的计划，KV 大小为上下文长度 + 100,000，MTP，`ROWS` 为 16,384；默认的 8,192 余量更多）：全部 3 bit 可容纳上下文长度 737,280 个 token；一半 3 bit / 一半 4 bit 为 737,280；四分之一 3 bit / 四分之三 4 bit 为 622,592；全部 4 bit 为 327,680；全部 5 bit 无法容纳。`CONTEXT_LENGTH` 1,048,576 和 `KV_SIZE` 1,049,088 时，70 % 3 bit、30 % 4 bit 可容纳，每个节点约余 7.5 GiB，足以进行运行中的重排。使用 DFlash2 时，启动器为每个节点多计划约 3 GiB（八个节点时约 4.3 GiB）。相同内存下，含 2 bit 专家的配置离基准远 13 至 30 %。

## 9. 它能做什么（实测）

- **速度**（使用发布版本 r2-102 和各检查点自身的位宽测量，未针对本发布版本重新测量；四个节点的 `ROWS` 当时为 16,384，现在为 8,192；GPU 时钟上限 2.1 GHz，一个请求，尚未缓存的编程 prompt，reasoning effort 为 low；速度因环境而异，这些数字仅供参考）：

| 节点 | prompt | MTP：首 token 时间 / 解码 | DFlash2：首 token 时间 / 解码 |
|---|---|---|---|
| 8 | 128K | 82.8 s / 47.5 tokens/s | 85.2 s / 53.2 |
| 8 | 256K | 172.9 s / 45.7 | 177.6 s / 52.6 |
| 8 | 512K | 376.5 s / 42.8 | 386.0 s / 47.1 |
| 4 | 128K | 120.1 s / 35.3 | 121.9 s / 42.9 |
| 4 | 256K | 253.3 s / 33.2 | 256.9 s / 40.7 |
| 4 | 512K | 565.4 s / 32.2 | 571.5 s / 37.8 |

- **已缓存的 prompt**：再次发送的对话从池中继续；一个 703,916 个 token 的对话在四个节点上不到 3 s 就给出首个 token。
- **精度**（相对于以官方实现、FP32 运行的官方 BF16 模型的 KL 散度）：八个节点在 2,048 个 token 的文本上约 0.013（top-1 token 一致率约 96 %）；在 8,192 个 token 的文本上为 0.012 至 0.019；四个节点在相同文本上为 0.019 至 0.032。所有路由专家为同一位宽，位置 6,144 至 8,190：2 bit 0.0517，3 bit 0.0203，4 bit 0.0127，5 bit 0.0114。
- **工具和流式**：OpenAI 风格的函数调用和带思考增量的流式回复。

## 10. 它不能做什么

- 在四台或八台 GB10 系统以外的环境运行（不支持其他 GPU、其他 rank 数或单节点模式）。两台和三台的配置正在测试中。
- 默认并行处理请求：一次运行一个，最多 32 个等待（聊天标题请求和 `"priority": "background"` 的请求排在最后），更多的以 503 "the engine is busy; retry shortly" 拒绝。并行运行子代理的代理，把子代理发送到另一个模型效果更好。设置 `MULTI=4` 时最多保持 4 个对话活动并一起解码（MTP 层、DFlash2 或无草稿模型；暂不支持 3 个节点）；它占用节点内存，默认关闭。
- 在请求运行时更改专家位宽（重排会等待请求结束），或在启动后增大 KV 缓存（它在启动时预留）。
- 读取图片（聊天 API 拒绝图片，`/v1/messages` 把图片变为一条说明），或接受 `n` > 1、`logprobs`、JSON 模式或强制工具调用（`tool_choice` `required` 或指定工具）。
- 在提供服务的配置中保证温度 0 时各次运行的回复逐比特相同：某些内核的求和顺序会变化。（精度测量使用为评分而设的确定性配置。）
- 包含 DFlash2：它另外下载，有自己的许可证（CC BY-NC-ND 4.0，非商业性使用，研究和评估）；某种使用是否在许可范围内，由其运营者判断。

## 11. 故障排除

| 症状 | 可能的原因 | 要做什么 |
|---|---|---|
| `check` 输出 `NG: files` / `weights` | 某个节点上的目录或权重不同 | `./tf-glm53 deploy`；`./tf-glm53 verify` |
| `NG: NCCL(...)`、`cuBLAS-13`、`libibverbs` | 该节点上缺少某个库 | 安装它（`RUNTIME.zh-CN.md` 的“出现问题时”） |
| `NG: RoCE-v2-GID` 或 RDMA 设备 | 该端口没有 RoCE v2 IPv4 条目，或 `IB_HCA` 错误 | 为端口配置地址或设置 `GID_INDEX`；`rdma link`、`ibdev2netdev` |
| `NG: GPU-busy` | 其他进程占用 GPU | 停止它（`nvidia-smi`） |
| `NG: memory(N<M GiB)` | 可用内存太少 | 释放节点上的内存，或减小 `KV_SIZE` |
| 某个节点上每项都是 `NG` | 第一个节点无法通过 ssh 访问它 | `ssh -o BatchMode=yes ADDRESS true` 必须成功 |
| 启动时：`a rank stopped` 和 `NCCL WARN` | 网络名称或地址错误，端口被阻止 | `NODES` 在 RoCE 网络上；每个节点上的 `SOCKET_IFNAME`、`IB_HCA` 正确 |
| 启动时：`cannot listen on` | `PORT` 在该节点上已被占用 | 释放它或更改 `PORT` |
| 启动时：`the smoke test failed` | 某个回复没有回答其问题 | 失败的问题会显示回复；rank 0 的日志中有该请求 |
| 启动时：`not ready after N s` | 磁盘慢或使用网络存储 | 增大 `READY_TIMEOUT`；确认 `MASTER_PORT` 空闲 |
| `PoolValuesDiffer` | 各 rank 以不同设置启动 | 用 `tf-glm53` 从同一个 `glm53.conf` 启动每个 rank |
| 服务器不再应答，所有 rank 仍在运行 | 挂起 | `./tf-glm53 down`、`./tf-glm53 start` |
| 节点内存耗尽 | KV 大小过大 | 每个 rank 以 `oom_score_adj` 1000 运行，因此内核停止该 rank，监视进程停止其余 rank，节点继续运行；减小 `KV_SIZE` |
| 面板：某个位宽显示 "No data" | 某个节点上缺少其 bank | 在那里下载；`./tf-glm53 verify-banks N` |
| 日志中的 `PoolWriteRefused` | 某个 rank 拒绝了对池中请求块之外的写入 | 每个 rank 在该请求结束时停止；`./tf-glm53 start` |

日志：在内存中，位于每个节点的日志保管进程（`tools/logkeep.py`）中，不写入磁盘：每个 rank 的日志（以及上一次启动的，`R.prev`）、监视进程的日志（它发现已消失的 rank）和启动器的日志，`tf-glm53 logs R | R.prev | watch | launcher`。设置 `LOG_LISTEN` 后，监控程序可通过 HTTP 读取并保存它们。

## 12. 容易出错的地方

- `max_tokens` 也计入思考 token。未指定的请求得到 `CONTEXT_LENGTH` 的 70 %。思考的上限为回答保留其余部分，因此 `max_tokens` 较小时回答较短；`THINKING_SHARE=0` 时，回复可能在思考中结束。
- 无论使用哪种草稿模型，引擎的启动行都显示 `drafts: on`；`tf-glm53 status` 给出正在使用的草稿模型。
- 每次启动都需要 Base 权重（或完整检查点）和 3 bit bank。
- 重排的测量相对于全 4 bit 模型，而不是 BF16 模型；`ATLAS-TESTS.zh-CN.md` 有具体说明。
- 在聊天模板的用户轮次内部，模型对用户自己文本的预测很差。这涉及对用户文本的预测（如 prompt 上的 KL 散度），与回复无关。`ATLAS-TESTS.zh-CN.md` 中的测量对 prompt 的 token 评分时不含模板的开头（“测量方法”）。
- 池缓存要求 `KV_SIZE` 至少为 `CONTEXT_LENGTH` + 512。
- 容器需要 GPU、`/dev/infiniband`、主机网络和 `--ulimit memlock=-1 --ulimit nofile=1048576:1048576`。
- `/v1/thinking` 和 `/v1/schedule` 只在第一个节点上响应；从其他机器需经由启动器的端口。
- `glm53.nodes.conf` 在 `TOPOLOGY`、`NODES`、`SOCKET_IFNAME` 和 `IB_HCA` 上优先于 `glm53.conf`。
- API 默认监听所有地址（`HOST=0.0.0.0`）且没有密钥：任何能访问到它的人都可以使用（`HOST=127.0.0.1` 把它限制在第一个节点）。控制面板的启动器也一样（`LAUNCHER_HOST`）。
