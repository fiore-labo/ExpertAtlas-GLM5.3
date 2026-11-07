[English](README.md) | [日本語](README.ja.md) | 简体中文 | [한국어](README.ko.md)

# 在四台或八台 DGX Spark 上运行完整版 GLM-5.3

完整版 GLM-5.3（755B，`glm_moe_dsa`）由四台或八台 NVIDIA DGX Spark 系统（每台一个 GB10）以张量并行 4 或 8 提供服务，连接方式为一台 RoCE 交换机，四个节点时也可以是直连线缆组成的环：速度快，并为智能体（agent）和编程提供长上下文。默认一次处理一个请求；`MULTI` 可同时解码最多四个对话。引擎用 Zig 编写，使用纯 CUDA 内核，运行时不需要 Python。

这是 Fiore-lab 基于 [TensorFold](https://github.com/ashhart/TensorFold) 1.0.2 制作的非官方衍生版本，并非 TensorFold 项目的发布版本。它包含 TensorFold 的代码，增加了完整版 GLM-5.3 的引擎（`zig/src/families/glm_moe_dsa/`、`zig/kernels/cuda/glm53/`），并去掉了 TensorFold 的文档（这些文档描述的是其他模型、权重和机器）。关于 TensorFold 本身，请参阅其仓库。

## 文档

每个页面都有英文、日文（`.ja.md`）和简体中文（`.zh-CN.md`）版本。在发布归档和镜像中，它们都放在顶层目录。

| 页面 | 内容 |
|---|---|
| [QUICKSTART](glm53/release/QUICKSTART.zh-CN.md) | 首次启动：权重、一行 `docker run`、所有选项、常见的 `STOP` 消息 |
| [BEFORE-START](glm53/release/BEFORE-START.zh-CN.md) | 首次启动前要决定的事项，以及可以以后再决定的事项 |
| [PANEL-GUIDE](glm53/release/PANEL-GUIDE.zh-CN.md) | 控制面板的各个界面，逐项说明 |
| [SCREENS](glm53/release/SCREENS.zh-CN.md) | 控制面板的每个界面，附截图 |
| [FEATURES](glm53/release/FEATURES.zh-CN.md) | 每个功能做什么、为什么需要它、如何使用 |
| [CONFIG](glm53/release/CONFIG.zh-CN.md) | `glm53.conf` 的全部设置项：默认值、由谁读取、修改何时生效 |
| [RUNTIME](packaging/RUNTIME.zh-CN.md) | 从归档或手动运行：所有命令、缓存、Claude Code、故障 |
| [T4](glm53/release/T4.zh-CN.md) | 四个节点：权重、设置、测量结果 |
| [RING](glm53/release/RING.zh-CN.md) | 不用交换机、连成环的四个节点：连线、fenced NCCL、测量结果 |
| [AGENT-API](glm53/release/AGENT-API.zh-CN.md) | 面向 AI 代理的控制 API，不需要仪表板 |
| [AI-GUIDE](glm53/release/AI-GUIDE.zh-CN.md) | 用一页说明整个系统，供帮助你理解或运行它的 AI 助手使用 |
| [ATLAS-TESTS](glm53/release/ATLAS-TESTS.zh-CN.md) | 专家重新排序的测量结果 |
| [weights](glm53/weights/README.zh-CN.md) | 从 Z.AI 官方的 BF16 检查点自行制作所有权重文件 |

## 功能

- 张量并行 8 或 4，KV 缓存在 rank 对之间切分（解码上下文并行 2）。引擎和模型的上下文长度最多为 1M（1,048,576 个 token），**八个节点和四个节点都能达到 1M**。八个节点：使用 4 bit KV 缓存（`KV=fp4`，`NEED_GB=100`）时为 `CONTEXT_LENGTH=1048576`、`KV_SIZE=2097152`（每个节点约余 23 GiB），使用 8 bit 缓存时为 `KV_SIZE=1638400`。四个节点：在控制面板中将 70 % 的路由专家设为 3 bit、其余设为 4 bit，可设 `CONTEXT_LENGTH=1048576`（`KV=fp4`，`KV_SIZE=1049088`，即上下文长度加 512，是 KV 池的最小值；`ROWS` 为 16,384 时每个节点约余 7.5 GiB，足以在运行中重排专家）。默认启动不论节点数都请求 `CONTEXT_LENGTH=1048576`，放不下时会缩短并说明。八个节点的 4 bit KV 缓存和四个节点通过控制面板的启动，各用一个约 1,019,000 个 token 的提示词实测。4 bit KV 缓存会略微降低精度（相同权重下 KL 为 0.0190，8 bit 缓存为 0.0156：[glm53/release/T4.zh-CN.md](glm53/release/T4.zh-CN.md)）；下文八个节点的速度使用 8 bit 缓存。设置示例中上下文长度为 819,200 和 1,048,576，KV 大小为 1,638,400 和 1,049,088（KV 大小即整个 KV 缓存，是所有对话共用的一个池：一个请求最多用到上下文长度，其余部分把之前对话的 KV 留在任意空闲的块中）。
- 长提示词按序列并行分段读入；解码步的集合通信走 RoCE。
- 所有对话共用一个对话缓存，以 256 个 token为一块：请求会从任意对话中已缓存的最长前缀接着处理；开头相同的对话（智能体的系统提示词、其子智能体）共享的块只保存一份。
- 每一轮由模型校验草稿：草稿默认来自 GLM-5.3 自带的 MTP 层，也可来自可选的 DFlash2 草稿模型。无论哪种情况，回复都由模型本身决定。
- 控制思考的长度：上限按 `max_tokens`（单次回复的上限）的比例设定，过长的思考可以在中途或结束时整理；既可以按请求设置，也可以在引擎运行中设置。
- 结构化输出：把回复的答案限定为 JSON 或 JSON Schema（`response_format`），开启或关闭思考都可以。
- `/metrics`（Prometheus 文本格式）和 `/health` 也统计正在运行的请求，因此监控程序在请求运行期间也能读取 prefill 与解码速度、缓存命中和草稿接受情况。设置 `METRICS_LISTEN` 后，这两个端点会在单独的地址上以只读方式开放，供其他主机上的监控程序使用。

## r3.3.3-102 的变化

- **结构化输出（默认开启）：** 回复的答案可以限定为 JSON 对象或 JSON Schema：在 `/v1/chat/completions` 上用 `response_format`（也读取 vLLM 的 `guided_json` 和 `structured_outputs`），在 `/v1/responses` 上用 `text.format`，在 `/v1/messages` 上用 `output_config.format`。模型在每个 token 处只从语法允许的 token 中选择，因此答案可以解析并符合 schema（被 `max_tokens` 截断时除外）。开启思考时，模型先自由思考。语法只能比所写更宽松地遵守的 schema 会以 400 拒绝，而不是只遵守一半；与工具一起或使用 `MULTI` 时也返回 400。语法由 llguidance 1.9.1（MIT）提供，包含在归档和镜像中；`STRUCTURED=0` 可将其关闭（[FEATURES](glm53/release/FEATURES.zh-CN.md#13-结构化输出)）。
- 结构化请求在没有 drafter 的情况下 decode，因此较慢。普通请求与 r3.3.2-102 相同：相同的 token、相同的速度。
- 从源代码构建还需要 Rust 1.95.0（“从源代码构建”）。

## r3.3.2-102 的变化

- **修复：** 值中含有 `</arg_value>` 或 `</tool_call>` 的 GLM 工具调用（值为代码，或为包含工具调用示例的文字时）会读取完整的值：只执行一个调用，文字中不会漏出任何内容；以前值会被截断，其中的示例可能被执行。以 DSML 标签开头但过短的工具调用块不再使服务器 panic 并使引擎崩溃。工具结果很多的长对话中，渲染聊天模板的内存不再随对话长度的平方增长：3,000 轮从 738 MB 降到 85 MB。
- **启动更快：** 所有 rank 同时启动，并在 1 秒内察觉引擎就绪；在交换机的八个节点上，整个启动从 115 秒缩短到 86 秒。
- **DFlash2 的默认值：** 使用 `MULTI` 时，与其他对话一起运行的对话从两个对话起使用 4 行的窗口（八个节点上写文章的四个对话 +12 %，四个节点上写代码慢 2 %），置信度为 0.6。默认的草稿模型 MTP 在同一测量中比两者都快（73.1 tok/s）。要恢复以前的行为，在 `EXTRA_ENV` 中写 `TF_GLM53_MULTI_DRAFT_ROWS=8 TF_GLM53_DFLASH_CONF=0.4`。

## r3.3.1-102 的变化

- **无事可做时引擎休眠。** 连续 1 秒无事可做时（`EXTRA_ENV` 中的 `TF_GLM53_IDLE_PARK_MS`），所有 rank 进入休眠：其他 rank 在与 rank 0 的连接上休眠，所有 RoCE 代理线程停止。请求、专家位宽的切换或停止到来时，它们在 0.6 至 2.0 ms 内被唤醒，短请求的首个 token 与以前一样快。在环上测得：休眠时每个节点的引擎占用不到一个核心的 1 %，GPU 为 0 %；不休眠时其他 rank 的 GPU 为 96 %。使用 `ENGINE_CORES=os` 时，这些节点的 SoC 温度从 64 至 85 °C 降到 50 至 54 °C。状态见 `/health` 的 `idle`，面板的 RDMA 栏显示为 idle（[CONFIG](glm53/release/CONFIG.zh-CN.md)）。
- **修复：有请求在排队或运行时停止引擎**在 r3.3-102 中需要 36 至 37 秒，并使其他 rank 的 GPU 在没有进程的情况下停留在工作状态（P0，约 11 W）。现在会结束这些请求并通知每个 rank：有负载时与空闲时一样 6 秒停止，所有 GPU 都回到空闲状态（P8）。

## r3.3-102 的变化

**需要你处理的变化：**

- **需要下载的权重变了：** `fiore-lab/GLM-5.3-EXL3-Base`（21.2 GB）和 3 bit bank（277.9 GB），四节点和八节点都一样。启动时所有路由专家都从 bank 读取，因此不再需要完整检查点；如果已有，用 `WEIGHTS` 指定即可使用。没有计划的启动以 3 bit 载入所有专家。`BITS` 或已保存的分配可以选择其他位宽，此时也需要相应的 bank。
- **设置改名：** `CONTEXT_LENGTH`（单个请求的最大长度）和 `KV_SIZE`（整个 KV 缓存）取代 `CONTEXT` 和 `STORE`。旧的 `glm53.conf` 仍可读取。
- **回复的默认上限**从 4,096 个 token 改为上下文长度的 70 %（`MAX_TOKENS_SHARE`）。客户端的 `max_tokens` 优先。
- **设置中的选项更少了：** `CACHE` 为 `pool`，`DCP` 为 2，`ROCE_PROXY_CORES` 为 `a725` 或 `os`。
- **`./tf-glm53 up` 只启动控制面板；** 引擎从面板启动，或用 `./tf-glm53 start` 启动。`./tf-glm53 engine-up` 手动启动引擎。

**新增：**

- **一行命令让引擎运行：** `docker run ... IMAGE up` 把发布内容写到第一个节点，找到节点，查找权重，启动引擎并发送一次请求。最后输出 `READY` 和地址，或输出 `STOP`、`FIX` 和 `THEN`，说明接下来要做什么。引擎设置界面中的每项设置都是一个 `-e NAME=VALUE` 选项（[QUICKSTART](glm53/release/QUICKSTART.zh-CN.md)）。
- **四个节点连成环**，不需要交换机（[RING](glm53/release/RING.zh-CN.md)）。
- **镜像包含带 AArch64 fence 的 NCCL：** 交换机和环都使用。这是 NVIDIA/nccl#2393 的修复，NCCL 的正式版本尚未包含。没有它时，Arm 上的代理线程可能在没有任何错误的情况下挂起（[RING](glm53/release/RING.zh-CN.md) 的“fenced NCCL”）。
- **节点检查**根据连线找到环或交换机上的节点，检查并登记（`glm53.nodes.conf`），因此不必填写 `TOPOLOGY`、`SOCKET_IFNAME` 和 `IB_HCA`。
- **Reasoning Control：** 思考的上限（`max_tokens` 的比例）、中途整理和思考结束时的整理，可以在引擎运行中通过面板或 `POST /v1/thinking` 更改（[CONFIG](glm53/release/CONFIG.zh-CN.md)）。
- **同时处理多个对话：** `MULTI`（1 至 4）现在也可以与 MTP 层一起使用。运行中可以通过面板或 `POST /v1/schedule` 设置 GPU 在 prefill 和解码之间的分配，短提示词也可以优先处理。
- **Byte Resume**（默认开启）：模型切分文本的位置与分词器不同时，下一轮提示词中相同的字节会对应其他 token id；现在缓存会越过该位置继续，不再重新计算其余部分。
- **面板：** 引擎设置界面、权重的位置及其检查（[ WEIGHTS ]）、启动流程、圆上的 KV 池、各节点的温度。
- **面向 AI 代理的控制 API：** 不需要仪表板即可启动、停止和设置引擎（[AGENT-API](glm53/release/AGENT-API.zh-CN.md)）。
- **更快、温度更低：** 权重并行载入（在环上测得，就绪时间从 366 s 缩短到 127 s）。草稿的置信度下降时，MTP 会停止该轮的草稿（`TF_GLM53_MTP_CONF`，默认 0.7）。RoCE 代理线程留在 Cortex-A725 核心上（`ROCE_PROXY_CORES`），相同速度下 SoC 温度降低 5 至 16 °C。

## r3.2-102 的变化

- 引擎运行时不向磁盘写入任何内容。每个 rank、监视进程和启动器的日志保存在每个节点上日志保管进程的内存中（`tools/logkeep.py`；每个节点都需要 `python3`，DGX OS 自带；归档和镜像都一样），每条保留最近的 `LOG_KEEP_MB`（8），用 `./tf-glm53 logs R` 读取。`RUNTIME=docker` 时容器也不保存日志。r3.1-102 只有在用户 linger 时才把日志放在内存中；r3.2-102 不需要 linger。
- 记录由另一台主机上的监控程序保存：设置 `LOG_LISTEN=HOST:PORT` 后，每个节点的保管进程以只读、无密钥的方式返回日志（`GET /streams`、`GET /logs`），指标监听（`METRICS_LISTEN`）也返回专家的使用次数（`GET /v1/experts/usage`）（[packaging/RUNTIME.zh-CN.md](packaging/RUNTIME.zh-CN.md)“为其他机器提供服务”）。

## r3.1-102 的变化

- API 和控制面板默认在所有地址（`HOST=0.0.0.0`、`LAUNCHER_HOST=0.0.0.0`）上监听，没有密钥：网络中的其他机器可以直接访问。请将节点留在封闭网络中（`HOST=127.0.0.1` / `LAUNCHER_HOST=127.0.0.1` 后只能从第一个节点访问）。
- 通过 Messages API 使用 Claude Code：`claude-glm53` 让 Claude Code 连接到服务器并启动，服务器把它的对话保留在缓存中、让长提示的流保持连接，并把图片换成注记（[packaging/RUNTIME.zh-CN.md](packaging/RUNTIME.zh-CN.md)“Claude Code”）。
- 控制面板也能用于由 `./tf-glm53 up` 启动的引擎（引擎默认统计专家的使用次数）；运行中的重新排序会先替换不会变大的层，并在开始前按替换过程中的内存峰值检查每个节点；内存不足的节点会在面板上标出。
- `glm53.conf` 的全部设置项：[glm53/release/CONFIG.zh-CN.md](glm53/release/CONFIG.zh-CN.md)。

## Expert Atlas（r3-102 新功能）

GLM-5.3 为每个 token 在每层 256 个路由专家中选出 8 个，而不同专家被选中的频率相差很大。Expert Atlas 让 19,200 个路由专家（第 3 至 77 层）各自以自己的位宽（2、3、4 或 5 bit）载入，在有限的内存中让使用最多的专家得到最多的位：

- **bank**：每种位宽的路由专家，位于 Hugging Face：[fiore-lab/GLM-5.3-EXL3-Experts-2bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-2bit)、[-3bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-3bit)、[-4bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-4bit) 和 [-5bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-5bit)。面板提供每个节点都具备的位宽。
- **控制面板**：由启动器（用 `./tf-glm53 up` 启动）提供的页面。启动前它规划位宽，并显示每个节点的内存、计划能否容纳以及计划留下的 KV 大小；也从这里启动和停止引擎。引擎运行时，它显示每个 token 选中的专家、被选中专家的平均位宽和速度。
- **使用记录**：引擎统计每个提示行和每个解码 token 选中的专家。保存的配置带有保存它的那些运行的使用记录，记录随配置一起移动：打开后再次运行，这次运行的部分会加入记录。计划按该记录为专家排序（没有记录的计划按 Fiore-lab 测得并随页面提供的次数排序）。不会自动保存任何内容：停止时为配置及其记录命名并保存。
- **专家优化**：保持每种位宽的专家数量不变，按本次运行的使用记录重新排列已载入的专家：相同内存下，模型实际选中的专家的平均位宽更高。在运行中的引擎上进行（在四个节点上，75 层的完整更改用了 83 至 90 秒，在请求之间进行）；需要时为结果命名并保存，供以后启动时打开。

控制面板可在四个或八个节点上运行，节点数在启动界面上选择。不使用面板时，`./tf-glm53 start` 和 `./tf-glm53 engine-up` 按上一次启动的计划启动，或以 3 bit 载入所有专家。详见 [PANEL-GUIDE](glm53/release/PANEL-GUIDE.zh-CN.md)、`packaging/RUNTIME.zh-CN.md`（“控制面板”）和 `glm53/release/T4.zh-CN.md`。

## Hugging Face 上的权重

| 内容 | 仓库 | 大小 | 用途 |
|---|---|---|---|
| Base 权重：路由专家以外的全部 | [fiore-lab/GLM-5.3-EXL3-Base](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Base) | 21.2 GB | **每次启动都必需**（四节点和八节点都一样） |
| 专家 bank，2 位 | [fiore-lab/GLM-5.3-EXL3-Experts-2bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-2bit) | 187.3 GB | 可选：使用 2 位专家时 |
| 专家 bank，3 位 | [fiore-lab/GLM-5.3-EXL3-Experts-3bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-3bit) | 277.9 GB | **每次启动都必需** |
| 专家 bank，4 位 | [fiore-lab/GLM-5.3-EXL3-Experts-4bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-4bit) | 368.5 GB | 可选：使用 4 位专家时（有八节点检查点时不需要） |
| 专家 bank，5 位 | [fiore-lab/GLM-5.3-EXL3-Experts-5bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-5bit) | 459.1 GB | 可选：使用 5 位专家时 |
| 八节点检查点：全部权重（路由专家为 4 位） | [fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks) | 390 GB | 可选：启动不需要。用 `WEIGHTS` 指定即可直接使用，并可作为 4 位专家 bank |
| 四节点检查点：全部权重（使用最少的一半路由专家为 3 位） | [fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks) | 344.6 GB | 可选：启动不需要。用 `WEIGHTS` 指定即可直接使用 |

专家 bank 是把 19,200 个路由专家统一为一种位宽的权重；引擎为每个专家从其计划位宽的 bank 中加载。
**每次启动都需要 Base 权重和 3 bit bank，四节点和八节点都是共 299.1 GB；其他 bank 只需下载你想用的位宽**（面板只提供每个节点都具备的位宽）。
每个节点都需要在相同路径下放置相同的文件。在一台机器上下载后复制到其他节点（`python3 -m venv` 需要 `python3-venv` 软件包），例如：

```sh
python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install -U huggingface_hub
~/hf-cli/bin/hf download fiore-lab/GLM-5.3-EXL3-Base --local-dir ~/models/GLM-5.3-EXL3-Base
~/hf-cli/bin/hf download fiore-lab/GLM-5.3-EXL3-Experts-3bit --local-dir ~/models/GLM-5.3-EXL3-Experts-3bit
```

复制到其他节点的方法见 [QUICKSTART](glm53/release/QUICKSTART.zh-CN.md)，校验（`./tf-glm53 verify`、`./tf-glm53 verify-banks`）见 [packaging/RUNTIME.zh-CN.md](packaging/RUNTIME.zh-CN.md)。

## 自己制作权重

权重不能像代码那样阅读：被改动过、在某些情况下表现不同的权重，看起来与其他权重没有区别，因此你可能不愿依赖无法核实其制作过程的
权重。你不必依赖 Fiore-lab 的权重。本方案加载的每个权重文件（Base 权重、八节点和四节点检查点，以及 2、3、4、5 位的专家 bank）都可以用
[glm53/weights](glm53/weights/README.zh-CN.md) 中的工具，从 Z.AI 官方的 BF16 检查点（[zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16)）
制作；步骤、硬件和实测时间都在那里。制作出的文件与 Fiore-lab 的布局相同，加载方式也相同。

## 所需环境

- 四台或八台 DGX Spark 系统（GB10），位于同一 RoCE v2 网络上，或四台连成环。第一台通过 ssh 访问每个节点，包括它自己，使用没有口令的密钥，并已登记它们的 host key。ssh 和 ConnectX-7 网络的步骤见 [QUICKSTART](glm53/release/QUICKSTART.zh-CN.md) 第 1 节。四个节点接在交换机上时，每个节点一个 100 Gb/s 端口即可。
- 使用镜像时：每个节点上有 NVIDIA 驱动 580 或更高版本（镜像使用主机的驱动）、Docker 和 NVIDIA Container Toolkit（DGX OS 两者都有）以及 `python3`，并且启动前每个节点约有 115 GiB 空闲内存。
- 改用归档时：DGX OS（Ubuntu 24.04），NVIDIA 驱动 580 或更高版本，以及 cuBLAS 13；每个节点上都有 NCCL 2.30.7：即 NVIDIA 的 `nvidia-nccl-cu13` wheel，由 pip 解包（见 `packaging/RUNTIME.zh-CN.md`）。
- 使用归档时，带 AArch64 fence 的 NCCL 在交换机上推荐使用，在环上必须使用；NVIDIA 的 wheel 没有这个 fence（[RING](glm53/release/RING.zh-CN.md)）。镜像已包含它。
- Fiore-lab 的 EXL3 Base 权重（Hugging Face），在每个节点上放在相同路径下：[fiore-lab/GLM-5.3-EXL3-Base](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Base)（21.2 GB），四节点和八节点都一样。也可以用完整检查点代替，即 [fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks)（390 GB）或 [fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks)（344.6 GB）；启动时不读取其路由专家。
- 所有权重文件也都可以不使用 Fiore-lab 的任何文件，从 Z.AI 官方的 BF16 检查点自行制作（[glm53/weights](glm53/weights/README.zh-CN.md)）。
- 3 bit bank（277.9 GB），在每个节点上放在相同路径下：每次启动都需要它。其他 bank 只需要用到的位宽（八节点检查点可以作为 4 bit bank；只有 4 bit 时上下文长度会更短）。
- 各节点上的 `python3`，供启动器和日志保管进程使用。

## 运行

发布以容器镜像和内容相同的归档两种形式提供；推荐使用镜像。权重就位后，在第一个节点上运行一行命令即可启动引擎：

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host ghcr.io/fiore-labo/expertatlas-glm5.3:r3.3.3-102 up
```

最后输出 `READY` 和地址：API 在 `http://FIRST-NODE:8000/v1`，控制面板在 `http://FIRST-NODE:8010/panel`。如果缺少什么，则以 `STOP` 结束，并说明要做什么以及接下来要运行的一行。步骤和所有选项见 [QUICKSTART](glm53/release/QUICKSTART.zh-CN.md)。

手动完成同样的事：使用镜像时，在每个节点上 `docker pull` 它，运行 `mkdir glm53 && docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/glm53":/out IMAGE install /out`，并在 `glm53.conf` 中设置 `RUNTIME=docker` 和 `IMAGE`（[packaging/RUNTIME.zh-CN.md](packaging/RUNTIME.zh-CN.md)“在容器中运行”）。不使用 Docker 时：下载发布归档 `tensorfold-r3.3.3-102-linux-aarch64.tar.gz`，并按照其中的 `RUNTIME.md`（本仓库中为 [packaging/RUNTIME.zh-CN.md](packaging/RUNTIME.zh-CN.md)）操作。两种方式都要将 `glm53.conf.example`（八个节点）或 `glm53.t4.conf.example`（四个节点，见 [glm53/release/T4.zh-CN.md](glm53/release/T4.zh-CN.md)）复制为 `glm53.conf`，填写其中要求的内容，然后执行 `./tf-glm53 deploy` 和 `./tf-glm53 up`（启动控制面板），再在那里启动引擎。`./tf-glm53 start` 按上一次启动的计划启动，`./tf-glm53 down` 停止。

服务器默认在所有地址（`HOST=0.0.0.0`）上提供 OpenAI chat API，以及供 Claude Code 使用的 Anthropic Messages API。API 和控制面板都没有密钥，任何能访问到它们的人都可以使用：请把节点放在封闭网络中，或在其前面加一层 TLS 和访问控制（设置 `HOST=127.0.0.1` 和 `LAUNCHER_HOST=127.0.0.1` 后只能从第一个节点访问）。

## 草稿模型

- `mtp`（默认）：权重中的 MTP 层，属于 GLM-5.3 权重的一部分。解码速度约为不使用草稿时的两倍，预填充（prefill）速度相同。
- `dflash2`：incoai/GLM-5.3-DFlash2。**由于其许可证，本仓库和归档均不包含它**：其模型卡注明的许可证为 CC BY-NC-ND 4.0（非商业性使用，禁止演绎），用于研究和评估；商业许可由 Inco AI 提供。如果你的使用在该许可证允许的范围内，请按照 [packaging/RUNTIME.zh-CN.md](packaging/RUNTIME.zh-CN.md#使用-dflash2-的步骤) 中的“使用 DFlash2 的步骤”操作：以固定的 revision 下载一次，复制到每个节点的相同路径并核对副本，然后设置 `DRAFTER_KIND=dflash2` 和 `DRAFTER`。引擎在启动时于内存中将其量化为 4 bit，不向磁盘写入任何内容。解码会更快（见“速度”）。
- `none`：不使用草稿。

由 `glm53.conf` 中的 `DRAFTER_KIND` 选择；在启动界面上选择会写入那里；`./tf-glm53 -d KIND start`（或 `engine-up`）只为这一次启动选择。

## 速度

**使用 r2-102 版本测量，测量时每个节点的 GPU 时钟上限均设为 2.1 GHz（2,100 MHz）。** 此后引擎已有变化，这些数字没有针对本版本重新测量。使用尚未缓存的编程提示词，一次一个请求（reasoning effort 为 low，回复最多 4,000 个 token），采用 `glm53.conf.example`（八个节点，FP8 KV 缓存）和 `glm53.t4.conf.example`（四个节点，4 bit KV 缓存；`ROWS` 当时为 16,384，现在为 8,192）的设置，以及 Fiore-lab 公开发布的检查点，各用其自身的位宽（八个节点所有路由专家为 4 bit，四个节点一半为 3 bit、一半为 4 bit）。默认启动以 3 bit 载入所有专家，这些数字没有测量该配置。速度因环境而异（检查点、提示词、时钟等），这些数字仅供参考。完整的测量条件见 `packaging/RUNTIME.zh-CN.md`。

| 节点数 | 提示词 | MTP（k=3）：首 token 时间 / 解码 | DFlash2：首 token 时间 / 解码 |
|---|---|---|---|
| 8 | 128K | 82.8 s / 47.5 tokens/s | 85.2 s / 53.2 |
| 8 | 256K | 172.9 s / 45.7 | 177.6 s / 52.6 |
| 8 | 512K | 376.5 s / 42.8 | 386.0 s / 47.1 |
| 4 | 128K | 120.1 s / 35.3 | 121.9 s / 42.9 |
| 4 | 256K | 253.3 s / 33.2 | 256.9 s / 40.7 |
| 4 | 512K | 565.4 s / 32.2 | 571.5 s / 37.8 |

本发布版在四个节点上按 `ROWS` 各测一次的 prefill：8,192 时 1,054 token/s，16,384 时 1,102，32,768 时 1,110。仅供参考，条件见 [glm53/release/T4.zh-CN.md](glm53/release/T4.zh-CN.md)。

## 精度

本方案的输出分布相对于原始 BF16 模型输出分布的 KL 散度（原始模型指以官方实现运行官方 BF16 权重、以 FP32 计算），使用本方案的服务配置：

- 八个节点（八节点检查点，所有路由专家为 4 bit，FP8 KV 缓存，DCP 2；用本引擎的较早版本在相同权重上测得），四段 2,048 token 的文本（对话、法律、散文、推理），每段取最后 1,023 个 token，运行五次：平均 0.0133（0.0128 至 0.0146）；top-1 token 的一致率约为 96%。
- 八个节点，四段 8,192 token 的文本（两段英文、两段代码），按token 区间：0.012（token 1,024 至 2,047）、0.018、0.019 和 0.013（6,144 至 8,190）。作为参照，官方 BF16 权重若以 BF16 而非 FP32 计算，在相同区间上与同一基准的差异为 0.007 至 0.010。
- 四个节点（四节点检查点自身的位宽，一半为 3 bit、一半为 4 bit，4 bit KV 缓存，DCP 2），四段 8,192 token 的文本，按相同的token 区间：0.019（token 1,024 至 2,047）、0.032、0.029 和 0.019（6,144 至 8,190）。
- Expert Atlas 的各位宽（所有路由专家为同一位宽，其他权重同上，相同的四段文本，token 6,144 至 8,190）：2 bit 0.0517、3 bit 0.0203、4 bit 0.0127（四个节点），5 bit 0.0114（八个节点）。默认启动以 3 bit 载入所有专家。在相同内存下，含 2 bit 专家的配置比不含的配置离基准远 13% 至 30%；把其他专家降到 3 bit 来换取 5 bit 专家，没有可测出的改善。

## 从源代码构建

在一台 GB10 系统上，需要 `PATH` 中有 Zig 0.17.0（或为 `build_libs.sh` 用 `ZIG=` 指定其路径）、CUDA 13（`nvcc`）、libibverbs 头文件（`libibverbs-dev`）以及通过 rustup 安装的 Rust 1.95.0（`rustup toolchain install 1.95.0`；`build_libs.sh` 使用 `~/.cargo/bin/cargo`，或 `CARGO=` 指定的那个）：

```sh
zig build fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 --prefix /tmp/fatbins
sh glm53/tools/release/build_libs.sh /tmp/lib        # lib/libroce_proxy.so, lib/libllguidance.so
zig build dist -Dversion=r3.3.3-102 -Ddist-commit=$(git rev-parse HEAD) -Ddist-platforms=linux-aarch64 \
    -Ddist-fatbins=/tmp/fatbins/fatbin -Ddist-lib=/tmp/lib -Ddist-extra=glm53/release
```

归档会生成在 `zig-out/dist/` 中。`build_libs.sh` 从 `third_party/llguidance/` 及其中附带的 crate 离线构建 llguidance。只使用镜像或归档则不需要 Rust。

## 限制

- 支持四台或八台 GB10 系统。两台和三台的配置正在测试中。不支持其他 GPU，也不支持其他 rank 数量。
- 默认一次只运行一个请求；其他请求最多 32 个按到达顺序等待（聊天标题生成请求排在最后），超过 32 个时服务器会拒绝（返回 503，流式请求则为错误事件）。`MULTI` 可同时解码最多四个请求，但要占用节点内存。除此之外，对于并行运行子智能体的智能体，最好让子智能体改用另一个模型。
- 结构化输出只支持 JSON 对象和 JSON Schema（尚不支持正则表达式、选项和语法），不能与工具一起使用，也不能在 `MULTI` 下使用。
- 控制面板的界面只有英文和日文。
- 其余限制见 `packaging/RUNTIME.zh-CN.md`。

## 许可证与致谢

代码采用 Apache-2.0 许可（`LICENSE`）；`NOTICE` 和 `THIRD_PARTY_NOTICES.md` 说明了各部分的来源。权重保留其自身的许可证（GLM-5.3 License）；可选的 DFlash2 草稿模型也保留其自身的许可证。

- [TensorFold](https://github.com/ashhart/TensorFold)，由 TensorFold 贡献者开发：本方案所扩展的服务器、构建系统和 CUDA 引擎。
- drowzeys 为 TensorFold 编写的完整版 GLM-5.3 Python 引擎（[ashhart/TensorFold#159](https://github.com/ashhart/TensorFold/pull/159)）：GLM-5.3 引擎用 Zig 和纯 CUDA 重写了它的设计（引擎结构、Triton 内核以及 MTP 和 DFlash2 草稿模型的处理流程）。其中部分 CUDA 文件被收录，其中几个经过了修改。
- [b12x](https://github.com/local-inference-lab/b12x)：将其 RoCE one-shot 集合通信用纯 CUDA 重写（`roce.cu`）；收录了其 RoCE 代理（`zig/native/roce_proxy.c`），并修改为按每个 RDMA 设备选择 GID 索引。
- [MiaAI-Lab 的 GLM-5.3-Flash on 2x DGX Spark](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)：参考其提示词处理内核编写了 GLM-5.3 的提示词处理内核。
- [MiaAI-Lab 的 GLM-5.3 EXL3 on 3x DGX Spark](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold)：4 bit KV 缓存采用其布局，并用本引擎自己的代码实现。
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3)：检查点用它量化，引擎读取其 EXL3 格式。
- [gr8 方案](https://github.com/knapcio/GLM-5.3-gr8-4x-DGX-Spark-TP4)和[引擎](https://github.com/kindlingai/gr8)：若干解码加速借鉴了其思路，并用本引擎自己的代码实现。
- [ollama/ollama#12375](https://github.com/ollama/ollama/issues/12375)：Byte Resume 的思路来自这份报告。

## TensorFold 的贡献者

摘自 TensorFold 的 README：感谢每一位向 TensorFold 提交过 pull request、测量数据或错误报告的人（名单截至 TensorFold 1.0.2；最新名单请见 [TensorFold 仓库](https://github.com/ashhart/TensorFold)）：

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
