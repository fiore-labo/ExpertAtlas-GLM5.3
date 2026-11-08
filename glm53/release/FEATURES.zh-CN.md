[English](FEATURES.md) | [日本語](FEATURES.ja.md) | 简体中文

# Expert Atlas：功能说明

本页逐一说明 Expert Atlas 的功能：它做什么、为什么需要它、如何使用。界面见 `SCREENS.zh-CN.md`，每个控件见 `PANEL-GUIDE.zh-CN.md`，
每项设置见 `CONFIG.zh-CN.md`。面板的界面语言只有英文和日文，本页保留界面上的英文名称。

下面的实测数字引自旁边注明的发布文档，条件与该文档相同。速度和内存因节点、prompt 和设置而异，请作为参考。

## 目录

1. [环与交换机，以及节点检查](#1-环与交换机以及节点检查)
2. [2、3、4、5 bit 的专家](#2-2345-bit-的专家)
3. [位宽分配](#3-位宽分配)
4. [保存和打开分配](#4-保存和打开分配)
5. [专家优化](#5-专家优化)
6. [上下文长度、KV 大小与 KV pool](#6-上下文长度kv-大小与-kv-pool)
7. [并发推理与 GPU 分配](#7-并发推理与-gpu-分配)
8. [Drafter：MTP 与 DFlash2](#8-draftermtp-与-dflash2)
9. [思考控制](#9-思考控制)
10. [健康检查](#10-健康检查)
11. [引擎核心与 RDMA 行](#11-引擎核心与-rdma-行)
12. [显示设置](#12-显示设置)
13. [结构化输出](#13-结构化输出)
14. [确定性模式](#14-确定性模式)

## 1. 环与交换机，以及节点检查

**做什么。** Expert Atlas 运行在连接到同一台 RoCE v2 交换机的 4 台或 8 台 DGX Spark 上，也可以运行在不用交换机、彼此连成环的 4 台上。
节点检查会判断你使用的是哪一种，检查每个节点并注册。

**为什么。** 环不需要交换机：每个节点的两个 ConnectX-7 端口直接连到两侧的邻居。两种方式下引擎、权重和面板都相同。
在环上，每个节点与对面的节点之间没有直连，发往对面节点的数据由邻居转发。四个节点上的实测（`RING.zh-CN.md`）：首个 token 比交换机晚
4 至 5 %，decode 一样快或略快，精度相同。

**如何使用。**

- 尚未注册节点时，面板在节点检查界面打开。按一次按钮：它会判断连线，在每个节点上运行 `tf-glm53 check`（ssh、GPU、内存、rendezvous、RDMA、
  NCCL、文件），在缺少软件包的节点上复制软件包，并把节点写入 `glm53.nodes.conf`。
- **环**：head 通过 LLDP 追踪连线，因此每个节点都需要 `lldpd`。把每个节点的端口 0 连到下一个节点的端口 1。环还会测量每条通道的带宽。
- **交换机**：head 在自己的 ConnectX-7 /24 上尝试 ssh，并列出响应的节点。选择节点数并勾选要使用的节点。交换机上有其他人的节点时，请先选择再按按钮。
- `docker run ... IMAGE up` 在命令行上做同样的检查（`QUICKSTART.zh-CN.md`）。
- `glm53.nodes.conf` 保存 `TOPOLOGY`、`NODES`、`SOCKET_IFNAME` 和 `IB_HCA`，其中的行优先于 `glm53.conf`。

## 2. 2、3、4、5 bit 的专家

**做什么。** 模型有 19,200 个路由专家（第 3 至 77 层，每层 256 个）。每个专家都可以以 2、3、4 或 5 bit 载入，来源是四个 bank：
每种位宽一个目录，在每个节点上路径相同。

**为什么。** 位宽越宽越接近官方模型，但占用的内存越多，留给上下文和 KV 缓存的就越少。你可以按自己的用途选择平衡点。

| 位宽 | bank 大小 | KL 散度，全部专家为该位宽（`AI-GUIDE.zh-CN.md`） |
|---|---|---|
| 2 bit | 187.3 GB | 0.0517 |
| 3 bit | 277.9 GB | 0.0203 |
| 4 bit | 368.5 GB | 0.0127 |
| 5 bit | 459.1 GB | 0.0114 |

（与官方 BF16 模型的 KL 散度，位置 6,144 至 8,190。越低越接近。）

四个节点上（`T4.zh-CN.md`，4 bit KV 缓存，计划请求的上下文长度最多 737,280）：全部 3 bit 以及一半 3 bit、一半 4 bit 都能放下所请求的 737,280。
全部 4 bit 能放下 327,680。全部 5 bit 放不下。在相同内存下，混入 2 bit 专家的组合比不混入的组合离参考远 13 至 30 %。
把其他专家降到 3 bit 来换取 5 bit 专家，没有可测量的收益。

**如何使用。**

- Base 权重和 3 bit 的 bank 始终必需。2、4、5 bit 的 bank 可选：没有 bank 的位宽不会提供。
- 在启动设置的 [ WEIGHTS ] 中，或用 `BANK_2` 至 `BANK_5` 设置目录（`QUICKSTART.zh-CN.md`）。
- 在 `docker run` 命令中加 `-e BITS=N`，所有专家以同一位宽启动。面板可以为每个专家分别设定位宽（下一节）。

## 3. 位宽分配

**做什么。** 启动界面的 [ BIT ALLOC ] 设定每种位宽在专家中所占的比例。使用最多的专家得到最宽的位宽。

**为什么。** 有些专家被选中的次数远多于其他专家。把多出的位宽给模型最常用的专家，可以在相同内存下提高实际被选中专家的平均位宽。

**如何使用。**

- 输入 5、4、3 bit 的比例。有数据的最窄位宽占其余部分，因此比例之和为 100 %。
- 专家按使用记录（每个专家被选中的次数）排序。排序时，专家的 decode 占比权重为 0.7，prefill 占比权重为 0.3。这组权重最多地保留了模型所写的文字（`ATLAS-TESTS.zh-CN.md`）。
- 没有自身使用记录的计划按页面内置的次数排序：为本发布测得的 12 类工作（编程、代理、日文和英文文本、Linux 运维、安全）。
- Count（decode、prefill 或两者）只改变页面的显示，计划的位宽不变。
- 每个节点的内存和判定（Can start、Not enough memory）会随输入而变化。

## 4. 保存和打开分配

**做什么。** 分配是所有专家的位宽加上使用记录。你可以以一个名称保存它，以后再打开。

**为什么。** 适合你工作的计划，或专家优化的结果，可以在之后的启动中再次使用。使用记录随分配一起移动，因此之后的运行会继续累加。

**如何使用。**

- 不会自动保存任何内容。
- 运行面板 [ BIT ALLOC ] 中的 **Save** 可随时保存。**Save and stop** 在停止引擎时保存。
- 启动界面的 **Open** 列出上一次运行的分配和已保存的分配。打开的分配按位原样成为计划，启动时使用它。
- 已保存的分配可以 **Rename** 和 **Delete**。删除的或被覆盖的旧版本进入启动器的回收站。
- 文件保存在 head 节点上发布目录的 `atlas/allocs/user/<name>.json` 中（`CONFIG.zh-CN.md` 中的 `ATLAS_DIR`）。
- 命令行：在 `docker run` 命令中加 `-e ALLOC=NAME`，或用 `./tf-glm53 start` 以上一次启动的计划启动（`QUICKSTART.zh-CN.md`，第 6 节）。

## 5. 专家优化

**做什么。** 引擎运行时，Optimise experts 按本次运行的使用记录（运行开始时的记录加上本次运行的选择）重新排列已载入的专家。
每种位宽的专家数量不变，因此内存不变。

**为什么。** 内置的次数只是一般性的估计。你自己的工作会选中自己的一组专家。重新排列后，你最常用的专家得到最宽的位宽。
四个节点、一半 3 bit 一半 4 bit 时的实测（`AI-GUIDE.zh-CN.md`、`ATLAS-TESTS.zh-CN.md`）：与全 4 bit 模型的 KL 散度，在回复的 token 上，
8,192 token 时下降 24 %，128K 和 256K token 时下降 32 至 34 %。对次数中没有包含的工作类型，收益较小（约三分之二）。

**如何使用。**

- 按 [ BIT ALLOC ] 中的 **Optimise experts**。估算显示会改变的专家数、推理大约停止多久，以及每个节点是否有交换所需的内存。
- 按 **Apply**。没有请求在运行或等待时开始更改。期间到达的请求会等待。
- 在四个节点上，完整更改 75 层用了 83 至 90 秒。
- KV pool 中保留的对话会被丢弃，因为它们的 KV 是用旧专家计算的。
- 每个节点都需要能容纳最大一层的新副本外加 2 GiB 的空间。若有节点不足，面板会拒绝更改并指出该节点。
- 结果不会自动保存：请按 **Save**。启动只载入给定的位宽，从不重新排列。
- 并发推理时也同样有效。

## 6. 上下文长度、KV 大小与 KV pool

**做什么。** KV 缓存是所有对话共享的一个池。之前的对话保留在其中，因此下一轮不必重新计算 prompt。

- **Context length**（`CONTEXT_LENGTH`）：最长的请求，即 prompt 与回复之和。最多 1,048,576。
- **KV size**（`KV_SIZE`）：整个池容纳的 token 数，至少为上下文长度 + 512。其余部分容纳之前的对话。

**为什么。** 代理和聊天会一次又一次地发送相同的长开头，而计算长 prompt 需要几分钟（四个节点上 128K token 约 120 秒，`T4.zh-CN.md`）。
从缓存读取则几乎不花时间。四个节点上的实测（`T4.zh-CN.md`）：再次发送一个 703,916 token 的对话，从池中恢复了 703,899 个位置，首个 token 不到 3 秒。

**工作方式**（`RUNTIME.zh-CN.md`）。

- 池由 256 个位置的块组成。请求从任一对话中最长的已缓存相同开头处恢复。
- 开头相同的对话（相同的 system prompt，或子代理）只持有一份共同的块。
- 最多保留 64 个对话。池满时，最久未使用的对话先从末尾交出块。
- 每个块在重用前都会与请求的 token 核对。只要有疑问，该请求就从头计算（cold）。
- 专家优化和重启会丢弃对话。
- Byte Resume（`TF_GLM53_BYTE_RESUME`，默认开启）在模型对文本的切分与分词器不同时也能保留缓存。你无需做任何事。

**如何使用。**

- 在启动界面的 [ NODE ] 中设定上下文长度、KV 大小和 KV 格式。默认情况下，KV 使用计划剩下的全部空间。
- **KV 格式**：FP4 每个位置占用的内存比 FP8 少（每 100,000 个位置 1.73 GiB 对 2.55 GiB，`RUNTIME.zh-CN.md`），精度略低（KL 散度增加 0.003 至 0.005，`T4.zh-CN.md`）。
- **Keep for the OS**（`RESERVE_GB`）：为操作系统保留得越多，留给 KV 的就越少。
- 运行面板的 KV POOL 显示池的状态：使用比例、保留的对话，以及每个 prompt 有多少来自缓存（`SCREENS.zh-CN.md`）。

## 7. 并发推理与 GPU 分配

**做什么。** 默认情况下，引擎一次处理一个请求。最多再有 32 个请求等待，超过的会以繁忙为由拒绝。启用并发推理（`MULTI`，面板上为 1 至 4）后，
引擎最多同时保持这么多对话，并一起 decode。

**为什么。** 这样可以同时服务多个用户，或带子代理的代理，而不必轮流处理。它会占用节点内存：在环上，上下文长度 131,072、为操作系统保留 16 GiB 时
（`BEFORE-START.zh-CN.md`，首次启动的估算），KV 从使用 MTP 时的 753,664 token 降到使用 DFlash2 和 `MULTI=4` 时的 360,448。

**如何使用。**

- 在启动界面选择 **Concurrent** 1 至 4（1 表示一次一个），或在 `docker run` 命令中加 `-e MULTI=N`。可与 MTP、DFlash2 或不使用 drafter 一起使用。3 个节点上尚不可用。
- `MULTI_WAIT`（默认 600 秒）：请求等待空间的最长时间。超过后以繁忙为由拒绝，客户端会重试。
- 运行时，[ NODE ] 显示 **Concurrent** 和三个控件：
  - **GPU split**：在 decode 中的对话旁边计算 prompt 时 GPU 时间如何分配。prefill 100 % 先计算 prompt，decode 100 % 让回复先进行。启动时的值为 decode 30 %（`MULTI_DECODE_SHARE`）。
  - **Priority prompt**：新 token 不超过此数（默认 8,192）的 prompt 排在较长的 prompt 之前。只计算缓存之后的部分，因此长对话的下一轮只计其新的部分。
  - **Overtake a running prefill too**（默认关闭）：正在运行的长 prefill 也会让位给优先 prompt，之后从已完成的块继续。
- 在运行面板上的更改持续到引擎停止。要在之后的启动中保留，请在引擎设置中设定。

## 8. Drafter：MTP 与 DFlash2

**做什么。** drafter 提出接下来的 token，由模型核对。被接受的 token 一次性输出，因此 decode 更快。回复始终是模型自己的。

**为什么。** 为了 decode 速度。用发布 r2-102 实测（`RUNTIME.zh-CN.md`，本发布未重新测量）：四个节点、128K prompt 时，decode 使用 MTP 为每秒 35.3 token，
使用 DFlash2 为 42.9；八个节点时为 47.5 和 53.2。

**如何使用。**

- **MTP**（默认）：权重中已有的 MTP 层。无需下载。
- **DFlash2**：单独的模型 `incoai/GLM-5.3-DFlash2`，有自己的许可证。它不随发布提供：请下载到每个节点并设置其目录（`DRAFTER`，或 [ WEIGHTS ] 中的 drafter 行）。
  在每个节点都有之前，面板中 DFlash2 呈灰色。它每个节点多占约 3 GiB（四个节点）或约 4 GiB（八个节点）（启动器的规划值），因此 KV 更小。
- **None**：不使用 drafter。
- 在启动界面的 **Drafter** 中选择，或用 `-e DRAFTER_KIND=mtp|dflash2|none`。

## 9. 思考控制

**做什么。** GLM-5.3 在回答之前先思考，`max_tokens` 也计入思考。[ REASONING CONTROL ] 设定回复中思考可以使用多少，以及何时整理思考。

**为什么。** 思考过长可能用尽回复的 max_tokens，没有留给答案的空间；很长的思考还会占满 KV。这些设置为答案留出空间，并使思考的 KV 保持有界。客户端仍会收到全部思考内容。

**如何使用。** 在运行面板上更改后，对之后开始的请求生效，并写入 `glm53.conf`。请求自身的设置优先。

- **Thinking limit**（`THINKING_SHARE`）：每个回复 max_tokens 的 5 % 至 100 %（No limit）。达到上限时，引擎在换行处结束思考，模型用剩余部分回答。
- **Organize midway**（`THINKING_FOLD`）：模型写出思考的检查点（目前的结论、已排除的内容、尚未确定的内容、接下来要查看的内容）。引擎用检查点替换上下文中的思考。
  这时上限按写出的 token 计数，因此可能超过上下文长度。
- **Organize every**（`THINKING_FOLD_SHARE`）：以 max_tokens 的比例表示多久整理一次（至少 1,024 token）。100 % 表示只在达到上限时整理。
- **Organize at the close**（`THINKING_FOLD_AT_CLOSE_SHARE`）：模型结束思考时，先整理思考。长的推理变成答案之前的计划，之后的轮次只得到整理后的文字。
- **Notice of what is left**（`THINKING_NOTICE`，仅在引擎设置中，需要重启）：每隔一定数量的思考 token，引擎告诉模型已用了多少、还剩多少，让模型自己掌握节奏。
- 在思考预算接近用尽时，引擎还会让模型重写其思考（`THINKING_FOLD_AT_BUDGET`，默认开启）。因此回复写出的内容可能超过上限。

引擎设置界面可以设定相同的值以及与之相关的大小（`CONFIG.zh-CN.md`）。

## 10. 健康检查

**做什么。** [ NODE ] 中的 **Health check** 向运行中的引擎发送一个约 1,000 token 的请求并判定回复。正确的答案只能通过读取该 prompt 得出。

**为什么。** 为了确认运行中的引擎能读取 prompt 并正确回答，而不仅仅是在运行。

**如何使用。** 按 **Health check**。回复正确时，圆的上方显示 **Inference OK** 几秒钟。**Inference error** 会连同原因留在按钮下方。
通常约 3 秒完成；排在长请求之后时会等待，最多 600 秒。

## 11. 引擎核心与 RDMA 行

**做什么。** GB10 的 CPU 有两种核心：Cortex-X925 和 Cortex-A725。两个设置决定引擎的线程在哪里运行。

- **RDMA cores**（`ROCE_PROXY_CORES`，默认 A725）：RoCE 代理线程，每个 rank 两个，在 decode 期间持续轮询。A725 让它们在核心 0 和 10 上运行。
  OS 交由调度器决定，调度器常把它们放在 X925 核心上。
- **Engine cores**（`ENGINE_CORES`，默认 OS）：OS 让每个 rank 使用所有核心。A725 只让各 rank 使用 A725 核心。

**为什么。** 持续轮询的线程在快速的 X925 核心上运行会使芯片发热。在四个节点上，把代理线程放在 A725 上，decode 速度不变，芯片温度低 5 至 16 °C（`CONFIG.zh-CN.md`）。
引擎只在 A725 上运行时的速度尚未测量。

**如何使用。** 两者都在引擎设置（Nodes and parallelism）中设定，在下一次启动时生效。引擎运行时，[ NODE ] 显示结果：RDMA 列列出每个节点两个代理线程的 CPU（例如 0·10），
表格下方的行显示 RDMA proxy threads 和 Engine cores。

## 12. 显示设置

**做什么。** **Display** 抽屉改变面板的外观：语言、圆的排列顺序、余辉和亮度、光的颜色、外圈的文字、数字、中央以及配色。它不改变引擎。

**为什么。** 让面板在你的屏幕上更易读。设置保存在通道上而不是浏览器中，因此每个打开面板的人看到的都相同。

**如何使用。**

- 设置保存在 head 节点上软件包目录的 `launch/panel-settings.json` 中（`EXTRA_ENV` 中的 `TF_PANEL_SETTINGS` 可以改变位置）。该通道的每个浏览器显示相同的面板。
- 节点检查的节点列表和勾选也保存在同一文件中。
- **Back to defaults**（按两次）为该通道的所有浏览器清除显示设置。语言和节点保持不变。
- 环和交换机通道在其设置未指定其他配色时，以按位宽着色的光和 Material · indigo 配色打开。

## 13. 结构化输出

**做什么。** 把回复的答案限定为 JSON：一个 JSON 对象，或符合你发送的 JSON Schema 的 JSON。引擎在每个 token 处只让模型从能使答案保持该形式的
token 中选择，因此答案可以解析并符合 schema（除非先被 `max_tokens` 截断）。语法由 llguidance 1.9.1 提供，在第一个节点上运行。默认开启（`STRUCTURED`）。

**为什么。** 读取回复的程序（信息提取、表单、智能体的一步等）可以直接得到能解析、带有预期的键和类型的 JSON，无需再次询问模型。

**如何使用。** 在请求中指定：

- `/v1/chat/completions`（OpenAI 的 chat）：JSON 对象用 `"response_format": {"type": "json_object"}`，符合 schema 的 JSON 用
  `{"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}`。也读取 vLLM 的字段：`"guided_json": SCHEMA`，以及
  `"structured_outputs"` 的 `{"json": SCHEMA}` 或 `{"json_object": true}`。`{"type": "text"}` 表示不加约束。
- `/v1/responses`（OpenAI 的 Responses API）：`"text": {"format": {"type": "json_schema", "name": "...", "schema": {...}}}`，
  或 `{"type": "json_object"}`。
- `/v1/messages`（Anthropic 的 Messages API）：`"output_config": {"format": {"type": "json_schema", "schema": {...}}}`。

在 chat API 上关闭思考的例子：

```sh
curl -s http://NODE:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3", "max_tokens": 300,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": "A city in Japan and its prefecture."}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": {
    "type": "object",
    "properties": {"city": {"type": "string"}, "prefecture": {"type": "string"}},
    "required": ["city", "prefecture"]}}}}'
```

使用 Python 的 `openai` 包并开启思考的例子：

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://NODE:8000/v1", api_key="none")
r = client.chat.completions.create(
    model="glm-5.3", max_tokens=4000,
    messages=[{"role": "user", "content": "Three prime numbers above 100."}],
    response_format={"type": "json_schema", "json_schema": {"name": "primes", "schema": {
        "type": "object",
        "properties": {"primes": {"type": "array", "items": {"type": "integer"},
                                  "minItems": 3, "maxItems": 3}},
        "required": ["primes"]}}})
choice = r.choices[0]
if choice.finish_reason == "length":
    raise SystemExit("cut by max_tokens: the JSON is not finished")
print(json.loads(choice.message.content)["primes"])
```

- **开启思考时。** 在 chat API 和 Responses API 上 GLM-5.3 默认会思考，在 Messages API 上则在 `thinking` 要求时思考。思考不受约束，约束从思考结束处
  开始：思考在 `reasoning_content`（Messages API 中为 `thinking` 块），JSON 在答案中。引擎会及早结束思考，为答案至少留出 256 个 token，因此开启思考时
  `max_tokens`（Responses API 中为 `max_output_tokens`）至少要 545，更小的会以 400 拒绝。简短的回复请像 curl 例子那样用
  `"chat_template_kwargs": {"enable_thinking": false}` 关闭思考。
- **以 400 拒绝的情况**，在回复的任何头部之前（流式也一样），每种都附带说明原因的消息：
  - 与 `tools`（非空列表）一起，或 `tool_choice` 为 `required` 或指定函数：尚不支持；
  - 与 `stop` 字符串（可能截断 JSON）或 `ignore_eos` 一起；
  - 以 `MULTI` 启动的引擎：尚不支持；
  - `regex`、`choice` 和 `grammar`（`guided_regex`、`guided_choice`、`guided_grammar`，以及 `structured_outputs` 中的同类字段）：尚不支持，
    请使用 `json_object` 或 `json_schema`；
  - 语法只能比所写更宽松地遵守的 schema。服务器先检查 schema 并拒绝它，而不是返回不符合它的 JSON：JSON Schema 2020-12 以外的 `$schema`；
    以数组给出的 `items` 或 `additionalItems`（draft 4 的元组，请改用 `prefixItems`）；`#`、`#/$defs/NAME`、`#/definitions/NAME` 以外的 `$ref`；
    `nullable`（OpenAPI 的写法，请写 `"type": ["integer", "null"]`）；超过 ±2^53 的整数边界或常量；既不是 1 至 65,536 的整数、也不是最多 8 位小数
    且去掉小数点后不超过 65,536 的小数（0.25、0.01）的 `multipleOf`，或同一个值上的两个 `multipleOf`；允许其他键的对象上的 `minProperties` 或
    `maxProperties`；`x-guidance`（由服务器设定）；超过 64 KiB 或嵌套超过 32 层的 schema；孤立的代理项、NaN 或 Infinity。
- **须知。**
  - 有 `properties` 而没有 `additionalProperties` 的对象只接受它声明的键（`allOf` 的分支内除外）。要允许其他键，请写 `"additionalProperties": true`。
  - 被 `max_tokens` 截断的回复以 `finish_reason` `length` 结束，其答案是未完成的 JSON：解析前请检查 `finish_reason`。
  - JSON 中不出现 `\u` 转义：其他字符按原样写出，因此一个键只有一种写法；没有短转义（`\n`、`\r`、`\t`、`\b`、`\f`）的控制字符无法写出。两个
    token 之间的空白最多为一个换行和最多 64 个空格或制表符。
  - 结构化请求在没有 drafter（MTP 或 DFlash2）的情况下 decode，因此比同样长度的普通请求慢。普通请求与以前相同：与 r3.3.2-102 相同的 token、相同的速度。
  - 关闭方法：引擎设置（Server）中的 **Structured output**、`docker run` 行的 `-e STRUCTURED=0` 或 `glm53.conf` 中的 `STRUCTURED=0`，
    从下一次启动起生效；此类请求随后得到 400（`CONFIG.zh-CN.md`）。如果第一个节点无法加载 `lib/libllguidance.so`，引擎会在没有它的情况下启动
    （rank 0 的日志：`structured output unavailable`），并返回同样的 400。
  - 在交换机的 8 个节点上确认过：JSON 对象和 schema、开启思考、流式、被 `max_tokens` 截断、客户端中断的流、三种 API，以及所有拒绝的情况，之后每个 rank
    都仍在运行。普通请求在 20 个 prompt 上给出与 r3.3.2-102 相同的 token。

## 14. 确定性模式

**做什么。** 设置 `DETERMINISTIC=1` 后，从启动起按相同顺序发送相同的请求，每次重新启动都得到相同的 token（相同的发布版、节点和设置）。
任何温度下都成立：没有 `seed` 的请求在两种模式下都由 prompt 决定 seed，因此采样的回复也会重复。默认关闭。

**为什么。** 默认情况下，有两件事可能在两次启动之间不同：引擎在每次启动时对 linear 的内核 tile 计时并选用最快的，长 prompt 的路由专家按计算完成的顺序
相加结果。两者都可能改变数值的最后一位，偶尔因此改变一个 token。要比较两种设置、重复一次测试或重现一个回复，相同的请求必须得到相同的答案。

**如何使用。** 在 `glm53.conf` 中设 `DETERMINISTIC=1`、在 `docker run` 行加 `-e DETERMINISTIC=1`，或使用引擎设置（Server）中的
**Same answers (deterministic)**，从下一次启动起生效。之后 `/health` 返回 `"deterministic": true`，chat API 和 Responses API 上每个回复的
`tensorfold` 对象也是如此；其中的 `token_sha`（回复 token 的哈希）可用来比较两次启动的回复。

在此模式下，EXL3 的 linear 使用由形状决定的 tile，而不是计时选出的 tile；prompt 的路由专家按 slot 的顺序（而不是到达的顺序）相加
每行的 slot，也不使用 fp16 的和。启动时每个 rank 比对其设置、tile、加载的库和 NCCL 的设置，只要有一项不同就停止启动。在交换机上，
NCCL 与环上一样，使用 Ring 算法和 2 个通道。

- **限制。**
  - 一次一个对话：`MULTI` 为 1 或以上时拒绝启动。启动界面把并发推理的 2 到 4 显示为灰色并说明原因。
  - 模式所固定的内容不能写在 `EXTRA_ENV` 和 `LAUNCH_ENV` 中：其中有 `TF_GLM53_TUNE`、`TF_GLM53_PE_F16`、`TF_GLM53_DETERMINISTIC`、`TF_EXL3_PROMPT_EXPERTS`、`NCCL_ALGO`、`NCCL_PROTO`、`NCCL_PAT_ENABLE`、`NCCL_NVLS_ENABLE`、`NCCL_MIN_NCHANNELS`、`NCCL_MAX_NCHANNELS` 中的任何一个时拒绝启动。引擎设置界面也不会把此模式与并发推理 2 或以上、
    或与这些名称一起保存。
  - `CONTEXT_LENGTH` 和 `KV_SIZE` 按设置使用。启动时既不按空闲内存缩短上下文长度，也不把剩余空间全部给 KV（两者都会随节点的空闲内存而变）：
    放不下时停止启动。
- **仍会不同的地方。**
  - 工具调用的 id 是随机的，每次都不同。
  - 在同一次启动中，再次询问已被逐出缓存的 prompt，回复可能与第一次不同：缓存块的分界会改变 prompt 的计算顺序。重新启动并按相同顺序发送，
    这些请求会以相同的方式不同。
- **代价**（交换机的 8 个节点）：prefill 约慢 20 %（4K token 3.31 s，通常 2.75 s；40K 29.1 s，通常 23.6 s），解码大致相同（45.3 与
  45.9 tokens/s）。每个 rank 为 prompt 专家多占 384 MiB（`ROWS` 8,192、`SP_PARTS` 4；`SP_PARTS` 2 或 `ROWS` 16,384 时为 768 MiB），已计入 KV 计划。
- **确认过**（各启动两次）：在交换机（8 个节点）上，23 个请求（短请求 12 个、4K 与 40K 的 prompt、工具、结构化 4 个、温度 1.0 的 4 个）的 token
  23 个全部相同，KV 缓存在全部 232 处比较中相同；在环（4 个节点）上为 23/23 和 112/112。思考的中途整理、三轮对话（池的续接与 Byte Resume）和
  KV 逐出：使用 MTP 和 DFlash2 均为 13/13，冷启动时也一样。
