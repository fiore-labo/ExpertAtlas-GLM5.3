[English](PANEL-GUIDE.md) | [日本語](PANEL-GUIDE.ja.md) | 简体中文

# Expert Atlas：控制面板

面板的界面语言只有英文和日文。本页保留界面上的英文名称。

启动器在 `http://<first node>:<launcher port>/panel` 提供面板，默认监听所有地址，且没有密钥：任何能访问到面板的人都可以启动、停止和更改引擎，因此请把节点放在封闭的网络中（或设置 `LAUNCHER_HOST=127.0.0.1`，从其他机器通过 ssh 隧道打开：先运行 `ssh -L 8010:127.0.0.1:8010
<first node>`，再在那台机器上打开 `http://127.0.0.1:8010/panel`）。页面有两个界面：引擎停止时是启动界面，引擎运行时是运行面板。页面会自行在两者之间切换。本发布的面板在四个或八个节点上运行（节点数在启动界面的 Nodes 中选择）。

圆显示 19,200 个路由专家：第 3 至 77 层由内向外排列，每圈 256 个专家。

## 启动界面（引擎停止时）

- **Plan**（计划，卡片 [ BIT ALLOC ]）：每个专家载入时使用的位宽。界面打开时是一个新计划，比例取上一次运行的比例（还没有任何运行时全部为 3 bit）；Open（打开）可选择原样的上一次运行的分配，或已保存的分配。卡片的标题显示这是哪个计划：上一次运行、已保存分配的名称，或 New · unsaved。
  - **Shares**（比例）：在表格中输入每种位宽的专家所占比例（5、4 和 3 bit；有数据的最窄位宽占其余部分，因此总和为 100 %），或拖动曲线上的拖动点。刻度从 5 bit 一端开始计，该端为 0 %：5 bit 10 % 即范围 0–10 %。混合使用时，使用最多的专家得到最宽的位宽，按 prefill 0.3 / decode 0.7 排序（专家的 decode 占比权重为 0.7，prefill 占比权重为 0.3；`ATLAS-TESTS.zh-CN.md`）。
  - **Count**（次数；选项为 decode、prefill、prefill 0.3 / decode 0.7）：仅供查看。曲线、拖动点的读数（到某一点为止的专家承担了多大比例的选择）、圆的排列顺序和 Use 都跟随所选的次数；计划的位宽不会改变。
  - **A new plan**（新计划）：更改比例（输入或使用拖动点）会产生一个新计划（New · unsaved），专家按计划的使用记录排序（见下文）。
  - **Open**：列出上一次运行的分配和已保存的分配，并显示它们的比例、平均位宽，以及各自的制作时间和方式。打开的分配会按位原样成为计划，启动时使用该分配。它的使用记录随之而来：Count 和曲线显示该记录，从它更改比例得到的计划按该记录排序，从这两者之一启动的运行都会累加到该记录上。在更改比例之前，其位宽保持保存时的样子。需要没有数据的位宽或本通道（lane）没有的 bank 的分配无法打开；原因显示在它的下方。每个已保存的分配都可以 Rename 和 Delete（再按一次确认）；删除的分配进入启动器的回收站。
  - **Saving**（保存）：不会自动保存任何内容。计划在其运行停止时连同使用记录一起保存（Save and stop，见下文；运行面板的 Save 可随时保存）。
- **Usage records**（使用记录）：每个已保存的分配都保留保存它的那些运行的使用情况（每个专家被选中的次数，分 decode 和 prefill），因此其数据随分配一起移动。没有所有运行共用的数据池。
  - New，然后运行，然后 Save：该分配只保留那次运行的使用情况。
  - 打开（或更改其比例），然后运行，然后 Save：该分配保留到目前为止的使用情况，再加上那次运行的使用情况。
  - 没有使用记录的计划（界面的第一个新计划、上一次运行的分配）按发布自带的次数排序：即为本发布测得的 12 个领域的次数，内置在页面中。

  12 个领域：Coding、Agent、Japanese、English、Agent A、Agent B、Coding (ja)、Coding (en)、Linux ops (ja)、Linux ops (en)、Security (ja)、Security (en)。每个领域权重相同：专家按它在任一领域中占其所在层的最大比例排序。
- **[ WEIGHTS ]**（权重位置）：检查点和专家 bank 的位置，每个节点上路径相同。默认情况下，检查点为 `~/models/GLM-5.3-EXL3-Base`（Hugging Face 的 `fiore-lab/GLM-5.3-EXL3-Base`），每个 bank 为 `~/models/GLM-5.3-EXL3-Experts-<N>bit`（`QUICKSTART.zh-CN.md` 第 2 节）。权重在其他位置时，在该行写入目录或用 Browse 选择，然后按 Check and save。界面打开时、某行更改时，以及缺少必需项期间每 15 秒，都会查看每个节点；这只是读取，不保存任何内容。检查点和 3 bit bank 是必需的：缺少时该行显示为红色，并列出节点和仓库名称，Start 无法按下。某行显示的目录尚未保存时，Start 同样无法按下。其他位宽是可选的：留空的行或找不到的 bank 只是不使用该位宽。
- **Banks and No data**（bank 和无数据）：每种位宽的专家来自 Hugging Face 的 `fiore-lab/GLM-5.3-EXL3-Experts-<N>bit`（N = 2, 3, 4, 5），在每个节点上各放在自己的目录中，默认为 `~/models/GLM-5.3-EXL3-Experts-<N>bit`，并附带其清单。任一节点缺少 bank 的位宽会显示 No data，以及哪个节点缺少什么。它无法用于计划：它在表格中的行显示 No data，且不接受比例。
- **Memory**（内存，卡片 [ NODE ]）：每个节点用于该计划的内存（expert、Other weights、KV、Work、Margin）和判定结果：Can start、Can start (little room) 或 Not enough memory。KV FP4 / FP8 显示每种格式将分配的 KV 大小（token 数，以及每个节点的 GiB）；按下其中之一即可选择。KV 总是占用计划留下的全部空间：一个对话使用 Context length，其余用于之前的对话。Keep for the OS 设置每个节点保留的空闲内存。
- **Context length**（上下文长度）：启动所允许的最长请求（prompt 和回复合计），单位为 token：131K、262K、524K、glm53.conf 的 CONTEXT_LENGTH，或 Max（在所选 KV 格式下计划能留出的最大值，再减去引擎在此之外需要的部分：它绝不会等于整个 KV 大小，pool 缓存至少保留 512 个 token）。放不下的大小显示为灰色；启动器不得不截短的上下文显示在该行下方。面板写入之前默认为 Max：此界面上的选择在选择时写入 glm53.conf（Max 写入其数字，数字变化时重写；未点击时在启动时写入），下次打开时会选中该值。启动器根据在本通道上使用该草稿模型启动时占用的内存来求出 Max（每次启动都测量，按草稿模型分别测量）；在此之前，它按内置大小规划，并多保留 2 GiB，因此 Max 较短（est.）；KV 行在大小的左边标出 est. 或 Measured。测量在启动后立即进行，此时没有任何任务在运行，因此在 Max 下，请求运行期间节点的空闲内存可能略低于为操作系统保留的量。名称旁边的 i 会说明这一点，并说明所选的草稿模型是否已测量。
- **Keep for the OS**（为操作系统保留）：启动后每个节点保留的空闲内存，4、6、8 或 12 GiB（glm53.conf 的 RESERVE_GB，默认 4）。判定结果和引擎自身的准入检查都使用它；保留越多，留给 KV 的越少。
- **Drafter**（草稿模型）：MTP、DFlash2 或 None，在 KV 格式下方选择；内存按所选项规划。DFlash2 不随发布提供（因其许可证）：请下载并在 glm53.conf 中设置 DRAFTER。在每个节点都具备之前，它显示为灰色并说明原因。
- **Nodes**（节点，在卡片标题中，引擎停止时显示）：head（rank 0）是启动器运行所在的节点。worker 只需登记一次，与节点数无关（输入地址，按 Add），每种节点数所用的 worker 从列表中勾选（4 个节点时 Use 3 / 3，8 个节点时 7 / 7）；rank 按列表的顺序排列。列表和每种节点数的勾选保存在本通道的面板设置中，因此在任何浏览器上都会恢复。Delete 把 worker 从列表中移除（也从每种节点数的勾选中移除）；本通道当前使用的 worker 始终在列表中，取消勾选并保存后才会删除。Node count（4 或 8，当启动器同时提供两者时）选择节点数。Check 把发布包复制到 head 和已勾选的 worker（tf-glm53 deploy），然后在它们上面运行 tf-glm53 check（ssh、网络、内存、NCCL、文件；每个节点显示 ✓ 或 ✗ 及原因）；这需要一些时间。该列表的每个节点都通过后，Save 把节点写入 glm53.nodes.conf，供下一次启动使用，glm53.conf 不变（旧的节点文件的副本保留在旁边）；保存另一种节点数会把本通道切换到该节点数。Save 变灰时，编辑器在旁边说明原因。这些地址在交换机上是 ConnectX-7 端口的地址（NCCL 使用的地址），在环上是局域网地址，并且每个节点上都必须有每个 bank。在新节点数的启动被测量之前，计划都是估算值。
- **Start**（启动）：按两次（Start，然后在 8 s 内按 Confirm）。圆会随每一层的载入而点亮（真实数据：引擎的载入进度），然后页面变为运行面板。
- **Startup flow**（启动流程）：从页眉按钮打开的界面（该按钮只在步骤栏显示时出现，即引擎运行之前的界面）：0 start（在 head 节点上运行 `./tf-glm53 up`）› 1 › 2 › 3 › 运行面板，集中在一页上，编号与步骤栏相同，每一步都标有步骤栏自身的按钮。Close 或 Escape 返回。
- **Optimised allocations**（优化后的分配）：启动只载入给定的表，从不重新排序，因此启动界面没有 Optimise experts。请在引擎运行时优化（见下文）：结果就是引擎在停止时持有的内容，因此启动界面会把它作为上一次运行打开；Save 把它保留给以后的启动，Open 把它恢复回来。

## 运行面板（引擎运行时）

- **Circle**（圆）：哪些是测量值，哪些只是视觉效果。
  - decode：模型为每个 token 选中的专家会点亮并渐暗（真实数据：引擎记录的选择）。
  - prefill：单元格的闪烁只是视觉效果。由内向外扩展的圆环标出当前正在计算的层，每个 prompt 分块扫过一圈（根据 prompt 的进度估算）。分块使用的专家不会画出：一个有数千个 token 的分块几乎把每个专家都用到一点，点亮它们看起来一片平淡。
  - Start：圆随专家载入而填满，这是真实数据（见上文 Start）。

  prompt 进行 prefill 时，专家按编号顺序排列；decode 开始后，转为按使用量排序，使用最多的在最前，从 9 点钟方向开始。
- **Rim**（外圈）：计算 prompt 时显示 prefill 的进度（真实数据）；decode 时，回复的文字沿外圈滚动。
- **Centre and below**（中心和下方）：中心显示 bpw（被选中专家的平均位宽）和每种位宽在选择中的占比；圆的下方显示 prefill 和 decode 的 tok/s（decode：所有请求的 token 合计，多个请求同时运行时显示为 decode total；多个请求同时运行时的 prefill：圆所跟随的那个请求，按其 prompt 的计算时间计算，不含等待 decode 的时间，引擎不提供时显示 —），以及最近 3 分钟的 bpw。
- **[ NODE ]**：Stop 和服务器列表：每台服务器的名称旁标有 head 或 worker，并显示其空闲内存以及 GPU 和 CPU 温度（CPU 温度取节点各温度区中的最高值：GB10 的 CPU 和 GPU 共用一颗芯片，负载下比 GPU 自身的读数高 6-10 °C），由启动器每 5 s 读取一次；启动器不提供的项显示“—”。RDMA（引擎会说明其 RoCE 代理线程运行位置时显示）：每个节点两个代理线程（节点之间的和 DCP 对的）所在的 CPU（例如 0·10）及其类型，A725 或 X925；表格下方的一行说明启动设置（ROCE_PROXY_CORES：固定在 A725，这是默认值，或交给操作系统）。head 的是它们最后运行的位置；其他节点的是启动时固定的位置，交给操作系统时显示“—”。没有启动器时，同样的内容单独成一张卡片 [ RDMA ]。Concurrent：引擎同时处理的对话数，以及在 2 或以上时正在处理的请求数。
- **GPU split**（GPU 分配；仅当引擎能同时运行多个对话并提供其调度信息时显示，否则不显示）：在 prompt 的计算与正在 decode 的对话并行时，GPU 时间如何分配。滑块从 prefill 100%（prompt 优先：prefill 运行时 decode 停止）到 decode 100%（有对话正在生成时，新 prompt 等待，正在运行的 prefill 也停止），步长为 5；两者之间，该百分比的时间分给 decode。只有一方有工作时，它使用全部 GPU。松开滑块时发送该值（Applying…，然后 Applied），并对正在运行的请求也生效；被拒绝的更改会回到正在使用的值，并以红色显示原因。滑块下方的圆点是引擎启动时的值（引擎提供时显示）。下方：引擎在最近一次重叠期间测得的 decode 占比（重叠期间显示 Overlapping）；在 decode 100% 时，显示等待中的 prompt 数和最长等待时间。每一行的说明通过名称旁的 ⓘ 打开（与 Context length 的相同），再按一次或按 Escape 关闭；拒绝的原因不经 ⓘ 也会以红色显示。Priority prompt（API 中为 short_prompt）：需要计算的新 token（缓存之后的部分，不是 prompt 的长度）不超过此数的 prompt，无论分配比例如何都排在较长的 prompt 之前，其 prefill 使用全部 GPU（在此期间 decode 停止；提示中不给出秒数）。− / + 在 0、256、512、1K、2K、4K、8K、16K、32K 之间切换；也可以直接输入值（0 至 1,048,576，按 Enter 或离开输入框时发送）；0 表示关闭。Overtake a running prefill too（默认关闭）：正在运行的长 prefill 也让给 Priority prompt，并从已完成的分块继续（它最多晚一个分块）。每个值每秒都跟随引擎，因此在别处所做的更改也会显示在这里。不经启动器启动的引擎（没有 [ NODE ] 卡片）在单独的卡片 [ GPU SPLIT ] 中显示同样的内容，Concurrent 显示在其标题中。
- **[ REASONING CONTROL ]**（在 [ NODE ] 下方，与 GPU split 相同；这里是引擎运行时可以更改的思考设置；API 和设置项保持原有名称，即 `/v1/thinking` 和 `THINKING_*`）：Thinking limit（思考上限）是一个滑块，取值为 max_tokens 的 5% 至 100%（No limit）：达到这一比例时，引擎让模型结束思考，用剩余部分回答，因此较短的回复（harness 的 max_tokens）也能为回答留出空间；起始值以圆点表示。Organize midway（中途整理）为 Yes（checkpoint）/ No；Organize every（整理间隔）是max_tokens 的 10% 至 100% 的滑块（至少 1,024 个 token；右端 100% 为默认值：到达上限之前不整理），没有中途整理时变灰，否则须低于思考上限；Organize at the close（思考结束时整理）为 Yes（取它原有的思考上限比例，没有则取 glm53.conf 中的，再没有则为 100%）/ No。每一项都对此后开始的请求生效（Applying…，然后 Applied；被拒绝时恢复原值，并以红色显示原因），并由启动器写入 glm53.conf，因此下一次启动时仍保留；请求自身指定的思考预算或摘要优先。在启动界面上（引擎停止时），或引擎无法在运行中更改它们时，同一张卡片会立即把下次启动的值保存到 glm53.conf（From the next start）。引擎设置界面设置同样的四项；任何一方的更改都会显示在另一方。
- **[ BIT ALLOC ]**：已载入的位宽，标题中显示启动时使用的分配（新计划显示 New · unsaved）。Save 以一个名称保存它们。每种位宽显示：Experts（数量）、Share（占专家的比例）、Use（在选择中的占比）和 GiB。还有 Mean bits、Expert size、KV size（token 数和格式，例如 837K FP4）、Drafter、MemAvailable 和 Allocated（rank 0）。运行面板只统计本次运行的选择；这里不显示发布自带的次数。
- **Stop**（停止）：Save and stop 把已载入的位宽和本次运行的使用记录（运行开始时带有的记录加上本次运行的）以 Allocation name 保存，然后停止；Stop without saving 什么都不保存。提供的名称是本次运行所基于的分配，保存时覆盖（旧版本移到启动器的回收站）；也可以给出其他名称，若名称已存在，覆盖前要再按一次。

## 运行中的 Optimise experts（专家优化）

- [ BIT ALLOC ] 菜单中的 Optimise experts 按本次运行的使用记录（运行开始时带有的记录加上本次运行的选择）重新排列已载入的专家；每种位宽的数量保持不变。被选中最多的专家得到最宽的位宽，因此在相同内存下，被选中专家的平均位宽会提高。它的面板有自己的 Count（decode、prefill、prefill 0.3 / decode 0.7）：每次按下都从 prefill 0.3 / decode 0.7 开始，选择其他项会重新估算。在本次运行还没有 prefill 次数之前，单独的 prefill 显示为灰色，默认只按 decode 计数。
- 估算结果给出会改变的专家、推理停止的时长，以及对话缓存会被丢弃这一点。估算显示期间每 30 s 重新询问一次，Apply 按按下时的次数排序。按 Apply：面板等待正在运行的请求结束，然后逐层替换。结果不会自动保存；用 Save 保留。一次完整更改（9,756 个专家，75 层）在四个节点上用了 83 至 90 s。
- 使用 `TF_GLM53_MULTI` 运行的引擎在运行中不更改位宽。此时 Optimise experts 给出估算，并以 Save for a start 代替 Apply：估算的位宽以一个名称（`opt-<time>`）保存，推理继续进行。停止引擎，Open 该分配，然后 Start。

## Engine settings（引擎设置界面）

右上角的切换开关 Inference (Launch while stopped) | Engine settings 打开需要重启的设置，这些设置在 glm53.conf 中（没有这些设置的启动器不显示切换开关）。运行中可更改的设置留在推理界面上：GPU split、[ REASONING CONTROL ]（其更改在生效时写入 glm53.conf）和 Optimise experts。

- 每一行有名称、一个打开说明的 (i)（作用于什么、设置的是什么、原因、示例、与什么配合，以及默认值）、与 glm53.conf 不同时显示的 Changed，以及键名。违反规则的值（与 tf-glm53 相同的检查）显示为红色，Save 变灰；当前无法选择的选项显示为灰色，鼠标指向时说明原因。
- **KV and context length**（KV 和上下文长度）：Context length（CONTEXT_LENGTH：单个请求的最长长度）和 KV size（KV 大小，KV_SIZE：整个 KV 缓存，是所有对话共享的一个池；从面板启动时总是占用计划留下的全部空间，所以此项用于 `tf-glm53 up`）以及 KV 格式。
- **Thinking and answer sizes**（思考和回答的大小）：max_tokens（单次回复的上限；MAX_TOKENS_SHARE，一个滑块：上下文长度的比例，默认 70%，用于未指定 max_tokens 的请求；客户端自己的 max_tokens 优先，旧版 glm53.conf 的 MAX_TOKENS 在保存滑块之前也优先，保存时会将其清除）和思考上限，以max_tokens 的比例表示（THINKING_SHARE；保存它会清除旧的 THINKING_BUDGET）。思考上限、Organize midway、Organize every（THINKING_FOLD_SHARE）和思考结束时的整理标有 Applied at once：引擎运行时，保存会立即应用它们（消息会说明是哪些），推理界面的 [ REASONING CONTROL ] 设置的也是同样的值。使用中途整理时，这些上限标有 Virtual：它们计算已写出的 token，整理从 KV 中移出的思考仍然计入，因此可能超过上下文长度。组名旁边的 (i) 按所设的值画出一次回复的划分，以比例表示：实际大小随每个 prompt 而变化。Organize midway（THINKING_FOLD，以及大小目标，以被整理思考的比例表示，默认 70%，中途和结束时都适用）和思考结束时的整理（Yes 或 No，默认 No；选 Yes 时，当思考达到思考上限的 N% 或更多时整理，从上次整理起计，初始值为 100%：THINKING_FOLD_AT_CLOSE_SHARE，0 表示 No）各自成组。剩余量通知也是 Yes 或 No（选 Yes 时，每隔多少个思考 token 通知一次，初始值为 5,000：THINKING_NOTICE，0 表示 No）。
- 引擎运行期间，草稿模型、并发推理、KV 格式、上下文长度、KV 大小和保留的内存显示引擎启动时取用的值；启动后保存过的行，旁边会显示 Next start: X。只有你更改的行才会保存；停止时它们显示 glm53.conf 的值。
- 权重、DFlash2 模型和节点在这里只显示，要在检查它们的地方更改：启动界面的 [ WEIGHTS ]（Check and save：每个节点都通过后保存；失败的行说明原因，不保存任何内容）和节点检查。API 端口只能在引擎停止时更改；面板端口在启动器再次启动后更改。Advanced (development) 包含 prompt 分块、SP、token 修正和 EXTRA_ENV。
- 更改在发生时写入：选项立即写入，输入框在离开或按 Enter 时写入，滑块在松开时写入（写入 glm53.conf，副本为 glm53.conf.bak-<time>，最多每十分钟一份）。规则不允许的值不写入并说明原因。由规则关联的行（例如并发推理与 DFlash2 的保存数）在满足规则后一起写入；Revert 取消等待中的更改。引擎运行时保持它的值：这些行显示 Saved · next start。**Restart to apply** 询问一次，然后是停止时的问题（是否以一个名称保存本次运行的使用情况），接着启动器停止引擎，再以保存的设置和相同的位宽重新启动。

## Display（显示设置）

底部的抽屉（Display）：

- **General**（常规）：Language（English 或 Japanese；默认取浏览器的语言）、Data（Live 或 Simulation，以及 Pause），以及 Memory 条的外观。
- **Circle**：Order（默认 Auto，从 9 点钟方向开始）、Afterglow（默认 Short）、Unpicked 和 Picked 的亮度（默认为 24 级中的 6 和 19）、Light colour（By bit，或 One colour：默认为蓝色）。
- **Text and centre**（文字和中心）：外圈的 Text（默认显示且不滚动）、Digits（默认 7-segment）以及 Centre 的单元格。
- **Colours**（配色）：预设（默认 Green · one hue；Random 选择另一个）。

这些设置保存在一个文件中，即发布包目录下的 `launch/panel-settings.json`（`EXTRA_ENV` 中的 `TF_PANEL_SETTINGS` 可以移动它），因此每个浏览器打开的都是相同的面板。Back to defaults（按两次）从该文件中删除显示设置，因此本通道的每个浏览器都会得到默认值；语言和节点保持不变。
