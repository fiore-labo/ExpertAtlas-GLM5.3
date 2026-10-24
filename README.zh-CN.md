[English](README.md) | [日本語](README.ja.md) | 简体中文 | [한국어](README.ko.md)

# 在四台或八台 DGX Spark 上运行完整版 GLM-5.3

完整版 GLM-5.3（755B，`glm_moe_dsa`）由四台或八台 NVIDIA DGX Spark 系统（每台一个 GB10）以张量并行 4 或 8 提供服务：一次处理一个请求，速度快，并为智能体（agent）和编程提供长上下文。引擎用 Zig 编写，使用纯 CUDA 内核，运行时不需要 Python。

这是 Fiore-lab 基于 [TensorFold](https://github.com/ashhart/TensorFold) 1.0.2 制作的非官方衍生版本，并非 TensorFold 项目的发布版本。它包含 TensorFold 的代码，增加了完整版 GLM-5.3 的引擎（`zig/src/families/glm_moe_dsa/`、`zig/kernels/cuda/glm53/`），并去掉了 TensorFold 的文档（这些文档描述的是其他模型、权重和机器）。关于 TensorFold 本身，请参阅其仓库。

## 功能

- 张量并行 8 或 4，KV 缓存在 rank 对之间切分（解码上下文并行 2）。引擎和模型最多支持 1M（1,048,576 个 token）；上下文上限由 DGX Spark 的内存决定：**八个节点（FP8 KV 缓存）为 1M，四个节点（4 bit KV 缓存）约 800K（819,200）**。设置示例中为 819,200 和 737,280；在相同内存下，八个节点可设 `CONTEXT=1048576` `STORE=589824`，四个节点可设 `CONTEXT=819200` `STORE=18080`（KV 池保存 `CONTEXT` + `STORE` 个 token）。
- 长提示词按序列并行分段读入；解码步的集合通信走 RoCE。
- 所有对话共用一个对话缓存，以 256 个 token为一块：请求会从任意对话中已缓存的最长前缀接着处理；开头相同的对话（智能体的系统提示词、其子智能体）共享的块只保存一份。
- 每一轮由模型校验草稿：草稿默认来自 GLM-5.3 自带的 MTP 层，也可来自可选的 DFlash2 草稿模型。无论哪种情况，回复都由模型本身决定。
- `/metrics`（Prometheus 文本格式）和 `/health` 也统计正在运行的请求，因此监控程序在请求运行期间也能读取 prefill 与解码速度、缓存命中和草稿接受情况。设置 `METRICS_LISTEN` 后，这两个端点会在单独的地址上以只读方式开放，供其他主机上的监控程序使用。

## Expert Atlas（r3-102 新功能，四个节点）

GLM-5.3 为每个 token 在每层 256 个路由专家中选出 8 个，而不同专家被选中的频率相差很大。Expert Atlas 让 19,200 个路由专家（第 3 至 77 层）各自以自己的位宽（2、3、4 或 5 bit）载入，在有限的内存中让使用最多的专家得到最多的位：

- **bank**：每种位宽的路由专家，位于 Hugging Face：`fiore-lab/GLM-5.3-EXL3-Experts-2bit`、`-3bit`、`-4bit` 和 `-5bit`。面板提供每个节点都具备的位宽。
- **控制面板**：由启动器（`./tf-glm53 launcher`）提供的页面。启动前它规划位宽，并显示每个节点的内存、计划能否容纳以及 KV 缓存还剩多少；也从这里启动和停止引擎。引擎运行时，它显示每个 token 选中的专家、被选中专家的平均位宽和速度。
- **使用次数**：引擎统计每个提示行和每个解码 token 选中的专家；停止时可以把这次运行的次数加入学习数据。计划按这些次数为专家排序（在保存第一次运行之前，按 Fiore-lab 测得并随页面提供的次数排序）。
- **专家优化**：保持每种位宽的专家数量不变，按使用情况重新排列专家：相同内存下，模型实际选中的专家的平均位宽更高。可在运行中的引擎上进行（在四个节点上，75 层的完整更改用了 83 至 90 秒，在请求之间进行），也可在停止的引擎上为下次启动进行。

本版本的控制面板在四个节点上运行；八个节点仍像以前一样，所有路由专家为 4 bit。详见 `packaging/RUNTIME.zh-CN.md`（“控制面板”）和 `glm53/release/T4.zh-CN.md`。

## 所需环境

- 四台或八台 DGX Spark 系统（GB10），位于同一 RoCE v2 网络上，且每台都能从第一台通过 ssh 免密码访问。四个节点时，每个节点一个 100 Gb/s 端口即可。
- DGX OS（Ubuntu 24.04），NVIDIA 驱动 580 或更高版本，以及 cuBLAS 13。
- 每个节点上都有 NCCL 2.30.7：即 NVIDIA 的 `nvidia-nccl-cu13` wheel，由 pip 解包（见 `packaging/RUNTIME.md`）。
- 与节点数相符的 Fiore-lab EXL3 检查点（Hugging Face），在每个节点上放在相同路径下：八个节点用 `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks`（390 GB），四个节点用 `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks`（344.6 GB）。四节点检查点中，第 3 至 77 层的路由专家里使用最少的一半为 3 bit，其余为 4 bit；其他部分与八节点检查点相同。
- 控制面板（四个节点）需要 3 bit 和 4 bit 的专家 bank（277.9 GB 和 368.5 GB；八节点检查点可以作为 4 bit bank；只有 4 bit 时上下文会更短），以及供启动器使用的各节点上的 `python3`。

## 运行

下载发布归档 `tensorfold-r3-102-linux-aarch64.tar.gz`，并按照其中的 `RUNTIME.md`（本仓库中为 [packaging/RUNTIME.zh-CN.md](packaging/RUNTIME.zh-CN.md)）操作。四个节点和八个节点使用同一个归档和同一个启动脚本：将 `glm53.conf.example`（八个节点）或 `glm53.t4.conf.example`（四个节点，见 [glm53/release/T4.zh-CN.md](glm53/release/T4.zh-CN.md)）复制为 `glm53.conf`，填写节点、网络名称、NCCL、权重和草稿模型，然后执行 `./tf-glm53 deploy` 和 `./tf-glm53 up`。除非为服务器指定地址和 API 密钥文件，否则它在 `127.0.0.1` 上提供 OpenAI chat API。

## 草稿模型

- `mtp`（默认）：检查点自带的 MTP 层，属于 GLM-5.3 权重的一部分。解码速度约为不使用草稿时的两倍，预填充（prefill）速度相同。
- `dflash2`：incoai/GLM-5.3-DFlash2。**由于其许可证，本仓库和归档均不包含它**：其模型卡注明的许可证为 CC BY-NC-ND 4.0（非商业性使用，禁止演绎），用于研究和评估；商业许可由 Inco AI 提供。如果你的使用在该许可证允许的范围内，请按照 [packaging/RUNTIME.zh-CN.md](packaging/RUNTIME.zh-CN.md#使用-dflash2-的步骤) 中的“使用 DFlash2 的步骤”操作：以固定的 revision 下载一次，复制到每个节点的相同路径并核对副本，然后设置 `DRAFTER_KIND=dflash2` 和 `DRAFTER`。引擎在启动时于内存中将其量化为 4 bit，不向磁盘写入任何内容。解码会更快（见“速度”）。
- `none`：不使用草稿。

由 `glm53.conf` 中的 `DRAFTER_KIND` 选择；`./tf-glm53 -d KIND up` 只为这一次启动选择。

## 速度

**测量时每个节点的 GPU 时钟上限均设为 2.1 GHz（2,100 MHz）。** 使用尚未缓存的编程提示词，一次一个请求（reasoning effort 为 low，回复最多 4,000 个 token），本归档分别采用 `glm53.conf.example`（八个节点，FP8 KV 缓存）和 `glm53.t4.conf.example`（四个节点，4 bit KV 缓存）的设置，以及 Fiore-lab 公开发布的检查点。你的结果取决于检查点、提示词和时钟。完整的测量条件见 `packaging/RUNTIME.md`。

| 节点数 | 提示词 | MTP（k=3）：首 token 时间 / 解码 | DFlash2：首 token 时间 / 解码 |
|---|---|---|---|
| 8 | 128K | 82.8 s / 47.5 tokens/s | 85.2 s / 53.2 |
| 8 | 256K | 172.9 s / 45.7 | 177.6 s / 52.6 |
| 8 | 512K | 376.5 s / 42.8 | 386.0 s / 47.1 |
| 4 | 128K | 120.1 s / 35.3 | 121.9 s / 42.9 |
| 4 | 256K | 253.3 s / 33.2 | 256.9 s / 40.7 |
| 4 | 512K | 565.4 s / 32.2 | 571.5 s / 37.8 |

## 精度

本方案的输出分布相对于原始 BF16 模型输出分布的 KL 散度（原始模型指以官方实现运行官方 BF16 权重、以 FP32 计算），使用本方案的服务配置：

- 八个节点（FP8 KV 缓存，DCP 2；用本引擎的较早版本在相同权重上测得），四段 2,048 token 的文本（对话、法律、散文、推理），每段取最后 1,023 个 token，运行五次：平均 0.0133（0.0128 至 0.0146）；top-1 token 的一致率约为 96%。
- 八个节点，四段 8,192 token 的文本（两段英文、两段代码），按token 区间：0.012（token 1,024 至 2,047）、0.018、0.019 和 0.013（6,144 至 8,190）。作为参照，官方 BF16 权重若以 BF16 而非 FP32 计算，在相同区间上与同一基准的差异为 0.007 至 0.010。
- 四个节点（四节点检查点，4 bit KV 缓存，DCP 2），四段 8,192 token 的文本，按相同的token 区间：0.019（token 1,024 至 2,047）、0.032、0.029 和 0.019（6,144 至 8,190）。
- Expert Atlas 的各位宽（所有路由专家为同一位宽，其他权重同上，相同的四段文本，token 6,144 至 8,190）：2 bit 0.0517、3 bit 0.0203、4 bit 0.0127（四个节点），5 bit 0.0114（八个节点）。在相同内存下，含 2 bit 专家的配置比不含的配置离基准远 13% 至 30%；把其他专家降到 3 bit 来换取 5 bit 专家，没有可测出的改善。

## 从源代码构建

在一台 GB10 系统上，需要 `PATH` 中有 Zig 0.17.0（或为 `build_libs.sh` 用 `ZIG=` 指定其路径）、CUDA 13（`nvcc`）以及 libibverbs 头文件（`libibverbs-dev`）：

```sh
zig build fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 --prefix /tmp/fatbins
sh glm53/tools/release/build_libs.sh /tmp/lib        # lib/libroce_proxy.so
zig build dist -Dversion=r3-102 -Ddist-commit=$(git rev-parse HEAD) -Ddist-platforms=linux-aarch64 \
    -Ddist-fatbins=/tmp/fatbins/fatbin -Ddist-lib=/tmp/lib -Ddist-extra=glm53/release
```

归档会生成在 `zig-out/dist/` 中。

## 限制

- 仅支持四台或八台 GB10 系统；不支持其他 GPU，也不支持其他 rank 数量。
- 一次只运行一个请求；其他请求最多 32 个按到达顺序等待（聊天标题生成请求排在最后），超过 32 个时服务器会拒绝（返回 503，流式请求则为错误事件）。对于并行运行子智能体的智能体，最好让子智能体改用另一个模型。
- 本版本的控制面板在四个节点上运行。
- 其余限制见 `packaging/RUNTIME.md`。

## 许可证与致谢

代码采用 Apache-2.0 许可（`LICENSE`）；`NOTICE` 和 `THIRD_PARTY_NOTICES.md` 说明了各部分的来源。权重保留其自身的许可证（GLM-5.3 License）；可选的 DFlash2 草稿模型也保留其自身的许可证。

- [TensorFold](https://github.com/ashhart/TensorFold)，由 TensorFold 贡献者开发：本方案所扩展的服务器、构建系统和 CUDA 引擎。
- drowzeys 为 TensorFold 编写的完整版 GLM-5.3 Python 引擎（[ashhart/TensorFold#159](https://github.com/ashhart/TensorFold/pull/159)）：GLM-5.3 引擎用 Zig 和纯 CUDA 重写了它的设计（引擎结构、Triton 内核以及 MTP 和 DFlash2 草稿模型的处理流程）。其中部分 CUDA 文件被收录，其中几个经过了修改。
- [b12x](https://github.com/local-inference-lab/b12x)：将其 RoCE one-shot 集合通信用纯 CUDA 重写（`roce.cu`）；收录了其 RoCE 代理（`zig/native/roce_proxy.c`），并修改为按每个 RDMA 设备选择 GID 索引。
- [MiaAI-Lab 的 GLM-5.3-Flash on 2x DGX Spark](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)：参考其提示词处理内核编写了 GLM-5.3 的提示词处理内核。
- [MiaAI-Lab 的 GLM-5.3 EXL3 on 3x DGX Spark](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold)：4 bit KV 缓存采用其布局，并用本引擎自己的代码实现。
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3)：检查点用它量化，引擎读取其 EXL3 格式。

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
