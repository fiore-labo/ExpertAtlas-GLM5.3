[English](RUNTIME.md) | [日本語](RUNTIME.ja.md) | 简体中文

# 从本归档运行完整版 GLM-5.3

第一次使用？`QUICKSTART.zh-CN.md` 用一行命令带你从镜像到引擎运行。Base 权重和 3 bit bank 始终是必需的，未指定时所有专家都以 3 bit 载入。

本归档在四台或八台 NVIDIA DGX Spark 系统上以张量并行 4 或 8 提供完整版 GLM-5.3（`glm_moe_dsa`，755B）服务，每个 GB10 一个 rank。服务器不需要 Python、编译器或源代码检出；控制面板的启动器和日志保管进程使用系统的 `python3`，只用其标准库。它是 Fiore-lab 基于 TensorFold 1.0.2 制作的非官方衍生版本，并非 TensorFold 项目的发布版本。本页同时适用于四个节点和八个节点；四个节点的不同之处及其测量结果见 `T4.zh-CN.md`，不经交换机连成环的四个节点见 `RING.zh-CN.md`。

## 归档内容

- `bin/tensorfold-native`：服务器，内含为 GB10（sm_121）构建的 CUDA 内核。Linux aarch64，glibc 2.28 或更高版本。
- `lib/libroce_proxy.so`：解码集合通信使用的 RDMA 代理（需要 `libibverbs.so.1`）。
- `lib/libllguidance.so`：结构化输出的语法（llguidance 1.9.1，MIT），由 rank 0 加载（`CONFIG.zh-CN.md` 的 `STRUCTURED`）。
- `tf-glm53`、`glm53.conf.example`（八个节点）和 `glm53.t4.conf.example`（四个节点）：用于启动、停止和检查各 rank。`glm53.conf` 的全部设置项见 `CONFIG.zh-CN.md`。
- `tools/quickstart.py`：从全新安装到引擎运行（`docker run ... IMAGE up` 和 `./tf-glm53 quickstart` 会运行它）。`tools/topology.py`：判定节点的连接方式。`tools/logkeep.py`：日志保管进程。
- `claude-glm53`：在服务器上使用 Claude Code（“Claude Code”）。
- 控制面板（四个或八个节点）：`launcher/tf_launcher.py`（启动器）、`panel.html`（其页面；服务器内含同一页面）、`tools/quant/expert_plan.py`（编写专家计划）和 `expert-plan/banks.json`（专家 bank 及其位置）。
- `docker/Dockerfile` 和 `docker/entry.sh`：把本发布做成容器镜像（见“在容器中运行”）。
- 文档，每份都有英文版、日文版（`.ja.md`）和简体中文版（`.zh-CN.md`）：`QUICKSTART.zh-CN.md`（第一次启动）、`BEFORE-START.zh-CN.md`（启动前要决定的事项）、`RUNTIME.zh-CN.md`（本页）、`CONFIG.zh-CN.md`、`T4.zh-CN.md`（四个节点）、`RING.zh-CN.md`（环）、`PANEL-GUIDE.zh-CN.md`（面板的各个界面）、`AGENT-API.zh-CN.md`（供 AI 代理使用的控制 API）、`AI-GUIDE.zh-CN.md`（把整个系统写在一页上，供 AI 助手使用）和 `ATLAS-TESTS.zh-CN.md`（专家重新排序的测量）。
- `LICENSE`、`NOTICE`、`LICENSES/`、`THIRD_PARTY_NOTICES.md`、`VERSION`、`SHA256SUMS`。

其中不含模型权重、草稿模型，也不含 NCCL。镜像包含带 AArch64 fence 的 NCCL（见“在容器中运行”）。

## 每个节点的要求

- DGX OS（Ubuntu 24.04），NVIDIA 驱动 580 或更高版本（由其提供 `libcuda.so.1`）。
- 加载器路径上有 cuBLAS 13（`libcublas.so.13`、`libcublasLt.so.13`）：DGX OS 会随 CUDA 13 一起安装它（`libcublas-13-0`）。
- 每个节点相同路径下的 NCCL 2.30.7（见下文）。
- `libibverbs.so.1`，以及连接各系统的同一个 RoCE v2 网络（四个节点时，每个节点一个 100 Gb/s 端口即可），或连成环的四个节点（`RING.zh-CN.md`）；每个节点（包括第一台本身）都能从第一台通过其在 `NODES` 中的地址以 ssh 免密码访问，并已登记其 host key（`QUICKSTART.zh-CN.md` 第 1 节）。
- 磁盘空间，在每个节点上放在相同路径下（见下文）：Base 权重 21.2 GB（19.8 GiB）和 3 bit 专家 bank 277.9 GB，共 299.1 GB（四节点和八节点都一样），每次启动都需要。
- 八个节点：启动前每个节点需有 112 GiB 可用内存（使用 DFlash2 时为 115），以 `free -g` 的显示为准：当 `CONTEXT_LENGTH` 为 819,200、`KV_SIZE` 为 1,638,400 时，各 rank 约使用 108 GiB（使用 DFlash2 时为 111）。`KV_SIZE` 每减少 100,000 个 token，所需内存减少 2.55 GiB（4 bit KV 缓存时为 1.73）。可用内存不足的节点会被 `tf-glm53 check` 拒绝。使用 4 bit KV 缓存（`KV=fp4`）时，八个节点可将 `CONTEXT_LENGTH` 设为 1,048,576、`KV_SIZE` 设为 2,097,152，各 rank 约使用 94 GiB（用 1,018,993 个 token 的提示词实测）；`tf-glm53 check` 的估算按 8 bit 缓存计算，因此这时请设置 `NEED_GB=100`。
- 四个节点：内存按只运行本服务器来分配（见 `T4.zh-CN.md`）；使用 DFlash2 时测得的最低可用内存为每个节点约 3.2 GiB（512K 时，`ROWS` 为 16,384；现在默认为 8,192）。
- 每个节点上的 `python3`，供日志保管进程使用，归档和镜像都需要（DGX OS 自带）。
- 其他位宽：所用专家 bank 的磁盘空间（见下文“控制面板”）。
- 在容器中运行时（见“在容器中运行”）：每个节点需要 Docker 和 NVIDIA Container Toolkit，而不是 cuBLAS、libibverbs 和 NCCL。

## 首次启动之前

NCCL：NVIDIA 以 Python wheel 的形式发布 NCCL 2.30.7。pip（软件包 `python3-pip`）只负责解包该库，没有任何组件会运行 Python。在每个节点上执行：

```sh
pip3 install --no-deps --target ~/nccl-2.30.7 nvidia-nccl-cu13==2.30.7
```

`glm53.conf` 中的 `NCCL_LIB` 默认指向 `~/nccl-2.30.7/nvidia/nccl/lib/libnccl.so.2`。从源代码构建的 NCCL 2.30.7（NVIDIA 的 DGX Spark playbook “NCCL for Two Sparks”，<https://build.nvidia.com/spark/nccl>）同样可用：将 `NCCL_LIB` 设为它即可。在交换机上 wheel 也能运行，但推荐使用带 AArch64 fence 的构建：没有 fence 时，Arm 上的代理线程可能在没有任何错误的情况下挂起（`RING.zh-CN.md` 的“fenced NCCL”）。环必须使用该构建。

Base 权重和 3 bit bank 都是必需的；bank 见下文“控制面板”。Base 权重：Hugging Face 上的 `fiore-lab/GLM-5.3-EXL3-Base`，四节点和八节点都一样，是路由专家以外的全部权重，路由专家在启动时从 bank 读取。用 Hugging Face CLI 下载一次（`python3 -m venv` 需要 `python3-venv` 软件包），然后通过 RoCE 网络复制到其他节点。八个节点时：

```sh
python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install -U huggingface_hub
D=~/models/GLM-5.3-EXL3-Base
~/hf-cli/bin/hf download fiore-lab/GLM-5.3-EXL3-Base --local-dir $D
for n in NODE2 NODE3 NODE4 NODE5 NODE6 NODE7 NODE8; do    # the other seven addresses in NODES
  rsync -a --mkpath -e "ssh -c aes128-gcm@openssh.com" $D/ $n:$D/ &
done; wait
```

四个节点时，对其余三个节点使用相同命令（见 `T4.zh-CN.md`）。

要校验副本，请在每个节点上运行 `cd ~/models/GLM-5.3-EXL3-Base && sha256sum -c --quiet SHA256SUMS`：每个节点需几分钟，所有文件都一致时不输出任何内容（部署本目录之后，`./tf-glm53 verify` 可一次校验所有节点）。`glm53.conf` 中的 `WEIGHTS` 指定该目录（此处为 `~/models/GLM-5.3-EXL3-Base`）。

完整检查点 `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks`（390 GB，363 GiB）和 `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks`（344.6 GB，321 GiB）还包含路由专家，但启动时不读取它们。如果已有，用 `WEIGHTS` 指定即可直接使用。

所有权重文件（Base 权重、检查点和专家 bank）也都可以不使用 Fiore-lab 的任何文件，在有 GPU 的主机上从 Z.AI 官方的 BF16 检查点自行制作。步骤见源代码仓库中的 `glm53/weights/README.zh-CN.md`（<https://github.com/fiore-labo/ExpertAtlas-GLM5.3/blob/main/glm53/weights/README.zh-CN.md>；归档中不包含该目录）。

## 启动

```sh
tar xzf tensorfold-VERSION-linux-aarch64.tar.gz
cd tensorfold-VERSION-linux-aarch64
sha256sum -c SHA256SUMS
cp glm53.conf.example glm53.conf    # nodes, network names, NCCL, weights, drafter, port
./tf-glm53 deploy                   # the same directory on the other nodes
./tf-glm53 up                       # the control panel: open the address it prints, start the engine there
./tf-glm53 down                     # stops the engine, as the panel's Stop does
./tf-glm53 start                    # starts it again with the last start's plan, as the panel's Start does
./tf-glm53 -d dflash2 start         # another start with the DFlash2 drafter (Drafters, below)
```

`up` 在第一个节点上启动控制面板的启动器，并输出面板的地址（`http://FIRST-NODE:LAUNCHER_PORT/panel`）。第一次时，面板从节点检查开始：判定节点的连接方式（环或交换机；交换机使用空闲的服务器，要用其他服务器时更改勾选后再按一次），检查后登记到 `glm53.nodes.conf`；之后从启动设置开始，权重的位置也可以在那里指定。从面板启动时，按在那里选择的计划的位宽，从专家 bank 载入路由专家（见下文“控制面板”）。`start` 用上一次启动的计划启动（没有计划时所有专家都用 3 bit），并等待引擎开始运行。手动执行 `./tf-glm53 engine-up` 与 `start` 相同。有启动器的计划时，它检查每个节点、启动各 rank、等待服务器并运行冒烟测试（`./tf-glm53 engine-down` 停止所有 rank）；从面板启动时以这种方式执行。

四个节点时，复制 `glm53.t4.conf.example` 而不是 `glm53.conf.example`（见下一节）。`tf-glm53` 既不使用 `sudo`，也不更改系统设置。它的冒烟测试检查服务器能否应答、能否复用已缓存的提示词；它不衡量精度。

## 四节点与八节点的设置

`tf-glm53` 根据 `NODES` 中的地址个数以张量并行 4 或 8 运行。两个设置示例的区别在于下列各行；网络名称、NCCL、草稿模型和服务器在两者中的设置方式相同：

| 设置 | `glm53.conf.example`（八个节点） | `glm53.t4.conf.example`（四个节点） |
|---|---|---|
| `NODES` | 八个地址，rank 0 在前 | 四个地址，rank 0 在前 |
| `WEIGHTS` | `~/models/GLM-5.3-EXL3-Base` | `~/models/GLM-5.3-EXL3-Base` |
| `KV` | `fp8`（默认） | `fp4` |
| `CONTEXT_LENGTH` | 819200 | 1048576 |
| `KV_SIZE` | 1638400 | 1049088 |
| `DCP` | 2 | 2 |
| `ROWS` / `SP_PARTS` | 8192 / 4（默认） | 8192 / 4（默认） |

`KV` 是 KV 缓存的格式：FP8 或 4 bit。`DCP` 是解码上下文并行，`ROWS` 和 `SP_PARTS` 分别是提示词分块的行数和每块拆分成的序列并行段数。`CONTEXT_LENGTH` 和 `KV_SIZE` 见下文“请求之间保留的对话”。以上是两个示例中的取值；四节点取值的说明见 `T4.zh-CN.md`。

`glm53.nodes.conf` 由节点检查写入，在 `glm53.conf` 之后读取并以它为准：它一旦存在，`TOPOLOGY`、`NODES`、`SOCKET_IFNAME` 和 `IB_HCA` 都取自它。`glm53.conf` 的全部设置项、默认值、由谁读取以及修改何时生效，见 `CONFIG.zh-CN.md`。为早先版本编写的 `glm53.conf` 仍可读取：其中的 `CONTEXT` 按 `CONTEXT_LENGTH` 读取，`CONTEXT` + `STORE` 按 `KV_SIZE` 读取（见 `CONFIG.zh-CN.md`）。

## 第一个请求

服务器在 `http://HOST:PORT/v1` 上提供 OpenAI chat API，模型名由 `NAME` 指定（在 `glm53.conf.example` 中为 `glm-5.3`，端口 8000）。在第一个节点上：

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3", "max_tokens": 300,
  "messages": [{"role": "user", "content": "Say hello in one short sentence."}]}'
```

回复的 `message` 中，`reasoning_content` 是模型的推理过程，其后 `content` 是答案；`max_tokens` 同时计入两者。从其他机器访问时，将请求发送到 `http://FIRST-NODE:PORT`（默认的 `HOST=0.0.0.0` 时）。也可以用 `response_format` 把回复的答案限定为 JSON 或 JSON Schema（`FEATURES.zh-CN.md` 的“结构化输出”）。

## 草稿模型

引擎每轮起草几个 token，由模型进行校验；无论哪种情况，回复都由模型本身决定。`glm53.conf` 中的 `DRAFTER_KIND` 选择由什么来起草。面板的启动界面也可以选择，并写入那里；在 `start` 或 `engine-up` 前加 `-d KIND` 可以不改文件、只为这一次启动选择（`./tf-glm53 -d dflash2 start`、`./tf-glm53 -d mtp engine-up`），`./tf-glm53 status` 会显示上一次启动所用的草稿模型：

- `mtp`（默认）：`WEIGHTS` 中的 MTP 层，属于 GLM-5.3 权重的一部分，适用其许可证。它要求权重包含该层；`tf-glm53 check` 会拒绝不含该层的权重。每轮最多 `MTP_K` 个草稿，默认 3（四个并不比三个快）；一轮的草稿在其概率之积低于 `TF_GLM53_MTP_CONF`（默认 0.7，0 表示关闭）处停止。
- `dflash2`：incoai/GLM-5.3-DFlash2，位于 `DRAFTER` 指定的目录。**由于其许可证，本归档和仓库均不包含它**：其模型卡注明的许可证为 CC BY-NC-ND 4.0（非商业性使用，禁止演绎），用于研究和评估；商业许可由 Inco AI 提供。如果你的使用在该许可证允许的范围内，请按下文“使用 DFlash2 的步骤”进行设置；这样解码会更快（见“速度”）。
- `none`：不使用草稿。

### 使用 DFlash2 的步骤

1. 阅读模型卡（<https://huggingface.co/incoai/GLM-5.3-DFlash2>）上的许可证，确认你的使用在其允许的范围内。引擎按原样读取公开的文件：启动时，每个 rank 取出草稿模型权重中由自己负责的部分，并在内存中量化为 4 bit。不需要事先转换，也不会向磁盘写入任何内容。
2. 用为权重准备的 Hugging Face CLI，在第一个节点上下载一次，并固定为本归档核对过的 revision。无需账号或 token；其权重为一个文件 `model.safetensors`（4,918,859,112 字节）。请使用单独的目录（`--local-dir`），而不是 Hugging Face 缓存：`DRAFTER` 所指的目录中必须直接包含 `config.json` 和 `model.safetensors`。

   ```sh
   R=~/models/GLM-5.3-DFlash2
   ~/hf-cli/bin/hf download incoai/GLM-5.3-DFlash2 --revision 425aa615ce320caac34400208b30808c8f14f76c --local-dir $R
   ```

3. 复制到其余节点的相同路径下。`tf-glm53 deploy` 只复制本归档的目录，不复制草稿模型：

   ```sh
   for n in NODE2 NODE3 NODE4 NODE5 NODE6 NODE7 NODE8; do    # the other addresses in NODES (three on four nodes)
     rsync -a --mkpath -e "ssh -c aes128-gcm@openssh.com" $R/ $n:$R/ &
   done; wait
   ```

4. 在每个节点（包括第一个节点）上核对文件。每个 rank 读取的是自己节点上的副本，而 `tf-glm53 check` 只检查运行它的节点上是否有 `config.json`，不会在节点之间比较草稿模型。

   ```sh
   for n in NODE1 NODE2 NODE3 NODE4 NODE5 NODE6 NODE7 NODE8; do    # every address in NODES
     printf '%s  %s\n' \
       f59e1da17d41d24a1aba588aecee1607788adb34a03805f2c883add8ca954e9b config.json \
       3105f14043bef642baa49a7d533fdf0b8b2895737ec84b6305601da662656161 model.safetensors |
       ssh $n "cd $R && sha256sum -c --quiet" && echo "ok $n" || echo "FAIL $n"
   done
   ```

   每个节点都必须输出 `ok`。
5. 在 `glm53.conf` 中填写相同的路径：

   ```sh
   DRAFTER_KIND=dflash2
   DRAFTER=~/models/GLM-5.3-DFlash2
   ```

   如果设置了 `DRAFTER` 而 `DRAFTER_KIND` 仍为 `mtp`，在启动界面上选择 DFlash2 会写入 `DRAFTER_KIND`；只用于这一次启动时用 `./tf-glm53 -d dflash2 start`。引擎停止期间，启动器会自行读取编辑后的内容。使用 DFlash2 时不使用 `MTP_K`。八个节点时，启动前每个节点需要 115 GiB 可用内存，而不是 112 GiB（见“每个节点的要求”）；`tf-glm53 check` 会检查这一点。最近对话的草稿模型状态也会被保留（`DRAFT_SAVES`，见“请求之间保留的对话”）。
6. 照常启动：如果正在运行，先执行 `./tf-glm53 down`，再从面板（或用 `./tf-glm53 start`）启动。`tf-glm53 check` 会输出 `drafter: dflash2 (PATH)`，启动时也会输出，见 `tf-glm53 logs launcher`；无论使用哪种草稿模型，引擎自身的启动行都显示 `drafts: on`。
7. 要确认草稿是否起作用，请在一次请求的前后读取 `curl -s http://127.0.0.1:8000/health` 中的 `completion_tokens_total` 和 `rounds_total`。前者的增量除以后者的增量，就是该请求每轮确定的 token 数：为 1 表示没有采用任何草稿。对于与“速度”一节类似的编程提示词，DFlash2 为 4.3 到 5.0。

要改回 MTP 层，请在启动界面上选择 MTP，或设置 `DRAFTER_KIND=mtp`，然后执行 `./tf-glm53 down` 并重新启动；若只为一次启动，执行 `./tf-glm53 -d mtp start`。

## 请求之间保留的对话

上下文长度（`glm53.conf` 中的 `CONTEXT_LENGTH`）是单个请求最多可达到的 token 数（提示词加回复），最多为模型的 1,048,576。KV 大小（`KV_SIZE`）是整个 KV 缓存的 token 数，由所有对话共享：上下文长度供该请求使用，其余部分保留先前对话的 KV。

KV 缓存是一个容纳 `KV_SIZE` 个 token 的 KV 池，按 256 个 token 分块（`KV_SIZE` 至少为 `CONTEXT_LENGTH` + 512）；每个对话的 KV 是这些块组成的表。先前的对话保留在任意空闲的块中，不为它们另外预留空间。请求会从任意对话（不只是紧接其前的那个）中已缓存的最长前缀接着处理。同一对话的下一轮接着使用该对话自己的块。开头与先前某个对话相同的新对话（相同的系统提示词，例如有些智能体框架每次请求都会发送约 27K token 的系统提示词；子智能体也是如此）会共享先前对话中位于两者分歧点之前的完整块，这些块只保存一份，并且只从那一块开始计算。池满时，最久未使用的对话先从末尾交出块。

每个已缓存的块在复用之前都会与请求的 token 进行核对（通过抽样行的哈希）。只要有任何疑点，就丢弃所有对话，从头计算该请求（“cold”）；绝不会错误地复用某个块。使用 DFlash2 时，池还会保留最多 `DRAFT_SAVES` 个（默认 8 个）对话的草稿模型状态，使这些对话的下一轮也能以全速起草。回复由模型本身决定；从缓存继续处理的请求，与从第 0 个 token 开始计算的请求相比，最后几位可能不同（舍入顺序不同）。

Byte Resume（`TF_GLM53_BYTE_RESUME`，默认 `on`）在模型的切分方式与分词器不同时保住缓存：这时下一轮的提示词以不同的 token id 持有相同的字节，按 id 匹配会停在那里。从第一个不同的 id 起，把提示词与保留的对话按字节比较，字节一致之处使用它们的 id，缓存因此越过该处继续。文本不变；回复可能与按分词器的 id 计算的结果略有不同。`log` 只报告这些位置，`off` 将其关闭。

`tf-glm53 status` 会显示池的情况，启动时也会显示，见 `tf-glm53 logs launcher`；采用四节点设置时如下：

```text
conversation cache (pool): 1049088 positions in 4098 blocks of 256, shared by up to 64 conversations (context 1048576)
```

此时服务器的 `/health` 中有一个 `pool` 对象：`blocks`、`free_blocks`、`slots`（保留的对话数）、`shared_blocks`（由多个对话共同持有的块数）、`evicted_total`（为腾出空间而丢弃的对话数）和 `cold_total`（从头计算的请求数）。

## 同时处理多个对话

默认情况下一次只运行一个请求。`MULTI`（在 `glm53.conf` 中或启动界面上设为 1 至 4）让最多这么多个对话同时保持活动并一起解码，可与 MTP 层、DFlash2 或不使用草稿模型一起使用，代价是占用节点内存（`BEFORE-START.zh-CN.md`）。在对话缓存中没有空间与其他对话并存的请求最多等待 `MULTI_WAIT` 秒，之后以繁忙被拒绝。

运行期间，面板或 `POST /v1/schedule` 可以设置在正在解码的对话旁计算提示词时如何分配 GPU（`MULTI_DECODE_SHARE`），短提示词也可以排在较长的提示词之前（`MULTI_SHORT_PROMPT`、`MULTI_SHORT_PREEMPT`）。每一项的说明见 `CONFIG.zh-CN.md`。

## 思考：上限与整理

`max_tokens` 也计入思考。默认情况下，思考在上下文中最多可保留提示词在上下文长度中剩余部分与 `max_tokens` 两者中较小者的 90 %（`THINKING_SHARE`）；到达这一点时，引擎在换行处结束思考，其余部分留给回答。未指定 `max_tokens` 的请求得到上下文长度的 70 %（`MAX_TOKENS_SHARE`）。

这个上限针对的是思考保留在上下文中的量，而不是它写出的量。接近上限时（上限之前 10 %，`THINKING_FOLD_MARGIN`），引擎让模型改写思考，并用结果替换上下文中的思考（`THINKING_FOLD_AT_BUDGET`，默认开启）；之后由模型决定继续还是结束。因此一次回复写出的量可以超过上限。整理也可以在中途进行（`THINKING_FOLD`，默认关闭），或在模型结束思考时进行（`THINKING_FOLD_AT_CLOSE_SHARE`，默认关闭）。客户端仍会收到全部推理内容。`THINKING_NOTICE`（默认关闭）告诉模型已经用了多少。

`CONFIG.zh-CN.md` 中标有 **运行中也可改** 的设置可以在引擎运行时更改，通过面板的 [ REASONING CONTROL ] 或 `POST /v1/thinking`，并写入 `glm53.conf`。请求自身的字段优先：`thinking_budget`、`thinking_fold`、`thinking_fold_at_budget`、`thinking_fold_margin` 和 `thinking_fold_at_close`（以 token 数指定）。详情和测量见 `AGENT-API.zh-CN.md`。

## 控制面板（Expert Atlas）

在四个或八个节点上，第 3 至 77 层的路由专家（19,200 个，每层 256 个）可以逐个以 2、3、4 或 5 bit 从四个 bank 载入；控制面板选择位宽，并能在引擎运行时更改。第一次时，面板的节点检查把节点（首节点加三个或七个 worker）登记到 `glm53.nodes.conf`。无论启动时是否使用，每个节点上都必须有 3 bit bank。未指定位宽的启动以 3 bit 载入所有专家。八个节点时每个节点只持有一半的路由专家，因此能放下更宽的位宽。四个节点需要哪些 bank、什么配置能容纳，见 `T4.zh-CN.md`“控制面板”。

**bank**：每个 bank 以一种位宽保存路由专家，并附带清单（`<id>.json`：每个文件的大小、头部和 sha256）：Hugging Face 上的 `fiore-lab/GLM-5.3-EXL3-Experts-2bit`、`-3bit`、`-4bit` 和 `-5bit`（187.3、277.9、368.5 和 459.1 GB）。`expert-plan/banks.json` 指定每个 bank 的目录，在每个节点上相同（`~/models/GLM-5.3-EXL3-Experts-<N>bit`；如放在别处，请在其中修改 `dir`，然后执行 `./tf-glm53 deploy`）。四个节点所需的 3 bit 和 4 bit bank：

```sh
for w in 3 4; do
  D=~/models/GLM-5.3-EXL3-Experts-${w}bit
  ~/hf-cli/bin/hf download fiore-lab/GLM-5.3-EXL3-Experts-${w}bit --local-dir $D
  for n in NODE2 NODE3 NODE4; do    # the other three addresses in NODES
    rsync -a --mkpath -e "ssh -c aes128-gcm@openssh.com" $D/ $n:$D/ &
  done; wait
done
./tf-glm53 verify-banks 3 4    # every file against its manifest's sha256 on every node, at low priority (minutes)
```

八节点检查点的路由专家与 4 bit bank 逐个文件相同：若以该检查点作为 `WEIGHTS`，请在每个节点上把 4 bit 清单放进它的目录（`hf download fiore-lab/GLM-5.3-EXL3-Experts-4bit glm53-exl3-experts-4bit.json --local-dir ~/models/GLM-5.3-EXL3-8x-DGX-Sparks`），并把 4 bit bank 的 `dir` 设为 `~/models/GLM-5.3-EXL3-8x-DGX-Sparks`。启动器会在自身启动时、页面打开期间最多每分钟一次、以及每次启动之前，在每个节点上检查每种位宽的 bank（清单，以及每个层文件是否为清单中的大小）；面板只提供每个节点都具备的位宽，需要其他位宽的启动或更改会在任何改变之前被拒绝。启动时会将每个层文件的大小和头部与清单核对；`verify-banks` 会读取整个文件。

**启动器**：在第一台节点上运行，并且无论引擎是否运行都一直运行：

```sh
./tf-glm53 up                 # start it and print the panel's address (./tf-glm53 launcher starts it too)
./tf-glm53 launcher status    # also ./tf-glm53 launcher stop
```

引擎停止期间，编辑后启动器会自行重新读取 `glm53.conf`。`LAUNCHER_HOST` 和 `LAUNCHER_PORT` 在启动器重新启动后生效（`./tf-glm53 launcher stop`，然后 `./tf-glm53 up`；引擎可以继续运行）。它读取的项目见 `CONFIG.zh-CN.md`。

打开 `up` 输出的地址 `http://FIRST-NODE:LAUNCHER_PORT/panel`：在 `glm53.t4.conf.example` 中端口为 8010（没有 `LAUNCHER_PORT` 时为 `PORT` + 10）。启动器默认监听所有地址（`LAUNCHER_HOST=0.0.0.0`），且没有密钥，任何能访问到它的人都可以启动、停止和替换引擎：请只在封闭网络中使用，或设置 `LAUNCHER_HOST=127.0.0.1`，从其他机器通过 ssh 隧道打开面板（`ssh -L 8010:127.0.0.1:8010 FIRST-NODE`）。页面的其他调用经由启动器到达引擎，因此无论引擎是否运行，页面只有一个地址。在启动器（以默认的所有地址）运行期间启动的引擎，会把打开引擎自身 `/panel`（`PORT`）的浏览器转到启动器的面板。使用面板期间，请从面板启动和停止引擎。页面说明见 `PANEL-GUIDE.zh-CN.md`；页面以英文或日文显示（按浏览器的语言，或 Display > Language）。

- **启动**（从面板）：启动器检查每个节点（可连通、没有引擎或其他 GPU 进程、计划能放进其内存），把位宽写成专家计划（`launch/plan-<时间>.json`），复制到每个节点，并用它执行 `tf-glm53 engine-up`（`tf-glm53 start` 用上一次启动的计划进行同样的启动）；页面会显示每一层的载入。此后所有路由专家都来自 bank，`WEIGHTS` 提供其他权重。
- **草稿模型**在启动界面上与 KV 格式并列选择：MTP（`WEIGHTS` 中的层）、DFlash2（只有当 `glm53.conf` 的 `DRAFTER` 在每个节点上都有其模型时才提供，见上文“使用 DFlash2 的步骤”）或不用；选择后写入 `DRAFTER_KIND`。各自占用的内存不同：使用 DFlash2 时，启动器为每个节点比 MTP 多计划约 3 GiB（八个节点时约 4.3 GiB），因此会按所选的草稿模型重新规划。
- **上下文长度**（界面上的 Context length）也在启动界面上选择：128K、256K、512K、`CONTEXT_LENGTH` 或能容纳的最长值（最多 1,048,576 个 token）；计划内存容纳不下的长度会缩短到能容纳的长度。所选的值写入 `glm53.conf`（启动时写入计划采用的数字），下次打开时会选中该值；面板写入之前以最长值打开。
- **每个节点为操作系统保留的内存**（`RESERVE_GB`，默认 4 GiB，2 至 32）也在那里以 GiB 整数输入。此时 KV 缓存使用计划剩下的全部空间（上下文长度供单个请求使用，其余供之前的对话使用），启动界面以 token 数显示它。也可以改为输入 KV 大小，此时显示每个节点将保留的内存；重新输入保留的内存，KV 又会使用全部空间。`tf-glm53 start` 从 `glm53.conf` 取上下文长度和 KV 大小，`docker run` 这一行上的 `-e KV_SIZE=N` 为那次启动设置它。计划根据每个节点为操作系统保留的内存确定 KV 缓存的大小，启动时引擎自身的内存检查也保留同样的量（输入 KV 大小时为最小的 2 GiB）。KV 缓存在启动时分配，运行中不会增长。
- **节点**：引擎停止时，可以在启动界面上更改 worker 的地址；第一台节点保持不变。Check 会把本目录复制到这些节点（`tf-glm53 deploy`），并对该列表执行 `tf-glm53 check`；只有该列表通过检查后，Save 才会写入 `glm53.nodes.conf`，并在旁边保留之前的文件。`glm53.conf` 不会被改动。
- **权重**：启动界面的 [ WEIGHTS ] 指定权重（`WEIGHTS`）、各 bank 和 DFlash2 的模型；只有每个节点上都有它们所指的内容时，Check and save 才会写入（`glm53.conf` 中的 `WEIGHTS` 和 `DRAFTER`，`expert-plan/banks.json` 中各 bank 的目录）。
- **引擎设置界面**（右上角的开关）更改需要重启的设置，并提供 Restart to apply（`PANEL-GUIDE.zh-CN.md`）。
- **配置与使用记录**：引擎统计每个提示行和每个解码 token 选中的专家（`glm53.conf` 中的 `EXPERT_LIVE=1`，默认开启；从面板启动时始终统计）。未开启统计而启动的引擎（`EXPERT_LIVE=0`）仍会显示运行面板和“停止”，但不显示选择，也没有专家优化。计划的位宽作为配置保存在 `ATLAS_DIR/allocs/user/<名称>.json`（默认为本目录下的 `atlas/`），并带有保存它的那些运行的使用记录（运行开始时带有的记录加上该次运行的选择）。不会自动保存任何内容：停止时的“保存并停止”或运行中的“保存”会为其命名。默认名称是该次运行所基于的配置，并覆盖保存（旧版本移到 `ATLAS_DIR/allocs/trash/`）。启动界面以上一次运行的比例的新计划打开（还没有任何运行时为全部 3 bit）；“打开”会按位原样恢复上一次运行的配置或已保存的配置，从它启动时使用其位宽和保存时的 bank。计划按其来源配置的使用记录为专家排序，使用最多的专家得到最宽的位宽；没有记录的计划按页面自带的次数排序，这些次数是 Fiore-lab 在 12 类工作（编程、智能体、日文和英文文本、Linux 运维、安全）上测得的。
- **专家优化**在引擎运行时按本次运行的使用记录重新排列已载入的专家，每种位宽的专家数量保持不变，因此在相同内存下，被选中专家的平均位宽会提高。排序所用的次数在面板的 Count 中选择：decode、prefill 或 Both（Both 以 0.7 加权专家的 decode 占比、以 0.3 加权其 prefill 占比，这组权重最能保持模型所写文本的精度，见 `ATLAS-TESTS.zh-CN.md`）。在没有正在运行或等待的请求后才开始更改（在此之前页面会重试）；引擎在每个 rank 上就地更改各层，丢弃所保留的对话（它们的 KV 是用旧专家计算的），然后继续服务，期间到达的请求会等待；在四个节点上，75 层的完整更改用了 83 至 90 秒。位宽减小的层先换，增大的层后换；每个节点在之前各层增加的量之上，还需要最大一层的新副本以及 2 GiB（`TF_GLM53_SWAP_MARGIN_GIB`）的可用内存：估算会显示每个节点是否足够，不足时页面会给出该节点和所需的量并拒绝更改（下次启动时为操作系统保留更多内存即可腾出空间，因为 KV 会用掉计划留下的全部空间；输入 KV 大小时，以及不经面板、只用 glm53.conf 启动时，则减小 `KV_SIZE`）。超过 `TF_GLM53_SWAP_DEADLINE_S`（默认 600 秒）仍卡住的 rank 会结束其进程，`tf-glm53` 的监视进程会停止所有 rank。结果不会自动保存：用“保存”保留。启动只载入给定的位宽，不会重新排序。
- 设置了 `MULTI`（`glm53.conf`）时也一样：更改会等到没有正在运行或等待的请求。
- 页面的设置（配色、圆的排列方式等）是所有浏览器共用的一个文件：`launch/panel-settings.json`（可用 `EXTRA_ENV` 中的 `TF_PANEL_SETTINGS` 更改位置）。
- `glm53.conf` 中的 `LAUNCH_SNAPSHOTS=1` 会在每次启动和停止的前后，把每个节点的 GPU、内存和日志末尾写入 `snapshots/`；`LAUNCH_DROP_CACHES=1` 会在启动前于每个节点丢弃页面缓存（需要免密码的 `sudo`）。两者默认关闭。

## 当某个 rank 停止时

每次启动（`tf-glm53 engine-up`；从面板启动和 `tf-glm53 start` 都会执行它）都会留下一个监视进程，每 5 秒检查一次各 rank。如果一部分 rank 已退出而其他 rank 仍在运行（连续两次），它会停止所有 rank，因为剩下的 rank 会永远等待，并在它自己的日志中记下已退出的 rank（`tf-glm53 logs watch`）。请从面板或用 `tf-glm53 start` 重新启动。请只用 `tf-glm53` 启动和停止各 rank：手动启动的 rank 没有监视进程，若其中一个在启动时失败，其他 rank 会在集合通信中一直等待，直到被停止。

启动失败时（某个 rank 停止、服务器未在 `READY_TIMEOUT` 内就绪、冒烟测试失败，或按下 Ctrl-C），`tf-glm53 engine-up` 会在返回前停止所有 rank（从面板启动时由页面、用 `tf-glm53 start` 时由其输出说明原因）。每个节点保留其 rank 的日志以及上一次启动的日志；`tf-glm53 logs R` 持续跟踪 rank R 的日志（`R.prev`：上一次启动的）。

日志保存在内存中，不向磁盘写入任何内容：`tf-glm53 engine-up` 在每个节点上、`up` 在第一个节点上为启动器启动一个日志保管进程（`tools/logkeep.py`，只用标准库的 `python3`；它通过抽象 unix 套接字读写，因此自身也没有文件），每个 rank、监视进程、启动器以及 `tf-glm53` 本身的输出都进入其中。每条日志保留最近的 `LOG_KEEP_MB`（默认 8），较旧的行被丢弃；重启或保管进程停止时日志也会消失。保管进程在停止引擎后仍然运行，因此失败的启动的日志仍可读取；下一次启动时保留上一次启动的日志。保管进程只响应启动它的用户的连接（抽象套接字没有文件权限，因此它检查对方的 uid）。修改 `LOG_LISTEN` 或 `LOG_KEEP_MB` 后，引擎的下一次启动会重新启动保管进程使其生效（之前保存的行会丢失）。`tf-glm53 logs R | R.prev | watch | launcher` 跟踪其中一条。若要在别处长期保存，请设置 `LOG_LISTEN=HOST:PORT`：每个节点的保管进程还会以只读、无密钥的方式响应 `GET /streams`（所持有的日志）和 `GET /logs?stream=rank0&after=N`（第 N 行之后带编号的行），供另一台主机上的监控程序轮询并保存（“为其他机器提供服务”）。服务器不为监控或面板重复发出的请求（`/health`、`/metrics`、`/v1/experts/*`、`/v1/memory`）写访问行，因此日志保持很小。

每个 rank 以 `oom_score_adj` 1000 运行。节点内存耗尽时，内核的 OOM killer 会结束该 rank，监视进程再停止其余 rank，节点本身继续运行。（通过 ssh 启动的 shell 会继承 sshd 的 -1000；保持该值的 rank 不会被 OOM killer 选中，节点反而会卡死。）

## 在容器中运行

最短的方式是一行命令 `docker run ... IMAGE up`：见 `QUICKSTART.zh-CN.md`。本节是同样过程的手动步骤。

本发布也以容器镜像形式提供：`/opt/glm53` 中是归档的文件，另有服务器运行所需的内容（cuBLAS 13、libibverbs 及其 provider、带 AArch64 fence 的 NCCL 2.30.7）。在 `glm53.conf` 中设置 `RUNTIME=docker` 和 `IMAGE` 后，`tf-glm53` 会通过 ssh 在每个节点上把各 rank 作为 `IMAGE` 的容器启动。这时每个节点需要的是 Docker 和 NVIDIA Container Toolkit（DGX OS 两者都有），而不是 cuBLAS、libibverbs 和 NCCL；运行 `tf-glm53` 的用户必须能在每个节点上运行 `docker`（`docker` 组）。`tf-glm53`、控制面板的启动器和设置与使用归档时一样留在第一个节点的主机上；权重和专家 bank 也留在主机上。

```sh
I=ghcr.io/fiore-labo/expertatlas-glm5.3:r3.3.5-102
docker pull $I                                                # 在每个节点上
mkdir glm53 && docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/glm53":/out $I install /out
cd glm53 && cp glm53.conf.example glm53.conf                  # 或 glm53.t4.conf.example；设置 RUNTIME=docker 和 IMAGE=$I
./tf-glm53 deploy && ./tf-glm53 check && ./tf-glm53 up        # 控制面板：在那里启动引擎
```

- `install` 把本发布的文件（与解开的归档相同）写到该目录。`./tf-glm53 pull` 在每个节点上执行 `docker pull IMAGE`。
- 镜像也可以由归档构建：在刚解开的归档中执行 `docker build --build-context fenced=DIR -f docker/Dockerfile -t glm53-tensorfold:r3.3.5-102 .`，`DIR` 中放带 fence 的 NCCL（`RING.zh-CN.md`；没有它构建会失败）。只要有归档 `SHA256SUMS` 未列出的文件，构建就会失败，因此安装环境中的设置或日志不会进入镜像。用 `docker save` 和 `docker load` 复制到其他节点。
- `tf-glm53 check` 在每个节点上检查：`IMAGE` 存在且层与第一个节点相同；容器能访问 GPU（`--gpus all`）；镜像中是与本目录相同的发布（否则为 `NG: image-not-this-release`）。
- 每个 rank 的容器获得 GPU、`/dev/infiniband`、主机网络和 IPC、`--ulimit memlock=-1 --ulimit nofile=1048576:1048576`、`--oom-score-adj 1000`、主机用户和一个 init。以相同路径挂载：只读的 `WEIGHTS`、`DRAFTER`（DFlash2）以及该节点上 `expert-plan/banks.json` 中的每个 bank；可写的 `launch/`。Docker 不保存容器的日志（`--log-driver none`）：其输出进入节点的日志保管进程。容器名为 `glm53-p<PORT>-r<rank>`，停止后即删除。`tf-glm53 down`、`status`、监视进程和控制面板对它们的处理与从主机运行的 rank 相同。
- 引擎在交换机和环上都读取 `/opt/nccl-fenced` 的 NCCL。
- 四个节点上的测量（八节点检查点，上下文长度 131,072 个 token，MTP）：启动用了 4 分钟，与在主机上相同；冒烟测试通过；`down` 在 4 秒内停止所有 rank；强制终止一个 rank 的容器后，监视进程停止了其余 rank，43 秒后所有容器都已消失。
- 主机上的 `systemctl daemon-reload` 可能使运行中的 GPU 容器失去 GPU（容器内 `nvidia-smi` 失败）：执行 `tf-glm53 down`，再从面板（或用 `tf-glm53 start`）重新启动。

## 为其他机器提供服务

服务器默认监听所有地址（`HOST=0.0.0.0`）并提供 OpenAI chat API 和 Messages API（“Claude Code”），网络中的其他机器可以通过 `http://FIRST-NODE:PORT` 访问。API 没有密钥：任何能访问到它的人都可以使用（`tf-glm53 check` 会提示这一点）；设置 `HOST=127.0.0.1` 后只能从第一个节点访问。服务器使用明文 HTTP，因此请只在封闭网络中使用，或在其前面加一层 TLS 和访问控制（反向代理）。控制面板的启动器同样监听所有地址，且没有密钥（“控制面板”）。服务器自身的页面（`http://FIRST-NODE:PORT/panel`）只能在第一个节点本机上读取专家的次数和内存（`/v1/experts/*`、`/v1/memory`），因此从其他机器打开时只显示一条消息：请打开控制面板的页面（`http://FIRST-NODE:LAUNCHER_PORT/panel`），它的调用由启动器在第一个节点上发出。`/v1/thinking` 和 `/v1/schedule` 同样只在第一个节点上应答：从其他机器访问时，请把它们发送到启动器的端口。启动器的控制 API 见 `AGENT-API.zh-CN.md`。

其他主机上的监控程序无需访问 API 即可读取服务器状态：设置 `METRICS_LISTEN=HOST:PORT`（例如 `0.0.0.0:9894`）后，第一个节点也会在该地址上响应 `GET /metrics`（Prometheus 文本格式）、`GET /health` 和 `GET /v1/experts/usage`（专家的使用次数，不含文本），且只响应这些（其他请求一律返回 404），API 仍留在 `HOST` 上。设置 `LOG_LISTEN` 时，每个节点的日志保管进程也会应答其日志（见上文“当某个 rank 停止时”）：同时轮询两者的监控程序负责保存记录，节点不向磁盘写入任何内容。`tf-glm53 status` 会显示该地址。请只在封闭网络中开放此端口：它不显示提示词或回复，但会显示负载和 token 数。`tensorfold_health:*_total` 计数器包含正在运行的请求且不会回退，因此速度由两次读取的差值得出：prefill tokens/s = Δ`prefill_tokens_total` / Δ`prefill_seconds_total`，解码 = Δ`completion_tokens_total` / Δ`decode_seconds_total`，缓存命中率 = Δ`cached_tokens_total` / Δ`prompt_tokens_total`，草稿接受率 = Δ`accepted_total` / Δ`drafted_total`。`tensorfold:requests_prefilling` 和 `tensorfold:requests_decoding` 表示正在运行的请求所处阶段，`tensorfold:prefill_tokens_remaining` 表示其 prefill 剩余量，`tensorfold:pool_*` 表示对话缓存，`tensorfold:engine_info` 表示设置。

## Claude Code

服务器也响应 Anthropic 的 Messages API（`POST /v1/messages` 和 `/v1/messages/count_tokens`），因此 Claude Code 可以用它代替 Anthropic 的服务。本归档中的 `claude-glm53` 会让 Claude Code 连接到服务器并启动：

```sh
ENDPOINT=http://FIRST-NODE:8000 ./claude-glm53      # 后面可以接 Claude Code 自己的参数
```

它从服务器的 `/health` 读取模型和上下文长度，发送一个简短的请求，然后启动 Claude Code：所有模型别名（包括小型快速模型）都指向服务器的模型，上下文窗口为服务器的上下文长度减去为回复保留的 `MAX_OUT`（40,000），流的超时为 30 分钟。请在安装了 Claude Code 的机器上运行（还需要 `curl` 和 `python3`）；在 macOS 上，Claude Code 需要“本地网络”权限才能访问局域网地址。Anthropic 不支持通过网关让 Claude Code 使用其他模型：请自行承担风险。

服务器为它所做的事（无需设置）：

- 每个请求都由其内容块重新构建，因此 Claude Code 的缓存标记和两种内容写法会渲染出相同的提示；Claude Code 系统提示开头的归属行（`x-anthropic-billing-header`）的值随每个对话变化，因此会被去掉。实测：同一对话的后续请求有 99 % 以上的提示从对话缓存中读取，新对话的第一个请求约为 84 %。
- 读取长提示时不发送任何内容，因此服务器在每 10 秒的静默后发送一个 keep-alive 事件；Claude Code 在 300 秒没有收到时会断开流。
- 图片会变成注记 `[image omitted: this model reads text only]`（`tf-glm53` 设置 `TF_MESSAGES_OMIT_IMAGES=1`）：模型只读文本，拒绝图片会使包含它的对话之后的每个请求都被拒绝（Claude Code 用 Read 读取图片文件时会发送图片）。

限制：

- 默认情况下请求逐个处理（`MULTI`：见“同时处理多个对话”）。Claude Code 的后台请求（不思考、只有几个 token，例如为对话命名）会等待主请求，而且开头与主对话的请求不同，因此与它们不共享缓存；后台请求之间会共享。
- 结构化输出（`output_config.format` 的 JSON Schema）与 chat API 一样把答案限定为该 schema（`FEATURES.zh-CN.md` 的“结构化输出”），但不能与工具一起使用：带工具的请求要求它时会被拒绝（400），Claude Code 会在没有它的情况下继续。
- 思考强度：GLM-5.3 有 low、high 和 max；minimal 按 low 运行，medium 按 high 运行，xhigh 按 max 运行（Claude Code 发送 xhigh）。Claude Code 每次回复最多请求 32,000 个 token（`CLAUDE_CODE_MAX_OUTPUT_TOKENS` 可以提高）。思考的上限会把其余部分留给回答；`THINKING_SHARE=0` 时，在思考途中达到上限的回复没有正文。

## 速度

**测量使用发布版本 r2-102，每个节点的 GPU 时钟上限均设为 2.1 GHz（2,100 MHz）。** 引擎此后已有变化，这些数字尚未针对本发布重新测量。它们载入的是各检查点自身的位宽，而不是默认启动时所有专家都用的 3 bit。使用尚未缓存的编程提示词，一次一个请求（reasoning effort 为 low，回复最多 4,000 个 token）。八个节点：`glm53.conf.example` 的设置（FP8 KV 缓存，`CONTEXT_LENGTH` 为 819,200）和 `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks`（路由专家 4 bits，其他线性层 5 bits）。四个节点：当时的设置（4 bit KV 缓存，`CONTEXT_LENGTH` 为 737,280，`ROWS` 为 16,384；现在的示例为 1,048,576 和 8,192）和 `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks`。MTP 为两次运行的中位数；DFlash2 在八个节点上为一次运行，在四个节点上为两次运行的中位数。每轮模型前进的 token 数：MTP 为 3.4 至 3.5，DFlash2 为 4.6 至 4.8。速度因环境而异（检查点、提示词、时钟等），这些数字仅供参考。

| 节点数 | 提示词 | MTP（k=3）：首 token 时间 / 解码 | DFlash2：首 token 时间 / 解码 |
|---|---|---|---|
| 8 | 128K | 82.8 s / 47.5 tokens/s | 85.2 s / 53.2 |
| 8 | 256K | 172.9 s / 45.7 | 177.6 s / 52.6 |
| 8 | 512K | 376.5 s / 42.8 | 386.0 s / 47.1 |
| 4 | 128K | 120.1 s / 35.3 | 121.9 s / 42.9 |
| 4 | 256K | 253.3 s / 33.2 | 256.9 s / 40.7 |
| 4 | 512K | 565.4 s / 32.2 | 571.5 s / 37.8 |

## 出现问题时

`tf-glm53 check`（也是每次启动 `engine-up` 的第一步）在 `NG:` 之后列出每个节点的问题。如果某个节点的所有项都是 NG，说明第一个节点无法访问它：必须能从第一个节点对 `NODES` 中的每个地址成功执行 `ssh -o BatchMode=yes ADDRESS true`。

| `NG:` | 处理方法 |
|---|---|
| `files` | 再次执行 `tf-glm53 deploy` |
| `weights` | `WEIGHTS` 处缺少权重，或其文件名、大小或 `config.json` 与第一个节点上的不同；内容可用 `tf-glm53 verify` 校验 |
| `NCCL(...)` | 该节点的 `NCCL_LIB` 处没有 NCCL（见“首次启动之前”） |
| `cuBLAS-13` | 安装 CUDA 13 的 cuBLAS（`libcublas-13-0`） |
| `libibverbs` | 安装 `libibverbs1` 和 `ibverbs-providers` |
| RDMA 设备 | `IB_HCA` 在该节点上不是 RDMA 设备：`rdma link` 列出这些设备，`ibdev2netdev` 给出其端口名称 |
| `RoCE-v2-GID` | 该端口上没有带 IPv4 地址的 RoCE v2 条目：为该端口配置地址，或设置 `GID_INDEX` |
| `GPU-busy` | 另一个进程占用了 GPU（`nvidia-smi`） |
| `memory(N<M GiB)` | 释放该节点上的内存，或调低 `KV_SIZE` |
| `GPU-unreadable` | 该节点上 `nvidia-smi` 失败：检查驱动，如果一直如此，请重启该节点 |
| `python3(the log keeper)` | 在该节点上安装 `python3` |
| `rendezvous(...)` | `SOCKET_IFNAME` 在该节点上没有 IPv4 地址 |
| `lane-down(...)` | 环上：该节点上某条链路的一个半边处于 down 状态：检查线缆（`RING.zh-CN.md`） |
| `ring: NG: ...` | 连接方式不是环：消息会说明要修正什么（`RING.zh-CN.md`） |
| `NCCL-not-the-fenced-build(...)` | 环需要 fenced NCCL，并在 `NCCL_SHA256` 中写入其 sha256（`RING.zh-CN.md`） |
| `image-missing`、`image-differs-from-the-first-node` | `tf-glm53 pull`，或用 `docker save` 和 `docker load` 复制镜像 |
| `image-not-this-release` | 本目录和 `IMAGE` 是不同的发布：从 `IMAGE` 再次 `install`，或把 `IMAGE` 设为本发布 |
| `image-without-...` | 此镜像没有 fenced NCCL：使用发布的镜像，或用 `--build-context fenced=DIR` 构建（`RING.zh-CN.md`） |
| `docker-GPU(...)` | `--gpus all` 失败：检查该节点上的 NVIDIA Container Toolkit |
| `bank 3 bit: nothing at ...` | 把 3 bit bank 放到那里，每个节点都要：每次启动都需要它 |

启动失败时会停止所有 rank 并说明原因（从面板启动和 `tf-glm53 start` 时也见 `tf-glm53 logs launcher`）；每个节点都在内存中保留其 rank 的日志（`tf-glm53 logs R`）：

- `a rank stopped (N/M running)`：其上方的行会列出停止的 rank 及其日志中最后的错误。若出现关于连接的 `NCCL WARN` 行：使用交换机时，`NODES` 必须是 RoCE 网络上的地址，`SOCKET_IFNAME` 和 `IB_HCA` 必须是每个节点上该网络的名称，并且各节点之间必须能在该网络上通过任意端口互相访问（环：见 `RING.zh-CN.md`）。`cannot listen on`：该节点上的 `PORT` 已被占用。`PoolValuesDiffer`：各 rank 以不同的池设置启动，因而拒绝启动；请用 `tf-glm53` 从同一个 `glm53.conf` 启动所有 rank。
- `not ready after N s`：权重位于本地 NVMe 硬盘时，启动约需两分半钟；更慢的磁盘或网络存储需要更长时间：请调高 `READY_TIMEOUT`。始终无法完成的 rendezvous 也会一直等待：请确认第一个节点上的 `MASTER_PORT` 未被占用。
- `the smoke test failed`：失败的问题会显示它收到的回复；rank 0 的日志中记录了该请求。

服务运行期间，`tf-glm53 status` 显示每个 rank、其可用内存、上一次启动所用的草稿模型、对话缓存以及服务器的 `/health`。如果所有 rank 仍在运行但服务器不再应答，请执行 `tf-glm53 down` 和 `tf-glm53 start`：监视进程只会在部分 rank 已退出时才停止各 rank。如果某个 rank 拒绝了一次超出该请求自身块范围的池写入（日志中出现 `PoolWriteRefused`），所有 rank 会在该请求结束时一起停止；请从面板或用 `tf-glm53 start` 重新启动。

## 限制

- 支持四台或八台 GB10 系统（张量并行 4 或 8），使用交换机，或四台时使用环。两台和三台的配置正在测试中。不支持其他 GPU 型号，也不支持其他 rank 数量。
- 默认情况下一次只运行一个请求（见上文 `MULTI`）。其他请求最多 32 个按到达顺序等待；聊天标题生成请求（以及带 `"priority": "background"` 的请求）排在其他请求之后。超过 32 个时，请求会以 “the engine is busy; retry shortly” 被拒绝：非流式请求返回 503，流式请求则在流中收到一个错误事件。
- 使用 MTP 草稿模型时，在保留对话的时刻，回复最后一轮中被接受的行尚未写入 MTP 层的缓存。对话只保留到 MTP 层缓存完整之处（往前几个 token），并重新计算这几个 token。回复由模型本身决定。
- 无论使用哪种草稿模型，引擎的启动行都会显示 `drafts: on`：其含义是请求可以使用草稿。`tf-glm53` 会在其后打印实际使用的草稿模型。
- 容器需要 GPU、`/dev/infiniband`、宿主机网络、不受限制的锁定内存上限，以及较高的打开文件数上限（`--ulimit memlock=-1 --ulimit nofile=1048576:1048576`）。
