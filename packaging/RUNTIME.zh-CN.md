[English](RUNTIME.md) | [日本語](RUNTIME.ja.md) | 简体中文

# 从本归档运行完整版 GLM-5.3

本归档在四台或八台 NVIDIA DGX Spark 系统上以张量并行 4 或 8 提供完整版 GLM-5.3（`glm_moe_dsa`，755B）服务，每个 GB10 一个 rank。它不需要 Python、编译器或源代码检出。它是 Fiore-lab 基于 TensorFold 1.0.2 制作的非官方衍生版本，并非 TensorFold 项目的发布版本。本页同时适用于四个节点和八个节点；四个节点的不同之处及其测量结果见 `T4.zh-CN.md`。

## 归档内容

- `bin/tensorfold-native`：服务器，内含为 GB10（sm_121）构建的 CUDA 内核。Linux aarch64，glibc 2.28 或更高版本。
- `lib/libroce_proxy.so`：解码集合通信使用的 RDMA 代理（需要 `libibverbs.so.1`）。
- `tf-glm53`、`glm53.conf.example`（八个节点）和 `glm53.t4.conf.example`（四个节点）：用于启动、停止和检查各 rank。
- `RUNTIME.md`（本页）和 `T4.md`（四个节点），各自另有日文版（`.ja.md`）和简体中文版（`.zh-CN.md`）。
- `LICENSE`、`NOTICE`、`LICENSES/`、`THIRD_PARTY_NOTICES.md`、`VERSION`、`SHA256SUMS`。

其中不含模型权重、草稿模型，也不含 NCCL。

## 每个节点的要求

- DGX OS（Ubuntu 24.04），NVIDIA 驱动 580 或更高版本（由其提供 `libcuda.so.1`）。
- 加载器路径上有 cuBLAS 13（`libcublas.so.13`、`libcublasLt.so.13`）：DGX OS 会随 CUDA 13 一起安装它（`libcublas-13-0`）。
- 每个节点相同路径下的 NCCL 2.30.7（见下文）。
- `libibverbs.so.1`，以及连接各系统的同一个 RoCE v2 网络（四个节点时，每个节点一个 100 Gb/s 端口即可）；每个节点（包括第一台本身）都能从第一台通过其在 `NODES` 中的地址以 ssh 免密码访问。
- 检查点所需的磁盘空间，在每个节点上放在相同路径下（见下文）：八个节点为 390 GB（363 GiB），四个节点为 344.6 GB（321 GiB）。
- 八个节点：启动前每个节点需有 112 GiB 可用内存（使用 DFlash2 时为 115），以 `free -g` 的显示为准：当 `CONTEXT` 和 `STORE` 均为 819,200 时，各 rank 约使用 108 GiB（使用 DFlash2 时为 111）；使用 DFlash2 时，池（默认的对话缓存）每个 rank 比 slots（`CACHE=slots`）多用约 0.5 GiB。`STORE` 每减少 100,000 个位置，所需内存减少 2.7 GiB。可用内存不足的节点会被 `tf-glm53 check` 拒绝。
- 四个节点：内存按只运行本服务器来分配（见 `T4.zh-CN.md`）；使用 DFlash2 时测得的最低可用内存为每个节点约 3.2 GiB（512K 时）。

## 首次启动之前

NCCL：NVIDIA 以 Python wheel 的形式发布 NCCL 2.30.7。pip（软件包 `python3-pip`）只负责解包该库，没有任何组件会运行 Python。在每个节点上执行：

```sh
pip3 install --no-deps --target ~/nccl-2.30.7 nvidia-nccl-cu13==2.30.7
```

`glm53.conf` 中的 `NCCL_LIB` 默认指向 `~/nccl-2.30.7/nvidia/nccl/lib/libnccl.so.2`。从源代码构建的 NCCL 2.30.7（NVIDIA 的 DGX Spark playbook “NCCL for Two Sparks”，<https://build.nvidia.com/spark/nccl>）同样可用：将 `NCCL_LIB` 设为它即可。

检查点：Hugging Face 上与节点数相符的 Fiore-lab EXL3 检查点，八个节点用 `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks`，四个节点用 `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks`。用 Hugging Face CLI 下载一次（`python3 -m venv` 需要 `python3-venv` 软件包），然后通过 RoCE 网络复制到其他节点。八个节点时：

```sh
python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install -U huggingface_hub
D=~/models/GLM-5.3-EXL3-8x-DGX-Sparks
~/hf-cli/bin/hf download fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks --local-dir $D
for n in NODE2 NODE3 NODE4 NODE5 NODE6 NODE7 NODE8; do    # the other seven addresses in NODES
  rsync -a --mkpath -e "ssh -c aes128-gcm@openssh.com" $D/ $n:$D/ &
done; wait
```

四个节点时，`T4.zh-CN.md` 给出了针对四节点检查点和其余三个节点的相同命令。

要校验副本，请在每个节点上运行 `cd ~/models/GLM-5.3-EXL3-8x-DGX-Sparks && sha256sum -c --quiet SHA256SUMS`：每个节点需几分钟，所有文件都一致时不输出任何内容（部署本目录之后，`./tf-glm53 verify` 可一次校验所有节点）。`glm53.conf` 中的 `WEIGHTS` 指定该目录（此处为 `~/models/GLM-5.3-EXL3-8x-DGX-Sparks`）。

## 启动

```sh
tar xzf tensorfold-VERSION-linux-aarch64.tar.gz
cd tensorfold-VERSION-linux-aarch64
sha256sum -c SHA256SUMS
cp glm53.conf.example glm53.conf    # nodes, network names, NCCL, weights, drafter, port
./tf-glm53 deploy                   # the same directory on the other nodes
./tf-glm53 up                       # checks, starts the ranks, waits, runs the smoke test
./tf-glm53 -d dflash2 up            # another start with the DFlash2 drafter (Drafters, below)
./tf-glm53 down
```

四个节点时，复制 `glm53.t4.conf.example` 而不是 `glm53.conf.example`（见下一节）。`tf-glm53` 既不使用 `sudo`，也不更改系统设置。它的冒烟测试检查服务器能否应答、能否复用已缓存的提示词；它不衡量精度。

## 四节点与八节点的设置

`tf-glm53` 根据 `NODES` 中的地址个数以张量并行 4 或 8 运行。两个设置示例的区别在于下列各行；网络名称、NCCL、草稿模型和服务器在两者中的设置方式相同：

| 设置 | `glm53.conf.example`（八个节点） | `glm53.t4.conf.example`（四个节点） |
|---|---|---|
| `NODES` | 八个地址，rank 0 在前 | 四个地址，rank 0 在前 |
| `WEIGHTS` | `~/models/GLM-5.3-EXL3-8x-DGX-Sparks` | `~/models/GLM-5.3-EXL3-4x-DGX-Sparks` |
| `KV` | `fp8`（默认） | `fp4` |
| `CONTEXT` | 819200 | 737280 |
| `STORE` | 819200 | 100000 |
| `CACHE` | `pool`（默认） | `pool` |
| `DCP` | 2（默认） | 2 |
| `ROWS` / `SP_PARTS` | 8192 / 4（默认） | 16384 / 4 |

`KV` 是 KV 缓存的格式：FP8 或 4 bit。`DCP` 是解码上下文并行，`ROWS` 和 `SP_PARTS` 分别是提示词分块的行数和每块拆分成的序列并行段数。`CONTEXT`、`STORE` 和 `CACHE` 见下文“请求之间保留的对话”。四个节点需要四节点检查点、`KV=fp4` 以及 `glm53.t4.conf.example` 中的取值；详见 `T4.zh-CN.md`。

## 第一个请求

服务器在 `http://HOST:PORT/v1` 上提供 OpenAI chat API，模型名由 `NAME` 指定（在 `glm53.conf.example` 中为 `glm-5.3`，端口 8000）。在第一个节点上：

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3", "max_tokens": 300,
  "messages": [{"role": "user", "content": "Say hello in one short sentence."}]}'
```

回复的 `message` 中，`reasoning_content` 是模型的推理过程，其后 `content` 是答案；`max_tokens` 同时计入两者。从其他机器访问时，在已设置 `HOST` 和 `API_KEY_FILE` 的前提下，将请求发送到 `http://HOST:PORT`，并加上 `-H "Authorization: Bearer KEY"`。

## 草稿模型

引擎每轮起草几个 token，由模型进行校验；无论哪种情况，回复都由模型本身决定。`glm53.conf` 中的 `DRAFTER_KIND` 选择由什么来起草。在命令前加 `-d KIND` 可以不改文件、只为这一次启动选择（`./tf-glm53 -d dflash2 up`、`./tf-glm53 -d mtp up`），`./tf-glm53 status` 会显示上一次启动所用的草稿模型：

- `mtp`（默认）：检查点自带的 MTP 层，属于 GLM-5.3 权重的一部分，适用其许可证。它要求检查点包含该层；`tf-glm53 check` 会拒绝不含该层的检查点。`MTP_K` 为每轮的草稿数，默认 3（四个并不比三个快）。
- `dflash2`：incoai/GLM-5.3-DFlash2，位于 `DRAFTER` 指定的目录。**由于其许可证，本归档和仓库均不包含它**：其模型卡注明的许可证为 CC BY-NC-ND 4.0（非商业性使用，禁止演绎），用于研究和评估；商业许可由 Inco AI 提供。如果你的使用在该许可证允许的范围内，请按下文“使用 DFlash2 的步骤”进行设置；这样解码会更快（见“速度”）。
- `none`：不使用草稿。

### 使用 DFlash2 的步骤

1. 阅读模型卡（<https://huggingface.co/incoai/GLM-5.3-DFlash2>）上的许可证，确认你的使用在其允许的范围内。引擎按原样读取公开的文件：启动时，每个 rank 取出草稿模型权重中由自己负责的部分，并在内存中量化为 4 bit。不需要事先转换，也不会向磁盘写入任何内容。
2. 用为检查点准备的 Hugging Face CLI，在第一个节点上下载一次，并固定为本归档核对过的 revision。无需账号或 token；其权重为一个文件 `model.safetensors`（4,918,859,112 字节）。请使用单独的目录（`--local-dir`），而不是 Hugging Face 缓存：`DRAFTER` 所指的目录中必须直接包含 `config.json` 和 `model.safetensors`。

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

   如果设置了 `DRAFTER` 而 `DRAFTER_KIND` 仍为 `mtp`，`./tf-glm53 -d dflash2 up` 会只在这一次启动中使用 DFlash2。使用 DFlash2 时不使用 `MTP_K`。八个节点时，启动前每个节点需要 115 GiB 可用内存，而不是 112 GiB（见“每个节点的要求”）；`tf-glm53 check` 会检查这一点。使用池时，最近对话的草稿模型状态也会被保留（`DRAFT_SAVES`，见“请求之间保留的对话”）。
6. 照常启动：如果正在运行，先执行 `./tf-glm53 down`，再执行 `./tf-glm53 up`。`tf-glm53 check` 和 `up` 会输出 `drafter: dflash2 (PATH)`；无论使用哪种草稿模型，引擎自身的启动行都显示 `drafts: on`。
7. 要确认草稿是否起作用，请在一次请求的前后读取 `curl -s http://127.0.0.1:8000/health`（无需密钥）中的 `completion_tokens_total` 和 `rounds_total`。前者的增量除以后者的增量，就是该请求每轮确定的 token 数：为 1 表示没有采用任何草稿。对于与“速度”一节类似的编程提示词，DFlash2 为 4.3 到 5.0。

要改回 MTP 层，请设置 `DRAFTER_KIND=mtp`，然后执行 `./tf-glm53 down` 和 `./tf-glm53 up`。

## 请求之间保留的对话

`CONTEXT` 是单个请求最多可达到的位置数（提示词加回复）。`STORE` 是在此之外为先前对话保留的位置数。`glm53.conf` 中的 `CACHE` 选择保留方式：

- `pool`（默认）：一个容纳 `CONTEXT` + `STORE` 个位置的 KV 池，按 256 个位置分块；每个对话的 KV 是这些块组成的表。请求会从任意对话（不只是紧接其前的那个）中已缓存的最长前缀接着处理。同一对话的下一轮接着使用该对话自己的块。开头与先前某个对话相同的新对话（相同的系统提示词，例如有些智能体框架每次请求都会发送约 27K token 的系统提示词；子智能体也是如此）会共享先前对话中位于两者分歧点之前的完整块，这些块只保存一份，并且只从那一块开始计算。池满时，最久未使用的对话先从末尾交出块。
- `slots`：按对话的保存区：在窗口之外为先前对话保留 `STORE` 个位置。

每个已缓存的块在复用之前都会与请求的 token 进行核对（通过抽样行的哈希）。只要有任何疑点，就丢弃所有对话，从头计算该请求（“cold”）；绝不会错误地复用某个块。使用 DFlash2 时，池还会保留最多 `DRAFT_SAVES` 个（默认 8 个）对话的草稿模型状态，使这些对话的下一轮也能以全速起草。无论哪种方式，回复都由模型本身决定；从缓存继续处理的请求，与从位置 0 计算的请求相比，最后几位可能不同（舍入顺序不同），这一点在 slots 下也一样。

`tf-glm53 up` 和 `status` 会显示池的情况；采用四节点设置时如下：

```text
conversation cache (pool): 837280 positions in 3271 blocks of 256, shared by up to 64 conversations (context 737280)
```

此时服务器的 `/health` 中有一个 `pool` 对象：`blocks`、`free_blocks`、`slots`（保留的对话数）、`shared_blocks`（由多个对话共同持有的块数）、`evicted_total`（为腾出空间而丢弃的对话数）和 `cold_total`（从头计算的请求数）。

## 当某个 rank 停止时

`tf-glm53 up` 会留下一个监视进程，每 5 秒检查一次各 rank。如果一部分 rank 已退出而其他 rank 仍在运行（连续两次），它会停止所有 rank，因为剩下的 rank 会永远等待，并在 `logs/watch.log` 中记下已退出的 rank。请用 `tf-glm53 up` 重新启动。请只用 `tf-glm53` 启动和停止各 rank：手动启动的 rank 没有监视进程，若其中一个在启动时失败，其他 rank 会在集合通信中一直等待，直到被停止。

启动失败时（某个 rank 停止、服务器未在 `READY_TIMEOUT` 内就绪、冒烟测试失败，或按下 Ctrl-C），`tf-glm53 up` 会在返回前停止所有 rank。每个节点将其 rank 的日志保存在 `logs/rankR.log` 中，上一次启动的日志保存在 `logs/rankR.log.prev` 中；`tf-glm53 logs R` 持续跟踪 rank R 的日志。

## 为其他机器提供服务

服务器默认监听 `127.0.0.1`，提供 OpenAI chat API。要为其他机器提供服务，请设置 `HOST` 和 `API_KEY_FILE`；对于没有密钥的非回环地址，`tf-glm53` 会拒绝。服务器使用明文 HTTP：请在其前面加一层 TLS，或只在封闭网络中使用，否则密钥将以明文传输。

API 密钥文件只在第一个节点上读取。文件中每行一个密钥（`KEY` 或 `LABEL: KEY`，`#` 开始注释），且必须是其他用户无法读取的普通文件（`chmod 600`），否则服务器拒绝启动；`tf-glm53 check` 会检查这两点。`tf-glm53 deploy` 不会把它复制到其他节点。客户端发送 `Authorization: Bearer KEY`。

## 速度

**测量时每个节点的 GPU 时钟上限均设为 2.1 GHz（2,100 MHz）。** 使用尚未缓存的编程提示词，一次一个请求（reasoning effort 为 low，回复最多 4,000 个 token），本归档采用 `CACHE=pool`。八个节点：`glm53.conf.example` 的设置（FP8 KV 缓存，`CONTEXT` 为 819,200）和 `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks`（路由专家 4 bits，其他线性层 5 bits）。四个节点：`glm53.t4.conf.example` 的设置（4 bit KV 缓存，`CONTEXT` 为 737,280）和 `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks`。MTP 为两次运行的中位数；DFlash2 在八个节点上为一次运行，在四个节点上为两次运行的中位数。每轮模型前进的 token 数：MTP 为 3.4 至 3.5，DFlash2 为 4.6 至 4.8。你的结果取决于检查点、提示词和时钟。

| 节点数 | 提示词 | MTP（k=3）：首 token 时间 / 解码 | DFlash2：首 token 时间 / 解码 |
|---|---|---|---|
| 8 | 128K | 82.8 s / 47.5 tokens/s | 85.2 s / 53.2 |
| 8 | 256K | 172.9 s / 45.7 | 177.6 s / 52.6 |
| 8 | 512K | 376.5 s / 42.8 | 386.0 s / 47.1 |
| 4 | 128K | 120.1 s / 35.3 | 121.9 s / 42.9 |
| 4 | 256K | 253.3 s / 33.2 | 256.9 s / 40.7 |
| 4 | 512K | 565.4 s / 32.2 | 571.5 s / 37.8 |

## 出现问题时

`tf-glm53 check`（也是 `up` 的第一步）在 `NG:` 之后列出每个节点的问题。如果某个节点的所有项都是 NG，说明第一个节点无法访问它：必须能从第一个节点对 `NODES` 中的每个地址成功执行 `ssh -o BatchMode=yes ADDRESS true`。

| `NG:` | 处理方法 |
|---|---|
| `files` | 再次执行 `tf-glm53 deploy` |
| `weights` | `WEIGHTS` 处缺少检查点，或其文件名、大小或 `config.json` 与第一个节点上的不同；内容可用 `tf-glm53 verify` 校验 |
| `NCCL(...)` | 该节点的 `NCCL_LIB` 处没有 NCCL（见“首次启动之前”） |
| `cuBLAS-13` | 安装 CUDA 13 的 cuBLAS（`libcublas-13-0`） |
| `libibverbs` | 安装 `libibverbs1` 和 `ibverbs-providers` |
| RDMA 设备 | `IB_HCA` 在该节点上不是 RDMA 设备：`rdma link` 列出这些设备，`ibdev2netdev` 给出其端口名称 |
| `RoCE-v2-GID` | 该端口上没有带 IPv4 地址的 RoCE v2 条目：为该端口配置地址，或设置 `GID_INDEX` |
| `GPU-busy` | 另一个进程占用了 GPU（`nvidia-smi`） |
| `memory(N<M GiB)` | 释放该节点上的内存，或调低 `STORE` |
| `API_KEY_FILE ...` | 按该行的提示处理 |

`up` 失败时会停止所有 rank 并说明原因；每个节点都保留其 rank 的日志（`logs/rankR.log`，`tf-glm53 logs R`）：

- `a rank stopped (N/M running)`：其上方的行会列出停止的 rank 及其日志中最后的错误。若出现关于连接的 `NCCL WARN` 行：`NODES` 必须是 RoCE 网络上的地址，`SOCKET_IFNAME` 和 `IB_HCA` 必须是每个节点上该网络的名称，并且各节点之间必须能在该网络上通过任意端口互相访问。`cannot listen on`：该节点上的 `PORT` 已被占用。`PoolValuesDiffer`：各 rank 以不同的池设置启动，因而拒绝启动；请用 `tf-glm53` 从同一个 `glm53.conf` 启动所有 rank。
- `not ready after N s`：检查点位于本地 NVMe 硬盘时，启动约需两分半钟；更慢的磁盘或网络存储需要更长时间：请调高 `READY_TIMEOUT`。始终无法完成的 rendezvous 也会一直等待：请确认第一个节点上的 `MASTER_PORT` 未被占用。
- `the smoke test failed`：失败的问题会显示它收到的回复；rank 0 的日志中记录了该请求。

服务运行期间，`tf-glm53 status` 显示每个 rank、其可用内存、上一次启动所用的草稿模型、对话缓存以及服务器的 `/health`。如果所有 rank 仍在运行但服务器不再应答，请执行 `tf-glm53 down` 和 `tf-glm53 up`：监视进程只会在部分 rank 已退出时才停止各 rank。如果某个 rank 拒绝了一次超出该请求自身块范围的池写入（日志中出现 `PoolWriteRefused`），所有 rank 会在该请求结束时一起停止；请用 `tf-glm53 up` 重新启动。以 `DCP=1` 启动时，日志中会警告提示词分块不按序列并行分段处理（此时不使用 `SP_PARTS`）。

使用 `CACHE=slots` 时，如果启动时某个 rank 没有足够内存容纳 `STORE` 个位置，`up` 和 `status` 会显示 `slot cache: WARNING: no store`：服务器仍会应答，但请求只能从紧接其前的对话继续。请释放节点上的内存或调低 `STORE`，然后执行 `tf-glm53 down` 和 `tf-glm53 up`。

## 限制

- 仅支持四台或八台 GB10 系统（张量并行 4 或 8）；不支持其他 GPU 型号，也不支持其他 rank 数量。四个节点需要四节点检查点和 `KV=fp4`。
- 一次只运行一个请求。其他请求最多 32 个按到达顺序等待；聊天标题生成请求（以及带 `"priority": "background"` 的请求）排在其他请求之后。超过 32 个时，请求会以 “the engine is busy; retry shortly” 被拒绝：非流式请求返回 503，流式请求则在流中收到一个错误事件。
- 使用 MTP 草稿模型时，在保留对话的时刻，回复最后一轮中被接受的行尚未写入 MTP 层的缓存。使用 `CACHE=slots` 时，之后从该处继续的请求在这几个位置上的起草效果会略差；使用池时，对话只保留到 MTP 层缓存完整的位置（往前几个位置），并重新计算这几个位置。无论哪种情况，回复都由模型本身决定。
- 无论使用哪种草稿模型，引擎的启动行都会显示 `drafts: on`：其含义是请求可以使用草稿。`tf-glm53` 会在其后打印实际使用的草稿模型。
- 在容器中，`tf-glm53 check` 看不到宿主机或其他容器的 GPU 进程；但可用内存检查仍会阻止在繁忙节点上启动。容器需要 GPU、`/dev/infiniband`、宿主机网络、不受限制的锁定内存上限，以及较高的打开文件数上限（`--ulimit memlock=-1 --ulimit nofile=1048576:1048576`）。
