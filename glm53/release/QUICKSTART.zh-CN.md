[English](QUICKSTART.md) | [日本語](QUICKSTART.ja.md) | 简体中文

# 首次启动：从镜像到引擎运行

本页面向第一次启动 Expert Atlas 的人。Expert Atlas 在 NVIDIA DGX Spark 上运行完整版 GLM-5.3。下载权重，在第一个节点上运行一行 `docker run`，引擎就会启动。如果缺少什么，这一行命令会告诉你该怎么做。

## 什么在哪里运行

- **引擎**在每个节点的一个容器中运行。由 `docker run ... up` 启动。
- **启动器**在第一个节点的主机上运行（容器之外）。它提供仪表板和控制 API，并负责启动和停止引擎。
- **`up` 容器**只做准备工作，引擎运行后它就退出。引擎和启动器继续运行。
- **权重**放在主机上，不在镜像中。

仪表板和 API 都没有密钥，并且都对局域网开放。请把节点放在封闭的网络中。

## 1. 节点需要什么

| 项目 | 内容 |
|---|---|
| 节点 | 接在同一台 RoCE v2 交换机上的 4 台或 8 台 DGX Spark，或不用交换机连成环的 4 台（`RING.zh-CN.md`）。2 台和 3 台正在测试中 |
| 每个节点的软件 | Docker 和 NVIDIA Container Toolkit（DGX OS 都带有），以及 `python3` |
| ssh | 从第一个节点到每个节点，包括它自己，使用没有口令的密钥 |
| Docker 权限 | 该用户能在每个节点上运行 `docker`（`docker` 组） |
| 内存 | 启动前每个节点约有 115 GiB 空闲，GPU 上没有其他程序 |
| 磁盘 | 两项必需的下载，每个节点至少 622.5 GB |

发布的镜像包含带 AArch64 fence 的 NCCL，在交换机和环上都读取它（`RING.zh-CN.md` 的“fenced NCCL”）。

## 2. 下载权重

**必需的有两项：检查点和 3 bit bank。** 即使打算使用其他位宽，没有这两项引擎也不会启动。每个节点上的路径必须相同。

| 内容 | Hugging Face 仓库 | 大小 | 默认目录 | 选项 |
|---|---|---|---|---|
| 检查点，4 节点（**必需**） | `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks` | 344.6 GB | `~/models/GLM-5.3-EXL3-4x-DGX-Sparks` | `WEIGHTS` |
| 检查点，8 节点（**必需**） | `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks` | | `~/models/GLM-5.3-EXL3-8x-DGX-Sparks` | `WEIGHTS` |
| 3 bit bank（**必需**） | `fiore-lab/GLM-5.3-EXL3-Experts-3bit` | 277.9 GB | `~/models/GLM-5.3-EXL3-Experts-3bit` | `BANK_3` |
| 2 bit bank（可选） | `fiore-lab/GLM-5.3-EXL3-Experts-2bit` | 187.3 GB | `~/models/GLM-5.3-EXL3-Experts-2bit` | `BANK_2` |
| 4 bit bank（可选） | `fiore-lab/GLM-5.3-EXL3-Experts-4bit` | 368.5 GB | `~/models/GLM-5.3-EXL3-Experts-4bit` | `BANK_4` |
| 5 bit bank（可选） | `fiore-lab/GLM-5.3-EXL3-Experts-5bit` | 459.1 GB | `~/models/GLM-5.3-EXL3-Experts-5bit` | `BANK_5` |
| DFlash2（可选，另有许可证） | `incoai/GLM-5.3-DFlash2`（`RUNTIME.zh-CN.md` 中 DFlash2 一节） | | 无 | `DRAFTER` |

下载一次，再复制到其他节点。四个节点时：

```sh
python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install -U huggingface_hub
for r in EXL3-4x-DGX-Sparks EXL3-Experts-3bit; do
  D=~/models/GLM-5.3-$r
  ~/hf-cli/bin/hf download fiore-lab/GLM-5.3-$r --local-dir $D
  for n in NODE2 NODE3 NODE4; do    # 其他三个节点
    rsync -a --mkpath -e "ssh -c aes128-gcm@openssh.com" $D/ $n:$D/ &
  done; wait
done
```

权重已经在别的位置时，不必移动，用选项指定目录即可（第 4 节）。

### 位宽是什么

模型的路由专家可以以 2、3、4 或 5 bit 载入。**默认启动以 3 bit 载入所有专家。** 四个节点时，上下文长度 737,280 个 token 可以放下。用 `BITS` 可以让所有专家以其他位宽启动。在仪表板中，可以从已有的 bank 为每个专家选择位宽。

## 3. 启动

在第一个节点上，以运行它的用户执行：

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host IMAGE up
```

`IMAGE` 是发布的镜像 `ghcr.io/fiore-labo/expertatlas-glm5.3:r3.3-102`。这一行把发布内容写到 `~/expertatlas`，找到节点（环或交换机）并检查，查找权重，启动引擎并发送一次请求。首次启动需要几分钟。

最后有两种结果。

**`READY`：引擎正在运行。**

```
READY
  inference API       http://203.0.113.10:8000/v1 (OpenAI compatible, no key, model glm-5.3)
  dashboard (no key)  http://203.0.113.10:8010/panel
  control API         http://203.0.113.10:8010/v1/launch/status
```

**`STOP`：需要你处理。** `STOP` 说明问题，`FIX` 说明要做什么，`THEN` 给出接下来要运行的一行。启动器保持运行，所以仪表板也能打开。

```
STOP: the model's files are not on the nodes:
  3 bit (always needed): ~/models/GLM-5.3-EXL3-Experts-3bit: not on node-1, node-2 (...)
    put fiore-lab/GLM-5.3-EXL3-Experts-3bit there, the same path on every node (...)
FIX:  ...
  hf download fiore-lab/GLM-5.3-EXL3-Experts-3bit --local-dir ~/models/GLM-5.3-EXL3-Experts-3bit
THEN: docker run --rm --network host -e USER="$USER" -v "$HOME":/host IMAGE up
```

按 `FIX` 做完后，运行 `THEN` 这一行。再次运行总是安全的：它不会重启正在运行的引擎。

## 4. 选项

**启动设置中能选择的内容，都可以作为 `docker run` 的选项指定：** 在 `IMAGE` 之前加上 `-e 名称=值`。遇到不认识的名称时会停止，所以拼写错误不会被忽略。

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host \
  -e WEIGHTS=/data/models/checkpoint \
  -e BANK_3=/data/models/3bit \
  -e BANK_4=/data/models/4bit \
  -e BITS=4 -e CONTEXT_LENGTH=131072 \
  IMAGE up
```

文件的位置（每个节点上路径相同）：

| 选项 | 默认值 | 含义 |
|---|---|---|
| `WEIGHTS` | `~/models/GLM-5.3-EXL3-4x-DGX-Sparks`（8 节点为 `-8x-`） | 检查点。必需 |
| `BANK_3` | `~/models/GLM-5.3-EXL3-Experts-3bit` | 3 bit bank。必需 |
| `BANK_2`、`BANK_4`、`BANK_5` | `~/models/GLM-5.3-EXL3-Experts-<N>bit` | 其他 bank。可选 |
| `DRAFTER` | 空 | DFlash2 的模型，与 `DRAFTER_KIND=dflash2` 一起使用 |

启动：

| 选项 | 默认值 | 含义 |
|---|---|---|
| `BITS` | `3` | 所有专家使用此位宽（2 至 5）。每个节点上都必须有对应的 bank |
| `ALLOC` | 无 | 已保存的分配（每个专家的位宽和使用记录），代替 `BITS` |
| `NODES` | 自动查找 | 交换机上使用的 worker 地址，在空闲节点多于一条 lane 所需时指定。环会自己找到 |
| `CONTEXT_LENGTH` | `737280` | prompt 加回复的最大长度。过长时会缩短到能放下，并在输出中说明 |
| `KV_SIZE` | `max` | KV 缓存保存的 token 总数：一个数，或使用全部剩余内存的 `max`。只对本次启动有效 |
| `KV` | `fp4` | KV 缓存的格式：`fp4` 或 `fp8` |
| `DRAFTER_KIND` | `mtp` | `mtp`、`dflash2` 或 `none` |
| `MULTI`、`MULTI_WAIT` | `0`、`600` | 同时解码的对话数（0 至 4），以及请求等待它们的秒数 |
| `RESERVE_GB` | `4` | 每个节点留给系统的内存 |

引擎设置的其他项目也以同样方式指定，例如 `MAX_TOKENS`、`THINKING_SHARE`、`THINKING_FOLD_PROMPT`、`NAME`、`HOST`。每项的含义和默认值见 `CONFIG.zh-CN.md`。

安装：

| 选项 | 默认值 | 含义 |
|---|---|---|
| `EA_DIR` | `expertatlas` | 主目录下的目录 |
| `PORT` | `8000` | 推理 API |
| `LAUNCHER_PORT` | `PORT` + 10 | 仪表板和控制 API |
| `IMAGE` | 此镜像 | 节点拉取并运行的名称 |

需要知道：

- 指定的值保存在 `~/expertatlas/glm53.conf` 中，直到你指定其他值。`KV_SIZE` 和 `BITS` 只对那次启动有效。
- `TOPOLOGY`、`SOCKET_IFNAME` 和 `IB_HCA` 不是选项：节点检查根据连线决定它们。
- 选项不会改变正在运行的引擎。请先停止，再运行这一行。如果忽略了选项，输出中会说明。

## 5. 使用

API 使用 OpenAI 的聊天格式。把 `READY` 输出的地址填入客户端：

```sh
curl -s http://NODE:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3","messages":[{"role":"user","content":"hello"}]}'
```

仪表板 `http://NODE:8010/panel` 显示引擎的状态，并可启动和停止它。AI 代理可以通过控制 API 做同样的事：步骤见 `AGENT-API.zh-CN.md`，整个系统的说明见 `AI-GUIDE.zh-CN.md`。

## 6. 停止，再次启动

```sh
cd ~/expertatlas
./tf-glm53 down              # 停止引擎
./tf-glm53 launcher stop     # 也想停止启动器时
```

再次启动时，运行同一行 `docker run`。节点、权重的位置和你的设置都会保留。

## 7. 常见的 STOP

| 消息 | 要做什么 |
|---|---|
| `up needs the host's home at /host` | 照原样使用这一行：`-e USER="$USER" -v "$HOME":/host` |
| `this container cannot ssh to the host` | 生成没有口令的密钥，并加入自己的 `authorized_keys`。消息中有命令 |
| `IMAGE is not on NODE and docker pull failed` | 用消息中的 `docker save \| ssh NODE docker load` 复制 |
| `3 bit (always needed)`、`checkpoint (always needed)` | 下载（第 2 节），或用 `BANK_3`、`WEIGHTS` 指定目录 |
| `N free worker nodes found` | 用 `-e NODES="ADDR ADDR ADDR"` 选择 |
| `port N answers another lane's launcher` | 停止那个启动器，或加上 `-e PORT=N` |
| `does not fit with context length N` | 减小 `CONTEXT_LENGTH`、减小 `BITS`，或释放节点内存 |
| `not an option` | 检查拼写。消息会提示最接近的名称 |

更多内容：`BEFORE-START.zh-CN.md`（启动前要决定的事项）、`RUNTIME.zh-CN.md`（所有命令和故障）、`T4.zh-CN.md`（四个节点及其测量）、`RING.zh-CN.md`（环）、`CONFIG.zh-CN.md`（所有设置）。
