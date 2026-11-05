[English](BEFORE-START.md) | [日本語](BEFORE-START.ja.md) | 简体中文

# 启动前要决定的事项

必须决定的只有两项：在哪些节点上运行，以及权重放在哪里。其余项目都可以用默认值启动，出现 `STOP` 时按提示修复。启动步骤见 `QUICKSTART.zh-CN.md`。

## 启动前要准备和决定的事项

先决定节点和权重的位置。每个节点上的权重路径必须相同。

| 决定事项 | 可选项 | 默认值 | 选项 |
|---|---|---|---|
| 运行的节点 | 交换机上的 4 台或 8 台，或环上的 4 台（2 台和 3 台正在测试中） | 自动查找。只有交换机上空闲节点多于所需时才选择 | `NODES` |
| 检查点的位置（必需） | 4 节点 344.6 GB，或 8 节点 | `~/models/GLM-5.3-EXL3-4x-DGX-Sparks`（8 节点为 `-8x-`） | `WEIGHTS` |
| 3 bit bank 的位置（必需） | 277.9 GB | `~/models/GLM-5.3-EXL3-Experts-3bit` | `BANK_3` |
| 其他位宽的 bank | 2 bit 187.3 GB，4 bit 368.5 GB，5 bit 459.1 GB。只需要用到的位宽 | `~/models/GLM-5.3-EXL3-Experts-<N>bit` | `BANK_2`、`BANK_4`、`BANK_5` |
| DFlash2 的模型 | 仅在使用时需要。另有许可证 | 无 | `DRAFTER` |
| 安装目录 | 主目录下的目录 | `expertatlas` | `EA_DIR` |
| 端口 | API 和仪表板 | 8000 和 8010（API + 10） | `PORT`、`LAUNCHER_PORT` |
| API 的开放范围 | 所有地址，或仅第一个节点（`127.0.0.1`） | 所有地址（`0.0.0.0`） | `HOST` |

API 和仪表板都没有密钥。请把节点放在封闭的网络中。

以下不是需要决定的事项，但启动时也需要：每个节点约 115 GiB 空闲内存，GPU 上没有其他程序，没有口令的 ssh 密钥，能够使用 `docker`。

## 每次启动时要决定的事项

更改以下设置需要重启引擎。如果不确定，用默认值即可启动。

| 决定事项 | 选项 | 可选项 | 默认值 | 如何决定 |
|---|---|---|---|---|
| 专家的位宽 | `BITS` 或 `ALLOC` | 2、3、4、5 bit，或已保存的分配 | 3 bit | 位宽越大占用内存越多，接下来两项会减少 |
| 单次最大长度 | `CONTEXT_LENGTH` | 65,536 至 1,048,576 个 token，16,384 的倍数 | 737,280（4 节点，3 bit） | 需要处理的最长 prompt 加回复 |
| KV 总量 | `KV_SIZE` | token 数，或 `max` | `max` | 通常用 `max`。越多则缓存中保留的对话越多 |
| KV 格式 | `KV` | `fp4`、`fp8` | `fp4` | `fp8` 精度更高，但占用更多内存 |
| 草稿方式 | `DRAFTER_KIND` | `mtp`、`dflash2`、`none` | `mtp` | `dflash2` 需要另一个模型，并会减少 KV |
| 同时解码的对话数 | `MULTI` | 0 至 4 | 0（逐个处理） | 按同时使用的人数决定 |
| 等待时间 | `MULTI_WAIT` | 秒数 | 600 秒 | 使用 `MULTI` 时，超时的请求因繁忙被拒绝 |
| 留给系统的内存 | `RESERVE_GB` | 2 至 32 GiB | 4 GiB | 节点上还运行其他程序时调高 |
| 回复上限的默认值 | `MAX_TOKENS_SHARE` | `CONTEXT_LENGTH` 的 1 至 100% | 70% | 请求未指定 `max_tokens` 时使用。客户端的值优先 |
| Byte Resume | `TF_GLM53_BYTE_RESUME` | `on`、`log`、`off` | `on` | 防止切分差异导致缓存失效。通常用 `on` |
| API 中的模型名称 | `NAME` | 文本 | `glm-5.3` | 与客户端的设置保持一致 |

- **思考与 `max_tokens`：** 思考计入回复。客户端发送的 `max_tokens` 就是上限，值较小时思考也限于其 90%。
- **DFlash2、`MULTI` 与内存：** 在环上，`CONTEXT_LENGTH` 131,072、`RESERVE_GB` 16 时，计划的 KV 从 `mtp` 的 753,664 个 token 减少到 `dflash2` 加 `MULTI=4` 的 360,448 个 token。这是按估算规划的首次启动的值，之后的启动使用上次的实测值。
- **放不下时：** `CONTEXT_LENGTH` 过长时会缩短到能放下的长度，并在输出中说明。

## 启动前不必决定的事项

Reasoning Control（思考的上限和中途整理）和 GPU 的分配可以在运行中更改，可以边用边调整。

| 设置 | 选项 | 默认值 | 是否保留到下次启动 |
|---|---|---|---|
| 思考的上限 | `THINKING_SHARE` | `max_tokens` 与 context 剩余量中较小者的 90% | 保留 |
| 中途整理 | `THINKING_FOLD` | 不做（`off`） | 保留 |
| 整理的间隔 | `THINKING_FOLD_SHARE` | 到上限前不整理（100%）。可从 `max_tokens` 的 10% 起指定 | 保留 |
| 整理次数的上限 | `THINKING_FOLD_MAX` | 不限（0） | 保留 |
| 预算将尽时的整理 | `THINKING_FOLD_AT_BUDGET`、`THINKING_FOLD_MARGIN` | 开启，从预算结束前 10% 开始 | 保留 |
| 思考结束时的整理 | `THINKING_FOLD_AT_CLOSE_SHARE` | 不做（0） | 保留 |
| 向模型提出的文本 | `THINKING_FOLD_PROMPT`、`THINKING_CLOSE_PROMPT` | 内置文本 | 保留 |
| prefill 与解码的 GPU 分配（使用 `MULTI` 时） | `MULTI_DECODE_SHARE` | 解码 30% | 仅在仪表板中指定时保留 |
| 短 prompt 的处理（使用 `MULTI` 时） | `MULTI_SHORT_PROMPT`、`MULTI_SHORT_PREEMPT` | 8,192 个 token 以下，不抢先 | 仅在仪表板中指定时保留 |

可在推理界面的卡片和引擎设置界面中更改。AI 代理在启动器的端口上调用 `POST /v1/thinking` 和 `POST /v1/schedule`。

## 自动决定的事项

`TOPOLOGY`、`SOCKET_IFNAME` 和 `IB_HCA` 由节点检查根据连线决定。在 `docker run` 中指定会被拒绝。环也会自动决定 `NODES`。

## 如何指定

在 `docker run` 这一行中，在 `IMAGE` 之前加上 `-e 名称=值`。启动界面提供同样的选择。

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host \
  -e WEIGHTS=/data/models/checkpoint \
  -e BANK_3=/data/models/3bit \
  -e CONTEXT_LENGTH=131072 \
  IMAGE up
```

- **`docker run` 的指定：** 保存在 `~/expertatlas/glm53.conf` 中，之后的启动也会使用。只有 `KV_SIZE` 和 `BITS` 仅对那次启动有效。
- **启动界面的选择：** 只在那次启动期间有效，不会保存。更改默认值请使用引擎设置界面。
- **正在运行的引擎：** 选项不起作用。请先停止，再运行同一行。
- **名称写错：** 不认识的名称会导致 `STOP`，并提示最接近的名称。

每项的详细含义见 `QUICKSTART.zh-CN.md` 和 `CONFIG.zh-CN.md`。
