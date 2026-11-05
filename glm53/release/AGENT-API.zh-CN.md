[English](AGENT-API.md) | [日本語](AGENT-API.ja.md) | 简体中文

# ExpertAtlas 代理操作 API 参考

- 读者：不使用仪表板操作 ExpertAtlas 的 AI 代理，以及使用该代理的人
- 范围：查看状态、服务器设置、引擎设置、启动、推理、检查缓存、停止
- 范围外：专家优化（Optimise、Apply、`/v1/launch/resort`、`/v1/experts/swap`）

本文档汇总了 AI 代理按步骤操作所需的内容。所有设置项见 `CONFIG.zh-CN.md`。

## 0. 从镜像到启动

在第 1 个节点上，以运行它的用户执行：

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host IMAGE up
```

这一行把发布内容写到 `~/expertatlas`，在主机上启动启动器，然后依次完成节点登记、权重检查、计划、启动和 1 次推理。这个容器退出后，启动器和引擎继续运行。**检查点和 3 bit bank 始终必需；未指定时，所有专家都以 3 bit 启动。**

- 出现以 `READY` 开头的行，就表示已经启动。后面的行给出仪表板和 API 的地址。
- 出现 `STOP` 时，它说明停止的原因。`FIX` 说明修复方法，`THEN` 给出重新执行的 1 行命令。修复后，照原样运行 `THEN` 这一行。启动器保持运行，所以本文档的 API 也可以使用。
- 启动设置都可以用 `-e 名称=值` 指定：`WEIGHTS`、`BANK_2` 至 `BANK_5`、`DRAFTER`、`BITS`、`ALLOC`、`NODES`、`PORT`、`LAUNCHER_PORT`、`EA_DIR`、`IMAGE`，以及第 4 节的引擎设置（`TOPOLOGY`、`SOCKET_IFNAME`、`IB_HCA` 由节点检查决定，不能指定）。不认识的名称会导致 `STOP`。列表见 `QUICKSTART.zh-CN.md`。
- 主机需要：NVIDIA 驱动 580 或更高版本、Docker 和 NVIDIA Container Toolkit、`python3`、ConnectX-7 网络的地址，以及所有节点之间和到自身的没有口令的 ssh，并已登记它们的 host key（`QUICKSTART.zh-CN.md` 第 1 节）。
- 发布的镜像包含带 AArch64 fence 的 NCCL，交换机和环都读取它（`RING.zh-CN.md`）。
- 如果已在运行，则不启动任何内容，只做 1 次推理并输出 `READY`。

## 1. 首先要了解的事项

### 1.1 有 2 个端口

| 角色 | 设置项 | 默认值 | 用法 |
|---|---|---|---|
| 引擎 | `PORT` | 8000 | 推理发到这里。停止时不响应 |
| 启动器 | `LAUNCHER_PORT` | `PORT` + 10 | 所有操作发到这里。引擎停止时也响应 |

- 先用 `./tf-glm53 launcher start` 启动一次启动器。API 无法启动它。`./tf-glm53 launcher status` 显示状态和 URL。
- 从局域网直接在引擎端口上调用 `/v1/thinking`、`/v1/schedule`、`/v1/pool`、`/v1/experts/*` 会返回 403。**请在启动器端口上调用。** 启动器从 127.0.0.1 转发给引擎。
- 启动器没有密钥。只在封闭的网络中使用。

### 1.2 操作前必须确认的事项

1. **该端口是否属于目标集群。** 可能有另一个目录使用相同的端口。`GET /v1/launch/config` 的 `conf_path` 是设置文件的绝对路径（`GET /v1/launch/settings` 的 `conf` 只是文件名）。也可以在节点上这样确认：
   ```bash
   ss -ltnp | grep ":$PORT "          # 读取 pid
   readlink /proc/$PID/cwd             # 该 lane 的目录
   ```
   节点通过 `GET /v1/launch/config` 的 `nodes` 和 `names` 确认。刚登记后 `names` 为空，几秒后恢复。
2. **是否有推理正在运行。** 查看 `GET /health` 的 `live.connections` 和 `progress.phase`。
3. **停止前是否已保存。** 不保存就停止，会丢失这次运行的使用记录（第 7 节）。

### 1.3 禁止的操作

- 在没有人要求的情况下，停止或重启生产引擎。
- 停止或重启正在运行你自己的引擎。工具运行期间 `live.connections` 看起来是 0，所以即使看起来空闲也不能停止。
- 把 `GET /v1/launch/status`、`GET /v1/launch/usage`、`GET /v1/launch/alloc`、`GET /v1/pool` 的结果原样放进上下文。其中的 base64 有几万到几十万个字符（`alloc` 实测 435,780 个字符）。只取出需要的项目（1.5）。

### 1.4 异步操作

耗时的操作中，POST 立即响应（202 或 200），处理在后台继续。连接断开也不会停止。用 GET 读取状态。

| POST | 读取的 GET | 完成的标志 | 实测 |
|---|---|---|---|
| `/v1/launch/start` | `/v1/launch/status` | `state` 为 `running` 或 `failed` | 100 秒（权重已在内存中时）。需要读取时 10 分钟以上 |
| `/v1/launch/stop` | `/v1/launch/status` | `state` 为 `stopped` 或 `failed` | 6 秒 |
| `/v1/launch/restart` | `/v1/launch/status` | `restart` 为 false 且 `state` 为 `running`，或 `restart` 为 false 且有 `error` | 与启动相同 |
| `/v1/launch/check` | `/v1/launch/check` | `state` 为 `done` | 2 至 4 秒（`deploy: false`） |
| `/v1/launch/discover` | `/v1/launch/discover` | `state` 为 `done` | 1 至 3 秒 |
| `/v1/launch/weights/check` | `/v1/launch/weights` 的 `check` | `check.state` 为 `done` | 少于 5 秒 |

读取间隔 2 至 10 秒即可。

### 1.5 工具

所有示例都使用这些函数。`LP` 是启动器端口，`EP` 是引擎端口。

```python
import base64, json, time, urllib.error, urllib.request

def call(port, method, path, body=None, host="127.0.0.1", timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://{host}:{port}{path}", data,
                                 {"Content-Type": "application/json"}, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}
    except urllib.error.URLError as e:      # 已停止的引擎等
        return 0, {"error": str(e.reason)}

def brief(status):
    """去掉 base64 后的 status"""
    return {k: v for k, v in status.items() if k not in ("done", "experts_done", "bits", "banks", "weights")}

def stream(port, body, host="127.0.0.1", timeout=7200):
    """读取 SSE。(思考, 回答, usage, reasoning_details, finish_reason)"""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"}, method="POST")
    think, text, usage, details, finish = [], [], {}, None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            for ch in d.get("choices") or []:
                de = ch.get("delta") or {}
                think.append(de.get("reasoning_content") or "")
                text.append(de.get("content") or "")
                details = de.get("reasoning_details") or details
                finish = ch.get("finish_reason") or finish
    return "".join(think), "".join(text), usage, details, finish
```

节点的日志用 `./tf-glm53 logs 0` 查看，但**它会持续跟随，不会结束。** 必须写成 `timeout 10 ./tf-glm53 logs 0 | grep ...` 的形式，并在 lane 的目录中读取。

## 2. 读取状态

| 想了解的内容 | 调用 | 查看的项目 |
|---|---|---|
| 是否在运行、启动进度 | `GET /v1/launch/status` | `state`（stopped, starting, running, stopping, failed）、`phase`、`layers_done`、`eta_ms`、`error`、`alloc` |
| 当前的启动选择 | `GET /v1/launch/status` | `kv.format`、`kv.context`、`kv.store`、`kv.reserve_gib`、`drafter.running`、`multi.running` |
| 哪些节点 | `GET /v1/launch/config` | `nodes`、`names`、`world`、`topology`、`registered` |
| 各节点的内存和温度 | `GET /v1/launch/nodes` | `nodes[].mem_available`、`engine`、`other`、`gpu_temp_c` |
| 可用的位宽和草稿模型 | `GET /v1/launch/status` | `banks.available`、`banks.checking`、`drafter.available`、`drafter.why` |
| 引擎的状况 | `GET /health`（引擎端口） | `model`、`context_length`、`live`、`progress`、`thinking` |
| 思考的设置 | `GET /v1/thinking` | 5.3 |
| 已保存的分配 | `GET /v1/launch/allocs` | `allocs[].name`、`valid`、`problem`、`usage` |

- **运行中的值和设置文件中的值是两回事。** `GET /v1/launch/settings` 返回设置文件中的值。启动时传入选择后，即使 `pending` 为空，值也可能不同。运行中的值看 `status` 的 `kv.*` 和 `/health` 的 `context_length`。
- 停止后，`status` 的 `kv`、`drafter.running`、`multi.running` 仍是上次的值。
- 引擎停止时，`/v1/thinking` 和 `/v1/schedule` 经启动器返回 503。设置用 `settings` 读取。

## 3. 服务器设置（节点和权重）

节点的登记和权重的位置。**两者都只能在引擎停止时更改。**

### 3.1 节点的确认和登记

```python
call(LP, "POST", "/v1/launch/discover", {})     # 202。只读，只查 ConnectX-7 连接的对端
s, d = call(LP, "GET", "/v1/launch/discover")   # 反复读取，直到 state 变为 done
# d["kind"]: switch | ring | direct | none
```

`head` 是**启动器所在节点自身的 IP 地址**。使用名称会返回 400。

**交换机的情况：**

```python
head = d["fabric"]["head_addr"]                 # rank 0 在 fabric 一侧的地址
hosts = [h for h in d["hosts"] if h["access"] == "ok" and h["gb10"] and not h["other"]]
# hosts[] 中也包含交换机（gb10 为 null）和其他 lane 正在使用的节点（other 中有名称）。不包含 head
# in_lane 表示是否在该 lane 当前的 NODES 中
workers = [h["addr"] for h in hosts][:3]         # 4 台时为 3 个。选择哪些要向人确认
call(LP, "POST", "/v1/launch/check", {"head": head, "workers": workers, "deploy": False})
```

**环的情况：** `hosts` 为空。沿着线缆找到其余 3 台。省略 `workers` 时会检查设置文件中的 `NODES`，所以新的 lane 必须按这个顺序操作。

```python
call(LP, "POST", "/v1/launch/ring", {})          # 202
r = call(LP, "GET", "/v1/launch/ring")[1]        # 反复读取，直到 state 变为 done
# r["ring"]: closed（4 台连成的环）| open | none，r["nodes"]: 以 head 开头的顺序，r["problems"]: 无法到达的节点等
assert r["ring"] == "closed", r["problems"]
head = d["head"]["lan"]["addr"]                  # 管理局域网的地址
call(LP, "POST", "/v1/launch/check", {"head": head, "workers": r["nodes"][1:]})
```

**确认的结果**（`GET /v1/launch/check`。反复读取，直到 `state` 变为 `done`）：

- `nodes_ok`：能否登记。为 false 时不能登记。
- `workers`：已确认的 worker。登记时原样传入。
- `topology`：判定的连线。`mode` 是 `full`（交换机）或 `ring4`，`order` 是节点的顺序，`hca` 和 `ifname` 是将要登记的值，`why` 是原因。
- `nodes[].items[]`：每个项目的结果。`key`、`state`（ok, warn, fail）、`value`、`fix`（修复方法）。
- 省略 `deploy` 或设为 true 时，确认前会把发布包复制到各节点。`false` 只能确认已有发布包的节点。`glm53.conf`、`*.nodes.conf`、`logs` 不会复制。节点上已有发布包时用 `false` 即可。

```python
c = call(LP, "GET", "/v1/launch/check")[1]
call(LP, "POST", "/v1/launch/config", {"head": head, "workers": c["workers"]})
# TOPOLOGY、NODES、SOCKET_IFNAME、IB_HCA 写入 glm53.nodes.conf，此后以它为准
```

- 未经确认的组合返回 412，引擎运行时返回 409。
- 环上节点的顺序按 `topology.order` 登记。不要自行调整。
- 节点数为 4 或 8。两台和三台的配置正在测试中（`worlds` 中也列有 2）。环（`ring4`）只有 4 台。

### 3.2 权重的位置

```python
s, w = call(LP, "GET", "/v1/launch/weights")
# w["weights"], w["drafter"], w["drafter_kind"], w["banks"] = [{"id", "bits", "dir"}, ...]

body = {"weights": w["weights"], "drafter": w["drafter"],
        "banks": [{"id": b["id"], "dir": b["dir"]} for b in w["banks"]]}
call(LP, "POST", "/v1/launch/weights/check", body)   # 202
while (c := call(LP, "GET", "/v1/launch/weights")[1]["check"])["state"] != "done":
    time.sleep(2)
# c: state, ok, why, items[] (key, state, value, fix)

if c["ok"]:
    call(LP, "POST", "/v1/launch/weights", body)     # 以与确认相同的值保存。仅在停止时
# 回复的 copied 中是每个 worker 的 banks.json 复制结果
```

- `banks` 中要把 `banks.json` 的所有条目各传 1 次。
- 未经确认的值，以及在确认结束前保存，都返回 412。
- `dir` 为空的位宽不使用。
- 保存后 bank 的 `id` 可能变化。下次调用前重新读取。
- 复制失败时保存也返回 200。请检查 `copied[].ok`。
- 第一个节点的目录可以用 `GET /v1/launch/browse?path=/data/models` 列出。仅限 `~`、`/opt`、`/mnt`、`/media`、`/data`、`/srv` 之下，其他返回 400。
- 没有数据的位宽为 warn，不能在启动计划中选择。**只有 3 bit 为 fail。** 没有 3 bit bank，既不能保存位置，也不能启动。`GET /v1/launch/status` 的 `banks.required` 和 `GET /v1/launch/weights` 的 `banks[].required` 表示必需的位宽。
- **刚保存后 `banks.available` 为空。** 等到 `GET /v1/launch/status` 的 `banks.checking` 变为 false。

## 4. 引擎设置

### 4.1 读写

```python
s, v = call(LP, "GET", "/v1/launch/settings")
# v["conf"], v["state"], v["pending"], v["keys"] = [{key, value, default, set, from, when, rule}, ...]

s, r = call(LP, "POST", "/v1/launch/settings", {"values": {"THINKING_SHARE": "90"}})
# r["changed"]: 写入的设置项，r["kept"]: 保留下来的原文件副本，r.get("applied"): 也已应用到运行中引擎的设置项
# r.get("apply_error"): 引擎拒绝的原因（此时设置文件中也已写入）
```

- 400 时，`errors` 中有每个设置项的原因，不会写入任何内容。
- 在运行中更改 `when` 为 `stopped` 的设置项返回 409。
- `when` 表示更改何时生效。

| `when` | 含义 |
|---|---|
| `live` | 立即生效。也会送达运行中的引擎 |
| `start` | 从下次启动起。运行中更改时列入 `pending` |
| `stopped` | 只能在停止时更改（`PORT`） |
| `launcher` | 启动器重启后（`LAUNCHER_PORT`） |
| `readonly` | 这里不能更改。节点见 3.1，权重见 3.2 |

- `from` 是值的来源。`node_file` 的设置项随节点登记而变化。
- `glm53.conf` 在更改前复制到 `glm53.conf.bak-<时间>`。**时间精确到秒，同一秒内的更改会覆盖。** 如果要用它恢复，两次更改之间要间隔 1 秒以上。
- API 中没有的设置项（`RUNTIME`、`IMAGE`、`NCCL_LIB`、`NCCL_SHA256`、`GID_INDEX`、`NEED_GB`、`LOG_LISTEN` 等）直接编辑 `glm53.conf`。先向人确认再做。

### 4.2 主要设置项

所有设置项和规则由 `GET /v1/launch/settings` 返回。说明见 `CONFIG.zh-CN.md`。

下表的默认值是 `tf-glm53` 的默认值。`docker run ... IMAGE up` 根据 `glm53.t4.conf.example` 写入 `glm53.conf`，所以初始值为 `CONTEXT_LENGTH` 737280、`KV_SIZE` 837280、`KV` fp4、`ROWS` 16384、`LAUNCHER_PORT` 8010，启动时的 KV 大小为 `max`。

| 设置项 | 默认值 | when | 含义 |
|---|---|---|---|
| `CONTEXT_LENGTH` | 819200 | start | 1 次的最大长度。新值为 16,384 的倍数，不小于 65,536，不大于 1,048,576 |
| `KV_SIZE` | 1638400 | start | 整个 KV。不小于 `CONTEXT_LENGTH` + 512 |
| `KV` | fp8 | start | KV 的格式。fp4 或 fp8 |
| `DRAFTER_KIND` | mtp | start | mtp、dflash2、none |
| `MULTI` | 0 | start | 同时推理的数量。0 至 4 |
| `RESERVE_GB` | 4 | start | 留给操作系统的内存 |
| `MAX_TOKENS_SHARE` | 70 | start | 请求未指定时的 `max_tokens`。`CONTEXT_LENGTH` 的 %。1 至 100 |
| `MAX_TOKENS` | （无） | start | 以 token 数指定同一项。有值时优先。保存 `MAX_TOKENS_SHARE` 时清空 |
| `THINKING_SHARE` | 90 | live | 思考预算的比例（5.1）。0 表示没有预算 |
| `THINKING_BUDGET` | 0 | start | 以 token 数指定同一项。0 表示由 `THINKING_SHARE` 决定 |
| `THINKING_NOTICE` | 0 | start | 剩余量通知的间隔（思考的 token 数）。0 表示关闭 |
| `THINKING_FOLD_AT_BUDGET` | 1 | live | 是否在预算用完前整理。0 或 1 |
| `THINKING_FOLD_MARGIN` | 10 | live | 在预算用完前百分之多少开始整理。0 至 50 |
| `THINKING_FOLD_PROMPT` | org 文本 | live | 整理时向模型提出的文本 |
| `THINKING_FOLD_AT_CLOSE_SHARE` | 0 | live | 模型结束思考时进行整理所需的思考最小量（思考上限的 %）。0 表示关闭 |
| `THINKING_CLOSE_PROMPT` | org 文本 | live | 模型结束思考时的整理向模型提出的文本 |
| `THINKING_FOLD` | off | live | 中途整理。off、checkpoint、recent、branch |
| `THINKING_FOLD_SHARE` | 100 | live | 中途整理的间隔。回复上限的 %。100 表示到上限前不整理 |
| `THINKING_FOLD_MAX` | 0 | live | 1 次回复的中途整理次数上限。0 表示不限 |
| `THINKING_FOLD_CAP_SHARE` | 70 | start | 整理大小的上限。被整理思考的 %。10 至 100 |
| `TF_GLM53_BYTE_RESUME` | on | start | 跨过切分差异，从缓存继续（9.3）。on、log、off |
| `MULTI_DECODE_SHARE` | 30 | start | prefill 与 decode 的 GPU 分配初始值 |

旧的设置文件中可能留有不符合当前规则的 `CONTEXT_LENGTH`（例如 450560）。更改后无法改回原值。

### 4.3 更改整理的文本，恢复默认

```python
call(LP, "POST", "/v1/launch/settings", {"values": {"THINKING_FOLD_PROMPT": "整理思路。"}})  # 更改
call(LP, "POST", "/v1/launch/settings", {"values": {"THINKING_FOLD_PROMPT": ""}})                  # 恢复默认
```

- `value` 是当前生效的文本，`default` 是内置文本，`set` 表示是否已更改。
- 可以使用任何语言和换行。最多 4,096 字节（启动器按字符数接受，但运行中的更改由引擎按字节计数，超限时拒绝。此时返回 `apply_error`）。
- 在设置文件中以 % 编码保存。不要直接读写。

### 4.4 运行中的更改

| 调用 | 可更改的内容 |
|---|---|
| `POST /v1/thinking` | `share`、`fold`、`at_close_share`（0 至 100）、`fold_share`、`fold_max`、`at_budget`、`fold_margin`、`fold_prompt`、`close_prompt` |
| `POST /v1/schedule` | `decode_share`（0 至 100）、`short_prompt`、`short_preempt`。`MULTI` 为 0 的引擎返回 404 |

- 在启动器端口上调用时，`/v1/thinking` 的更改也会保留在设置文件中。
- `/v1/schedule` 的更改不会保留到下次启动。要保留，把 `MULTI_DECODE_SHARE`、`MULTI_SHORT_PROMPT`、`MULTI_SHORT_PREEMPT` 写入 `settings`。
- `GET /v1/thinking` 的 `start` 是启动时的值。
- `/v1/thinking` 的更改不会生成 `glm53.conf.bak-<时间>`。

## 5. 思考的预算和整理

### 5.1 机制

- **思考预算：** 除去 prompt 后剩余部分的 `THINKING_SHARE` %。`max_tokens` 较小时，取它的该百分比。`THINKING_SHARE` 为 0 时没有预算。有中途整理（`THINKING_FOLD`）时，预算是 `max_tokens` 减去回答部分的值，可能超过上下文的剩余量。**这是上下文中可放置的思考量，不是生成量的上限。**
- **整理的开始：** 预算的 `100 - THINKING_FOLD_MARGIN` %（默认 90%）。之后要等到行的分隔处，所以会稍晚一些。
- **整理：** 用向模型提出的文本让模型改写思考，并把上下文中的思考替换为结果。
- **整理之后：** 由模型决定继续思考还是结束思考。上下文中的思考达到预算时，引擎结束思考。回复或上下文的剩余量只够回答部分时，也会结束思考。
- **回答部分：** 预算的剩余部分（默认 10%）是思考用完预算时留给回答的最低量。不是上限。
- **模型结束思考时的整理：** 默认关闭。

**生成量会超过预算。** 即使预算为 2,200，加上整理和后续思考，思考也会达到 2,700 至 4,300 个 token，1 个请求需要 90 至 160 秒。

**请把 `max_tokens` 指定得足够大。** 过小时回答部分不够。`max_tokens` 为 20,000 时，回答在 2,000 个 token 处被截断。

### 5.2 每个请求的指定（`/v1/chat/completions`）

| 项目 | 含义 |
|---|---|
| `thinking_budget` | 以 token 数指定预算 |
| `thinking_fold_at_budget` | true / false。预算用完前的整理 |
| `thinking_fold_margin` | 0 至 50 |
| `thinking_fold_at_close` | token 数。0 表示关闭 |
| `thinking_fold` | off、checkpoint、recent、branch |
| `thinking_fold_window` | 以 token 数指定中途整理的间隔 |
| `thinking_fold_max` | 中途整理次数的上限 |
| `thinking_fold_cap` | 以 token 数指定整理大小的上限。128 至 16384 |
| `priority` | `"background"` 排到队列后面 |
| `chat_template_kwargs` | `{"enable_thinking": false}` 表示不思考 |
| `reasoning_effort` | none（不思考）、minimal 和 low（Low）、medium 和 high（High）、xhigh 和 max（Max）。省略时为 Max。**值变化时 prompt 也变化，缓存不会命中** |

`/v1/messages` 的形式不同（8.1）。

### 5.3 `GET /v1/thinking`

`share`、`budget`、`fold`、`at_close_share`、`fold_share`、`fold_max`、`at_budget`、`fold_margin`、`fold_prompt`、`fold_prompt_default`、`close_prompt`、`close_prompt_default`、`start`、`fold_window`、`fold_cap`、`fold_cap_share`、`notice`、`max_tokens`。

## 6. 启动

### 6.1 启动的选择

**分配（`alloc`）只保存 bit、bank 和使用记录。** 不保存下面的选择。省略时使用设置文件中的值，可能与上一次运行不同。

| 请求的项目 | 读取当前值的位置 | 含义 |
|---|---|---|
| `kv` | `status.kv.format` | fp4 或 fp8 |
| `context` | `status.kv.context` | 1 次的最大长度 |
| `store` | `status.kv.store` | KV 中保留对话的部分。一个数或 `"max"`（放得下多少算多少） |
| `reserve_gib` | `status.kv.reserve_gib` | 留给操作系统的内存。2 至 32 |
| `drafter` | `status.drafter.running` | mtp、dflash2、none |
| `multi` | `status.multi.running` | 同时推理的数量。0 至 4 |
| `multi_wait` | `status.multi.wait` | 秒 |

**要以相同配置恢复运行，请在停止前按此表从 `status` 读取。** `"max"` 由当时的空闲内存决定，不一定得到相同的值。需要相同的值时，传入数字。

### 6.2 步骤

```python
# 1. 是否已停止
s, st = call(LP, "GET", "/v1/launch/status"); assert st["state"] == "stopped"

# 2. 选择可用的分配
s, a = call(LP, "GET", "/v1/launch/allocs")
names = [x["name"] for x in a["allocs"] if x["valid"]]

# 3. 读取分配的 bit，确认能否放下（不启动）
s, al = call(LP, "GET", f"/v1/launch/alloc?name={names[0]}")
opts = {"kv": "fp4", "drafter": "mtp", "context": 131072, "store": "max", "reserve_gib": 12, "multi": 4}
s, plan = call(LP, "POST", "/v1/launch/plan", dict(opts, bits=al["bits"]))
# plan["verdict"]: fits | tight | no，plan["nodes"][].margin，plan["missing_bits"]，plan["multi"]["why"]
# 放不下的 context 不会被拒绝，而是被缩短。比较 plan["kv"]["context"] 和传入的值

# 4. 启动
s, r = call(LP, "POST", "/v1/launch/start", dict(opts, alloc=names[0]))   # {"state": "starting"}

# 5. 等待
while True:
    st = call(LP, "GET", "/v1/launch/status")[1]
    if st["state"] in ("running", "failed"): break
    print(st["phase"], st["layers_done"], st["eta_ms"], flush=True)
    time.sleep(10)
assert st["state"] == "running", st["error"]
```

- 用 `alloc` 启动时，继承该分配的 bit、bank 和使用记录。**恢复运行时使用这种方式。** `tf-glm53 start` 不继承使用记录。
- `bits` 和 `alloc` 都省略时，所有专家以 3 bit 启动。
- 要让所有专家以另一种相同位宽启动，bit 为 `base64.b64encode(bytes([W]) * (75 * 256)).decode()`。`W` 是 `banks.available` 中的位宽。
- 大小是 `GET /v1/launch/alloc` 的 `layers` 和 `experts`。目前是 75 和 256。

### 6.3 启动后的确认

```python
s, r = call(EP, "POST", "/v1/chat/completions", {"model": MODEL, "max_tokens": 64, "temperature": 0,
    "messages": [{"role": "user", "content": "What is 17 times 23? Answer with the number only."}],
    "chat_template_kwargs": {"enable_thinking": False}}, timeout=180)
# content 中应包含 391
```

`MODEL` 是 `GET /health` 的 `model`。**启动后的第 1 个请求 prefill 较慢。** 测速时丢弃它。

### 6.4 错误

| 代码 | 含义 | 处理 |
|---|---|---|
| 400 | 值不正确 | 读取 `error` |
| 404 `NoSuchAlloc` | 分配不存在 | `GET /v1/launch/allocs` |
| 409 | 正在运行，或 bank 不匹配（`BankMissing`、`BankDiffers`） | 停止，或读取 `allocs[].problem` |
| 412 | 计划放不下、没有数据的位宽、草稿模型不可用 | 查看 `plan` 的结果，减小 `context`、`store`、`multi` |

`state` 为 `failed` 时，读取 `error`，并查看节点的日志（1.5）。

## 7. 停止

### 7.1 先保存再停止（标准）

```python
NAME = "run-1"

# 1. 没有正在运行的推理
h = call(EP, "GET", "/health")[1]
assert not h["live"]["connections"]

# 2. 保存运行中引擎的 bit 和使用记录
s, r = call(LP, "POST", "/v1/launch/allocs/save",
            {"name": NAME, "engine": True, "with_usage": True, "source": "engine", "note": "before stop"})
assert s == 200, r

# 3. 确认已保存（确认后再停止）
a = call(LP, "GET", "/v1/launch/allocs")[1]
assert any(x["name"] == NAME and x["usage"] for x in a["allocs"])

# 4. 如果打算以相同配置恢复运行，记下 status 中的选择（6.1）

# 5. 停止。已保存，所以 save_usage 为 false
s, r = call(LP, "POST", "/v1/launch/stop", {"save_usage": False})
assert s == 200, r

# 6. 等待
while call(LP, "GET", "/v1/launch/status")[1]["state"] not in ("stopped", "failed"):
    time.sleep(3)
```

- **不要改变顺序。** `with_usage` 的保存只能在运行期间进行。停止后无法找回。
- 名称已存在时返回 409 `Exists`。请换一个名称，或加上 `"replace": true`。
- 已经停止时 `stop` 返回 409。
- 省略 `stop` 的 `save_usage` 时，记录进入待定状态（`GET /v1/launch/usage` 的 `pending`）。待定的记录只在启动器的内存中，停止启动器后就会丢失。

### 7.2 重启

```python
call(LP, "POST", "/v1/launch/allocs/save", {"name": NAME, "engine": True, "with_usage": True, "source": "engine"})
# 确认已保存（7.1 的第 3 步）
call(LP, "POST", "/v1/launch/restart", {"alloc": NAME})   # 202
```

- 继承运行中的启动选择（`kv`、`drafter`、`context`、`store`、`reserve_gib`、`multi`、`multi_wait`）。只有列在 `pending` 中的设置项使用设置文件的值。
- 省略 `alloc` 时，不保存使用记录就停止。

### 7.3 取消启动

启动过程中的 `POST /v1/launch/stop` 会取消启动。

## 8. 推理

### 8.1 API

| 路径 | 格式 | 使用场景 |
|---|---|---|
| `/v1/chat/completions` | OpenAI | 标准。pi、opencode |
| `/v1/messages` | Anthropic | Claude Code |
| `/v1/completions` | 原始 prompt | 用于直接发送特殊 token 的测试。思考预算和整理不起作用 |
| `/v1/responses` | OpenAI Responses | Responses API 的客户端 |
| `/v1/models` | | 模型名称列表 |
| `/v1/tokenize`、`/v1/detokenize` | | token 数和 id |
| `/v1/messages/count_tokens` | Anthropic | `/v1/messages` 的 token 数 |

**`/v1/messages` 的不同之处：**

- 思考用 `"thinking": {"type": "enabled", "budget_tokens": 2200}` 指定（也接受 `adaptive`）。`thinking_budget` 不起作用。**省略 `thinking` 时不思考。**
- `max_tokens` 必需。`budget_tokens` 不小于 `max_tokens` 时返回 400。
- 回复的 `content` 中有 `thinking` 块和 `text` 块。
- `usage.input_tokens` 不包含命中部分。`usage.cache_read_input_tokens` 是命中部分，`usage.output_tokens_details.thinking_tokens` 是思考部分。

### 8.2 回复（`/v1/chat/completions`）

```python
s, j = call(EP, "POST", "/v1/chat/completions",
            {"model": MODEL, "max_tokens": 276480, "temperature": 0, "messages": messages}, timeout=7200)
m = j["choices"][0]["message"]
m["content"]              # 回答
m["reasoning_content"]    # 思考。有整理时，整理前的内容、整理的结果和后续都包含在内
m.get("reasoning_details")  # 附在引擎改动过思考的回复上。是上下文中持有的思考
j["usage"]["prompt_tokens"]
j["usage"]["prompt_tokens_details"]["cached_tokens"]
j["usage"]["completion_tokens_details"]["reasoning_tokens"]
```

流式响应用 `stream()`（1.5）读取。思考和回答分别在 `delta.reasoning_content` 和 `delta.content` 中，`delta.reasoning_details` 只在结束前送达 1 次。

### 8.3 下一轮

把回复放进下一个请求的 `messages` 中返回。**`reasoning_content` 和 `reasoning_details` 按收到的原样返回。**

```python
a = {"role": "assistant", "content": m.get("content") or ""}
for k in ("reasoning_content", "reasoning_details"):
    if m.get(k): a[k] = m[k]
messages += [a, {"role": "user", "content": "下一个问题"}]
```

| 返回的内容 | 有整理的回复之后的下一轮 |
|---|---|
| 有 `reasoning_details` | 几乎全部命中。用上下文中的思考组装 |
| 只有 `reasoning_content` | 缓存从该回复思考的开头处起不命中。prompt 也会因全文而变长 |
| 不返回思考 | 缓存从该回复思考的开头处起不命中 |

### 8.4 Harness

**pi**（`~/.pi/agent/models.json`）：按收到的原样返回 `reasoning_details`，所以整理后缓存仍然保持。

```json
{"providers": {"expertatlas": {
  "name": "ExpertAtlas", "baseUrl": "http://HOST:PORT/v1", "api": "openai-completions", "apiKey": "none",
  "compat": {"supportsDeveloperRole": false, "maxTokensField": "max_tokens", "thinkingFormat": "openai"},
  "models": [{"id": "MODEL", "name": "MODEL", "reasoning": true, "input": ["text"],
              "contextWindow": 131072, "maxTokens": 64000,
              "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
              "samplingParams": {"thinking_budget": 52000}}]}}}
```

- `contextWindow` 与引擎的 `context_length` 保持一致。
- pi 可能根据 prompt 的长度减小 `maxTokens` 后发送。
- 每个请求的项目写在模型内的 `samplingParams` 中。
- pi 默认发送的 `reasoning_effort` 是 medium（日志中为 high）。

**opencode**（`opencode.json`，1.19.0 已确认）：**只返回 `reasoning_content`。** 有整理的回复之后的下一轮，缓存从该回复思考的开头处起不命中。

```json
{"$schema": "https://opencode.ai/config.json",
 "enabled_providers": ["expertatlas"],
 "provider": {"expertatlas": {"npm": "@ai-sdk/openai-compatible", "name": "ExpertAtlas",
   "options": {"baseURL": "http://HOST:PORT/v1", "apiKey": "none", "headerTimeout": 3600000, "chunkTimeout": 3600000},
   "models": {"MODEL": {"name": "MODEL", "tool_call": true, "reasoning": true,
                        "limit": {"context": 131072, "output": 64000},
                        "options": {"thinking_fold_at_budget": false}}}}}}
```

- 每个请求的项目写在 `models.MODEL.options` 中。上面的例子是关闭整理的指定（已在直接请求中确认：没有整理时，只返回 `reasoning_content` 也能命中。尚未在 opencode 中确认）。
- 生成对话标题的请求会另外运行。它带思考运行，可能需要约 50 秒。可以用 `small_model` 指定另一个模型。

**Claude Code：** 指定 `ANTHROPIC_BASE_URL=http://HOST:PORT`。发布包中的 `claude-glm53` 负责启动。

Harness 生成的代码在 sandbox 中运行。从 `--network host` 的 sandbox 也能访问启动器。

## 9. 检查缓存

### 9.1 确认方法

| 查看的内容 | 调用 | 含义 |
|---|---|---|
| 该轮命中的量 | 回复的 `usage.prompt_tokens_details.cached_tokens` | `prompt_tokens` 中取自缓存的数量 |
| 正在运行的请求 | `GET /health` 的 `progress` | `phase`、`prompt_tokens`、`cached_tokens`、`prompt_done` |
| 保留的对话 | 启动器的 `GET /v1/pool` | `convs[].positions`、`blocks`、`shared_blocks`、`idle_ms`、`active`。被逐出的数量见 `GET /health` 的 `pool.evicted_total` |
| 日志 | `timeout 10 ./tf-glm53 logs 0 \| grep` | `[tensorfold] done ... prompt=N cached=M` 和 `glm53: thinking:` 的行 |

`/v1/pool` 的 `owner` 和 `convs[].table` 是 base64。不要放进上下文。

### 9.2 实测

| 情况 | prompt | cached |
|---|---|---|
| 同一对话的下一轮 | 2,610 | 2,593 |
| 插入无关对话后的下一轮 | 2,713 | 2,696 |
| 整理后模型结束思考，按 8.3 返回 | 1,017 | 1,000 |
| 整理后引擎结束思考，按 8.3 返回 | 4,196 | 4,179 |
| 整理后只返回 `reasoning_content` | 3,252 | 256 |
| 整理后不返回思考 | 506 | 256 |

- 通常只有约 17 个 token 不命中。这是末尾模板的部分。
- 与其他对话共享的开头部分按 256 为单位向下取整后命中。
- **大量不命中时：** 依次怀疑 8.3 的返回方式、`reasoning_effort` 的差异、`/v1/pool` 中是否还有该对话。

### 9.3 日志行

| 行 | 含义 |
|---|---|
| `a summary beside the reply asked for at the budget's end at N think tokens` | 在预算用完前要求了整理 |
| `fold 1 (branch at the budget's end) at N think tokens: the context's reply A -> B tokens` | 上下文中的思考从 A 变为 B |
| `after the budget's summary the model goes on, until the thinking's memory of N tokens is full` | 模型继续思考 |
| `past the budget's summary the thinking's memory is full` | 思考填满预算，引擎结束思考 |
| `past the budget's summary only the answer's part is left` | 剩余量只够回答部分，引擎结束思考 |
| `fold 1 (branch at the close)` | 模型结束思考时的整理 |
| `the budget of N think tokens reached` | 未整理就达到预算并结束思考 |
| `ids remade (fold\|close): A -> B think ids, the first difference at N` | 把思考的 token 序列调整为从文本重新切分的结果 |
| `the think block closed after N reply tokens; the answer goes on from its remade ids` | 模型结束思考，重新切分后进入回答 |
| `pool: conversation N done: cancelled` | 同时推理（`MULTI`）时，因整理或重新切分而切换了区间。正常 |
| `ended reason=client-left` | 客户端断开。不会出现 `done` 行 |
| `pool:` 的 NEW、CONTINUE、SHARE | 新对话、同一对话的继续、与其他对话共享开头 |

要快速试验，在请求中把 `thinking_budget` 设为 2,000 左右。想触发整理时，调大 `thinking_fold_margin` 会更早开始。思考不足 1,024 个 token 时不整理。

### 切分差异

模型的切分方式与分词器不同时，下一轮的 prompt 以不同的 id 持有相同的字节，缓存就停在那里。默认的 `TF_GLM53_BYTE_RESUME=on` 把 prompt 与保存的对话按字节比较，一致之处使用它们的 id。`off` 时缓存停在那里。`log` 与 `off` 相同，并比较和记录。日志的 `byteresume:` 行显示 id 停止的位置和字节到达的位置。

## 10. 遇到问题时

| 症状 | 要确认的内容 |
|---|---|
| 无法连接 | `./tf-glm53 launcher status`。`HOST` 是否为 127.0.0.1 |
| 403 | 正从局域网调用引擎端口（1.1）。引擎的 `/v1/experts`、`/v1/schedule`、`/v1/thinking` 只响应同一节点的请求 |
| `/v1/schedule` 返回 404 | `MULTI` 为 0 的引擎（默认） |
| `/v1/thinking` 返回 503 | 引擎已停止。设置用 `settings` 读取 |
| 设置不生效 | `when` 和 `pending`，`from` 是否为 `node_file`，请求中的值是否优先 |
| 运行中更改后出现 `apply_error` | 引擎拒绝了。设置文件中已写入 |
| 回答被截断 | `max_tokens` 过小（5.1） |
| 恢复运行后配置不同 | 没有传入启动选择（6.1） |
| 刚保存后无法选择位宽 | 等到 `banks.checking` 变为 false（3.2） |
| 启动为 `tight` 或 `no` | `POST /v1/launch/plan` 的 `nodes[].margin`、`kv.formats`。减小 `context`、`store`、`multi` |
| `GET /v1/launch/status` 为 `failed` | `error`、日志（1.5） |
| 想停止却返回 409 | 已经停止，或正在停止或重启。正在启动时可以取消（7.3） |
| 日志命令不返回 | `tf-glm53 logs` 会持续跟随。加上 `timeout`（1.5） |
