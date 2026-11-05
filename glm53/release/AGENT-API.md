English | [日本語](AGENT-API.ja.md) | [简体中文](AGENT-API.zh-CN.md)

# Expert Atlas agent control API reference

- Readers: an AI agent that operates Expert Atlas without the dashboard, and the person behind it
- Covers: reading the state, the server settings, the engine settings, the start, inference, checking the cache, the stop
- Does not cover: expert optimisation (Optimise, Apply, `/v1/launch/resort`, `/v1/experts/swap`)

This page gathers what an agent needs to operate step by step. Every setting is in `CONFIG.md`.

## 0. From the image to a running engine

On the first node, as the user who will run it:

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host IMAGE up
```

It writes the release to `~/expertatlas`, starts the launcher on the host, and goes on through registering the nodes, checking the weights, the plan, the start and 1 inference request. The launcher and the engine keep running after this container ends. **The checkpoint and the 3-bit bank are always needed, and unless you name a width every expert starts at 3 bits.**

- A line that begins with `READY` means the engine has started. The lines after it give the addresses of the dashboard and the API.
- `STOP` gives the reason it stopped. `FIX` says how to fix it, and `THEN` gives the 1 line to run again. Fix it, then run the `THEN` line as it is. The launcher stays up, so the API on this page works too.
- Every start setting can be given as `-e NAME=VALUE`: `WEIGHTS`, `BANK_2` to `BANK_5`, `DRAFTER`, `BITS`, `ALLOC`, `NODES`, `PORT`, `LAUNCHER_PORT`, `EA_DIR`, `IMAGE` and the engine settings of section 4 (`TOPOLOGY`, `SOCKET_IFNAME` and `IB_HCA` cannot be given, since the node check decides them). A name it does not know ends in `STOP`. The list is in `QUICKSTART.md`.
- What the host needs: the NVIDIA driver 580 or later, Docker and the NVIDIA container toolkit, `python3`, addresses on the ConnectX-7 network, and ssh without a password between all the nodes and to itself, with their host keys known (`QUICKSTART.md`, section 1).
- The release's image holds NCCL with the AArch64 fence, for a switch and a ring (`RING.md`).
- If the engine already runs, it starts nothing, sends 1 inference request and prints `READY`.

## 1. First things to know

### 1.1 There are 2 ports

| Role | Settings key | Default | Use |
|---|---|---|---|
| Engine | `PORT` | 8000 | Inference goes here. It does not answer while stopped |
| Launcher | `LAUNCHER_PORT` | `PORT` + 10 | All control goes here. It answers while the engine is stopped too |

- Start the launcher once with `./tf-glm53 launcher start`. The API cannot start it. `./tf-glm53 launcher status` shows its state and URL.
- `/v1/thinking`, `/v1/schedule`, `/v1/pool` and `/v1/experts/*` answer 403 when called from the LAN on the engine's port. **Call them on the launcher's port.** The launcher relays them to the engine from 127.0.0.1.
- The launcher has no key. Use it only inside a closed network.

### 1.2 Always confirm before you act

1. **Is that port the cluster you mean?** Another directory may hold the same port. `conf_path` in `GET /v1/launch/config` is the settings file's absolute path (`conf` in `GET /v1/launch/settings` is the file's name only). On the node you can also confirm it like this:
   ```bash
   ss -ltnp | grep ":$PORT "          # read the pid
   readlink /proc/$PID/cwd             # that lane's directory
   ```
   Confirm the nodes with `nodes` and `names` in `GET /v1/launch/config`. Just after registering, `names` is empty; it comes back within seconds.
2. **Is any inference running?** Look at `live.connections` and `progress.phase` in `GET /health`.
3. **Did you save before stopping?** A stop without a save loses that run's usage record (section 7).

### 1.3 What you must not do

- Stopping or restarting a production engine nobody asked you to.
- Stopping or restarting the engine that runs you. While a tool runs, `live.connections` reads 0, so do not stop it even if it looks free.
- Putting `GET /v1/launch/status`, `GET /v1/launch/usage`, `GET /v1/launch/alloc` or `GET /v1/pool` into the context as they are. They hold tens to hundreds of thousands of base64 characters (measured for `alloc`: 435,780 characters). Take only the fields you need (1.5).

### 1.4 Asynchronous operations

For an operation that takes time, the POST answers at once (202 or 200) and the work goes on behind it. A dropped connection does not stop it. Read the state with GET.

| POST | GET to read | Done when | Measured |
|---|---|---|---|
| `/v1/launch/start` | `/v1/launch/status` | `state` is `running` or `failed` | 100 s (with the weights in memory). 10 minutes or more when they load |
| `/v1/launch/stop` | `/v1/launch/status` | `state` is `stopped` or `failed` | 6 s |
| `/v1/launch/restart` | `/v1/launch/status` | `restart` is false and `state` is `running`, or `restart` is false with an `error` | The same as a start |
| `/v1/launch/check` | `/v1/launch/check` | `state` is `done` | 2 to 4 s (`deploy: false`) |
| `/v1/launch/discover` | `/v1/launch/discover` | `state` is `done` | 1 to 3 s |
| `/v1/launch/weights/check` | `check` in `/v1/launch/weights` | `check.state` is `done` | Under 5 s |

Reading every 2 to 10 s is enough.

### 1.5 The tools

Every example uses these functions. `LP` is the launcher's port, `EP` the engine's.

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
    except urllib.error.URLError as e:      # a stopped engine, for example
        return 0, {"error": str(e.reason)}

def brief(status):
    """status without its base64 fields"""
    return {k: v for k, v in status.items() if k not in ("done", "experts_done", "bits", "banks", "weights")}

def stream(port, body, host="127.0.0.1", timeout=7200):
    """Reads the SSE. (thinking, answer, usage, reasoning_details, finish_reason)"""
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

A node's log is `./tf-glm53 logs 0`, but **it follows and does not end.** Always read it as `timeout 10 ./tf-glm53 logs 0 | grep ...`, in the lane's directory.

## 2. Read the state

| What you want to know | Call | Fields |
|---|---|---|
| Whether it runs, how far the start is | `GET /v1/launch/status` | `state` (stopped, starting, running, stopping, failed), `phase`, `layers_done`, `eta_ms`, `error`, `alloc` |
| The current start's choices | `GET /v1/launch/status` | `kv.format`, `kv.context`, `kv.store`, `kv.reserve_gib`, `drafter.running`, `multi.running` |
| Which nodes | `GET /v1/launch/config` | `nodes`, `names`, `world`, `topology`, `registered` |
| Each node's memory and temperature | `GET /v1/launch/nodes` | `nodes[].mem_available`, `engine`, `other`, `gpu_temp_c` |
| The widths and drafters you can use | `GET /v1/launch/status` | `banks.available`, `banks.checking`, `drafter.available`, `drafter.why` |
| The engine's state | `GET /health` (the engine's port) | `model`, `context_length`, `live`, `progress`, `thinking` |
| The thinking's settings | `GET /v1/thinking` | 5.3 |
| The saved allocations | `GET /v1/launch/allocs` | `allocs[].name`, `valid`, `problem`, `usage` |

- **The values in use and the settings file's values are different things.** `GET /v1/launch/settings` gives the settings file's values. When a start is given choices, the values may differ even with `pending` empty. For the values in use, read `kv.*` in `status` and `context_length` in `/health`.
- After a stop, `kv`, `drafter.running` and `multi.running` in `status` keep the last values.
- While the engine is stopped, `/v1/thinking` and `/v1/schedule` answer 503 through the launcher. Read the settings with `settings`.

## 3. Server settings (nodes and weights)

The nodes' registration and the weights' locations. **Both can change only while the engine is stopped.**

### 3.1 Check and register the nodes

```python
call(LP, "POST", "/v1/launch/discover", {})     # 202. Reads only, and only past the ConnectX-7 ports
s, d = call(LP, "GET", "/v1/launch/discover")   # read until state is done
# d["kind"]: switch | ring | direct | none
```

`head` is **the launcher's node's own IP address**. A name gets 400.

**On a switch:**

```python
head = d["fabric"]["head_addr"]                 # rank 0's address on the fabric
hosts = [h for h in d["hosts"] if h["access"] == "ok" and h["gb10"] and not h["other"]]
# hosts[] also holds the switch (gb10 is null) and nodes another lane uses (its name in other). head is not in it
# in_lane says whether it is in that lane's current NODES
workers = [h["addr"] for h in hosts][:3]         # 3 for 4 nodes. Ask the person which to pick
call(LP, "POST", "/v1/launch/check", {"head": head, "workers": workers, "deploy": False})
```

**On a ring:** `hosts` is empty. Follow the cables to find the other 3 nodes. Leaving out `workers` checks `NODES` in the settings file, so on a new lane always use this order.

```python
call(LP, "POST", "/v1/launch/ring", {})          # 202
r = call(LP, "GET", "/v1/launch/ring")[1]        # read until state is done
# r["ring"]: closed (a ring of 4 nodes) | open | none, r["nodes"]: in order, head first, r["problems"]: nodes it cannot reach and so on
assert r["ring"] == "closed", r["problems"]
head = d["head"]["lan"]["addr"]                  # the address on the management LAN
call(LP, "POST", "/v1/launch/check", {"head": head, "workers": r["nodes"][1:]})
```

**The check's result** (`GET /v1/launch/check`. Read until `state` is `done`):

- `nodes_ok`: whether they can be registered. If false, they cannot.
- `workers`: the workers checked. Pass these as they are to register.
- `topology`: the cabling it judged. `mode` is `full` (a switch) or `ring4`, `order` the nodes' order, `hca` and `ifname` the values to be registered, `why` the reason.
- `nodes[].items[]`: each item's result. `key`, `state` (ok, warn, fail), `value`, `fix` (how to fix it).
- Leaving out `deploy` or setting it true copies the package to each node before the check. `false` checks only nodes that already have the package. `glm53.conf`, `*.nodes.conf` and `logs` are not copied. If the nodes already have the package, `false` is enough.

```python
c = call(LP, "GET", "/v1/launch/check")[1]
call(LP, "POST", "/v1/launch/config", {"head": head, "workers": c["workers"]})
# glm53.nodes.conf gets TOPOLOGY, NODES, SOCKET_IFNAME and IB_HCA, and from then on it comes first
```

- A combination not checked gets 412; a running engine gets 409.
- A ring's node order is registered as `topology.order` has it. Do not reorder it yourself.
- The node count is 4 or 8. Two- and three-node forms are being tested (`worlds` also lists 2). A ring (`ring4`) is 4 nodes only.

### 3.2 Where the weights are

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
    call(LP, "POST", "/v1/launch/weights", body)     # saves the values just checked. Only while stopped
# copied in the answer: how the copy of banks.json went on each worker
```

- Pass each entry of `banks.json` in `banks`, 1 time each.
- Values not checked, and a save before the check has ended, get 412.
- A width whose `dir` is empty is not used.
- A save may change a bank's `id`. Read it again before the next call.
- A failed copy still gets 200 for the save. Check `copied[].ok`.
- `GET /v1/launch/browse?path=/data/models` lists the first node's directories. Only under `~`, `/opt`, `/mnt`, `/media`, `/data` and `/srv`; anything else gets 400.
- A width without data shows warn and cannot be picked in a start's plan. **Only 3 bits shows fail.** Without a 3-bit bank, neither saving the locations nor a start is possible. `banks.required` in `GET /v1/launch/status` and `banks[].required` in `GET /v1/launch/weights` give the required widths.
- **Just after a save, `banks.available` is empty.** Wait until `banks.checking` in `GET /v1/launch/status` is false.

## 4. Engine settings

### 4.1 Read and write

```python
s, v = call(LP, "GET", "/v1/launch/settings")
# v["conf"], v["state"], v["pending"], v["keys"] = [{key, value, default, set, from, when, rule}, ...]

s, r = call(LP, "POST", "/v1/launch/settings", {"values": {"THINKING_SHARE": "90"}})
# r["changed"]: the keys written, r["kept"]: the copy of the old file, r.get("applied"): the keys also applied to the running engine
# r.get("apply_error"): why the engine refused (the settings file is written all the same)
```

- On 400, `errors` holds each key's reason and nothing is written.
- Changing a key whose `when` is `stopped` while running gets 409.
- `when` says when a change takes effect.

| `when` | Meaning |
|---|---|
| `live` | At once. It reaches the running engine too |
| `start` | From the next start. A change while running shows in `pending` |
| `stopped` | Can change only while stopped (`PORT`) |
| `launcher` | After the launcher restarts (`LAUNCHER_PORT`) |
| `readonly` | Cannot change here. The nodes: 3.1. The weights: 3.2 |

- `from` says where the value comes from. A `node_file` key changes by registering the nodes.
- Before a change, `glm53.conf` is copied to `glm53.conf.bak-<time>`. **The time is in seconds, and changes in the same second overwrite it.** To use it for going back, leave at least 1 s between changes.
- Keys not in the API (`RUNTIME`, `IMAGE`, `NCCL_LIB`, `NCCL_SHA256`, `GID_INDEX`, `NEED_GB`, `LOG_LISTEN` and others): edit `glm53.conf` directly. Ask the person first.

### 4.2 Main keys

`GET /v1/launch/settings` returns every key and its rule. `CONFIG.md` explains them.

The defaults below are `tf-glm53`'s. `docker run ... IMAGE up` writes `glm53.conf` from `glm53.t4.conf.example`, so it begins with `CONTEXT_LENGTH` 737280, `KV_SIZE` 837280, `KV` fp4 and `LAUNCHER_PORT` 8010, and the start's KV size is `max`.

| Key | Default | when | Meaning |
|---|---|---|---|
| `CONTEXT_LENGTH` | 819200 | start | 1 request's longest. A new value is a multiple of 16,384, at least 65,536, at most 1,048,576 |
| `KV_SIZE` | 1638400 | start | The whole KV. At least `CONTEXT_LENGTH` + 512 |
| `KV` | fp8 | start | The KV's format. fp4 or fp8 |
| `DRAFTER_KIND` | mtp | start | mtp, dflash2, none |
| `MULTI` | 0 | start | Conversations decoded together. 0 to 4 |
| `RESERVE_GB` | 4 | start | The memory kept for the OS |
| `MAX_TOKENS_SHARE` | 70 | start | `max_tokens` when a request names none. A percent of `CONTEXT_LENGTH`. 1 to 100 |
| `MAX_TOKENS` | (none) | start | The same as a token count. Comes first when set. A save of `MAX_TOKENS_SHARE` empties it |
| `THINKING_SHARE` | 90 | live | The thinking's budget as a share (5.1). 0 is no budget |
| `THINKING_BUDGET` | 0 | start | The same as a token count. 0 leaves it to `THINKING_SHARE` |
| `THINKING_NOTICE` | 0 | start | How often the notice of what is left comes (in thinking tokens). 0 is off |
| `THINKING_FOLD_AT_BUDGET` | 1 | live | Whether to organize near the end of the budget. 0 or 1 |
| `THINKING_FOLD_MARGIN` | 10 | live | How many percent of the budget before its end organizing starts. 0 to 50 |
| `THINKING_FOLD_PROMPT` | the org text | live | The text that asks for organizing |
| `THINKING_FOLD_AT_CLOSE_SHARE` | 0 | live | The least thinking (a percent of the thinking's limit) to organize when the model closes it. 0 is off |
| `THINKING_CLOSE_PROMPT` | the org text | live | The text that asks for organizing at the close |
| `THINKING_FOLD` | off | live | Organizing midway. off, checkpoint, recent, branch |
| `THINKING_FOLD_SHARE` | 100 | live | How often to organize midway. A percent of the reply's limit. 100 is no organizing until the limit |
| `THINKING_FOLD_MAX` | 0 | live | The most organizing midway in 1 reply. 0 is no limit |
| `THINKING_FOLD_CAP_SHARE` | 70 | start | The most an organized text may be. A percent of the thinking it organizes. 10 to 100 |
| `TF_GLM53_BYTE_RESUME` | on | start | Resumes from the cache past a difference in how a text is cut (9.3). on, log, off |
| `MULTI_DECODE_SHARE` | 30 | start | The initial split of the GPU between prefill and decode |

An older settings file may still hold a `CONTEXT_LENGTH` against today's rule (for example 450560). Once changed, the old value cannot be set again.

### 4.3 Change the organizing text, go back to the default

```python
call(LP, "POST", "/v1/launch/settings", {"values": {"THINKING_FOLD_PROMPT": "Organize the thinking."}})  # change
call(LP, "POST", "/v1/launch/settings", {"values": {"THINKING_FOLD_PROMPT": ""}})                        # back to the default
```

- `value` is the text in effect, `default` the built-in text, `set` whether it was changed.
- Any language and line breaks work. Up to 4,096 bytes (the launcher accepts it by character count, but for a change while running the engine counts bytes and refuses; `apply_error` then comes back).
- The settings file keeps it percent-encoded. Do not read or write it directly.

### 4.4 Changes while running

| Call | What it can change |
|---|---|
| `POST /v1/thinking` | `share`, `fold`, `at_close_share` (0 to 100), `fold_share`, `fold_max`, `at_budget`, `fold_margin`, `fold_prompt`, `close_prompt` |
| `POST /v1/schedule` | `decode_share` (0 to 100), `short_prompt`, `short_preempt`. An engine whose `MULTI` is 0 answers 404 |

- Called on the launcher's port, a `/v1/thinking` change also stays in the settings file.
- A `/v1/schedule` change does not last to the next start. To keep it, write `MULTI_DECODE_SHARE`, `MULTI_SHORT_PROMPT` and `MULTI_SHORT_PREEMPT` with `settings`.
- `start` in `GET /v1/thinking` holds the values at start.
- A `/v1/thinking` change makes no `glm53.conf.bak-<time>`.

## 5. The thinking's budget and organizing

### 5.1 How it works

- **The thinking's budget:** `THINKING_SHARE` % of what the prompt leaves. If `max_tokens` is smaller, that percent of it. With `THINKING_SHARE` at 0 there is no budget. With organizing midway (`THINKING_FOLD`) it is `max_tokens` less the answer's part, and it may pass what the context has left. **It is how much thinking the context can hold, not a limit on what is generated.**
- **The start of organizing:** `100 - THINKING_FOLD_MARGIN` % of the budget (90% by default). It then waits for the next line break, so it comes a little later.
- **Organizing:** the text asks the model to rewrite the thinking, and the rewrite replaces the thinking in the context.
- **After organizing:** the model decides whether to think on or close. When the context's thinking reaches the budget, the engine closes it. It also closes when what the reply or the context has left is only the answer's part.
- **The answer's part:** what the budget leaves (10% by default) is the least the answer has when the thinking uses up the budget. It is not a limit.
- **Organizing at the close, when the model closes:** off by default.

**What is generated goes past the budget.** Even with a budget of 2,200, organizing and what follows take the thinking to 2,700 to 4,300 tokens, and 1 reply takes 90 to 160 s.

**Give `max_tokens` a large enough value.** Too small leaves too little for the answer. With `max_tokens` 20,000 an answer was cut at 2,000 tokens.

### 5.2 Per-request fields (`/v1/chat/completions`)

| Field | Meaning |
|---|---|
| `thinking_budget` | The budget in tokens |
| `thinking_fold_at_budget` | true / false. Organizing near the end of the budget |
| `thinking_fold_margin` | 0 to 50 |
| `thinking_fold_at_close` | In tokens. 0 is off |
| `thinking_fold` | off, checkpoint, recent, branch |
| `thinking_fold_window` | How often to organize midway, in tokens |
| `thinking_fold_max` | The most organizing midway |
| `thinking_fold_cap` | The most an organized text may be, in tokens. 128 to 16384 |
| `priority` | `"background"` goes to the back of the queue |
| `chat_template_kwargs` | `{"enable_thinking": false}` means no thinking |
| `reasoning_effort` | none (no thinking), minimal and low (Low), medium and high (High), xhigh and max (Max). Left out: Max. **A change of value changes the prompt, and the cache does not hit** |

`/v1/messages` has another form (8.1).

### 5.3 `GET /v1/thinking`

`share`, `budget`, `fold`, `at_close_share`, `fold_share`, `fold_max`, `at_budget`, `fold_margin`, `fold_prompt`, `fold_prompt_default`, `close_prompt`, `close_prompt_default`, `start`, `fold_window`, `fold_cap`, `fold_cap_share`, `notice`, `max_tokens`.

## 6. Start

### 6.1 The start's choices

**An allocation (`alloc`) holds only the bits, the banks and the usage record.** It does not hold the choices below. Left out, they take the settings file's values, which may differ from the last run.

| Request field | Where to read the current value | Meaning |
|---|---|---|
| `kv` | `status.kv.format` | fp4 or fp8 |
| `context` | `status.kv.context` | 1 request's longest |
| `store` | `status.kv.store` | The part of the KV that holds conversations. A number, or `"max"` (all that fits) |
| `reserve_gib` | `status.kv.reserve_gib` | The memory kept for the OS. 2 to 32 |
| `drafter` | `status.drafter.running` | mtp, dflash2, none |
| `multi` | `status.multi.running` | Conversations decoded together. 0 to 4 |
| `multi_wait` | `status.multi.wait` | Seconds |

**To resume with the same setup, read these from `status` before the stop, as this table says.** `"max"` depends on the free memory at that time, so it may not give the same value. For the same value, pass a number.

### 6.2 Steps

```python
# 1. Is it stopped?
s, st = call(LP, "GET", "/v1/launch/status"); assert st["state"] == "stopped"

# 2. Pick a usable allocation
s, a = call(LP, "GET", "/v1/launch/allocs")
names = [x["name"] for x in a["allocs"] if x["valid"]]

# 3. Read the allocation's bits and check that it fits (this starts nothing)
s, al = call(LP, "GET", f"/v1/launch/alloc?name={names[0]}")
opts = {"kv": "fp4", "drafter": "mtp", "context": 131072, "store": "max", "reserve_gib": 12, "multi": 4}
s, plan = call(LP, "POST", "/v1/launch/plan", dict(opts, bits=al["bits"]))
# plan["verdict"]: fits | tight | no, plan["nodes"][].margin, plan["missing_bits"], plan["multi"]["why"]
# A context that does not fit is not refused but cut. Compare plan["kv"]["context"] with the value you passed

# 4. Start
s, r = call(LP, "POST", "/v1/launch/start", dict(opts, alloc=names[0]))   # {"state": "starting"}

# 5. Wait
while True:
    st = call(LP, "GET", "/v1/launch/status")[1]
    if st["state"] in ("running", "failed"): break
    print(st["phase"], st["layers_done"], st["eta_ms"], flush=True)
    time.sleep(10)
assert st["state"] == "running", st["error"]
```

- A start with `alloc` takes over that allocation's bits, banks and usage record. **Use this to resume.** `tf-glm53 start` does not take over the usage record.
- With neither `bits` nor `alloc`, every expert starts at 3 bits.
- To start every expert at another single width, the bits are `base64.b64encode(bytes([W]) * (75 * 256)).decode()`. `W` is a width in `banks.available`.
- The sizes are `layers` and `experts` in `GET /v1/launch/alloc`. Today they are 75 and 256.

### 6.3 Check after the start

```python
s, r = call(EP, "POST", "/v1/chat/completions", {"model": MODEL, "max_tokens": 64, "temperature": 0,
    "messages": [{"role": "user", "content": "What is 17 times 23? Answer with the number only."}],
    "chat_template_kwargs": {"enable_thinking": False}}, timeout=180)
# content holds 391
```

`MODEL` is `model` in `GET /health`. **The first request after a start has a slow prefill.** Drop it when you measure speed.

### 6.4 Errors

| Code | Meaning | What to do |
|---|---|---|
| 400 | A value is not valid | Read `error` |
| 404 `NoSuchAlloc` | No such allocation | `GET /v1/launch/allocs` |
| 409 | Running, or the banks do not match (`BankMissing`, `BankDiffers`) | Stop it, or read `allocs[].problem` |
| 412 | The plan does not fit, a width without data, or the drafter cannot be used | Read the `plan` result and reduce `context`, `store` or `multi` |

When `state` is `failed`, read `error` and then the node's log (1.5).

## 7. Stop

### 7.1 Save, then stop (the usual way)

```python
NAME = "run-1"

# 1. No inference is running
h = call(EP, "GET", "/health")[1]
assert not h["live"]["connections"]

# 2. Save the running engine's bits and usage record
s, r = call(LP, "POST", "/v1/launch/allocs/save",
            {"name": NAME, "engine": True, "with_usage": True, "source": "engine", "note": "before stop"})
assert s == 200, r

# 3. Confirm the save (confirm it before stopping)
a = call(LP, "GET", "/v1/launch/allocs")[1]
assert any(x["name"] == NAME and x["usage"] for x in a["allocs"])

# 4. If you will resume with the same setup, note the choices in status (6.1)

# 5. Stop. It is saved, so save_usage is false
s, r = call(LP, "POST", "/v1/launch/stop", {"save_usage": False})
assert s == 200, r

# 6. Wait
while call(LP, "GET", "/v1/launch/status")[1]["state"] not in ("stopped", "failed"):
    time.sleep(3)
```

- **Do not change the order.** A save with `with_usage` works only while running. It cannot be recovered after the stop.
- If the name exists it gets 409 `Exists`. Use another name, or add `"replace": true`.
- `stop` when already stopped gets 409.
- Leaving out `save_usage` in `stop` puts the record on hold (`pending` in `GET /v1/launch/usage`). What is on hold is only in the launcher's memory and goes when the launcher stops.

### 7.2 Restart

```python
call(LP, "POST", "/v1/launch/allocs/save", {"name": NAME, "engine": True, "with_usage": True, "source": "engine"})
# Confirm the save (step 3 of 7.1)
call(LP, "POST", "/v1/launch/restart", {"alloc": NAME})   # 202
```

- It takes over the running start's choices (`kv`, `drafter`, `context`, `store`, `reserve_gib`, `multi`, `multi_wait`). Only the keys in `pending` take the settings file's value.
- Leaving out `alloc` stops without saving the usage record.

### 7.3 Cancel a start

`POST /v1/launch/stop` during a start cancels it.

## 8. Inference

### 8.1 The API

| Route | Format | When to use it |
|---|---|---|
| `/v1/chat/completions` | OpenAI | The standard. pi, opencode |
| `/v1/messages` | Anthropic | Claude Code |
| `/v1/completions` | Raw prompt | For tests that send special tokens as they are. The thinking's budget and organizing do not work |
| `/v1/responses` | OpenAI Responses | Clients of the Responses API |
| `/v1/models` | | The list of model names |
| `/v1/tokenize`, `/v1/detokenize` | | Token counts and ids |
| `/v1/messages/count_tokens` | Anthropic | The token count of `/v1/messages` |

**How `/v1/messages` differs:**

- Name the thinking with `"thinking": {"type": "enabled", "budget_tokens": 2200}` (`adaptive` is accepted too). `thinking_budget` has no effect. **Leaving out `thinking` means no thinking.**
- `max_tokens` is required. A `budget_tokens` at or above `max_tokens` gets 400.
- The reply's `content` holds `thinking` blocks and `text` blocks.
- `usage.input_tokens` leaves out what hit the cache. `usage.cache_read_input_tokens` is what hit, `usage.output_tokens_details.thinking_tokens` the thinking.

### 8.2 The reply (`/v1/chat/completions`)

```python
s, j = call(EP, "POST", "/v1/chat/completions",
            {"model": MODEL, "max_tokens": 276480, "temperature": 0, "messages": messages}, timeout=7200)
m = j["choices"][0]["message"]
m["content"]              # the answer
m["reasoning_content"]    # the thinking. With organizing, it holds the text before organizing, the organized text and what follows
m.get("reasoning_details")  # on a reply whose thinking the engine changed. The thinking the context holds
j["usage"]["prompt_tokens"]
j["usage"]["prompt_tokens_details"]["cached_tokens"]
j["usage"]["completion_tokens_details"]["reasoning_tokens"]
```

Read a stream with `stream()` (1.5). The thinking and the answer come apart in `delta.reasoning_content` and `delta.content`, and `delta.reasoning_details` comes only 1 time, just before the end.

### 8.3 The next turn

Put the reply into the next request's `messages` and send it back. **Send back `reasoning_content` and `reasoning_details` just as you got them.**

```python
a = {"role": "assistant", "content": m.get("content") or ""}
for k in ("reasoning_content", "reasoning_details"):
    if m.get(k): a[k] = m[k]
messages += [a, {"role": "user", "content": "The next question"}]
```

| What you send back | The next turn after a reply with organizing |
|---|---|
| `reasoning_details` present | Nearly all of it hits. It is built from the context's thinking |
| `reasoning_content` only | It misses from the start of that reply's thinking. The prompt also grows by the full text |
| No thinking | It misses from the start of that reply's thinking |

### 8.4 Harnesses

**pi** (`~/.pi/agent/models.json`): it returns `reasoning_details` just as it got them, so the cache holds after organizing too.

```json
{"providers": {"expertatlas": {
  "name": "ExpertAtlas", "baseUrl": "http://HOST:PORT/v1", "api": "openai-completions", "apiKey": "none",
  "compat": {"supportsDeveloperRole": false, "maxTokensField": "max_tokens", "thinkingFormat": "openai"},
  "models": [{"id": "MODEL", "name": "MODEL", "reasoning": true, "input": ["text"],
              "contextWindow": 131072, "maxTokens": 64000,
              "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
              "samplingParams": {"thinking_budget": 52000}}]}}}
```

- Match `contextWindow` to the engine's `context_length`.
- pi may send a smaller `maxTokens`, by the prompt's length.
- Write per-request fields in the model's `samplingParams`.
- By pi's default, `reasoning_effort` is sent as medium (high in the log).

**opencode** (`opencode.json`, confirmed with 1.19.0): **it returns `reasoning_content` only.** The next turn after a reply with organizing misses from the start of that reply's thinking.

```json
{"$schema": "https://opencode.ai/config.json",
 "enabled_providers": ["expertatlas"],
 "provider": {"expertatlas": {"npm": "@ai-sdk/openai-compatible", "name": "ExpertAtlas",
   "options": {"baseURL": "http://HOST:PORT/v1", "apiKey": "none", "headerTimeout": 3600000, "chunkTimeout": 3600000},
   "models": {"MODEL": {"name": "MODEL", "tool_call": true, "reasoning": true,
                        "limit": {"context": 131072, "output": 64000},
                        "options": {"thinking_fold_at_budget": false}}}}}}
```

- Per-request fields go in `models.MODEL.options`. The example above turns organizing off (in direct requests, `reasoning_content` alone was confirmed to hit without organizing; with opencode it is not confirmed yet).
- A separate request runs to make the conversation's title. It runs with thinking and can take about 50 s. `small_model` can name another model for it.

**Claude Code:** set `ANTHROPIC_BASE_URL=http://HOST:PORT`. The package's `claude-glm53` does the launch.

Run the code a harness generates inside a sandbox. A `--network host` sandbox can reach the launcher too.

## 9. Check the cache

### 9.1 How to check

| What to look at | Call | Meaning |
|---|---|---|
| How much hit in that turn | `usage.prompt_tokens_details.cached_tokens` in the reply | How many of `prompt_tokens` came from the cache |
| The running request | `progress` in `GET /health` | `phase`, `prompt_tokens`, `cached_tokens`, `prompt_done` |
| The conversations held | The launcher's `GET /v1/pool` | `convs[].positions`, `blocks`, `shared_blocks`, `idle_ms`, `active`. The number evicted is `pool.evicted_total` in `GET /health` |
| The log | `timeout 10 ./tf-glm53 logs 0 \| grep` | The `[tensorfold] done ... prompt=N cached=M` and `glm53: thinking:` lines |

`owner` and `convs[].table` in `/v1/pool` are base64. Do not put them in the context.

### 9.2 Measured

| Case | prompt | cached |
|---|---|---|
| The next turn of the same conversation | 2,610 | 2,593 |
| The next turn with an unrelated conversation between | 2,713 | 2,696 |
| After organizing, the model closed, sent back as in 8.3 | 1,017 | 1,000 |
| After organizing, the engine closed, sent back as in 8.3 | 4,196 | 4,179 |
| After organizing, only `reasoning_content` sent back | 3,252 | 256 |
| After organizing, no thinking sent back | 506 | 256 |

- Usually about 17 tokens miss. That is the template at the end.
- A head shared with another conversation hits rounded down to units of 256.
- **When much misses:** suspect, in this order, how you sent it back (8.3), a difference in `reasoning_effort`, and whether the conversation is still in `/v1/pool`.

### 9.3 Log lines

| Line | Meaning |
|---|---|
| `a summary beside the reply asked for at the budget's end at N think tokens` | Asked for organizing near the end of the budget |
| `fold 1 (branch at the budget's end) at N think tokens: the context's reply A -> B tokens` | The context's thinking went from A to B |
| `after the budget's summary the model goes on, until the thinking's memory of N tokens is full` | The model goes on |
| `past the budget's summary the thinking's memory is full` | The thinking filled the budget and the engine closed it |
| `past the budget's summary only the answer's part is left` | Only the answer's part was left and the engine closed it |
| `fold 1 (branch at the close)` | Organizing when the model closed |
| `the budget of N think tokens reached` | It reached the budget without organizing and closed |
| `ids remade (fold\|close): A -> B think ids, the first difference at N` | The thinking's token ids were set to what cutting its text again gave |
| `the think block closed after N reply tokens; the answer goes on from its remade ids` | The model closed, and the text was cut again and went on to the answer |
| `pool: conversation N done: cancelled` | With concurrent inference (`MULTI`), organizing or cutting again switched the segment. Normal |
| `ended reason=client-left` | The client disconnected. No `done` line comes |
| NEW, CONTINUE, SHARE in `pool:` | A new conversation, the same conversation going on, a head shared with another conversation |

For a short try, set `thinking_budget` to about 2,000 in the request. To bring organizing on, a larger `thinking_fold_margin` starts it earlier. With under 1,024 thinking tokens it does not organize.

### When the cut differs

When a model cuts a text differently from the tokenizer, the next turn's prompt holds the same bytes under other ids, and the cache stops there. With `TF_GLM53_BYTE_RESUME=on`, the default, the prompt is compared with the saved conversations in bytes, and where they agree it takes their ids. With `off` the cache stops there. `log` is as `off`, and compares and logs. The `byteresume:` lines in the log show where the ids stopped and where the bytes reached.

## 10. When something is wrong

| Symptom | What to check |
|---|---|
| Cannot connect | `./tf-glm53 launcher status`. Whether `HOST` is 127.0.0.1 |
| 403 | You are calling the engine's port from the LAN (1.1). The engine's `/v1/experts`, `/v1/schedule` and `/v1/thinking` answer only from the same node |
| `/v1/schedule` gets 404 | An engine whose `MULTI` is 0 (the default) |
| `/v1/thinking` gets 503 | The engine is stopped. Read the settings with `settings` |
| A setting has no effect | `when` and `pending`, whether `from` is `node_file`, whether the request's value comes first |
| `apply_error` after a change while running | The engine refused. The settings file has it |
| The answer is cut | `max_tokens` is small (5.1) |
| The setup differs after a resume | The start's choices were not passed (6.1) |
| No width can be picked just after a save | Wait until `banks.checking` is false (3.2) |
| The start says `tight` or `no` | `nodes[].margin` and `kv.formats` of `POST /v1/launch/plan`. Reduce `context`, `store` or `multi` |
| `GET /v1/launch/status` says `failed` | `error`, the log (1.5) |
| 409 when you want to stop | Already stopped, or in the middle of a stop or restart. In the middle of a start, it can be cancelled (7.3) |
| The log command does not return | `tf-glm53 logs` follows. Add `timeout` (1.5) |
