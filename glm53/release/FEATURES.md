English | [日本語](FEATURES.ja.md) | [简体中文](FEATURES.zh-CN.md)

# Expert Atlas: the features

This page explains each feature of Expert Atlas: what it does, why it is there, and how to use it. `SCREENS.md` shows
the screens, `PANEL-GUIDE.md` every control, and `CONFIG.md` every setting.

Measured numbers below come from the release documents named beside them, with their conditions. Speeds and memory
differ with the nodes, the prompt and the settings: take them as a guide.

## Contents

1. [Ring or switch, and the node check](#1-ring-or-switch-and-the-node-check)
2. [Experts at 2, 3, 4 and 5 bits](#2-experts-at-2-3-4-and-5-bits)
3. [The bit allocation](#3-the-bit-allocation)
4. [Saving and opening allocations](#4-saving-and-opening-allocations)
5. [Expert optimisation](#5-expert-optimisation)
6. [Context length, KV size and the KV pool](#6-context-length-kv-size-and-the-kv-pool)
7. [Concurrent inference and the GPU split](#7-concurrent-inference-and-the-gpu-split)
8. [Drafters: MTP and DFlash2](#8-drafters-mtp-and-dflash2)
9. [Reasoning control](#9-reasoning-control)
10. [Health check](#10-health-check)
11. [Engine cores and the RDMA line](#11-engine-cores-and-the-rdma-line)
12. [Display settings](#12-display-settings)
13. [Structured output](#13-structured-output)
14. [Same answers (deterministic mode)](#14-same-answers-deterministic-mode)

## 1. Ring or switch, and the node check

**What it does.** Expert Atlas runs on four or eight DGX Sparks on one RoCE v2 switch, or on four cabled to each other
as a ring with no switch. The node check finds which one you have, checks every node and registers them.

**Why.** A ring needs no switch: each node's two ConnectX-7 ports go straight to its two neighbours. The engine,
weights and panel are the same on both. On a ring the node opposite each node has no link of its own, so its
neighbours relay what goes to it. Measured on four nodes (`RING.md`): the first token came 4 to 5 % later than on a
switch, decode was as fast or slightly faster, and the accuracy was the same.

**How.**

- The panel opens on the node check while no nodes are registered. Press the button once: it detects the cabling, runs
  `tf-glm53 check` on every node (ssh, GPU, memory, rendezvous, RDMA, NCCL, files), copies the package where it is
  missing, and writes the nodes to `glm53.nodes.conf`.
- **Ring**: the head follows the cables over LLDP, so every node needs `lldpd`. Cable each node's port 0 to the next
  node's port 1. The ring step also measures each lane's bandwidth.
- **Switch**: the head tries ssh on its ConnectX-7 /24 and lists the nodes that answer. Choose the node count and tick
  the nodes to use. On a switch with other people's nodes, choose them first, then press.
- `docker run ... IMAGE up` does the same check from the command line (`QUICKSTART.md`).
- `glm53.nodes.conf` holds `TOPOLOGY`, `NODES`, `SOCKET_IFNAME` and `IB_HCA`. Its lines win over `glm53.conf`.

## 2. Experts at 2, 3, 4 and 5 bits

**What it does.** The model has 19,200 routed experts (layers 3 to 77, 256 a layer). Each one can load at 2, 3, 4 or
5 bits, from four banks: one directory a width, the same path on every node.

**Why.** Wider experts are closer to the official model but take more memory, which leaves less for the context and the
KV cache. You choose the balance for your work.

| Width | Bank size | KL divergence, every expert at this width (`AI-GUIDE.md`) |
|---|---|---|
| 2 bits | 187.3 GB | 0.0517 |
| 3 bits | 277.9 GB | 0.0203 |
| 4 bits | 368.5 GB | 0.0127 |
| 5 bits | 459.1 GB | 0.0114 |

(KL divergence from the official BF16 model, positions 6,144 to 8,190. Lower is closer.)

On four nodes (`T4.md`, the 4-bit KV cache, plans asking for a context length of at most 737,280): all 3 bits and
half 3 / half 4 bits both fit the 737,280 asked. All 4 bits fit 327,680. All 5 bits did not fit. At the same memory, mixes with
2-bit experts were 13 to 30 % further from the reference than mixes without them. 5-bit experts paid for by moving
others down to 3 bits gained nothing measurable.

**How.**

- The Base weights and the 3-bit bank are always required. The 2, 4 and 5-bit banks are optional: a width without its
  bank is not offered.
- Set the directories in the launch settings' [ WEIGHTS ], or with `BANK_2` to `BANK_5` (`QUICKSTART.md`).
- `-e BITS=N` on the `docker run` line starts every expert at one width. The panel can give each expert its own width
  (next section).

## 3. The bit allocation

**What it does.** [ BIT ALLOC ] on the launch screen sets each width's share of the experts. The most used experts get
the widest bits.

**Why.** Some experts are picked far more often than others. Giving the extra bits to the experts the model uses most
raises the mean bits of the experts actually picked, at the same memory.

**How.**

- Type the shares of 5, 4 and 3 bits. The narrowest width with data takes the rest, so the shares add up to 100 %.
- The experts are ranked by a usage record: how often each expert was picked. The ranking weighs an expert's decode
  share 0.7 and its prefill share 0.3. These weights kept the most of the text the model writes (`ATLAS-TESTS.md`).
- A plan with no usage record of its own is ranked by the counts shipped in the page: 12 kinds of work, measured for
  this release (coding, agents, Japanese and English text, Linux operations, security).
- Count (decode, prefill, or both) changes only what the page shows. The plan's bits do not change.
- Each node's memory and the verdict (Can start, Not enough memory) follow the plan as you type.

## 4. Saving and opening allocations

**What it does.** An allocation is the bits of every expert plus its usage record. You can save it under a name and
open it again.

**Why.** A plan that suits your work, or the result of an expert optimisation, can be used again at later starts. The
usage record travels with the allocation, so later runs add to it.

**How.**

- Nothing is saved by itself.
- **Save** in [ BIT ALLOC ] on the live panel saves at any time. **Save and stop** saves when you stop the engine.
- **Open** on the launch screen lists the last run's allocation and the saved ones. Opened, an allocation becomes the
  plan bit for bit, and the start uses it.
- **Rename** and **Delete** work on saved allocations. A deleted or overwritten one goes to the launcher's trash.
- The files are kept on the head node, in `atlas/allocs/user/<name>.json` in the release's directory (`ATLAS_DIR` in
  `CONFIG.md`).
- From the command line: `-e ALLOC=NAME` on the `docker run` line, or `./tf-glm53 start` for the last start's plan
  (`QUICKSTART.md`, section 6).

## 5. Expert optimisation

**What it does.** While the engine runs, Optimise experts re-sorts the loaded experts by this run's usage record (the
record the run started with, plus this run's picks). Each width keeps its number of experts, so the memory stays the
same.

**Why.** The shipped counts are a general guess. Your own work picks its own experts. After the re-sort, your most used
experts have the widest bits. Measured on four nodes, half 3 / half 4 bits (`AI-GUIDE.md`, `ATLAS-TESTS.md`): the KL
divergence from the all-4-bit model fell 24 % on the reply's tokens at 8,192 tokens, and 32 to 34 % at 128K and 256K
tokens. On kinds of work not in the counts, the gain was smaller (about two thirds).

**How.**

- Press **Optimise experts** in [ BIT ALLOC ]. The estimate shows how many experts change, about how long inference
  stops, and whether every node has the memory for the swap.
- Press **Apply**. The change starts once no request runs or waits. Requests that arrive meanwhile wait.
- A full change of 75 layers took 83 to 90 s on four nodes.
- The conversations kept in the KV pool are dropped, since their KV was computed with the old experts.
- Every node needs room for the largest layer's new copy, plus 2 GiB. The panel refuses the change, naming the node,
  when one does not have it.
- The result is not saved by itself: press **Save**. A start loads the widths it is given and never re-sorts.
- With concurrent inference it works the same way.

## 6. Context length, KV size and the KV pool

**What it does.** The KV cache is one pool shared by every conversation. Earlier conversations stay in it, so their next
turn does not compute the prompt again.

- **Context length** (`CONTEXT_LENGTH`): the longest request, prompt and reply together. At most 1,048,576.
- **KV size** (`KV_SIZE`): the tokens the whole pool holds. At least the context length + 512. The rest of the pool
  holds earlier conversations.

**Why.** Agents and chats send the same long prefix again and again, and computing a long prompt takes minutes (about
120 s for 128K tokens on four nodes, `T4.md`). Read from the cache, it is almost free. Measured on four nodes (`T4.md`):
a conversation of 703,916 tokens sent again resumed 703,899 positions from the pool and gave its first token in under
3 s.

**How it works** (`RUNTIME.md`, "Conversations kept between requests").

- The pool is made of blocks of 256 positions. A request resumes from the longest cached prefix of any conversation.
- Conversations that begin alike (the same system prompt, or a sub-agent) hold their common blocks once.
- Up to 64 conversations are kept. When the pool is full, the least recently used conversations give up blocks from
  their ends first.
- Every block is checked against the request's tokens before it is reused. On any doubt the request is computed from the
  start ("cold").
- Expert optimisation and a restart drop the conversations.
- Byte Resume (`TF_GLM53_BYTE_RESUME`, on by default) keeps the cache when the model cut a text into tokens differently
  from the tokenizer. You need not do anything.

**How to use it.**

- The launch screen's [ NODE ] sets the context length, the KV size and the KV format. By default the KV takes all the
  room the plan leaves.
- **KV format**: FP4 uses less memory a position than FP8 (1.73 GiB against 2.55 GiB for each 100,000 positions,
  `RUNTIME.md`) and is a little less accurate (it added 0.003 to 0.005 KL divergence, `T4.md`).
- **Keep for the OS** (`RESERVE_GB`): more kept for the OS leaves less for the KV.
- The live panel's KV POOL shows the pool: the share in use, the conversations kept, and how much of each prompt came
  from the cache (`SCREENS.md`).

## 7. Concurrent inference and the GPU split

**What it does.** By default the engine runs one request at a time. Up to 32 more wait, and more than that are refused
as busy. With concurrent inference (`MULTI`, 1 to 4 on the panel) the engine keeps up to that many conversations live
and decodes them together.

**Why.** Several users, or an agent with sub-agents, can then be served at once instead of in turn. It costs node
memory: on a ring with a context length of 131,072 and 16 GiB kept for the OS (`BEFORE-START.md`, the first start's
estimate), the KV fell from 753,664 tokens with MTP to 360,448 with DFlash2 and `MULTI=4`.

**How.**

- Choose **Concurrent** 1 to 4 on the launch screen (1 is one at a time), or `-e MULTI=N` on the `docker run` line. It
  works with MTP, DFlash2 or no drafter. Not on 3 nodes yet.
- `MULTI_WAIT` (600 s by default): how long a request waits for room. Past it the request is refused as busy, so the
  client retries.
- While it runs, [ NODE ] shows **Concurrent** and three controls:
  - **GPU split**: how the GPU's time is shared while a prompt is computed beside conversations that are decoding.
    Prefill 100 % computes prompts first. Decode 100 % lets the replies go first. The start value is 30 % for decode
    (`MULTI_DECODE_SHARE`).
  - **Priority prompt**: a prompt with at most this many new tokens (8,192 by default) goes before longer ones. Only the
    part past the cache counts, so a long conversation's next turn counts only its new part.
  - **Overtake a running prefill too** (off by default): a long prefill already running gives way to a priority prompt
    and resumes from the chunks it finished.
- A change on the live panel lasts until the engine stops. To keep it for later starts, set it in the engine settings.

## 8. Drafters: MTP and DFlash2

**What it does.** A drafter proposes the next tokens and the model checks them. The tokens it accepts come at once, so
decoding is faster. The reply is always the model's own.

**Why.** Decode speed. Measured with release r2-102 (`RUNTIME.md`, not measured again for this release): on four nodes
with a 128K prompt, decode was 35.3 tokens/s with MTP and 42.9 with DFlash2; on eight nodes, 47.5 and 53.2.

**How.**

- **MTP** (default): the multi-token prediction layer already in the weights. Nothing to download.
- **DFlash2**: a separate model, `incoai/GLM-5.3-DFlash2`, with its own licence. It is not shipped: download it to every
  node and set its directory (`DRAFTER`, or the drafter row of [ WEIGHTS ]). The panel greys DFlash2 until every node has
  it. It takes about 3 GiB more a node on four nodes and about 4 GiB on eight (the launcher's planned sizes), so the
  KV is smaller.
- **None**: no drafter.
- Choose it with **Drafter** on the launch screen, or `-e DRAFTER_KIND=mtp|dflash2|none`.

## 9. Reasoning control

**What it does.** GLM-5.3 thinks before it answers, and `max_tokens` counts the thinking too. [ REASONING CONTROL ] sets
how much of a reply the thinking may use and when the thinking is organized.

**Why.** A long thinking can use up a reply's max_tokens and leave no room for the answer, and a very long thinking
fills the KV. These settings keep room for the answer and keep the thinking's KV bounded. The client still gets all of
the reasoning.

**How.** On the live panel, a change applies to the requests that start after it and is written to `glm53.conf`. A
request's own settings come first.

- **Thinking limit** (`THINKING_SHARE`): 5 % to 100 % (No limit) of each reply's max_tokens. At the limit the engine
  closes the thinking at a line break, and the model answers in the rest.
- **Organize midway** (`THINKING_FOLD`): the model writes a checkpoint of its thinking (its conclusions so far, what was
  ruled out, what is still open, what to examine next). The engine puts the checkpoint in place of the thinking in the
  context. The limits then count the tokens written, so they can pass the context length.
- **Organize every** (`THINKING_FOLD_SHARE`): how often it organizes, as a share of max_tokens (at least 1,024 tokens).
  100 % organizes only at the limit.
- **Organize at the close** (`THINKING_FOLD_AT_CLOSE_SHARE`): when the model closes its thinking, it first organizes it.
  A long reasoning becomes a plan before the answer, and the next turns get only the organized text.
- **Notice of what is left** (`THINKING_NOTICE`, engine settings only, needs a restart): every so many thinking tokens,
  the engine tells the model how many it has used and how many are left, so it can pace itself.
- Near the end of the thinking's budget the engine also asks the model to rewrite its thinking (`THINKING_FOLD_AT_BUDGET`,
  on by default). So a reply can write more than the limit.

The engine settings screen sets the same values and the sizes they work with (`CONFIG.md`, "Context, cache and KV").

## 10. Health check

**What it does.** **Health check** in [ NODE ] sends the running engine one request of about 1,000 tokens and judges the
reply. The right answer can only come from reading the prompt.

**Why.** To confirm that the running engine reads a prompt and answers it correctly, not only that it is up.

**How.** Press **Health check**. **Inference OK** shows over the circle for a few seconds. **Inference error** stays under
the button with its reason. It normally takes about 3 s; behind a long request it waits its turn, up to 600 s.

## 11. Engine cores and the RDMA line

**What it does.** The GB10 has two kinds of CPU core: Cortex-X925 and Cortex-A725. Two settings choose where the engine's
threads run.

- **RDMA cores** (`ROCE_PROXY_CORES`, default A725): the RoCE proxy threads, two a rank, which spin while the engine
  decodes. A725 keeps them on cores 0 and 10. OS leaves them to the scheduler, which often puts them on X925 cores.
- **Engine cores** (`ENGINE_CORES`, default OS): OS lets every rank use every core. A725 keeps the ranks on the A725
  cores only.

**Why.** Spinning threads on the fast X925 cores heat the chip. On four nodes, keeping the proxy threads on A725 ran the
chip 5 to 16 °C cooler with the same decode speed (`CONFIG.md`). The speed with the engine on A725 alone has not been
measured.

**How.** Set both in the engine settings (Nodes and parallelism); they take effect at the next start. While the engine
runs, [ NODE ] shows the result: the RDMA column lists the CPUs of each node's two proxy threads (for example 0·10), and
the lines under the table say RDMA proxy threads and Engine cores.

## 12. Display settings

**What it does.** The **Display** drawer changes how the panel looks: the language, the circle's order, afterglow and
brightness, the light's colour, the rim's text, the digits, the centre and the colours. It does not change the engine.

**Why.** So the panel reads well on your screen. The settings are kept on the lane, not in the browser, so everyone
who opens the panel sees the same one.

**How.**

- The settings are saved on the head node, in `launch/panel-settings.json` in the package's directory
  (`TF_PANEL_SETTINGS` in `EXTRA_ENV` moves it). Every browser of the lane shows the same panel.
- The node list and the ticks of the node check are kept in the same file.
- **Back to defaults** (press it twice) drops the display settings for every browser of the lane. The language and the
  nodes stay.
- A ring or switch lane opens with the light by bit and the Material · indigo colours, unless its settings name others.

## 13. Structured output

**What it does.** A reply's answer is held to JSON: a JSON object, or the JSON that a JSON Schema you send accepts. At
each token the engine lets the model choose only among the tokens that keep the answer in that form, so the answer
parses and follows the schema, unless `max_tokens` cuts it first. The grammar is llguidance 1.9.1's, on the first node.
It is on by default (`STRUCTURED`).

**Why.** A program that reads the reply, such as an extraction, a form or an agent's step, gets JSON it can parse, with
the keys and the types it expects, without asking the model again.

**How.** Ask for it in the request:

- `/v1/chat/completions` (OpenAI's chat): `"response_format": {"type": "json_object"}` for a JSON object, or
  `{"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}` for JSON the schema accepts. vLLM's
  fields are read too: `"guided_json": SCHEMA`, and `"structured_outputs"` with `{"json": SCHEMA}` or
  `{"json_object": true}`. `{"type": "text"}` asks for no constraint.
- `/v1/responses` (OpenAI's Responses API): `"text": {"format": {"type": "json_schema", "name": "...", "schema": {...}}}`,
  or `{"type": "json_object"}`.
- `/v1/messages` (Anthropic's Messages API): `"output_config": {"format": {"type": "json_schema", "schema": {...}}}`.

On the chat API, with the thinking off:

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

With Python's `openai` package, and the thinking on:

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

- **With thinking.** GLM-5.3 thinks by default on the chat API and the Responses API, and on the Messages API when
  `thinking` asks for it. The thinking is free and the constraint starts where it closes: the thinking comes in
  `reasoning_content` (a `thinking` block on the Messages API) and the JSON in the answer. The engine closes the
  thinking early enough to leave the answer at least 256 tokens, so with the thinking on, `max_tokens`
  (`max_output_tokens` on the Responses API) must be at least 545; a smaller one is refused (400). For a short reply,
  turn the thinking off: `"chat_template_kwargs": {"enable_thinking": false}`, as in the curl line.
- **Refused with 400**, before any header of the reply (a stream too), each with a message that says why:
  - with `tools` (a list that is not empty), or `tool_choice` `required` or a named function: not yet;
  - with `stop` strings (they could cut the JSON) or `ignore_eos`;
  - `regex`, `choice` and `grammar` (`guided_regex`, `guided_choice`, `guided_grammar`, and the same under
    `structured_outputs`): not yet; use `json_object` or `json_schema`;
  - a schema the grammar would follow less strictly than it says. The server checks the schema first and refuses it,
    rather than return JSON that does not follow it: a `$schema` other than JSON Schema 2020-12; `items` as an array, or
    `additionalItems` (draft 4's tuples: write `prefixItems`); a `$ref` other than `#`, `#/$defs/NAME` or
    `#/definitions/NAME`; `nullable` (OpenAPI's: write `"type": ["integer", "null"]`); an integer bound or constant
    past ±2^53; a `multipleOf` that is neither an integer from 1 to 65,536 nor a decimal of at most 8 places that reads
    at most 65,536 without its point (0.25, 0.01), or two `multipleOf` for one value; `minProperties` or `maxProperties`
    on an object open to other keys; `x-guidance` (the server sets it); a schema over 64 KiB or nested deeper than 32
    levels; a lone surrogate, NaN or Infinity.
- **Good to know.**
  - An object with `properties` and no `additionalProperties` takes only the keys it declares (`allOf`'s branches
    aside). Write `"additionalProperties": true` to allow others.
  - A reply cut by `max_tokens` ends with `finish_reason` `length`, and its answer is JSON not yet finished: check
    `finish_reason` before parsing.
  - The JSON has no `\u` escapes: other characters are written as themselves, so a key has one spelling, and a control
    character without a short escape (`\n`, `\r`, `\t`, `\b`, `\f`) cannot be written. Between two tokens it has at
    most one line break and up to 64 spaces or tabs.
  - A structured request decodes without the drafter (MTP or DFlash2), so it decodes more slowly than a plain request
    of the same length. With `MULTI` it is one of the conversations decoded together (one row of the window a round,
    beside the others' drafted rows), from r3.3.7-102. Plain requests are as before: the same tokens and the same speed
    as on r3.3.2-102.
  - To turn it off: **Structured output** in the engine settings (Server), `-e STRUCTURED=0` on the `docker run` line or
    `STRUCTURED=0` in `glm53.conf`, from the next start; such requests then get 400 (`CONFIG.md`). If the first node cannot load
    `lib/libllguidance.so`, the engine starts without it (rank 0's log: `structured output unavailable`) and they get
    the same 400.
  - Checked on the switch's eight nodes: JSON objects and schemas, with the thinking, streamed, cut by `max_tokens`, a
    stream the client dropped, on the three APIs, and every refusal, with every rank running after it. Plain requests
    gave the same tokens as on r3.3.2-102 for 20 prompts.

## 14. Same answers (deterministic mode)

**What it does.** With `DETERMINISTIC=1`, the same requests sent in the same order from a start give the same tokens
after every start, on the same release, nodes and settings. This holds at any temperature: a request without a `seed`
gets one from its prompt (in either mode), so a sampled reply repeats too. It is off by default.

**Why.** By default two things can differ from one start to the next: the engine times its linears' kernel tiles at
each start and keeps the fastest, and a long prompt's routed experts add their results in the order they finish. Either
can move the last bit of a number, and now and then that changes a token. To compare two settings, repeat a test or
reproduce a reply, the same request has to give the same answer.

**How.** `DETERMINISTIC=1` in `glm53.conf`, `-e DETERMINISTIC=1` on the `docker run` line, or **Same answers
(deterministic)** in the engine settings (Server), from the next start. `/health` then says `"deterministic": true`,
and so does each reply's `tensorfold` object on the chat and Responses APIs; its `token_sha`, a hash of the reply's
tokens, compares the replies of two starts.

In this mode the EXL3 linears take the tiles their shapes fix, never timed ones, and the prompt's routed experts add
each row's slots in slot order, not as they arrive, and never as fp16 sums. At the start every rank compares its
settings, its tiles, the libraries it loaded and NCCL's settings, and the start stops if one differs. On the switch,
NCCL runs the Ring algorithm on two channels, as on the ring.

- **Limits.**
  - One conversation at a time: with `MULTI` 1 or more the start is refused. The launch screen greys concurrent
    inference 2 to 4 and says why.
  - `EXTRA_ENV` and `LAUNCH_ENV` may not set what the mode fixes: a start with any of `TF_GLM53_TUNE`, `TF_GLM53_PE_F16`, `TF_GLM53_DETERMINISTIC`, `TF_EXL3_PROMPT_EXPERTS`, `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_PAT_ENABLE`, `NCCL_NVLS_ENABLE`, `NCCL_MIN_NCHANNELS`, `NCCL_MAX_NCHANNELS` in them is refused. The
    engine settings screen does not save the mode beside concurrent inference 2 or more, or beside one of those names.
  - `CONTEXT_LENGTH` and `KV_SIZE` are taken as set. A start never cuts the context length to the free memory or gives
    the KV all the room left, since both would change with the nodes' free memory: when they do not fit, the start
    stops.
- **What still differs.**
  - A tool call's id is random: a new one every time.
  - Within one start, a prompt asked again after the cache dropped it can get a reply different from the first: where
    the cache's blocks begin changes the order the prompt is computed in. Started again and sent in the same order,
    the requests differ the same way.
- **Cost** (the switch's eight nodes): prefill about 20 % slower (4K tokens in 3.31 s against 2.75 s, 40K in 29.1 s
  against 23.6 s), decode about the same (45.3 against 45.9 tokens/s). Each rank holds 384 MiB more for the prompt
  experts (`ROWS` 8,192 and `SP_PARTS` 4; 768 MiB with `SP_PARTS` 2 or `ROWS` 16,384), counted in the KV plan.
- **Checked** with two starts each: on the switch (eight nodes) 23 requests (12 short ones, 4K and 40K prompts, tools,
  4 structured, 4 at temperature 1.0) gave the same tokens, 23 of 23, and the KV cache was the same in all 232
  comparisons; on the ring (four nodes) 23 of 23 and 112 of 112. Organizing the thinking midway, three-turn
  conversations (the pool's continuation and Byte Resume) and KV eviction: 13 of 13 with MTP and with DFlash2, on a
  cold start too.
