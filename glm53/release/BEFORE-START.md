English | [日本語](BEFORE-START.ja.md) | [简体中文](BEFORE-START.zh-CN.md)

# What to decide before the first start

You must decide two things: which nodes to run on and where the weights live. Everything else starts with its default, and a
`STOP` tells you what to fix. `QUICKSTART.md` has the steps.

## Prepare and decide before you start

Decide the nodes and the weights' location first. The weights use the same path on every node.

| Decision | Choices | Default | Option |
|---|---|---|---|
| The nodes | 4 or 8 on a switch, or 4 in a ring (2 or 3 are being tested) | Found automatically. Choose only when a switch has more free nodes than a lane uses | `NODES` |
| Where the checkpoint is (required) | 4 nodes 344.6 GB, or 8 nodes 389.9 GB | `~/models/GLM-5.3-EXL3-4x-DGX-Sparks` (8 nodes: `-8x-`) | `WEIGHTS` |
| Where the 3-bit bank is (required) | 277.9 GB | `~/models/GLM-5.3-EXL3-Experts-3bit` | `BANK_3` |
| Banks of other widths | 2 bit 187.3 GB, 4 bit 368.5 GB, 5 bit 459.1 GB. Only the widths you use | `~/models/GLM-5.3-EXL3-Experts-<N>bit` | `BANK_2`, `BANK_4`, `BANK_5` |
| DFlash2's model | Only if you use it. It has a separate licence | None | `DRAFTER` |
| The install directory | A directory under your home | `expertatlas` | `EA_DIR` |
| The ports | The API and the dashboard | 8000 and 8010 (the API + 10) | `PORT`, `LAUNCHER_PORT` |
| Who can reach the API | Every address, or the first node only (`127.0.0.1`) | Every address (`0.0.0.0`) | `HOST` |

Neither the API nor the dashboard has a key. Keep the nodes on a closed network.

The start also needs these, which are not decisions: about 115 GiB free on each node, nothing else on the GPU, addresses on
the ConnectX-7 network, an ssh key without a passphrase with every node's host key known to the first node, and access to
`docker`. `QUICKSTART.md`, section 1, has the steps.

## Decide for each start

Changing these settings needs a restart of the engine. If you are unsure, the defaults start.

| Decision | Option | Choices | Default | How to decide |
|---|---|---|---|---|
| The experts' width | `BITS` or `ALLOC` | 2, 3, 4 or 5 bits, or a saved allocation | 3 bits | A wider one uses more memory, which reduces the next two |
| The longest request | `CONTEXT_LENGTH` | 65,536 to 1,048,576 tokens, a multiple of 16,384 | 737,280 (4 nodes, 3 bits) | Your longest prompt plus reply |
| The KV cache's total | `KV_SIZE` | A token count, or `max` | `max` | Usually `max`. More keeps more conversations cached |
| The KV cache's format | `KV` | `fp4`, `fp8` | `fp4` | `fp8` is more precise and uses more memory |
| What drafts tokens | `DRAFTER_KIND` | `mtp`, `dflash2`, `none` | `mtp` | `dflash2` needs another model and reduces the KV cache |
| Conversations decoded together | `MULTI` | 0 to 4 | 0 (one at a time) | How many use it at once |
| How long a request waits | `MULTI_WAIT` | Seconds | 600 s | With `MULTI`, a request past this is refused as busy |
| Memory left for the OS | `RESERVE_GB` | 2 to 32 GiB | 4 GiB | Raise it if other programs run on the nodes |
| The default reply limit | `MAX_TOKENS_SHARE` | 1 to 100% of `CONTEXT_LENGTH` | 70% | Used when a request names no `max_tokens`. A client's value comes first |
| Byte Resume | `TF_GLM53_BYTE_RESUME` | `on`, `log`, `off` | `on` | Keeps the cache when a text is cut differently. Usually `on` |
| The model's name in the API | `NAME` | Text | `glm-5.3` | Match your client's setting |

- **Thinking and `max_tokens`:** thinking counts as part of the reply. A client's `max_tokens` is the limit, so with a small
  value the thinking is limited to 90% of that too.
- **DFlash2, `MULTI` and memory:** on a ring with `CONTEXT_LENGTH` 131,072 and `RESERVE_GB` 16, the plan's KV cache falls from 753,664
  tokens with `mtp` to 360,448 with `dflash2` and `MULTI=4` on the first start, which estimates; later starts use what the last one measured.
- **When it does not fit:** a `CONTEXT_LENGTH` that is too long is cut to what fits, and the output says so.

## What you need not decide before the start

Reasoning Control (the thinking's limit and its organising midway) and the GPU's split change while the engine runs, so you can tune them as you go.

| Setting | Option | Default | Kept for the next start |
|---|---|---|---|
| The thinking's limit | `THINKING_SHARE` | 90% of the smaller of `max_tokens` and what the context has left | Yes |
| Summaries midway | `THINKING_FOLD` | None (`off`) | Yes |
| The interval between summaries | `THINKING_FOLD_SHARE` | None before the limit (100%). From 10% of `max_tokens` | Yes |
| The most summaries | `THINKING_FOLD_MAX` | No limit (0) | Yes |
| Organising near the budget's end | `THINKING_FOLD_AT_BUDGET`, `THINKING_FOLD_MARGIN` | On, from 10% before the end | Yes |
| A summary at the thinking's end | `THINKING_FOLD_AT_CLOSE_SHARE` | None (0) | Yes |
| The texts the model is asked | `THINKING_FOLD_PROMPT`, `THINKING_CLOSE_PROMPT` | Built in | Yes |
| The GPU's split between prefill and decode (with `MULTI`) | `MULTI_DECODE_SHARE` | Decode 30% | Only when the dashboard is told to |
| Short prompts (with `MULTI`) | `MULTI_SHORT_PROMPT`, `MULTI_SHORT_PREEMPT` | 8,192 tokens or fewer, no overtaking | Only when the dashboard is told to |

Change them on the inference screen's cards or the engine settings screen. An AI agent calls `POST /v1/thinking` and
`POST /v1/schedule` on the launcher's port.

## What is decided for you

The node check decides `TOPOLOGY`, `SOCKET_IFNAME` and `IB_HCA` from the cabling. `docker run` refuses them. A ring also finds
its own `NODES`.

## How to give your choices

On the `docker run` line, add `-e NAME=VALUE` before `IMAGE`. The start screen offers the same choices.

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host \
  -e WEIGHTS=/data/models/checkpoint \
  -e BANK_3=/data/models/3bit \
  -e CONTEXT_LENGTH=131072 \
  IMAGE up
```

- **On the `docker run` line:** values are saved in `~/expertatlas/glm53.conf` and used by later starts. `KV_SIZE` and `BITS`
  count for that start only.
- **On the start screen:** choices count for that start only and are not saved. The engine settings screen changes the defaults.
- **A running engine:** options do not change it. Stop it, then run the line again.
- **A wrong name:** a name it does not know gives a `STOP`, which suggests the nearest name.

`QUICKSTART.md` and `CONFIG.md` explain each item in full.
