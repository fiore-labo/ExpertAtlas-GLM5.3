English | [日本語](QUICKSTART.ja.md) | [简体中文](QUICKSTART.zh-CN.md)

# First start: from the image to a running engine

This page is for someone starting Expert Atlas for the first time. Expert Atlas serves full GLM-5.3 on NVIDIA DGX Spark.
You download the weights, run one `docker run` line on the first node, and the engine starts. If something is missing,
the same line tells you what to do.

## What runs where

- **The engine** runs in one container on each node. `docker run ... up` starts these containers for you.
- **The launcher** runs on the first node's host, outside any container. It serves the dashboard and the control API, and
  it starts and stops the engine.
- **The `up` container** only sets things up. It exits when the engine is running. The engine and the launcher stay on.
- **The weights** stay on the hosts. They are not in the image.

Neither the dashboard nor the API has a key, and both listen on every address of the first node. Keep the nodes on a
closed network (section 4 has how to keep them to the first node).

## 1. What the nodes need

| What | Detail |
|---|---|
| Nodes | 4 or 8 DGX Sparks on one RoCE v2 switch, or 4 cabled as a ring without a switch (`RING.md`). 2 or 3 nodes are being tested |
| Software on each node | The NVIDIA driver 580 or later (the image uses the host's driver; `nvidia-smi` shows it), Docker and the NVIDIA container toolkit (DGX OS has both), and `python3` |
| ConnectX-7 network | An address on each half of the cabled ports, MTU 9000 (below) |
| ssh | From the first node to every node, and to itself, with a key that has no passphrase and host keys it already knows (below) |
| Docker access | Your user can run `docker` on every node (the `docker` group) |
| Memory | About 115 GiB free on each node before the start, with nothing else on the GPU |
| Disk | For the two required downloads, on each node: at least 299.1 GB, on four nodes and on eight |

The release's image holds NCCL with the AArch64 fence, and reads it on a switch and on a ring (`RING.md`, "The fenced NCCL").

### The ConnectX-7 network

A ConnectX-7 port is two PCIe halves, each a network device with its own address: port 0 is `enp1s0f0np0` and
`enP2p1s0f0np0`, port 1 is `enp1s0f1np1` and `enP2p1s0f1np1`. Each half needs its own /24 and an MTU of 9000.

On a switch, give the two halves of the cabled port an address on every node: the same two /24s everywhere, a different
last number on each node. For port 1, on one node, with example addresses:

```sh
sudo tee /etc/netplan/01-cx7.yaml >/dev/null <<'EOF'
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    enp1s0f1np1:
      mtu: 9000
      addresses: [172.16.88.1/24]
    enP2p1s0f1np1:
      mtu: 9000
      addresses: [172.16.89.1/24]
EOF
sudo chmod 600 /etc/netplan/01-cx7.yaml
sudo netplan apply
ip -br addr    # both halves UP, with their addresses
rdma link      # both halves ACTIVE
```

On a ring, all four halves have addresses, and every node also needs `lldpd`: `RING.md`, "Cabling and addresses".

DGX OS already sets the kernel settings this needs (`arp_ignore=1`, `arp_announce=2`, `rp_filter=2`), so there is nothing
to set by hand.

### ssh

The launcher connects with a key that has no passphrase, and it never adds a host key it does not know. So the first node
must already know every node's host key, its own too. On the first node:

```sh
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
for a in NODE1 NODE2 NODE3 NODE4; do    # every node's address, the first node's too
  ssh-copy-id -o StrictHostKeyChecking=accept-new "$a"    # asks for the password
  ssh -o BatchMode=yes "$a" true && echo "$a OK"
done
```

`NODE1` and the others are the addresses the launcher uses. On a switch, each node's ConnectX-7 address on the `enp1s0f…`
half (`172.16.88.N` above): the first node finds the others by trying ssh's port on that /24. On a ring, each node's LAN
address, the LAN being the interface of its default route. If your management network is another interface, write the node
settings by hand (`RING.md`, "Settings").

## 2. Download the weights

**Two downloads are required: the Base weights and the 3-bit bank.** The engine does not start without them, even if you
plan to use another width. Every path must be the same on every node.

| What | Hugging Face repository | Size | Default directory | Option |
|---|---|---|---|---|
| Base weights (**required**) | `fiore-lab/GLM-5.3-EXL3-Base` | 21.2 GB | `~/models/GLM-5.3-EXL3-Base` | `WEIGHTS` |
| 3-bit bank (**required**) | `fiore-lab/GLM-5.3-EXL3-Experts-3bit` | 277.9 GB | `~/models/GLM-5.3-EXL3-Experts-3bit` | `BANK_3` |
| 2-bit bank (optional) | `fiore-lab/GLM-5.3-EXL3-Experts-2bit` | 187.3 GB | `~/models/GLM-5.3-EXL3-Experts-2bit` | `BANK_2` |
| 4-bit bank (optional) | `fiore-lab/GLM-5.3-EXL3-Experts-4bit` | 368.5 GB | `~/models/GLM-5.3-EXL3-Experts-4bit` | `BANK_4` |
| 5-bit bank (optional) | `fiore-lab/GLM-5.3-EXL3-Experts-5bit` | 459.1 GB | `~/models/GLM-5.3-EXL3-Experts-5bit` | `BANK_5` |
| Full checkpoint, 4 nodes (optional) | `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks` | 344.6 GB | none | `WEIGHTS` |
| Full checkpoint, 8 nodes (optional) | `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks` | 389.9 GB | none | `WEIGHTS` |
| DFlash2 (optional, separate licence) | `incoai/GLM-5.3-DFlash2` (`RUNTIME.md`, "Using DFlash2") | | none | `DRAFTER` |

The Base weights are everything but the routed experts, which a start reads from the banks. The full checkpoints hold those experts as well, and a start does not need them. If you already have one, name it with `WEIGHTS` and it works as it is. The 8-node one also serves as the 4-bit bank (`RUNTIME.md`, "The control panel").

Download once, then copy to the other nodes. If `python3 -m venv` fails, install the `python3-venv` package. For four nodes:

```sh
python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install -U huggingface_hub
for r in EXL3-Base EXL3-Experts-3bit; do
  D=~/models/GLM-5.3-$r
  ~/hf-cli/bin/hf download fiore-lab/GLM-5.3-$r --local-dir $D
  for n in NODE2 NODE3 NODE4; do    # the other three nodes
    rsync -a --mkpath -e "ssh -c aes128-gcm@openssh.com" $D/ $n:$D/ &
  done; wait
done
```

Weights already somewhere else? Keep them there and name the directory with an option (section 4).

### What the widths mean

The routed experts of the model can be loaded at 2, 3, 4 or 5 bits. **The default start loads every expert at 3 bits.** On
four nodes this fits with the default context length of 1,048,576 tokens. You can start every expert at another width with
`BITS`. The dashboard can also choose a width for each expert from the banks you have.

## 3. Start

On the first node, as the user who will run it:

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host IMAGE up
```

`IMAGE` is the release's image, `ghcr.io/fiore-labo/expertatlas-glm5.3:r3.3.5-102`. The line writes the release to `~/expertatlas`, finds the nodes (a ring or a switch), checks
them, looks for the weights, starts the engine and sends one request. The first start takes a few minutes.

It ends in one of two ways.

**`READY`: the engine runs.**

```
READY
  inference API       http://203.0.113.10:8000/v1 (OpenAI compatible, no key, model glm-5.3)
  dashboard (no key)  http://203.0.113.10:8010/panel
  control API         http://203.0.113.10:8010/v1/launch/status
```

**`STOP`: something needs you.** `STOP` says what is wrong, `FIX` says what to do, and `THEN` gives the line to run next.
The launcher stays up, so the dashboard opens too.

```
STOP: the model's files are not on the nodes:
  3 bit (always needed): ~/models/GLM-5.3-EXL3-Experts-3bit: not on node-1, node-2 (...)
    put fiore-lab/GLM-5.3-EXL3-Experts-3bit there, the same path on every node (...)
FIX:  ...
  hf download fiore-lab/GLM-5.3-EXL3-Experts-3bit --local-dir ~/models/GLM-5.3-EXL3-Experts-3bit
THEN: docker run --rm --network host -e USER="$USER" -v "$HOME":/host IMAGE up
```

Do what `FIX` says, then run the `THEN` line. Running the line again is always safe: it does not restart a running engine.
`hf` in a `FIX` line is the Hugging Face CLI of section 2, `~/hf-cli/bin/hf`.

## 4. Options

**Everything you can choose in the launch settings is a `docker run` option:** add `-e NAME=VALUE` before `IMAGE`. A name it
does not know stops the line, so a typo cannot pass unnoticed.

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host \
  -e WEIGHTS=/data/models/base \
  -e BANK_3=/data/models/3bit \
  -e BANK_4=/data/models/4bit \
  -e BITS=4 -e CONTEXT_LENGTH=131072 \
  IMAGE up
```

Where the files are (the same path on every node):

| Option | Default | Meaning |
|---|---|---|
| `WEIGHTS` | `~/models/GLM-5.3-EXL3-Base` | The Base weights, or a full checkpoint. Required |
| `BANK_3` | `~/models/GLM-5.3-EXL3-Experts-3bit` | The 3-bit bank. Required |
| `BANK_2`, `BANK_4`, `BANK_5` | `~/models/GLM-5.3-EXL3-Experts-<N>bit` | The other banks. Optional |
| `DRAFTER` | empty | DFlash2's model, used with `DRAFTER_KIND=dflash2` |

The start:

| Option | Default | Meaning |
|---|---|---|
| `BITS` | `3` | Every expert at this width (2 to 5). Its bank must be on every node |
| `ALLOC` | none | A saved allocation (its per-expert widths and usage), instead of `BITS` |
| `NODES` | found | The workers' addresses on a switch, when more nodes are free than a lane uses. A ring finds its own |
| `CONTEXT_LENGTH` | `1048576` | The longest prompt plus reply. Cut to fit when too long, and the output says so |
| `KV_SIZE` | `max` | The tokens the KV cache holds in all: a number, or `max` for all the memory left. For this start only |
| `KV` | `fp4` | The KV cache's format: `fp4` or `fp8` |
| `DRAFTER_KIND` | `mtp` | `mtp`, `dflash2` or `none` |
| `MULTI`, `MULTI_WAIT` | `0`, `600` | Conversations decoded together (0 to 4), and how long a request waits for them, in seconds |
| `RESERVE_GB` | `4` | The memory each node keeps for the system |

Every other key of the engine settings works the same way, for example `MAX_TOKENS`, `THINKING_SHARE`, `THINKING_FOLD_PROMPT`,
`NAME`, `HOST`, `STRUCTURED` (`0` turns structured output off) or `DETERMINISTIC` (`1`: the same answers after every start). `CONFIG.md` gives each key's meaning and the keys that are not options. Its defaults are `tf-glm53`'s: for this line, the
tables here come first.

This install:

| Option | Default | Meaning |
|---|---|---|
| `EA_DIR` | `expertatlas` | The directory under your home |
| `PORT` | `8000` | The inference API |
| `LAUNCHER_PORT` | `PORT` + 10 | The dashboard and the control API |
| `IMAGE` | this image | The name the nodes pull and run |

Good to know:

- Values you give are saved in `~/expertatlas/glm53.conf` and stay until you give another value. `KV_SIZE` and `BITS` count
  for that start only. Widths are not kept: without `BITS` or `ALLOC`, the line starts every expert at 3 bits, whatever the
  dashboard chose.
- `TOPOLOGY`, `SOCKET_IFNAME` and `IB_HCA` are not options: the node check decides them from the cabling.
- Options do not change a running engine. Stop it first, then run the line. The output says so when it ignored any.
- To keep the API to the first node, add `-e HOST=127.0.0.1`. `LAUNCHER_HOST`, for the dashboard and the control API, is not
  an option: write `LAUNCHER_HOST=127.0.0.1` in `~/expertatlas/glm53.conf`, stop the launcher (section 6), and run the line again.

## 5. Use it

The API speaks the OpenAI chat format. Point any client at the address `READY` printed:

```sh
curl -s http://NODE:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3","messages":[{"role":"user","content":"hello"}]}'
```

A reply's thinking is in `reasoning_content` and its answer in `content` (`delta.reasoning_content` and `delta.content` in a
stream).

For JSON, add `"response_format": {"type": "json_object"}`, or a JSON Schema with `json_schema`: the answer is held to
it. With the thinking on (the default), `max_tokens` must be at least 545; for a short reply add
`"chat_template_kwargs": {"enable_thinking": false}`. `FEATURES.md`, "Structured output", has the rest.

The dashboard at `http://NODE:8010/panel` shows the engine and starts and stops it. An AI agent can do the same through
the control API: `AGENT-API.md` has the steps, and `AI-GUIDE.md` describes the whole system.

- `SCREENS.md` shows every screen of the dashboard, with pictures.
- `FEATURES.md` explains each feature: what it does, why it is there and how to use it.

On a switch that also has other people's nodes, give your own to the start line (`-e NODES=...`). In the dashboard's node
check, choose the node count and the ticks first, then press (`PANEL-GUIDE.md`, "Node check").

## 6. Stop and start again

```sh
cd ~/expertatlas
./tf-glm53 down              # stops the engine
./tf-glm53 launcher stop     # stops the launcher too, if you want
```

To start again, run the same `docker run` line. The nodes, the weights' locations and your settings are remembered, but not
the widths chosen in the dashboard: the line starts every expert at `BITS`, 3 by default. To start with the last start's plan,
run `./tf-glm53 start` instead, or give a saved allocation with `-e ALLOC=NAME`.

## 7. Common STOP messages

| Message | What to do |
|---|---|
| `up needs the host's home at /host` | Use the line exactly: `-e USER="$USER" -v "$HOME":/host` |
| `this container cannot ssh to the host` | Make a key without a passphrase and add it to your own `authorized_keys`. The message gives the commands |
| `IMAGE is not on NODE and docker pull failed` | Copy it with the `docker save \| ssh NODE docker load` lines the message gives |
| `3 bit (always needed)`, `checkpoint (always needed)` | Download it (section 2), or name its directory with `BANK_3` or `WEIGHTS` |
| `no other node answers over ConnectX-7` | Check the cables and the addresses (section 1) |
| `its host key is not known here`, `it refuses this node's ssh key` | Do the ssh steps of section 1 for that address |
| `N free worker nodes found` | Choose them with `-e NODES="ADDR ADDR ADDR"` |
| `no ring found`, `the ring is open` | Every node needs `lldpd` (section 1). Check the cables and ssh too |
| `its LAN address is not known here` | From the first node, ping that node's LAN address, then run the line again. If it still stops, write the node settings by hand (`RING.md`, "Settings") |
| `port N answers another lane's launcher` | Stop that launcher, or add `-e PORT=N` |
| `does not fit with context length N` | Add a smaller `CONTEXT_LENGTH` or a narrower `BITS`, or free the nodes' memory |
| `not an option` | Check the spelling. The message suggests the nearest name |

More: `BEFORE-START.md` (what to decide before the first start), `RUNTIME.md` (every command and failure), `T4.md` (four nodes and their measurements), `RING.md` (a ring),
`CONFIG.md` (every setting). In the source repository, `RUNTIME.md` is `packaging/RUNTIME.md` and the others are in
`glm53/release/`.
