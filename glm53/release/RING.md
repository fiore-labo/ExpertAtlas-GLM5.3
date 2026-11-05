English | [日本語](RING.ja.md) | [简体中文](RING.zh-CN.md)

# Four DGX Sparks in a ring, without a switch

The four-node form (`T4.md`) also runs on four NVIDIA DGX Spark systems cabled to each other as a ring of direct
200 Gb/s links, with no switch. The engine, the checkpoint, the banks and the control panel are the same; only how the
ranks reach each other changes. It is Fiore-lab's unofficial derivative of TensorFold 1.0.2, not a release of the
TensorFold project.

## How it runs

- Each node reaches its two neighbours directly. The node opposite it (two links away) has no link of its own; what
  goes to it is relayed by the neighbours:
  - the prompt's all-to-all sends the opposite node's block in two halves, one through each neighbour;
  - the decode collectives on RoCE (`TF_GLM53_ROCE=1`) forward the opposite node's part through the neighbour that
    holds it in the same slot, so the GPU kernels are unchanged.
- NCCL connects rings only (`NCCL_ALGO=Ring`), never the opposite node, over the four PCIe halves of each node
  (4 channels).
- The ranks are the nodes' positions on the ring; each decode context parallel group (DCP 2) is a cabled pair.
- The prompt's partial sums are reduced over the ring (`PREFILL_REDUCE=ring`; reduce-scatter is refused). The
  multi-conversation reduce in shares (AR2) is not used on a ring; `MULTI` runs there all the same, up to 4
  conversations.
- NCCL 2.30.7 with the AArch64 send fence of NVIDIA/nccl#2393 (commit `05d026f4`), built for sm_121. With `RUNTIME=docker`
  the release's image holds it at `/opt/nccl-fenced`, and `tf-glm53 check` looks for it. With `RUNTIME=host` it goes at the
  same path on every node, and the launcher checks its SHA-256. "The fenced NCCL" below gives its build.

## Cabling and addresses

- Cable each node's port 0 to the next node's port 1, around the ring. With port 0 to port 0 and port 1 to port 1
  instead, NCCL's subnet-aware routing finds no device on some links (a listener advertises only its own device's
  address) and connecting fails.
- Give each link one /24 for each PCIe half (a ConnectX-7 port is two halves, `rocep1s0f*` and `roceP2p1s0f*`), with an
  MTU of 9000. DGX OS already sets `arp_ignore=1`, `arp_announce=2` and `rp_filter=2`, so there is nothing to set by hand.
  The example used here (ring order node-0, 1, 3, 2):

| Link | half 0 | half 1 |
|---|---|---|
| node-0 – node-1 | 172.16.88.0/24 | 172.16.89.0/24 |
| node-3 – node-2 | 172.16.90.0/24 | 172.16.91.0/24 |
| node-0 – node-2 | 172.16.92.0/24 | 172.16.93.0/24 |
| node-1 – node-3 | 172.16.94.0/24 | 172.16.95.0/24 |

node-0's addresses, with its port 0 cabled to node-1 and its port 1 to node-2; the `enp1s0f*` devices are half 0 and the
`enP2p1s0f*` devices half 1, and each node takes its own last number:

```sh
sudo tee /etc/netplan/01-cx7.yaml >/dev/null <<'EOF'
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    enp1s0f0np0:
      mtu: 9000
      addresses: [172.16.88.10/24]
    enP2p1s0f0np0:
      mtu: 9000
      addresses: [172.16.89.10/24]
    enp1s0f1np1:
      mtu: 9000
      addresses: [172.16.92.10/24]
    enP2p1s0f1np1:
      mtu: 9000
      addresses: [172.16.93.10/24]
EOF
sudo chmod 600 /etc/netplan/01-cx7.yaml
sudo netplan apply
ip -br addr    # all four halves UP, with their addresses
rdma link      # all four halves ACTIVE
```

- Check every link in both directions with `ib_write_bw` after cabling: here one stream reached 98 Gb/s a half, and all
  four links in both directions at once 48 to 49 Gb/s a half, about 194 Gb/s sent and 194 Gb/s received by each node.
  A port re-plugged while the node ran stayed at 5 to 13 Gb/s until the node restarted.
- The rendezvous and ssh go over the management network (`SOCKET_IFNAME`), which needs no link between the opposite
  nodes. `QUICKSTART.md`, section 1, has the ssh steps; on a ring they use the management addresses.
- `docker run ... IMAGE up` and the control panel's node check follow the cables with LLDP, so every node needs `lldpd`. DGX OS's
  updates bring it. `lldpctl` shows a node's neighbours, and `sudo apt install lldpd` adds it when it is missing.

## The fenced NCCL

On Arm, a proxy thread of NCCL can read a stale count from the clear-to-send slot, because nothing orders the two reads.
It then spins forever with no error, or posts too few writes. The fix is one acquire fence (NVIDIA/nccl#2393, commit
`05d026f4`). It is on NCCL's development branch and in no release up to 2.32.3, so NVIDIA's wheel does not have it. NCCL's
developers think it also fixes a silent hang reported on four GB10 on a switch (NVIDIA/nccl#2334); that report has not
been confirmed fixed.

The release's image has this build and reads it on a switch and on a ring, so with the image there is nothing to do. With
`RUNTIME=host`, a ring requires it, and on a switch, where the wheel runs too, it is recommended. To build it (about two
minutes on a node with CUDA 13):

```sh
git clone https://github.com/NVIDIA/nccl && cd nccl
git checkout v2.30.7-1
git cherry-pick 05d026f4
make -j8 src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
sha256sum build/lib/libnccl.so.2.30.7
```

Copy `build/lib/libnccl.so*` to the same path on every node. The build used here has the sha256
`f75ac06821bbf42793a4951d6fc89061526a3fa1ad37b565c4146b0995a53299`, and `docker/Dockerfile` accepts only that file. A build of the
image needs it, with `--build-context fenced=DIR`: `DIR` holds `build/lib/libnccl.so.2.30.7` and the source's `LICENSE.txt`
and `ThirdPartyNotices.txt`, and the build fails if one is missing. With `RUNTIME=host`, set `NCCL_LIB` to your own build, and on a ring
`NCCL_SHA256`; its sha256 can differ with the toolchain.

## Settings

`docker run ... IMAGE up` and the control panel's node check find a ring by themselves and write `TOPOLOGY`, `NODES`,
`SOCKET_IFNAME` and `IB_HCA` to `glm53.nodes.conf`. By hand, beside the four-node settings of `glm53.t4.conf.example`:

```sh
TOPOLOGY=ring4
NODES=(172.16.10.200 172.16.10.201 172.16.10.203 172.16.10.202)   # management addresses, in ring order
SOCKET_IFNAME=enP7s7                                                  # the management network
IB_HCA=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1                # all four RDMA devices
RING_CHANNELS=4
NCCL_LIB=$HOME/nccl-fenced/libnccl.so.2.30.7                          # the fenced NCCL
NCCL_SHA256=<its sha256>
```

When `docker run ... IMAGE up` cannot find the ring, write the four lines `TOPOLOGY`, `NODES`, `SOCKET_IFNAME` and `IB_HCA` in
`~/expertatlas/glm53.nodes.conf` (its first run makes the directory) and run it again. The image holds the fenced NCCL, so
`NCCL_LIB` and `NCCL_SHA256` are not needed.

`./tf-glm53 check` confirms the ring: the nodes in ring order, two subnets a link, no subnet shared with the opposite
node, every device with an IPv4 RoCE v2 address. It refuses settings that would reach the opposite node directly.

`TOPOLOGY=auto` in place of `ring4` judges the cabling at each `check` and start (`./tf-glm53 topology`, or `--json`,
shows the judgment): four nodes cabled as a ring take `ring4`, NODES put in ring order (rank 0 kept), the four RDMA
devices in `IB_HCA` and the interface of rank 0's address in `SOCKET_IFNAME`, so those two may be left out. The same
settings on a switch take `full`. A cabling that is neither (a cable out, a lane down, a node's two cables swapped,
an opposite pair on one /24) is refused with what to fix; nothing falls back.

## Speed

Measured when the ring was brought up, on release r3-102's engine. The four ring nodes, routed experts half at 3 and
half at 4 bits (the panel's plan, 9,600 each), the settings of `T4.md` with the MTP drafter and the 4-bit KV cache. A coding prompt with no cached prefix, one request at a time
(reasoning effort low, up to 4,000 reply tokens), one run after a warm-up; GPU clocks not capped (2,080 to 2,405 MHz
seen). The four-node form on a switch, from `T4.md` (GPU clock capped at 2.1 GHz, median of two):

Speeds differ with the environment (the checkpoint, the prompt, the clocks): take these numbers as a guide.

| Prompt | Ring: time to first token (s) | Ring: decode (tokens/s) | Switch: time to first token (s) | Switch: decode (tokens/s) |
|---|---|---|---|---|
| 129,441 tokens (128K) | 125.4 | 35.9 | 120.1 | 35.3 |
| 260,230 tokens (256K) | 264.8 | 34.6 | 253.3 | 33.2 |
| 395,322 tokens (384K) | 421.9 | 33.0 | - | - |

- The ring's first token comes 4 to 5 % later; decode is as fast or slightly faster.
- That start sized the context length at 573,440 positions: the panel had no measured start on the ring and kept a
  margin. Each node kept 6 to 7 GiB available during the 384K prompt.

## Accuracy

KL divergence from the official BF16 model, over the four 8,192-token windows of the reference, by position range
(a: 1,024 prompt rows, then decode windows; b: 6,144 prompt rows, then decode windows), in the deterministic
configuration with the 8-bit KV cache. On the ring the prompt's partial sums are reduced over the ring; the switch's
runs used reduce-scatter.

| Positions | Ring: half 3 / half 4 bits | Ring: all 3 bits | Switch: half 3 / half 4 bits | Switch: all 3 bits | Switch: all 4 bits |
|---|---|---|---|---|---|
| a 1,024 to 2,047 | 0.0154 | 0.0206 | 0.0156 | 0.0204 | 0.0120 |
| a 2,048 to 4,095 | 0.0236 | 0.0347 | 0.0258 | 0.0346 | 0.0208 |
| a 4,096 to 6,143 | 0.0250 | 0.0324 | 0.0259 | 0.0321 | 0.0198 |
| a 6,144 to 8,190 | 0.0166 | 0.0211 | 0.0160 | 0.0209 | 0.0132 |
| b 6,144 to 8,190 | 0.0158 | 0.0199 | 0.0171 | 0.0203 | 0.0127 |

- With the same all-3-bit weights the ring and the switch differ by 0.0001 to 0.0004 in every range, under 2 %
  (top-1 agreement 93.8 to 95.4 % on the ring, 93.6 to 95.5 % on the switch): the relays and the ring's reduce of
  the prompt's partial sums leave the accuracy as it was.
- The two half-and-half columns are not the same experts at 3 bits: the switch's is the four-node checkpoint's mix
  (`T4.md`), the ring's the panel's plan from its usage counts.

## Not tested yet

- Failures: a node, a cable or a relay stopped while the others run.
- Long runs (hours) under load.
- A start with `TOPOLOGY=auto` (its judgment and `check` pass on the ring and on a switch).
