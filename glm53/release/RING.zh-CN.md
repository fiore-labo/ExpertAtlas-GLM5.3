[English](RING.md) | [日本語](RING.ja.md) | 简体中文

# 不用交换机，把四台 DGX Spark 连成环

四节点配置（`T4.zh-CN.md`）也能在四台 NVIDIA DGX Spark 系统上运行：它们不用交换机，而是以 200 Gb/s 直连链路互相连成环。引擎、检查点、bank 和控制面板都相同，变化的只是 rank 之间的通信方式。它是 Fiore-lab 基于 TensorFold 1.0.2 制作的非官方衍生版本，并非 TensorFold 项目的发布版本。

## 运行方式

- 每个节点直接连到两个相邻节点。对面的节点（隔两条链路）没有自己的链路，发往它的数据由相邻节点中继：
  - 提示词的 all-to-all 把发往对面节点的块分成两半，分别经两个相邻节点发送；
  - RoCE 上的解码集合通信（`TF_GLM53_ROCE=1`）把对面节点的那部分数据，经由在同一槽位持有它的相邻节点转发，因此 GPU 内核不变。
- NCCL 只连接环（`NCCL_ALGO=Ring`），从不连接对面节点，使用每个节点的四个 PCIe 半边（4 个通道）。
- rank 是节点在环上的位置；每个解码上下文并行组（DCP 2）都是一对直接用线缆相连的节点。
- 提示词的部分和通过环归约（`PREFILL_REDUCE=ring`；reduce-scatter 会被拒绝）。环上不使用按份额进行的多对话归约（AR2）；但 `MULTI` 在环上照常运行，最多 4 个对话。
- NCCL 使用 2.30.7，带 NVIDIA/nccl#2393 的 AArch64 发送 fence（commit `05d026f4`），为 sm_121 构建。使用 `RUNTIME=docker` 时，发布的镜像把它放在 `/opt/nccl-fenced`，`tf-glm53 check` 会查找它。使用 `RUNTIME=host` 时，把它放在每个节点的相同路径下，启动器检查其 SHA-256。下面的“fenced NCCL”给出构建方法。

## 连线和地址

- 沿着环，把每个节点的端口 0 连到下一个节点的端口 1。如果改为端口 0 连端口 0、端口 1 连端口 1，NCCL 的 subnet-aware routing 在某些链路上找不到设备（监听方只通告自己设备的地址），连接会失败。
- 为每条链路的每个 PCIe 半边分配一个 /24（一个 ConnectX-7 端口分为两个半边，`rocep1s0f*` 和 `roceP2p1s0f*`），MTU 设为 9000。DGX OS 已经设置了 `arp_ignore=1`、`arp_announce=2` 和 `rp_filter=2`，无需手动设置。这里使用的示例（环的顺序为 node-0、1、3、2）：

| 链路 | 半边 0 | 半边 1 |
|---|---|---|
| node-0 – node-1 | 172.16.88.0/24 | 172.16.89.0/24 |
| node-3 – node-2 | 172.16.90.0/24 | 172.16.91.0/24 |
| node-0 – node-2 | 172.16.92.0/24 | 172.16.93.0/24 |
| node-1 – node-3 | 172.16.94.0/24 | 172.16.95.0/24 |

node-0 的地址，其端口 0 连接 node-1，端口 1 连接 node-2；`enp1s0f*` 是半边 0，`enP2p1s0f*` 是半边 1，每个节点使用自己的最后一个数字：

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
ip -br addr    # 四个半边都是 UP，并且有地址
rdma link      # 四个半边都是 ACTIVE
```

- 连线后用 `ib_write_bw` 检查每条链路的两个方向：这里单条流每个半边达到 98 Gb/s；四条链路的两个方向同时运行时，每个半边为 48 至 49 Gb/s，每个节点约发送 194 Gb/s、接收 194 Gb/s。在节点运行时重新插拔的端口一直停留在 5 至 13 Gb/s，直到节点重启。
- rendezvous 和 ssh 走管理网络（`SOCKET_IFNAME`），因此对面的节点之间不需要链路。ssh 的步骤见 `QUICKSTART.zh-CN.md` 第 1 节；在环上使用管理网络的地址。
- `docker run ... IMAGE up` 和控制面板的节点检查通过 LLDP 沿线缆追踪，因此每个节点都需要 `lldpd`。DGX OS 的更新会安装它。`lldpctl` 显示节点的邻居，缺少时用 `sudo apt install lldpd` 安装。

## fenced NCCL

在 Arm 上，由于两次读取的顺序没有保证，NCCL 的代理线程可能从 clear-to-send 槽读到旧的计数，于是在没有任何错误的情况下一直等待，或者写入不足。修复方法是增加一个 acquire fence（NVIDIA/nccl#2393，commit `05d026f4`）。它在 NCCL 的开发分支上，2.32.3 及之前的正式版本都没有，因此 NVIDIA 的 wheel 也没有。NCCL 的开发者认为，在四台 GB10 加交换机上报告的无声挂起（NVIDIA/nccl#2334）也由此修复，但该报告尚未确认已修复。

发布的镜像包含此构建，在交换机和环上都读取它，所以使用镜像时无需任何操作。使用 `RUNTIME=host` 时，环必须使用它；交换机上 wheel 也能运行，但推荐使用它。构建方法（在装有 CUDA 13 的节点上约两分钟）：

```sh
git clone https://github.com/NVIDIA/nccl && cd nccl
git checkout v2.30.7-1
git cherry-pick 05d026f4
make -j8 src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
sha256sum build/lib/libnccl.so.2.30.7
```

把 `build/lib/libnccl.so*` 复制到每个节点的相同路径下。这里使用的构建的 sha256 为 `f75ac06821bbf42793a4951d6fc89061526a3fa1ad37b565c4146b0995a53299`，`docker/Dockerfile` 只接受这个文件。构建镜像需要它（`--build-context fenced=DIR`）：`DIR` 中放 `build/lib/libnccl.so.2.30.7` 以及源代码中的 `LICENSE.txt` 和 `ThirdPartyNotices.txt`，缺少任何一个，构建都会失败。使用 `RUNTIME=host` 时，把 `NCCL_LIB` 设为你自己的构建，在环上还要设置 `NCCL_SHA256`；sha256 可能随工具链而不同。

## 设置

`docker run ... IMAGE up` 和控制面板的节点检查会自行发现环，并把 `TOPOLOGY`、`NODES`、`SOCKET_IFNAME` 和 `IB_HCA` 写入 `glm53.nodes.conf`。手动设置时，在 `glm53.t4.conf.example` 的四节点设置之外加上：

```sh
TOPOLOGY=ring4
NODES=(172.16.10.200 172.16.10.201 172.16.10.203 172.16.10.202)   # 管理地址，按环的顺序
SOCKET_IFNAME=enP7s7                                                  # 管理网络
IB_HCA=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1                # 全部四个 RDMA 设备
RING_CHANNELS=4
NCCL_LIB=$HOME/nccl-fenced/libnccl.so.2.30.7                          # fenced NCCL
NCCL_SHA256=<its sha256>
```

`./tf-glm53 check` 会确认环：节点按环的顺序排列，每条链路两个子网，不与对面节点共享子网，每个设备都有 IPv4 RoCE v2 地址。它拒绝会直接连到对面节点的设置。

用 `TOPOLOGY=auto` 代替 `ring4` 时，每次 `check` 和启动都会判定连线（`./tf-glm53 topology` 或 `--json` 显示判定结果）：连成环的四个节点取 `ring4`，NODES 排成环的顺序（rank 0 不变），`IB_HCA` 使用四个 RDMA 设备，`SOCKET_IFNAME` 使用 rank 0 地址所在的接口，因此这两项可以不写。相同设置接在交换机上时取 `full`。两者都不是的连线（线缆拔出、一个半边断开、一个节点的两根线缆接反、一对对面节点在同一个 /24 上）会被拒绝，并指出要修正的地方，不会自动退回到别的方式。

## 速度

在环刚搭建好时，用发布版本 r3-102 的引擎测量。环上的四个节点，路由专家一半 3 bit、一半 4 bit（面板的计划，各 9,600 个），采用 `T4.zh-CN.md` 的设置、MTP 草稿模型和 4 bit KV 缓存。使用没有已缓存前缀的编程提示词，一次一个请求（reasoning effort 为 low，回复最多 4,000 个 token），预热后运行一次；GPU 时钟不设上限（观测到 2,080 至 2,405 MHz）。交换机上的四节点配置数据来自 `T4.zh-CN.md`（GPU 时钟上限 2.1 GHz，两次运行的中位数）：

速度因环境而异（检查点、提示词、时钟等），这些数字仅供参考。

| 提示词 | 环：首 token 时间（s） | 环：解码（tokens/s） | 交换机：首 token 时间（s） | 交换机：解码（tokens/s） |
|---|---|---|---|---|
| 129,441 token（128K） | 125.4 | 35.9 | 120.1 | 35.3 |
| 260,230 token（256K） | 264.8 | 34.6 | 253.3 | 33.2 |
| 395,322 token（384K） | 421.9 | 33.0 | - | - |

- 环的首 token 晚 4 至 5 %；解码同样快或略快。
- 那次启动把上下文长度定为 573,440 个位置：面板还没有环上的实测启动数据，因此留了余量。在 384K 提示词期间，每个节点保持 6 至 7 GiB 可用内存。

## 精度

相对于官方 BF16 模型的 KL 散度：基于参考数据的四个 8,192 token 窗口，按位置区间（a：1,024 行提示词，之后是解码窗口；b：6,144 行提示词，之后是解码窗口），在确定性配置下使用 8 bit KV 缓存测得。在环上，提示词的部分和通过环归约；交换机上的运行使用 reduce-scatter。

| 位置 | 环：一半 3 bit / 一半 4 bit | 环：全部 3 bit | 交换机：一半 3 bit / 一半 4 bit | 交换机：全部 3 bit | 交换机：全部 4 bit |
|---|---|---|---|---|---|
| a 1,024 至 2,047 | 0.0154 | 0.0206 | 0.0156 | 0.0204 | 0.0120 |
| a 2,048 至 4,095 | 0.0236 | 0.0347 | 0.0258 | 0.0346 | 0.0208 |
| a 4,096 至 6,143 | 0.0250 | 0.0324 | 0.0259 | 0.0321 | 0.0198 |
| a 6,144 至 8,190 | 0.0166 | 0.0211 | 0.0160 | 0.0209 | 0.0132 |
| b 6,144 至 8,190 | 0.0158 | 0.0199 | 0.0171 | 0.0203 | 0.0127 |

- 使用相同的全部 3 bit 权重时，环和交换机在每个区间相差 0.0001 至 0.0004，不到 2 %（top-1 一致率：环上 93.8 至 95.4 %，交换机上 93.6 至 95.5 %）：中继和环上对提示词部分和的归约没有改变精度。
- 两个“一半 / 一半”列中，以 3 bit 载入的专家并不相同：交换机一列是四节点检查点的混合（`T4.zh-CN.md`），环一列是面板根据使用计数制定的计划。

## 尚未测试

- 故障：在其余部分运行时停止一个节点、一根线缆或一个中继。
- 负载下的长时间运行（数小时）。
- 使用 `TOPOLOGY=auto` 的启动（其判定和 `check` 在环和交换机上都能通过）。
