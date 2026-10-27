[English](README.md) | [日本語](README.ja.md) | 简体中文

# 自己制作权重

本方案加载的每一个权重文件，都可以用本目录中的工具，从 Z.AI 官方的 GLM-5.3 检查点自行制作：八节点和四节点检查点，以及
Expert Atlas 的 2、3、4、5 位专家库。你不需要 Fiore-lab 在 Hugging Face 上发布的任何文件。

为什么这很重要：权重不能像代码那样阅读。被改动过、在某些情况下表现不同的权重，看起来和其他权重没有区别。Fiore-lab 制作权重时
很谨慎，但你无法从文件本身确认这一点，你也可能不愿依赖无法核实其制作过程的权重。自己制作的话，你唯一依赖的权重就是 Z.AI 自己的。
本页给出步骤。

## 你依赖什么，不需要什么

| | |
|---|---|
| **Z.AI 官方检查点** | [zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16) 的版本 `9d2398f478cab2de883137db3a36ad2c96205e24`，Fiore-lab 的权重也由这个版本制作。`check_official.py` 会把你下载的文件与 Hugging Face 为该版本给出的清单（每个文件的大小和哈希）逐一核对。 |
| **量化工具** | [ExLlamaV3](https://github.com/turboderp-org/exllamav3) 1.6.1（MIT 许可证）和 PyTorch，由 `convert.sh setup` 从各自的正式发布安装。 |
| **本目录的工具** | 简短的 Python 和 shell 脚本，与本仓库其他部分一样采用 Apache-2.0。运行前请先阅读；每个文件的作用见文末的表。 |
| **不需要** | Fiore-lab 的任何权重文件。四节点检查点可以选用 Fiore-lab 的一个小数据文件（`q3-selection.json`：一串专家编号，见第 6 步），也可以自己编写。 |

文件本身不能携带代码：safetensors 文件只有一个 JSON 头和原始数值，本方案的引擎用自己的代码读取它们（Zig，运行时没有 Python，
没有 pickle，也不执行模型仓库中的代码）。聊天模板和分词器是官方文件，原样复制。权重文件的风险在于数值被改动，自己制作就消除了
这一依赖。

## 你会得到什么

| 内容 | 文件 | 大小 | 用途 |
|---|---|---|---|
| 4 位专家库 | `experts-layer-003` … `077.safetensors` | 368.5 GB | 八节点检查点的路由专家；Expert Atlas 的 4 位 |
| 八节点检查点 | 4 位专家库的文件、`nonexperts-5bit.safetensors`、`unquantized-bf16.safetensors`、三个 `mtp-layer-78-*` 文件、官方小文件、`config.json`、索引、`SHA256SUMS` | 389.9 GB | 八节点上的 `./tf-glm53 up`；每次用 Expert Atlas 启动时专家以外的部分 |
| 3 位专家库 | `experts-layer-*.safetensors` | 277.9 GB | Expert Atlas 的 3 位；四节点检查点 |
| 2 位 / 5 位专家库 | `experts-layer-*.safetensors` | 187.3 / 459.1 GB | Expert Atlas 的 2 位和 5 位（可选） |
| 四节点检查点 | 把一半路由专家换成 3 位的八节点检查点 | 344.6 GB | 四节点上的 `./tf-glm53 up` |

需要哪些取决于运行方式：不用控制面板、八节点运行只需要八节点检查点；不用控制面板、四节点运行需要四节点检查点（由八节点检查点和
3 位专家库制作）；使用控制面板需要一个检查点和你想用的位宽的专家库（四节点上为 3 位和 4 位；八节点检查点的目录可以直接作为 4 位
专家库）。

## 与 Fiore-lab 文件的比较

- **布局相同。** 相同的张量，名称、形状、位宽和码本都相同，每个专家切成同样的八个张量并行分片，在文件中的排列也相同。引擎读取它们
  的方式与读取 Fiore-lab 的文件完全一样，`tf-glm53 verify` 和 `verify-banks` 用你自己的 `SHA256SUMS` 和清单文件核对。
- **字节不相同。** 量化在 GPU 上计算，换用其他 GPU 或其他版本的库重新转换能否得到相同的字节，Fiore-lab 没有测试过。2 位和 5 位
  专家库在做法上也不同：Fiore-lab 用 ExLlamaV3 默认校准数据加上日语文本和智能体 / 编程会话来校准这两个，而本页对所有位宽都只用
  ExLlamaV3 默认校准数据（与 Fiore-lab 的 3 位、4 位专家库和两个检查点相同）。文件中的元数据文字也不同（Fiore-lab 的以
  "Fiore-lab:" 开头）。
- **精度。** README 中的 KL 散度是用 Fiore-lab 的文件测得的。你的文件做法相同，应当接近，但没有测量过。

### 已经核对的内容

组装工具（`assemble.py`、`bank_manifest.py`）在 Fiore-lab 保存的转换输出上运行，每个结果都与已发布的文件逐一比较：2、3、5 位
专家库（逐层）、非专家线性层、MTP 层的文件、保持不量化的张量、索引和 `config.json`、四节点混合，以及专家库清单。张量及其字节完全
相同（文件只在上述元数据文字上不同）。

转换步骤（`convert.sh` 和两个补丁）是 Fiore-lab 实际运行过的 ExLlamaV3 命令和补丁，只是把路径改成了设置项。Fiore-lab 还没有从头到尾
运行过 `convert.sh` 本身。已确认两个补丁能直接应用到 ExLlamaV3 1.6.1 的发布源码上，`convert.sh` 传递的每个选项在该版本中都存在。

## 硬件与时间

### 租用的 GPU（Fiore-lab 使用的方式）

Fiore-lab 在租用的 NVIDIA RTX PRO 6000（每块 96 GB）上用 ExLlamaV3 1.6.1 转换：

| 步骤 | GPU | 时间 |
|---|---|---|
| `full4`（所有线性层 4 位） | 4 块 | 5.3 小时 |
| `experts 3`（路由专家 3 位） | 2 块 | 8.1 小时 |
| `nonexperts`（其他线性层 5 位） | 1 块 | 2.6 小时 |
| `mtp` | 1 块 | 几分钟（量化本身只需几秒） |
| `experts 2` / `experts 5` | | 未用这种方法测量；每层的时间几乎不随位宽变化，同样的 GPU 预计与 3 位相近 |
| `assemble.py`、`bank_manifest.py` | 不需要 | 每步几分钟到一小时，取决于磁盘 |

磁盘：官方检查点占 1.51 TB。每个转换步骤把量化后的张量写入它的工作目录（大小约等于它量化的部分），ExLlamaV3 还会把它自己的完整
模型写到 `<步骤>-out`，本方案不使用它：空间不足时可在该步骤完成后删除。若每个步骤完成后删除 `<步骤>-out`，八节点、四节点检查点加 3 位专家库总共请准备约 4 TB。下载 1.51 TB
所需的时间取决于你的网络。

### DGX Spark（尚未尝试）

Fiore-lab 没有在 DGX Spark 上运行过这些转换，因此这里是已知的情况，而不是经过测试的步骤：

- DGX Spark 是 aarch64、CUDA 13，GB10 GPU 与 CPU 共享 128 GB 内存。ExLlamaV3 1.6.1 没有为它预编译的安装包：先安装支持 GB10
  的 aarch64 版带 CUDA 的 PyTorch，再从源码安装 ExLlamaV3 1.6.1（`pip install git+https://github.com/turboderp-org/exllamav3@v1.6.1`，
  会编译其 CUDA 扩展）。1.6.1 能否在那里编译和运行尚未测试。`convert.sh setup` 只安装 x86_64 的安装包；在 DGX Spark 上请自己
  创建 venv，并用 `VENV` 指向它来运行其他步骤。
- 一台 GB10 的 GPU 算力远小于上面的 GPU，每个步骤预计要长很多倍（未测量）。`full4` 之后各位宽相互独立：有多台节点时，每台运行
  一个位宽（每台都需要一份官方检查点和 `full4` 工作目录的副本）。
- 注意内存。DGX Spark 内存耗尽时，可能会在断电重启之前一直没有响应。可用内存低于 20 GiB 时，`assemble.py` 会自行停止。

## 步骤

下面的命令以 `/data` 为例；任何有上述空间的目录都可以。

### 1. 获取工具和官方检查点

```sh
git clone https://github.com/fiore-labo/ExpertAtlas-GLM5.3 && cd ExpertAtlas-GLM5.3/glm53/weights
pip install -U "huggingface_hub[hf_xet]"
hf download zai-org/GLM-5.3-BF16 --revision 9d2398f478cab2de883137db3a36ad2c96205e24 --local-dir /data/GLM-5.3-BF16
python3 check_official.py /data/GLM-5.3-BF16          # 最后一行必须是 "OK: ... 0 problems"
```

`check_official.py` 在运行时从 huggingface.co 读取该版本的文件清单，并核对每个文件（权重文件用 sha256，小文件用 Git blob 哈希）。
它要读取 1.51 TB，需要一些时间。

### 2. 安装量化工具

```sh
./convert.sh setup                                     # 在 ./venv 中安装 PyTorch 2.8（CUDA 12.8）和 ExLlamaV3 1.6.1
```

### 3. 转换

```sh
export BF16=/data/GLM-5.3-BF16 WORK=/data/work
./convert.sh full4            # 最先运行：4 位专家，也是后续步骤校准所经过的模型
./convert.sh nonexperts       # 其他线性层 5 位
./convert.sh mtp              # MTP 层的线性层
./convert.sh experts 3        # 3 位专家（用于四节点检查点和 Expert Atlas）
./convert.sh experts 2        # 可选
./convert.sh experts 5        # 可选
```

每个步骤记录到 `$WORK/<步骤>.log`，在前台运行（较长的步骤请用 `tmux` 或 `nohup`），中断后再次运行会从最后的检查点继续。
`GPUS=0,1` 可限制所用的 GPU。

### 4. 组装专家库

```sh
python3 -m venv /data/tools && /data/tools/bin/pip install numpy xxhash   # 或使用 ./venv
P=/data/tools/bin/python
$P assemble.py bank $WORK/full4/qtensors    /data/banks/4bit
$P assemble.py bank $WORK/experts3/qtensors /data/banks/3bit
$P assemble.py bank $WORK/experts2/qtensors /data/banks/2bit     # 如果制作了
$P assemble.py bank $WORK/experts5/qtensors /data/banks/5bit     # 如果制作了
```

每层的专家被切成引擎读取的八个张量并行分片；不会重新量化。

### 5. 组装八节点检查点

```sh
$P assemble.py checkpoint --official $BF16 --experts /data/banks/4bit --mtp-qtensors $WORK/full4/qtensors \
   --nonexperts $WORK/nonexperts/qtensors --mtp-linears $WORK/mtp-linears-5bit.safetensors --out /data/GLM-5.3-8x
```

专家文件是指向 4 位专家库文件的硬链接（加 `--copy` 改为复制；两个目录在不同文件系统上时也需要）。保持不量化的张量（嵌入、
`lm_head`、norm、路由器、`kv_b_proj`、indexer 的一部分、MTP 层的 `eh_proj`）按原样从官方文件复制。命令最后会核对索引与文件，并写出
`SHA256SUMS`。

### 6. 组装四节点检查点（仅四节点运行时需要）

四节点检查点把一半的路由专家存为 3 位。哪一半由一个清单 `q3-selection.json` 决定（对第 3 到 77 层为 `{"层号": [专家, ...]}`）。
可以使用 Fiore-lab 的清单（只包含专家编号：在日语和英语的智能体工作、编程、Linux 操作和服务器安全中使用最少的一半；它不能改变
专家的计算），也可以自己编写：

```sh
hf download fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks q3-selection.json --local-dir /data
$P assemble.py mix --base /data/GLM-5.3-8x --bank3 /data/banks/3bit --selection /data/q3-selection.json --out /data/GLM-5.3-4x
```

### 7. 专家库清单与专家库列表

只有当专家库清单文件的 sha256 与 `expert-plan/banks.json` 中给出的一致时，引擎才会加载该专家库。为每个专家库制作清单，并为它们
制作列表：

```sh
CAL="exllamav3 1.6.1 default calibration data (250 x 2,048 tokens)"
$P bank_manifest.py make /data/banks/4bit --id own-experts-4bit --bits 4 --calibration "$CAL"
$P bank_manifest.py make /data/banks/3bit --id own-experts-3bit --bits 3 --calibration "$CAL"
$P bank_manifest.py banks banks.json /data/banks/3bit /data/banks/4bit      # 若制作了 2 位和 5 位也一并列出
```

每个清单以 `<id>.json` 写入各自专家库的目录。用生成的 `banks.json` 替换发布包中的 `expert-plan/banks.json`；专家库目录在每个节点
上必须是同一路径。用其他专家库（Fiore-lab 的）保存的分配无法用你的专家库打开（面板会提示专家库不同）；请重新制作。

### 8. 使用

把检查点复制到每个节点的同一路径，并把 `glm53.conf` 的 `WEIGHTS` 指向它，把专家库复制到 `banks.json` 中的目录，然后：

```sh
./tf-glm53 deploy
./tf-glm53 verify            # 在所有节点上用你的 SHA256SUMS 核对权重
./tf-glm53 verify-banks      # 在所有节点上用你的清单核对专家库
./tf-glm53 up
```

## 本目录中的文件

| 文件 | 作用 |
|---|---|
| `check_official.py` | 把 zai-org/GLM-5.3-BF16 的下载与 Hugging Face 为该版本给出的文件清单核对 |
| `convert.sh` | 安装 ExLlamaV3 1.6.1 并运行转换（`full4`、`experts BITS`、`nonexperts`、`mtp`） |
| `patch_experts.py` | 修改 ExLlamaV3 的转换器，只量化路由专家，其他线性层取自 `full4` |
| `patch_nonexperts.py` | 修改它只量化其他线性层，路由专家取自 `full4` |
| `convert_mtp_linears.py` | 不使用校准数据，把 MTP 层的线性层量化为 5 位（mul1） |
| `assemble.py` | 组装专家库、八节点和四节点检查点，`check` 检查一个检查点目录 |
| `bank_manifest.py` | 专家库清单（`make`、`check`）和专家库列表（`banks`） |

你制作的权重派生自 GLM-5.3，适用 GLM-5.3 License（检查点中附带的官方 `LICENSE`）。
