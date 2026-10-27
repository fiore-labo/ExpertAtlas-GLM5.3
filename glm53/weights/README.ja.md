[English](README.md) | 日本語 | [简体中文](README.zh-CN.md)

# 重みを自分で作る

このレシピが読み込む重みのファイルは、どれも Z.AI の公式の GLM-5.3 チェックポイントから、このディレクトリの道具で作れます。
8 台用と 4 台用のチェックポイント、Expert Atlas の 2・3・4・5 bit の expert バンクのすべてです。Fiore-lab が Hugging Face に
公開したファイルは一つも要りません。

なぜ大事か: 重みはコードのようには読めません。ある場面だけ違うふるまいをするように手を加えた重みも、見た目はほかの重みと
変わりません。Fiore-lab は重みを注意して作っていますが、それをファイルから確かめる方法はなく、作り方を確かめられない重みには
頼りたくないこともあるでしょう。自分で作れば、頼る重みは Z.AI 自身のものだけになります。このページはその手順です。

## 頼るもの、要らないもの

| | |
|---|---|
| **Z.AI の公式チェックポイント** | [zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16) の版 `9d2398f478cab2de883137db3a36ad2c96205e24`。Fiore-lab の重みもこの版から作りました。`check_official.py` が、ダウンロードしたものを Hugging Face がその版について示す一覧 (全ファイルの大きさとハッシュ) と照合します。 |
| **量子化の道具** | [ExLlamaV3](https://github.com/turboderp-org/exllamav3) 1.6.1 (MIT ライセンス) と PyTorch。`convert.sh setup` がそれぞれの公式の配布物から入れます。 |
| **ここの道具** | 短い Python とシェルのスクリプトで、このリポジトリのほかの部分と同じ Apache-2.0 です。動かす前に読んでください。何をするかは最後の表にあります。 |
| **要らないもの** | Fiore-lab の重みのファイルすべて。4 台用のチェックポイントにだけ、Fiore-lab の小さなデータ (`q3-selection.json`: expert の番号の一覧、手順 6) を使うこともできますが、自分で作ることもできます。 |

ファイルそのものがコードを運ぶことはありません。safetensors のファイルは JSON のヘッダーと数値の並びだけで、このレシピのエンジンは
それを自前のコードで読みます (Zig、実行時に Python なし、pickle なし、モデルのリポジトリからコードを取り込まない)。チャット
テンプレートとトークナイザーは公式のファイルをそのまま写します。重みのファイルにある危険は数値を変えられることで、自分で作れば
その心配がなくなります。

## 作るもの

| もの | ファイル | 大きさ | 使い道 |
|---|---|---|---|
| 4 bit のバンク | `experts-layer-003` … `077.safetensors` | 368.5 GB | 8 台用チェックポイントの routed expert。Expert Atlas の 4 bit |
| 8 台用チェックポイント | 4 bit バンクのファイル、`nonexperts-5bit.safetensors`、`unquantized-bf16.safetensors`、`mtp-layer-78-*` の 3 ファイル、公式の小さなファイル、`config.json`、インデックス、`SHA256SUMS` | 389.9 GB | 8 台での `./tf-glm53 up`。Expert Atlas で起動するときの expert 以外の部分 |
| 3 bit のバンク | `experts-layer-*.safetensors` | 277.9 GB | Expert Atlas の 3 bit。4 台用チェックポイント |
| 2 bit / 5 bit のバンク | `experts-layer-*.safetensors` | 187.3 / 459.1 GB | Expert Atlas の 2 bit と 5 bit (任意) |
| 4 台用チェックポイント | 8 台用チェックポイントの routed expert の半分を 3 bit にしたもの | 344.6 GB | 4 台での `./tf-glm53 up` |

要るものは動かし方で決まります。コントロールパネルを使わず 8 台で動かすなら 8 台用チェックポイントだけ。使わずに 4 台なら 4 台用
チェックポイント (8 台用と 3 bit バンクから作ります)。コントロールパネルを使うなら、チェックポイントと使いたい幅のバンク (4 台なら
3 bit と 4 bit。8 台用チェックポイントのディレクトリを 4 bit バンクとして使えます) です。

## Fiore-lab のファイルとの違い

- **形は同じです。** 同じテンソルが同じ名前・形・ビット幅・符号帳で入り、どの expert も同じ 8 つのテンソル並列の分け前に切られ、
  ファイルの中の並びも同じです。エンジンは Fiore-lab のものとまったく同じように読み込み、`tf-glm53 verify` と `verify-banks` は
  あなた自身の `SHA256SUMS` とマニフェストで照合します。
- **バイト列は同じになりません。** 量子化は GPU の上で計算するもので、別の GPU や別の版のライブラリで変換し直して同じバイト列になるか
  は、Fiore-lab は試していません。2 bit と 5 bit のバンクは作り方からして違います。Fiore-lab はこの 2 つを ExLlamaV3 の標準の
  校正データに日本語の文章とエージェント・コーディングの会話を足して作りましたが、このページはどの幅も ExLlamaV3 の標準の
  校正データだけを使います (Fiore-lab の 3 bit・4 bit バンクと両チェックポイントと同じ)。ファイルに付く説明文も違います
  (Fiore-lab のものは「Fiore-lab:」で始まります)。
- **精度。** README の KL ダイバージェンスの値は Fiore-lab のファイルで測ったものです。自分で作ったものも同じ作り方なので近い
  はずですが、測ってはいません。

### 確かめてあること

組み立ての道具 (`assemble.py`、`bank_manifest.py`) を、Fiore-lab が保管していた変換直後のファイルに対して動かし、できたものを
公開中のファイルと一つずつ比べました。2・3・5 bit のバンク (層ごと)、expert 以外の線形層、MTP 層のファイル、量子化しないテンソル、
インデックスと `config.json`、4 台用の混ぜ合わせ、バンクのマニフェストです。テンソルとそのバイト列はすべて同じでした (違うのは
上に書いたファイルの説明文だけです)。

変換の手順 (`convert.sh` と 2 つのパッチ) は、Fiore-lab が実際に動かした ExLlamaV3 の命令とパッチで、置き場所を設定にしたものです。
`convert.sh` そのものを最初から最後まで動かしたことはまだありません。パッチが ExLlamaV3 1.6.1 の公開ソースにそのまま当たること、
`convert.sh` が渡す引数がすべてその版にあることは確かめてあります。

## 機材と時間

### レンタルの GPU (Fiore-lab が使ったもの)

Fiore-lab は NVIDIA RTX PRO 6000 (1 枚 96 GB) を借りて、ExLlamaV3 1.6.1 で変換しました。

| 手順 | GPU | 時間 |
|---|---|---|
| `full4` (すべての線形層を 4 bit) | 4 枚 | 5.3 時間 |
| `experts 3` (routed expert を 3 bit) | 2 枚 | 8.1 時間 |
| `nonexperts` (それ以外の線形層を 5 bit) | 1 枚 | 2.6 時間 |
| `mtp` | 1 枚 | 数分 (量子化そのものは数秒) |
| `experts 2` / `experts 5` | | この方法では測っていません。1 層の時間はビット幅でほとんど変わらないので、同じ GPU なら 3 bit と同じくらいの見込みです |
| `assemble.py`、`bank_manifest.py` | 不要 | 1 手順あたり数分から 1 時間 (ディスクの速さしだい) |

ディスク: 公式チェックポイントが 1.51 TB。変換の各手順は量子化したテンソルを作業ディレクトリに書き (量子化した分とほぼ同じ大きさ)、
ExLlamaV3 が自前の完成品を `<手順>-out` に書きます。後者はこのレシピでは使わないので、容量が足りなければ手順のあとで消してください。
各手順のあとで `<手順>-out` を消すとして、8 台用・4 台用チェックポイントと 3 bit バンクまでで合計 4 TB ほどを見込んでください。1.51 TB のダウンロードの時間は回線しだいです。

### DGX Spark (まだ試していません)

Fiore-lab は DGX Spark でこの変換を動かしたことがありません。ここに書くのは分かっていることで、試した手順ではありません。

- DGX Spark は aarch64 で CUDA 13、GB10 の GPU が CPU と 128 GB のメモリを共有します。ExLlamaV3 1.6.1 にはこれ用の配布物が
  ありません。GB10 に対応した aarch64 用の CUDA 付き PyTorch を入れ、ExLlamaV3 1.6.1 をソースから入れます
  (`pip install git+https://github.com/turboderp-org/exllamav3@v1.6.1`、CUDA の拡張をビルドします)。1.6.1 がそこでビルドでき、
  動くかは試していません。`convert.sh setup` は x86_64 用の配布物しか入れないので、DGX Spark では venv を自分で作り、ほかの手順は
  `VENV` でそれを指して動かします。
- GB10 1 台の GPU の計算力は上の GPU よりずっと小さいので、各手順は何倍も長くかかる見込みです (測っていません)。`full4` のあとは
  幅ごとに独立しているので、台数があれば 1 台に 1 つの幅を割り振れます (どの台にも公式チェックポイントと `full4` の作業ディレクトリ
  の写しが要ります)。
- メモリに気をつけてください。DGX Spark はメモリが尽きると、電源を入れ直すまで応答しなくなることがあります。`assemble.py` は空きが
  20 GiB を切ると自分で止まります。

## 手順

下のコマンドは置き場所の例として `/data` を使います。上の容量のあるディレクトリなら、どこでも構いません。

### 1. 道具と公式チェックポイントを用意する

```sh
git clone https://github.com/fiore-labo/ExpertAtlas-GLM5.3 && cd ExpertAtlas-GLM5.3/glm53/weights
pip install -U "huggingface_hub[hf_xet]"
hf download zai-org/GLM-5.3-BF16 --revision 9d2398f478cab2de883137db3a36ad2c96205e24 --local-dir /data/GLM-5.3-BF16
python3 check_official.py /data/GLM-5.3-BF16          # 最後が "OK: ... 0 problems" になること
```

`check_official.py` は、動かした時点でその版のファイル一覧を huggingface.co から読み、全ファイルを照合します (重みのファイルは
sha256、小さなファイルは Git の blob ハッシュ)。1.51 TB を読むので時間がかかります。

### 2. 量子化の道具を入れる

```sh
./convert.sh setup                                     # ./venv に PyTorch 2.8 (CUDA 12.8) と ExLlamaV3 1.6.1
```

### 3. 変換する

```sh
export BF16=/data/GLM-5.3-BF16 WORK=/data/work
./convert.sh full4            # 最初に: 4 bit の expert と、ほかの手順の校正の通り道
./convert.sh nonexperts       # それ以外の線形層を 5 bit
./convert.sh mtp              # MTP 層の線形層
./convert.sh experts 3        # 3 bit の expert (4 台用チェックポイントと Expert Atlas 用)
./convert.sh experts 2        # 任意
./convert.sh experts 5        # 任意
```

各手順は `$WORK/<手順>.log` に記録し、前面で動きます (長いものは `tmux` か `nohup` で)。途中で止まっても、もう一度動かせば
最後のチェックポイントから続けます。`GPUS=0,1` で使う GPU を絞れます。

### 4. バンクを組み立てる

```sh
python3 -m venv /data/tools && /data/tools/bin/pip install numpy xxhash   # または ./venv を使う
P=/data/tools/bin/python
$P assemble.py bank $WORK/full4/qtensors    /data/banks/4bit
$P assemble.py bank $WORK/experts3/qtensors /data/banks/3bit
$P assemble.py bank $WORK/experts2/qtensors /data/banks/2bit     # 作った場合
$P assemble.py bank $WORK/experts5/qtensors /data/banks/5bit     # 作った場合
```

各層の expert を、エンジンが読み込む 8 つのテンソル並列の分け前に切ります。量子化し直すことはありません。

### 5. 8 台用チェックポイントを組み立てる

```sh
$P assemble.py checkpoint --official $BF16 --experts /data/banks/4bit --mtp-qtensors $WORK/full4/qtensors \
   --nonexperts $WORK/nonexperts/qtensors --mtp-linears $WORK/mtp-linears-5bit.safetensors --out /data/GLM-5.3-8x
```

expert のファイルは 4 bit バンクのファイルへのハードリンクです (写すなら `--copy`。2 つのディレクトリが別のファイルシステムに
あるときも)。量子化しないテンソル (埋め込み、`lm_head`、norm、ルーター、`kv_b_proj`、indexer の一部、MTP 層の `eh_proj`) は
公式のファイルからそのまま写します。最後にインデックスとファイルの照合をして、`SHA256SUMS` を書きます。

### 6. 4 台用チェックポイントを組み立てる (4 台で動かす場合だけ)

4 台用チェックポイントは routed expert の半分を 3 bit で持ちます。どの半分かは `q3-selection.json` という一覧
(層 3〜77 について `{"層": [expert, ...]}`) で決まります。Fiore-lab の一覧を使う (入っているのは expert の番号だけで、日本語と
英語のエージェント作業・コーディング・Linux の操作・サーバーのセキュリティで使われ方の少ない半分です。expert の計算を変える
ことはできません) か、自分で書きます。

```sh
hf download fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks q3-selection.json --local-dir /data
$P assemble.py mix --base /data/GLM-5.3-8x --bank3 /data/banks/3bit --selection /data/q3-selection.json --out /data/GLM-5.3-4x
```

### 7. バンクのマニフェストとバンクの一覧

エンジンは、マニフェストの sha256 が `expert-plan/banks.json` に書かれたものと同じバンクしか読み込みません。バンクごとに
マニフェストを作り、それらの一覧を作ります。

```sh
CAL="exllamav3 1.6.1 default calibration data (250 x 2,048 tokens)"
$P bank_manifest.py make /data/banks/4bit --id own-experts-4bit --bits 4 --calibration "$CAL"
$P bank_manifest.py make /data/banks/3bit --id own-experts-3bit --bits 3 --calibration "$CAL"
$P bank_manifest.py banks banks.json /data/banks/3bit /data/banks/4bit      # 作ったなら 2 bit と 5 bit も
```

マニフェストはそれぞれのバンクのディレクトリに `<id>.json` として書かれます。できた `banks.json` を、リリースの
`expert-plan/banks.json` と置き換えてください。バンクのディレクトリはどのノードでも同じパスに置きます。ほかのバンク
(Fiore-lab のもの) で保存した割り当ては、自分のバンクでは開けません (パネルがバンクの違いを示します)。新しく作り直してください。

### 8. 使う

チェックポイントをどのノードにも同じパスで写して `glm53.conf` の `WEIGHTS` をそこに向け、バンクを `banks.json` のディレクトリに
写してから:

```sh
./tf-glm53 deploy
./tf-glm53 verify            # 全ノードで重みをあなたの SHA256SUMS と照合
./tf-glm53 verify-banks      # 全ノードでバンクをあなたのマニフェストと照合
./tf-glm53 up
```

## このディレクトリのファイル

| ファイル | すること |
|---|---|
| `check_official.py` | zai-org/GLM-5.3-BF16 のダウンロードを、Hugging Face がその版について示すファイル一覧と照合する |
| `convert.sh` | ExLlamaV3 1.6.1 を入れ、変換を動かす (`full4`、`experts BITS`、`nonexperts`、`mtp`) |
| `patch_experts.py` | ExLlamaV3 の変換を routed expert だけを量子化するように変え、ほかの線形層は `full4` から取る |
| `patch_nonexperts.py` | それ以外の線形層だけを量子化するように変え、routed expert は `full4` から取る |
| `convert_mtp_linears.py` | MTP 層の線形層を校正データなしで 5 bit (mul1) にする |
| `assemble.py` | バンク、8 台用と 4 台用のチェックポイントを組み立てる。`check` でチェックポイントのディレクトリを確かめる |
| `bank_manifest.py` | バンクのマニフェスト (`make`、`check`) とバンクの一覧 (`banks`) |

自分で作った重みは GLM-5.3 から派生したもので、GLM-5.3 License に従います (チェックポイントに入る公式の `LICENSE`)。
