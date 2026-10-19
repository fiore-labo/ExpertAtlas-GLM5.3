[English](RUNTIME.md) | 日本語 | [简体中文](RUNTIME.zh-CN.md)

# このアーカイブでフルサイズの GLM-5.3 を動かす

このアーカイブは、フルサイズの GLM-5.3（`glm_moe_dsa`、755B）を NVIDIA DGX Spark 4 台または 8 台で、GB10 1 基につき 1 ランク、テンソル並列 4 または 8 で動かします。Python、コンパイラ、ソースのチェックアウトは不要です。これは TensorFold 1.0.2 から Fiore-lab が作った非公式の派生物であり、TensorFold プロジェクトのリリースではありません。このページは 4 台と 8 台の両方を扱います。4 台で異なる点とその測定結果は `T4.ja.md` に記載しています。

## アーカイブの中身

- `bin/tensorfold-native`: サーバー本体です。GB10（sm_121）向けにビルドした CUDA カーネルを内蔵しています。Linux aarch64、glibc 2.28 以降。
- `lib/libroce_proxy.so`: decode の集団通信に使う RDMA プロキシです（`libibverbs.so.1` が必要）。
- `tf-glm53`、`glm53.conf.example`（8 台）、`glm53.t4.conf.example`（4 台）: ランクの起動、停止、確認を行います。
- `RUNTIME.md`（このページ）と `T4.md`（4 台）。それぞれ日本語版（`.ja.md`）と簡体字中国語版（`.zh-CN.md`）もあります。
- `LICENSE`、`NOTICE`、`LICENSES/`、`THIRD_PARTY_NOTICES.md`、`VERSION`、`SHA256SUMS`。

モデルの重み、ドラフトモデル、NCCL は含みません。

## 各ノードに必要なもの

- DGX OS（Ubuntu 24.04）と、`libcuda.so.1` を提供する NVIDIA ドライバ 580 以降。
- ローダーのパス上にある cuBLAS 13（`libcublas.so.13`、`libcublasLt.so.13`）。DGX OS では CUDA 13 と一緒にインストールされます（`libcublas-13-0`）。
- 全ノードの同じパスにある NCCL 2.30.7（後述）。
- `libibverbs.so.1` と、各システムをつなぐ 1 つの RoCE v2 ネットワーク（4 台なら、各ノードに 100 Gb/s のポートが 1 つあれば足ります）。1 台目自身を含むすべてのノードに、1 台目から `NODES` に書いたアドレスでパスワードなしの ssh で接続できること。
- チェックポイント用のディスク。全ノードの同じパスに置きます（後述）。8 台では 390 GB（363 GiB）、4 台では 344.6 GB（321 GiB）です。
- 8 台: 起動前の各ノードで、`free -g` が示す利用可能メモリが 112 GiB（DFlash2 では 115）。`CONTEXT` と `STORE` が 819,200 のとき、ランクは約 108 GiB（DFlash2 では 111）を使います。DFlash2 では、プール（既定の会話キャッシュ）は slots（`CACHE=slots`）よりランクあたり約 0.5 GiB 多く使います。`STORE` を 100,000 位置減らすごとに、必要量は 2.7 GiB 減ります。これより少ないノードは `tf-glm53 check` が拒否します。
- 4 台: メモリはこのサーバー専用として割り当てます（`T4.ja.md`）。DFlash2 で測定した空きメモリの最小値は、512K のときノードあたり約 3.2 GiB でした。

## 初回起動の前に

NCCL: NVIDIA は NCCL 2.30.7 を Python の wheel として公開しています。pip（パッケージ `python3-pip`）はライブラリを展開するだけで、Python で何かを実行することはありません。全ノードで次を実行します。

```sh
pip3 install --no-deps --target ~/nccl-2.30.7 nvidia-nccl-cu13==2.30.7
```

`glm53.conf` の `NCCL_LIB` は、既定で `~/nccl-2.30.7/nvidia/nccl/lib/libnccl.so.2` を指しています。ソースからビルドした NCCL 2.30.7（NVIDIA の DGX Spark playbook「NCCL for Two Sparks」、<https://build.nvidia.com/spark/nccl>）でも動きます。その場合は `NCCL_LIB` にそれを指定してください。

チェックポイント: Hugging Face 上の、ノード数に合った Fiore-lab の EXL3 チェックポイントです。8 台では `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks`、4 台では `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks` を使います。Hugging Face CLI で一度だけダウンロードし（`python3 -m venv` には `python3-venv` パッケージが必要です）、RoCE ネットワーク経由で残りのノードにコピーします。8 台の場合は次のとおりです。

```sh
python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install -U huggingface_hub
D=~/models/GLM-5.3-EXL3-8x-DGX-Sparks
~/hf-cli/bin/hf download fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks --local-dir $D
for n in NODE2 NODE3 NODE4 NODE5 NODE6 NODE7 NODE8; do    # the other seven addresses in NODES
  rsync -a --mkpath -e "ssh -c aes128-gcm@openssh.com" $D/ $n:$D/ &
done; wait
```

4 台の場合は、4 台用のチェックポイントと残りの 3 台を使う同じコマンドを `T4.ja.md` に記載しています。

コピーを確認するには、全ノードで `cd ~/models/GLM-5.3-EXL3-8x-DGX-Sparks && sha256sum -c --quiet SHA256SUMS` を実行します。1 台あたり数分かかり、すべてのファイルが一致すれば何も出力しません（このディレクトリを配置した後なら、`./tf-glm53 verify` で全ノードを一度に確認できます）。`glm53.conf` の `WEIGHTS` にこのディレクトリ（ここでは `~/models/GLM-5.3-EXL3-8x-DGX-Sparks`）を指定します。

## 起動

```sh
tar xzf tensorfold-VERSION-linux-aarch64.tar.gz
cd tensorfold-VERSION-linux-aarch64
sha256sum -c SHA256SUMS
cp glm53.conf.example glm53.conf    # nodes, network names, NCCL, weights, drafter, port
./tf-glm53 deploy                   # the same directory on the other nodes
./tf-glm53 up                       # checks, starts the ranks, waits, runs the smoke test
./tf-glm53 -d dflash2 up            # another start with the DFlash2 drafter (Drafters, below)
./tf-glm53 down
```

4 台の場合は、`glm53.conf.example` の代わりに `glm53.t4.conf.example` をコピーします（次の節）。`tf-glm53` は `sudo` を使わず、システム設定も変更しません。スモークテストは、サーバーが応答することと、キャッシュしたプロンプトを再利用できることを確認するもので、精度の指標ではありません。

## 4 台と 8 台の設定

`tf-glm53` は、`NODES` に書いたアドレスの数に応じてテンソル並列 4 または 8 で動かします。2 つの設定例の違いは次の行です。ネットワーク名、NCCL、drafter、サーバーの設定方法はどちらも同じです。

| 設定 | `glm53.conf.example`（8 台） | `glm53.t4.conf.example`（4 台） |
|---|---|---|
| `NODES` | 8 つのアドレス（rank 0 が先頭） | 4 つのアドレス（rank 0 が先頭） |
| `WEIGHTS` | `~/models/GLM-5.3-EXL3-8x-DGX-Sparks` | `~/models/GLM-5.3-EXL3-4x-DGX-Sparks` |
| `KV` | `fp8`（既定） | `fp4` |
| `CONTEXT` | 819200 | 737280 |
| `STORE` | 819200 | 100000 |
| `CACHE` | `pool`（既定） | `pool` |
| `DCP` | 2（既定） | 2 |
| `ROWS` / `SP_PARTS` | 8192 / 4（既定） | 16384 / 4 |

`KV` は KV キャッシュの形式で、FP8 または 4 ビットです。`DCP` は decode コンテキスト並列、`ROWS` と `SP_PARTS` はプロンプトのチャンクの行数と、チャンクを分けるシーケンス並列の部分の数です。`CONTEXT`、`STORE`、`CACHE` は後述の「リクエスト間で保持する会話」で説明します。4 台には、4 台用のチェックポイント、`KV=fp4`、`glm53.t4.conf.example` の値が必要です。詳しくは `T4.ja.md` を参照してください。

## 最初のリクエスト

サーバーは `http://HOST:PORT/v1` で OpenAI chat API を提供し、モデル名は `NAME` で指定したものです（`glm53.conf.example` では `glm-5.3`、ポート 8000）。1 台目のノードで次を実行します。

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3", "max_tokens": 300,
  "messages": [{"role": "user", "content": "Say hello in one short sentence."}]}'
```

応答の `message` には、モデルの思考過程が `reasoning_content` に、続いて回答が `content` に入ります。`max_tokens` は両方を合わせて数えます。別のマシンから送る場合は、`HOST` と `API_KEY_FILE` を設定したうえで、リクエストを `http://HOST:PORT` に送り、`-H "Authorization: Bearer KEY"` を付けます。

## Drafter

エンジンはラウンドごとに数トークンのドラフトを作り、モデルがそれを検証します。どの場合も、応答はモデル自身が選んだものです。ドラフトを作るものは `glm53.conf` の `DRAFTER_KIND` で選びます。コマンドの前に `-d KIND` を付けると、ファイルを書き換えずにその起動だけ選べます（`./tf-glm53 -d dflash2 up`、`./tf-glm53 -d mtp up`）。直前の起動で使ったものは `./tf-glm53 status` が表示します。

- `mtp`（既定）: チェックポイント自身の MTP 層です。GLM-5.3 の重みの一部であり、重みと同じライセンスに従います。この層を含むチェックポイントが必要で、含まないものは `tf-glm53 check` が拒否します。1 ラウンドあたりのドラフト数は `MTP_K` で、既定は 3 です（4 にしても 3 より速くなりませんでした）。
- `dflash2`: incoai/GLM-5.3-DFlash2。置き場所は `DRAFTER` で指定します。**ライセンス上の理由で、このアーカイブにもリポジトリにも同梱していません**。モデルカードには CC BY-NC-ND 4.0（非営利利用のみ、改変禁止）、研究と評価のため、と記載されています。商用ライセンスは Inco AI が提供しています。その利用がこのライセンスの範囲内であれば、下の「DFlash2 を使う手順」で設定してください。decode はさらに速くなります（「速度」）。
- `none`: ドラフトなし。

### DFlash2 を使う手順

1. モデルカード（<https://huggingface.co/incoai/GLM-5.3-DFlash2>）のライセンスを読み、ご自身の利用がその範囲内かを確認してください。エンジンは公開されているファイルをそのまま読みます。起動時に各ランクが drafter の重みのうち自分の担当分を取り出し、メモリ上で 4 ビットに量子化します。事前の変換はなく、ディスクにも何も書き出しません。
2. チェックポイント用に用意した Hugging Face CLI で、先頭のノードで一度だけダウンロードします。版は、このアーカイブで確認した revision に固定します。アカウントもトークンも不要です。重みは `model.safetensors` の 1 ファイル（4,918,859,112 バイト）です。Hugging Face のキャッシュではなく専用のディレクトリ（`--local-dir`）を使ってください。`DRAFTER` に指定するディレクトリには、`config.json` と `model.safetensors` そのものが必要です。

   ```sh
   R=~/models/GLM-5.3-DFlash2
   ~/hf-cli/bin/hf download incoai/GLM-5.3-DFlash2 --revision 425aa615ce320caac34400208b30808c8f14f76c --local-dir $R
   ```

3. 残りのノードの同じパスにコピーします。`tf-glm53 deploy` がコピーするのはこのアーカイブのディレクトリだけで、drafter はコピーしません。

   ```sh
   for n in NODE2 NODE3 NODE4 NODE5 NODE6 NODE7 NODE8; do    # the other addresses in NODES (three on four nodes)
     rsync -a --mkpath -e "ssh -c aes128-gcm@openssh.com" $R/ $n:$R/ &
   done; wait
   ```

4. 先頭のノードを含む全ノードでファイルを確認します。各ランクは自分のノードにあるコピーを読みます。`tf-glm53 check` は実行したノードに `config.json` があるかを見るだけで、ノード間で drafter を比べません。

   ```sh
   for n in NODE1 NODE2 NODE3 NODE4 NODE5 NODE6 NODE7 NODE8; do    # every address in NODES
     printf '%s  %s\n' \
       f59e1da17d41d24a1aba588aecee1607788adb34a03805f2c883add8ca954e9b config.json \
       3105f14043bef642baa49a7d533fdf0b8b2895737ec84b6305601da662656161 model.safetensors |
       ssh $n "cd $R && sha256sum -c --quiet" && echo "ok $n" || echo "FAIL $n"
   done
   ```

   全ノードで `ok` と表示されることを確認してください。
5. `glm53.conf` に同じパスを指定します。

   ```sh
   DRAFTER_KIND=dflash2
   DRAFTER=~/models/GLM-5.3-DFlash2
   ```

   `DRAFTER` を設定して `DRAFTER_KIND` を `mtp` のままにしておけば、`./tf-glm53 -d dflash2 up` でその起動だけ DFlash2 を使えます。DFlash2 では `MTP_K` は使いません。8 台では、起動前に各ノードで使えるメモリが 112 GiB ではなく 115 GiB 必要です（「各ノードに必要なもの」）。`tf-glm53 check` がこれを確認します。プールでは、最近の会話の drafter の状態も保持します（`DRAFT_SAVES`。「リクエスト間で保持する会話」）。
6. いつもどおり起動します。動いていれば先に `./tf-glm53 down`、その後 `./tf-glm53 up` です。`tf-glm53 check` と `up` は `drafter: dflash2 (PATH)` と表示します。エンジン自身の起動時の行は、drafter に関係なく `drafts: on` と表示します。
7. ドラフトが効いているかは、`curl -s http://127.0.0.1:8000/health`（キー不要）の `completion_tokens_total` と `rounds_total` をリクエストの前後で読んで確かめます。前者の増分を後者の増分で割ると、そのリクエストで 1 ラウンドあたりに確定したトークン数になります。1 ならドラフトは 1 つも採用されていません。「速度」と同じようなコーディング用のプロンプトでは、DFlash2 で 4.3〜5.0 でした。

MTP 層に戻すには、`DRAFTER_KIND=mtp` にしてから `./tf-glm53 down` と `./tf-glm53 up` を実行します。

## リクエスト間で保持する会話

`CONTEXT` は、1 件のリクエストが到達できる位置の数です（プロンプトと応答の合計）。`STORE` は、それに加えて以前の会話のために保持する位置の数です。保持の方式は `glm53.conf` の `CACHE` で選びます。

- `pool`（既定）: `CONTEXT` + `STORE` 位置の KV を 1 つのプールに持ち、256 位置ずつのブロックに分けます。各会話の KV は、それらのブロックの表で表します。リクエストは、直前の会話に限らず、どの会話のものであってもキャッシュにある最も長い先頭部分から再開します。会話の次のターンは、その会話自身のブロックの続きから処理します。以前の会話と同じように始まる新しい会話（同じシステムプロンプトを持つもの。たとえば、エージェントのハーネスにはリクエストのたびに約 27K トークンのシステムプロンプトを送るものがあります。サブエージェントも同様です）は、両者が分かれる位置より手前の、以前の会話の埋まったブロックを共有して 1 回だけ保持し、そのブロックから先だけを計算します。プールがいっぱいになると、最も長く使われていない会話から、末尾のブロックを先に手放します。
- `slots`: 会話ごとの保存領域です。以前の会話のために、窓とは別に `STORE` 位置を持ちます。

キャッシュにあるブロックは、再利用する前に、リクエストのトークンと照合します（抜き取った行のハッシュ）。少しでも疑わしい点があれば、すべての会話を捨て、リクエストを最初から計算します（「cold」）。ブロックを誤って再利用することはありません。DFlash2 では、プールは最大 `DRAFT_SAVES` 件（既定は 8）の会話の drafter の状態も保持するため、それらの会話の次のターンでも本来の速さでドラフトを作れます。どちらの方式でも、応答はモデル自身のものです。キャッシュから再開したリクエストは、位置 0 から計算したものと末尾のビットが異なることがあります（丸めの順序が変わるため）。これは slots でも同じです。

`tf-glm53 up` と `status` はプールを表示します。4 台の設定では次のようになります。

```text
conversation cache (pool): 837280 positions in 3271 blocks of 256, shared by up to 64 conversations (context 737280)
```

このとき、サーバーの `/health` には `pool` オブジェクトがあり、`blocks`、`free_blocks`、`slots`（保持している会話の数）、`shared_blocks`（複数の会話が持つブロックの数）、`evicted_total`（空きを作るために捨てた会話の数）、`cold_total`（最初から計算したリクエストの数）が入ります。

## ランクが停止したとき

`tf-glm53 up` は、5 秒ごとにランクを確認する監視プロセスを残します。一部のランクが消え、他のランクがまだ動いている状態を（2 回続けて）検出すると、残りのランクは永久に待ち続けることになるため、すべてのランクを停止し、消えていたランクを `logs/watch.log` に記録します。`tf-glm53 up` で再び起動してください。ランクの起動と停止は `tf-glm53` だけで行ってください。手動で起動したランクには監視プロセスがなく、そのうち 1 つが起動中に失敗すると、他のランクは停止されるまで集団通信の中で待ち続けます。

起動に失敗した場合（ランクが停止した、`READY_TIMEOUT` 以内にサーバーの準備ができなかった、スモークテストに失敗した、または Ctrl-C）、`tf-glm53 up` は戻る前にすべてのランクを停止します。各ノードは自分のランクのログを `logs/rankR.log` に、その前の起動のログを `logs/rankR.log.prev` に残します。`tf-glm53 logs R` でランク R のログを追えます。

## 他のマシンへの提供

サーバーは既定で `127.0.0.1` で待ち受け、OpenAI chat API を提供します。他のマシンに提供するには `HOST` と `API_KEY_FILE` を設定します。キーなしでループバック以外のアドレスを指定すると、`tf-glm53` は拒否します。サーバーは平文の HTTP で通信します。前段に TLS を置くか、閉じたネットワーク内に留めてください。そうしないとキーが平文で流れます。

API キーファイルは 1 台目のノードでのみ読み込まれます。1 行に 1 つのキーを書き（`KEY` または `LABEL: KEY`、`#` 以降はコメント）、他のユーザーが読めない通常ファイル（`chmod 600`）でなければなりません。そうでなければサーバーは起動を拒否します。`tf-glm53 check` はこの両方を検査します。`tf-glm53 deploy` はこのファイルを他のノードにコピーしません。クライアントは `Authorization: Bearer KEY` を送ります。

## 速度

**全ノードで GPU クロックを 2.1 GHz（2,100 MHz）に制限して測定しています**。まだキャッシュにないコーディング用のプロンプト、リクエストは一度に 1 件（reasoning effort low、応答は最大 4,000 トークン）、このアーカイブを `CACHE=pool` で使用しました。8 台: `glm53.conf.example` の設定（FP8 KV キャッシュ、`CONTEXT` 819,200）と `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks`（routed expert は 4 ビット、その他の線形層は 5 ビット）。4 台: `glm53.t4.conf.example` の設定（4 ビット KV キャッシュ、`CONTEXT` 737,280）と `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks`。MTP は 2 回の中央値、DFlash2 は 8 台で 1 回、4 台で 2 回の中央値です。1 ラウンドあたりモデルが進んだトークン数は MTP で 3.4〜3.5、DFlash2 で 4.6〜4.8 でした。実際の数値はチェックポイント、プロンプト、クロックによって変わります。

| ノード数 | プロンプト | MTP（k=3）: 最初のトークンまでの時間 / decode | DFlash2: 最初のトークンまでの時間 / decode |
|---|---|---|---|
| 8 | 128K | 82.8 s / 47.5 tokens/s | 85.2 s / 53.2 |
| 8 | 256K | 172.9 s / 45.7 | 177.6 s / 52.6 |
| 8 | 512K | 376.5 s / 42.8 | 386.0 s / 47.1 |
| 4 | 128K | 120.1 s / 35.3 | 121.9 s / 42.9 |
| 4 | 256K | 253.3 s / 33.2 | 256.9 s / 40.7 |
| 4 | 512K | 565.4 s / 32.2 | 571.5 s / 37.8 |

## 問題が起きたとき

`tf-glm53 check`（`up` の最初のステップでもあります）は、各ノードの問題を `NG:` の後に列挙します。すべての項目が NG のノードは、1 台目から到達できないノードです。`NODES` のすべてのアドレスについて、1 台目から `ssh -o BatchMode=yes ADDRESS true` が成功する必要があります。

| `NG:` | 対処 |
|---|---|
| `files` | `tf-glm53 deploy` をやり直します |
| `weights` | `WEIGHTS` にチェックポイントがないか、ファイル名、サイズ、`config.json` のいずれかが 1 台目のものと異なっています。内容は `tf-glm53 verify` で確認できます |
| `NCCL(...)` | そのノードの `NCCL_LIB` に NCCL がありません（「初回起動の前に」を参照） |
| `cuBLAS-13` | CUDA 13 の cuBLAS（`libcublas-13-0`）をインストールします |
| `libibverbs` | `libibverbs1` と `ibverbs-providers` をインストールします |
| RDMA デバイス名 | そのノードでは `IB_HCA` が RDMA デバイスではありません。`rdma link` でデバイスを一覧でき、`ibdev2netdev` でそのポート名がわかります |
| `RoCE-v2-GID` | そのポートに、IPv4 アドレスを持つ RoCE v2 のエントリがありません。ポートにアドレスを割り当てるか、`GID_INDEX` を設定します |
| `GPU-busy` | 別のプロセスが GPU を使っています（`nvidia-smi`） |
| `memory(N<M GiB)` | そのノードのメモリを空けるか、`STORE` を下げます |
| `API_KEY_FILE ...` | その行の表示に従います |

`up` が失敗すると、すべてのランクを停止して理由を表示します。各ノードには自分のランクのログが残ります（`logs/rankR.log`、`tf-glm53 logs R`）。

- `a rank stopped (N/M running)`: その上の行に、停止したランクと、そのログの最後のエラーが表示されます。接続に関する `NCCL WARN` 行がある場合: `NODES` は RoCE ネットワーク上のアドレス、`SOCKET_IFNAME` と `IB_HCA` は全ノードでそのネットワークの名前でなければならず、ノード同士がそのネットワーク上で任意のポートに到達できる必要があります。`cannot listen on` の場合: そのノードで `PORT` が使用中です。`PoolValuesDiffer` の場合: ランクごとに異なるプールの設定で起動されたため、起動を拒否しています。すべてのランクを 1 つの `glm53.conf` から `tf-glm53` で起動してください。
- `not ready after N s`: チェックポイントがローカルの NVMe ドライブにある場合、起動には約 2 分半かかります。遅いディスクやネットワークストレージではさらに時間がかかるので、`READY_TIMEOUT` を引き上げてください。ランデブーが完了しない場合も待ち続けます。1 台目のノードで `MASTER_PORT` が空いているか確認してください。
- `the smoke test failed`: 失敗した質問と、それに対して返ってきた応答が表示されます。リクエストは rank 0 のログで確認できます。

稼働中は、`tf-glm53 status` で全ランク、各ランクの空きメモリ、直前の起動で使った drafter、会話キャッシュ、サーバーの `/health` を確認できます。すべてのランクが動いているのにサーバーが応答しなくなった場合は、`tf-glm53 down` と `tf-glm53 up` を実行してください。監視プロセスがランクを停止するのは、一部のランクが消えたときだけです。あるランクが、プールのうちそのリクエストに割り当てたブロックの外への書き込みを拒否した場合（ログに `PoolWriteRefused`）、そのリクエストの終わりにすべてのランクが一緒に停止します。`tf-glm53 up` で再び起動してください。`DCP=1` で起動すると、プロンプトのチャンクをシーケンス並列の部分に分けずに処理する旨の警告がログに出ます（このとき `SP_PARTS` は使われません）。

`CACHE=slots` では、起動時にあるランクで `STORE` 位置分のメモリが足りなかった場合、`up` と `status` は `slot cache: WARNING: no store` と表示します。サーバーは応答しますが、リクエストが再開できるのは直前の会話だけです。ノードのメモリを空けるか `STORE` を下げてから、`tf-glm53 down` と `tf-glm53 up` を実行してください。

## 制限

- GB10 システム 4 台または 8 台のみ（テンソル並列 4 または 8）に対応します。他の GPU モデルや他のランク数には対応していません。4 台には、4 台用のチェックポイントと `KV=fp4` が必要です。
- 同時に実行するリクエストは 1 件です。他のリクエストは 32 件まで到着順に待ちます。チャットのタイトル作成のリクエスト（と `"priority": "background"` を付けたもの）は他の後に回ります。32 件を超えたリクエストは「the engine is busy; retry shortly」で断られます。ストリーミングでないリクエストには 503、ストリーミングのリクエストにはストリーム中のエラーイベントとして返ります。
- MTP drafter では、会話を保持する時点で、応答の最後のラウンドで受理された行がまだ MTP 層のキャッシュに書き込まれていません。`CACHE=slots` では、そこから再開する後続のリクエストで、その数位置のドラフトの質がわずかに下がります。プールでは、MTP 層のキャッシュがそろっている位置（数位置手前）までだけ会話を保持し、その数位置を計算し直します。どちらの場合も、応答はモデル自身のものです。
- エンジンの起動時の行は、drafter の種類にかかわらず `drafts: on` と表示します。これはリクエストがドラフトを使えるという意味です。使用中の drafter は、その後に `tf-glm53` が表示します。
- コンテナ内では、`tf-glm53 check` はホストや他のコンテナの GPU プロセスを見ることができません。ただし空きメモリの確認によって、使用中のノードでの起動は止まります。コンテナには GPU、`/dev/infiniband`、ホストネットワーク、無制限のロックメモリ上限、高いオープンファイル数上限（`--ulimit memlock=-1 --ulimit nofile=1048576:1048576`）が必要です。
