[English](README.md) | 日本語 | [简体中文](README.zh-CN.md) | [한국어](README.ko.md)

# 4 / 8 台のDGX Spark用　GLM-5.3-full


フルサイズの GLM-5.3（755B、`glm_moe_dsa`）を、GB10 を 1 基ずつ搭載した NVIDIA DGX Spark 4 台または 8 台で、テンソル並列 4 または 8 で動かします。リクエストは一度に 1 件ずつ、高速に、エージェントやコーディング向けの長いコンテキストで処理します。エンジンは Zig と素の CUDA カーネルで書かれており、実行時に Python は不要です。

これは [TensorFold](https://github.com/ashhart/TensorFold) 1.0.2 から Fiore-lab が作った非公式の派生物であり、TensorFold プロジェクトのリリースではありません。TensorFold のコードを含み、フルサイズの GLM-5.3 用のエンジン（`zig/src/families/glm_moe_dsa/`、`zig/kernels/cuda/glm53/`）を追加し、TensorFold のドキュメントは除いています。それらのドキュメントは別のモデル、重み、マシンについて書かれたものだからです。TensorFold 自体については、そのリポジトリを参照してください。

**AI アシスタントを使ってこのシステムを理解したり動かしたりする場合**は、[glm53/release/AI-GUIDE.ja.md](glm53/release/AI-GUIDE.ja.md)（リリースアーカイブでは `AI-GUIDE.ja.md`。英語版は `AI-GUIDE.md`）を渡してください。AI アシスタント向けに書いた 1 ページの説明で、システムの全体像、目的、準備と使い方、できることとできないこと（実測の数値つき）、詳細の置き場所をまとめています。

## できること

- テンソル並列 8 または 4。KV キャッシュはランクのペアに分割します（decode コンテキスト並列 2）。エンジンとモデルは 1M（1,048,576 トークン）まで扱え、**8 台でも 4 台でも 1M に届きます**。8 台: 4 ビット KV キャッシュ（`KV=fp4`、`NEED_GB=100`）なら `CONTEXT=1048576` に加えて `STORE=1048576`（1 ノードあたり約 23 GiB の余裕）、8 ビットのキャッシュなら `STORE=589824`。4 台: コントロールパネルで routed expert の 70 % を 3 ビット、残りを 4 ビットにして `CONTEXT=1048576`（`KV=fp4`、`STORE` 512。1 ノードあたり約 7.5 GiB の余裕があり、動作中の並べ替えもできます）。4 台用チェックポイントだけなら `CONTEXT=819200` `STORE=18080` です。どれも約 1,019,000 トークンのプロンプト 1 本で実測しました。4 ビット KV キャッシュは精度が少し下がります（同じ重みで KL 0.0190、8 ビットのキャッシュでは 0.0156: [glm53/release/T4.ja.md](glm53/release/T4.ja.md)）。下の速度の 8 台は 8 ビットのキャッシュです。設定例は 819,200 と 737,280 になっています（KV プールは `CONTEXT` + `STORE` トークンを持ちます）。
- 長いプロンプトはシーケンス並列で分割して読み込みます。decode ステップの集団通信は RoCE 上で行います。
- すべての会話で 1 つの会話キャッシュを使い、256 トークンのブロック単位で保持します。リクエストは、どの会話のものであってもキャッシュにある最も長い先頭部分から再開します。同じように始まる会話（エージェントのシステムプロンプト、そのサブエージェント）は、共通のブロックを 1 回だけ保持します。
- ラウンドごとにドラフトをモデルが検証します。ドラフトを作るのは、既定では GLM-5.3 自身の MTP 層、またはオプションの DFlash2 drafter です。どの場合も、応答はモデル自身が選んだものです。
- `/metrics`（Prometheus のテキスト）と `/health` は実行中のリクエストの分も数えるので、モニタはリクエストの実行中にも prefill と decode の速さ、キャッシュのヒット、ドラフトの受理を読めます。`METRICS_LISTEN` を設定すると、別のホストのモニタ向けに、この 2 つを専用のアドレスで読み取り専用で開きます。

## r3.2-102 で変わったこと

- エンジンの動作中、ディスクには何も書きません。各ランク、監視プロセス、ランチャーのログは、各ノードのログ係（`tools/logkeep.py`。各ノードに `python3` が要ります。DGX OS にはあります。アーカイブでもイメージでも同じです）のメモリに、1 本ごとに直近の `LOG_KEEP_MB`（8）だけ置き、`./tf-glm53 logs R` で読みます。`RUNTIME=docker` では、コンテナもログを持ちません。r3.1-102 は、ユーザーが linger のときだけログをメモリに置きました。r3.2-102 は linger が要りません。
- 記録は別のホストのモニタが持ちます。`LOG_LISTEN=HOST:PORT` を設定すると、各ノードのログ係が読み取り専用・キーなしでログを返し（`GET /streams`、`GET /logs`）、メトリクスの待ち受け（`METRICS_LISTEN`）は expert の使用回数も返します（`GET /v1/experts/usage`）（[packaging/RUNTIME.ja.md](packaging/RUNTIME.ja.md)「他のマシンへの提供」）。

## r3.1-102 で変わったこと

- API とコントロールパネルは、既定で全アドレス（`HOST=0.0.0.0`、`LAUNCHER_HOST=0.0.0.0`）でキーなしで待ち受けます。ネットワーク上の他のマシンから直接使えます。ノードは閉じたネットワーク内に置いてください（`HOST=127.0.0.1` / `LAUNCHER_HOST=127.0.0.1` で 1 台目のノードからだけになります）。
- Messages API で Claude Code が使えます。`claude-glm53` が Claude Code をサーバーに向けて起動し、サーバーはその会話をキャッシュに保ち、長いプロンプトのストリームを保ち、画像を注記に置き換えます（[packaging/RUNTIME.ja.md](packaging/RUNTIME.ja.md)「Claude Code」）。
- コントロールパネルは、`./tf-glm53 up` で起動したエンジンでも使えます（エンジンは既定で expert の使用回数を数えます）。動作中の並べ替えは、大きくならない層から先に入れ替え、始める前に入れ替えの途中で最も多く使うメモリを全ノードで確かめます。足りないノードはパネルが示します。
- `glm53.conf` の全設定項目: [glm53/release/CONFIG.ja.md](glm53/release/CONFIG.ja.md)。

## Expert Atlas（r3-102 の新機能）

GLM-5.3 はトークンごとに、1 層 256 個の routed expert から 8 個を選びます。expert によって選ばれる頻度は大きく違います。Expert Atlas は 19,200 個の routed expert（層 3〜77）を、それぞれ自分の幅（2・3・4・5 ビット）で読み込み、限られたメモリの中で、よく使われる expert ほど多くのビットを持たせます。

- **バンク**: 各幅の routed expert で、Hugging Face にあります。[fiore-lab/GLM-5.3-EXL3-Experts-2bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-2bit)、[-3bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-3bit)、[-4bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-4bit)、[-5bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-5bit)。パネルが選ばせるのは、全ノードにそろっている幅です。
- **コントロールパネル**: ランチャー（`./tf-glm53 launcher`）が出すページです。起動の前には幅を計画し、各ノードのメモリ、計画が収まるか、KV キャッシュに残る量を表示します。エンジンの起動と停止もここから行います。稼働中は、各トークンが選ぶ expert、選ばれた expert の平均の幅、速さを表示します。
- **利用記録**: エンジンは、プロンプトの各行と decode の各トークンが選ぶ expert を数えます。保存した割り当ては保存元の稼働の利用記録を持ち、記録は割り当てと一緒に動きます。開いてもう一度稼働させると、その稼働の分が記録に加わります。計画はこの記録で expert を並べます（記録の無い計画は、Fiore-lab が測ってページに同梱した回数で並べます）。自動では何も保存しません。停止時に、割り当てとその記録に名前を付けて保存します。
- **エキスパート最適化**: 幅ごとの expert の個数はそのままで、稼働の利用記録で読み込み済みの expert を並べ替えます。同じメモリで、モデルが実際に選ぶ expert の平均の幅が上がります。稼働中のエンジンで使えます（4 台で 75 層全部の変更に 83〜90 秒、リクエストの合間に行います）。結果は必要なときに名前を付けて保存し、後の起動で開きます。

コントロールパネルは 4 台または 8 台で動き、台数は起動画面で選びます。`./tf-glm53 up` はこれまでどおりチェックポイントの幅で起動します。詳しくは `packaging/RUNTIME.ja.md`（「コントロールパネル」）と `glm53/release/T4.ja.md` にあります。

## Hugging Face の重み

| 何 | リポジトリ | 大きさ | 使う場面 |
|---|---|---|---|
| 8 台用チェックポイント：すべての重み（routed expert は 4 bit） | [fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks) | 390 GB | 8 台での起動。4 台・8 台のコントロールパネル（4 bit のバンクも兼ねる） |
| 4 台用チェックポイント：使われ方の少ない半分の routed expert を 3 bit に | [fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks) | 344.6 GB | コントロールパネルを使わない 4 台での起動 |
| エキスパートのバンク 2 bit | [fiore-lab/GLM-5.3-EXL3-Experts-2bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-2bit) | 187.3 GB | コントロールパネル |
| エキスパートのバンク 3 bit | [fiore-lab/GLM-5.3-EXL3-Experts-3bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-3bit) | 277.9 GB | コントロールパネル（4 台で文脈を最大まで取るときに必要） |
| エキスパートのバンク 4 bit | [fiore-lab/GLM-5.3-EXL3-Experts-4bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-4bit) | 368.5 GB | コントロールパネル（8 台用チェックポイントがないノードで） |
| エキスパートのバンク 5 bit | [fiore-lab/GLM-5.3-EXL3-Experts-5bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-5bit) | 459.1 GB | コントロールパネル |

バンクは、19,200 個の routed expert を 1 つの幅でそろえたものです。コントロールパネルは、エキスパートごとに、計画で決めた幅の
バンクから読み込みます。**バンクは、使いたい幅の分だけダウンロードすれば足ります**（パネルが選ばせるのは、全ノードにそろって
いる幅です）。4 台でコントロールパネルを使うなら、8 台用チェックポイントと 3 bit のバンク（合わせて 668 GB）で足ります。
どのノードにも、同じファイルを同じパスに置きます。1 台でダウンロードして、ほかのノードに写します。例:

```sh
python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install -U huggingface_hub
~/hf-cli/bin/hf download fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks --local-dir ~/models/GLM-5.3-EXL3-8x-DGX-Sparks
```

ほかのノードへの写し方と照合（`./tf-glm53 verify`、`./tf-glm53 verify-banks`）は [packaging/RUNTIME.ja.md](packaging/RUNTIME.ja.md)
にあります。

## 重みを自分で作る

重みはコードのようには読めません。ある場面だけ違うふるまいをするように手を加えた重みも見た目はほかと変わらないので、作り方を
確かめられない重みには頼りたくないこともあるでしょう。Fiore-lab の重みに頼る必要はありません。このレシピが読み込む重みのファイル（8 台用・4 台用のチェックポイントと、
2・3・4・5 ビットの expert のバンク）はどれも、Z.AI の公式 BF16 チェックポイント（[zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16)）
から [glm53/weights](glm53/weights/README.ja.md) の道具で作れます。手順・機材・実測の時間はそこにあります。作ったファイルは
Fiore-lab のものと形が同じで、同じように読み込まれます。

## 必要なもの

- 1 つの RoCE v2 ネットワークでつながった DGX Spark（GB10）4 台または 8 台。どのノードにも 1 台目からパスワードなしの ssh で接続できること。4 台なら、各ノードに 100 Gb/s のポートが 1 つあれば足ります。
- DGX OS（Ubuntu 24.04）、NVIDIA ドライバ 580 以降、cuBLAS 13。
- 全ノードに NCCL 2.30.7。NVIDIA の `nvidia-nccl-cu13` wheel を pip で展開したものです（`packaging/RUNTIME.md`）。
- ノード数に合った Fiore-lab の EXL3 チェックポイント（Hugging Face）。全ノードの同じパスに置きます。8 台では [fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks)（390 GB）、4 台では [fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks)（344.6 GB）です。4 台用のチェックポイントは、層 3〜77 の routed expert のうち使用頻度の低い半分を 3 ビット、残りを 4 ビットで持ち、それ以外は 8 台用のチェックポイントと同じです。
- 重みのファイルはどれも、Fiore-lab のファイルを使わずに、Z.AI の公式 BF16 チェックポイントから自分で作ることもできます（[glm53/weights](glm53/weights/README.ja.md)）。
- コントロールパネルには、使う幅の expert のバンク。4 台では 3 ビットと 4 ビットのバンク（277.9 GB と 368.5 GB。8 台用のチェックポイントを 4 ビットのバンクとして使えます。4 ビットだけでは文脈が短くなります）と、ランチャー用に各ノードの `python3`。

## 実行

リリースは、コンテナイメージと、同じファイルを持つアーカイブの 2 つの形で配布します。おすすめはイメージです。イメージを使う場合、各ノードに要るのは Docker と NVIDIA Container Toolkit（DGX OS には両方あります）だけで、サーバーのライブラリは要りません。

```sh
docker pull ghcr.io/fiore-labo/expertatlas-glm5.3:r3.2-102            # 各ノードで
mkdir glm53 && docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/glm53":/out ghcr.io/fiore-labo/expertatlas-glm5.3:r3.2-102 install /out
```

そのあと `glm53.conf` に `RUNTIME=docker` と `IMAGE` を設定し、以下と同じように進めます（[packaging/RUNTIME.ja.md](packaging/RUNTIME.ja.md)「コンテナで動かす」）。Docker を使わない場合は、リリースアーカイブ `tensorfold-r3.2-102-linux-aarch64.tar.gz` をダウンロードし、同梱の `RUNTIME.md`（このリポジトリでは [packaging/RUNTIME.ja.md](packaging/RUNTIME.ja.md)）に従ってください。4 台でも 8 台でも、同じアーカイブと起動スクリプトを使います。`glm53.conf.example`（8 台）または `glm53.t4.conf.example`（4 台。[glm53/release/T4.ja.md](glm53/release/T4.ja.md) を参照）を `glm53.conf` にコピーし、ノード、ネットワーク名、NCCL、重み、drafter を記入してから、`./tf-glm53 deploy` と `./tf-glm53 up` を実行します。サーバーは既定で全アドレス（`HOST=0.0.0.0`）で OpenAI chat API を提供します。API にキーはないので、届く人なら誰でも使えます。閉じたネットワーク内に留めるか、前段に TLS とアクセス制御を置いてください（`HOST=127.0.0.1` にすると 1 台目のノードからだけになります）。

`./tf-glm53 up` の代わりにコントロールパネルから使うときは、1 台目のノードで `./tf-glm53 launcher` を実行し、`http://FIRST-NODE:LAUNCHER_PORT/panel`（4 台の例ではポート 8010）を開きます。エンジンの起動と停止はそこで行います（[packaging/RUNTIME.ja.md](packaging/RUNTIME.ja.md)「コントロールパネル」）。`glm53.conf` の全設定項目、読む側、変更が効く時期は [glm53/release/CONFIG.ja.md](glm53/release/CONFIG.ja.md) にあります。

## Drafter

- `mtp`（既定）: チェックポイント自身の MTP 層で、GLM-5.3 の重みの一部です。decode 速度はドラフトなしの約 2 倍で、prefill 速度は変わりません。
- `dflash2`: incoai/GLM-5.3-DFlash2。**ライセンス上の理由で、このリポジトリにもアーカイブにも同梱していません**。モデルカードには CC BY-NC-ND 4.0（非営利利用のみ、改変禁止）、研究と評価のため、と記載されています。商用ライセンスは Inco AI が提供しています。その利用がこのライセンスの範囲内であれば、[packaging/RUNTIME.ja.md](packaging/RUNTIME.ja.md#dflash2-を使う手順) の「DFlash2 を使う手順」に従ってください。固定した版で一度だけダウンロードし、全ノードの同じパスにコピーしてコピーを確認してから、`DRAFTER_KIND=dflash2` と `DRAFTER` を設定します。エンジンは起動時にメモリ上でこれを 4 ビットに量子化し、ディスクには何も書き出しません。decode はさらに速くなります（「速度」）。
- `none`: ドラフトなし。

`glm53.conf` の `DRAFTER_KIND` で選びます。`./tf-glm53 -d KIND up` なら、その起動だけ選べます。

## 速度

**全ノードで GPU クロックを 2.1 GHz（2,100 MHz）に制限して測定しています**。まだキャッシュにないコーディング用のプロンプトで、リクエストは一度に 1 件（reasoning effort low、応答は最大 4,000 トークン）。このアーカイブを `glm53.conf.example`（8 台、FP8 KV キャッシュ）と `glm53.t4.conf.example`（4 台、4 ビット KV キャッシュ）の設定で使い、Fiore-lab が公開しているチェックポイントで測りました。実際の数値はチェックポイント、プロンプト、クロックによって変わります。測定条件の詳細は `packaging/RUNTIME.md` に記載しています。

| ノード数 | プロンプト | MTP（k=3）: 最初のトークンまでの時間 / decode | DFlash2: 最初のトークンまでの時間 / decode |
|---|---|---|---|
| 8 | 128K | 82.8 s / 47.5 tokens/s | 85.2 s / 53.2 |
| 8 | 256K | 172.9 s / 45.7 | 177.6 s / 52.6 |
| 8 | 512K | 376.5 s / 42.8 | 386.0 s / 47.1 |
| 4 | 128K | 120.1 s / 35.3 | 121.9 s / 42.9 |
| 4 | 256K | 253.3 s / 33.2 | 256.9 s / 40.7 |
| 4 | 512K | 565.4 s / 32.2 | 571.5 s / 37.8 |

## 精度

元の BF16 モデル（公式の BF16 重みを公式実装で動かし、FP32 で計算したもの）の出力分布に対する、このレシピの出力分布の KL ダイバージェンスです。このレシピのサービング構成で求めました。

- 8 台（FP8 KV キャッシュ、DCP 2。同じ重みを、このエンジンの以前のビルドで測定）、2,048 トークンのテキスト 4 本（対話、法律、散文、推論）、それぞれ最後の 1,023 トークン、5 回実行: 平均 0.0133（0.0128〜0.0146）。top-1 トークンの一致率は約 96% です。
- 8 台、8,192 トークンのテキスト 4 本（英語 2 本、コード 2 本）、トークンの範囲ごと: 0.012（トークン 1,024〜2,047）、0.018、0.019、0.013（6,144〜8,190）。比較の目安として、公式の BF16 重みを FP32 ではなく BF16 で計算すると、同じ範囲で同じ基準との差は 0.007〜0.010 です。
- 4 台（4 台用のチェックポイント、4 ビット KV キャッシュ、DCP 2）、8,192 トークンのテキスト 4 本、同じトークンの範囲ごと: 0.019（トークン 1,024〜2,047）、0.032、0.029、0.019（6,144〜8,190）。
- Expert Atlas の各幅（routed expert をすべて 1 つの幅にし、それ以外の重みは上と同じ、同じテキスト 4 本、トークン 6,144〜8,190）: 2 ビット 0.0517、3 ビット 0.0203、4 ビット 0.0127（4 台）、5 ビット 0.0114（8 台）。同じメモリで比べると、2 ビットの expert を混ぜた配分は混ぜない配分より正解から 13〜30 % 遠く、ほかを 3 ビットに下げて 5 ビットの expert を足しても、測れるほどの改善はありませんでした。

## ソースからのビルド

Zig 0.17.0 が `PATH` 上にあり（または `build_libs.sh` には `ZIG=` でそのパスを指定）、CUDA 13（`nvcc`）と libibverbs のヘッダ（`libibverbs-dev`）がある GB10 システムで次を実行します。

```sh
zig build fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 --prefix /tmp/fatbins
sh glm53/tools/release/build_libs.sh /tmp/lib        # lib/libroce_proxy.so
zig build dist -Dversion=r3.2-102 -Ddist-commit=$(git rev-parse HEAD) -Ddist-platforms=linux-aarch64 \
    -Ddist-fatbins=/tmp/fatbins/fatbin -Ddist-lib=/tmp/lib -Ddist-extra=glm53/release
```

アーカイブは `zig-out/dist/` に出力されます。

## 制限

- GB10 システム 4 台または 8 台のみに対応します。他の GPU や他のランク数には対応していません。
- 同時に実行するリクエストは 1 件です。他のリクエストは 32 件まで到着順に待ち（チャットのタイトル作成は最後）、それを超えるとサーバーは断ります（503、ストリーミングではエラーイベント）。サブエージェントを並列に動かすエージェントでは、サブエージェントを別のモデルに向けるほうが適しています。
- その他の制限は `packaging/RUNTIME.md` に記載しています。

## ライセンスとクレジット

コードは Apache-2.0（`LICENSE`）です。どの部分がどこに由来するかは `NOTICE` と `THIRD_PARTY_NOTICES.md` に記載しています。重みはそれ自身のライセンス（GLM-5.3 License）に従い、オプションの DFlash2 drafter もそれ自身のライセンスに従います。

- TensorFold のコントリビューターによる [TensorFold](https://github.com/ashhart/TensorFold): このレシピが拡張しているサーバー、ビルド、CUDA エンジン。
- drowzeys による TensorFold のフルサイズ GLM-5.3 用 Python エンジン（[ashhart/TensorFold#159](https://github.com/ashhart/TensorFold/pull/159)）: GLM-5.3 エンジンは、その設計（エンジンの構造、Triton カーネル、MTP と DFlash2 の drafter の処理）を Zig と素の CUDA で書き直したものです。その CUDA ファイルの一部を収録しており、いくつかは変更を加えています。
- [b12x](https://github.com/local-inference-lab/b12x): RoCE の one-shot 集団通信を素の CUDA で書き直しました（`roce.cu`）。RoCE プロキシは収録し（`zig/native/roce_proxy.c`）、RDMA デバイスごとに GID インデックスを選ぶよう変更しました。
- [MiaAI-Lab の GLM-5.3-Flash on 2x DGX Spark](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold): そのプロンプト処理用カーネルを参考に、GLM-5.3 のプロンプト処理用カーネルを書きました。
- [MiaAI-Lab の GLM-5.3 EXL3 on 3x DGX Spark](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold): 4 ビット KV キャッシュはそのレイアウトを採用し、このエンジン独自のコードで実装しました。
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3): チェックポイントはこれで量子化しており、エンジンはその EXL3 形式を読み込みます。

## TensorFold のコントリビューター

TensorFold の README より: TensorFold にプルリクエスト、測定結果、バグ報告を送ってくださったすべての方に感謝します（TensorFold 1.0.2 時点のリストです。最新のリストは [TensorFold のリポジトリ](https://github.com/ashhart/TensorFold) を参照してください）。

[@0mao0](https://github.com/0mao0),
[@321sssrt-bit](https://github.com/321sssrt-bit),
[@aditya1503](https://github.com/aditya1503),
[@AdrianBinDC](https://github.com/AdrianBinDC),
[@akol1](https://github.com/akol1),
[@Alexbob0](https://github.com/Alexbob0),
[@anvilsong](https://github.com/anvilsong),
[@Arminova](https://github.com/Arminova),
[@b-ostrov](https://github.com/b-ostrov),
[@barelyworkingcode](https://github.com/barelyworkingcode),
[@Benjamin-Wegener](https://github.com/Benjamin-Wegener),
[@benthecarman](https://github.com/benthecarman),
[@benwilson](https://github.com/benwilson),
[@BHCC2025](https://github.com/BHCC2025),
[@Bizuayeu](https://github.com/Bizuayeu),
[@BlivionIaG](https://github.com/BlivionIaG),
[@BobClawblaw](https://github.com/BobClawblaw),
[@borodach23](https://github.com/borodach23),
[@Boscoeuk](https://github.com/Boscoeuk),
[@boxabirds](https://github.com/boxabirds),
[@brandonmmusic-max](https://github.com/brandonmmusic-max),
[@bunnyfu](https://github.com/bunnyfu),
[@CerebralCoding](https://github.com/CerebralCoding),
[@cesarswong](https://github.com/cesarswong),
[@chadhurley25075-png](https://github.com/chadhurley25075-png),
[@chaog992](https://github.com/chaog992),
[@Charlie-Louis](https://github.com/Charlie-Louis),
[@Chedrian07](https://github.com/Chedrian07),
[@chris247474](https://github.com/chris247474),
[@christrade215](https://github.com/christrade215),
[@crescit](https://github.com/crescit),
[@cshintov](https://github.com/cshintov),
[@cwschroeder](https://github.com/cwschroeder),
[@DakotaTexas](https://github.com/DakotaTexas),
[@danyo1399](https://github.com/danyo1399),
[@Deesha08](https://github.com/Deesha08),
[@Defilan](https://github.com/Defilan),
[@DevRico003](https://github.com/DevRico003),
[@di37](https://github.com/di37),
[@drowzeys](https://github.com/drowzeys),
[@ecohash-co](https://github.com/ecohash-co),
[@edurdias](https://github.com/edurdias),
[@eleqtrizit](https://github.com/eleqtrizit),
[@ericlsimplifi](https://github.com/ericlsimplifi),
[@EugeneClaw](https://github.com/EugeneClaw),
[@feni6](https://github.com/feni6),
[@gbgbgbg](https://github.com/gbgbgbg),
[@GDACONSULT](https://github.com/GDACONSULT),
[@gecobattya](https://github.com/gecobattya),
[@gilby](https://github.com/gilby),
[@Gogo6969](https://github.com/Gogo6969),
[@gprot42](https://github.com/gprot42),
[@GraithSecurity](https://github.com/GraithSecurity),
[@grantoverton](https://github.com/grantoverton),
[@grearjake-star](https://github.com/grearjake-star),
[@greatyingzi](https://github.com/greatyingzi),
[@harrisonfriia](https://github.com/harrisonfriia),
[@haxudev](https://github.com/haxudev),
[@heitke](https://github.com/heitke),
[@hichaiuse](https://github.com/hichaiuse),
[@ivanfioravanti](https://github.com/ivanfioravanti),
[@jasontitus](https://github.com/jasontitus),
[@jayleaton](https://github.com/jayleaton),
[@jeffpeng3](https://github.com/jeffpeng3),
[@jeidbugs404](https://github.com/jeidbugs404),
[@jetnet](https://github.com/jetnet),
[@jkuepker](https://github.com/jkuepker),
[@johnymoo](https://github.com/johnymoo),
[@JordiPosthumus](https://github.com/JordiPosthumus),
[@JRaxworthy](https://github.com/JRaxworthy),
[@jregan-beasley-smc](https://github.com/jregan-beasley-smc),
[@jschmied](https://github.com/jschmied),
[@juliankang4](https://github.com/juliankang4),
[@kingjamez](https://github.com/kingjamez),
[@kky42](https://github.com/kky42),
[@lcgutierrez](https://github.com/lcgutierrez),
[@LECYWZA](https://github.com/LECYWZA),
[@lijian1999](https://github.com/lijian1999),
[@liumorrisclaw](https://github.com/liumorrisclaw),
[@LXD-8](https://github.com/LXD-8),
[@m-naoki-m](https://github.com/m-naoki-m),
[@mapamalu](https://github.com/mapamalu),
[@mcclanahanaman](https://github.com/mcclanahanaman),
[@MESevenJourney](https://github.com/MESevenJourney),
[@mgoldwasser](https://github.com/mgoldwasser),
[@MiaAI-Lab](https://github.com/MiaAI-Lab),
[@mikolaj92](https://github.com/mikolaj92),
[@millaguie](https://github.com/millaguie),
[@Mirrdhyn](https://github.com/Mirrdhyn),
[@Moutonc](https://github.com/Moutonc),
[@MovieMaker93](https://github.com/MovieMaker93),
[@mrpmorris](https://github.com/mrpmorris),
[@MV10](https://github.com/MV10),
[@NeoAiLabs](https://github.com/NeoAiLabs),
[@Nipale-ai](https://github.com/Nipale-ai),
[@nood-co1](https://github.com/nood-co1),
[@nullburn](https://github.com/nullburn),
[@olexale](https://github.com/olexale),
[@omar16100](https://github.com/omar16100),
[@optimisme](https://github.com/optimisme),
[@outcastofmusic](https://github.com/outcastofmusic),
[@paragontasx](https://github.com/paragontasx),
[@peacockesq](https://github.com/peacockesq),
[@philip-pentatonic](https://github.com/philip-pentatonic),
[@plotarmordev](https://github.com/plotarmordev),
[@pmeenan](https://github.com/pmeenan),
[@pulseandthread](https://github.com/pulseandthread),
[@quigles1977](https://github.com/quigles1977),
[@rafafortes](https://github.com/rafafortes),
[@raymondkpwong](https://github.com/raymondkpwong),
[@robertpitt](https://github.com/robertpitt),
[@RoscoeTT](https://github.com/RoscoeTT),
[@salmanarshad321](https://github.com/salmanarshad321),
[@samwang0041-star](https://github.com/samwang0041-star),
[@sanjaibalajee](https://github.com/sanjaibalajee),
[@satindergrewal](https://github.com/satindergrewal),
[@scottleimroth](https://github.com/scottleimroth),
[@sethforprivacy](https://github.com/sethforprivacy),
[@sfxnz](https://github.com/sfxnz),
[@shantanugoel](https://github.com/shantanugoel),
[@simon-lin88](https://github.com/simon-lin88),
[@simonmd](https://github.com/simonmd),
[@spenchey](https://github.com/spenchey),
[@squarrier](https://github.com/squarrier),
[@ss-cong](https://github.com/ss-cong),
[@styles01](https://github.com/styles01),
[@SxMShaDoW](https://github.com/SxMShaDoW),
[@sxuff](https://github.com/sxuff),
[@taussoe](https://github.com/taussoe),
[@tfolkman](https://github.com/tfolkman),
[@ThinkOffApp](https://github.com/ThinkOffApp),
[@Thotheris](https://github.com/Thotheris),
[@tinyapps](https://github.com/tinyapps),
[@tolewis](https://github.com/tolewis),
[@tomByrer](https://github.com/tomByrer),
[@tonydehnke](https://github.com/tonydehnke),
[@tournierjc](https://github.com/tournierjc),
[@tpischke](https://github.com/tpischke),
[@urtho](https://github.com/urtho),
[@vcruz305](https://github.com/vcruz305),
[@vinicius-symetrix](https://github.com/vinicius-symetrix),
[@wojo](https://github.com/wojo),
[@xjqx2z](https://github.com/xjqx2z),
[@Yuepixel](https://github.com/Yuepixel),
[@YvesLaRose](https://github.com/YvesLaRose).
