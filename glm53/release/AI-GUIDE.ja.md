[English](AI-GUIDE.md) | 日本語 | [简体中文](AI-GUIDE.zh-CN.md)

# AI アシスタント向けガイド: DGX Spark で動かすフルサイズの GLM-5.3（リリース r3.3-102）

このページは、システム全体を 1 か所にまとめたものです: これが何か、各部分が何をするか、どう使うか、何ができて何ができないか。利用者がこのリリースを理解する、インストールする、動かす、不具合を切り分けるのを手伝う AI アシスタント向けに書いていますが、人が読んでも同じ内容です。数値は他のページから取っています。詳細はそれらのページにあり、このページより詳しく書いています: `QUICKSTART.ja.md`（初めての起動）、`BEFORE-START.ja.md`（起動前に決めること）、`RUNTIME.ja.md`（動かし方。4 台と 8 台の両方）、`CONFIG.ja.md`（全設定）、`T4.ja.md`（4 台）、`RING.ja.md`（リングにつないだ 4 台）、`PANEL-GUIDE.ja.md`（コントロールパネルの画面）、`AGENT-API.ja.md`（ダッシュボードを使わない操作 API）、`ATLAS-TESTS.ja.md`（expert の並べ替えの測定）。ソースのリポジトリでは `RUNTIME.md` は `packaging/RUNTIME.md` で、ほかは `glm53/release/` にあります。どのページにも英語版（`.md`）と簡体字中国語版（`.zh-CN.md`）があります。

## 1. これは何か

フルサイズの GLM-5.3 モデル（755B パラメータ、アーキテクチャ `glm_moe_dsa`）を、NVIDIA DGX Spark（GB10 を 1 基ずつ）4 台または 8 台で、GB10 1 基につき 1 ランク、テンソル並列 4 または 8 で動かすサーバーです。TensorFold 1.0.2（Zig と素の CUDA カーネルで書かれたエンジン）から Fiore-lab が作った非公式の派生物であり、TensorFold プロジェクトのリリースではありません。サーバーは実行時に Python を必要としません。

| 項目 | 値 |
|---|---|
| モデル | フルサイズの GLM-5.3。Fiore-lab が Hugging Face で公開している EXL3 量子化チェックポイント |
| ノード | DGX Spark（GB10）4 台または 8 台を 1 つの RoCE v2 ネットワークに、またはスイッチなしのリングで 4 台。2 台と 3 台の構成はテスト中で、それ以外には対応しません |
| Context length | 8 台でも 4 台でも最大 1,048,576 トークン。8 台: 4 ビット KV（`KV=fp4`、`NEED_GB=100`）なら `KV_SIZE` 2,097,152 で、8 ビットなら 1,638,400。4 台: コントロールパネルで routed expert の 70 % を 3 ビット、30 % を 4 ビットにして（4 ビット KV、`KV_SIZE` 1,049,088）。既定の起動（routed expert をすべて 3 ビット）は 737,280。8 台の 4 ビット KV と、4 台のコントロールパネルからの起動は、それぞれ約 1,019,000 トークンのプロンプト 1 本で実測。4 ビット KV は精度が少し下がります（KL 0.0190、8 ビットでは 0.0156、`T4.ja.md`）。設定例は 819,200 と 737,280 です |
| API | OpenAI 互換の chat API（`/v1/chat/completions`）と Anthropic の Messages API（`/v1/messages`）。既定は全アドレスの 8000 番（`0.0.0.0:8000`） |
| リクエスト | 既定では同時に実行するのは 1 件で、さらに 32 件まで待ち行列で待ちます。`MULTI` を設定すると最大 4 つの会話をまとめて decode します |
| drafter | MTP（チェックポイント自身の層。既定）、DFlash2（任意。同梱せず、別のライセンス）、なし |
| コントロールパネル | 「Expert Atlas」: expert ごとの bit 幅（2/3/4/5 ビット）を決める Web ページ。4 台または 8 台 |
| 配布の形 | コンテナイメージ（おすすめ）と、同じファイルのアーカイブ。どちらも Linux aarch64 用 |
| ライセンス | コードは Apache-2.0。重みは GLM-5.3 License のまま。DFlash2 は独自のライセンス（CC BY-NC-ND 4.0）のまま |

## 2. 構成

| アーカイブ内のファイル・ディレクトリ | 内容 |
|---|---|
| `bin/tensorfold-native` | サーバー本体: GLM-5.3 のエンジンと HTTP API。GB10（sm_121）向けの CUDA カーネルを内蔵 |
| `lib/libroce_proxy.so` | decode の集団通信に使う RDMA プロキシ（`libibverbs.so.1` が必要） |
| `tf-glm53` | ランクの配布、確認、起動、停止、監視を行うスクリプト。1 台目のノードで実行します |
| `claude-glm53` | Claude Code をサーバーの Messages API に向けて起動します（RUNTIME.ja.md「Claude Code」） |
| `glm53.conf.example` / `glm53.t4.conf.example` | 8 台 / 4 台の設定。どちらかを `glm53.conf` にコピーします |
| `CONFIG.ja.md` | `glm53.conf` の全設定項目（既定値、読む側、変更が効く時期） |
| `launcher/tf_launcher.py` | コントロールパネルのランチャー（システムの `python3`、標準ライブラリのみ） |
| `panel.html` | コントロールパネルのページ（サーバーも同じページを持っています） |
| `tools/quant/expert_plan.py` | expert の計画（各 expert を何ビットで読むか）を書きます |
| `expert-plan/banks.json` | 各 expert バンク（1 つの幅）がノードのどこにあるか |
| `QUICKSTART.ja.md`、`BEFORE-START.ja.md`、`RUNTIME.ja.md`、`T4.ja.md`、`RING.ja.md`、`PANEL-GUIDE.ja.md`、`ATLAS-TESTS.ja.md` | 文書。どれも英語版（`.md`）と簡体字中国語版（`.zh-CN.md`）もあります |
| `AI-GUIDE.ja.md` | このページ（英語版 `.md` と簡体字中国語版 `.zh-CN.md` もあります） |
| `docker/Dockerfile`、`docker/entry.sh` | リリースをコンテナイメージにするもの（`up`、`install`、`serve`、`version`） |
| `tools/quickstart.py` | 入れたばかりの状態からエンジン起動まで（`up` と `./tf-glm53 quickstart` が使う） |
| `tools/topology.py` | ノードの配線を判定します（`./tf-glm53 topology`） |
| `tools/logkeep.py` | 各ノードのログ係。ログはメモリに置きます |
| `AGENT-API.ja.md` | ダッシュボードを使わず AI エージェントから操作する API のリファレンス（英語版 `.md` と簡体字中国語版 `.zh-CN.md` もあります） |
| `VERSION`、`SHA256SUMS`、`LICENSE`、`NOTICE`、`LICENSES/`、`THIRD_PARTY_NOTICES.md` | リリースの情報とライセンス |

アーカイブに含まれないもの: モデルの重み、expert バンク、DFlash2 の drafter、NCCL。これらは別にダウンロードします（4 節）。
重みとバンクは、Fiore-lab のファイルを使わずに、Z.AI の公式 BF16 チェックポイントから GPU のあるホストで作ることもできます。手順はソースリポジトリの `glm53/weights/README.ja.md`（<https://github.com/fiore-labo/ExpertAtlas-GLM5.3/blob/main/glm53/weights/README.ja.md>）にあり、アーカイブには入っていません。

## 3. 仕組みの要点

- **ランク。** GB10 1 基に 1 プロセス（ランク）。ランク 0 は `NODES` の 1 台目で動き、HTTP API を提供します。テンソル並列は各層をランクに分け、KV キャッシュはランクの組に分けます（decode の context 並列、`DCP=2`）。
- **prefill と decode。** プロンプトはチャンクに分けて読み（prefill）、応答は 1 トークンずつ生成します（decode）。prefill の速さは計算で、decode の速さはメモリ帯域で決まります。
- **ドラフト。** decode の各ラウンドで drafter が数トークンを提案し、モデルが 1 ステップでそれを確かめます。モデル自身が選ぶトークンだけを残すので、どちらでも応答はモデル自身のものです。
- **会話のキャッシュ。** KV キャッシュは `KV_SIZE` 位置（KV サイズ）を 256 位置ずつのブロックに分けた 1 つのプールで、すべての会話で共有します。1 件のリクエストが届くのは最大 `CONTEXT_LENGTH` 位置（context length）で、以前の会話は空いているブロックに残ります。リクエストは、どの会話のものであってもキャッシュにある最も長い先頭部分から再開し、同じように始まる会話（同じシステムプロンプト）は共通のブロックを 1 回だけ保持します。キャッシュにあるブロックは再利用の前にリクエストのトークンと照合し、少しでも疑わしければすべての会話を捨ててリクエストを最初から計算します（「cold」）。ブロックを誤って再利用することはありません。
- **expert。** GLM-5.3 は層 3〜77 のそれぞれに 256 個の routed expert（合計 19,200 個）を持ち、トークンごとに層あたり 8 個を選びます。チェックポイントはこれを量子化して持ちます: 8 台用はすべて 4 ビット、4 台用は半分（使われる回数の少ない方）を 3 ビット、残りの半分を 4 ビット。起動では expert ごとに別の幅でバンクから読むことができ、計画がなければすべての expert を 3 ビットで読みます。
- **Byte Resume。** モデルがテキストをトークナイザーと違う切り方にすることがあり、そのとき次のターンのプロンプトは同じバイトを別のトークン ID で持ちます。最初に違う ID より先は、プロンプトを保持している会話とバイトで比べ、キャッシュはその箇所を越えて再開します（`TF_GLM53_BYTE_RESUME`、既定で有効）。
- **思考。** `max_tokens` は思考も数えます。既定では、思考が持てるのは、プロンプトが残した context length か、それより小さければ `max_tokens` の 90 %（`THINKING_SHARE`）までです。この上限が近づくとモデルは思考を書き直し、書き直したものが context の中で元の思考に置き換わります。上限に達するとエンジンが思考を閉じます。上限は context が持っている量に対するものなので、1 つの応答がそれより多くの思考を書くことはあります。これらの設定はエンジンの動作中に変えられます（`POST /v1/thinking`、パネルの [ REASONING CONTROL ]）。
- **同時推論。** `MULTI` を設定すると、その数（最大 4）までの会話をまとめて decode します。その間は `POST /v1/schedule` かパネルで prefill と decode の GPU の配分を決められ、短いプロンプトを先に通せます。

## 4. 2 通りの動かし方

| | `docker run ... up` か `./tf-glm53 start` | コントロールパネル（`./tf-glm53 up`） |
|---|---|---|
| ノード | 4 台または 8 台。ノード確認が見つける | 4 台または 8 台。ノード確認が登録する（`glm53.nodes.conf`） |
| expert の幅 | すべて 3 ビット（別の幅は `BITS`。`start` は直前の起動の計画） | バンク（2、3、4、5 ビット）から expert ごとに選ぶ |
| 必要なもの | チェックポイントと 3 ビットのバンク（常に） | チェックポイント（expert 以外の重み）、3 ビットのバンク、使う幅のバンク |
| 操作 | コマンドライン | Web ページ（起動、停止、計画、稼働中の表示、エキスパート最適化）。`./tf-glm53 start` は直前の起動の計画で起動し直します |

どちらも同じ `glm53.conf`、同じサーバー、同じ API を使います。`./tf-glm53 up` はパネルのランチャーを起動してパネルのアドレスを表示し、エンジンはそこで起動します。`./tf-glm53 down` はどちらの起動でもエンジンを止めます。

人がページを操作しないままエンジンを起動する必要がある AI アシスタントは、下の 1 行か `./tf-glm53 start` を使います。`start` はパネルの「起動」と同じく直前の起動の計画で起動し（`launch/plan-*.bits`。なければすべての expert を 3 ビットで）、エンジンが動くまで待ちます。`./tf-glm53 engine-up` を手で実行しても同じです。前に `-d KIND` を付けるとその起動の drafter を選べます。

### 1 行で動かす（AI から操作する場合のおすすめ）

1 台目のノードで、動かすユーザーとして実行します（全ノードの間でパスワードなしの ssh が通り、host key を登録してあり (`QUICKSTART.ja.md` の 1 章)、イメージがノードにあるか pull できること）。

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host IMAGE up
```

リリースを `~/expertatlas` に書き、`glm53.conf` を作り、ランチャーをホストで起動し、ノード（リングかスイッチか）を見つけて確認し、重みを探し、すべての expert を 3 ビットにしてランクをコンテナで起動し、1 回推論します。このコンテナが終わってもランチャーとエンジンは動き続けます。1 段階 1 行で進み、最後は `READY`（ダッシュボード `http://ノード:8010/panel`、API `http://ノード:8000/v1`。キーなし、どちらも LAN に開く）か、`STOP`（何が足りないか。どのディレクトリに、どのリポジトリを、どのノードに）、`FIX`（することそのもの）、`THEN`（やり直す 1 行）です。起動設定はすべて `-e 名前=値` で指定できます: `WEIGHTS`、`BANK_2`〜`BANK_5`、`DRAFTER`、`BITS`、`ALLOC`、`NODES`、`PORT`、`LAUNCHER_PORT`、`EA_DIR`、`IMAGE` と、エンジン設定の全項目です（ノード確認が決める `TOPOLOGY`、`SOCKET_IFNAME`、`IB_HCA` は除きます）。一覧と初回の手順は `QUICKSTART.ja.md` にあります。その後の操作は `AGENT-API.ja.md` にあります。リリースのイメージには AArch64 の fence を入れた NCCL が入っていて、スイッチでもリングでもそれを読みます（`RING.ja.md`）。

### コンテナで動かす

`glm53.conf` に `RUNTIME=docker` と `IMAGE` を設定すると、`tf-glm53` は各ランクをリリースのイメージのコンテナとして各ノードで起動します（1 ランクに 1 コンテナ。1 台目から ssh で）。各ノードに要るのは、cuBLAS・libibverbs・NCCL の代わりに Docker と NVIDIA Container Toolkit で（DGX OS には両方あります）、下の手順 1 と 2 は要りません。各ノードでイメージを `docker pull` し（または `./tf-glm53 pull`）、1 台目で `mkdir glm53 && docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/glm53":/out IMAGE install /out` でリリースのファイルをディレクトリに書き（ディレクトリは先に作ります。マウントのために Docker が作ると root のものになり、書き込めません）、手順 4 から進めます。重みとバンクはホストに置いたまま、読み取り専用でマウントします。詳細は `RUNTIME.ja.md`「コンテナで動かす」にあります。イメージは、fence を入れた NCCL があればアーカイブから作ることもできます（`docker/Dockerfile`、`RING.ja.md`）。

### 準備（共通）

1. 全ノード: DGX OS（Ubuntu 24.04）、NVIDIA ドライバ 580 以降、cuBLAS 13（`libcublas-13-0`）、`libibverbs`。ノードは 1 つの RoCE v2 ネットワークにつなぎ、1 台目から全ノードにパスワードなしの ssh が通り、host key を登録してあること (`QUICKSTART.ja.md` の 1 章)。
2. 全ノードに NCCL 2.30.7: `pip3 install --no-deps --target ~/nccl-2.30.7 nvidia-nccl-cu13==2.30.7`（pip はライブラリを展開するだけで、Python は何も実行しません）。スイッチではこれで動きますが、AArch64 の fence を入れたビルドを推奨します。リングではそのビルドが必須です（`RING.ja.md`）。
3. チェックポイントを全ノードの同じパスに: `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks`（390 GB、8 台）または `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks`（344.6 GB、4 台）。Hugging Face の CLI で 1 回ダウンロードし、`rsync` で他のノードにコピーし、`./tf-glm53 verify` で確認します。チェックポイントは公式の BF16 チェックポイントから作ることもできます（2 節、`glm53/weights/`）。
4. 1 台目のノードでアーカイブを展開し、`sha256sum -c SHA256SUMS`、合う方の設定例を `glm53.conf` にコピーして、`NODES`、`SOCKET_IFNAME`、`IB_HCA`、`NCCL_LIB`、`WEIGHTS` を設定します。
5. `./tf-glm53 deploy`（このディレクトリを他のノードにコピー）、続いてページを使わないなら `./tf-glm53 start`（すべての expert を 3 ビットで。全ノードを確認し、ランクを起動し、サーバーを待ち、スモークテストを実行）、パネルを使うなら `./tf-glm53 up`（下）。ローカルの NVMe ドライブからの起動は約 2 分半です。
6. `./tf-glm53 down` でエンジン（ランチャーが動いていればランチャーを通して）と全ランクを止めます。

### コントロールパネルの準備

1. 計画で使う expert バンクを、全ノードの `expert-plan/banks.json` が示すディレクトリ（既定は `~/models/GLM-5.3-EXL3-Experts-<N>bit`）に置きます: 4 台では通常 `fiore-lab/GLM-5.3-EXL3-Experts-3bit` と `-4bit`（277.9 GB と 368.5 GB）。`-2bit`（187.3 GB）と `-5bit`（459.1 GB）は任意です。8 台用のチェックポイントは 4 ビットのバンクとして使えます（`RUNTIME.ja.md`「コントロールパネル（Expert Atlas）」）。`./tf-glm53 verify-banks 3 4` で確認します。
2. 1 台目のノードで `./tf-glm53 up` を実行すると、ランチャーが起動してパネルのアドレスを表示します。そのアドレス `http://FIRST-NODE:8010/panel` を開きます（`LAUNCHER_PORT` が無ければポートは `PORT` + 10）。ランチャーは既定で全アドレスで待ち受け、キーはないので、届く人なら誰でもエンジンの起動・停止・入れ替えができます（`LAUNCHER_HOST=127.0.0.1` にすると 1 台目のノードからだけになり、他のマシンからは ssh トンネル `ssh -L 8010:127.0.0.1:8010 FIRST-NODE` で開きます）。ランチャーはエンジンが動いていてもいなくても動き続け、ページはランチャーを通してエンジンに届きます。ランチャーが動いている間に起動したエンジンは、エンジン自身の `/panel` を開いたブラウザをランチャーへ送ります。エンジンが止まっている間は、編集した `glm53.conf` をランチャーが自分で読み直します。`./tf-glm53 launcher stop` と `./tf-glm53 up` が要るのは `LAUNCHER_HOST` と `LAUNCHER_PORT` だけです。初回は、ページがノード確認で開きます。ノード確認はリングかスイッチのノードを見つけて確かめ、`glm53.nodes.conf` を書きます。このファイルは `glm53.conf` の後に読まれ、`TOPOLOGY`、`NODES`、`SOCKET_IFNAME`、`IB_HCA` ではこちらが優先します。重みの場所は起動画面の [ WEIGHTS ] で、再起動が要る設定はエンジン設定の画面で決めます。
3. ページで: 計画、KV の形式、context length、各ノードが OS に残す量、drafter、同時推論の数を選び（KV サイズは計画が残す空きの全部で、KV の行に出ます）、メモリの判定を確かめ、「起動」を 2 回（起動 → 確定）押します。停止もページから行います（または `./tf-glm53 down`）。同じ計画での次の起動は `./tf-glm53 start` でもできます。

## 5. 設定（`glm53.conf`）

| 設定 | 意味 | 8 台の例 / 4 台の例 |
|---|---|---|
| `NODES` | RoCE ネットワーク上のノードのアドレス。ランク 0 が先頭（リングでは管理ネットワーク上のアドレス） | 8 / 4 個のアドレス |
| `TOPOLOGY` | `full`（スイッチ）、`ring4`、または `auto`（`check` と `./tf-glm53 up` のたびに配線から判定。既知の不具合は `CONFIG.ja.md`） | `full` |
| `SOCKET_IFNAME`、`IB_HCA`、`GID_INDEX` | 全ノードでのそのネットワークのインターフェースと RDMA デバイス、GID の番号（空なら自動選択） | 同じ |
| `NCCL_LIB` | NCCL 2.30.7 の `libnccl.so.2`。全ノードで同じパス | 同じ |
| `WEIGHTS` | チェックポイントのディレクトリ。全ノードで同じパス | 8x / 4x のチェックポイント |
| `DRAFTER_KIND`、`MTP_K`、`DRAFTER` | drafter（`mtp`、`dflash2`、`none`）、MTP の 1 ラウンドのドラフト数（3）、DFlash2 のディレクトリ | `mtp`、3、空 |
| `NAME`、`HOST`、`PORT` | API でのモデル名、待ち受けのアドレスとポート | `glm-5.3`、`0.0.0.0`、8000 |
| `METRICS_LISTEN` | `/metrics`、`/health`、`/v1/experts/usage` だけに答える、読み取り専用の追加のアドレス | 無効 |
| `LOG_LISTEN` | 各ノードのログ係が、このアドレスで自分のログに答えます（読み取り専用） | 無効 |
| `CONTEXT_LENGTH` | context length: 1 件のリクエストが届く位置の数（プロンプトと応答の合計、最大 1,048,576） | 819,200 / 737,280 |
| `KV_SIZE` | KV サイズ: KV キャッシュ全体の位置の数。すべての会話で共有する 1 つのプールで、リクエストが使わない分に以前の会話が残ります（`CONTEXT_LENGTH` + 512 以上） | 1,638,400 / 837,280 |
| `KV` | KV キャッシュの形式、`fp8` または `fp4` | `fp8` / `fp4` |
| `DCP`、`ROWS`、`SP_PARTS` | decode の context 並列、プロンプトのチャンクの行数、チャンクを分けるシーケンス並列の数 | 2、8,192、4 |
| `NEED_GB` | 起動前に各ノードで使えるべきメモリ（KV サイズに応じて変わり、DFlash2 では 3 増。計算は `CONFIG.ja.md`） | 112（既定）/ 133（112 GiB を要求） |
| `MASTER_PORT`、`READY_TIMEOUT` | ランクが落ち合うポート、サーバーを待つ秒数 | 29551、900 |
| `RUNTIME`、`IMAGE` | ランクの動かし方: `host`（既定。このディレクトリから）か `docker`（それぞれ `IMAGE` のコンテナで） | `host`、空 |
| `MULTI`、`MULTI_WAIT` | まとめて decode する会話の数（1〜4）、リクエストがその横の空きを待てる秒数 | `0`（1 件ずつ）、600 |
| `MULTI_DECODE_SHARE`、`MULTI_SHORT_PROMPT`、`MULTI_SHORT_PREEMPT` | `MULTI` のとき: prefill の横で decode に回す GPU の時間の割合（%）、先に通す短いプロンプトの長さ、それが実行中の prefill を追い越すかどうか | 30、8192、0 |
| `MAX_TOKENS_SHARE`、`MAX_TOKENS` | 応答の上限（リクエストが `max_tokens` を指定しないときの値）: `CONTEXT_LENGTH` に対する割合（%）、またはそれより優先するトークン数 | 70、空 |
| `THINKING_SHARE`、`THINKING_BUDGET` | 思考の上限: 割合（%）、またはそれより優先するトークン数 | 90、0 |
| `THINKING_FOLD`、`THINKING_FOLD_SHARE`、`THINKING_FOLD_MAX` | 中間整理: 方法（`off`、`recent`、`checkpoint`、`branch`）、間隔、回数 | `off`、100、0 |
| `THINKING_FOLD_AT_BUDGET`、`THINKING_FOLD_MARGIN` | 予算の手前の整理: するかどうかと、上限のどれだけ手前で行うか | 1、10 |
| `THINKING_FOLD_AT_CLOSE_SHARE` | 思考完了時整理: モデルが思考を閉じるときの整理。上限のこの割合（%）以上のときに行います。0 はオフ | 0 |
| `THINKING_NOTICE` | 残りのお知らせ: 思考のトークンがこの数だけ進むごとに、使った量を 1 行でモデルに知らせます。0 はオフ | 0 |
| `TF_GLM53_BYTE_RESUME` | Byte Resume: `on`、`log`、`off` | `on` |
| `ROCE_PROXY_CORES` | RoCE プロキシのスレッドがポーリングするコア: `a725`、`os` | `a725` |
| `EXTRA_ENV` | エンジンの変数（`TF_GLM53_MULTI` はここでなく `MULTI`） | 空 |
| `LAUNCHER_HOST`、`LAUNCHER_PORT`、`ATLAS_DIR`、`RESERVE_GB` | コントロールパネル: アドレス、ポート、保存した配分の置き場所、OS に残すメモリ | 4 台の例: `0.0.0.0`、8010、`atlas/`、4 |

古い `glm53.conf` もそのまま読めます: `CONTEXT` を `CONTEXT_LENGTH`、`CONTEXT` + `STORE` を `KV_SIZE` として読みます。全設定は `CONFIG.ja.md` にあります。

## 6. コマンド（`./tf-glm53`、1 台目のノードで実行）

| コマンド | 内容 |
|---|---|
| `deploy` | このディレクトリを全ノード（同じパス）にコピーし、そこで `SHA256SUMS` を確かめます |
| `check` | ノード、ファイル、重み、ライブラリ、RDMA デバイス、GPU、メモリを確かめ、問題を `NG:` の後に並べます |
| `verify` | 全ノードで重みを `SHA256SUMS` と照合します（数分） |
| `verify-banks [BITS...]` | expert バンクのファイルを manifest と照合します |
| `quickstart` | 入れたばかりの状態からエンジン起動まで。`docker run ... IMAGE up` が実行するものと同じです |
| `topology [--json]` | ノードの配線を判定します |
| `up` | コントロールパネルのランチャーを起動し、パネルのアドレスを表示します。エンジンはパネルから起動します |
| `start` | パネルの「起動」と同じく、ランチャーを通して直前の起動の計画でエンジンを起動し、動くまで待ちます |
| `down` | エンジンを止めます: ランチャーが動いていればランチャーを通して、そのあと全ランクを直接 |
| `engine-up` | 手で実行すると `start` と同じです。ランチャーの計画があれば: `check` の後、ランクを（最後から先頭へ）起動し、サーバーを待ち、スモークテストを実行します |
| `engine-down` | 全ランクを直接止めます（TERM、30 秒後に KILL） |
| `status` | ランク、空きメモリ、drafter、会話のキャッシュ、サーバーの `/health` |
| `smoke` | 答えを確かめる 3 件のリクエスト |
| `logs [R]` | ランク R のログを追います（既定は 0。`R.prev`、`watch`、`launcher` も）。ログは各ノードのメモリにあります |
| `launcher [stop\|status]` | コントロールパネルのランチャーを起動（`up` と同じ）、停止、または状態表示 |
| `-d KIND`（コマンドの前） | この起動の drafter（`start` か `engine-up`: `mtp`、`dflash2`、`none`） |
| `pull` | `RUNTIME=docker` のとき、各ノードで `docker pull IMAGE` |
| `-c FILE`（コマンドの前） | 別の設定ファイル |

`tf-glm53` は `sudo` を使わず、システムの設定を変えません。ランクの起動と停止はこれだけで行う前提です: 一部のランクが消えたときに全ランクを止める監視プロセスを残します（相手を失ったランクは、そうしないと永久に待ち続けます）。

## 7. API

| パス | メソッド | 説明 |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI chat: ストリーミング（`stream`、`stream_options.include_usage`）、ツール（`tools`、`tool_choice` は `auto` または `none`） |
| `/v1/messages`、`/v1/messages/count_tokens` | POST | Anthropic の Messages API。Claude Code 用（`RUNTIME.ja.md`「Claude Code」） |
| `/v1/responses` | POST | OpenAI の Responses API |
| `/v1/completions` | POST | プロンプトのテキスト補完（chat template は付けません） |
| `/v1/models` | GET | 提供しているモデル名 |
| `/v1/tokenize`、`/v1/detokenize` | POST | トークン数と ID。プロンプト、または chat template で展開したメッセージについて |
| `/health` | GET | 状態、メモリ、稼働中の速さ、会話のプール、トークンとドラフトのカウンタ |
| `/metrics` | GET | 同じカウンタを Prometheus のテキストで |
| `/v1/experts/...`、`/v1/memory` | GET / POST | expert の使用状況、ビット、入れ替え。コントロールパネルが使います |
| `/v1/thinking` | GET / POST | エンジンの動作中の思考の設定 |
| `/v1/schedule` | GET / POST | GPU の配分と短いプロンプト。`MULTI` のとき（なければ 404） |
| `/v1/pool` | GET | 保持している会話 |

`/v1/experts/...`、`/v1/memory`、`/v1/thinking`、`/v1/schedule`、`/v1/pool` は 1 台目のノードからの要求にだけ答えます（他からは 403）。ネットワークにはランチャーのポートが提供します。

- 応答の `message.reasoning_content` にモデルの推論、`message.content` に答えが入ります。`max_tokens` は両方を数えます。`chat_template_kwargs` は `{"enable_thinking": false}`（推論なし）と `reasoning_effort` を受け付けます。リクエスト自身の `thinking_budget`、`thinking_fold`、`thinking_fold_at_budget`、`thinking_fold_margin`、`thinking_fold_at_close`（トークン単位）は設定より優先します。`reasoning_details` は、context が持っている思考を次のターンのために運びます。
- 400 で断るもの: 1 以外の `n`、`logprobs`、`response_format` の `json_object` と `json_schema`、`guided_*`、`tool_choice` の `required` か名前を指定したツール、chat API での画像（`/v1/messages` では画像は注記になります。テキストのみ）、中間整理も予算の終わりの整理もオフのときに context の残りを超える `max_tokens` (中間整理がオンのときの条件は `CONFIG.ja.md` の `MAX_TOKENS`)。収まらないストリーミングのリクエストはエラーイベントで終わります。
- 接続を切ったクライアントのリクエストは取り消されます。
- キーはありません: API は既定で全アドレス（`HOST=0.0.0.0`）で待ち受けるので、届く人なら誰でも使えます（`HOST=127.0.0.1` にすると 1 台目のノードからだけになります）。サーバーは平文の HTTP なので、閉じたネットワーク内に留めるか前段に TLS とアクセス制御を置かない限り、文章は平文でネットワークを流れます。
- カウンタからの速さ: decode の tokens/s = Δ`completion_tokens_total` / Δ`decode_seconds_total`、prefill の tokens/s = Δ`prefill_tokens_total` / Δ`prefill_seconds_total`、ドラフトのラウンドあたりに進んだトークン = Δ`completion_tokens_total` / Δ`rounds_total`（1 ならドラフトは 1 つも残っていない）、キャッシュのヒット率 = Δ`cached_tokens_total` / Δ`prompt_tokens_total`。カウンタは実行中のリクエストの分も含み、後戻りしません。

コントロールパネルのランチャーは `/v1/launch/` の下に独自の API（status、plan、start、stop、restart、settings、weights、discover、ring、check、config、usage、resort、保存した割り当ての allocs、alloc、allocs/save、allocs/rename、allocs/delete）を持ちます。エージェントに必要な部分は `AGENT-API.ja.md` にあります。項目はリリースの間で変わることがあります。

## 8. Expert Atlas（コントロールパネル）

- **バンク**: routed expert を 1 つの幅（2、3、4、5 ビット）で持つもの。別にダウンロードし、それぞれ manifest を持ちます。ページは全ノードがそろえている幅だけを出します。
- **計画（割り当て）**: 19,200 個の expert それぞれの幅。起動画面は前回の稼働の割合の新しい計画で開きます（稼働が 1 つも無いときはすべて 3 ビット）。割合は幅ごとに入力するか、曲線上でつまみを動かして決めます。「開く」で前回の稼働の割り当てか保存した割り当てをビット単位でそのまま戻し、そこからの起動はそのビットと保存時のバンクを使います。起動前に、各ノードのメモリと判定（収まる / ぎりぎり / 入らない）を示します。
- **利用記録**: エンジンは、プロンプトの各行（prefill）と生成した各トークン（decode）が選ぶ expert を数えます。保存した割り当ては、それぞれ保存元の稼働の利用記録を持ちます。全稼働の共通の記録はありません。新しい計画 → 稼働 → 保存なら、その稼働の記録だけ。開いた割り当て → 稼働 → 保存なら、その記録にその稼働の分を足したもの。計画は元になった割り当ての記録で expert を並べます（記録の無い計画は、ページに同梱した 12 種類の作業で測った回数で）。よく使われる expert ほど広いビットになります。
- **保存**: 自動では何も保存しません。停止時の「保存して停止」は、読み込んでいるビットと利用記録に名前を付けて保存します（既定は稼働の元になった割り当てで、上書きします。前の版は `ATLAS_DIR/allocs/trash/` へ移ります）。稼働中の「保存」も、いつでも同じことをします。割り当ては `ATLAS_DIR/allocs/user/<名前>.json` のファイルで、「開く」が一覧を出し、名前の変更と削除もできます。
- **利用記録（並べる基準）**: decode、prefill、または両方。両方は、層の中での expert の decode の選択の割合を 0.7、prefill の選択の割合を 0.3 で重みづけします。prefill と decode は別の expert を選びます（層ごとの上位 64 個の重なりは 42 %。でたらめに選べば 25 %。深い層ほど重なりが少ない）。
- **エキスパート最適化（並べ替え）**: エンジンの稼働中だけ使えます。稼働の利用記録（稼働が始めに持っていた記録 + この稼働の選択）で、読み込み済みの expert の幅を付け直します。幅ごとの expert の個数はそのままなので、メモリは変わりません。実行中や待機中のリクエストが無くなってから始め、全ランクで層をその場で入れ替え、保持している会話を捨て（古い expert で計算した KV のため）、また応答を始めます。その間に届いたリクエストは待ちます（75 層全部の変更は 4 台で 83〜90 秒でした）。各ノードには、それより前の層で増えた分の上に最大の層の新しい写しと 2 GiB の空きが要り、見積もりがノードごとに足りるかを示します。足りないノードがあれば、ページはそのノードを示して変更を断ります。結果は自動では保存しません。起動は渡された幅を読むだけで、並べ替えはしません。`MULTI` を設定していると稼働中は何も変えず、見積もりの幅に名前（`opt-<時刻>`）を付けて保存するので、停止してそれで起動します。
- **並べ替えの効果**（4 台、半分 3 ビット / 半分 4 ビット、両方）: すべて 4 ビットのモデルからの KL ダイバージェンスが、応答のトークン（decode）で 8,192 トークンのとき 24 %、128K と 256K トークンのとき 32〜34 %、プロンプトのトークン（prefill）で 15〜22 % 減りました（`ATLAS-TESTS.ja.md`）。並べ替えに使った記録に無い種類の作業では、効果は小さくなります（約 3 分の 2）。
- **メモリ、4 台**（4 ビット KV、KV サイズは context length + 100,000、MTP、`ROWS` 16,384。既定の 8,192 ならもっと余ります）: すべて 3 ビットで context length 737,280 トークンが収まり、半分 3 / 半分 4 ビットで 737,280、4 分の 1 が 3 / 4 分の 3 が 4 ビットで 622,592、すべて 4 ビットで 327,680、すべて 5 ビットは入りません。`CONTEXT_LENGTH` 1,048,576・`KV_SIZE` 1,049,088 では、70 % を 3 ビット、30 % を 4 ビットにして 1 台あたり約 7.5 GiB の余裕があり、動作中の並べ替えもできます。DFlash2 では 1 台あたり約 4 GiB 多く要ります。同じメモリでは、2 ビットの expert を混ぜた配分は参照から 13〜30 % 遠くなりました。

## 9. できること（実測）

- **速さ**（リリース r2-102 と各チェックポイント自身の幅で測定し、このリリースでは測り直していません。4 台の `ROWS` は当時 16,384、今は 8,192。GPU クロックを 2.1 GHz に制限、リクエストは 1 件、まだキャッシュにないコーディング用のプロンプト、reasoning effort low。速度は環境によって違うので、数値は目安です）:

| ノード数 | プロンプト | MTP: 最初のトークンまでの時間 / decode | DFlash2: 最初のトークンまでの時間 / decode |
|---|---|---|---|
| 8 | 128K | 82.8 s / 47.5 tokens/s | 85.2 s / 53.2 |
| 8 | 256K | 172.9 s / 45.7 | 177.6 s / 52.6 |
| 8 | 512K | 376.5 s / 42.8 | 386.0 s / 47.1 |
| 4 | 128K | 120.1 s / 35.3 | 121.9 s / 42.9 |
| 4 | 256K | 253.3 s / 33.2 | 256.9 s / 40.7 |
| 4 | 512K | 565.4 s / 32.2 | 571.5 s / 37.8 |

- **キャッシュにあるプロンプト**: もう一度送った会話はプールから再開します。4 台で、703,916 トークンの会話の最初のトークンが 3 秒以内に出ました。
- **精度**（公式の BF16 モデルを公式実装で FP32 計算したものからの KL ダイバージェンス）: 8 台で 2,048 トークンの文章で約 0.013（最も確率の高いトークンの一致は約 96 %）、8,192 トークンの文章で 0.012〜0.019。4 台では同じ文章で 0.019〜0.032。routed expert をすべて 1 つの幅にしたとき（位置 6,144〜8,190）: 2 ビット 0.0517、3 ビット 0.0203、4 ビット 0.0127、5 ビット 0.0114。
- **ツールとストリーミング**: OpenAI 形式の関数呼び出しと、推論の差分つきのストリーミング応答。

## 10. できないこと

- GB10 システム 4 台または 8 台以外で動かすこと（他の GPU、他のランク数、1 台での動作には対応しません）。2 台と 3 台の構成はテスト中です。
- 既定でリクエストを並列に処理すること: 同時に実行するのは 1 件で、32 件まで待ち（チャットのタイトル作成と `"priority": "background"` のリクエストは最後）、それを超えると 503「the engine is busy; retry shortly」で断ります。サブエージェントを並列に動かすエージェントは、サブエージェントを別のモデルに向けた方がうまくいきます。`MULTI=4` を設定すると最大 4 つの会話を保ったまままとめて decode します（MTP 層、DFlash2、drafter なしのどれでも。3 台ではまだ使えません）。ノードのメモリを使い、既定では無効です。
- リクエストの実行中に expert の幅を変えること（並べ替えはリクエストの終わりを待ちます）、起動後に KV キャッシュを増やすこと（起動時に確保します）。
- 画像を読むこと（chat API は断り、`/v1/messages` は注記に変えます）、1 より大きい `n`、`logprobs`、JSON モード、ツール呼び出しの強制（`tool_choice` の `required` か名前を指定したツール）を受け付けること。
- 提供している設定で、temperature 0 の応答が実行ごとにビット単位で同じになると約束すること: いくつかのカーネルの足し合わせの順序が変わります（精度の測定には、採点用に作った決定的な設定を使っています）。
- DFlash2 を同梱すること: 独自のライセンス（CC BY-NC-ND 4.0、非商用、研究と評価用）のもとで別にダウンロードします。使い方がそのライセンスの範囲かどうかは、運用する人が判断します。

## 11. 不具合の切り分け

| 症状 | 考えられる原因 | 対処 |
|---|---|---|
| `check` が `NG: files` / `weights` を出す | あるノードでディレクトリかチェックポイントが違う | `./tf-glm53 deploy`、`./tf-glm53 verify` |
| `NG: NCCL(...)`、`cuBLAS-13`、`libibverbs` | そのノードにライブラリが無い | 入れる（`RUNTIME.ja.md`「問題が起きたとき」） |
| `NG: RoCE-v2-GID` または RDMA デバイス | ポートに IPv4 の RoCE v2 のエントリが無い、または `IB_HCA` が違う | ポートにアドレスを付けるか `GID_INDEX` を設定。`rdma link`、`ibdev2netdev` |
| `NG: GPU-busy` | 別のプロセスが GPU を使っている | それを止める（`nvidia-smi`） |
| `NG: memory(N<M GiB)` | 空きメモリが足りない | ノードのメモリを空けるか `KV_SIZE` を下げる |
| あるノードで全項目が `NG` | 1 台目からそのノードに ssh が通らない | `ssh -o BatchMode=yes ADDRESS true` が通ること |
| 起動: `a rank stopped` と `NCCL WARN` | ネットワークの名前かアドレスの誤り、ポートが塞がれている | `NODES` は RoCE ネットワークのアドレスに。`SOCKET_IFNAME`、`IB_HCA` を全ノードで正しく |
| 起動: `cannot listen on` | そのノードで `PORT` が使われている | 空けるか `PORT` を変える |
| 起動: `the smoke test failed` | 応答が質問に答えなかった | 失敗した質問が受け取った応答を示し、ランク 0 のログにリクエストが出ます |
| 起動: `not ready after N s` | ディスクやネットワークストレージが遅い | `READY_TIMEOUT` を上げる。`MASTER_PORT` が空いているか確かめる |
| `PoolValuesDiffer` | ランクが違う設定で起動された | すべてのランクを 1 つの `glm53.conf` から `tf-glm53` で起動 |
| サーバーが答えなくなり、全ランクはまだ動いている | 固まった | `./tf-glm53 down`、`./tf-glm53 start` |
| ノードのメモリが尽きた | KV サイズが大きすぎる | 各ランクは `oom_score_adj` 1000 で動くので、カーネルがそのランクを止め、監視が残りを止め、ノードは動き続けます。`KV_SIZE` を下げる |
| パネル: ある幅が「データなし」 | どこかのノードにそのバンクが無い | そこにダウンロードし、`./tf-glm53 verify-banks N` |
| ログに `PoolWriteRefused` | あるランクが、リクエスト自身のブロック以外へのプールの書き込みを拒んだ | そのリクエストの終わりで全ランクが止まります。`./tf-glm53 start` |

ログ: 各ノードのログ係（`tools/logkeep.py`）のメモリにあり、ディスクには書きません。各ランクの分（その前の起動の分は `R.prev`）、監視プロセスの分（消えていたと判断したランク）、ランチャーの分を、`tf-glm53 logs R | R.prev | watch | launcher` で読みます。`LOG_LISTEN` を設定すると、モニタが HTTP で読んで保存できます。

## 12. 間違えやすい点

- `max_tokens` は推論のトークンも数えます。指定しないリクエストには `CONTEXT_LENGTH` の 70 % が使われます。思考の上限が残りを答えのために取っておくので、`max_tokens` が小さいと答えが短くなります。`THINKING_SHARE=0` では、応答が推論の途中で終わることがあります。
- エンジンの起動時の行は、drafter にかかわらず `drafts: on` と表示します。使っている drafter は `tf-glm53 status` が示します。
- どの起動にもチェックポイントと 3 ビットのバンクが要ります。コントロールパネルでは、どちらのチェックポイントでもバンクと一緒に使えます。
- 並べ替えの測定は、BF16 モデルではなく、すべて 4 ビットのモデルに対するものです。`ATLAS-TESTS.ja.md` にどちらかを書いています。
- chat template の user のターンの中では、モデルは利用者自身の文章をうまく予測しません。これはプロンプトに対する KL ダイバージェンスのように利用者の文章を予測するときの話で、応答の話ではありません。`ATLAS-TESTS.ja.md` の測定は、プロンプトのトークンを template の冒頭を除いて採点しています（「測り方」）。
- プールの会話のキャッシュには、`CONTEXT_LENGTH` + 512 以上の `KV_SIZE` が必要です。
- コンテナには GPU、`/dev/infiniband`、ホストネットワーク、`--ulimit memlock=-1 --ulimit nofile=1048576:1048576` が必要です。
- `/v1/thinking` と `/v1/schedule` は 1 台目のノードからの要求にだけ答えます。別のマシンからはランチャーのポートを通します。
- `TOPOLOGY`、`NODES`、`SOCKET_IFNAME`、`IB_HCA` では、`glm53.nodes.conf` が `glm53.conf` より優先します。
- API は既定で全アドレス（`HOST=0.0.0.0`）で待ち受け、キーはないので、届く人なら誰でも使えます（`HOST=127.0.0.1` にすると 1 台目のノードからだけになります）。コントロールパネルのランチャーも同じです（`LAUNCHER_HOST`）。
