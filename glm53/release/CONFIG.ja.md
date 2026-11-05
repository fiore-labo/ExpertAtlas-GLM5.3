[English](CONFIG.md) | 日本語 | [简体中文](CONFIG.zh-CN.md)

# glm53.conf の設定項目

`docker run ... IMAGE up` では、エンジン設定の項目を `-e 項目=値` で指定できます (`QUICKSTART.ja.md`)。次の項目は `not an option` で止まるので、`~/expertatlas/glm53.conf` に書きます: `GID_INDEX`、`RUNTIME`、`NCCL_LIB`、`EXPERT_LIVE`、`LOG_LISTEN`、`CONTEXT`、`STORE`、`THINKING_FOLD_KEEP`、`NEED_GB`、`LAUNCHER_HOST`、`ATLAS_DIR`、`LAUNCH_SNAPSHOTS`。`TOPOLOGY`、`SOCKET_IFNAME`、`IB_HCA` はノード確認が決めます。

下の既定は `tf-glm53` の既定です。`docker run ... IMAGE up` は台数によらず `glm53.t4.conf.example` から `glm53.conf` を書くので、`CONTEXT_LENGTH` 1048576、`KV_SIZE` 1049088、`KV` fp4、`NEED_GB` 115 で始まり、起動の KV サイズは `max` です。

`glm53.conf` には、レーンの設定（ノード、ネットワーク、重み、drafter、サーバー、メモリ、コントロールパネル）を書きます。シェルのファイルで、1 台目のノードの `tf-glm53` と同じディレクトリに置き、手で書いて編集します。`glm53.t4.conf.example`（4 台）か `glm53.conf.example`（8 台）を `glm53.conf` にコピーし、ノード、ネットワークのデバイス、パスを書き入れてください。それ以外の項目は、例に実測した値が入っています。

## 誰が読むか、変更がいつ効くか

- `tf-glm53` は、コマンドを実行するたびに読みます。読むのは 1 台目のノードのファイルだけです。`tf-glm53 deploy` はこのディレクトリを `glm53.conf` を除いて他のノードにコピーし、各ランクは起動するときに 1 台目から設定を受け取ります。
- 動いているエンジンは、起動したときの設定のままです。変更がエンジンに届くのは次の起動からです（コントロールパネルで「停止」のあと「起動」。または `./tf-glm53 down` のあと `./tf-glm53 start`）。
- コントロールパネルのランチャーは、ランチャーを起動したときに読みます。エンジンが止まっている間にファイルが変わると、ランチャーが自分で読み直します（エンジンが動いている間の編集は、止まったときに読みます）。`LAUNCHER_HOST` と `LAUNCHER_PORT` は、ランチャーを起動し直したときに効きます: `./tf-glm53 launcher stop` のあと `./tf-glm53 up`。その間エンジンは動いたままでかまいません。ランチャーは動いているエンジンを見つけます。
- ほかに書き換えるものはありません。例外は 2 つで、どちらもコントロールパネルです。どちらも変えた行だけを書き（1 項目 1 行。値の後ろのコメントは残し、行が無ければ末尾に足す）、前のファイルを `glm53.conf.bak-<時刻>` として横に残します。
  - 起動設定で、ウェイトの「保存」を押すと `WEIGHTS` と `DRAFTER` を書きます。書くのは、指す場所が全ノードにあると確かめてからです（expert のバンクのディレクトリは同じように `expert-plan/banks.json` に書き、全ノードに配ります）。ランチャーはそれをすぐに読み直します。
  - エンジン設定の画面（見出しの切り替えで開く）は、再起動が必要な設定を書きます: `NAME`, `CONTEXT_LENGTH`, `KV_SIZE`, `KV`, `MAX_TOKENS`, `THINKING_*`, `MULTI*`, `DRAFTER_KIND`, `MTP_K`, `DRAFT_SAVES`, `DCP`, `MASTER_PORT`, `RESERVE_GB`, `READY_TIMEOUT`, `LAUNCH_DROP_CACHES`, `HOST`, `PORT`, `METRICS_LISTEN`, `LOG_KEEP_MB`, `LAUNCHER_PORT`, `ROWS`, `SP_PARTS`, `ROCE_PROXY_CORES`, `ENGINE_CORES`, `ATLAS_GENERIC`, `EXTRA_ENV`（エンジンの `TF_GLM53_CANON_CLOSE` と `TF_GLM53_CANON_NEAR` はその中の 1 語として）。ノード確認でノードを登録するまでは `TOPOLOGY`、`SOCKET_IFNAME`、`IB_HCA` も。値は先に `tf-glm53` と同じ規則で確かめます。`PORT` はエンジンが止まっている間だけ。変更はその時点で書き、次の起動から使います（動いているエンジンは起動したときの設定のまま）。ただし思考の動作中に効く項目（`THINKING_SHARE`、`THINKING_FOLD`、`THINKING_FOLD_AT_CLOSE_SHARE`、`THINKING_FOLD_SHARE`）は、動いているエンジンにもすぐ渡し、推論画面の [ THINKING ] で変えた値もここに書くので、両方の画面が同じ値になります。「再起動して反映」（エンジンが動いているとき）: 「停止」と同じく、この稼働の利用記録を残すかを聞いてから止め、同じ expert のまま、保存した設定で起動し直します。
- コントロールパネルのノード確認で登録したノードは、別のファイル `glm53.nodes.conf`（このファイルの横）に書きます。中身は `TOPOLOGY`、`NODES`、`SOCKET_IFNAME`、`IB_HCA` で、確認で配線を判定したとおりの値です（前のファイルは `glm53.nodes.conf.bak-<時刻>` として残ります）。`tf-glm53` とランチャーは `glm53.conf` のあとに読むので、この 4 項目はそちらが優先します。このファイルがない間、パネルはノード確認から開きます。できたあとは起動設定から開きます。`NODES_CONF=FILE` で別のファイルを、`NODES_CONF=`（空）でどれも読みません。`tf-glm53 deploy` はこのファイルを配りません。
- 起動画面で選ぶもの（KV の形式、context length、drafter、同時推論の数、各ノードが OS に残す量）は、選んだ時点でこのファイルに書きます（`KV`、`CONTEXT_LENGTH`、`DRAFTER_KIND`、`MULTI`、`RESERVE_GB`）。起動のときは計画が取った context length を書くので、次に開くとその値が選ばれています。`KV_SIZE` は context length に足りないときだけ上げます。expert の幅と計画が取る KV はその起動だけのもので、ランチャーが環境変数（`LAUNCH_*`）で `tf-glm53` に渡します。このファイルの設定項目ではありません。
- ページ自身の設定（色、円の並べ方など）は `launch/panel-settings.json` に残ります。
- `glm53.conf` はリリースのファイルではありません。`SHA256SUMS` に載っていないので、`tf-glm53 check` も `tf-glm53 deploy` も照合しません。大きく変える前に、動いているファイルの控えを取ってください。
- 別の設定ファイル: `tf-glm53 -c FILE コマンド`（ランチャーも `./tf-glm53 -c FILE up`）。同じノードで 2 つのレーンを動かすなら、`PORT`、`MASTER_PORT`、`LAUNCHER_PORT` をそれぞれ別にします。

## 設定項目

「既定」は、項目が無いか空のときに `tf-glm53` が使う値です。「（必須）」には既定がありません。**パネル** と書いた項目は起動画面でも選べ、選ぶとここに書きます。起動画面を通らない起動（`tf-glm53 start`、`engine-up`）はファイルの値を使います。

### ノードとネットワーク

| 項目 | 既定 | 意味 |
|---|---|---|
| `NODES` | （必須） | ノードをランク順に、RDMA ネットワークのアドレスで、シェルの配列として書きます: `NODES=(192.0.2.11 192.0.2.12 192.0.2.13 192.0.2.14)`。4 台か 8 台です (2 台と 3 台はテスト中)。`tf-glm53` は 1 台目で実行します。各ノードは他のノードにパスワードなしの ssh で入れること。コントロールパネルのノード確認は、ノードをここではなく `glm53.nodes.conf` に登録します。そちらがこの項目と次の 3 項目より優先します。 |
| `SOCKET_IFNAME` | （必須） | 各ノードでのそのネットワークのインターフェース（`ip -br addr`）。例: `enp1s0f1np1`。 |
| `IB_HCA` | （必須） | そのインターフェースの RDMA デバイス（`ibv_devices`）。例: `rocep1s0f1`。 |
| `TOPOLOGY` | `full` | ノードのつなぎ方。`full`: スイッチ、または 1 本の線。`ring4`: スイッチなしでリングにつないだ 4 台。NODES はリングの順（RING.ja.md）。`auto`: `check` と `./tf-glm53 up` のたびに、各ノードの RDMA デバイスと LLDP の隣から判定します（判定は `tf-glm53 topology` で見られます）。リングなら `ring4` で、NODES をリングの順に並べ替えます（ランク 0 はそのまま）。スイッチか 1 本の線なら `full` です。`IB_HCA` と `SOCKET_IFNAME` は判定のものを使うので、書かなくてかまいません（`IB_HCA` はスイッチのとき、どちらの半分を使うかの希望になります）。どちらでもないつなぎ方は、直すところを示して断ります。黙って別の形に切り替えることはしません。**既知の不具合:** `TOPOLOGY=auto` で `IB_HCA` と `SOCKET_IFNAME` を書かず、まだ判定がない状態で、最初の起動を `./tf-glm53 start` か `engine-up` で行うと、`NG: auto RoCE-v2-GID(gid=)` で止まります。この 2 つのコマンドは判定せず、`check` は判定しても保存せず、保存するのは `./tf-glm53 up` だけだからです。先に `./tf-glm53 up` を実行してください。この 2 つの項目を書いてあればこの `NG` にはなりませんが、判定がないまま `full` として起動するので、リングでは使えません。`docker run ... IMAGE up` と `TOPOLOGY=full`、`ring4` には影響しません。 |
| `GID_INDEX` | 空 | RoCE v2 の GID の番号。空なら各ノードで探します（IPv4 アドレスを持つ最後の RoCE v2 の項目）。 |
| `MASTER_PORT` | `29551` | NCCL のランデブーに使う 1 台目のポート。 |

### ランクの動かし方

| 項目 | 既定 | 意味 |
|---|---|---|
| `RUNTIME` | `host` | `host`: このディレクトリから動かします。各ノードに cuBLAS 13、libibverbs、NCCL が要ります。`docker`: 各ランクを `IMAGE` のコンテナで動かします（RUNTIME.ja.md「コンテナで動かす」）。 |
| `IMAGE` | 空 | `RUNTIME=docker` のとき、このリリースのイメージ。 |
| `NCCL_LIB` | `~/nccl-2.30.7/nvidia/nccl/lib/libnccl.so.2` | 各ノードのこのパスにある NCCL 2.30.7（`RUNTIME=host` のときだけ。RUNTIME.ja.md「初回起動の前に」）。スイッチでは NVIDIA の wheel でも動きますが、AArch64 の fence を入れたビルドを推奨します。リングではそのビルドが必須です（RING.ja.md）。 |

### 重みと drafter

| 項目 | 既定 | 意味 |
|---|---|---|
| `WEIGHTS` | （必須） | Base の重み (`GLM-5.3-EXL3-Base`) のディレクトリ。全ノードで同じパスにします (4 台でも 8 台でも)。完全なチェックポイント (`GLM-5.3-EXL3-4x-DGX-Sparks`、`GLM-5.3-EXL3-8x-DGX-Sparks`) でも使えます。起動設定で指定できます。expert のバンクのディレクトリ（`expert-plan/banks.json`）も同じ画面で指定できます。どちらも全ノードにあることを確かめてから書きます。 |
| `DRAFTER_KIND` | `mtp` | **パネル**。モデルが確かめるトークンを何が下書きするか: `mtp`（`WEIGHTS` の MTP 層）、`dflash2`（`DRAFTER` のモデル）、`none`。`./tf-glm53 -d KIND start`（または `engine-up`）でその起動だけ別のものにできます。 |
| `MTP_K` | `3` | MTP の 1 ラウンドの下書きの数。1〜7。 |
| `DRAFTER` | 空 | DFlash2 のモデルのディレクトリ。全ノードで同じパスにします。`DRAFTER_KIND=dflash2` にはこれが要ります。起動画面が DFlash2 を選ばせるのも、これが設定され、全ノードにモデルがあるときだけです。リリースには含まれません（RUNTIME.ja.md「DFlash2 を使う手順」）。起動設定で指定できます。 |
| `DRAFT_SAVES` | `8` | DFlash2 のとき、リクエストの間で drafter の状態を残す会話の数。 |

### サーバー

| 項目 | 既定 | 意味 |
|---|---|---|
| `NAME` | `glm-5.3` | API でのモデルの名前。`/v1/models` に載り、応答に付きます（リクエストはどの名前を指定してもかまいません）。 |
| `HOST` | `0.0.0.0` | API が待ち受けるアドレス。`0.0.0.0` は全アドレスです。API にキーはないので、届く人なら誰でも使えます。閉じたネットワーク内に留めるか、前段に TLS とアクセス制御を置いてください。`127.0.0.1` にすると 1 台目のノードからだけになり、他のマシンからは ssh トンネルで使います。 |
| `PORT` | `8000` | API のポート。 |
| `METRICS_LISTEN` | 空 | `HOST:PORT`。1 台目のノードがそのアドレスでも `GET /metrics`（Prometheus のテキスト）、`GET /health`、`GET /v1/experts/usage`（expert の使用回数）に読み取り専用で答えます。別のホストのモニタ用です。空なら使いません。 |
| `EXPERT_LIVE` | `1` | `1`: 1 台目のノードが、プロンプトの各行と decode の各トークンが選ぶ expert を、コントロールパネルのために数えます（実測の表示、利用記録、エキスパート最適化）。`0`: 数えません。パネルは動いているエンジンと「停止」を出しますが、選択の表示と最適化はありません。パネルからの起動はいつも数えます。 |
| `LOG_LISTEN` | 空 | `HOST:PORT`。各ノードのログ係が、読み取り専用・キーなしで `GET /streams` と `GET /logs?stream=S&after=N` にも答えます。別のホストのモニタがログを保存するためです。空なら、ログは各ノードのメモリにだけあります（`tf-glm53 logs`）。変更は次のエンジンの起動で反映します。そのときログ係を起動し直すので、それまでの行は消えます。 |
| `LOG_KEEP_MB` | `8` | ログ（各ランク、監視プロセス、ランチャー）1 本ごとにメモリに持つ MB。古い行から消えます。ディスクには何も書きません。変更の反映は `LOG_LISTEN` と同じです。 |

### Context、キャッシュ、KV

| 項目 | 既定 | 意味 |
|---|---|---|
| `CONTEXT_LENGTH` | `819200` | **パネル**。1 つのリクエストが届く位置の数（プロンプト + 応答）。65,536〜1,048,576（モデルの上限）で 16,384 の倍数。 |
| `KV_SIZE` | `1638400` | **パネル**。KV 全体の位置の数で、全会話が共有します。`CONTEXT_LENGTH` + 512 以上。パネルで設定すると (ここか起動画面の KV サイズ)、パネルからの起動はこの値を使います。起動画面で OS に残す量を入力すると、KV はまた計画の残りを全部使います。`glm53.conf` だけで起動するときもこの値です。 |
| `CONTEXT`、`STORE` | | `CONTEXT_LENGTH` と `KV_SIZE` の前の名前。新しい名前が無いときはこれを読みます: `CONTEXT_LENGTH` = `CONTEXT`、`KV_SIZE` = `CONTEXT` + `STORE`（`STORE` が無ければ 819,200）。エンジン設定の画面で保存すると、新しい名前で書き、この行は消します。 |
| `KV` | `fp8` | **パネル**。KV キャッシュの形式: `fp8`、または `fp4`（1 位置あたりのメモリが少なく、精度が少し下がります。T4.ja.md）。 |
| `DCP` | `2` | decode の context 並列: `2` は KV をペアのランクで分けます。 |
| `MULTI` | `0` | 1 ラウンドに 1 つのウィンドウでまとめて decode する会話の最大数（1〜4）。`0` はリクエストを 1 件ずつ処理します。MTP 層、DFlash2（`DRAFT_SAVES` が 1 以上）、drafter なしのどれでも使えます。3 台ではまだ使えません（起動前に断ります）。エキスパート最適化は `0` のときと同じで、変更は推論中と待ちの要求がなくなるまで待ちます。 |
| `MULTI_WAIT` | `600` | `MULTI` のとき、処理中の会話の横に会話キャッシュの空きができるのをリクエストが待てる秒数。超えると混雑として断り、クライアントが送り直せるようにします: HTTP 503（Messages エンドポイントでは `overloaded_error`）。すでに始まったストリームでは、種類が `overloaded_error`（Messages）か、`server_error` でコード `overloaded`（Chat Completions）のエラーのイベント、またはコード `rate_limit_exceeded` の失敗した応答（Responses）です。待ち時間は、空きがなくて最初に入れなかった時から数え、decode のラウンドと prompt のチャンクの合間に判定するので、断るのが 1 ラウンド遅れることがあります。待っている間に取り消したリクエストは、取り消しとして終わります。`0` は上限なし。 |
| `MULTI_DECODE_SHARE` | `30` | `MULTI` のとき、decode 中の会話の横で prompt が待っているか計算されている間の、GPU の時間のうち decode に回す割合（%）。`0` は prompt を先にします（prefill の途中に decode を挟まず、待っているリクエストを続けて受け入れます）。`100` は decode を先にします（どれかの会話が decode している間は prompt を待たせ、実行中の prefill も会話が終わるまで止めます。短い prompt は通します。`MULTI_SHORT_PROMPT` を参照）。その間の値では、エンジンが両方の時間を測ってその割合に保ちます。片方の仕事しかないときは、その仕事が GPU を全部使います。これは起動時の値で、動作中は `POST /v1/schedule {"decode_share": N}`（API キー、または先頭ノードのクライアントから）で変えられ、実行中の prefill にも数層のうちに効きます。`GET /v1/schedule` と `/health`（`multi.decode_share`、`multi.measured`）で見られます。取り分による待ちには `MULTI_WAIT` の上限がかかりません（`MULTI_WAIT` は空きを待つ時間です）。 |
| `MULTI_SHORT_PROMPT` | `8192` | `MULTI` のとき、計算するトークン（会話キャッシュで再利用できる分より後ろ。長い会話の次のターンは新しく足した分だけを数えます）がこれ以下の prompt を短いものとして扱います。短い prompt は、待っている長い prompt より先に入り、`MULTI_DECODE_SHARE` で待たされず、prefill を decode のラウンドを挟まずに終わりまで進めます（取り分にかかわらず、その間 GPU は prompt のものです。decode 中の会話はその間待ちます。4 台で 8,192 トークンなら約 7 秒）。実行中の prefill は、`MULTI_SHORT_PREEMPT=1` でなければ終わるまで待ちます。`0` は短いものなし（待ち行列の順のまま）。これは起動時の値で、動作中は `POST /v1/schedule {"short_prompt": N}` で変えられ、次の受け入れから効きます。`GET /v1/schedule` と `/health`（`multi.short_prompt`）で見られます。 |
| `MULTI_SHORT_PREEMPT` | `0` | `MULTI` と `MULTI_SHORT_PROMPT` のとき、`1` にすると、実行中の長い prefill も、待っている短い prompt に譲ります。譲るのは次に decode のために止まる所で、ほかに decode している会話がなければ次のチャンクです。終わったチャンクはキャッシュに残り、ストリームを開いたまま、あとでそこから再開します。止まる所で手放したチャンクの分は計算し直します（最大 1 チャンク。4 台・取り分 30 で長い prompt が約 14 秒遅れます）。再開した後は、1 チャンク終えるまで次は譲りません。`0`（既定）は、短い prompt が実行中の prefill の終わりを待ちます。動作中は `POST /v1/schedule {"short_preempt": true}` で変えられ、`GET /v1/schedule` と `/health`（`multi.short_preempt`）で見られます。 |
| `ROWS` | `8192` | プロンプトのチャンクの行数: 256〜65,536、256 刻み。増やすとメモリを多く使います（1,024 行あたり 1 ノード約 0.41 GiB、4 台で測定）。起動の計画はその分を KV から差し引くので、KV が小さくなります。 |
| `SP_PARTS` | `4` | プロンプトのチャンクを分ける系列並列の数: 2〜4。 |
| `ROCE_PROXY_CORES` | `a725` | RDMA (RoCE) の通信スレッド（proxy、1 ランクに 2 本: ノード間のものと DCP の組のもの）が待つコア。decode の間ずっと回ります。`a725` は GB10 の Cortex-A725 のコア 0 と 10 に留め（クラスタに 1 本ずつ）、`os` はスケジューラに任せます（X925 に載ることが多い）。4 台で、A725 はスケジューラ任せより SoC が 5〜16 °C 低く、decode の速さは同じでした。その種類のコアがない CPU ではスケジューラに任せます。どこで待つかはエンジンのログに出て、`/health` の `roce` で全ランクの分が見られます（rank 0 の分は今の値）。 |
| `ENGINE_CORES` | `all` | 全ランクを動かすコア。`all` はノードのすべてのコア、`a725` は各ノードの Cortex-A725 のコアだけです。`a725` と `ROCE_PROXY_CORES=x925` は一緒に使えず、`tf-glm53` が断ります。A725 だけで動かしたときの速さは未測定です。 |
| `MAX_TOKENS_SHARE` | `70` | 要求が max_tokens を指定しないときの応答の max_tokens を、`CONTEXT_LENGTH` の割合 (1〜100 %) で決めます。クライアントの `max_tokens` が優先します。 |
| `MAX_TOKENS` | (空) | 同じものをトークン数で指定します。値があれば `MAX_TOKENS_SHARE` より優先し、パネルで `MAX_TOKENS_SHARE` を保存すると空になります。中間整理も予算の終わりの整理もオフのとき、context の残りより多くを求める要求は断ります (400)。中間整理がオンのときは、エンジンが間隔を空きに合わせます。要求が自分で `thinking_fold_window` を指定し、それが収まらないときは断ります (400)。 |
| `THINKING_SHARE` | `90` | **動作中も**。 応答の max_tokens のうち思考に使ってよい割合 (%。ただし prompt が `CONTEXT_LENGTH` に残す分までの割合。`0` は上限なし)。残りは答えの分です。思考がそのトークン数に届くと、次の改行で (遅くとも 256 トークン後)、エンジンが締めの一文と `</think>` を書き、残りで答えが続きます。`THINKING_FOLD` のときは閉じずに続けます: 思考が context のその割合を埋めたら畳み、予算は max_tokens から答えの分を引いた長さです。要求の `thinking_budget` (Anthropic では `thinking.budget_tokens`) が優先です。 |
| `THINKING_BUDGET` | `0` | 1 回の応答でモデルが考えてよい思考のトークン数です。`THINKING_FOLD` を使うと、たたんだ思考は context から外れるので、`CONTEXT_LENGTH` より大きくできます (仮想の長さ)。予算に達すると、エンジンが改行のところで思考を締め、答えに進ませます。`0` は `THINKING_SHARE` で決めます。要求の `thinking_budget` が優先します。 |
| `THINKING_NOTICE` | `0` | 思考がこのトークン数だけ進むごとに (`256` 以上。`0` はなし)、次の改行で、使った思考のトークン数と残り (予算の。予算がなければ max_tokens の) を伝える一行を、エンジンが思考の中に書きます。その行は応答の思考 (`reasoning_content`) の一部です。1 回ごとに短い受け入れ直しがかかります (キャッシュはその行以外を再開します)。 |
| `THINKING_FOLD` | `off` | **動作中も**。 長い思考をモデルの context の中で畳み、KV が増え続けないようにします (クライアントには思考がすべて届きます)。`recent` は思考の最初の 100 トークンと最後の `THINKING_FOLD_KEEP` を残します。`checkpoint` はまずモデルにチェックポイントを書かせ (ここまでの結論、うまくいかなかった方針、まだ決まっていないこと、次に調べること。`END CHECKPOINT` で終わり、最大 `THINKING_FOLD_CAP` トークン)、それがそれより前の思考に代わります。モデルはまだ決まっていないことだけを考え続けます。`branch` は応答を脇に置き、応答の横の答え (思考を閉じ、ユーザーの番、思考なしの答え) で、前の要約以降の思考を要約させます。答えは EOS で必ず終わります (思考の中で書かせるチェックポイントは `END CHECKPOINT` をほとんど書きません)。要約は思考の中に「(My reasoning grew long, so I summarized it once ...)」の見出しで積み上がり、クライアントにも思考として届きます。畳むのは、思考が prompt の残す context の `THINKING_SHARE` (`THINKING_SHARE=0` なら全部) から要約の分を引いた所を埋めた時で、その後は context を越えて考え続けます。要求の `thinking_fold`・`thinking_fold_window`・`thinking_fold_max`・`thinking_fold_keep` でも指定できます。 |
| `THINKING_FOLD_SHARE` | `100` | **動作中も**。`THINKING_FOLD` のとき: 要求ごとの max_tokens のこの % に届いたら、それより遅くならずに畳みます (1024 トークン以上、10 から `THINKING_SHARE` 未満)。`100` は上限まで整理しません。要求の `thinking_fold_window` が先です。 |
| `THINKING_FOLD_MAX` | `0` | **動作中も**。`THINKING_FOLD` のとき: 1 つの応答で畳む回数の上限 (`0` は制限なし)。使い切った後、または積んだ要約ともう 1 つの要約が context の思考の割合を越えるとき (要約はそこに積み上がる) は、もう畳みません: 思考はその割合を埋めるまで続き、埋めたら予算の終わりと同じに閉じます (`THINKING_FOLD_AT_CLOSE_SHARE` があれば締めの文の代わりに要約)。思考の予算の残りに中間整理と思考完了時整理が入らないときは、畳む所で閉じます。要求の `thinking_fold_max` が先です。 |
| `THINKING_FOLD_KEEP` | `0` | `THINKING_FOLD=recent` のとき: 残す最後の思考のトークン数 (`0` は窓の半分)。 |
| `THINKING_FOLD_AT_CLOSE_SHARE` | `0` | **動作中も**。 思考完了時整理: モデルが思考を閉じた時、context の思考 (最後に畳んでから) が思考の上限のこの割合以上なら (`0` はしない、1〜100 %。要求の `thinking_fold_at_close` はトークン数で指定し、こちらが優先)、答える前に推論で得たことのチェックポイント (確かめた事実と決めた結果、条件つきの途中の結果、退けた考えとその理由、自分で直した誤り、まだ確かめていない仮説、答えの計画、書くときに気をつけること。試行錯誤や言い換えは残しません。要求の `thinking_fold_cap` が無ければ最大 `THINKING_FOLD_CAP` トークン) を書かせ、それで思考を置き換えてから答えを書かせます。次のターンにも要約が渡ります。思考の予算でエンジンが思考を閉じる時も、締めの文の代わりに同じ要約を書かせます (このときは、どこで思考が止まったかと、決めきれなかった点ごとに、その時点の最善の答え・確からしさ・答えでの扱いも書かせます)。要約とそれと同じ長さの答えが応答の `max_tokens` に収まる時だけ求めます。`0` は使いません。`THINKING_FOLD=off` でも使えます。要求の `thinking_fold_at_close` が優先します。 |
| `THINKING_FOLD_AT_BUDGET` | `1` | **動作中も**。思考の予算の終わり近くで思考を整理します。context の思考を書き直しに置き換え、閉じるかどうかはその後モデルが決めます。`0` はオフ。要求の `thinking_fold_at_budget` が優先します。 |
| `THINKING_FOLD_MARGIN` | `10` | **動作中も**。思考の予算の終わりのこの % 手前で整理を始めます (0〜50)。 |
| `THINKING_FOLD_PROMPT` | org の文面 | **動作中も**。その整理の指示文です。空にすると既定に戻ります。 |
| `THINKING_CLOSE_PROMPT` | org の文面 | **動作中も**。思考完了時整理の指示文です。空にすると既定に戻ります。 |
| `TF_GLM53_BYTE_RESUME` | `on` | モデルはトークナイザーと違う切り方をすることがあり、その時は次のターンの prompt が同じバイトを別の id で持つので、キャッシュがそこで止まります。`on` は、最初に違う id より先を保存した会話とバイトで比べ、バイトが一致する所はその id を使うので、キャッシュはその先から再開します。文は変わりません。キャッシュなしで計算した応答とはわずかに違うことがあります。`off` はキャッシュがそこで止まります。`log` は `off` と同じで、比べて記録します (`tf-glm53 logs 0` の `byteresume:`)。 |
| `THINKING_FOLD_CAP_SHARE` | `70` | 要約 1 回の上限を、整理する思考の割合で決めます (10〜100 %、128 トークン以上)。`THINKING_FOLD=checkpoint` のチェックポイントと `THINKING_FOLD_AT_CLOSE_SHARE` の要約に効きます。要求の `thinking_fold_cap` (トークン数、128〜16384) が優先します。 |

### メモリと起動

| 項目 | 既定 | 意味 |
|---|---|---|
| `NEED_GB` | `112` | 起動の前に各ノードで使えなければならない GiB（`tf-glm53 check`）。DFlash2 では 3 多く、`KV_SIZE` が 1,638,400 より 100,000 位置少ないごとに、`KV=fp8` で 2.55、`KV=fp4` で 1.73 少なくなります。4 台の例は 115 で、その `KV_SIZE` で 105 になります。パネルからの起動は、計画の値を代わりに使います。 |
| `READY_TIMEOUT` | `900` | 起動（`tf-glm53 engine-up`。パネルからの起動もこれを実行します）がサーバーを待つ秒数。 |

### コントロールパネル

| 項目 | 既定 | 意味 |
|---|---|---|
| `LAUNCHER_HOST` | `0.0.0.0` | パネルが待ち受けるアドレス。キーはないので、届く人なら誰でもエンジンの起動・停止・変更ができます。`127.0.0.1` にすると 1 台目のノードからだけになり、他のマシンからは ssh トンネルで開きます（`ssh -L 8010:127.0.0.1:8010 FIRST-NODE`）。 |
| `LAUNCHER_PORT` | `PORT` + 10 | パネルのポート（4 台の例では 8010）。`http://FIRST-NODE:LAUNCHER_PORT/panel` で開きます。 |
| `RESERVE_GB` | `4` | **パネル**。パネルから起動したあと、各ノードが OS に残す GiB。2〜32。 |
| `ATLAS_DIR` | このディレクトリの `atlas/` | 保存した割り当ての置き場所（`allocs/user/<名前>.json`）。 |
| `ATLAS_GENERIC` | `1` | `1`: パネルからの起動で全層を汎用の grouped カーネルで動かし、エキスパート最適化がどの expert でも動かせるようにします。 |
| `LAUNCH_SNAPSHOTS` | `0` | `1`: 起動と停止の前後に、各ノードの GPU、メモリ、ログの末尾を `snapshots/` に書きます。 |
| `LAUNCH_DROP_CACHES` | `0` | `1`: 起動の前に全ノードでページキャッシュを捨てます（パスワードなしの `sudo` が要ります）。 |

### ランクに渡す環境変数

| 項目 | 既定 | 意味 |
|---|---|---|
| `EXTRA_ENV` | 空 | 全ランクに渡す環境変数（`名前=値` を空白で区切り、引用符で囲みます）。まとめて decode する会話の数はここではなく `MULTI` で指定します（`TF_GLM53_MULTI` を書くと `tf-glm53` が断ります）。ランチャーも、いくつかを自分で読みます。`TF_PANEL_HTML`（別のページを出す）と `TF_PANEL_SETTINGS`（ページの設定の別のファイル）などです。 |
