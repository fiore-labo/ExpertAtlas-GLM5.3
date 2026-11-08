[English](QUICKSTART.md) | 日本語 | [简体中文](QUICKSTART.zh-CN.md)

# 初めての起動: イメージからエンジンが動くまで

このページは、Expert Atlas を初めて立ち上げる人のためのものです。Expert Atlas は、フルサイズの GLM-5.3 を NVIDIA DGX Spark で動かします。ウェイトをダウンロードし、1 台目のノードで `docker run` を 1 行実行すると、エンジンが起動します。足りないものがあれば、同じ 1 行が何をすればよいかを表示します。

## どこで何が動くか

- **エンジン**は、各ノードのコンテナで 1 つずつ動きます。`docker run ... up` が起動します。
- **ランチャー**は、1 台目のノードのホストで動きます (コンテナの外)。ダッシュボードと操作 API を提供し、エンジンの起動と停止を行います。
- **`up` のコンテナ**は準備をするだけで、エンジンが動くと終わります。エンジンとランチャーは動き続けます。
- **ウェイト**はホストに置きます。イメージには入っていません。

ダッシュボードにも API にもキーはなく、どちらも 1 台目のノードの全アドレスで待ち受けます。ノードは閉じたネットワークに置いてください (1 台目だけに絞る方法は 4 章にあります)。

## 1. ノードに必要なもの

| 項目 | 内容 |
|---|---|
| ノード | 1 つの RoCE v2 スイッチにつないだ DGX Spark 4 台または 8 台、またはスイッチなしのリングで 4 台 (`RING.ja.md`)。2 台と 3 台はテスト中です |
| 各ノードのソフトウェア | NVIDIA ドライバ 580 以降 (イメージはホストのドライバを使います。`nvidia-smi` で確かめられます)、Docker と NVIDIA Container Toolkit (DGX OS には両方あります)、`python3` |
| ConnectX-7 のネットワーク | つないだポートの半分ごとにアドレス、MTU 9000 (下記) |
| ssh | 1 台目から全ノードへ、自分自身へも。パスフレーズのない鍵で、host key を登録しておきます (下記) |
| Docker の権限 | そのユーザーが全ノードで `docker` を実行できること (`docker` グループ) |
| メモリ | 起動前に各ノードで約 115 GiB の空き。GPU で他のものが動いていないこと |
| ディスク | 必須のダウンロード 2 つに、各ノードで 299.1 GB 以上（4 台でも 8 台でも） |

リリースのイメージには AArch64 の fence を入れた NCCL が入っていて、スイッチでもリングでもそれを読みます (`RING.ja.md`「fence 入りの NCCL」)。

### ConnectX-7 のネットワーク

ConnectX-7 の 1 ポートは PCIe の半分 2 つで、半分ごとにネットワークデバイスがあり、アドレスも別々です。ポート 0 は `enp1s0f0np0` と `enP2p1s0f0np0`、ポート 1 は `enp1s0f1np1` と `enP2p1s0f1np1` です。半分ごとに別々の /24 と MTU 9000 を設定します。

スイッチでは、全ノードで、つないだポートの半分 2 つにアドレスを付けます。/24 は全ノードで同じ 2 つにし、最後の数字をノードごとに変えます。ポート 1 の場合の、1 台分の例です (アドレスは例です):

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
ip -br addr    # 半分 2 つが UP で、アドレスが付いている
rdma link      # 半分 2 つが ACTIVE
```

リングでは半分 4 つすべてにアドレスを付け、全ノードに `lldpd` も要ります。`RING.ja.md`「配線と番地」にあります。

これに要るカーネルの設定 (`arp_ignore=1`、`arp_announce=2`、`rp_filter=2`) は DGX OS に入っているので、手で設定する必要はありません。

### ssh

ランチャーはパスフレーズのない鍵で接続し、知らない host key を自分では足しません。そのため、1 台目は全ノードの host key を、自分のものも含めて先に知っている必要があります。1 台目で:

```sh
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
for a in NODE1 NODE2 NODE3 NODE4; do    # 全ノードのアドレス。1 台目自身も
  ssh-copy-id -o StrictHostKeyChecking=accept-new "$a"    # パスワードを聞かれます
  ssh -o BatchMode=yes "$a" true && echo "$a OK"
done
```

`NODE1` などには、ランチャーが使うアドレスを書きます。スイッチでは各ノードの ConnectX-7 の `enp1s0f…` の半分のアドレス (上の例の `172.16.88.N`) です。1 台目はこの /24 の ssh のポートを順に試して、他のノードを見つけます。リングでは各ノードの LAN のアドレスです。LAN は既定ルートのインターフェースです。管理用のネットワークがそれと別なら、ノードの設定を手で書きます (`RING.ja.md`「設定」)。

## 2. ウェイトをダウンロードする

**必須は、Base の重みと 3 ビットのバンクの 2 つです。** 別の幅を使う予定でも、この 2 つがないとエンジンは起動しません。パスは全ノードで同じにします。

| 種類 | Hugging Face のリポジトリ | 大きさ | 既定のディレクトリ | オプション |
|---|---|---|---|---|
| Base の重み (**必須**) | `fiore-lab/GLM-5.3-EXL3-Base` | 21.2 GB | `~/models/GLM-5.3-EXL3-Base` | `WEIGHTS` |
| 3 ビットのバンク (**必須**) | `fiore-lab/GLM-5.3-EXL3-Experts-3bit` | 277.9 GB | `~/models/GLM-5.3-EXL3-Experts-3bit` | `BANK_3` |
| 2 ビットのバンク (任意) | `fiore-lab/GLM-5.3-EXL3-Experts-2bit` | 187.3 GB | `~/models/GLM-5.3-EXL3-Experts-2bit` | `BANK_2` |
| 4 ビットのバンク (任意) | `fiore-lab/GLM-5.3-EXL3-Experts-4bit` | 368.5 GB | `~/models/GLM-5.3-EXL3-Experts-4bit` | `BANK_4` |
| 5 ビットのバンク (任意) | `fiore-lab/GLM-5.3-EXL3-Experts-5bit` | 459.1 GB | `~/models/GLM-5.3-EXL3-Experts-5bit` | `BANK_5` |
| 完全なチェックポイント、4 台 (任意) | `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks` | 344.6 GB | なし | `WEIGHTS` |
| 完全なチェックポイント、8 台 (任意) | `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks` | 389.9 GB | なし | `WEIGHTS` |
| DFlash2 (任意、別ライセンス) | `incoai/GLM-5.3-DFlash2` (`RUNTIME.ja.md` の DFlash2 の節) | | なし | `DRAFTER` |

Base の重みは routed expert 以外の全部で、routed expert は起動のときにバンクから読みます。完全なチェックポイントには routed expert も入っていますが、起動には要りません。すでに持っていれば `WEIGHTS` に指定すると、そのまま使えます。8 台用は 4 ビットのバンクも兼ねられます（`RUNTIME.ja.md`「コントロールパネル」）。

1 回ダウンロードして、他のノードにコピーします。`python3 -m venv` が失敗するときは、`python3-venv` パッケージを入れます。4 台の場合:

```sh
python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install -U huggingface_hub
for r in EXL3-Base EXL3-Experts-3bit; do
  D=~/models/GLM-5.3-$r
  ~/hf-cli/bin/hf download fiore-lab/GLM-5.3-$r --local-dir $D
  for n in NODE2 NODE3 NODE4; do    # 他の 3 台
    rsync -a --mkpath -e "ssh -c aes128-gcm@openssh.com" $D/ $n:$D/ &
  done; wait
done
```

ウェイトが既に別の場所にあるなら、動かさずにオプションでディレクトリを指定できます (4 章)。

### 幅とは

モデルの routed expert は、2、3、4、5 ビットのどれかで読み込めます。**既定の起動は、すべての expert を 3 ビットで読み込みます。** 4 台でも、既定の context length 1,048,576 トークンが収まります。`BITS` を指定すると、すべての expert を別の幅で起動できます。ダッシュボードでは、持っているバンクから expert ごとに幅を選べます。

## 3. 起動する

1 台目のノードで、動かすユーザーとして実行します。

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host IMAGE up
```

`IMAGE` はリリースのイメージ `ghcr.io/fiore-labo/expertatlas-glm5.3:r3.3.4-102` です。リリースを `~/expertatlas` に書き、ノード (リングかスイッチか) を見つけて確認し、ウェイトを探し、エンジンを起動して、1 回推論します。初回は数分かかります。

最後は次のどちらかになります。

**`READY`: エンジンが動いています。**

```
READY
  inference API       http://203.0.113.10:8000/v1 (OpenAI compatible, no key, model glm-5.3)
  dashboard (no key)  http://203.0.113.10:8010/panel
  control API         http://203.0.113.10:8010/v1/launch/status
```

**`STOP`: 対応が必要です。** `STOP` に原因、`FIX` にすること、`THEN` に次に実行する 1 行が出ます。ランチャーは動いたままなので、ダッシュボードも開けます。

```
STOP: the model's files are not on the nodes:
  3 bit (always needed): ~/models/GLM-5.3-EXL3-Experts-3bit: not on node-1, node-2 (...)
    put fiore-lab/GLM-5.3-EXL3-Experts-3bit there, the same path on every node (...)
FIX:  ...
  hf download fiore-lab/GLM-5.3-EXL3-Experts-3bit --local-dir ~/models/GLM-5.3-EXL3-Experts-3bit
THEN: docker run --rm --network host -e USER="$USER" -v "$HOME":/host IMAGE up
```

`FIX` のとおりにしてから、`THEN` の行を実行します。もう一度実行しても問題はありません。動いているエンジンを起動し直すことはありません。`FIX` の行の `hf` は Hugging Face CLI で、2 章の `~/hf-cli/bin/hf` のことです。

## 4. オプション

**起動設定で選べるものは、すべて `docker run` のオプションで指定できます。** `IMAGE` の前に `-e 名前=値` を足します。知らない名前があると止まるので、打ち間違いに気づかないまま進むことはありません。

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host \
  -e WEIGHTS=/data/models/base \
  -e BANK_3=/data/models/3bit \
  -e BANK_4=/data/models/4bit \
  -e BITS=4 -e CONTEXT_LENGTH=131072 \
  IMAGE up
```

ファイルの場所 (全ノードで同じパス):

| オプション | 既定 | 意味 |
|---|---|---|
| `WEIGHTS` | `~/models/GLM-5.3-EXL3-Base` | Base の重み、または完全なチェックポイント。必須 |
| `BANK_3` | `~/models/GLM-5.3-EXL3-Experts-3bit` | 3 ビットのバンク。必須 |
| `BANK_2`、`BANK_4`、`BANK_5` | `~/models/GLM-5.3-EXL3-Experts-<N>bit` | 他のバンク。任意 |
| `DRAFTER` | 空 | DFlash2 のモデル。`DRAFTER_KIND=dflash2` と一緒に使います |

起動:

| オプション | 既定 | 意味 |
|---|---|---|
| `BITS` | `3` | すべての expert をこの幅 (2〜5) で。そのバンクが全ノードに必要です |
| `ALLOC` | なし | 保存した割り当て (expert ごとの幅と利用記録)。`BITS` の代わりに使います |
| `NODES` | 自動で見つける | スイッチで使うワーカーの住所。空きノードがレーンの台数より多い時に指定します。リングは自分で見つけます |
| `CONTEXT_LENGTH` | `1048576` | prompt と応答を合わせた最長。長すぎると収まるまで切り詰め、そのことを表示します |
| `KV_SIZE` | `max` | KV キャッシュが持つトークンの総数。数か、残りのメモリ全部を使う `max`。この起動だけに効きます |
| `KV` | `fp4` | KV キャッシュの形式。`fp4` か `fp8` |
| `DRAFTER_KIND` | `mtp` | `mtp`、`dflash2`、`none` |
| `MULTI`、`MULTI_WAIT` | `0`、`600` | 同時に推論する会話の数 (0〜4) と、要求がそれを待つ秒数 |
| `RESERVE_GB` | `4` | 各ノードがシステムのために残すメモリ |

エンジン設定の他の項目も、同じように指定できます。たとえば `MAX_TOKENS`、`THINKING_SHARE`、`THINKING_FOLD_PROMPT`、`NAME`、`HOST`、`STRUCTURED` (`0` で構造化出力をオフ)、`DETERMINISTIC` (`1` で起動し直しても同じ答え) です。各項目の意味と、オプションにできない項目は `CONFIG.ja.md` にあります。そこの既定は `tf-glm53` のものなので、この行ではここの表の値が優先です。

インストール:

| オプション | 既定 | 意味 |
|---|---|---|
| `EA_DIR` | `expertatlas` | ホームの下のディレクトリ |
| `PORT` | `8000` | 推論 API |
| `LAUNCHER_PORT` | `PORT` + 10 | ダッシュボードと操作 API |
| `IMAGE` | このイメージ | ノードが pull して動かす名前 |

知っておくこと:

- 指定した値は `~/expertatlas/glm53.conf` に保存され、別の値を指定するまで残ります。`KV_SIZE` と `BITS` はその起動だけに効きます。幅は残りません。`BITS` も `ALLOC` もなければ、ダッシュボードで選んだ幅に関係なく、すべての expert を 3 ビットで起動します。
- `TOPOLOGY`、`SOCKET_IFNAME`、`IB_HCA` はオプションではありません。ノード確認が配線から決めます。
- オプションは動いているエンジンを変えません。先に止めてから実行します。無視したオプションがあれば、そう表示します。
- API を 1 台目だけに絞るには `-e HOST=127.0.0.1` を足します。ダッシュボードと操作 API の `LAUNCHER_HOST` はオプションではないので、`~/expertatlas/glm53.conf` に `LAUNCHER_HOST=127.0.0.1` と書き、ランチャーを止めて (6 章)、もう一度この行を実行します。

## 5. 使う

API は OpenAI のチャット形式です。`READY` に出た住所をクライアントに指定します。

```sh
curl -s http://NODE:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3","messages":[{"role":"user","content":"hello"}]}'
```

応答の思考は `reasoning_content`、答えは `content` に入ります (ストリームでは `delta.reasoning_content` と `delta.content`)。

JSON で答えさせるなら `"response_format": {"type": "json_object"}` を、スキーマを決めるなら `json_schema` を足します。答えはその形に保たれます。思考あり (既定) では `max_tokens` が 545 以上要ります。短い返事なら `"chat_template_kwargs": {"enable_thinking": false}` も足してください。詳しくは `FEATURES.ja.md` の「構造化出力」にあります。

ダッシュボード `http://NODE:8010/panel` では、エンジンの状態を見て、起動と停止ができます。AI エージェントは同じことを操作 API でできます。手順は `AGENT-API.ja.md`、システム全体の説明は `AI-GUIDE.ja.md` にあります。

- `SCREENS.ja.md`: ダッシュボードの画面を、画像つきで 1 つずつ説明します。
- `FEATURES.ja.md`: 機能ごとに、何をするか、なぜあるか、どう使うかを説明します。

スイッチに他の人のノードがあるときは、起動の行で自分のノードを指定してください (`-e NODES=...`)。ダッシュボードのノード確認では、先に台数とチェックを選んでから押します (`PANEL-GUIDE.ja.md` の「ノード確認」)。

## 6. 止める、もう一度起動する

```sh
cd ~/expertatlas
./tf-glm53 down              # エンジンを止める
./tf-glm53 launcher stop     # ランチャーも止めたい時
```

もう一度起動するには、同じ `docker run` の行を実行します。ノード、ウェイトの場所、設定は覚えていますが、ダッシュボードで選んだ幅は覚えていません。この行はすべての expert を `BITS` (既定は 3) で起動します。前回の計画で起動するには、代わりに `./tf-glm53 start` を実行するか、保存した割り当てを `-e ALLOC=NAME` で指定します。

## 7. よくある STOP

| メッセージ | すること |
|---|---|
| `up needs the host's home at /host` | 行をそのまま使います: `-e USER="$USER" -v "$HOME":/host` |
| `this container cannot ssh to the host` | パスフレーズのない鍵を作り、自分の `authorized_keys` に足します。コマンドはメッセージにあります |
| `IMAGE is not on NODE and docker pull failed` | メッセージの `docker save \| ssh NODE docker load` の行でコピーします |
| `3 bit (always needed)`、`checkpoint (always needed)` | ダウンロードするか (2 章)、`BANK_3` や `WEIGHTS` でディレクトリを指定します |
| `no other node answers over ConnectX-7` | ケーブルとアドレスを確かめます (1 章) |
| `its host key is not known here`、`it refuses this node's ssh key` | そのアドレスについて 1 章の ssh の手順を行います |
| `N free worker nodes found` | `-e NODES="ADDR ADDR ADDR"` で選びます |
| `no ring found`、`the ring is open` | 全ノードに `lldpd` が要ります (1 章)。ケーブルと ssh も確かめます |
| `its LAN address is not known here` | 1 台目からそのノードの LAN のアドレスへ ping してから、もう一度実行します。それでも止まるなら、ノードの設定を手で書きます (`RING.ja.md`「設定」) |
| `port N answers another lane's launcher` | そのランチャーを止めるか、`-e PORT=N` を足します |
| `does not fit with context length N` | `CONTEXT_LENGTH` を小さく、`BITS` を狭くするか、ノードのメモリを空けます |
| `not an option` | 綴りを確かめます。近い名前をメッセージが示します |

詳しくは `BEFORE-START.ja.md` (起動前に決めること)、`RUNTIME.ja.md` (全コマンドと障害)、`T4.ja.md` (4 台と実測)、`RING.ja.md` (リング)、`CONFIG.ja.md` (全設定) にあります。ソースのリポジトリでは、`RUNTIME.ja.md` は `packaging/RUNTIME.ja.md` に、ほかのページは `glm53/release/` にあります。
