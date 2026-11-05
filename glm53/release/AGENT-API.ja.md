[English](AGENT-API.md) | 日本語 | [简体中文](AGENT-API.zh-CN.md)

# ExpertAtlas エージェント操作 API リファレンス

- 読者: ダッシュボードを使わずに ExpertAtlas を操作する AI エージェントと、その人
- 範囲: 状態の確認、サーバー設定、エンジン設定、起動、推論、キャッシュの確認、終了
- 範囲外: エキスパート最適化 (Optimise、Apply、`/v1/launch/resort`、`/v1/experts/swap`)

この文書は、エージェントが手順どおりに操作するために必要な分をまとめたものです。設定の全項目は `CONFIG.ja.md` にあります。

## 0. イメージから起動まで

1 台目のノードで、動かすユーザーとして実行します。

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host IMAGE up
```

リリースを `~/expertatlas` に書き、ランチャーをホストで起動し、ノードの登録、重みの確認、計画、起動、推論 1 回まで進めます。このコンテナが終わっても、ランチャーとエンジンは動き続けます。**チェックポイントと 3 ビットのバンクは常に必要で、指定がなければすべての expert を 3 ビットで起動します。**

- `READY` で始まる行が出れば起動しています。続く行にダッシュボードと API の住所があります。
- `STOP` なら止まった理由です。`FIX` に直し方、`THEN` にやり直す 1 行があります。直してから `THEN` をそのまま実行します。ランチャーは動いたままなので、この文書の API も使えます。
- 起動設定はすべて `-e 名前=値` で指定できます: `WEIGHTS`、`BANK_2`〜`BANK_5`、`DRAFTER`、`BITS`、`ALLOC`、`NODES`、`PORT`、`LAUNCHER_PORT`、`EA_DIR`、`IMAGE` と、4 章のエンジン設定です (`TOPOLOGY`、`SOCKET_IFNAME`、`IB_HCA` はノード確認が決めるので指定できません)。知らない名前は `STOP` になります。一覧は `QUICKSTART.ja.md` にあります。
- ホストに要るもの: NVIDIA ドライバ 580 以降、Docker と NVIDIA container toolkit、`python3`、ConnectX-7 のネットワークのアドレス、全ノード間と自分自身へのパスワードなしの ssh と登録済みの host key (`QUICKSTART.ja.md` の 1 章)。
- リリースのイメージには AArch64 の fence を入れた NCCL が入っていて、スイッチでもリングでも読みます (`RING.ja.md`)。
- 既に動いていれば何も起動せず、推論を 1 回して `READY` を出します。

## 1. 最初に知ること

### 1.1 ポートは 2 つ

| 役割 | 設定の鍵 | 既定 | 使い方 |
|---|---|---|---|
| エンジン | `PORT` | 8000 | 推論はここへ。止まっている時は応答しない |
| ランチャー | `LAUNCHER_PORT` | `PORT` + 10 | 操作はすべてここへ。エンジンが止まっていても応答する |

- ランチャーは `./tf-glm53 launcher start` で一度起動しておきます。API では起動できません。`./tf-glm53 launcher status` が状態と URL を表示します。
- `/v1/thinking`、`/v1/schedule`、`/v1/pool`、`/v1/experts/*` は、LAN からエンジンの port を直接呼ぶと 403 になります。**ランチャーの port で呼びます。** ランチャーは 127.0.0.1 からエンジンへ中継します。
- ランチャーに鍵はありません。閉じたネットワークの中でだけ使います。

### 1.2 操作の前に必ず確かめること

1. **そのポートが目的のクラスタか。** 同じポートを持つ別のディレクトリがありえます。`GET /v1/launch/config` の `conf_path` が設定ファイルの絶対パスです (`GET /v1/launch/settings` の `conf` はファイル名だけです)。ノードの上でも次で確かめられます。
   ```bash
   ss -ltnp | grep ":$PORT "          # pid を読む
   readlink /proc/$PID/cwd             # そのレーンのディレクトリ
   ```
   ノードは `GET /v1/launch/config` の `nodes` と `names` で確かめます。登録の直後は `names` が空で、数秒で戻ります。
2. **推論が走っていないか。** `GET /health` の `live.connections` と `progress.phase` を見ます。
3. **止める前に保存したか。** 保存せずに止めると、その稼働の利用記録を失います (7 章)。

### 1.3 してはいけないこと

- 人に頼まれていない本番のエンジンを止める、再起動する。
- 自分自身を動かしているエンジンを止める、再起動する。道具を実行している間は `live.connections` が 0 に見えるので、空いているように見えても止めてはいけません。
- `GET /v1/launch/status`、`GET /v1/launch/usage`、`GET /v1/launch/alloc`、`GET /v1/pool` をそのまま context に入れる。base64 が数万〜数十万文字あります (`alloc` の実測 435,780 文字)。必要な項目だけ取り出します (1.5)。

### 1.4 非同期の操作

時間のかかる操作は、POST がすぐ答えて (202 か 200)、処理は裏で続きます。接続が切れても止まりません。GET で状態を読みます。

| POST | 読む GET | 完了の印 | 実測 |
|---|---|---|---|
| `/v1/launch/start` | `/v1/launch/status` | `state` が `running` か `failed` | 100 秒 (重みがメモリにある時)。読み込むと 10 分以上 |
| `/v1/launch/stop` | `/v1/launch/status` | `state` が `stopped` か `failed` | 6 秒 |
| `/v1/launch/restart` | `/v1/launch/status` | `restart` が false で `state` が `running`、または `restart` が false で `error` あり | 起動と同じ |
| `/v1/launch/check` | `/v1/launch/check` | `state` が `done` | 2〜4 秒 (`deploy: false`) |
| `/v1/launch/discover` | `/v1/launch/discover` | `state` が `done` | 1〜3 秒 |
| `/v1/launch/weights/check` | `/v1/launch/weights` の `check` | `check.state` が `done` | 5 秒未満 |
| `/v1/launch/inference-check` | `/v1/launch/inference-check` | `state` が `done` | 約 3 秒。長い推論の後ろでは最大 600 秒 |

読む間隔は 2〜10 秒で足ります。

### 1.5 道具

例はすべてこの関数を使います。`LP` はランチャー、`EP` はエンジンの port です。

```python
import base64, json, time, urllib.error, urllib.request

def call(port, method, path, body=None, host="127.0.0.1", timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://{host}:{port}{path}", data,
                                 {"Content-Type": "application/json"}, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}
    except urllib.error.URLError as e:      # 止まっているエンジンなど
        return 0, {"error": str(e.reason)}

def brief(status):
    """status から base64 を除いたもの"""
    return {k: v for k, v in status.items() if k not in ("done", "experts_done", "bits", "banks", "weights")}

def stream(port, body, host="127.0.0.1", timeout=7200):
    """SSE を読む。(思考, 答え, usage, reasoning_details, finish_reason)"""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"}, method="POST")
    think, text, usage, details, finish = [], [], {}, None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            for ch in d.get("choices") or []:
                de = ch.get("delta") or {}
                think.append(de.get("reasoning_content") or "")
                text.append(de.get("content") or "")
                details = de.get("reasoning_details") or details
                finish = ch.get("finish_reason") or finish
    return "".join(think), "".join(text), usage, details, finish
```

ノードのログは `./tf-glm53 logs 0` ですが、**追従したまま終わりません。** 必ず `timeout 10 ./tf-glm53 logs 0 | grep ...` の形で、レーンのディレクトリで読みます。

## 2. 状態を読む

| 知りたいこと | 呼び方 | 見る項目 |
|---|---|---|
| 動いているか、起動の進み具合 | `GET /v1/launch/status` | `state` (stopped, starting, running, stopping, failed)、`phase`、`layers_done`、`eta_ms`、`error`、`alloc` |
| 今の起動の選択 | `GET /v1/launch/status` | `kv.format`、`kv.context`、`kv.store`、`kv.reserve_gib`、`drafter.running`、`multi.running` |
| どのノードか | `GET /v1/launch/config` | `nodes`、`names`、`world`、`topology`、`registered` |
| 各ノードのメモリと温度 | `GET /v1/launch/nodes` | `nodes[].mem_available`、`engine`、`other`、`gpu_temp_c` |
| 使える幅とドラフター | `GET /v1/launch/status` | `banks.available`、`banks.checking`、`drafter.available`、`drafter.why` |
| エンジンの状況 | `GET /health` (エンジンの port) | `model`、`context_length`、`live`、`progress`、`thinking` |
| 思考の設定 | `GET /v1/thinking` | 5.3 |
| 保存した割り当て | `GET /v1/launch/allocs` | `allocs[].name`、`valid`、`problem`、`usage` |

- **動作中の値と設定ファイルの値は別です。** `GET /v1/launch/settings` は設定ファイルの値です。起動の時に選択を渡すと、`pending` が空でも値が違うことがあります。動作中の値は `status` の `kv.*` と `/health` の `context_length` を見ます。
- 止めた後も、`status` の `kv`、`drafter.running`、`multi.running` は前回の値のままです。
- エンジンが止まっている時、`/v1/thinking` と `/v1/schedule` はランチャー経由で 503 です。設定は `settings` で読みます。

## 3. サーバー設定 (ノードと重み)

ノードの登録と重みの場所です。**どちらもエンジンが止まっている時だけ変更できます。**

### 3.1 ノードの確認と登録

```python
call(LP, "POST", "/v1/launch/discover", {})     # 202。読むだけ、ConnectX-7 の先だけ
s, d = call(LP, "GET", "/v1/launch/discover")   # state が done になるまで読む
# d["kind"]: switch | ring | direct | none
```

`head` は **ランチャーのノード自身の IP アドレス** です。名前は 400 になります。

**スイッチの場合:**

```python
head = d["fabric"]["head_addr"]                 # rank 0 の、ファブリック側の住所
hosts = [h for h in d["hosts"] if h["access"] == "ok" and h["gb10"] and not h["other"]]
# hosts[] にはスイッチ (gb10 が null) や、他のレーンが使っているノード (other に名前) も入る。head は入らない
# in_lane はそのレーンの今の NODES に入っているか
workers = [h["addr"] for h in hosts][:3]         # 4 台なら 3 つ。どれを選ぶかは人に確認する
call(LP, "POST", "/v1/launch/check", {"head": head, "workers": workers, "deploy": False})
```

**リングの場合:** `hosts` は空です。ケーブルをたどって残りの 3 台を見つけます。`workers` を省くと設定ファイルの `NODES` を調べるので、新しいレーンでは必ずこの順にします。

```python
call(LP, "POST", "/v1/launch/ring", {})          # 202
r = call(LP, "GET", "/v1/launch/ring")[1]        # state が done になるまで読む
# r["ring"]: closed (4 台の輪) | open | none、r["nodes"]: head が先頭の順、r["problems"]: 届かないノードなど
assert r["ring"] == "closed", r["problems"]
head = d["head"]["lan"]["addr"]                  # 管理 LAN の住所
call(LP, "POST", "/v1/launch/check", {"head": head, "workers": r["nodes"][1:]})
```

**確認の結果** (`GET /v1/launch/check`。`state` が `done` になるまで読む):

- `nodes_ok`: 登録できるか。false なら登録できません。
- `workers`: 確認したワーカー。登録にはこれをそのまま渡します。
- `topology`: 判定された配線。`mode` は `full` (スイッチ) か `ring4`、`order` はノードの順、`hca` と `ifname` は登録される値、`why` は理由。
- `nodes[].items[]`: 項目ごとの結果。`key`、`state` (ok, warn, fail)、`value`、`fix` (直し方)。
- `deploy` を省くか true にすると、確認の前にパッケージを各ノードへ複写します。`false` で確認できるのは、既にパッケージのあるノードだけです。`glm53.conf`、`*.nodes.conf`、`logs` は複写しません。ノードに既にパッケージがあるなら `false` で足ります。

```python
c = call(LP, "GET", "/v1/launch/check")[1]
call(LP, "POST", "/v1/launch/config", {"head": head, "workers": c["workers"]})
# glm53.nodes.conf に TOPOLOGY、NODES、SOCKET_IFNAME、IB_HCA が書かれ、以後これが優先される
```

- 確認していない組み合わせは 412、エンジンが動いていると 409 です。
- リングのノードの順は `topology.order` に従って登録されます。自分で並べ替えません。
- ノード数は 4 か 8 です。2 台と 3 台の構成はテスト中です (`worlds` には 2 も入ります)。リング (`ring4`) は 4 台だけです。

### 3.2 重みの場所

```python
s, w = call(LP, "GET", "/v1/launch/weights")
# w["weights"], w["drafter"], w["drafter_kind"], w["banks"] = [{"id", "bits", "dir"}, ...]

body = {"weights": w["weights"], "drafter": w["drafter"],
        "banks": [{"id": b["id"], "dir": b["dir"]} for b in w["banks"]]}
call(LP, "POST", "/v1/launch/weights/check", body)   # 202
while (c := call(LP, "GET", "/v1/launch/weights")[1]["check"])["state"] != "done":
    time.sleep(2)
# c: state, ok, why, items[] (key, state, value, fix)

if c["ok"]:
    call(LP, "POST", "/v1/launch/weights", body)     # 確認と同じ値で保存。止まっている時だけ
# 答えの copied に、ワーカーごとの banks.json の複写の結果
```

- `banks` には `banks.json` の全エントリを 1 回ずつ渡します。
- 確認していない値と、確認が終わる前の保存は 412 です。
- `dir` が空の幅は使いません。
- 保存でバンクの `id` が変わることがあります。次に呼ぶ前に読み直します。
- 複写に失敗しても保存は 200 です。`copied[].ok` を確かめます。
- 先頭ノードのディレクトリは `GET /v1/launch/browse?path=/data/models` で一覧できます。`~`、`/opt`、`/mnt`、`/media`、`/data`、`/srv` の下だけで、ほかは 400 です。
- データのない幅は warn になり、起動の計画で選べません。**3 ビットだけは fail になります。** 3 ビットのバンクがないと、場所の保存も起動もできません。`GET /v1/launch/status` の `banks.required` と `GET /v1/launch/weights` の `banks[].required` が必須の幅を示します。
- **保存の直後は `banks.available` が空です。** `GET /v1/launch/status` の `banks.checking` が false になるまで待ちます。

## 4. エンジン設定

### 4.1 読み書き

```python
s, v = call(LP, "GET", "/v1/launch/settings")
# v["conf"], v["state"], v["pending"], v["keys"] = [{key, value, default, set, from, when, rule}, ...]

s, r = call(LP, "POST", "/v1/launch/settings", {"values": {"THINKING_SHARE": "90"}})
# r["changed"]: 書いた鍵、r["kept"]: 複写した元のファイル、r.get("applied"): 動いているエンジンにも効かせた鍵
# r.get("apply_error"): エンジンが断った理由 (この時も設定ファイルには書かれている)
```

- 400 の時は `errors` に鍵ごとの理由が入り、何も書かれません。
- `when` が `stopped` の鍵を動作中に変えると 409 です。
- `when` が変更の効く時です。

| `when` | 意味 |
|---|---|
| `live` | 即時反映。動いているエンジンにも届く |
| `start` | 次回起動時反映。動作中に変えると `pending` に載る |
| `stopped` | 止まっている時だけ変更できる (`PORT`) |
| `launcher` | ランチャーの再起動後 (`LAUNCHER_PORT`) |
| `readonly` | ここでは変えられない。ノードは 3.1、重みは 3.2 |

- `from` が値の出どころです。`node_file` の鍵はノードの登録で変わります。
- `glm53.conf` は変更の前に `glm53.conf.bak-<時刻>` へ複写されます。**時刻は秒単位で、同じ秒の変更は上書きされます。** 元に戻す時に使うなら、変更の間を 1 秒以上空けます。
- API にない鍵 (`RUNTIME`、`IMAGE`、`NCCL_LIB`、`NCCL_SHA256`、`GID_INDEX`、`NEED_GB`、`LOG_LISTEN` など) は `glm53.conf` を直接編集します。人に確認してから行います。

### 4.2 主な鍵

全部の鍵と規則は `GET /v1/launch/settings` が返します。説明は `CONFIG.ja.md` にあります。

下の表の既定は `tf-glm53` の既定です。`docker run ... IMAGE up` は `glm53.t4.conf.example` から `glm53.conf` を書くので、`CONTEXT_LENGTH` 1048576、`KV_SIZE` 1049088、`KV` fp4、`LAUNCHER_PORT` 8010 で始まり、起動の KV サイズは `max` です。

| 鍵 | 既定 | when | 意味 |
|---|---|---|---|
| `CONTEXT_LENGTH` | 819200 | start | 1 回の最長。新しい値は 16,384 の倍数、65,536 以上、1,048,576 以下 |
| `KV_SIZE` | 1638400 | start | KV 全体。`CONTEXT_LENGTH` + 512 以上 |
| `KV` | fp8 | start | KV の形式。fp4 か fp8 |
| `DRAFTER_KIND` | mtp | start | mtp、dflash2、none |
| `MULTI` | 0 | start | 同時に推論する数。0〜4 |
| `RESERVE_GB` | 4 | start | OS に残すメモリ |
| `MAX_TOKENS_SHARE` | 70 | start | 要求が指定しない時の `max_tokens`。`CONTEXT_LENGTH` の %。1〜100 |
| `MAX_TOKENS` | (なし) | start | 同じものをトークン数で。値があれば優先。`MAX_TOKENS_SHARE` を保存すると空になる |
| `THINKING_SHARE` | 90 | live | 思考の予算の割合 (5.1)。0 は予算なし |
| `THINKING_BUDGET` | 0 | start | 同じものをトークン数で。0 は `THINKING_SHARE` が決める |
| `THINKING_NOTICE` | 0 | start | 残りのお知らせの間隔 (思考のトークン数)。0 はオフ |
| `THINKING_FOLD_AT_BUDGET` | 1 | live | 予算の手前で整理するか。0 か 1 |
| `THINKING_FOLD_MARGIN` | 10 | live | 整理の開始を予算の何 % 手前にするか。0〜50 |
| `THINKING_FOLD_PROMPT` | org の文 | live | 整理で求める文 |
| `THINKING_FOLD_AT_CLOSE_SHARE` | 0 | live | モデルが閉じた時に整理する思考の最小量 (思考の上限の %)。0 はオフ |
| `THINKING_CLOSE_PROMPT` | org の文 | live | モデルが閉じた時の整理で求める文 |
| `THINKING_FOLD` | off | live | 中間整理。off、checkpoint、recent、branch |
| `THINKING_FOLD_SHARE` | 100 | live | 中間整理の間隔。`max_tokens` の %。100 は上限まで整理しない |
| `THINKING_FOLD_MAX` | 0 | live | 1 回の応答の中間整理の回数の上限。0 は制限なし |
| `THINKING_FOLD_CAP_SHARE` | 70 | start | 整理の大きさの上限。整理する思考の %。10〜100 |
| `TF_GLM53_BYTE_RESUME` | on | start | 切り方のずれを越えてキャッシュから再開する (9.3)。on、log、off |
| `MULTI_DECODE_SHARE` | 30 | start | prefill と decode の GPU 配分の初期値 |

古い設定ファイルには、今の規則に合わない `CONTEXT_LENGTH` が残っていることがあります (例: 450560)。変えると元の値に戻せません。

### 4.3 整理の文面を変える、デフォルトに戻す

```python
call(LP, "POST", "/v1/launch/settings", {"values": {"THINKING_FOLD_PROMPT": "考えを整理する。"}})  # 変更
call(LP, "POST", "/v1/launch/settings", {"values": {"THINKING_FOLD_PROMPT": ""}})                  # デフォルトに戻す
```

- `value` は今効いている文、`default` は組み込みの文、`set` は変更されているかです。
- 日本語と改行が使えます。4,096 バイトまでです (ランチャーは文字数で受け付けますが、動作中の変更はエンジンがバイトで数えて断ります。その時は `apply_error` が返ります)。
- 設定ファイルには % 符号化して保存されます。直接読み書きしません。

### 4.4 動作中の変更

| 呼び方 | 変えられるもの |
|---|---|
| `POST /v1/thinking` | `share`、`fold`、`at_close_share` (0〜100)、`fold_share`、`fold_max`、`at_budget`、`fold_margin`、`fold_prompt`、`close_prompt` |
| `POST /v1/schedule` | `decode_share` (0〜100)、`short_prompt`、`short_preempt`。`MULTI` が 0 のエンジンは 404 |

- ランチャーの port で呼ぶと、`/v1/thinking` の変更は設定ファイルにも残ります。
- `/v1/schedule` の変更は次の起動に残りません。残すには `MULTI_DECODE_SHARE`、`MULTI_SHORT_PROMPT`、`MULTI_SHORT_PREEMPT` を `settings` に書きます。
- `GET /v1/thinking` の `start` は起動時の値です。
- `/v1/thinking` の変更では `glm53.conf.bak-<時刻>` は作られません。

## 5. 思考の予算と整理

### 5.1 仕組み

- **思考の予算:** prompt を除いた残りの `THINKING_SHARE` %。`max_tokens` の方が小さければ、その %。`THINKING_SHARE` が 0 なら予算はありません。中間整理 (`THINKING_FOLD`) があるときは `max_tokens` から答えの分を引いた値で、context の残りを超えることがあります。**context に置ける思考の量であって、生成量の上限ではありません。**
- **整理の開始:** 予算の `100 - THINKING_FOLD_MARGIN` % (既定 90%)。その後の行の区切りを待つので、少し遅れます。
- **整理:** モデルに求める文で思考を書き直させ、context の思考をその結果に置き換えます。
- **整理の後:** モデルが考え続けるか閉じるかを決めます。context の思考が予算に達すると、エンジンが閉じます。応答か context の残りが答えの分だけになった時も閉じます。
- **答えの分:** 予算の残り (既定 10%) は、思考が予算を使い切った時に答えに残る最低限です。上限ではありません。
- **モデルが閉じた時の整理:** 既定はオフです。

**生成量は予算を超えます。** 予算 2,200 でも、整理と続きで思考は 2,700〜4,300 トークンになり、1 本で 90〜160 秒かかります。

**`max_tokens` は十分大きく指定します。** 小さいと答えの分が足りません。`max_tokens` 20,000 では答えが 2,000 トークンで切れました。

### 5.2 要求ごとの指定 (`/v1/chat/completions`)

| 項目 | 意味 |
|---|---|
| `thinking_budget` | 予算をトークン数で指定 |
| `thinking_fold_at_budget` | true / false。予算の手前の整理 |
| `thinking_fold_margin` | 0〜50 |
| `thinking_fold_at_close` | トークン数。0 はオフ |
| `thinking_fold` | off、checkpoint、recent、branch |
| `thinking_fold_window` | 中間整理の間隔をトークン数で |
| `thinking_fold_max` | 中間整理の回数の上限 |
| `thinking_fold_cap` | 整理の大きさの上限をトークン数で。128〜16384 |
| `priority` | `"background"` で待ち行列の後ろへ |
| `chat_template_kwargs` | `{"enable_thinking": false}` で思考なし |
| `reasoning_effort` | none (思考なし)、minimal と low (Low)、medium と high (High)、xhigh と max (Max)。省くと Max。**値が変わると prompt が変わり、キャッシュは当たりません** |

`/v1/messages` は別の形です (8.1)。

### 5.3 `GET /v1/thinking`

`share`、`budget`、`fold`、`at_close_share`、`fold_share`、`fold_max`、`at_budget`、`fold_margin`、`fold_prompt`、`fold_prompt_default`、`close_prompt`、`close_prompt_default`、`start`、`fold_window`、`fold_cap`、`fold_cap_share`、`notice`、`max_tokens`。

## 6. 起動

### 6.1 起動の選択

**割り当て (`alloc`) が持つのは bit、bank、利用記録だけです。** 次の選択は持ちません。省くと設定ファイルの値になり、前の稼働と変わることがあります。

| 要求の項目 | 今の値を読む所 | 意味 |
|---|---|---|
| `kv` | `status.kv.format` | fp4 か fp8 |
| `context` | `status.kv.context` | 1 回の最長 |
| `store` | `status.kv.store` | KV のうち会話を保持する分。数か `"max"` (入るだけ) |
| `reserve_gib` | `status.kv.reserve_gib` | OS に残すメモリ。2〜32 |
| `drafter` | `status.drafter.running` | mtp、dflash2、none |
| `multi` | `status.multi.running` | 同時に推論する数。0〜4 |
| `multi_wait` | `status.multi.wait` | 秒 |

**同じ構成で再開するには、止める前の `status` からこの表のとおりに読みます。** `"max"` は、その時の空きで決まるので同じ値になるとは限りません。同じ値が欲しければ数で渡します。

### 6.2 手順

```python
# 1. 止まっているか
s, st = call(LP, "GET", "/v1/launch/status"); assert st["state"] == "stopped"

# 2. 使える割り当てを選ぶ
s, a = call(LP, "GET", "/v1/launch/allocs")
names = [x["name"] for x in a["allocs"] if x["valid"]]

# 3. 割り当ての bit を読み、収まるか確かめる (起動はしない)
s, al = call(LP, "GET", f"/v1/launch/alloc?name={names[0]}")
opts = {"kv": "fp4", "drafter": "mtp", "context": 131072, "store": "max", "reserve_gib": 12, "multi": 4}
s, plan = call(LP, "POST", "/v1/launch/plan", dict(opts, bits=al["bits"]))
# plan["verdict"]: fits | tight | no、plan["nodes"][].margin、plan["missing_bits"]、plan["multi"]["why"]
# 収まらない context は断られず、短く切られる。plan["kv"]["context"] と渡した値を比べる

# 4. 起動
s, r = call(LP, "POST", "/v1/launch/start", dict(opts, alloc=names[0]))   # {"state": "starting"}

# 5. 待つ
while True:
    st = call(LP, "GET", "/v1/launch/status")[1]
    if st["state"] in ("running", "failed"): break
    print(st["phase"], st["layers_done"], st["eta_ms"], flush=True)
    time.sleep(10)
assert st["state"] == "running", st["error"]
```

- `alloc` で起動すると、その割り当ての bit と bank と利用記録を引き継ぎます。**再開にはこれを使います。** `tf-glm53 start` は利用記録を引き継ぎません。
- `bits` も `alloc` も省くと、全 expert を 3 ビットで起動します。
- 全 expert を別の同じ幅で起動するなら、bit は `base64.b64encode(bytes([W]) * (75 * 256)).decode()` です。`W` は `banks.available` の幅です。
- 大きさは `GET /v1/launch/alloc` の `layers` と `experts` です。今は 75 と 256 です。

### 6.3 起動後の確認

```python
s, r = call(EP, "POST", "/v1/chat/completions", {"model": MODEL, "max_tokens": 64, "temperature": 0,
    "messages": [{"role": "user", "content": "What is 17 times 23? Answer with the number only."}],
    "chat_template_kwargs": {"enable_thinking": False}}, timeout=180)
# content に 391 があること
```

`POST /v1/launch/inference-check` は、同じことをランチャーから行います。約 1,000 token の要求を 1 本、裏で送ります (202。エンジンが動いていないときと確認中は 409)。`GET /v1/launch/inference-check` で `state` (idle、checking、done)、`ok`、異常のときの `why`、`ms`、`prompt_tokens`、`completion_tokens` を読みます。判定はランチャーが行います: HTTP 200、`finish_reason` が stop、答えがプロンプト内の乱数と一致。エンジンが止まると idle に戻ります。

`MODEL` は `GET /health` の `model` です。**起動直後の 1 本目は prefill が遅くなります。** 速さを測る時は捨てます。

### 6.4 エラー

| コード | 意味 | 対処 |
|---|---|---|
| 400 | 値が不正 | `error` を読む |
| 404 `NoSuchAlloc` | 割り当てがない | `GET /v1/launch/allocs` |
| 409 | 動作中、または bank が合わない (`BankMissing`、`BankDiffers`) | 止める、または `allocs[].problem` を読む |
| 412 | 計画が収まらない、データのない幅、ドラフターが使えない | `plan` の結果を見て `context`、`store`、`multi` を減らす |

`state` が `failed` の時は `error` を読み、ノードのログを見ます (1.5)。

## 7. 終了

### 7.1 保存してから止める (標準)

```python
NAME = "run-1"

# 1. 走っている推論がないこと
h = call(EP, "GET", "/health")[1]
assert not h["live"]["connections"]

# 2. 動いているエンジンの bit と利用記録を保存
s, r = call(LP, "POST", "/v1/launch/allocs/save",
            {"name": NAME, "engine": True, "with_usage": True, "source": "engine", "note": "before stop"})
assert s == 200, r

# 3. 保存を確かめる (確かめてから止める)
a = call(LP, "GET", "/v1/launch/allocs")[1]
assert any(x["name"] == NAME and x["usage"] for x in a["allocs"])

# 4. 同じ構成で再開する予定なら、status の選択を控える (6.1)

# 5. 止める。保存済みなので save_usage は false
s, r = call(LP, "POST", "/v1/launch/stop", {"save_usage": False})
assert s == 200, r

# 6. 待つ
while call(LP, "GET", "/v1/launch/status")[1]["state"] not in ("stopped", "failed"):
    time.sleep(3)
```

- **順番を変えません。** `with_usage` の保存は動いている間しかできません。止めた後では取り戻せません。
- 名前が既にあると 409 `Exists` です。別の名前にするか、`"replace": true` を付けます。
- 既に止まっている時の `stop` は 409 です。
- `stop` の `save_usage` を省くと、記録は保留になります (`GET /v1/launch/usage` の `pending`)。保留はランチャーのメモリにしかなく、ランチャーを止めると消えます。

### 7.2 再起動

```python
call(LP, "POST", "/v1/launch/allocs/save", {"name": NAME, "engine": True, "with_usage": True, "source": "engine"})
# 保存を確かめる (7.1 の 3)
call(LP, "POST", "/v1/launch/restart", {"alloc": NAME})   # 202
```

- 動いている起動の選択 (`kv`、`drafter`、`context`、`store`、`reserve_gib`、`multi`、`multi_wait`) を引き継ぎます。`pending` に載っている鍵だけ、設定ファイルの値になります。
- `alloc` を省くと、利用記録を保存せずに止めます。

### 7.3 起動の取り消し

起動中の `POST /v1/launch/stop` は起動を取り消します。

## 8. 推論

### 8.1 API

| 経路 | 形式 | 使う場面 |
|---|---|---|
| `/v1/chat/completions` | OpenAI | 標準。pi、opencode |
| `/v1/messages` | Anthropic | Claude Code |
| `/v1/completions` | 生の prompt | 特殊トークンをそのまま送る試験用。思考の予算と整理は効かない |
| `/v1/responses` | OpenAI Responses | Responses API のクライアント |
| `/v1/models` | | モデル名の一覧 |
| `/v1/tokenize`、`/v1/detokenize` | | トークン数と id |
| `/v1/messages/count_tokens` | Anthropic | `/v1/messages` のトークン数 |

**`/v1/messages` の違い:**

- 思考は `"thinking": {"type": "enabled", "budget_tokens": 2200}` で指定します (`adaptive` も受け付けます)。`thinking_budget` は効きません。**`thinking` を省くと思考なしです。**
- `max_tokens` は必須です。`budget_tokens` が `max_tokens` 以上なら 400 です。
- 応答の `content` に `thinking` ブロックと `text` ブロックが入ります。
- `usage.input_tokens` は当たった分を含みません。`usage.cache_read_input_tokens` が当たった分、`usage.output_tokens_details.thinking_tokens` が思考です。

### 8.2 応答 (`/v1/chat/completions`)

```python
s, j = call(EP, "POST", "/v1/chat/completions",
            {"model": MODEL, "max_tokens": 276480, "temperature": 0, "messages": messages}, timeout=7200)
m = j["choices"][0]["message"]
m["content"]              # 答え
m["reasoning_content"]    # 思考。整理があれば、整理前と整理の結果と続きがすべて入る
m.get("reasoning_details")  # エンジンが思考に手を入れた応答に付く。context が持っている思考
j["usage"]["prompt_tokens"]
j["usage"]["prompt_tokens_details"]["cached_tokens"]
j["usage"]["completion_tokens_details"]["reasoning_tokens"]
```

ストリームは `stream()` (1.5) で読みます。思考と答えは `delta.reasoning_content` と `delta.content` に分かれ、`delta.reasoning_details` は終わりの直前に 1 回だけ届きます。

### 8.3 次のターン

応答を次の要求の `messages` に入れて返します。**`reasoning_content` と `reasoning_details` は受け取ったまま返します。**

```python
a = {"role": "assistant", "content": m.get("content") or ""}
for k in ("reasoning_content", "reasoning_details"):
    if m.get(k): a[k] = m[k]
messages += [a, {"role": "user", "content": "次の質問"}]
```

| 返すもの | 整理があった応答の次のターン |
|---|---|
| `reasoning_details` あり | ほぼ全部当たる。context の思考で組み立てられる |
| `reasoning_content` だけ | その応答の思考の頭から外れる。prompt も全文の分だけ長くなる |
| 思考を返さない | その応答の思考の頭から外れる |

### 8.4 ハーネス

**pi** (`~/.pi/agent/models.json`): `reasoning_details` を受け取ったまま返すので、整理の後もキャッシュが保たれます。

```json
{"providers": {"expertatlas": {
  "name": "ExpertAtlas", "baseUrl": "http://HOST:PORT/v1", "api": "openai-completions", "apiKey": "none",
  "compat": {"supportsDeveloperRole": false, "maxTokensField": "max_tokens", "thinkingFormat": "openai"},
  "models": [{"id": "MODEL", "name": "MODEL", "reasoning": true, "input": ["text"],
              "contextWindow": 131072, "maxTokens": 64000,
              "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
              "samplingParams": {"thinking_budget": 52000}}]}}}
```

- `contextWindow` はエンジンの `context_length` に合わせます。
- pi は `maxTokens` を prompt の長さに応じて減らして送ることがあります。
- 要求ごとの項目は、モデルの中の `samplingParams` に書きます。
- `reasoning_effort` は pi の既定で medium (ログでは high) が送られます。

**opencode** (`opencode.json`、1.19.0 で確認): **`reasoning_content` だけを返します。** 整理があった応答の次のターンは、その応答の思考の頭から外れます。

```json
{"$schema": "https://opencode.ai/config.json",
 "enabled_providers": ["expertatlas"],
 "provider": {"expertatlas": {"npm": "@ai-sdk/openai-compatible", "name": "ExpertAtlas",
   "options": {"baseURL": "http://HOST:PORT/v1", "apiKey": "none", "headerTimeout": 3600000, "chunkTimeout": 3600000},
   "models": {"MODEL": {"name": "MODEL", "tool_call": true, "reasoning": true,
                        "limit": {"context": 131072, "output": 64000},
                        "options": {"thinking_fold_at_budget": false}}}}}}
```

- 要求ごとの項目は `models.MODEL.options` に書きます。上の例は整理をオフにする指定です (直接の要求で、整理なしなら `reasoning_content` だけで当たることを確認済み。opencode での確認はまだです)。
- 会話の題名を作る要求が別に走ります。思考つきで動き、50 秒ほどかかることがあります。`small_model` に別のモデルを指定できます。

**Claude Code:** `ANTHROPIC_BASE_URL=http://HOST:PORT` を指定します。パッケージの `claude-glm53` が起動を行います。

ハーネスが生成したコードは sandbox の中で動かします。`--network host` の sandbox からはランチャーにも届きます。

## 9. キャッシュの確認

### 9.1 確かめ方

| 見るもの | 呼び方 | 意味 |
|---|---|---|
| そのターンで当たった量 | 応答の `usage.prompt_tokens_details.cached_tokens` | `prompt_tokens` のうちキャッシュから取った数 |
| 走っている要求 | `GET /health` の `progress` | `phase`、`prompt_tokens`、`cached_tokens`、`prompt_done` |
| 保持している会話 | ランチャーの `GET /v1/pool` | `convs[].positions`、`blocks`、`shared_blocks`、`idle_ms`、`active`。追い出された数は `GET /health` の `pool.evicted_total` |
| ログ | `timeout 10 ./tf-glm53 logs 0 \| grep` | `[tensorfold] done ... prompt=N cached=M` と `glm53: thinking:` の行 |

`/v1/pool` の `owner` と `convs[].table` は base64 です。context に入れません。

### 9.2 実測

| 場合 | prompt | cached |
|---|---|---|
| 同じ会話の次のターン | 2,610 | 2,593 |
| 無関係な会話を挟んだ次のターン | 2,713 | 2,696 |
| 整理の後、モデルが閉じた、8.3 のとおり返す | 1,017 | 1,000 |
| 整理の後、エンジンが閉じた、8.3 のとおり返す | 4,196 | 4,179 |
| 整理の後、`reasoning_content` だけ返す | 3,252 | 256 |
| 整理の後、思考を返さない | 506 | 256 |

- 通常は 17 トークン前後だけ外れます。末尾のテンプレートの分です。
- 別の会話と共有する先頭は 256 単位に切り捨てて当たります。
- **大きく外れた時:** 8.3 の返し方、`reasoning_effort` の違い、`/v1/pool` に会話が残っているか、を順に疑います。

### 9.3 ログの行

| 行 | 意味 |
|---|---|
| `a summary beside the reply asked for at the budget's end at N think tokens` | 予算の手前で整理を求めた |
| `fold 1 (branch at the budget's end) at N think tokens: the context's reply A -> B tokens` | context の思考が A から B になった |
| `after the budget's summary the model goes on, until the thinking's memory of N tokens is full` | モデルが続ける |
| `past the budget's summary the thinking's memory is full` | 思考が予算を埋め、エンジンが閉じた |
| `past the budget's summary only the answer's part is left` | 残りが答えの分だけになり、エンジンが閉じた |
| `fold 1 (branch at the close)` | モデルが閉じた時の整理 |
| `the budget of N think tokens reached` | 整理なしで予算に達して閉じた |
| `ids remade (fold\|close): A -> B think ids, the first difference at N` | 思考のトークン列を、文字列から切り直した結果に合わせた |
| `the think block closed after N reply tokens; the answer goes on from its remade ids` | モデルが閉じ、切り直して答えに進んだ |
| `pool: conversation N done: cancelled` | 同時推論 (`MULTI`) のとき、整理や切り直しで区間を切り替えた。正常 |
| `ended reason=client-left` | クライアントが切った。`done` の行は出ない |
| `pool:` の NEW、CONTINUE、SHARE | 新しい会話、同じ会話の続き、他の会話と先頭を共有 |

短く試すには、要求で `thinking_budget` を 2,000 程度にします。整理を起こしたい時は、`thinking_fold_margin` を大きくすると早く始まります。1,024 トークン未満しか考えていない時は整理しません。

### 切り方のずれ

モデルの切り方がトークナイザーと違うと、次のターンの prompt は同じバイトを別の id で持ち、キャッシュがそこで止まります。既定の `TF_GLM53_BYTE_RESUME=on` では、prompt を保存した会話とバイトで比べ、一致する所はその id を使います。`off` はキャッシュがそこで止まります。`log` は `off` と同じで、比べて記録します。ログの `byteresume:` の行に、id が止まった位置とバイトが届いた位置が出ます。

## 10. 困った時

| 症状 | 確かめること |
|---|---|
| 接続できない | `./tf-glm53 launcher status`。`HOST` が 127.0.0.1 ではないか |
| 403 | エンジンの port を LAN から呼んでいる (1.1)。エンジンの `/v1/experts`、`/v1/schedule`、`/v1/thinking` は同じノードからだけ答える |
| `/v1/schedule` が 404 | `MULTI` が 0 のエンジン (既定) |
| `/v1/thinking` が 503 | エンジンが止まっている。設定は `settings` で読む |
| 設定が効かない | `when` と `pending`、`from` が `node_file` か、要求の値が優先されていないか |
| 動作中の変更の後に `apply_error` | エンジンが断った。設定ファイルには書かれている |
| 答えが切れる | `max_tokens` が小さい (5.1) |
| 再開したら構成が違う | 起動の選択を渡していない (6.1) |
| 保存の直後に幅が選べない | `banks.checking` が false になるまで待つ (3.2) |
| 起動が `tight` か `no` | `POST /v1/launch/plan` の `nodes[].margin`、`kv.formats`。`context`、`store`、`multi` を減らす |
| `GET /v1/launch/status` が `failed` | `error`、ログ (1.5) |
| 止めたいのに 409 | 既に止まっているか、停止か再起動の途中。起動の途中なら取り消せる (7.3) |
| ログのコマンドが返らない | `tf-glm53 logs` は追従する。`timeout` を付ける (1.5) |
