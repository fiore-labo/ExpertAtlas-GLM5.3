[English](BEFORE-START.md) | 日本語 | [简体中文](BEFORE-START.zh-CN.md)

# 起動前に決めること

必ず決めるのは、どのノードで動かすかと、重みをどこに置くかの 2 つです。それ以外は既定のまま起動でき、STOP が出たら画面の案内に従って直します。起動の手順は `QUICKSTART.ja.md` にあります。

## 起動の前に用意して決めること

ノードと重みの場所は、最初に決めます。重みは全ノードで同じパスに置きます。

| 決めること | 選択肢 | 既定 | オプション |
|---|---|---|---|
| 動かすノード | スイッチの 4 台か 8 台、またはリングの 4 台 (2 台と 3 台はテスト中) | 自動で探す。スイッチで空きが余るときだけ選ぶ | `NODES` |
| チェックポイントの場所 (必須) | 4 台用 344.6 GB、8 台用 | `~/models/GLM-5.3-EXL3-4x-DGX-Sparks` (8 台は `-8x-`) | `WEIGHTS` |
| 3 bit バンクの場所 (必須) | 277.9 GB | `~/models/GLM-5.3-EXL3-Experts-3bit` | `BANK_3` |
| ほかの幅のバンク | 2 bit 187.3 GB、4 bit 368.5 GB、5 bit 459.1 GB。使う幅だけ | `~/models/GLM-5.3-EXL3-Experts-<N>bit` | `BANK_2`、`BANK_4`、`BANK_5` |
| DFlash2 のモデル | 使うときだけ。許諾は別 | なし | `DRAFTER` |
| インストール先 | ホーム以下のディレクトリ | `expertatlas` | `EA_DIR` |
| ポート | API とパネル | 8000 と 8010 (API + 10) | `PORT`、`LAUNCHER_PORT` |
| API の公開範囲 | 全アドレス、または 1 台目だけ (`127.0.0.1`) | 全アドレス (`0.0.0.0`) | `HOST` |

API にもパネルにもキーはありません。ノードは閉じたネットワークに置きます。

決めることではありませんが、起動には次も必要です。各ノードの空きメモリ約 115 GiB、GPU にほかのプログラムがないこと、パスフレーズなしの ssh 鍵、`docker` を使えること。

## 起動のたびに決めること

次の設定は、変えるとエンジンの再起動が必要です。迷ったら既定のままで起動できます。

| 決めること | オプション | 選択肢 | 既定 | 決め方 |
|---|---|---|---|---|
| エキスパートの幅 | `BITS` または `ALLOC` | 2、3、4、5 bit、または保存した割り当て | 3 bit | 幅を上げるとメモリを多く使い、次の 2 つが減る |
| 1 回の最長 | `CONTEXT_LENGTH` | 65,536〜1,048,576 トークン、16,384 の倍数 | 737,280 (4 台、3 bit) | 扱いたい最長の prompt + 応答 |
| KV の総量 | `KV_SIZE` | トークン数、または `max` | `max` | 普通は `max`。多いほど多くの会話をキャッシュに残せる |
| KV の形式 | `KV` | `fp4`、`fp8` | `fp4` | `fp8` は精度が高いがメモリを多く使う |
| 下書きの仕組み | `DRAFTER_KIND` | `mtp`、`dflash2`、`none` | `mtp` | `dflash2` は別のモデルが必要で、KV が減る |
| 同時に decode する会話数 | `MULTI` | 0〜4 | 0 (1 件ずつ) | 同時に使う人数で決める |
| 待ち時間 | `MULTI_WAIT` | 秒数 | 600 秒 | `MULTI` のとき、超えた要求は混雑として断る |
| OS に残すメモリ | `RESERVE_GB` | 2〜32 GiB | 4 GiB | 同じノードで他のものを動かすなら上げる |
| 応答の上限の既定 | `MAX_TOKENS_SHARE` | `CONTEXT_LENGTH` の 1〜100 % | 70% | 要求が `max_tokens` を指定しないときの値。クライアントの指定が優先 |
| Byte Resume | `TF_GLM53_BYTE_RESUME` | `on`、`log`、`off` | `on` | 切り方のズレでキャッシュが外れるのを防ぐ。通常は `on` |
| API でのモデル名 | `NAME` | 文字列 | `glm-5.3` | クライアントの設定に合わせる |

- **思考と `max_tokens`:** 思考は応答に数えられます。クライアントが `max_tokens` を送ると、それが上限です。小さい値を送るクライアントでは、思考もその 90% までに限られます。
- **DFlash2 と `MULTI` のメモリ:** リング、`CONTEXT_LENGTH` 131,072、`RESERVE_GB` 16 では、計画の KV は `mtp` の 753,664 から `dflash2` + `MULTI=4` で 360,448 トークンに減ります。これは見積もりで計画する最初の起動の値で、次からは前回の実測を使います。
- **収まらないとき:** `CONTEXT_LENGTH` が長すぎると、収まる長さに切り詰めて、出力で知らせます。

## 起動の前に決めなくてよいこと

Reasoning Control (思考の上限と中間整理) と GPU の配分は、動作中に変えられます。使いながら調整できます。

| 設定 | オプション | 既定 | 次の起動に残るか |
|---|---|---|---|
| 思考の上限 | `THINKING_SHARE` | `max_tokens` と context の残りの小さい方の 90% | 残る |
| 中間整理 | `THINKING_FOLD` | しない (`off`) | 残る |
| 整理する間隔 | `THINKING_FOLD_SHARE` | 上限まで整理しない (100%)。指定は `max_tokens` の 10% から | 残る |
| 整理の回数の上限 | `THINKING_FOLD_MAX` | 制限なし (0) | 残る |
| 予算の終わりでの整理 | `THINKING_FOLD_AT_BUDGET`、`THINKING_FOLD_MARGIN` | する、予算の 10% 手前から | 残る |
| 思考完了時整理 | `THINKING_FOLD_AT_CLOSE_SHARE` | しない (0) | 残る |
| モデルに求める文 | `THINKING_FOLD_PROMPT`、`THINKING_CLOSE_PROMPT` | 組み込みの文 | 残る |
| prefill と decode の GPU 配分 (`MULTI` のとき) | `MULTI_DECODE_SHARE` | decode 30% | パネルで指定したときだけ残る |
| 短い prompt の扱い (`MULTI` のとき) | `MULTI_SHORT_PROMPT`、`MULTI_SHORT_PREEMPT` | 8,192 トークン以下、追い越しなし | パネルで指定したときだけ残る |

変える場所は、推論画面のカードとエンジン設定の画面です。AI エージェントは `POST /v1/thinking` と `POST /v1/schedule` をランチャーの port で呼びます。

## 自動で決まること

`TOPOLOGY`、`SOCKET_IFNAME`、`IB_HCA` は、ノード確認が配線から決めます。`docker run` で指定しても断られます。リングでは `NODES` も自動で決まります。

## 決めた内容の指定の仕方

`docker run` では、`IMAGE` の前に `-e 名前=値` を足します。起動画面で選んでも同じです。

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host \
  -e WEIGHTS=/data/models/checkpoint \
  -e BANK_3=/data/models/3bit \
  -e CONTEXT_LENGTH=131072 \
  IMAGE up
```

- **`docker run` の指定:** `~/expertatlas/glm53.conf` に保存され、次の起動でも使われます。`KV_SIZE` と `BITS` だけは、その起動限りです。
- **起動画面の選択:** その起動の間だけ有効で、保存されません。既定値を変えるのはエンジン設定の画面です。
- **動いているエンジン:** オプションは効きません。止めてから、もう一度同じ行を実行します。
- **名前の間違い:** 知らない名前は STOP になり、近い名前を案内します。

各項目の詳しい意味は `QUICKSTART.ja.md` と `CONFIG.ja.md` にあります。
