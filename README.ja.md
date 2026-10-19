[English](README.md) | 日本語 | [简体中文](README.zh-CN.md)

# 4 台または 8 台の DGX Spark で動かすフルサイズの GLM-5.3

フルサイズの GLM-5.3（755B、`glm_moe_dsa`）を、GB10 を 1 基ずつ搭載した NVIDIA DGX Spark 4 台または 8 台で、テンソル並列 4 または 8 で動かします。リクエストは一度に 1 件ずつ、高速に、エージェントやコーディング向けの長いコンテキストで処理します。エンジンは Zig と素の CUDA カーネルで書かれており、実行時に Python は不要です。

これは [TensorFold](https://github.com/ashhart/TensorFold) 1.0.2 から Fiore-lab が作った非公式の派生物であり、TensorFold プロジェクトのリリースではありません。TensorFold のコードを含み、フルサイズの GLM-5.3 用のエンジン（`zig/src/families/glm_moe_dsa/`、`zig/kernels/cuda/glm53/`）を追加し、TensorFold のドキュメントは除いています。それらのドキュメントは別のモデル、重み、マシンについて書かれたものだからです。TensorFold 自体については、そのリポジトリを参照してください。

## できること

- テンソル並列 8 または 4。KV キャッシュはランクのペアに分割します（decode コンテキスト並列 2）。8 台では FP8 で保持し、1 件のリクエストで 819,200 位置まで扱えます。4 台では 4 ビットで保持し、737,280 位置まで扱えます。
- 長いプロンプトはシーケンス並列で分割して読み込みます。decode ステップの集団通信は RoCE 上で行います。
- すべての会話で 1 つの会話キャッシュを使い、256 位置のブロック単位で保持します。リクエストは、どの会話のものであってもキャッシュにある最も長い先頭部分から再開します。同じように始まる会話（エージェントのシステムプロンプト、そのサブエージェント）は、共通のブロックを 1 回だけ保持します。
- ラウンドごとにドラフトをモデルが検証します。ドラフトを作るのは、既定では GLM-5.3 自身の MTP 層、またはオプションの DFlash2 drafter です。どの場合も、応答はモデル自身が選んだものです。

## 必要なもの

- 1 つの RoCE v2 ネットワークでつながった DGX Spark（GB10）4 台または 8 台。どのノードにも 1 台目からパスワードなしの ssh で接続できること。4 台なら、各ノードに 100 Gb/s のポートが 1 つあれば足ります。
- DGX OS（Ubuntu 24.04）、NVIDIA ドライバ 580 以降、cuBLAS 13。
- 全ノードに NCCL 2.30.7。NVIDIA の `nvidia-nccl-cu13` wheel を pip で展開したものです（`packaging/RUNTIME.md`）。
- ノード数に合った Fiore-lab の EXL3 チェックポイント（Hugging Face）。全ノードの同じパスに置きます。8 台では `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks`（390 GB）、4 台では `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks`（344.6 GB）です。4 台用のチェックポイントは、層 3〜77 の routed expert のうち使用頻度の低い半分を 3 ビット、残りを 4 ビットで持ち、それ以外は 8 台用のチェックポイントと同じです。

## 実行

リリースアーカイブ `tensorfold-r2-102-linux-aarch64.tar.gz` をダウンロードし、同梱の `RUNTIME.md`（このリポジトリでは [packaging/RUNTIME.ja.md](packaging/RUNTIME.ja.md)）に従ってください。4 台でも 8 台でも、同じアーカイブと起動スクリプトを使います。`glm53.conf.example`（8 台）または `glm53.t4.conf.example`（4 台。[glm53/release/T4.ja.md](glm53/release/T4.ja.md) を参照）を `glm53.conf` にコピーし、ノード、ネットワーク名、NCCL、重み、drafter を記入してから、`./tf-glm53 deploy` と `./tf-glm53 up` を実行します。アドレスと API キーファイルを指定しない限り、サーバーは `127.0.0.1` 上で OpenAI chat API を提供します。

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

- 8 台（FP8 KV キャッシュ、DCP 2。同じ重みを、このエンジンの以前のビルドで測定）、2,048 トークンのテキスト 4 本（対話、法律、散文、推論）、それぞれ最後の 1,023 位置、5 回実行: 平均 0.0133（0.0128〜0.0146）。top-1 トークンの一致率は約 96% です。
- 8 台、8,192 トークンのテキスト 4 本（英語 2 本、コード 2 本）、位置範囲ごと: 0.012（位置 1,024〜2,047）、0.018、0.019、0.013（6,144〜8,190）。比較の目安として、公式の BF16 重みを FP32 ではなく BF16 で計算すると、同じ範囲で同じ基準との差は 0.007〜0.010 です。
- 4 台（4 台用のチェックポイント、4 ビット KV キャッシュ、DCP 2）、8,192 トークンのテキスト 4 本、同じ位置範囲ごと: 0.019（位置 1,024〜2,047）、0.032、0.029、0.019（6,144〜8,190）。

## ソースからのビルド

Zig 0.17.0 が `PATH` 上にあり（または `build_libs.sh` には `ZIG=` でそのパスを指定）、CUDA 13（`nvcc`）と libibverbs のヘッダ（`libibverbs-dev`）がある GB10 システムで次を実行します。

```sh
zig build fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 --prefix /tmp/fatbins
sh glm53/tools/release/build_libs.sh /tmp/lib        # lib/libroce_proxy.so
zig build dist -Dversion=r2-102 -Ddist-commit=$(git rev-parse HEAD) -Ddist-platforms=linux-aarch64 \
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
