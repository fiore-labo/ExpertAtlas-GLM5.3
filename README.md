English | [日本語](README.ja.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md)

# Full GLM-5.3 on four or eight DGX Sparks

Full GLM-5.3 (755B, `glm_moe_dsa`) served by four or eight NVIDIA DGX Spark systems, one GB10 each, with tensor
parallelism 4 or 8: one request at a time, fast, with a long context for agents and coding. The engine is written in
Zig with plain CUDA kernels and needs no Python at run time.

This is Fiore-lab's unofficial derivative of [TensorFold](https://github.com/ashhart/TensorFold) 1.0.2, not a release
of the TensorFold project. It carries TensorFold's code, adds the engine for full GLM-5.3
(`zig/src/families/glm_moe_dsa/`, `zig/kernels/cuda/glm53/`) and leaves out TensorFold's documents, which describe
other models, weights and machines. For TensorFold itself, see its repository.

**Using an AI assistant to understand or run this system?** Give it [glm53/release/AI-GUIDE.md](glm53/release/AI-GUIDE.md)
(`AI-GUIDE.md` in the release archive; Japanese: `AI-GUIDE.ja.md`): one page written for AI assistants that describes
the whole system, its purpose, how to set it up and use it, what it can and cannot do with the measured numbers, and
where the details are.

## What it does

- Tensor parallelism 8 or 4 with the KV cache split across rank pairs (decode context parallelism 2). The engine and
  the model take up to 1M tokens (1,048,576); the DGX Spark's memory sets the context: **1M on eight nodes** (FP8 KV
  cache) and **about 800K (819,200) on four** (4-bit KV cache). The conf examples ship with 819,200 and 737,280; with
  the same memory, eight nodes take `CONTEXT=1048576` `STORE=589824` and four nodes `CONTEXT=819200` `STORE=18080`
  (the KV pool holds `CONTEXT` + `STORE` positions).
- Long prompts read in sequence-parallel parts; decode-step collectives over RoCE.
- One conversation cache for all conversations, in blocks of 256 positions: a request resumes from the longest cached
  prefix of any conversation, and conversations that begin alike (an agent's system prompt, its sub-agents) hold the
  blocks they share once.
- Drafts checked by the model in each round: by GLM-5.3's own MTP layer by default, or by the optional DFlash2 drafter.
  The reply is the model's own choice in every case.
- `/metrics` (Prometheus text) and `/health` count the running request too, so a monitor reads prefill and decode
  speed, cache hits and draft acceptance while a request runs. `METRICS_LISTEN` serves the two read-only on an address
  of their own, for a monitor on another host.

## Expert Atlas (new in r3-102)

GLM-5.3 picks 8 of 256 routed experts a layer for each token, and some experts are picked far more often than others.
Expert Atlas loads each of the 19,200 routed experts (layers 3 to 77) at its own width, 2, 3, 4 or 5 bits, so the
experts in most use get the most bits for the memory there is:

- **Banks**: the routed experts at each width, on Hugging Face: `fiore-lab/GLM-5.3-EXL3-Experts-2bit`, `-3bit`,
  `-4bit` and `-5bit`. The panel offers the widths every node holds.
- **The control panel**: a page served by the launcher (`./tf-glm53 launcher`). Before a start it plans the widths
  and shows each node's memory, whether the plan fits and the KV cache left; it starts and stops the engine. While
  the engine runs it shows the experts each token picks, the mean width of the picked experts and the speeds.
- **Usage records**: the engine counts the experts every prompt row and decoded token picks. A saved allocation keeps
  the usage record of the runs it was saved from, so its data travels with it: opened and run again, its record grows
  by that run. Plans rank the experts by the record (a plan without one, by counts Fiore-lab measured and ships with
  the page). Nothing is saved by itself: the stop saves the allocation and its record under a name.
- **Expert optimisation**: re-sorts the loaded experts by the run's record with each width's number of experts kept:
  the same memory, a higher mean width for the experts the model actually picks. It runs on a live engine (a full
  change took 83 to 90 s on four nodes, between requests); the result is saved under a name when wanted and opened
  for later starts.

The control panel runs on four or eight nodes, the count chosen on its launch screen; `./tf-glm53 up` still starts
them with the checkpoint's widths. `packaging/RUNTIME.md` ("The control panel") and `glm53/release/T4.md` give the
details.

## Making the weights yourself

Weights cannot be read like code: weights altered to behave differently in some cases look like any others, so you
may not want to rely on weights whose making you cannot check. You do not have to rely on Fiore-lab's. Every weight file this recipe loads, the eight-
and four-node checkpoints and the 2-, 3-, 4- and 5-bit expert banks, can be made from Z.AI's official BF16 checkpoint
([zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16)) with the tools in
[glm53/weights](glm53/weights/README.md), which give the steps, the hardware and the measured times. The files you make
have the same layout as Fiore-lab's and load the same way.

## What you need

- Four or eight DGX Spark systems (GB10) on one RoCE v2 network, each reachable from the first over ssh without a
  password. For four nodes, one 100 Gb/s port a node is enough.
- DGX OS (Ubuntu 24.04) with the NVIDIA driver 580 or later and cuBLAS 13.
- NCCL 2.30.7 on every node: NVIDIA's `nvidia-nccl-cu13` wheel, unpacked by pip (`packaging/RUNTIME.md`).
- Fiore-lab's EXL3 checkpoint for the number of nodes, on Hugging Face, at the same path on every node:
  `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks` (390 GB) for eight nodes, `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks` (344.6 GB)
  for four. The four-node checkpoint holds the routed experts of layers 3 to 77 at 3 bits for the least used half and
  at 4 bits for the rest; everything else is as in the eight-node checkpoint.
- Every weight file can also be made yourself from Z.AI's official BF16 checkpoint, without any of Fiore-lab's files:
  [glm53/weights](glm53/weights/README.md).
- For the control panel: the expert banks of the widths it uses, on four nodes the 3- and 4-bit banks (277.9 and
  368.5 GB; the eight-node checkpoint can serve as the 4-bit bank; with 4 bits alone the context is shorter), and
  `python3` on the nodes for the launcher.

## Run

The release comes as a container image and as an archive with the same files; the image is the way we recommend. With
the image, every node needs Docker and the NVIDIA container toolkit (DGX OS has both) and nothing else of the
server's libraries:

```sh
docker pull ghcr.io/fiore-labo/fiore-glm5.3:r3-102            # on every node
mkdir glm53 && docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/glm53":/out ghcr.io/fiore-labo/fiore-glm5.3:r3-102 install /out
```

then set `RUNTIME=docker` and `IMAGE` in `glm53.conf` and go on as below ([packaging/RUNTIME.md](packaging/RUNTIME.md),
"Running in containers"). Without Docker: download the release archive `tensorfold-r3-102-linux-aarch64.tar.gz` and follow its `RUNTIME.md`
([packaging/RUNTIME.md](packaging/RUNTIME.md) here). The same archive and launcher serve four or eight nodes: copy
`glm53.conf.example` (eight nodes) or `glm53.t4.conf.example` (four nodes, see
[glm53/release/T4.md](glm53/release/T4.md)) to `glm53.conf`, fill in the nodes, network names, NCCL, weights and
drafter, then `./tf-glm53 deploy` and `./tf-glm53 up`. The server speaks the OpenAI chat API on `127.0.0.1` unless you
give it an address and an API key file.

## Drafters

- `mtp` (default): the checkpoint's own MTP layer, part of the GLM-5.3 weights. About twice the decode speed of no
  drafts, with the same prefill speed.
- `dflash2`: incoai/GLM-5.3-DFlash2. **Not included in this repository or the archive, because of its license**: its
  model card states CC BY-NC-ND 4.0 (non-commercial use, no derivatives), for research and evaluation, with
  commercial licensing from Inco AI. If your use is covered by that license, follow "Using DFlash2" in
  [packaging/RUNTIME.md](packaging/RUNTIME.md#using-dflash2): download it once at a fixed revision, copy it to every
  node at the same path, check the copies, then set `DRAFTER_KIND=dflash2` and `DRAFTER`. The engine quantizes it to
  4 bits in memory at the start and writes nothing to disk. Decode is faster still (Speed).
- `none`: no drafts.

`DRAFTER_KIND` in `glm53.conf` picks one; `./tf-glm53 -d KIND up` picks one for a single start.

## Speed

**Measured with the GPU clock capped at 2.1 GHz (2,100 MHz) on every node.** One request at a time with a coding prompt
not yet cached (reasoning effort low, up to 4,000 reply tokens), this archive with the settings of
`glm53.conf.example` (eight nodes, FP8 KV cache) and `glm53.t4.conf.example` (four nodes, 4-bit KV cache), and
Fiore-lab's published checkpoints. Your numbers depend on the checkpoint, the prompt and the clocks.
`packaging/RUNTIME.md` gives the conditions in full.

| Nodes | Prompt | MTP (k=3): time to first token / decode | DFlash2: time to first token / decode |
|---|---|---|---|
| 8 | 128K | 82.8 s / 47.5 tokens/s | 85.2 s / 53.2 |
| 8 | 256K | 172.9 s / 45.7 | 177.6 s / 52.6 |
| 8 | 512K | 376.5 s / 42.8 | 386.0 s / 47.1 |
| 4 | 128K | 120.1 s / 35.3 | 121.9 s / 42.9 |
| 4 | 256K | 253.3 s / 33.2 | 256.9 s / 40.7 |
| 4 | 512K | 565.4 s / 32.2 | 571.5 s / 37.8 |

## Accuracy

KL divergence of this recipe's output distributions from the original BF16 model's (the official BF16 weights run by
the official implementation, computed in FP32), with this recipe's serving configurations:

- Eight nodes (FP8 KV cache, DCP 2; measured on the same weights with an earlier build of this engine): four 2,048-token texts (dialogue,
  legal, prose, reasoning), the last 1,023 positions each, five runs: mean 0.0133 (0.0128 to 0.0146); the top-1 token
  agrees at about 96%.
- Eight nodes, four 8,192-token texts (two English, two code), by position range: 0.012 (positions 1,024 to 2,047),
  0.018, 0.019 and 0.013 (6,144 to 8,190). For scale, the official BF16 weights computed in BF16 instead of FP32 differ
  from the same reference by 0.007 to 0.010 over the same ranges.
- Four nodes (the four-node checkpoint, 4-bit KV cache, DCP 2), four 8,192-token texts by the same position ranges:
  0.019 (positions 1,024 to 2,047), 0.032, 0.029 and 0.019 (6,144 to 8,190).
- Expert Atlas's widths, every routed expert at one width with the other weights as above, the same four texts,
  positions 6,144 to 8,190: 2 bits 0.0517, 3 bits 0.0203, 4 bits 0.0127 (four nodes), 5 bits 0.0114 (eight nodes).
  At the same memory, a mix with 2-bit experts was 13 to 30 % further from the reference than one without, and 5-bit
  experts paid for by moving others down to 3 bits gained nothing measurable.

## Build from source

On a GB10 system with Zig 0.17.0 on the `PATH` (or `ZIG=` its path for `build_libs.sh`), CUDA 13 (`nvcc`) and the
libibverbs headers (`libibverbs-dev`):

```sh
zig build fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 --prefix /tmp/fatbins
sh glm53/tools/release/build_libs.sh /tmp/lib        # lib/libroce_proxy.so
zig build dist -Dversion=r3-102 -Ddist-commit=$(git rev-parse HEAD) -Ddist-platforms=linux-aarch64 \
    -Ddist-fatbins=/tmp/fatbins/fatbin -Ddist-lib=/tmp/lib -Ddist-extra=glm53/release
```

The archive lands in `zig-out/dist/`.

## Limits

- Four or eight GB10 systems only; other GPUs and other rank counts are not supported.
- One request runs at a time; up to 32 others wait in arrival order, chat-title requests last, and past 32 the server
  refuses them (a 503, or an error event in a stream). Agents that run sub-agents in parallel are better served by
  pointing the sub-agents at another model.
- `packaging/RUNTIME.md` lists the rest.

## License and credits

The code is Apache-2.0 (`LICENSE`); `NOTICE` and `THIRD_PARTY_NOTICES.md` say what comes from where. The weights keep
their own license (GLM-5.3 License); the optional DFlash2 drafter keeps its own.

- [TensorFold](https://github.com/ashhart/TensorFold) by the TensorFold contributors: the server, the build and the
  CUDA engine this recipe extends.
- TensorFold's Python engine for full GLM-5.3 by drowzeys
  ([ashhart/TensorFold#159](https://github.com/ashhart/TensorFold/pull/159)): the GLM-5.3 engine rewrites its design in
  Zig and plain CUDA (the engine's structure, its Triton kernels and its MTP and DFlash2 drafter paths). Some of its
  CUDA files are included, a few of them modified.
- [b12x](https://github.com/local-inference-lab/b12x): its RoCE one-shot collectives are rewritten as plain CUDA
  (`roce.cu`), and its RoCE proxy is included (`zig/native/roce_proxy.c`), modified to pick a GID index for each RDMA
  device.
- [MiaAI-Lab's GLM-5.3-Flash on 2x DGX Spark](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold):
  the GLM-5.3 prompt kernels are written with reference to its prompt kernels.
- [MiaAI-Lab's GLM-5.3 EXL3 on 3x DGX Spark](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold): the
  4-bit KV cache takes its layout, implemented with this engine's own code.
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3): the checkpoints are quantized with it, and the engine reads
  its EXL3 format.

## TensorFold's contributors

From TensorFold's README: thank you to everyone who has sent TensorFold a pull request, a measurement or a bug report
(the list as of TensorFold 1.0.2; see [the TensorFold repository](https://github.com/ashhart/TensorFold) for its
current list):

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
