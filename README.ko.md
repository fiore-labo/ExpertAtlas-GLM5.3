[English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh-CN.md) | 한국어

# DGX Spark 4대 또는 8대로 실행하는 풀사이즈 GLM-5.3

GB10을 1개씩 탑재한 NVIDIA DGX Spark 4대 또는 8대에서 텐서 병렬 4 또는 8로 풀사이즈 GLM-5.3(755B, `glm_moe_dsa`)을 실행합니다. 요청은 한 번에 하나씩, 빠르게, 에이전트와 코딩에 필요한 긴 컨텍스트로 처리합니다. 엔진은 Zig와 순수 CUDA 커널로 작성되어 있으며, 실행할 때 Python이 필요하지 않습니다.

이것은 [TensorFold](https://github.com/ashhart/TensorFold) 1.0.2를 바탕으로 Fiore-lab이 만든 비공식 파생물이며, TensorFold 프로젝트의 릴리스가 아닙니다. TensorFold의 코드를 포함하고, 풀사이즈 GLM-5.3용 엔진(`zig/src/families/glm_moe_dsa/`, `zig/kernels/cuda/glm53/`)을 추가했으며, TensorFold의 문서는 제외했습니다. 그 문서들은 다른 모델, 가중치, 머신에 대해 설명하기 때문입니다. TensorFold 자체에 대해서는 해당 저장소를 참고하세요.

**AI 어시스턴트를 사용해 이 시스템을 이해하거나 실행하려면** [glm53/release/AI-GUIDE.md](glm53/release/AI-GUIDE.md)(릴리스 아카이브에서는 `AI-GUIDE.md`)를 건네주세요. AI 어시스턴트를 위해 쓴 한 페이지 설명으로, 시스템의 전체 구조, 목적, 준비와 사용 방법, 할 수 있는 것과 할 수 없는 것(실측 수치 포함), 자세한 내용이 있는 곳을 담고 있습니다(영어).

## 주요 기능

- 텐서 병렬 8 또는 4. KV 캐시는 랭크 쌍에 나누어 둡니다(decode 컨텍스트 병렬 2). 엔진과 모델은 최대 1M 토큰(1,048,576)까지 처리할 수 있으며, 컨텍스트 길이는 DGX Spark의 메모리로 정해집니다. **8대에서 1M**(FP8 KV 캐시), **4대에서 약 800K(819,200)**(4비트 KV 캐시)입니다. 설정 예시는 819,200과 737,280으로 되어 있습니다. 같은 메모리로 8대는 `CONTEXT=1048576` `STORE=589824`, 4대는 `CONTEXT=819200` `STORE=18080`으로 설정할 수 있습니다(KV 풀은 `CONTEXT` + `STORE`개의 위치를 보유합니다).
- 긴 프롬프트는 시퀀스 병렬로 나누어 읽고, decode 단계의 집합 통신은 RoCE로 수행합니다.
- 모든 대화가 하나의 대화 캐시를 256개 위치 단위의 블록으로 함께 씁니다. 요청은 어느 대화의 것이든 캐시에 있는 가장 긴 앞부분부터 이어서 처리하며, 같은 내용으로 시작하는 대화(에이전트의 시스템 프롬프트, 그 서브에이전트)는 공통 블록을 한 번만 보유합니다.
- 매 라운드 드래프트를 모델이 검증합니다. 드래프트는 기본적으로 GLM-5.3 자체의 MTP 레이어가 만들며, 선택 사항인 DFlash2 drafter를 쓸 수도 있습니다. 어느 경우든 응답은 모델 자신이 선택한 것입니다.
- `/metrics`(Prometheus 텍스트)와 `/health`는 실행 중인 요청도 집계하므로, 모니터는 요청이 실행되는 동안에도 prefill과 decode 속도, 캐시 적중, 드래프트 채택을 읽을 수 있습니다. `METRICS_LISTEN`을 설정하면 다른 호스트의 모니터를 위해 이 두 가지를 별도의 주소에서 읽기 전용으로 제공합니다.

## Expert Atlas(r3-102의 새 기능)

GLM-5.3은 토큰마다 레이어당 256개의 routed expert 가운데 8개를 고르며, expert마다 선택되는 빈도가 크게 다릅니다. Expert Atlas는 19,200개의 routed expert(레이어 3부터 77까지)를 각각 자기 폭(2, 3, 4 또는 5비트)으로 불러와, 한정된 메모리 안에서 많이 쓰이는 expert일수록 더 많은 비트를 갖게 합니다.

- **뱅크**: 폭별 routed expert로, Hugging Face에 있습니다: `fiore-lab/GLM-5.3-EXL3-Experts-2bit`, `-3bit`, `-4bit`, `-5bit`. 패널은 모든 노드에 갖춰진 폭만 고르게 합니다.
- **컨트롤 패널**: 런처(`./tf-glm53 launcher`)가 제공하는 페이지입니다. 시작 전에는 폭을 계획하고 노드별 메모리, 계획이 들어가는지, KV 캐시에 남는 양을 보여 주며, 엔진의 시작과 정지도 여기서 합니다. 엔진이 도는 동안에는 각 토큰이 고른 expert, 선택된 expert의 평균 폭, 속도를 보여 줍니다.
- **사용 기록**: 엔진은 프롬프트의 각 행과 디코딩된 각 토큰이 고른 expert를 셉니다. 저장한 할당은 저장에 쓰인 실행들의 사용 기록을 함께 가지며, 기록은 할당과 함께 움직입니다. 열어서 다시 실행하면 그 실행분이 기록에 더해집니다. 계획은 이 기록으로 expert를 정렬합니다(기록이 없는 계획은 Fiore-lab이 측정해 페이지에 함께 넣은 횟수로 정렬합니다). 자동으로는 아무것도 저장하지 않습니다. 정지할 때 할당과 그 기록에 이름을 붙여 저장합니다.
- **전문가 최적화**: 폭별 expert 수는 그대로 두고 실행의 사용 기록으로 로드된 expert를 다시 정렬합니다. 같은 메모리에서, 모델이 실제로 고르는 expert의 평균 폭이 올라갑니다. 실행 중인 엔진에서 씁니다(4대에서 75개 레이어 전체 변경에 83–90초, 요청 사이에 수행). 결과는 필요할 때 이름을 붙여 저장하고, 이후 시작에서 엽니다.

컨트롤 패널은 4대 또는 8대에서 동작하며, 대수는 시작 화면에서 고릅니다. `./tf-glm53 up`은 지금까지처럼 체크포인트의 폭으로 시작합니다. 자세한 내용은 `packaging/RUNTIME.md`("The control panel", 영어)와 `glm53/release/T4.md`(영어)에 있습니다.

## 요구 사항

- 하나의 RoCE v2 네트워크로 연결된 DGX Spark(GB10) 4대 또는 8대. 첫 번째 노드에서 모든 노드에 비밀번호 없이 ssh로 접속할 수 있어야 합니다. 4대라면 노드당 100 Gb/s 포트 하나로 충분합니다.
- DGX OS(Ubuntu 24.04), NVIDIA 드라이버 580 이상, cuBLAS 13.
- 모든 노드에 NCCL 2.30.7. NVIDIA의 `nvidia-nccl-cu13` wheel을 pip으로 풀어 놓은 것입니다(`packaging/RUNTIME.md`).
- 노드 수에 맞는 Fiore-lab의 EXL3 체크포인트(Hugging Face). 모든 노드의 같은 경로에 둡니다. 8대는 `fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks`(390 GB), 4대는 `fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks`(344.6 GB)입니다. 4대용 체크포인트는 레이어 3부터 77까지의 routed expert 가운데 사용 빈도가 낮은 절반을 3비트, 나머지를 4비트로 담고 있으며, 그 밖에는 8대용 체크포인트와 같습니다.
- 모든 가중치 파일은 Fiore-lab의 파일 없이 Z.AI의 공식 BF16 체크포인트에서 직접 만들 수도 있습니다(`glm53/weights/README.md`, 영어).
- 컨트롤 패널에는 사용하는 폭의 expert 뱅크가 필요합니다. 4대에서는 3비트와 4비트 뱅크(277.9 GB와 368.5 GB. 8대용 체크포인트를 4비트 뱅크로 쓸 수 있습니다. 4비트만으로는 문맥이 짧아집니다)와, 런처를 위한 각 노드의 `python3`이 필요합니다.

## 실행

릴리스는 컨테이너 이미지와, 같은 파일을 담은 아카이브의 두 가지 형태로 배포합니다. 권장하는 방법은 이미지입니다. 이미지를 쓰면 각 노드에는 Docker와 NVIDIA Container Toolkit(DGX OS에는 둘 다 있습니다)만 있으면 되고, 서버의 라이브러리는 필요하지 않습니다.

```sh
docker pull ghcr.io/fiore-labo/fiore-glm5.3:r3-102            # 각 노드에서
mkdir glm53 && docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/glm53":/out ghcr.io/fiore-labo/fiore-glm5.3:r3-102 install /out
```

그런 다음 `glm53.conf`에 `RUNTIME=docker`와 `IMAGE`를 설정하고 아래와 같이 진행합니다([packaging/RUNTIME.md](packaging/RUNTIME.md) "Running in containers", 영어). Docker를 쓰지 않는 경우: 릴리스 아카이브 `tensorfold-r3-102-linux-aarch64.tar.gz`를 내려받아 함께 들어 있는 `RUNTIME.md`(이 저장소에서는 [packaging/RUNTIME.md](packaging/RUNTIME.md), 영어)를 따르세요. 4대든 8대든 같은 아카이브와 실행 스크립트를 사용합니다. `glm53.conf.example`(8대) 또는 `glm53.t4.conf.example`(4대, [glm53/release/T4.md](glm53/release/T4.md) 참고, 영어)를 `glm53.conf`로 복사하고 노드, 네트워크 이름, NCCL, 가중치, drafter를 채운 다음 `./tf-glm53 deploy`와 `./tf-glm53 up`을 실행합니다. 주소와 API 키 파일을 지정하지 않으면 서버는 `127.0.0.1`에서 OpenAI chat API를 제공합니다.

## Drafter

- `mtp`(기본값): 체크포인트 자체의 MTP 레이어로, GLM-5.3 가중치의 일부입니다. decode 속도는 드래프트를 쓰지 않을 때의 약 2배이며, prefill 속도는 같습니다.
- `dflash2`: incoai/GLM-5.3-DFlash2. **라이선스 때문에 이 저장소에도 아카이브에도 포함하지 않았습니다**. 모델 카드에는 CC BY-NC-ND 4.0(비상업적 이용, 변경 금지), 연구 및 평가용이라고 적혀 있으며, 상업용 라이선스는 Inco AI가 제공합니다. 그 라이선스의 범위 안에서 사용한다면 [packaging/RUNTIME.md](packaging/RUNTIME.md#using-dflash2)의 "Using DFlash2"(영어)를 따르세요. 고정한 리비전으로 한 번만 내려받아 모든 노드의 같은 경로에 복사하고, 복사본을 확인한 다음 `DRAFTER_KIND=dflash2`와 `DRAFTER`를 설정합니다. 엔진은 시작할 때 이를 메모리에서 4비트로 양자화하며, 디스크에는 아무것도 쓰지 않습니다. decode는 더 빨라집니다("속도" 참고).
- `none`: 드래프트를 쓰지 않습니다.

`glm53.conf`의 `DRAFTER_KIND`로 하나를 고릅니다. `./tf-glm53 -d KIND up`으로 그 한 번의 시작에만 다른 것을 고를 수도 있습니다.

## 속도

**모든 노드에서 GPU 클럭을 2.1 GHz(2,100 MHz)로 제한하고 측정했습니다.** 아직 캐시되지 않은 코딩용 프롬프트로 한 번에 요청 하나씩(reasoning effort low, 응답 최대 4,000 토큰), 이 아카이브를 `glm53.conf.example`(8대, FP8 KV 캐시)와 `glm53.t4.conf.example`(4대, 4비트 KV 캐시)의 설정으로, Fiore-lab이 공개한 체크포인트와 함께 사용했습니다. 실제 수치는 체크포인트, 프롬프트, 클럭에 따라 달라집니다. 측정 조건 전체는 `packaging/RUNTIME.md`에 있습니다.

| 노드 수 | 프롬프트 | MTP(k=3): 첫 토큰까지의 시간 / decode | DFlash2: 첫 토큰까지의 시간 / decode |
|---|---|---|---|
| 8 | 128K | 82.8 s / 47.5 tokens/s | 85.2 s / 53.2 |
| 8 | 256K | 172.9 s / 45.7 | 177.6 s / 52.6 |
| 8 | 512K | 376.5 s / 42.8 | 386.0 s / 47.1 |
| 4 | 128K | 120.1 s / 35.3 | 121.9 s / 42.9 |
| 4 | 256K | 253.3 s / 33.2 | 256.9 s / 40.7 |
| 4 | 512K | 565.4 s / 32.2 | 571.5 s / 37.8 |

## 정확도

원래 BF16 모델(공식 BF16 가중치를 공식 구현으로 실행하고 FP32로 계산한 것)의 출력 분포에 대한 이 레시피 출력 분포의 KL 발산이며, 이 레시피의 서빙 구성으로 구했습니다.

- 8대(FP8 KV 캐시, DCP 2. 같은 가중치를 이 엔진의 이전 빌드로 측정), 2,048토큰 텍스트 4개(대화, 법률, 산문, 추론), 각각 마지막 1,023개 위치, 5회 실행: 평균 0.0133(0.0128–0.0146). top-1 토큰의 일치율은 약 96%입니다.
- 8대, 8,192토큰 텍스트 4개(영어 2개, 코드 2개), 위치 범위별: 0.012(위치 1,024–2,047), 0.018, 0.019, 0.013(6,144–8,190). 비교 기준으로, 공식 BF16 가중치를 FP32가 아닌 BF16으로 계산하면 같은 범위에서 같은 기준과의 차이는 0.007–0.010입니다.
- 4대(4대용 체크포인트, 4비트 KV 캐시, DCP 2), 8,192토큰 텍스트 4개, 같은 위치 범위별: 0.019(위치 1,024–2,047), 0.032, 0.029, 0.019(6,144–8,190).
- Expert Atlas의 각 폭(routed expert를 모두 한 폭으로, 그 밖의 가중치는 위와 같음, 같은 텍스트 4개, 위치 6,144–8,190): 2비트 0.0517, 3비트 0.0203, 4비트 0.0127(4대), 5비트 0.0114(8대). 같은 메모리에서 비교하면 2비트 expert를 섞은 배분은 섞지 않은 배분보다 기준에서 13–30% 멀었고, 다른 expert를 3비트로 낮춰 5비트 expert를 더해도 측정할 만한 개선은 없었습니다.

## 소스에서 빌드

Zig 0.17.0이 `PATH`에 있고(또는 `build_libs.sh`에는 `ZIG=`로 그 경로를 지정), CUDA 13(`nvcc`)과 libibverbs 헤더(`libibverbs-dev`)가 있는 GB10 시스템에서 다음을 실행합니다.

```sh
zig build fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 --prefix /tmp/fatbins
sh glm53/tools/release/build_libs.sh /tmp/lib        # lib/libroce_proxy.so
zig build dist -Dversion=r3-102 -Ddist-commit=$(git rev-parse HEAD) -Ddist-platforms=linux-aarch64 \
    -Ddist-fatbins=/tmp/fatbins/fatbin -Ddist-lib=/tmp/lib -Ddist-extra=glm53/release
```

아카이브는 `zig-out/dist/`에 만들어집니다.

## 제한 사항

- GB10 시스템 4대 또는 8대만 지원합니다. 다른 GPU나 다른 랭크 수는 지원하지 않습니다.
- 한 번에 요청 하나만 실행합니다. 다른 요청은 32개까지 도착한 순서대로 기다리고(채팅 제목을 만드는 요청은 맨 뒤), 32개를 넘으면 서버가 거절합니다(503, 스트리밍에서는 오류 이벤트). 서브에이전트를 병렬로 실행하는 에이전트라면 서브에이전트를 다른 모델에 연결하는 편이 좋습니다.
- 그 밖의 제한 사항은 `packaging/RUNTIME.md`에 있습니다.

## 라이선스와 크레딧

코드는 Apache-2.0(`LICENSE`)입니다. 어느 부분이 어디에서 왔는지는 `NOTICE`와 `THIRD_PARTY_NOTICES.md`에 적혀 있습니다. 가중치는 자체 라이선스(GLM-5.3 License)를 따르며, 선택 사항인 DFlash2 drafter도 자체 라이선스를 따릅니다.

- TensorFold 기여자들의 [TensorFold](https://github.com/ashhart/TensorFold): 이 레시피가 확장한 서버, 빌드, CUDA 엔진.
- drowzeys가 만든 TensorFold의 풀사이즈 GLM-5.3용 Python 엔진([ashhart/TensorFold#159](https://github.com/ashhart/TensorFold/pull/159)): GLM-5.3 엔진은 그 설계(엔진의 구조, Triton 커널, MTP와 DFlash2 drafter의 처리 경로)를 Zig와 순수 CUDA로 다시 작성한 것입니다. 그 CUDA 파일 가운데 일부를 포함했으며, 그중 몇 개는 수정했습니다.
- [b12x](https://github.com/local-inference-lab/b12x): 그 RoCE one-shot 집합 통신을 순수 CUDA로 다시 작성했습니다(`roce.cu`). RoCE 프록시는 포함했으며(`zig/native/roce_proxy.c`), RDMA 장치마다 GID 인덱스를 고르도록 수정했습니다.
- [MiaAI-Lab의 GLM-5.3-Flash on 2x DGX Spark](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold): 그 프롬프트 커널을 참고하여 GLM-5.3의 프롬프트 커널을 작성했습니다.
- [MiaAI-Lab의 GLM-5.3 EXL3 on 3x DGX Spark](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold): 4비트 KV 캐시는 그 레이아웃을 채택했으며, 이 엔진 자체의 코드로 구현했습니다.
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3): 체크포인트는 이것으로 양자화했으며, 엔진은 그 EXL3 형식을 읽습니다.

## TensorFold의 기여자

TensorFold의 README에서: TensorFold에 풀 리퀘스트, 측정 결과, 버그 리포트를 보내 주신 모든 분께 감사드립니다(TensorFold 1.0.2 시점의 목록입니다. 최신 목록은 [TensorFold 저장소](https://github.com/ashhart/TensorFold)를 참고하세요).

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
