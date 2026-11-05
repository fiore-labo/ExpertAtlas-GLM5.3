[English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh-CN.md) | 한국어

# DGX Spark 4대 또는 8대로 실행하는 풀사이즈 GLM-5.3

GB10을 1개씩 탑재한 NVIDIA DGX Spark 4대 또는 8대에서 텐서 병렬 4 또는 8로 풀사이즈 GLM-5.3(755B, `glm_moe_dsa`)을 실행합니다. 연결은 RoCE 스위치 1대로 하며, 4대라면 직결 케이블로 만든 링도 가능합니다. 에이전트와 코딩에 필요한 긴 컨텍스트를 빠르게 처리합니다. 요청은 기본적으로 한 번에 하나씩 실행하며, `MULTI`를 쓰면 최대 4개의 대화를 함께 decode합니다. 엔진은 Zig와 순수 CUDA 커널로 작성되어 있으며, 실행할 때 Python이 필요하지 않습니다.

이것은 [TensorFold](https://github.com/ashhart/TensorFold) 1.0.2를 바탕으로 Fiore-lab이 만든 비공식 파생물이며, TensorFold 프로젝트의 릴리스가 아닙니다. TensorFold의 코드를 포함하고, 풀사이즈 GLM-5.3용 엔진(`zig/src/families/glm_moe_dsa/`, `zig/kernels/cuda/glm53/`)을 추가했으며, TensorFold의 문서는 제외했습니다. 그 문서들은 다른 모델, 가중치, 머신에 대해 설명하기 때문입니다. TensorFold 자체에 대해서는 해당 저장소를 참고하세요.

## 문서

각 페이지는 영어, 일본어(`.ja.md`), 간체 중국어(`.zh-CN.md`)로 되어 있으며, 한국어판은 없습니다. 릴리스 아카이브와 이미지에서는 최상위에 함께 있습니다.

| 페이지 | 내용 |
|---|---|
| [QUICKSTART](glm53/release/QUICKSTART.md) | 첫 시작: 가중치, `docker run` 한 줄, 모든 옵션, 자주 나오는 `STOP` 메시지 |
| [BEFORE-START](glm53/release/BEFORE-START.md) | 첫 시작 전에 정할 것과 나중에 정해도 되는 것 |
| [PANEL-GUIDE](glm53/release/PANEL-GUIDE.md) | 컨트롤 패널의 화면을 항목별로 |
| [CONFIG](glm53/release/CONFIG.md) | `glm53.conf`의 모든 설정: 기본값, 읽는 쪽, 변경이 적용되는 시점 |
| [RUNTIME](packaging/RUNTIME.md) | 아카이브나 수동으로 실행하는 방법: 모든 명령, 캐시, Claude Code, 장애 |
| [T4](glm53/release/T4.md) | 4대: 가중치, 설정, 측정 |
| [RING](glm53/release/RING.md) | 스위치 없이 링으로 연결한 4대: 배선, fence가 들어간 NCCL, 측정 |
| [AGENT-API](glm53/release/AGENT-API.md) | AI 에이전트를 위한 제어 API(대시보드 없음) |
| [AI-GUIDE](glm53/release/AI-GUIDE.md) | 시스템 전체를 한 페이지로. 이해나 실행을 돕는 AI 어시스턴트용 |
| [ATLAS-TESTS](glm53/release/ATLAS-TESTS.md) | expert 재정렬의 측정 |
| [weights](glm53/weights/README.md) | 모든 가중치 파일을 Z.AI의 공식 BF16 체크포인트에서 직접 만들기 |

## 주요 기능

- 텐서 병렬 8 또는 4. KV 캐시는 랭크 쌍에 나누어 둡니다(decode 컨텍스트 병렬 2). 엔진과 모델은 컨텍스트 길이 최대 1M 토큰(1,048,576)까지 처리하며, **8대와 4대 모두 1M에 도달합니다**. 8대: 4비트 KV 캐시(`KV=fp4`, `NEED_GB=100`)라면 `CONTEXT_LENGTH=1048576`에 `KV_SIZE=2097152`(노드당 약 23 GiB 여유), 8비트 캐시라면 `KV_SIZE=1638400`. 4대: 컨트롤 패널에서 routed expert의 70 %를 3비트, 나머지를 4비트로 하여 `CONTEXT_LENGTH=1048576`(`KV=fp4`, `KV_SIZE=1049088`, 즉 컨텍스트 길이에 512를 더한 KV 풀의 최솟값. `ROWS` 16,384에서 노드당 약 7.5 GiB 여유가 있어 실행 중 재정렬도 가능합니다). 기본 시작은 노드 수와 관계없이 `CONTEXT_LENGTH=1048576`을 요청하며, 공간이 부족하면 줄이고 그 사실을 표시합니다. 8대의 4비트 KV 캐시와 4대의 컨트롤 패널 시작은 각각 약 1,019,000 토큰의 프롬프트 1개로 실측했습니다. 4비트 KV 캐시는 정확도가 조금 낮아집니다(같은 가중치에서 KL 0.0190, 8비트 캐시는 0.0156: [glm53/release/T4.md](glm53/release/T4.md), 영어). 아래 속도에서 8대는 8비트 캐시입니다. 설정 예시는 컨텍스트 길이 819,200과 1,048,576, KV 크기 1,638,400과 1,049,088으로 되어 있습니다(KV 크기는 KV 캐시 전체이며, 모든 대화가 함께 쓰는 하나의 풀입니다. 요청 하나가 컨텍스트 길이까지 쓰고, 나머지는 이전 대화의 KV를 비어 있는 아무 블록에나 남겨 둡니다).
- 긴 프롬프트는 시퀀스 병렬로 나누어 읽고, decode 단계의 집합 통신은 RoCE로 수행합니다.
- 모든 대화가 하나의 대화 캐시를 256개 위치 단위의 블록으로 함께 씁니다. 요청은 어느 대화의 것이든 캐시에 있는 가장 긴 앞부분부터 이어서 처리하며, 같은 내용으로 시작하는 대화(에이전트의 시스템 프롬프트, 그 서브에이전트)는 공통 블록을 한 번만 보유합니다.
- 매 라운드 드래프트를 모델이 검증합니다. 드래프트는 기본적으로 GLM-5.3 자체의 MTP 레이어가 만들며, 선택 사항인 DFlash2 drafter를 쓸 수도 있습니다. 어느 경우든 응답은 모델 자신이 선택한 것입니다.
- 사고(thinking)가 지나치게 길어지지 않게 합니다. 상한은 `max_tokens`(1회 응답의 상한)에 대한 비율로 정하고, 긴 사고는 중간이나 끝에서 정리할 수 있습니다. 요청마다 설정할 수도, 엔진 실행 중에 설정할 수도 있습니다.
- `/metrics`(Prometheus 텍스트)와 `/health`는 실행 중인 요청도 집계하므로, 모니터는 요청이 실행되는 동안에도 prefill과 decode 속도, 캐시 적중, 드래프트 채택을 읽을 수 있습니다. `METRICS_LISTEN`을 설정하면 다른 호스트의 모니터를 위해 이 두 가지를 별도의 주소에서 읽기 전용으로 제공합니다.

## r3.3-102에서 바뀐 점

**조치가 필요한 변경:**

- **내려받을 가중치가 바뀌었습니다.** `fiore-lab/GLM-5.3-EXL3-Base`(21.2 GB)와 3비트 뱅크(277.9 GB)입니다(4대든 8대든 같습니다). 시작할 때 routed expert를 모두 뱅크에서 읽으므로 전체 체크포인트는 더 이상 필요하지 않습니다. 이미 있다면 `WEIGHTS`로 지정해 쓸 수 있습니다. 계획 없이 시작하면 모든 expert를 3비트로 불러옵니다. `BITS`나 저장한 할당으로 다른 폭을 고르면 그 폭의 뱅크도 필요합니다.
- **설정 이름이 바뀌었습니다.** `CONTEXT`와 `STORE` 대신 `CONTEXT_LENGTH`(요청 하나의 최대 길이)와 `KV_SIZE`(KV 캐시 전체)를 씁니다. 예전 `glm53.conf`도 그대로 읽습니다.
- **응답의 기본 상한**이 4,096 토큰에서 context length의 70 %(`MAX_TOKENS_SHARE`)로 바뀌었습니다. 클라이언트의 `max_tokens`가 우선합니다.
- **설정의 선택지를 정리했습니다.** `CACHE`는 `pool`, `DCP`는 2, `ROCE_PROXY_CORES`는 `a725` 또는 `os`입니다.
- **`./tf-glm53 up`은 컨트롤 패널만 시작합니다.** 엔진은 패널이나 `./tf-glm53 start`로 시작합니다. `./tf-glm53 engine-up`은 엔진을 수동으로 시작합니다.

**새로운 점:**

- **한 줄로 엔진 실행:** `docker run ... IMAGE up`은 첫 번째 노드에 릴리스를 쓰고, 노드를 찾고, 가중치를 찾고, 엔진을 시작한 뒤 요청을 하나 보냅니다. 마지막에는 주소와 함께 `READY`를 표시하거나, 다음에 할 일을 알려 주는 `STOP`, `FIX`, `THEN`을 표시합니다. 엔진 설정 화면의 모든 설정은 `-e NAME=VALUE` 옵션입니다([QUICKSTART](glm53/release/QUICKSTART.md), 영어).
- **4대를 링으로 연결**할 수 있으며 스위치가 필요 없습니다([RING](glm53/release/RING.md), 영어).
- **이미지에 AArch64 fence가 들어간 NCCL:** 스위치와 링 모두에서 사용합니다. NVIDIA/nccl#2393의 수정으로, NCCL 릴리스에는 아직 들어 있지 않습니다. 이것이 없으면 Arm의 프록시 스레드가 오류 없이 멈출 수 있습니다([RING](glm53/release/RING.md), 영어).
- **노드 확인**은 배선에서 링이나 스위치의 노드를 찾아 확인하고 등록하므로(`glm53.nodes.conf`), `TOPOLOGY`, `SOCKET_IFNAME`, `IB_HCA`를 쓸 필요가 없습니다.
- **Reasoning Control:** 사고 상한(`max_tokens`에 대한 비율), 중간 정리, 사고 종료 시 정리를 엔진 실행 중에 패널이나 `POST /v1/thinking`으로 바꿉니다([CONFIG](glm53/release/CONFIG.md), 영어).
- **여러 대화를 동시에:** `MULTI`(1–4)가 이제 MTP 레이어에서도 동작합니다. 실행 중에 패널이나 `POST /v1/schedule`로 prefill과 decode의 GPU 배분을 정할 수 있고, 짧은 프롬프트를 먼저 처리할 수 있습니다.
- **Byte Resume**(기본값 켜짐): 모델이 토크나이저와 다른 위치에서 텍스트를 자르면, 다음 턴의 프롬프트에서 같은 바이트가 다른 토큰 ID가 됩니다. 이제 캐시는 나머지를 다시 계산하지 않고 그 위치를 넘어 이어 갑니다.
- **패널:** 엔진 설정 화면, 가중치 위치와 그 확인([ WEIGHTS ]), 시작 흐름, 원 위의 KV 풀, 각 노드의 온도.
- **AI 에이전트를 위한 제어 API:** 대시보드 없이 엔진을 시작, 정지, 설정합니다([AGENT-API](glm53/release/AGENT-API.md), 영어).
- **더 빠르고 더 시원하게:** 가중치를 병렬로 불러옵니다(링에서 측정, 준비 완료까지 366초가 127초로). MTP는 드래프트의 신뢰도가 떨어지면 그 라운드의 드래프트를 멈춥니다(`TF_GLM53_MTP_CONF`, 기본값 0.7). RoCE 프록시 스레드는 Cortex-A725 코어에 머무르며(`ROCE_PROXY_CORES`), 같은 속도에서 SoC 온도가 5–16 °C 낮아졌습니다.

## r3.2-102에서 바뀐 점

- 엔진이 동작하는 동안 디스크에는 아무것도 쓰지 않습니다. 각 rank, 감시 프로세스, 런처의 로그는 각 노드의 로그 보관 프로세스(`tools/logkeep.py`. 각 노드에 `python3`가 필요하며 DGX OS에는 들어 있습니다. 아카이브든 이미지든 같습니다)의 메모리에 로그마다 최근 `LOG_KEEP_MB`(8)만 두고, `./tf-glm53 logs R`로 읽습니다. `RUNTIME=docker`에서는 컨테이너도 로그를 남기지 않습니다. r3.1-102는 사용자가 linger일 때만 로그를 메모리에 두었고, r3.2-102는 linger가 필요 없습니다.
- 기록은 다른 호스트의 모니터가 보관합니다. `LOG_LISTEN=HOST:PORT`를 설정하면 각 노드의 보관 프로세스가 읽기 전용·키 없이 로그를 돌려주고(`GET /streams`, `GET /logs`), 메트릭 대기(`METRICS_LISTEN`)는 expert 사용 횟수도 돌려줍니다(`GET /v1/experts/usage`)([packaging/RUNTIME.md](packaging/RUNTIME.md) "Serving other machines", 영어).

## r3.1-102에서 바뀐 점

- API와 컨트롤 패널은 기본적으로 모든 주소(`HOST=0.0.0.0`, `LAUNCHER_HOST=0.0.0.0`)에서 키 없이 대기합니다. 네트워크의 다른 머신에서 바로 접근할 수 있습니다. 노드는 닫힌 네트워크 안에 두세요(`HOST=127.0.0.1` / `LAUNCHER_HOST=127.0.0.1`이면 첫 번째 노드에서만 사용할 수 있습니다).
- Messages API로 Claude Code를 쓸 수 있습니다. `claude-glm53`이 Claude Code를 서버에 연결해 시작하고, 서버는 그 대화를 캐시에 유지하고 긴 프롬프트의 스트림을 유지하며 이미지를 메모로 바꿉니다([packaging/RUNTIME.md](packaging/RUNTIME.md) "Claude Code", 영어).
- 컨트롤 패널은 `./tf-glm53 up`으로 시작한 엔진에서도 동작합니다(엔진은 기본적으로 expert 사용 횟수를 셉니다). 실행 중 재정렬은 커지지 않는 층부터 바꾸고, 시작하기 전에 교체 도중의 메모리 최대치로 모든 노드를 확인합니다. 메모리가 부족한 노드는 패널이 알려 줍니다.
- `glm53.conf`의 모든 설정 항목: [glm53/release/CONFIG.md](glm53/release/CONFIG.md)(영어).

## Expert Atlas(r3-102의 새 기능)

GLM-5.3은 토큰마다 레이어당 256개의 routed expert 가운데 8개를 고르며, expert마다 선택되는 빈도가 크게 다릅니다. Expert Atlas는 19,200개의 routed expert(레이어 3부터 77까지)를 각각 자기 폭(2, 3, 4 또는 5비트)으로 불러와, 한정된 메모리 안에서 많이 쓰이는 expert일수록 더 많은 비트를 갖게 합니다.

- **뱅크**: 폭별 routed expert로, Hugging Face에 있습니다: [fiore-lab/GLM-5.3-EXL3-Experts-2bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-2bit), [-3bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-3bit), [-4bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-4bit), [-5bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-5bit). 패널은 모든 노드에 갖춰진 폭만 고르게 합니다.
- **컨트롤 패널**: 런처(`./tf-glm53 up`으로 시작)가 제공하는 페이지입니다. 시작 전에는 폭을 계획하고 노드별 메모리, 계획이 들어가는지, 계획이 남기는 KV 크기를 보여 주며, 엔진의 시작과 정지도 여기서 합니다. 엔진이 도는 동안에는 각 토큰이 고른 expert, 선택된 expert의 평균 폭, 속도를 보여 줍니다.
- **사용 기록**: 엔진은 프롬프트의 각 행과 디코딩된 각 토큰이 고른 expert를 셉니다. 저장한 할당은 저장에 쓰인 실행들의 사용 기록을 함께 가지며, 기록은 할당과 함께 움직입니다. 열어서 다시 실행하면 그 실행분이 기록에 더해집니다. 계획은 이 기록으로 expert를 정렬합니다(기록이 없는 계획은 Fiore-lab이 측정해 페이지에 함께 넣은 횟수로 정렬합니다). 자동으로는 아무것도 저장하지 않습니다. 정지할 때 할당과 그 기록에 이름을 붙여 저장합니다.
- **전문가 최적화**: 폭별 expert 수는 그대로 두고 실행의 사용 기록으로 로드된 expert를 다시 정렬합니다. 같은 메모리에서, 모델이 실제로 고르는 expert의 평균 폭이 올라갑니다. 실행 중인 엔진에서 씁니다(4대에서 75개 레이어 전체 변경에 83–90초, 요청 사이에 수행). 결과는 필요할 때 이름을 붙여 저장하고, 이후 시작에서 엽니다.

컨트롤 패널은 4대 또는 8대에서 동작하며, 대수는 시작 화면에서 고릅니다. 패널 없이 시작할 때는 `./tf-glm53 start`와 `./tf-glm53 engine-up`이 직전 시작의 계획으로, 또는 모든 expert를 3비트로 하여 시작합니다. 자세한 내용은 [PANEL-GUIDE](glm53/release/PANEL-GUIDE.md)(영어), `packaging/RUNTIME.md`("The control panel", 영어), `glm53/release/T4.md`(영어)에 있습니다.

## Hugging Face의 가중치

| 항목 | 저장소 | 크기 | 용도 |
|---|---|---|---|
| Base 가중치: routed expert를 뺀 전부 | [fiore-lab/GLM-5.3-EXL3-Base](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Base) | 21.2 GB | **모든 시작에 필수**(4대든 8대든 같습니다) |
| 전문가 뱅크 2비트 | [fiore-lab/GLM-5.3-EXL3-Experts-2bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-2bit) | 187.3 GB | 선택: 2비트 expert를 쓸 때 |
| 전문가 뱅크 3비트 | [fiore-lab/GLM-5.3-EXL3-Experts-3bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-3bit) | 277.9 GB | **모든 시작에 필수** |
| 전문가 뱅크 4비트 | [fiore-lab/GLM-5.3-EXL3-Experts-4bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-4bit) | 368.5 GB | 선택: 4비트 expert를 쓸 때(8대용 체크포인트가 있으면 불필요) |
| 전문가 뱅크 5비트 | [fiore-lab/GLM-5.3-EXL3-Experts-5bit](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Experts-5bit) | 459.1 GB | 선택: 5비트 expert를 쓸 때 |
| 8대용 체크포인트: 모든 가중치(routed expert는 4비트) | [fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks) | 390 GB | 선택: 시작에는 필요 없습니다. `WEIGHTS`로 지정하면 그대로 쓸 수 있고 4비트 뱅크도 겸합니다 |
| 4대용 체크포인트: 모든 가중치(사용 빈도가 낮은 절반의 routed expert는 3비트) | [fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks) | 344.6 GB | 선택: 시작에는 필요 없습니다. `WEIGHTS`로 지정하면 그대로 쓸 수 있습니다 |

뱅크는 19,200개의 routed expert를 하나의 폭으로 모은 것입니다. 엔진은 expert마다 계획에서 정한 폭의 뱅크에서
불러옵니다. **모든 시작에는 Base 가중치와 3비트 뱅크가 필요하며(4대든 8대든 합계 299.1 GB), 다른 뱅크는 사용하려는 폭만 내려받으면 됩니다**(패널은 모든 노드에 갖춰진 폭만 고르게 합니다).
모든 노드에 같은 파일을 같은 경로에 둡니다. 한 대에서
내려받아 다른 노드로 복사합니다(`python3 -m venv`에는 `python3-venv` 패키지가 필요합니다). 예:

```sh
python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install -U huggingface_hub
~/hf-cli/bin/hf download fiore-lab/GLM-5.3-EXL3-Base --local-dir ~/models/GLM-5.3-EXL3-Base
~/hf-cli/bin/hf download fiore-lab/GLM-5.3-EXL3-Experts-3bit --local-dir ~/models/GLM-5.3-EXL3-Experts-3bit
```

다른 노드로 복사하는 방법은 [QUICKSTART](glm53/release/QUICKSTART.md)(영어)에, 검증(`./tf-glm53 verify`, `./tf-glm53 verify-banks`)은 [packaging/RUNTIME.md](packaging/RUNTIME.md)(영어)에
있습니다.

## 가중치 직접 만들기

가중치는 코드처럼 읽을 수 없습니다. 특정 상황에서만 다르게 동작하도록 손댄 가중치도 겉보기에는 다른 가중치와 같으므로, 만들어진
과정을 확인할 수 없는 가중치에 의존하고 싶지 않을 수 있습니다. Fiore-lab의 가중치에 의존할 필요는 없습니다. 이 레시피가 읽는 모든 가중치 파일(Base 가중치, 8대용·4대용 체크포인트와
2·3·4·5비트 expert 뱅크)은 Z.AI의 공식 BF16 체크포인트([zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16))에서
[glm53/weights](glm53/weights/README.md)(영어)의 도구로 만들 수 있으며, 절차·하드웨어·실측 시간이 그곳에 있습니다. 직접 만든 파일은
Fiore-lab의 것과 구조가 같고 같은 방식으로 로드됩니다.

## 요구 사항

- 하나의 RoCE v2 네트워크로 연결된 DGX Spark(GB10) 4대 또는 8대, 또는 링으로 연결한 4대. 첫 번째 노드에서 모든 노드와 자기 자신에 패스프레이즈 없는 키로 ssh로 접속할 수 있어야 하고, 호스트 키도 미리 등록해 둡니다. ssh와 ConnectX-7 네트워크 절차는 [QUICKSTART](glm53/release/QUICKSTART.md)(영어) 1장에 있습니다. 스위치에 연결한 4대라면 노드당 100 Gb/s 포트 하나로 충분합니다.
- 이미지를 쓰는 경우: 각 노드에 NVIDIA 드라이버 580 이상(이미지는 호스트의 드라이버를 씁니다), Docker와 NVIDIA Container Toolkit(DGX OS에는 둘 다 있습니다), `python3`. 시작 전에 각 노드에 약 115 GiB의 여유 메모리.
- 대신 아카이브를 쓰는 경우: DGX OS(Ubuntu 24.04), NVIDIA 드라이버 580 이상, cuBLAS 13. 모든 노드에 NCCL 2.30.7. NVIDIA의 `nvidia-nccl-cu13` wheel을 pip으로 풀어 놓은 것입니다(`packaging/RUNTIME.md`).
- 아카이브에서는 AArch64 fence가 들어간 NCCL을 스위치에서는 권장하고 링에서는 필수로 합니다. NVIDIA의 wheel에는 없습니다([RING](glm53/release/RING.md), 영어). 이미지에는 들어 있습니다.
- Fiore-lab의 EXL3 Base 가중치(Hugging Face). 모든 노드의 같은 경로에 둡니다. [fiore-lab/GLM-5.3-EXL3-Base](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-Base)(21.2 GB)이며 4대든 8대든 같습니다. 전체 체크포인트, [fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks)(390 GB)나 [fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks)(344.6 GB)를 대신 쓸 수도 있지만, 시작할 때 그 routed expert는 읽지 않습니다.
- 모든 가중치 파일은 Fiore-lab의 파일 없이 Z.AI의 공식 BF16 체크포인트에서 직접 만들 수도 있습니다([glm53/weights](glm53/weights/README.md), 영어).
- 3비트 뱅크(277.9 GB). 모든 노드의 같은 경로에 둡니다. 모든 시작에 필요합니다. 다른 뱅크는 사용하는 폭만 있으면 됩니다(8대용 체크포인트를 4비트 뱅크로 쓸 수 있습니다. 4비트만으로는 컨텍스트 길이가 짧아집니다).
- 런처와 로그 보관 프로세스를 위한 각 노드의 `python3`.

## 실행

릴리스는 컨테이너 이미지와, 같은 파일을 담은 아카이브의 두 가지 형태로 배포합니다. 권장하는 방법은 이미지입니다. 가중치를 준비해 두면 첫 번째 노드에서 다음 한 줄로 엔진이 시작됩니다.

```sh
docker run --rm --network host -e USER="$USER" -v "$HOME":/host ghcr.io/fiore-labo/expertatlas-glm5.3:r3.3-102 up
```

마지막에 주소와 함께 `READY`가 표시됩니다. API는 `http://FIRST-NODE:8000/v1`, 컨트롤 패널은 `http://FIRST-NODE:8010/panel`입니다. 부족한 것이 있으면 `STOP`으로 끝나고, 할 일과 다음에 실행할 줄을 알려 줍니다. 절차와 모든 옵션은 [QUICKSTART](glm53/release/QUICKSTART.md)(영어)에 있습니다.

같은 결과를 수동으로 얻으려면, 이미지의 경우 모든 노드에서 `docker pull`하고 `mkdir glm53 && docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/glm53":/out IMAGE install /out`을 실행한 뒤, `glm53.conf`에 `RUNTIME=docker`와 `IMAGE`를 설정합니다([packaging/RUNTIME.md](packaging/RUNTIME.md) "Running in containers", 영어). Docker를 쓰지 않는 경우: 릴리스 아카이브 `tensorfold-r3.3-102-linux-aarch64.tar.gz`를 내려받아 함께 들어 있는 `RUNTIME.md`(이 저장소에서는 [packaging/RUNTIME.md](packaging/RUNTIME.md), 영어)를 따르세요. 어느 쪽이든 `glm53.conf.example`(8대) 또는 `glm53.t4.conf.example`(4대, [glm53/release/T4.md](glm53/release/T4.md) 참고, 영어)를 `glm53.conf`로 복사하고 요구하는 항목을 채운 다음 `./tf-glm53 deploy`와 `./tf-glm53 up`(컨트롤 패널을 시작합니다)을 실행하고, 거기서 엔진을 시작합니다. `./tf-glm53 start`는 직전 시작의 계획으로 시작하고, `./tf-glm53 down`은 정지합니다.

서버는 기본적으로 모든 주소(`HOST=0.0.0.0`)에서 OpenAI chat API와, Claude Code를 위한 Anthropic Messages API를 제공합니다. API에도 컨트롤 패널에도 키가 없으므로 접근할 수 있는 누구나 사용할 수 있습니다. 노드를 닫힌 네트워크 안에 두거나 앞에 TLS와 접근 제어를 두세요(`HOST=127.0.0.1`과 `LAUNCHER_HOST=127.0.0.1`이면 첫 번째 노드에서만 사용할 수 있습니다).

## Drafter

- `mtp`(기본값): 가중치 안의 MTP 레이어로, GLM-5.3 가중치의 일부입니다. decode 속도는 드래프트를 쓰지 않을 때의 약 2배이며, prefill 속도는 같습니다.
- `dflash2`: incoai/GLM-5.3-DFlash2. **라이선스 때문에 이 저장소에도 아카이브에도 포함하지 않았습니다**. 모델 카드에는 CC BY-NC-ND 4.0(비상업적 이용, 변경 금지), 연구 및 평가용이라고 적혀 있으며, 상업용 라이선스는 Inco AI가 제공합니다. 그 라이선스의 범위 안에서 사용한다면 [packaging/RUNTIME.md](packaging/RUNTIME.md#using-dflash2)의 "Using DFlash2"(영어)를 따르세요. 고정한 리비전으로 한 번만 내려받아 모든 노드의 같은 경로에 복사하고, 복사본을 확인한 다음 `DRAFTER_KIND=dflash2`와 `DRAFTER`를 설정합니다. 엔진은 시작할 때 이를 메모리에서 4비트로 양자화하며, 디스크에는 아무것도 쓰지 않습니다. decode는 더 빨라집니다("속도" 참고).
- `none`: 드래프트를 쓰지 않습니다.

`glm53.conf`의 `DRAFTER_KIND`로 하나를 고릅니다. 시작 화면에서 고르면 거기에 기록합니다. `./tf-glm53 -d KIND start`(또는 `engine-up`)는 그 한 번의 시작에만 고릅니다.

## 속도

**릴리스 r2-102로, 모든 노드에서 GPU 클럭을 2.1 GHz(2,100 MHz)로 제한하고 측정했습니다.** 그 후 엔진이 바뀌었으며, 이 수치는 이번 릴리스에서 다시 측정하지 않았습니다. 아직 캐시되지 않은 코딩용 프롬프트로 한 번에 요청 하나씩(reasoning effort low, 응답 최대 4,000 토큰), `glm53.conf.example`(8대, FP8 KV 캐시)와 `glm53.t4.conf.example`(4대, 4비트 KV 캐시. `ROWS`는 당시 16,384, 지금은 8,192)의 설정으로, Fiore-lab이 공개한 체크포인트를 각각의 폭 그대로 사용했습니다(8대는 모든 routed expert가 4비트, 4대는 절반이 3비트, 절반이 4비트). 기본 시작은 모든 expert를 3비트로 불러오며, 이 표는 그 구성을 측정한 것이 아닙니다. 속도는 환경(체크포인트, 프롬프트, 클럭 등)에 따라 다르므로, 이 수치는 참고용입니다. 측정 조건 전체는 `packaging/RUNTIME.md`에 있습니다.

| 노드 수 | 프롬프트 | MTP(k=3): 첫 토큰까지의 시간 / decode | DFlash2: 첫 토큰까지의 시간 / decode |
|---|---|---|---|
| 8 | 128K | 82.8 s / 47.5 tokens/s | 85.2 s / 53.2 |
| 8 | 256K | 172.9 s / 45.7 | 177.6 s / 52.6 |
| 8 | 512K | 376.5 s / 42.8 | 386.0 s / 47.1 |
| 4 | 128K | 120.1 s / 35.3 | 121.9 s / 42.9 |
| 4 | 256K | 253.3 s / 33.2 | 256.9 s / 40.7 |
| 4 | 512K | 565.4 s / 32.2 | 571.5 s / 37.8 |

이번 릴리스에서 4대로 `ROWS`별로 한 번씩 측정한 prefill은 8,192에서 1,054 토큰/초, 16,384에서 1,102, 32,768에서 1,110입니다. 참고용이며, 조건은 [glm53/release/T4.md](glm53/release/T4.md)(영어)에 있습니다.

## 정확도

원래 BF16 모델(공식 BF16 가중치를 공식 구현으로 실행하고 FP32로 계산한 것)의 출력 분포에 대한 이 레시피 출력 분포의 KL 발산이며, 이 레시피의 서빙 구성으로 구했습니다.

- 8대(8대용 체크포인트, 모든 routed expert가 4비트, FP8 KV 캐시, DCP 2. 같은 가중치를 이 엔진의 이전 빌드로 측정), 2,048토큰 텍스트 4개(대화, 법률, 산문, 추론), 각각 마지막 1,023개 위치, 5회 실행: 평균 0.0133(0.0128–0.0146). top-1 토큰의 일치율은 약 96%입니다.
- 8대, 8,192토큰 텍스트 4개(영어 2개, 코드 2개), 위치 범위별: 0.012(위치 1,024–2,047), 0.018, 0.019, 0.013(6,144–8,190). 비교 기준으로, 공식 BF16 가중치를 FP32가 아닌 BF16으로 계산하면 같은 범위에서 같은 기준과의 차이는 0.007–0.010입니다.
- 4대(4대용 체크포인트의 폭 그대로, 절반이 3비트, 절반이 4비트, 4비트 KV 캐시, DCP 2), 8,192토큰 텍스트 4개, 같은 위치 범위별: 0.019(위치 1,024–2,047), 0.032, 0.029, 0.019(6,144–8,190).
- Expert Atlas의 각 폭(routed expert를 모두 한 폭으로, 그 밖의 가중치는 위와 같음, 같은 텍스트 4개, 위치 6,144–8,190): 2비트 0.0517, 3비트 0.0203, 4비트 0.0127(4대), 5비트 0.0114(8대). 기본 시작은 모든 expert를 3비트로 불러옵니다. 같은 메모리에서 비교하면 2비트 expert를 섞은 배분은 섞지 않은 배분보다 기준에서 13–30% 멀었고, 다른 expert를 3비트로 낮춰 5비트 expert를 더해도 측정할 만한 개선은 없었습니다.

## 소스에서 빌드

Zig 0.17.0이 `PATH`에 있고(또는 `build_libs.sh`에는 `ZIG=`로 그 경로를 지정), CUDA 13(`nvcc`)과 libibverbs 헤더(`libibverbs-dev`)가 있는 GB10 시스템에서 다음을 실행합니다.

```sh
zig build fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 --prefix /tmp/fatbins
sh glm53/tools/release/build_libs.sh /tmp/lib        # lib/libroce_proxy.so
zig build dist -Dversion=r3.3-102 -Ddist-commit=$(git rev-parse HEAD) -Ddist-platforms=linux-aarch64 \
    -Ddist-fatbins=/tmp/fatbins/fatbin -Ddist-lib=/tmp/lib -Ddist-extra=glm53/release
```

아카이브는 `zig-out/dist/`에 만들어집니다.

## 제한 사항

- GB10 시스템 4대 또는 8대를 지원합니다. 2대와 3대 구성은 테스트 중입니다. 다른 GPU나 다른 랭크 수는 지원하지 않습니다.
- 기본적으로 한 번에 요청 하나만 실행합니다. 다른 요청은 32개까지 도착한 순서대로 기다리고(채팅 제목을 만드는 요청은 맨 뒤), 32개를 넘으면 서버가 거절합니다(503, 스트리밍에서는 오류 이벤트). `MULTI`를 쓰면 최대 4개를 함께 decode하지만 노드 메모리가 더 듭니다. 그 밖의 경우, 서브에이전트를 병렬로 실행하는 에이전트라면 서브에이전트를 다른 모델에 연결하는 편이 좋습니다.
- 컨트롤 패널의 화면은 영어와 일본어만 있습니다.
- 그 밖의 제한 사항은 `packaging/RUNTIME.md`에 있습니다.

## 라이선스와 크레딧

코드는 Apache-2.0(`LICENSE`)입니다. 어느 부분이 어디에서 왔는지는 `NOTICE`와 `THIRD_PARTY_NOTICES.md`에 적혀 있습니다. 가중치는 자체 라이선스(GLM-5.3 License)를 따르며, 선택 사항인 DFlash2 drafter도 자체 라이선스를 따릅니다.

- TensorFold 기여자들의 [TensorFold](https://github.com/ashhart/TensorFold): 이 레시피가 확장한 서버, 빌드, CUDA 엔진.
- drowzeys가 만든 TensorFold의 풀사이즈 GLM-5.3용 Python 엔진([ashhart/TensorFold#159](https://github.com/ashhart/TensorFold/pull/159)): GLM-5.3 엔진은 그 설계(엔진의 구조, Triton 커널, MTP와 DFlash2 drafter의 처리 경로)를 Zig와 순수 CUDA로 다시 작성한 것입니다. 그 CUDA 파일 가운데 일부를 포함했으며, 그중 몇 개는 수정했습니다.
- [b12x](https://github.com/local-inference-lab/b12x): 그 RoCE one-shot 집합 통신을 순수 CUDA로 다시 작성했습니다(`roce.cu`). RoCE 프록시는 포함했으며(`zig/native/roce_proxy.c`), RDMA 장치마다 GID 인덱스를 고르도록 수정했습니다.
- [MiaAI-Lab의 GLM-5.3-Flash on 2x DGX Spark](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold): 그 프롬프트 커널을 참고하여 GLM-5.3의 프롬프트 커널을 작성했습니다.
- [MiaAI-Lab의 GLM-5.3 EXL3 on 3x DGX Spark](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold): 4비트 KV 캐시는 그 레이아웃을 채택했으며, 이 엔진 자체의 코드로 구현했습니다.
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3): 체크포인트는 이것으로 양자화했으며, 엔진은 그 EXL3 형식을 읽습니다.
- [gr8 레시피](https://github.com/knapcio/GLM-5.3-gr8-4x-DGX-Spark-TP4)와 [엔진](https://github.com/kindlingai/gr8): decode 고속화 몇 가지는 그 아이디어를 채택했으며, 이 엔진 자체의 코드로 구현했습니다.
- [ollama/ollama#12375](https://github.com/ollama/ollama/issues/12375): Byte Resume의 아이디어는 이 보고에서 가져왔습니다.

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
