# 📘 ECMiner™ Preprocessing Pipeline  
**DOCX → Sanitize → Image/Table Summary → Chunking → Chunk+Summary 통합 RAG 입력 데이터 생성**

이 프로젝트는 ECMiner™ 소프트웨어 매뉴얼 같은 **복잡한 DOCX 문서**를  
**RAG 시스템에서 바로 사용할 수 있는 구조**로 자동 변환하기 위한 전처리 파이프라인이다.

---

## ⭐ 주요 기능

1. **DOCX 파싱 & 정제**
   - 단락, 표, 리스트, 이미지 등 구조 정보 추출
   - `_sanitized` JSON 및 `_assets` 이미지 폴더 생성

2. **이미지 Semantic Summary (Qwen-VL)**
   - 문서 속 인라인 이미지/스크린샷들을 Qwen-VL로 요약
   - `_chunked/{docname}_image_llm.json` 생성

3. **테이블 HTML Summary (Qwen Text LLM) 🔥**
   - Sanitized JSON 내 테이블 HTML을 Qwen 텍스트 LLM에 넣어 요약
   - `_chunked/{docname}_table_llm.json` 생성

4. **Chunking**
   - Sanitized JSON을 문단 기반으로 잘라 `_chunked/{docname}_chunked.json` 생성
   - Context 안에는 `[image:rId#]`, `[table:tbl#]` 같은 placeholder가 포함됨

5. **이미지 + 테이블 요약 머지**
   - chunk의 `Context` 안에 있는 `[image:...]`, `[table:...]` 자리 뒤에  
     각각 이미지/테이블 요약을 인라인으로 삽입
   - 최종적으로 `_chunked/{docname}_chunked_with_imgsum.json` 생성  
   - 이 파일이 **RAG에 바로 넣을 최종 입력 데이터**

6. **GPU 기반 Qwen2.5-VL-3B-Instruct + Qwen 텍스트 LLM 사용 가능**
   - `start_preprocessing.sh`가 GPU 버전 torch/torchvision 설치 및 모델 다운로드까지 자동 처리

---

## 📂 프로젝트 구조 (요약)

```bash
.
├── README.md
├── run_full_preprocess.py              # 전체 DOCX → 최종 chunk_with_imgsum까지 자동 실행
├── start_preprocessing.sh              # 환경 세팅 + 모델 다운로드 (GPU)
├── merge_image_summaries_into_chunks.py# 이미지+테이블 요약을 chunk에 인라인 머지 (CLI)
├── config/
├── infra/
├── output/
│   └── processed/
│       └── {docname}/v0/
│           ├── _sanitized/
│           ├── _chunked/
│           └── _comp/
└── src/
    └── mypkg/
        ├── pipelines/
        │   ├── docx_parsing_pipeline.py
        │   ├── img_summary_gen.py           # 이미지 요약 파이프라인
        │   ├── table_summary_gen.py         # 🔥 테이블 요약 파이프라인
        │   └── chunking.py
        ├── components/
        └── utils/
````

---

## 🚀 환경 설정 (GPU + Qwen 모델)

### 1) 환경 세팅 스크립트 실행

```bash
cd /root/ecm-preprocess-1   # 프로젝트 루트
bash start_preprocessing.sh
```

이 스크립트는 다음을 자동으로 수행한다:

* 시스템 패키지 설치: `python3`, `git`, `curl`, **Tesseract OCR + kor 언어팩**, 등
* **Git-Xet 설치**

  * Hugging Face 공식 스크립트 사용
  * `git xet install --system`
* Python 가상환경 `.venv` 생성
* **GPU 버전 torch + torchvision** 설치 (CUDA 12.1 기준)
* `transformers`, `huggingface_hub`, `accelerate`, `safetensors`, `einops`, `sentencepiece`
* OCR용 `pillow`, `pytesseract`, HTTP용 `httpx`, `hf_transfer`
* **Qwen2.5-VL-3B-Instruct** 모델을 `/workspace/qwen`에 다운로드
* `.venv/bin/activate`에 환경 변수 자동 주입:

  * `QWEN_MODEL_PATH=/workspace/qwen`
  * `PYTHONPATH=<project_root>/src`

---

### 2) 가상환경 활성화

```bash
source .venv/bin/activate
```

이제부터는 `python` 실행 시 전부 `.venv` 환경을 사용하게 된다.

---

## 🧠 파이프라인 단계별 설명

### STEP 1. DOCX → Sanitized JSON (`DocxParsingPipeline`)

* 입력 DOCX 파일에서:

  * 단락, 리스트, 표, 이미지 등 구조 분석
* 출력:

  * `_sanitized/{docname}_sanitized.json`
  * `_comp` 및 `_assets` 폴더에 구성 요소 저장

---

### STEP 2. 이미지 Semantic Summary (`img_summary_gen.py`)

* sanitized JSON에서 **인라인 이미지/스크린샷 정보**를 추출
* Qwen-VL 모델 (예: Qwen2.5-VL-3B-Instruct)을 사용해 이미지 요약 생성
* 출력:

  * `_chunked/{docname}_image_llm.json`

형태 예시:

```json
[
  {"rId": "rId7", "llm_text": "이 이미지는 ECMiner 노드 속성 편집 화면을 보여준다..."},
  {"rId": "rId8", "llm_text": "이 그래프는 시계열 데이터를 시각화한 예시로..."}
]
```

---

### STEP 3. 테이블 HTML Summary (`table_summary_gen.py`) 🔥

경로: `src/mypkg/pipelines/table_summary_gen.py`

* 입력: sanitized JSON 경로
  예:
  `/root/ecm-preprocess-1/output/processed/{docname}/v0/_sanitized/{docname}_sanitized.json`
* Sanitized 구조 안에서 `"type": "table"` 형태의 요소를 찾아,
  그 안의 `"html"` 또는 `"table_html"`에 있는 HTML을 LLM에 넣어 요약
* 사용 모델:

  * 텍스트 전용 Qwen 모델 (기본: `Qwen/Qwen2.5-7B-Instruct`)
  * 환경 변수 `QWEN_TEXT_MODEL_PATH`로 변경 가능
* 출력:

  * `/root/ecm-preprocess-1/output/processed/{docname}/v0/_chunked/{docname}_table_llm.json`

예시:

```json
[
  {
    "table_id": "tbl1",
    "llm_text": "이 표는 ECMiner 프로젝트의 노드별 파라미터를 열로, 각 실험 케이스를 행으로 정리한다..."
  },
  {
    "table_id": "tbl2",
    "llm_text": "이 표는 날짜별로 수집된 센서 값의 요약 통계를 보여준다..."
  }
]
```

**CLI 사용 예:**

```bash
python -m mypkg.pipelines.table_summary_gen \
  /root/ecm-preprocess-1/output/processed/1장_v3.1/v0/_sanitized/1장_v3.1_sanitized.json
```

---

### STEP 4. Chunking (`chunking.py`)

* sanitized JSON을 문단/문맥 단위로 잘라 `_chunked/{docname}_chunked.json` 생성
* 각 chunk의 `Context` 안에는 다음과 같은 placeholder가 포함될 수 있다:

```text
... 본문 텍스트 ...
[image:rId7]
... 더 많은 텍스트 ...
[table:tbl1]
```

placeholder만 있고, 실제 요약 텍스트는 아직 들어가 있지 않은 상태.

---

### STEP 5. 이미지 + 테이블 요약 Merge

`merge_image_summaries_into_chunks.py`

#### 💡 역할

* chunk JSON / 이미지 요약 JSON / 테이블 요약 JSON을 읽어,
* 각 chunk의 `Context` 안에서:

  * `[image:rId#]` 또는 `[Image:rId#]`
    → 그 **바로 뒤에** 해당 이미지 요약을 인라인 삽입
  * `[table:tbl#]` 또는 `[Table:tbl#]`
    → 그 **바로 뒤에** 테이블 요약 인라인 삽입
* 최종적으로 `_chunked/{docname}_chunked_with_imgsum.json` 생성

#### 📌 경로 규칙

문서명이 `{docname}`일 때:

* chunked JSON
  `/root/ecm-preprocess-1/output/processed/{docname}/v0/_chunked/{docname}_chunked.json`
* 이미지 요약 JSON
  `/root/ecm-preprocess-1/output/processed/{docname}/v0/_chunked/{docname}_image_llm.json`
* 테이블 요약 JSON
  `/root/ecm-preprocess-1/output/processed/{docname}/v0/_chunked/{docname}_table_llm.json`
* 최종 merged JSON
  `/root/ecm-preprocess-1/output/processed/{docname}/v0/_chunked/{docname}_chunked_with_imgsum.json`

#### 🧾 동작 방식

`Context`에 다음과 같은 문자열이 있을 때:

```text
... some text ...
[image:rId7]
... more text ...
[table:tbl1]
```

이미지 요약이 `"이미지 요약..."`,
테이블 요약이 `"테이블 요약..."` 이라면,

머지 후:

```text
... some text ...
[image:rId7] 이미지 요약...
... more text ...
[table:tbl1] 테이블 요약...
```

처럼 **placeholder 바로 뒤에 요약이 인라인으로 삽입된다.**

#### 🧪 CLI 사용 예

```bash
# 문서명만 넘기면 됨
python merge_image_summaries_into_chunks.py 1장_v3.1
```

실행 시 내부에서 자동으로 다음 경로를 사용:

* `1장_v3.1_chunked.json`
* `1장_v3.1_image_llm.json`
* `1장_v3.1_table_llm.json`
* `1장_v3.1_chunked_with_imgsum.json`

---

## ⚙️ 전체 파이프라인 한 번에 실행 (`run_full_preprocess.py`)

`run_full_preprocess.py`는 아래 작업들을 **순서대로** 실행하는 오케스트레이션 스크립트다:

1. DOCX → Sanitized (`DocxParsingPipeline`)
2. 이미지 요약 생성 (`img_summary_gen.process_inline_images_to_chunked`)
3. 테이블 요약 생성 (`table_summary_gen.process_tables_to_chunked`) ※ 필요 시 추가
4. Chunking (`chunking.main`)
5. 이미지 + 테이블 요약 merge (`merge_image_summaries_into_chunks.py` 내부 함수 호출)

### 1) 단일 DOCX 파일 처리

```bash
source .venv/bin/activate
python run_full_preprocess.py ./docs/ECMiner_Manual.docx
```

### 2) 폴더 내 모든 DOCX 처리

```bash
python run_full_preprocess.py ./docs/
```

---

## 📦 최종 RAG 입력 파일

**가장 중요한 결과물은 이것 하나:**

```text
output/processed/{docname}/v0/_chunked/{docname}_chunked_with_imgsum.json
```

* 텍스트 + 이미지 요약 + 테이블 요약이 모두 포함된 chunk 단위 JSON
* RAG 인덱싱 단계에서 이 파일만 읽어 벡터화하면 됨
