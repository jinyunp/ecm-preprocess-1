# 📘 ECMiner™ Preprocessing Pipeline

**DOCX → Sanitize → Image Summary → Chunking → Chunk+ImageSummary 통합 RAG 입력 데이터 생성**

이 프로젝트는 ECMiner™ 소프트웨어 매뉴얼 같은 **복잡한 문서(DOCX)** 를
**RAG 시스템에서 바로 사용할 수 있는 구조**로 자동 변환하기 위한 **Preprocessing Pipeline**입니다.

---

# ⭐ 기능 요약

✔ DOCX 문서 파싱 (텍스트, 표, 리스트, 이미지 등 구조 분석)
✔ 이미지/그림/스크린샷 자동 OCR + Semantic Summary (Qwen2-VL-7B-Instruct)
✔ Text Chunking (문단 기반 chunking, 이미지 위치 포함)
✔ Chunk + Image Summary 자동 병합
✔ 다수의 DOCX 파일을 일괄 처리 가능
✔ RunPod 환경 & 일반 Ubuntu 환경 모두 지원
✔ Qwen2-VL 모델 xet clone 자동 다운로드
✔ 전체 파이프라인을 수행하는 단일 엔트리포인트 제공

---

# 📂 프로젝트 구조

```
.
├── README.md
├── run_full_preprocess.py        # 전체 파이프라인 실행 스크립트
├── start_preprocessing.sh        # 환경 구성 + 모델 다운로드 스크립트
├── config/
├── infra/
├── output/
│   └── processed/
│       └── <문서명>/v0/
│               ├── _sanitized/
│               ├── _comp/
│               └── _chunked/
├── scripts/
└── src/
    └── mypkg/
        ├── pipelines/
        │     ├── docx_parsing_pipeline.py
        │     ├── merge_image_summaries_into_chunks.py
        │     ├── img_summary_gen.py
        │     └── chunking.py
        ├── components/
        ├── cli/
        └── utils/
```

---

# 🚀 빠른 시작 (Quick Start)

## 1. RunPod 또는 Ubuntu 서버에 접속

```
cd /workspace/ECMinerPreprocess
```

## 2. 환경 설치 + 모델 다운로드

`start_preprocessing.sh` 실행:

```bash
bash start_preprocessing.sh
```

그러면 다음이 자동 수행됩니다:

* `/workspace/qwen` 에 Qwen2-VL-7B-Instruct 다운로드
* `.venv` 가상환경 생성
* pip 패키지 설치
* 환경 변수 설정

  * `QWEN_MODEL_PATH=/workspace/qwen`
  * `PYTHONPATH=<project_root>/src`

---

## 3. 가상환경 활성화

```bash
source .venv/bin/activate
```

---

## 4. 단일 DOCX 파일 변환

```bash
python run_full_preprocess.py ./docs/ECMiner_Manual.docx
```

출력 위치:

```
output/processed/ECMiner_Manual/v0/
    ├── _sanitized/
    ├── _chunked/
    └── _chunked/<docname>_chunked_with_imgsum.json
```

---

## 5. 폴더 내의 모든 DOCX 파일 일괄 처리

```bash
python run_full_preprocess.py ./docs/
```

---

# 🔧 `run_full_preprocess.py` CLI 사용법

```
usage: run_full_preprocess.py <input_path> [-o OUTPUT_ROOT] [--pattern PATTERN]
```

### 입력이 파일이면 → 그 파일만 처리

### 입력이 폴더이면 → 내부 모든 `.docx` 처리

---

### 📌 예시

#### 1) 출력 경로 기본값 사용

```bash
python run_full_preprocess.py ./manuals/Manual1.docx
```

→ `output/processed/Manual1/v0/` 자동 생성

#### 2) 출력 경로 지정

```bash
python run_full_preprocess.py ./manuals/Manual1.docx -o ./output/custom
```

→ `./output/custom/Manual1/v0/`에 저장

#### 3) 폴더 전체 처리

```bash
python run_full_preprocess.py ./manuals/
```

#### 4) 파일명 패턴 지정 (예: 대문자 DOCX)

```bash
python run_full_preprocess.py ./manuals/ --pattern "*.DOCX"
```

---

# 🧠 파이프라인 단계 요약

## STEP 1. DOCX Parsing

* 단락, 표, 리스트 파싱
* 이미지 `_assets`에 저장
* Sanitized JSON 생성

## STEP 2. Image Semantic Summary

* Qwen2-VL-7B-Instruct 사용
* inline/표/리스트에 포함된 이미지 요약

## STEP 3. Chunking

* 단락 기반으로 문서 chunk 생성
* 이미지 anchor(`[Image:rIdXX]`)도 유지

## STEP 4. Chunk + Image Summary Merge

* 각 이미지를 chunk 내 anchor 위치에 삽입
* 최종 RAG-ready JSON 생성

---

# 📦 출력 결과 구조

예시:

```
output/processed/Chapter1/v0/
│
├── _sanitized/
│      └── Chapter1_sanitized.json
│
├── _chunked/
│      ├── Chapter1_chunked.json
│      ├── Chapter1_image_llm.json
│      └── Chapter1_chunked_with_imgsum.json   # 최종 결과
│
└── _comp/   (이미지/표/리스트 구성 요소)
```

---

# 🎯 RAG에 바로 사용되는 파일

```
<docname>_chunked_with_imgsum.json
```

이 파일이 **텍스트 + 이미지 요약이 통합된 최종 RAG 입력 파일**입니다.

---

# 💡 추가 작업 요청 가능

원하면 다음도 지원해드립니다:

* RAG 인덱싱 자동화 스크립트 (`init_index.py` 통합)
* Vector DB(Qdrant) 자동 로딩 버전
* Streamlit 기반 테스트 챗봇 UI 연동
* RunPod Start Script (`start_runpod.sh`)
* GPU 최적화된 Qwen2-VL inference 모듈
