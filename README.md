# ECM Preprocess Pipeline

이 저장소는 EC민자료 DOCX 문서를 정제(sanitize)해 구조화된 JSON과 컴포넌트 파일들을 생성하는 파이프라인입니다. 현재 단계에서는 `_sanitized/*.json`과 `_comp/*.json` 산출물이 완성돼 있으며, 최종 목표는 이 결과물을 기반으로 DocJSON 포맷(섹션/블록 중심 구조)까지 생성하는 것입니다.

> ⚠️ DocJSON 포맷은 아직 미구현입니다. `src/mypkg/core/docjson_types.py`에 정의된 타입을 참고해 후속 작업을 설계하세요.

---

## 1. 준비하기

### 1.1 필수 도구
- Python 3.12 (시스템 파이썬이 3.12라면 자동 사용)
- [uv](https://docs.astral.sh/uv/) (가상환경/패키지 관리)

### 1.2 의존성 설치
```bash
uv venv          # .venv 생성 (1회)
uv sync          # pyproject.toml + uv.lock 기반 패키지 설치/업데이트
```
> `./start.sh`를 실행하면 위 두 명령을 자동으로 수행해 가상환경을 부트스트랩합니다. 수동으로 실행할 필요는 없지만, 환경 점검을 위해 직접 실행해 볼 수 있습니다.

### 1.3 실행 스크립트
`./start.sh`는 `.venv`의 파이썬을 이용해 `PYTHONPATH=src` 설정 후 CLI를 실행합니다. 모든 예시는 이 스크립트를 기준으로 합니다.

---

## 2. 데이터 위치

- 원본 DOCX는 프로젝트 외부의 형제 디렉터리인 `_datasets/ecminer/`에 있습니다.
- CLI의 `raw` 인자로 파일명만 넘기면 우선 이 디렉터리에서 검색합니다.

예)
```bash
./start.sh Appendix_v2.docx --version v0
./start.sh ../_datasets/ecminer --all --version v0   # 디렉터리 전체 처리
```

---

## 3. CLI 사용법

```bash
./start.sh --help
```

주요 옵션 정리:

- `raw` (필수): 처리할 DOCX 파일 또는 디렉터리
- `--version`: 출력 버전 태그 (디폴트 `v0`)
- `--processed-root`: 기본 출력 루트 (디폴트 `output/processed`)
- `--all` / `--one`: 디렉터리 입력 시 전체/특정 파일 선택
- `--inspect`: 기존 산출물 확인용 (`sanitized`, `tables`, `lists`, `inline_images`, `meta`, `comp`, `all`, `ls` 등)
  - 예: `./start.sh dummy --inspect sanitized --base-dir output/processed/1장_v3.1/v0`

---

## 4. 파이프라인 구조

`src/mypkg/pipelines/docx_parsing_pipeline.py`에 정의된 단일 파이프라인이 다음 단계를 수행합니다.

1. **Parsing**  
   - `DocxXmlParser`가 DOCX에서 문단/표/이미지 등의 XML 데이터를 추출.  
   - 결과: `_sanitized/<doc>_output_xml.json` 및 인라인 이미지 원본(`_sanitized/_assets/`).

2. **Sanitizing**  
   - 문단(`ParagraphSanitizer`), 리스트(`ListSanitizer`), 표(`TableSanitizer`), 이미지(`ImageSanitizer`)를 정제해 균일한 스키마로 변환.  
   - 결과: `_sanitized/<doc>_sanitized.json` (현재까지 확정된 최종물).

3. **Component 분리 저장**  
   - 후속 처리를 위해 `_comp/` 아래에 파트별 JSON을 생성.  
   - `parag_comp.json`, `list_comp.json`, `table_comp.json`, `image_comp.json`.

---

## 5. 출력 디렉터리 레이아웃

```
output/processed/<문서명>/<버전>/
├── _sanitized/
│   ├── <문서명>_output_xml.json   # 파서 원본
│   ├── <문서명>_sanitized.json    # 현재 주요 산출물
│   └── _assets/                   # 인라인 이미지 원본
└── _comp/
    ├── parag_comp.json   # 문단 (doc_index 포함)
    ├── list_comp.json    # 리스트 + consumed 정보
    ├── table_comp.json   # 표 데이터 (행/열, HTML, 앵커 등)
    └── image_comp.json   # 이미지 메타데이터
```

`*_sanitized.json` 구조 요약:
- `paragraphs`: `{text, doc_index, style, source_doc_indices, emphasized, math_texts, image_included}`
- `lists`: 리스트 문단(문단 구조와 동일)
- `tables`: `{tid, doc_index, rows, cols, data, table_html, anchors, ...}`
- `inline_images`: `ImageSanitizer`에서 채운 이미지 메타 (`doc_index`, 파일명 등)
- `relationships`: DOCX relationships 맵 (추가 후속용)

---

## 6. 개발/확장 가이드

### 6.1 DocJSON 단계 계획
DocJSON은 섹션/블록 기반 구조로 나중에 생성해야 합니다. 참고 파일:
- `src/mypkg/core/docjson_types.py`  
  - `ContentBlock`, `Section`, `DocumentDocJSON` 등 타입 정의/직렬화 로직.

### 6.2 DocJSON 생성 시 고려 사항
1. **문단 → 블록 매핑**  
   - 문단 컴포넌트를 순회하며 `doc_index` 기준으로 블록을 생성하고 섹션을 조직화.
2. **표(Table) 처리**  
   - `table_comp.json`의 `table_html`, `data`, `doc_index`를 DocJSON 블록에 포함.  
   - 텍스트 흐름 중 표 위치를 유지하려면 `doc_index`를 기준으로 블록 순서를 정렬하세요.
3. **이미지/리스트 연계**  
   - 이미지와 리스트도 `doc_index`를 기준으로 적절한 블록으로 통합하거나 메타데이터에 연결.
4. **섹션(heading) 분석**  
   - 내용 중 제목/헤더 정보를 기반으로 `Section` 트리를 만들고, 각 섹션이 참조하는 블록 ID를 지정.
5. **최종 직렬화**  
   - `DocumentDocJSON.to_dict()` 사용 → `docjson_output_path_from_sanitized()`로 경로 계산 → JSON 저장.

DocJSON 작업은 현재 `.old/`에 이동해 둔 레거시 코드(예: `section_analyzer.py`, `layout_assembler.py`)를 참고해 구현할 수 있습니다. 필요하면 해당 코드를 분석 후 일부 로직만 재사용하세요.

### 6.3 Inspect 모드로 디버깅
- `./start.sh dummy --inspect ls --base-dir <출력폴더>`  
  → 각 JSON 존재 여부 확인.
- `--inspect tables --base-dir ... --json`  
  → 표 데이터 전체 출력.
- `--inspect sanitized --sanitized <경로>`  
  → `_sanitized.json`을 바로 조회.

---

## 7. 레거시 코드 보관

더 이상 사용하지 않는 분석기/DocJSON 설정 코드는 `.old/` 디렉터리에 보관했습니다.

```
.old/src/mypkg/components/analyzer/      # 이전 섹션/레이아웃 조립 로직
.old/src/mypkg/components/paragraph_post.py
.old/src/mypkg/core/config/              # DocJSON/Section 설정 관련
```

필요 시 이 디렉터리에서 파일을 꺼내와 새로운 구조에 맞춰 재작성하세요.

---

## 8. 테스트 & 운영 팁

- 파이프라인 자체는 비동기로 동작하지만 CLI에서 자동으로 `asyncio.run`을 실행합니다.
- 새로운 문서를 추가할 때는 `_datasets/ecminer/`에 복사 후 `./start.sh <파일>`로 실행합니다.
- 대량 처리:
  ```bash
  ./start.sh ../_datasets/ecminer --all --version v0
  ```
- 산출물을 벡터DB에 적재할 때는 `_sanitized.json`에서 문단/표를 추출하고 각 항목에 메타데이터(`doc_index`, `version`, `source_doc_indices`)를 붙이는 것을 권장합니다.

---

## 9. 흔한 문제 해결

| 증상 | 해결 |
| --- | --- |
| `ModuleNotFoundError: httpx` | `uv sync` 재실행 |
| `error: 입력 경로가 존재하지 않습니다` | `_datasets/ecminer`에 파일이 있는지 확인하고 정확한 이름 전달 |
| `.venv`를 찾지 못함 | `uv venv` 실행 후 `uv sync` |
| DocJSON 파일 미생성 | 아직 구현되지 않았습니다. `_sanitized` + `_comp`를 활용해 DocJSON 변환 로직을 개발하세요. |

---

## 10. 이후 인수인계 체크리스트

1. `uv venv`, `uv sync`로 환경 준비되었는가?
2. `_datasets/ecminer` 접근 권한이 있는가?
3. `./start.sh <DOCX>` 실행 후 `output/processed/<doc>/<ver>/`에 산출물이 생성되는가?
4. `_sanitized/*.json` 구조를 이해하고 필요한 메타데이터를 추출할 수 있는가?
5. DocJSON 변환 스펙(섹션/블록/테이블/이미지)을 정의했는가?
6. 레거시 코드(.old)를 분석해 재사용할지 여부를 결정했는가?

이 문서를 인수자에게 전달하고, 궁금한 사항은 `src/mypkg/pipelines/docx_parsing_pipeline.py`와 `src/mypkg/core/docjson_types.py`를 먼저 참고하도록 안내하세요.
