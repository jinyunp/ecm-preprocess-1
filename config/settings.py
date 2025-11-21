# config/settings.py
from __future__ import annotations
from typing import Literal, List
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from datetime import datetime

Lang = Literal["en", "ko"]

class Settings(BaseSettings):
    """
    - 다중 파일 입력을 전제로 FILENAMES(List[str]) 사용
    - 전처리 산출물(.jsonl) 경로를 파일별/머지본으로 모두 제공
    - 임베딩은 언어별(en/ko)로 모델/프로바이더를 선택
    """
    model_config = SettingsConfigDict(extra="ignore")

    # --------------------------------------------------------------- 사용자 설정 ------------------------------------------------------------------

    # ------------------------------
    # 기본
    # ------------------------------
    LANGUAGE: Lang = "ko"  # 기본 질의 언어(런타임에서 바꿀 수 있음)

    VERSION: str = "v2"
    COLLECTION_NAME_BASE: str = f"manual_prep"
    # COLLECTION_NAME_BASE: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 여러 파일 입력 (예: ["manual.docx","policy.pdf"])
    FILENAMES: List[str] = ["manual.docx"]
    
    # ------------------------------
    # 전처리 옵션 (멀티파일 고려)
    # ------------------------------
    PREPROCESS_MERGE_BOOL: bool = True
    PREPROCESS_OUTPUT_FORMAT: Literal["jsonl"] = "jsonl"
    PREPROCESS_MERGED_BASENAME: str = f"merged.jsonl"    # 합쳐진 산출물 파일명

    # ------------------------------------------------------------------------------------------------------------------------------------------
    
    # ------------------------------
    # 경로
    # ------------------------------
    RAW_DATA_PATH: Path = Path("data/raw")
    PROCESSED_DATA_PATH: Path = Path("data/processed")
    INDEX_DIR: Path = Path("data/index")        # Qdrant 인덱싱
    PICKLE_DIR: Path = Path("data/pickles")     # BM25 아티팩트 저장
    CONFIG_DB_PATH: Path = Path("var/config.db")  # 세션 설정과 히스토리 저장
    LOG_DIR: Path = Path("logs")                # 세션별 상세 로그
    QDRANT_URL: str = "http://localhost:6333"

    # ------------------------------
    # 유효성/디렉터리 보장
    # ------------------------------
    @field_validator("RAW_DATA_PATH", "PROCESSED_DATA_PATH", "INDEX_DIR", "PICKLE_DIR", "CONFIG_DB_PATH", "LOG_DIR")
    @classmethod
    def _ensure_dirs(cls, v: Path) -> Path:
        (v if v.suffix == "" else v.parent).mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("FILENAMES", mode="before")
    @classmethod
    def _coerce_filenames(cls, v):
        """
        환경변수로 "a.docx,b.pdf"처럼 넘겨도 리스트로 파싱되게.
        (JSON 문자열도 자동 파싱됨)
        """
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("VERSION", mode="before")
    @classmethod
    def _sanitize_version(cls, v: str) -> str:
        import re
        """
        파일명/컬렉션에 안전하게 쓰도록 버전 문자열 정규화.
        공백 제거 후 [a-zA-Z0-9._-] 외 문자는 '_'로 치환.
        """
        if not v:
            return "v1"
        v = str(v).strip()
        return re.sub(r"[^a-zA-Z0-9._-]", "_", v)
    
    # ------------------------------
    # 내부 헬퍼
    # ------------------------------
    def _with_version_suffix(self, stem: str) -> str:
        """파일/이름 stem 뒤에 _<VERSION> 접미사 부착"""
        return f"{stem}_{self.VERSION}"

    # ------------------------------
    # 파생 경로 (멀티파일 대응)
    # ------------------------------
    @property
    def RAW_DATA_FILE_PATHS(self) -> List[Path]:
        return [self.RAW_DATA_PATH / name for name in self.FILENAMES]

    @property
    def PROCESSED_DATA_FILE_PATHS(self) -> List[Path]:
        """
        각 원본 파일별 전처리 산출물(.jsonl) 경로
        예) data/processed/manual.jsonl, policy.jsonl ...
        """
        out: List[Path] = []
        for name in self.FILENAMES:
            stem = name.rsplit(".", 1)[0]
            ver_stem = self._with_version_suffix(stem)  # dev_doc_v1
            out.append(self.PROCESSED_DATA_PATH / f"{ver_stem}.{self.PREPROCESS_OUTPUT_FORMAT}")
        return out

    @property
    def PROCESSED_DATA_MERGED_FILE_PATH(self) -> Path:
        """
        여러 파일을 합쳐 하나의 jsonl로 만들 때 쓰는 경로
        """
        base = Path(self.PREPROCESS_MERGED_BASENAME)
        stem = base.stem  # "merged"
        ver_stem = self._with_version_suffix(stem)  # "merged_v1"
        return self.PROCESSED_DATA_PATH / f"{ver_stem}.{base.suffix.lstrip('.') or self.PREPROCESS_OUTPUT_FORMAT}"

    # 머지본 파일명(stem): "merged.jsonl" -> "merged"
    @property
    def PROCESSED_DATA_MERGED_STEM(self) -> str:
        return self.PROCESSED_DATA_MERGED_FILE_PATH.stem

    # ------------------------------
    # 언어별 BM25 아티팩트 (완전 분리)
    # ------------------------------
    @property
    def BM25_PICKLE_EN(self) -> Path:
        # ex) data/pickles/merged_bm25_en.pkl
        return self.PICKLE_DIR / f"{self.PROCESSED_DATA_MERGED_STEM}_bm25_en.pkl"
    @property
    def BM25_PICKLE_KO(self) -> Path:
        # ex) data/pickles/merged_bm25_ko.pkl
        return self.PICKLE_DIR / f"{self.PROCESSED_DATA_MERGED_STEM}_bm25_ko.pkl"

    def get_bm25_pickle(self, lang: Lang) -> Path:
        return self.BM25_PICKLE_EN if lang == "en" else self.BM25_PICKLE_KO

    # ------------------------------
    # 언어별 컬렉션 네이밍
    # ------------------------------
    @property
    def COLLECTION_NAME_EN(self) -> str:
        # ex) testingmanageindex_v1_en
        return f"{self._with_version_suffix(self.COLLECTION_NAME_BASE)}_en"

    @property
    def COLLECTION_NAME_KO(self) -> str:
        # ex) testingmanageindex_v1_ko
        return f"{self._with_version_suffix(self.COLLECTION_NAME_BASE)}_ko"

    def get_collection_name(self, lang: Lang=LANGUAGE) -> str:
        return self.COLLECTION_NAME_EN if lang == "en" else self.COLLECTION_NAME_KO

# 싱글톤
_settings_instance: Settings | None = None
def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


## 예시
"""
    s = get_settings()
    print(s.FILENAMES)                       # ["manual.docx", "policy.pdf", ...]
    print(s.RAW_DATA_FILE_PATHS)             # [data/raw/manual.docx, data/raw/policy.pdf, ...]
    print(s.PROCESSED_DATA_FILE_PATHS)       # [data/processed/manual.jsonl, data/processed/policy.jsonl, ...]
    print(s.PROCESSED_DATA_MERGED_FILE_PATH) # data/processed/merged.jsonl (옵션)
    print(s.get_collection_name('en'))  # testingrefact_en
    print(s.get_collection_name('ko'))  # testingrefact_ko
    print(s.get_bm25_pickle('en'))      # data/pickles/bm25_en.pkl
    print(s.get_bm25_pickle('ko'))      # data/pickles/bm25_ko.pkl
    print(s.get_embedding('en').dense_model)
"""