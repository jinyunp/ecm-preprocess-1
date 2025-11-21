# config/configs.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Literal, Optional, Mapping, FrozenSet, Any
# ------------------------------
# 전처리 청킹(기존 유지)
# ------------------------------
class ChunkingType(Enum):
    NARRATIVE = "narrative"
    TABLE = "table"

class ChunkingStrategy(Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    HIERARCHICAL = "hierarchical"
    ROW = "row"
# ------------------------------
# 검색 타입
# ------------------------------
class SearchType(str, Enum):
    SPARSE = "SPARSE"
    DENSE = "DENSE"
    HYBRID = "HYBRID"
    MULTISTAGE = "MULTISTAGE"

# ------------------------------
# 임베딩 타입/프로바이더(기존)
# ------------------------------
class EmbeddingType(str, Enum):
    SPARSE  = "sparse"
    DENSE   = "dense"
    COLBERT = "colbert"

class Provider(str, Enum):
    FASTEMBED   = "fastembed"
    HUGGINGFACE = "huggingface"

# ---- 타입 그룹 (빌드 스크립트 단순화용) ----
SPARSE_TYPES: FrozenSet[EmbeddingType] = frozenset({EmbeddingType.SPARSE})
DENSE_TYPES: FrozenSet[EmbeddingType] = frozenset({EmbeddingType.DENSE})
MULTIVECTOR_TYPES: FrozenSet[EmbeddingType] = frozenset({EmbeddingType.COLBERT})

# ------------------------------
# 라우팅 스키마
# ------------------------------
@dataclass(frozen=True)
class SparseRoute:
    vectorizer_type: str 
    language: Literal["en", "ko", "auto"]

@dataclass(frozen=True)
class DenseRoute:
    provider: Provider
    model_name: str

@dataclass(frozen=True)
class ColbertRoute:
    provider: Provider
    model_name: str


Route = SparseRoute | DenseRoute | ColbertRoute 

# ------------------------------
# 언어 프로필 & 라우팅 팩토리
# ------------------------------
@dataclass(frozen=True)
class LanguageProfile:
    dense_provider: Provider
    dense_model: str
    colbert_provider: Provider
    colbert_model: str

LANGUAGE_PROFILES: Mapping[Literal["en","ko"], LanguageProfile] = {
    "en": LanguageProfile(
        dense_provider=Provider.FASTEMBED,
        dense_model="BAAI/bge-base-en-v1.5",
        colbert_provider=Provider.FASTEMBED,
        colbert_model="colbert-ir/colbertv2.0",
    ),
    "ko": LanguageProfile(
        dense_provider=Provider.FASTEMBED,
        dense_model="intfloat/multilingual-e5-large",
        colbert_provider=Provider.FASTEMBED,
        colbert_model="jinaai/jina-colbert-v2",
    ),
}

def assign_provider(route: Any, etype: EmbeddingType, *, use_cuda: bool = False):
    from infra.embedding import HuggingFaceEmbedding, FastEmbedEmbedding
    PROVIDER_TO_CLASS = {
        Provider.HUGGINGFACE: HuggingFaceEmbedding,
        Provider.FASTEMBED: FastEmbedEmbedding,
    }
    cls = PROVIDER_TO_CLASS.get(route.provider)
    if not cls:
        raise ValueError(f"Unknown provider: {route.provider}")
    return cls(route.model_name, etype, use_cuda=use_cuda)


def get_embedding_type_configs(lang: Literal["en","ko"]) -> Mapping[EmbeddingType, Route]:
    prof = LANGUAGE_PROFILES[lang]
    return {
        EmbeddingType.SPARSE:  SparseRoute(vectorizer_type="bm25", language=lang),
        EmbeddingType.DENSE:   DenseRoute(provider=prof.dense_provider,   model_name=prof.dense_model),
        EmbeddingType.COLBERT: ColbertRoute(provider=prof.colbert_provider, model_name=prof.colbert_model),
    }

# ------------------------------
# 모델 구성 컨테이너(벡터스토어 생성에 필요)
# ------------------------------

@dataclass(frozen=True)
class EmbeddingModelConfig:
    embed_type: Dict[EmbeddingType, bool]        # 벡터 종류
    dimensions: Dict[EmbeddingType, int]         # 각 벡터 차원
    dense_distance: Literal["Cosine","Dot","Euclidean"] = "Cosine"

# ------------------------------
# 인덱스/런타임 기본값(빌드에서 소비)
# ------------------------------
class IndexingType(str, Enum):
    HNSW = "hnsw"
    FLAT = "flat"

@dataclass(frozen=True)
class HnswDefaults:
    m: int = 32 # 한 노드 당 최대 연결 수 (그래프의 밀도), 커질수록 정확도가 올라가지만 경로가 많아지고 구조가 복잡해질 수 있음. 정확도가 중요하면 32~64
    ef_construct: int = 200 # 각 벡터에 대해 탐색할 후보 노드 수 (exploration factor for index building) 크게 설정할수록 더 정확한 그래프 생성, but 구축시간 느려짐. 대규모나 고정밀은 200~500까지 ㄱㄴ 
    full_scan_threshold: int = 10000 # 데이터 개수가 이 값보다 적으면 FLAT 스캔 수행. 작은 데이터라도 HNSW 쓰고싶으면 0으로 설정
    payload_m: int = 16

@dataclass(frozen=True)
class RuntimeOptions:
    indexing: IndexingType = IndexingType.HNSW
    dense_distance: Literal["Cosine","Dot","Euclidean"] = "Cosine"
    use_cuda: bool = False
    batch_size: int = 32
    update_colbert: bool = False
    colbert_batch_size: int = 1

# ------------------------------
# 생성 데이터 클래스 --> 세션에 저장되는 기본/선택 설정값으로 stateful config임. UI 에서 전달된 값으로 초기화 가능
# GenerateRequest는 runtime 요청 파라미터로 generaterequest가 없으면 generationconfig기본값을 불러와서 보정하는 것임.
# ------------------------------
@dataclass(frozen = False)
class GenerationConfig:
    prompt_template: str = """ 
        You are an AI assistant for ECMiner software.
        Do NOT use prior knowledge or invent answers.
        If information is not available, say: Information not found in context.
        Previous conversations: {history}
        Question: {query}
        Reference context: {context}
        Answer:
    """
    temperature: float = 0.5
    top_p: float = 0.7
    provider: str = "ollama"
    model_name: str = ""
    search_type: str = "DENSE"
    use_cross_encoder: bool = False
    # 추가적인 생성 관련 인자 (예: max_tokens, stop_words 등)
    max_tokens: Optional[int] = None
    language: Literal["en", "ko"] = "ko"
    history_limit: int = 10