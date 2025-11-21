#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/init_index.py

- ECMiner chunked JSON들을 읽어서
- Sparse / Dense / (옵션) ColBERT 임베딩을 생성하고
- Qdrant Docker 인스턴스에 컬렉션을 만들고 업로드한다.

실행:
    python scripts/init_index.py
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Any, Tuple

from qdrant_client.models import HnswConfigDiff, PointVectors

# 프로젝트 공용 모듈 (이미 build_index.py에서 쓰던 것 그대로 재사용)
from config.configs import (
    EmbeddingType,
    EmbeddingModelConfig,
    RuntimeOptions,
    HnswDefaults,
    IndexingType,
    assign_provider,
    get_embedding_type_configs,
)
from config.settings import get_settings
from infra.vectorstore import Qdrant
from infra.vectorizer import BM25SparseVectorizer
from utils.embedding_helpers import generate_points


# =============================================================================
# 1. Chunked JSON 로딩 헬퍼
# =============================================================================

def find_chunked_files(base_dir: Path) -> List[Path]:
    """
    base_dir 아래에서 *_chunked_with_imgsum.json / *_chunked.json 파일을 모두 찾는다.
    필요하면 이 부분만 자기 프로젝트 구조에 맞게 수정하면 됨.
    """
    if not base_dir.exists():
        raise RuntimeError(f"Chunked base dir not found: {base_dir}")

    # 이미지 요약이 들어간 파일이 우선
    files = sorted(base_dir.rglob("*_chunked_with_imgsum.json"))
    if not files:
        files = sorted(base_dir.rglob("*_chunked.json"))

    if not files:
        raise RuntimeError(f"No chunked json files found under: {base_dir}")

    return files


def load_documents_from_chunked_file(path: Path) -> List[Any]:
    """
    단일 chunked.json 파일을 Document-like 객체 리스트로 변환.
    - Context → page_content
    - 나머지 필드 → metadata
    Document 타입을 강하게 의존하지 않기 위해 SimpleNamespace 사용.
    (generate_points는 .page_content, .metadata만 쓰므로 충분)
    """
    with path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    docs: List[Any] = []
    for ch in chunks:
        # 필수 필드: Context, Context_id
        content = ch.get("Context", "") or ""
        if not content.strip():
            continue

        # 메타데이터 구성 (Context만 제외)
        metadata = {k: v for k, v in ch.items() if k != "Context"}

        doc = SimpleNamespace(
            page_content=content,
            metadata=metadata,
        )
        docs.append(doc)

    return docs


def load_documents_from_chunked_dir(base_dir: Path) -> List[Any]:
    """
    base_dir 아래 chunked JSON들을 모두 읽어서 하나의 documents 리스트로 합친다.
    """
    chunk_files = find_chunked_files(base_dir)
    print(f"[INFO] Found {len(chunk_files)} chunked json files under {base_dir}")

    all_docs: List[Any] = []
    for p in chunk_files:
        docs = load_documents_from_chunked_file(p)
        all_docs.extend(docs)
        print(f"[INFO] Loaded {len(docs)} docs from {p.name}")

    print(f"[INFO] Total documents loaded from chunked jsons: {len(all_docs)}")
    return all_docs


# =============================================================================
# 2. Embedding 준비 로직 (Sparse / Dense / ColBERT)
# =============================================================================

def prepare_embeddings(
    documents: List[Any],
    lang: str,
    runtime: RuntimeOptions,
) -> Tuple[Dict[EmbeddingType, Any], Dict[EmbeddingType, int], List[Dict[str, Any]], EmbeddingModelConfig]:
    """
    - BM25 Sparse 벡터 준비 (필요시 훈련 및 저장)
    - Dense 임베딩 모델 준비
    - ColBERT 임베딩 모델 준비 (옵션)
    - EmbeddingModelConfig 생성
    """
    s = get_settings()
    routes = get_embedding_type_configs(lang)  # {"sparse": SparseRoute, ...}

    embed_models: Dict[EmbeddingType, Any] = {}
    dimensions: Dict[EmbeddingType, int] = {}
    sparse_vectors: List[Dict[str, Any]] = []

    # ---------- SPARSE (BM25) ----------
    if EmbeddingType.SPARSE in routes:
        sr = routes.get(EmbeddingType.SPARSE)
        bm25_pkl = s.get_bm25_pickle(lang)
        vec = BM25SparseVectorizer(pkl_path=bm25_pkl, language=getattr(sr, "language", lang))

        # pickle이 없으면 현재 documents로 훈련
        if not vec.load():
            texts = [d.page_content for d in documents]
            print(f"[INFO] BM25 pickle not found. Training on {len(texts)} documents...")
            vec.fit(texts)
            vec.save()
            print("[INFO] BM25 training finished and saved.")
        else:
            print("[INFO] Loaded existing BM25 vectorizer pickle.")

        embed_models[EmbeddingType.SPARSE] = vec
        sparse_vectors = vec.transform([d.page_content for d in documents])
        print(f"[INFO] Sparse vectors prepared: {len(sparse_vectors)}")

    # ---------- DENSE ----------
    if EmbeddingType.DENSE in routes:
        dr = routes[EmbeddingType.DENSE]
        dense = assign_provider(dr, EmbeddingType.DENSE, use_cuda=runtime.use_cuda)
        embed_models[EmbeddingType.DENSE] = dense
        dimensions[EmbeddingType.DENSE] = dense.get_dimension()
        print(f"[INFO] Dense model loaded. Dimension = {dimensions[EmbeddingType.DENSE]}")

    # ---------- COLBERT ----------
    if runtime.update_colbert and EmbeddingType.COLBERT in routes:
        cr = routes[EmbeddingType.COLBERT]
        colbert = assign_provider(cr, EmbeddingType.COLBERT, use_cuda=runtime.use_cuda)
        embed_models[EmbeddingType.COLBERT] = colbert
        dimensions[EmbeddingType.COLBERT] = colbert.get_dimension()
        print(f"[INFO] ColBERT model loaded. Dimension = {dimensions[EmbeddingType.COLBERT]}")

    # ---------- EmbeddingModelConfig ----------
    def _has_sparse(batch):
        return any(bool(v.get("indices") and v.get("values")) for v in batch)

    embed_type = {
        EmbeddingType.SPARSE:  _has_sparse(sparse_vectors),
        EmbeddingType.DENSE:   EmbeddingType.DENSE   in embed_models,
        EmbeddingType.COLBERT: EmbeddingType.COLBERT in embed_models,
    }

    model_config = EmbeddingModelConfig(
        embed_type=embed_type,
        dimensions=dimensions,  # SPARSE 차원은 필요 없음
        dense_distance=runtime.dense_distance,
    )

    if not any(embed_type.values()):
        raise RuntimeError("No embeddings prepared (sparse/dense/colbert are all missing).")

    return embed_models, dimensions, sparse_vectors, model_config


# =============================================================================
# 3. Qdrant 컬렉션 생성 및 업로드
# =============================================================================

def create_qdrant_collection(
    qdrant: Qdrant,
    collection_name: str,
    model_config: EmbeddingModelConfig,
    runtime: RuntimeOptions,
    hnsw_defaults: HnswDefaults,
) -> None:
    """Qdrant 컬렉션 생성 (HNSW 설정 포함)"""
    if runtime.indexing == IndexingType.HNSW:
        hnsw_config_obj = HnswConfigDiff(
            m=hnsw_defaults.m,
            ef_construct=hnsw_defaults.ef_construct,
            full_scan_threshold=hnsw_defaults.full_scan_threshold,
            payload_m=hnsw_defaults.payload_m,
        )
    else:
        hnsw_config_obj = None

    qdrant.create_collection(
        collection_name=collection_name,
        model_config=model_config,
        hnsw_config=hnsw_config_obj,
    )
    print(f"[INFO] Qdrant collection '{collection_name}' created.")


def upload_dense_sparse_points(
    qdrant: Qdrant,
    collection_name: str,
    documents: List[Any],
    embed_models: Dict[EmbeddingType, Any],
    sparse_vectors: List[Dict[str, Any]],
    runtime: RuntimeOptions,
) -> List[Tuple[str, str]]:
    """
    Dense / Sparse 포인트를 배치로 업로드.
    ColBERT 업데이트에 사용할 (id, content) 리스트를 반환.
    """
    uploaded: List[Tuple[str, str]] = []
    total_uploaded = 0

    for i in range(0, len(documents), runtime.batch_size):
        batch_docs = documents[i : i + runtime.batch_size]
        batch_sparse = sparse_vectors[i : i + len(batch_docs)] if sparse_vectors else None

        points = generate_points(
            batch_docs,
            embed_models,
            batch_sparse,
            runtime,
        )
        qdrant.upload_vectors(collection_name, points)
        total_uploaded += len(points)

        uploaded.extend(
            (str(p.id), p.payload["content"])
            for p in points
            if p.payload is not None and "content" in p.payload
        )

        print(f"[INFO] Uploaded batch {i // runtime.batch_size + 1} "
              f"({len(points)} points, total {total_uploaded})")

    print(f"[INFO] Dense/Sparse uploaded: {total_uploaded} points")
    return uploaded


def update_colbert_vectors(
    qdrant: Qdrant,
    collection_name: str,
    uploaded: List[Tuple[str, str]],
    embed_models: Dict[EmbeddingType, Any],
    runtime: RuntimeOptions,
) -> None:
    """이미 업로드된 포인트에 대해 ColBERT 벡터를 추가/업데이트."""
    if not (runtime.update_colbert and uploaded and EmbeddingType.COLBERT in embed_models):
        print("[INFO] ColBERT update skipped.")
        return

    colbert_model = embed_models[EmbeddingType.COLBERT]
    total_updated = 0

    for i in range(0, len(uploaded), runtime.colbert_batch_size):
        chunk = uploaded[i : i + runtime.colbert_batch_size]
        ids   = [p for p, _ in chunk]
        texts = [t for _, t in chunk]

        # List[np.ndarray] with shape (T_i, H) per text
        col_vecs = list(colbert_model.encode(texts, task="document"))

        col_points = [
            PointVectors(
                id=ids[j],
                vector={EmbeddingType.COLBERT.value: col_vecs[j]},
            )
            for j in range(len(ids))
        ]

        if col_points:
            qdrant.update_colbert_vectors(collection_name, col_points)
            total_updated += len(col_points)

    print(f"[INFO] ColBERT updated: {total_updated} points")


# =============================================================================
# 4. main()
# =============================================================================

def main():
    s = get_settings()
    lang = s.LANGUAGE

    # 1) chunked JSON 위치 설정
    #    필요하면 여기만 자기 프로젝트 구조에 맞게 수정
    #    예: /root/ecm-preprocess-1/output/processed 아래 스캔
    chunk_base_dir = Path(s.OUTPUT_PROCESSED_DIR) if hasattr(s, "OUTPUT_PROCESSED_DIR") else Path("output/processed")
    documents = load_documents_from_chunked_dir(chunk_base_dir)
    if not documents:
        raise RuntimeError("No documents to index from chunked json. Check chunked path.")

    collection_name = s.get_collection_name()

    # 2) 런타임/인덱스 기본값
    runtime = RuntimeOptions()
    hnsw_defaults = HnswDefaults()

    # 3) Embedding 준비
    embed_models, dimensions, sparse_vectors, model_config = prepare_embeddings(
        documents=documents,
        lang=lang,
        runtime=runtime,
    )

    # 4) Qdrant 컬렉션 생성
    qdrant = Qdrant(qdrant_url=s.QDRANT_URL, qdrant_api_key=None)
    create_qdrant_collection(
        qdrant=qdrant,
        collection_name=collection_name,
        model_config=model_config,
        runtime=runtime,
        hnsw_defaults=hnsw_defaults,
    )

    # 5) Dense/Sparse 업로드
    uploaded = upload_dense_sparse_points(
        qdrant=qdrant,
        collection_name=collection_name,
        documents=documents,
        embed_models=embed_models,
        sparse_vectors=sparse_vectors,
        runtime=runtime,
    )

    # 6) ColBERT 업데이트 (옵션)
    update_colbert_vectors(
        qdrant=qdrant,
        collection_name=collection_name,
        uploaded=uploaded,
        embed_models=embed_models,
        runtime=runtime,
    )


if __name__ == "__main__":
    main()
