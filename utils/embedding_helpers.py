from typing import List, Dict, Any
from pathlib import Path
import json, uuid
from langchain_core.documents import Document
from config.configs import EmbeddingType
from qdrant_client.http.models import PointStruct, SparseVector

from config.settings import get_settings


# def _extract_filename_from_record(data: Dict[str, Any], fallback_name: str) -> str:
#     """
#     filename 후보 우선순위:
#     1) metadata.filename
#     2) metadata.general_info.filename
#     3) id의 'filename#chunk' 형태에서 '#' 앞 부분
#     4) 파일 경로명(fallback)
#     """
#     md = data.get("metadata") or {}
#     gi = md.get("general_info") or {}

#     if isinstance(md.get("filename"), str) and md["filename"].strip():
#         return md["filename"].strip()

#     if isinstance(gi.get("filename"), str) and gi["filename"].strip():
#         return gi["filename"].strip()

#     rid = data.get("id")
#     if isinstance(rid, str) and "#" in rid:
#         head = rid.split("#", 1)[0].strip()
#         if head:
#             return head

#     return fallback_name
from typing import List, Dict, Any
from pathlib import Path
import json
from config.settings import get_settings

def load_documents_from_jsonls(paths: List[Path]) -> List[Document]:
    s = get_settings()
    docs: List[Document] = []

    for p in paths:
        if not p.exists():
            continue

        with open(p, "r", encoding="utf-8") as f:
            seq = 0  # (id 누락 시에만 fallback 생성)
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # context/content 통일
                text = data.get("context") or data.get("content")
                if not text:
                    continue

                # metadata / general_info 안전 추출
                md: Dict[str, Any] = (data.get("metadata") or {})
                if not isinstance(md, dict):
                    md = {}
                gi: Dict[str, Any] = md.get("general_info") or {}
                if not isinstance(gi, dict):
                    gi = {}

                # filename / stem / versioned 보강 (기존 값 보존 우선)
                # - 우선순위: filename_stem > filename.stem > jsonl 파일명 stem
                filename = gi.get("filename")
                stem = gi.get("filename_stem") \
                       or (Path(filename).stem if isinstance(filename, str) and filename else None) \
                       or Path(p).stem
                gi.setdefault("filename_stem", stem)

                ver_stem = gi.get("filename_versioned") or f"{stem}_{s.VERSION}"
                gi.setdefault("filename_versioned", ver_stem)

                # point용 문자열 ID 연동
                # - 우선순위: unique_id > id(top-level) > metadata.id > (없으면 fallback)
                pid = data.get("unique_id") or data.get("id") or md.get("id")
                if not pid:
                    seq += 1
                    pid = f"{ver_stem}_{seq:06d}"

                # 메타데이터에 확정 반영 (generate_points에서 external_id/id 사용)
                md["id"] = pid
                md.setdefault("external_id", pid)

                # 정리한 general_info를 되돌려놓기
                md["general_info"] = gi

                docs.append(Document(page_content=text, metadata=dict(md)))

    return docs


def generate_points(documents: List[Document], embed_models: Dict, sparse_vectors: List, runtime: Any) -> List[PointStruct]:
    """문서들로부터 Qdrant 포인트 생성."""
    points: List[PointStruct] = []
    contents = [d.page_content for d in documents]
    metas = [d.metadata for d in documents]

    dense_vecs: List[Any] = []
    if EmbeddingType.DENSE in embed_models:
        dense_model = embed_models[EmbeddingType.DENSE]
        dense_vecs = list(dense_model.encode(contents, task="document"))

    sparse_batch = sparse_vectors if sparse_vectors else []

    for j in range(len(documents)):
        md: Dict[str, Any] = metas[j] or {}
        gi: Dict[str, Any] = md.get("general_info") or {}

        # 사람이 보는 문자열 ID(버전 포함)를 payload에 노출
        unique_id = md.get("external_id") or md.get("id") or md.get("unique_id")
        if not unique_id:
            # 최후 fallback: filename_versioned + 인덱스
            ver_stem = gi.get("filename_versioned") or gi.get("filename_stem") or "doc"
            unique_id = f"{ver_stem}_{(j+1):06d}"

        # Qdrant point_id는 UUID(v4)
        pid = str(uuid.uuid4())

        vectors: Dict[str, Any] = {}

        if dense_vecs:
            dv = dense_vecs[j]
            vectors[EmbeddingType.DENSE.value] = dv.tolist() if hasattr(dv, "tolist") else list(dv)

        if sparse_batch and j < len(sparse_batch):
            sv = sparse_batch[j]
            if sv and sv.get("indices") and sv.get("values"):
                vectors[EmbeddingType.SPARSE.value] = SparseVector(
                    indices = sv["indices"],
                    values  = sv["values"]
                )

        points.append(PointStruct(
            id      = pid,  # <-- Qdrant 저장용 UUID
            vector  = vectors,
            payload = {
                "content": contents[j],
                "unique_id": unique_id,   # <-- 요청대로 payload 최상위에 포함
                "metadata": md,           # 메타데이터는 그대로 유지(여기에 id/external_id가 있을 수 있음)
            },
        ))

    return points