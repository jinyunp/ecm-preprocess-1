# infra/vectorstore.py
from typing import Dict, Optional, List
from config.configs import EmbeddingType, EmbeddingModelConfig
import logging, time
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    PointVectors,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    Filter, 
    FieldCondition, 
    MatchValue, 
    PointIdsList,
    ExtendedPointId,
    PayloadSchemaType
)

class Qdrant():
    def __init__(self, qdrant_url: str, qdrant_api_key: Optional[str] = None):
        self.client = QdrantClient(
                url     = qdrant_url,
                api_key = qdrant_api_key,
        )

    # -------------------------------------------------------------
    #           컬렉션 생성/삭제
    # ------------------------------------------------------------- 
    def create_collection(self,
                      collection_name: str,
                      model_config: EmbeddingModelConfig,
                      hnsw_config: Optional[HnswConfigDiff] = None):
        try:
            dense_vector_params = {}
            sparse_vector_params = {}
            
            if model_config.embed_type.get(EmbeddingType.SPARSE):
                sparse_vector_params[EmbeddingType.SPARSE.value] = SparseVectorParams(
                    index = SparseIndexParams(on_disk = False)
                )
            
            if model_config.embed_type.get(EmbeddingType.DENSE):
                dense_vector_params[EmbeddingType.DENSE.value] = VectorParams(
                    size = model_config.dimensions[EmbeddingType.DENSE],
                    distance = Distance[model_config.dense_distance.upper()]
                )

            if model_config.embed_type.get(EmbeddingType.COLBERT):
                dense_vector_params[EmbeddingType.COLBERT.value] = VectorParams(
                    size = model_config.dimensions[EmbeddingType.COLBERT],
                    distance = Distance.COSINE,
                    multivector_config = MultiVectorConfig(comparator = MultiVectorComparator.MAX_SIM),
                    hnsw_config = HnswConfigDiff(m = 0)  # HNSW 비활성화
                )

            self.client.create_collection(
                collection_name = collection_name,
                vectors_config = dense_vector_params or None,
                sparse_vectors_config = sparse_vector_params or None,
                hnsw_config = hnsw_config or None
            )

            # metadata.filename에 인덱스 생성
            self.create_payload_index(collection_name, field_name="metadata.filename", field_schema=PayloadSchemaType.KEYWORD)

            logging.info(f"컬렉션 생성 완료: {collection_name}")
    
        except Exception as e:
            logging.error(f"컬렉션 생성 실패: {e}")
            raise

    def delete_collection(self, collection_name: str) -> None:
        try:
            # 현재 존재하는 컬렉션 목록 확인
            collections = [c.name for c in self.client.get_collections().collections]
            if collection_name not in collections:
                logging.info(f"'{collection_name}' 컬렉션이 존재하지 않습니다. 삭제하지 않습니다.")
                return

            # 컬렉션 삭제 요청
            self.client.delete_collection(collection_name = collection_name)
            
            # 삭제 완료 확인 (최대 5초 대기)
            max_attempts = 5
            for _ in range(max_attempts):
                collections = [c.name for c in self.client.get_collections().collections]
                if collection_name not in collections:
                    logging.info(f"'{collection_name}' 컬렉션이 성공적으로 삭제됐습니다.")
                    return
                time.sleep(0.1)
            
            logging.warning(f"'{collection_name}' 삭제 {max_attempts * 0.1} 초 대기")
        
        except Exception as e:
            logging.error(f"컬렉션을 삭제하지 못했습니다. '{collection_name}': {e}")
            raise

    def has_collection(self, collection_name: str) -> bool:
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            return collection_name in collections
        except Exception as e:
            logging.error(f"컬렉션 존재 여부 확인 실패: {e}")
            return False

    # -------------------------------------------------------------
    #           페이로드 인덱스 관리
    # -------------------------------------------------------------
    def create_payload_index(self, collection_name: str, field_name: str, field_schema: PayloadSchemaType) -> None:
        """페이로드 필드에 인덱스 생성."""
        try:
            self.client.create_payload_index(
                collection_name     = collection_name,
                field_name          = field_name,
                field_schema        = field_schema,
            )
            logging.info(f"페이로드 인덱스 생성 완료: {field_name} in {collection_name}")
        
        except Exception as e:
            logging.error(f"페이로드 인덱스 생성 실패: {field_name} in {collection_name}: {e}")
            raise

    # -------------------------------------------------------------
    #           기본적인 벡터 업로드
    # -------------------------------------------------------------
    def upload_vectors(self, collection_name: str, points: List[PointStruct]) -> None:
        try:
            self.client.upsert(collection_name = collection_name, points = points, wait = True)
            logging.info(f"포인트 업로드 완료: {len(points)} points to {collection_name}")
        except Exception as e:
            logging.error(f"포인트 업로드 실패: {e}")
            raise
    
    # -------------------------------------------------------------
    #           멀티벡터를 지원하기 때문에 Colbart 벡터 업로드
    # -------------------------------------------------------------
    def update_colbert_vectors(self, collection_name: str, vectors: List[PointVectors]) -> None:
        try:    
            self.client.update_vectors(collection_name = collection_name, points = vectors, wait = True)
        except Exception as e:
            logging.error(f"ColBERT 벡터 업데이트 실패: {e}")
            raise
    
    # -------------------------------------------------------------
    #           metadata.filename을 이용한 포인트 검색
    # -------------------------------------------------------------
    def search_points_by_filename(self, collection_name: str, filename: str) -> List[ExtendedPointId]:
        """metadata.filename으로 포인트 검색."""
        try:
            result = self.client.scroll(
                collection_name = collection_name,
                scroll_filter   = Filter(
                    must = [FieldCondition(
                        key = "metadata.general_info.filename", 
                        match = MatchValue(
                            value = filename)
                            )]
                ),
            )
            point_ids = [point.id for point in result[0]] if result[0] else []
            logging.info(f"검색 완료: {len(point_ids)} points found for filename={filename}")
            return point_ids
        
        except Exception as e:
            logging.error(f"포인트 검색 실패: {e}")
            return []

    def delete_points_by_ids(self, collection_name: str, point_ids: List[ExtendedPointId]) -> int:
        """ID 목록으로 포인트 삭제."""
        try:
            if not point_ids:
                logging.info("삭제할 포인트 ID 없음")
                return 0
            self.client.delete(
                collection_name = collection_name,
                points_selector = PointIdsList(points=point_ids)
            )
            logging.info(f"포인트 삭제 완료: {len(point_ids)} points from {collection_name}")
            return len(point_ids)
        except Exception as e:
            logging.error(f"포인트 삭제 실패: {e}")
            return 0

    def count_points(self, collection_name: str) -> int:
        """컬렉션 내 포인트 수 확인."""
        try:
            count = self.client.count(
                collection_name = collection_name
                ).count
            logging.info(f"컬렉션 {collection_name}에 {count} 포인트 존재")
            return count

        except Exception as e:
            logging.error(f"포인트 수 확인 실패: {e}")
            return 0