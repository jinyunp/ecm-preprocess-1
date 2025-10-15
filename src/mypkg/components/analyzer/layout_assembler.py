"""
레이아웃 어셈블러 (Layout Assembler)

개요
- 섹션, 리스트, 테이블, 인라인 이미지 등 다양한 분석 결과로부터 생성된 콘텐츠 블록을 받아 섹션 트리에 배치합니다.
- 최종적으로 각 섹션은 자신이 포함하는 블록 목록을 보유하게 됩니다.

핵심 로직
1.  블록 취합: `section_analyzer`, `list_table_analyzer` 등에서 생성된 블록을 하나로 모읍니다.
2.  블록 할당 (`assign_blocks_to_sections`): 섹션의 span 범위를 기준으로 블록을 재귀적으로 배치합니다.
3.  블록 생성 헬퍼:
    - `build_paragraph_blocks`: 다른 분석기에 소비되지 않은 순수 문단을 ContentBlock으로 생성합니다.
    - `build_inline_image_blocks`: 인라인 이미지 메타를 ContentBlock으로 변환합니다.

입력
- `sections`: 섹션 분석 결과 트리 구조
- `blocks`: 리스트/테이블 등에서 생성된 ContentBlock 리스트
- `paragraphs`: sanitized 데이터의 문단 리스트
- `skip_docidx`, `heading_idx`: 이미 사용된 문단의 doc_index 집합

출력
- 함수 호출 후 `sections` 객체가 in-place로 수정되어 각 섹션의 `blocks`/`block_ids`가 채워집니다.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from mypkg.core.docjson_types import ContentBlock, InlineImageData, Section


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text).strip()


def assign_blocks_to_sections(sections: List[Section], blocks: List[ContentBlock]) -> None:
    """섹션의 [start, end) 범위에 포함되는 블록을 재귀적으로 배치한다."""

    def put(section: Section, candidates: List[ContentBlock]):
        start, end = section.span
        mine = [b for b in candidates if start <= b.doc_index < end]

        for child in section.subsections:
            child_candidates = [b for b in mine if child.span[0] <= b.doc_index < child.span[1]]
            put(child, child_candidates)
            child_ids = {id(b) for b in child.blocks}
            mine = [b for b in mine if id(b) not in child_ids]

        mine.sort(key=lambda b: b.doc_index)
        section.blocks.extend(mine)
        section.block_ids.extend([b.id for b in mine])

    for root in sections:
        put(root, blocks)


def build_paragraph_blocks(paragraphs: List[Dict[str, Any]], skip_docidx: set, heading_idx: set) -> List[ContentBlock]:
    """제목/리스트 등에 소비되지 않은 문단만 ContentBlock으로 생성한다."""
    blocks: List[ContentBlock] = []
    for info in sorted(paragraphs, key=lambda x: x.get("doc_index", 0)):
        doc_index = info.get("doc_index")
        if doc_index in skip_docidx or doc_index in heading_idx:
            continue
        text = _collapse_spaces(info.get("text") or "")
        if not text:
            continue
        blocks.append(
            ContentBlock(
                id=f"p{doc_index}",
                type="paragraph",
                doc_index=doc_index,
                text=text,
            )
        )
    return blocks


def build_inline_image_blocks(inline_images: List[Dict[str, Any]] | None) -> List[ContentBlock]:
    """인라인 이미지 메타를 image 타입 ContentBlock으로 변환한다."""
    blocks: List[ContentBlock] = []
    for idx, info in enumerate(inline_images or []):
        data = InlineImageData.from_dict(info) if not isinstance(info, InlineImageData) else info
        doc_index = data.doc_index
        if doc_index is None:
            if data.doc_indices:
                doc_index = data.doc_indices[0]
            else:
                doc_index = -1

        block_id = info.get("id") if isinstance(info, dict) else None
        if not block_id:
            rid = getattr(data, "rid", None)
            block_id = f"img_{rid or idx}"

        blocks.append(
            ContentBlock(
                id=block_id,
                type="image",
                doc_index=doc_index,
                inline_image=data,
            )
        )
    return blocks
