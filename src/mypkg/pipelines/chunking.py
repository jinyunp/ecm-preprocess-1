#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Chunk ECMiner manual JSON (paragraphs/tables/inline_images) into
section-based chunks suitable for vector DB ingestion.

- Chunk unit: heading level = CHUNK_LEVEL (default: 3)
- Section prefix sentences are injected before actual content.
- Media IDs (Image_ids, Table_ids) are extracted from context text by
  scanning [Image:...] / [Table:...] markers.

Input JSON format (simplified):
{
  "paragraphs": [
    {
      "text": "Chapter 1 ECMiner™ Overview",
      "doc_index": 3,
      "style": "heading 1",
      "source_doc_indices": [3],
      ...
    },
    ...
  ],
  "tables": [
    {
      "tid": "t21",
      "doc_index": 21,
      "table_html": "<table>...</table>",
      ...
    },
    ...
  ],
  "inline_images": [
    {
      "rId": "rId26",
      "doc_index": 65,
      "ocr_text": "...",
      ...
    },
    ...
  ]
}
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


# === CONFIGURABLE PARTS =====================================================

# 기준이 되는 heading level들 (heading 2, heading 3에서 모두 chunk 시작)
CHUNK_LEVELS = {2, 3}

# style -> heading level 매핑 (필요할 때 자유롭게 수정)
STYLE_LEVEL_MAP = {
    "heading 1": 1,
    "heading1": 1,
    "heading 2": 2,
    "heading2": 2,
    "heading 3": 3,
    "heading3": 3,
    # "소제목2"가 포함되는 스타일은 level 4로 취급
    "소제목2": 4,
    "subtitle2": 4,
}


# === HELPER FUNCTIONS =======================================================

def normalize_style(style: Optional[str]) -> str:
    if not style:
        return ""
    return style.strip().lower()


def guess_heading_level(style: Optional[str], text: str) -> Tuple[Optional[int], bool]:
    """
    style과 text를 보고 heading level을 추정한다.
    반환: (level, is_heading)
    """
    s = normalize_style(style)

    # 1) style 기반 매핑
    for key, lvl in STYLE_LEVEL_MAP.items():
        if key in s:
            return lvl, True

    # 2) text 패턴 기반 (fallback)
    t = text.strip()

    # "Chapter 1 ECMiner™ Overview" 같은 경우
    if re.match(r"^Chapter\s+\d+\b", t, flags=re.IGNORECASE):
        return 1, True

    # "1", "1.1", "1.1.1" + 공백 + 제목
    if re.match(r"^\d+(\.\d+)*\s+", t):
        # 깊이에 따라 level 추정 (자유롭게 조정 가능)
        depth = t.split()[0].count(".") + 1
        return min(depth, 4), True

    return None, False


def clean_heading_text(text: str) -> str:
    """
    Section prefix에 넣기 전에 "Chapter #", "#.#" 같은 번호를 제거.
    예:
      "Chapter 1 ECMiner™ Overview" -> "ECMiner™ Overview"
      "1.1.1 Test Section" -> "Test Section"
    """
    t = text.strip()

    # "Chapter 1 xxx"
    t = re.sub(r"^Chapter\s+\d+\s+", "", t, flags=re.IGNORECASE)

    # "1.1.1 " or "1 " 등 번호 제거
    t = re.sub(r"^\d+(\.\d+)*\s+", "", t)

    return t.strip()


def build_section_intro(section_path: List[str]) -> str:
    """
    section_path: ["L1", "L2", "L3", "L4?"] 형태의 제목 리스트.
    길이에 따라 다른 템플릿으로 section prefix 문장 구성.
    여기서는 문장에 넣기 전에 번호(Chapter, 1.1, 1.1.1 등)를 제거한다.
    """
    # 빈 값 제거 + 문장용으로 번호 제거
    titles = [clean_heading_text(t) for t in section_path if t]

    if not titles:
        return ""

    if len(titles) == 3:
        L1, L2, L3 = titles
        return (
            f"This content is about the section {L1}, "
            f"and more specifically it belongs to {L2}, focusing on {L3}."
        )
    elif len(titles) >= 4:
        L1, L2, L3, L4 = titles[:4]
        return (
            f"This content is about the section {L1}, "
            f"and more specifically it belongs to {L2}, "
            f"under the subsection {L3}, with a detailed reference to {L4}."
        )
    else:
        # level 1 or 2만 있는 경우도 안전하게 처리
        if len(titles) == 1:
            return f"This content is about the section {titles[0]}."
        elif len(titles) == 2:
            return (
                f"This content is about the section {titles[0]}, "
                f"and more specifically it belongs to {titles[1]}."
            )
        return ""



def extract_media_ids_from_text(text: str) -> Tuple[List[str], List[str]]:
    """
    Context 텍스트 안에서 [Image:rId##], [Table:tid] 패턴을 찾아
    Image_ids, Table_ids 리스트를 순서대로 반환.
    """
    image_ids = re.findall(r"\[Image:([^\]]+)\]", text)
    table_ids = re.findall(r"\[Table:([^\]]+)\]", text)

    # 순서 유지하면서 중복 제거
    def unique(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return unique(image_ids), unique(table_ids)

def get_default_output_path(input_path: str) -> str:
    """
    입력: .../_sanitized/1장_v3.1_sanitized.json
    출력: .../_chunked/1장_v3.1_chunked.json
    """
    dirpath, filename = os.path.split(input_path)
    name, ext = os.path.splitext(filename)

    # 파일명에서 _sanitized 제거 후 _chunked로 변경
    if name.endswith("_sanitized"):
        base_name = name[: -len("_sanitized")]
    else:
        base_name = name
    out_filename = base_name + "_chunked" + ext

    # 디렉터리에서 _sanitized → _chunked로 변경
    parent_dir, last_dir = os.path.split(dirpath)
    if last_dir == "_sanitized":
        out_dir = os.path.join(parent_dir, "_chunked")
    else:
        # fallback: 그냥 같은 디렉터리에 저장
        out_dir = dirpath

    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, out_filename)


# === CORE CHUNKING LOGIC ====================================================

def build_elements(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    paragraphs, tables, inline_images를 doc_index 기준으로 하나의 시퀀스로 합친다.
    각 element: {"kind": "paragraph"|"table"|"image", "doc_index": int, "obj": original_dict}
    """
    elements: List[Dict[str, Any]] = []

    for p in doc.get("paragraphs", []):
        elements.append(
            {"kind": "paragraph", "doc_index": p.get("doc_index", 0), "obj": p}
        )

    for t in doc.get("tables", []):
        elements.append(
            {"kind": "table", "doc_index": t.get("doc_index", 0), "obj": t}
        )

    for im in doc.get("inline_images", []):
        elements.append(
            {"kind": "image", "doc_index": im.get("doc_index", 0), "obj": im}
        )

    elements.sort(key=lambda e: e["doc_index"])
    return elements


def chunk_document(doc: Dict[str, Any], file_name: str) -> List[Dict[str, Any]]:
    """
    메인 chunking 함수.
    반환: chunk 리스트 (각 chunk는 metadata 필드 포함)
    """
    elements = build_elements(doc)

    # level -> section title (cleaned)
    section_titles: Dict[int, str] = {}

    chunks: List[Dict[str, Any]] = []
    current_chunk: Dict[str, Any] = {}
    current_texts: List[str] = []
    current_pages: List[int] = []

    def flush_current_chunk():
        nonlocal current_chunk, current_texts, current_pages, chunks

        if not current_chunk:
            return

        intro = current_chunk.get("section_intro", "")
        body = "\n\n".join([t for t in current_texts if t.strip()])

        if intro and body:
            context = intro + "\n\n" + body
        elif intro:
            context = intro
        else:
            context = body

        image_ids, table_ids = extract_media_ids_from_text(context)

        chunk_meta = {
            "Context": context,
            "Context_id": current_chunk["context_id"],
            "Is_image": bool(image_ids),
            "Image_ids": image_ids,
            "Is_table": bool(table_ids),
            "Table_ids": table_ids,
            "Section_path": current_chunk["section_path"],
            "Section_length": len(
                [t for t in current_chunk["section_path"] if t]
            ),
            "Page_number": min(current_pages) if current_pages else None,
            "File_name": file_name,
        }

        chunks.append(chunk_meta)

        # reset
        current_chunk = {}
        current_texts = []
        current_pages = []

    # chunk ID 증가용
    chunk_counter = 0

    for el in elements:
        kind = el["kind"]
        obj = el["obj"]

        if kind == "paragraph":
            text = obj.get("text", "") or ""
            style = obj.get("style")
            page = obj.get("page_number")  # 있으면 사용, 없으면 None

            level, is_heading = guess_heading_level(style, text)

            if is_heading:
                # section path 업데이트
                if level is not None:
                    section_titles[level] = text.strip()
                    # 더 깊은 레벨 초기화
                    for k in list(section_titles.keys()):
                        if k > level:
                            del section_titles[k]

                # 🔁 UPDATED: heading 2, 3, 그리고 소제목2(level 4)에서 새 chunk 시작
                if level in CHUNK_LEVELS or level == 4:
                    flush_current_chunk()

                    chunk_counter += 1
                    # 현재까지의 모든 상위 제목을 path로 사용 (1,2,3,4...)
                    path_levels = sorted(section_titles.keys())
                    section_path = [section_titles[lvl] for lvl in path_levels]
                    section_intro = build_section_intro(section_path)

                    current_chunk = {
                        "context_id": f"{os.path.splitext(file_name)[0]}::chunk_{chunk_counter:04d}",
                        "section_path": section_path,
                        "section_intro": section_intro,
                    }
                    current_texts = []
                    current_pages = []

                # heading 문단 자체는 body 텍스트에 포함하지 않음
                continue

            # 일반 문단 (heading 아님)
            if not current_chunk:
                # 🔁 UPDATED: 아직 chunk가 없고, Chapter(heading1)만 있는 경우에는
                #            heading1 기준으로 intro를 가진 chunk를 시작
                available_levels = [lvl for lvl in section_titles.keys() if lvl in CHUNK_LEVELS]
                if not available_levels and 1 in section_titles:
                    base_level = 1

                    chunk_counter += 1
                    path_levels = sorted(
                        [lvl for lvl in section_titles.keys() if lvl <= base_level]
                    )
                    section_path = [section_titles[lvl] for lvl in path_levels]
                    section_intro = build_section_intro(section_path)

                    current_chunk = {
                        "context_id": f"{os.path.splitext(file_name)[0]}::chunk_{chunk_counter:04d}",
                        "section_path": section_path,
                        "section_intro": section_intro,
                    }
                    current_texts = []
                    current_pages = []
                else:
                    # heading2/3 기반 chunk가 이미 생성되어야 하는 상황인데
                    # 아직 없으면 그냥 스킵
                    continue

            if text.strip():
                current_texts.append(text)

            if page is not None:
                current_pages.append(page)


        elif kind == "table":
            # 일반 문단 (heading 아님)
            if not current_chunk:
                # 아직 chunk 시작 전이라면 스킵
                continue

            table = obj
            tid = table.get("tid")
            page = table.get("page_number")

            if tid:
                current_texts.append(f"[Table:{tid}]")
            if page is not None:
                current_pages.append(page)
                
        elif kind == "image":
            if not current_chunk:
                # 아직 chunk 시작 전이면 스킵
                continue

            im = obj
            rid = im.get("rId")
            page = im.get("page_number")

            if rid:
                current_texts.append(f"[Image:{rid}]")
            if page is not None:
                current_pages.append(page)


    # 마지막 chunk flush
    flush_current_chunk()

    return chunks


# === MAIN ===================================================================

def main(input_path: str, output_path: Optional[str] = None):
    with open(input_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    file_name = os.path.basename(input_path)

    chunks = chunk_document(doc, file_name=file_name)

    if not output_path:
        # UPDATED: sanitized → chunked 디렉터리 및 파일명 자동 생성
        output_path = get_default_output_path(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chunk_manual.py <input_json_path> [output_json_path]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else None
    main(in_path, out_path)
    
# python src/mypkg/pipelines/chunking.py /home/jinypark/vscodeProjects/ecm-preprocess-1/output/processed/Appendix_v2/v0/_sanitized/Appendix_v2_sanitized.json

# python src/mypkg/pipelines/chunking.py /home/jinypark/vscodeProjects/ecm-preprocess-1/output/processed/1장_v3.1/v0/_sanitized/1장_v3.1_sanitized.json