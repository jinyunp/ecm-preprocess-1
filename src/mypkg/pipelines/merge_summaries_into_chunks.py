#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
이미지 & 테이블 summary 를 chunk Context 내 placeholder 바로 뒤에 인라인 삽입.

- [image:rId#] / [Image:rId#] → "[image:rId#] <img_summary>"
- [table:tbl#] / [Table:tbl#] → "[table:tbl#] <tbl_summary>"

CLI:
    python merge_summaries_into_chunks.py <문서명>

예:
    python merge_summaries_into_chunks.py 1장_v3.1
"""

import json
import os
import re
import sys
from typing import Dict, Any, List, Tuple


# -------------------------------------------------------------------------
# 공통: JSON 로더
# -------------------------------------------------------------------------
def load_image_summaries(image_llm_path: str) -> Dict[str, str]:
    """
    image_llm JSON: [{"rId": "rId7", "llm_text": "..."}, ...]
    → { "rid7": "..." } (소문자 key)
    """
    try:
        with open(image_llm_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[WARN] image_llm file not found: {image_llm_path}")
        return {}

    mapping: Dict[str, str] = {}

    if isinstance(data, list):
        for item in data:
            rid_raw = item.get("rId")
            text = (item.get("llm_text") or "").strip()
            if not rid_raw or not text:
                continue
            rid_key = rid_raw.strip().lower()
            mapping[rid_key] = text
    else:
        print(f"[WARN] image_llm file is not a list: {image_llm_path}", file=sys.stderr)

    print(f"[INFO] Loaded {len(mapping)} image summaries from {image_llm_path}")
    return mapping


def load_table_summaries(table_llm_path: str) -> Dict[str, str]:
    """
    table_llm JSON: [{"table_id": "tbl1", "llm_text": "..."}, ...]
    → { "tbl1": "..." } (소문자 key)
    """
    try:
        with open(table_llm_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[WARN] table_llm file not found: {table_llm_path}")
        return {}

    mapping: Dict[str, str] = {}

    if isinstance(data, list):
        for item in data:
            tid_raw = item.get("table_id") or item.get("id")
            text = (item.get("llm_text") or "").strip()
            if not tid_raw or not text:
                continue
            tid_key = str(tid_raw).strip().lower()
            mapping[tid_key] = text
    else:
        print(f"[WARN] table_llm file is not a list: {table_llm_path}", file=sys.stderr)

    print(f"[INFO] Loaded {len(mapping)} table summaries from {table_llm_path}")
    return mapping


# -------------------------------------------------------------------------
# Context 내 placeholder 치환
# -------------------------------------------------------------------------
def inject_summaries_into_context(
    context: str,
    img_summaries: Dict[str, str],
    table_summaries: Dict[str, str],
) -> str:
    """
    [image:rId#]/[Image:rId#] and [table:tbl#]/[Table:tbl#] 바로 뒤에 summary를 붙인다.
    """

    def _replace_image(match: re.Match) -> str:
        tag = match.group("tag")      # image or Image
        rid_raw = match.group("rid")
        rid_key = rid_raw.strip().lower()
        summary = img_summaries.get(rid_key)
        if not summary:
            return match.group(0)
        return f"[{tag}:{rid_raw}] {summary}"

    def _replace_table(match: re.Match) -> str:
        tag = match.group("tag")      # table or Table
        tid_raw = match.group("tid")
        tid_key = tid_raw.strip().lower()
        summary = table_summaries.get(tid_key)
        if not summary:
            return match.group(0)
        return f"[{tag}:{tid_raw}] {summary}"

    # image placeholder
    img_pattern = re.compile(r"\[(?P<tag>[Ii]mage):(?P<rid>[^\]]+)\]")
    context = img_pattern.sub(_replace_image, context)

    # table placeholder
    tbl_pattern = re.compile(r"\[(?P<tag>[Tt]able):(?P<tid>[^\]]+)\]")
    context = tbl_pattern.sub(_replace_table, context)

    return context


# -------------------------------------------------------------------------
# 전체 merge
# -------------------------------------------------------------------------
def merge_image_and_table_summaries(
    chunked_path: str,
    image_llm_path: str,
    table_llm_path: str,
    output_path: str,
) -> str:
    with open(chunked_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    img_summaries = load_image_summaries(image_llm_path)
    table_summaries = load_table_summaries(table_llm_path)

    updated_chunks: List[Dict[str, Any]] = []

    for ch in chunks:
        ctx = ch.get("Context", "")
        if isinstance(ctx, str):
            ch["Context"] = inject_summaries_into_context(
                ctx,
                img_summaries=img_summaries,
                table_summaries=table_summaries,
            )
        updated_chunks.append(ch)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_chunks, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved updated chunks with image+table summaries → {output_path}")
    return output_path


# -------------------------------------------------------------------------
# CLI: 문서명만 받으면 경로 자동 구성
# -------------------------------------------------------------------------
def build_paths(docname: str) -> Tuple[str, str, str, str]:
    base = f"/root/ecm-preprocess-1/output/processed/{docname}/v0/_chunked"
    chunked = f"{base}/{docname}_chunked.json"
    img_llm = f"{base}/{docname}_image_llm.json"
    tbl_llm = f"{base}/{docname}_table_llm.json"
    out = f"{base}/{docname}_chunked_with_imgsum.json"
    return chunked, img_llm, tbl_llm, out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python merge_summaries_into_chunks.py <문서명>")
        print("예:")
        print("  python merge_summaries_into_chunks.py 1장_v3.1")
        sys.exit(1)

    docname = sys.argv[1].strip()
    chunked_json, image_llm_json, table_llm_json, output_json = build_paths(docname)

    print("[INFO] chunk JSON      :", chunked_json)
    print("[INFO] image LLM JSON  :", image_llm_json)
    print("[INFO] table LLM JSON  :", table_llm_json)
    print("[INFO] output JSON     :", output_json)

    merge_image_and_table_summaries(
        chunked_path=chunked_json,
        image_llm_path=image_llm_json,
        table_llm_path=table_llm_json,
        output_path=output_json,
    )
