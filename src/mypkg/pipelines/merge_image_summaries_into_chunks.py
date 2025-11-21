#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
이미지 summary를 chunk Context 내 [image:rID#] / [Image:rID#] 바로 뒤에 삽입하는 script.

CLI는 "파일명"만 받으면 전체 경로를 자동 구성한다.

사용 예:
    python merge_image_summaries_into_chunks.py 1장_v3.1

경로 규칙:
  chunked_json:
    /root/ecm-preprocess-1/output/processed/{파일명}/v0/_chunked/{파일명}_chunked.json

  image_llm_json:
    /root/ecm-preprocess-1/output/processed/{파일명}/v0/_chunked/{파일명}_image_llm.json

  output_json:
    /root/ecm-preprocess-1/output/processed/{파일명}/v0/_chunked/{파일명}_chunked_with_imgsum.json
"""

import json
import os
import re
import sys
from typing import Dict, Any, List


# -------------------------------------------------------------------------
# 1) 이미지 LLM 요약 로드: rId → summary (소문자 key)
# -------------------------------------------------------------------------
def load_image_summaries(image_llm_path: str) -> Dict[str, str]:
    with open(image_llm_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, str] = {}

    if isinstance(data, list):
        for item in data:
            rid_raw = item.get("rId")
            text = (item.get("llm_text") or "").strip()
            if not rid_raw or not text:
                continue

            # rId 대소문자 섞여 있을 수 있으므로 소문자로 통일해서 key 저장
            rid_key = rid_raw.strip().lower()
            mapping[rid_key] = text
    else:
        print(f"[WARN] image_llm file is not a list: {image_llm_path}", file=sys.stderr)

    print(f"[INFO] Loaded {len(mapping)} image summaries from {image_llm_path}")
    return mapping


# -------------------------------------------------------------------------
# 2) Context 내 [image:rID#] / [Image:rID#] 바로 뒤에 summary 삽입
# -------------------------------------------------------------------------
def inject_summaries_into_context(context: str, img_summaries: Dict[str, str]) -> str:
    """
    Context 문자열 안의 [image:rIdXX], [Image:rIdXX] 패턴을 찾아
    그대로 두되, 그 *바로 뒤에* summary 텍스트를 붙인다.

    예:
      "[image:rId7] 어떤 설명"  +  summary="(요약)"
      -> "[image:rId7] (요약) 어떤 설명"

    - 태그의 대소문자(Image / image)는 그대로 유지.
    - rId는 case-insensitive 로 lookup.
    """

    # [Image:rId7], [image:rId7] 둘 다 잡되 tag는 그대로 둠
    pattern = re.compile(r"\[(?P<tag>[Ii]mage):(?P<rid>[^\]]+)\]")

    def repl(match: re.Match) -> str:
        tag = match.group("tag")          # Image or image
        rid_raw = match.group("rid")      # rId7, rid7 등
        rid_key = rid_raw.strip().lower()

        summary = img_summaries.get(rid_key)
        if not summary:
            # 요약이 없는 rId는 그대로 놔둔다
            return match.group(0)

        # 아주 단순한 replace: "[image:rId7]" -> "[image:rId7] summary"
        return f"[{tag}:{rid_raw}] {summary}"

    return pattern.sub(repl, context)


# -------------------------------------------------------------------------
# 3) 전체 merge 수행
# -------------------------------------------------------------------------
def merge_image_summaries(chunked_path: str, image_llm_path: str, output_path: str) -> str:
    with open(chunked_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    img_summaries = load_image_summaries(image_llm_path)

    updated_chunks: List[Dict[str, Any]] = []

    for ch in chunks:
        ctx = ch.get("Context", "")
        if isinstance(ctx, str):
            ch["Context"] = inject_summaries_into_context(ctx, img_summaries)
        updated_chunks.append(ch)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_chunks, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved updated chunks with image summaries → {output_path}")
    return output_path


# -------------------------------------------------------------------------
# 4) CLI: 파일명 하나만 받으면 경로 자동 구성
# -------------------------------------------------------------------------
def build_paths(docname: str):
    base = f"/root/ecm-preprocess-1/output/processed/{docname}/v0/_chunked"
    chunked = f"{base}/{docname}_chunked.json"
    img_llm = f"{base}/{docname}_image_llm.json"
    out = f"{base}/{docname}_chunked_with_imgsum.json"
    return chunked, img_llm, out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python merge_image_summaries_into_chunks.py <문서명>")
        print("예:")
        print("  python merge_image_summaries_into_chunks.py 1장_v3.1")
        sys.exit(1)

    docname = sys.argv[1].strip()
    chunked_json, image_llm_json, output_json = build_paths(docname)

    print("[INFO] chunk JSON      :", chunked_json)
    print("[INFO] image LLM JSON  :", image_llm_json)
    print("[INFO] output JSON     :", output_json)

    merge_image_summaries(chunked_json, image_llm_json, output_json)
