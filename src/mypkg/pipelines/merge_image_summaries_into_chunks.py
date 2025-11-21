#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import re
import sys
from typing import Dict, Any, List


def load_image_summaries(image_llm_path: str) -> Dict[str, str]:
    """
    image_llm JSON에서 rId -> llm_text 매핑을 만든다.
    [
      {"rId": "rId7", "llm_text": "...."},
      ...
    ]
    """
    with open(image_llm_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, str] = {}
    if isinstance(data, list):
        for item in data:
            rId = item.get("rId")
            text = (item.get("llm_text") or "").strip()
            if rId and text:
                mapping[rId] = text
    else:
        print(f"[WARN] image_llm file is not a list: {image_llm_path}", file=sys.stderr)

    print(f"[INFO] Loaded {len(mapping)} image summaries from {image_llm_path}")
    return mapping


def inject_summaries_into_context(context: str, img_summaries: Dict[str, str]) -> str:
    """
    Context 문자열 안의 [Image:rIdXX] 패턴을 찾아
    [Image:rIdXX] <summary> 형태로 summary를 붙여준다.

    주의:
      - 대문자 I: [Image:...] 만 처리 (원본 텍스트의 [image:...] 는 그대로 둠)
      - summary가 없으면 원문 그대로 유지
    """
    pattern = re.compile(r"\[Image:(?P<rid>[^\]]+)\]")

    def repl(match: re.Match) -> str:
        rid = match.group("rid")
        summary = img_summaries.get(rid)
        if not summary:
            return match.group(0)
        # 이미 summary가 붙어 있는 경우 중복 방지 (간단 체크)
        full = match.group(0)
        # 예: "[Image:rId7] some text"
        # 이미 뒤에 summary 비슷한 게 있다면 안 붙이는게 좋지만,
        # 여기서는 단순하게 placeholder 바로 뒤에만 붙는다고 가정
        return f"{full} {summary}"

    return pattern.sub(repl, context)


def merge_image_summaries(chunked_path: str, image_llm_path: str, output_path: str = None) -> str:
    """
    chunked JSON의 각 Context에 image_llm 요약을 주입한다.
    - chunked_path: ..._chunked.json
    - image_llm_path: ..._image_llm.json
    - output_path: 없으면 <chunked>_with_imgsum.json 으로 저장
    """
    with open(chunked_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    img_summaries = load_image_summaries(image_llm_path)

    updated_chunks: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks, start=1):
        ctx = ch.get("Context", "")
        if not isinstance(ctx, str):
            updated_chunks.append(ch)
            continue

        new_ctx = inject_summaries_into_context(ctx, img_summaries)
        ch["Context"] = new_ctx
        updated_chunks.append(ch)

    if not output_path:
        stem, ext = os.path.splitext(chunked_path)
        output_path = stem + "_with_imgsum" + ext

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_chunks, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved updated chunks with image summaries to {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_image_summaries_into_chunks.py <chunked_json> <image_llm_json> [output_json]")
        sys.exit(1)

    chunked_json = sys.argv[1]
    image_llm_json = sys.argv[2]
    out_json = sys.argv[3] if len(sys.argv) >= 4 else None

    merge_image_summaries(chunked_json, image_llm_json, out_json)
