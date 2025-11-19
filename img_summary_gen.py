#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import json
import os
import sys
from typing import Any, Dict, List

import requests

# ---- Qwen (Ollama) 설정 ----
OLLAMA_URL = "http://localhost:11434/api/generate"
QWEN_MODEL = "qwen2.5vl:3b"

SUMMARY_PROMPT = (
    "You see a screenshot or UI image from ECMiner data mining software. "
    "In one concise English sentence (max 25 words), summarize the main purpose and content of this screen."
)


def encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def summarize_image_with_qwen(image_path: str) -> str:
    """
    saved_path에 있는 이미지를 Qwen-VL(qwen2.5vl:3b)에 보내
    한 줄짜리 semantic summary를 받아온다.
    """
    img_b64 = encode_image_to_base64(image_path)

    payload = {
        "model": QWEN_MODEL,
        "prompt": SUMMARY_PROMPT,
        "images": [img_b64],
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        summary = (data.get("response") or "").strip()
        return summary
    except Exception as e:
        print(f"[WARN] Failed to summarize image '{image_path}': {e}", file=sys.stderr)
        return ""


def get_chunked_output_path_for_images(sanitized_json_path: str) -> str:
    """
    입력:  .../_sanitized/1장_v3.1_sanitized.json
    출력: .../_chunked/1장_v3.1_image_llm.json
    """
    dirpath, filename = os.path.split(sanitized_json_path)
    name, ext = os.path.splitext(filename)

    # 파일명에서 _sanitized 제거
    if name.endswith("_sanitized"):
        base_name = name[: -len("_sanitized")]
    else:
        base_name = name

    out_filename = base_name + "_image_llm" + ext

    parent_dir, last_dir = os.path.split(dirpath)
    if last_dir == "_sanitized":
        out_dir = os.path.join(parent_dir, "_chunked")
    else:
        # fallback: 같은 디렉터리
        out_dir = dirpath

    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, out_filename)


def process_inline_images_to_chunked(input_path: str) -> str:
    """
    sanitized json에서 inline_images 정보만 읽고,
    각 이미지에 대해 llm_text를 생성한 뒤,
    _chunked 폴더에 이미지 요약 전용 JSON을 저장한다.
    (기존 sanitized json은 수정하지 않음)
    """
    with open(input_path, "r", encoding="utf-8") as f:
        doc: Dict[str, Any] = json.load(f)

    inline_images: List[Dict[str, Any]] = doc.get("inline_images", [])

    results: List[Dict[str, Any]] = []

    for img in inline_images:
        rId = img.get("rId")
        filename = img.get("filename")
        saved_path = img.get("saved_path")

        if not saved_path:
            print(f"[WARN] No saved_path for image rId={rId}", file=sys.stderr)
            continue

        if not os.path.exists(saved_path):
            print(f"[WARN] Image file not found: {saved_path}", file=sys.stderr)
            continue

        print(f"[INFO] Summarizing image: rId={rId}, path={saved_path}")
        summary = summarize_image_with_qwen(saved_path)

        results.append(
            {
                "rId": rId,
                "filename": filename,
                "saved_path": saved_path,
                "llm_text": summary,
            }
        )

    output_path = get_chunked_output_path_for_images(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved image LLM summaries to {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize_images_to_chunked.py <sanitized_json_path>")
        sys.exit(1)

    input_json = sys.argv[1]
    process_inline_images_to_chunked(input_json)

#python summarize_images_to_chunked.py \
#  /home/jinypark/vscodeProjects/ecm-preprocess-1/output/processed/1장_v3.1/v0/_sanitized/1장_v3.1_sanitized.json
