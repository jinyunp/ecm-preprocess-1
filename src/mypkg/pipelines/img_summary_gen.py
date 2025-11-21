#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
from typing import Any, Dict, List

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# ---------------- Qwen2-VL 설정 ----------------

# 원하는 Qwen2-VL 모델 이름 (필요시 변경)
MODEL_NAME = "/workspace/qwen" #"Qwen/Qwen2-VL-7B-Instruct"

# 한 줄 요약 프롬프트
SUMMARY_PROMPT = (
    "You see a screenshot or UI image from ECMiner data mining software. "
    "In concise English sentence (max 25 words), summarize the main purpose and content of this screen."
    "Explan exactly what the image means. According to whether ii's a screenshot, icon, diagram, etc"
)

print(f"[INFO] Loading model: {MODEL_NAME}")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto" if device == "cuda" else None,
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print(f"[INFO] Model loaded on device: {device}")

def clean_qwen_output(raw: str) -> str:
    """
    Qwen 출력에서 system/user/assistant 프롬프트 부분을 제거하고
    마지막 assistant 응답 문장만 남긴다.
    예:
      "system\\n...\\nuser\\n...\\nassistant\\n실제 요약문"
      -> "실제 요약문"
    """
    text = (raw or "").strip()

    # 가장 마지막 'assistant\n' 기준으로 잘라서 그 뒤만 사용
    for sep in ["\nassistant\n", "assistant\n", "assistant:"]:
        if sep in text:
            text = text.split(sep, 1)[1].strip()
    return text


def summarize_image_with_qwen(image_path: str) -> str:
    """
    로컬 Qwen2-VL Instruct 모델을 이용해 이미지 한 장을
    한 줄 영어 문장으로 요약.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to open image {image_path}: {e}", file=sys.stderr)
        return ""

    # Qwen2-VL Instruct는 messages + apply_chat_template 방식이 가장 안전함
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SUMMARY_PROMPT},
                {"type": "image"},  # 🔴 여기서 이미지 토큰 자리를 자동으로 잡아줌
            ],
        }
    ]

    # chat template 적용 → 내부적으로 올바른 이미지 토큰을 삽입
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    raw_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    cleaned = clean_qwen_output(raw_output)
    return cleaned

# ---------------- 경로 처리 ----------------

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


# ---------------- 메인 로직 ----------------

def process_inline_images_to_chunked(input_path: str) -> str:
    """
    sanitized json에서 inline_images 정보만 읽고,
    각 PNG 이미지에 대해 llm_text 생성 후,
    _chunked 폴더에 이미지 요약 전용 JSON을 저장한다.
    (기존 sanitized json은 수정하지 않음)

    - 파일 확장자가 .bin 인 경우: summary 스킵
    - 진행 상황: [INFO] (i/total_png) 형태로 출력
    """
    with open(input_path, "r", encoding="utf-8") as f:
        doc: Dict[str, Any] = json.load(f)

    inline_images: List[Dict[str, Any]] = doc.get("inline_images", [])

    # sanitized json 기준으로 _assets 디렉터리 계산
    json_dir = os.path.dirname(input_path)                # .../_sanitized
    assets_dir = os.path.join(json_dir, "_assets")        # .../_sanitized/_assets

    results: List[Dict[str, Any]] = []

    # --- 1) 먼저 요약 대상 PNG 이미지 목록만 뽑아서 total 계산 ---
    candidates: List[Tuple[Dict[str, Any], str]] = []

    for img in inline_images:
        original_saved_path = img.get("saved_path") or ""
        filename = img.get("filename")

        # 파일명 결정 (filename 우선, 없으면 saved_path에서 basename)
        filename_only = filename or os.path.basename(original_saved_path)
        if not filename_only:
            print(f"[WARN] No valid filename for image rId={img.get('rId')}", file=sys.stderr)
            continue

        ext = os.path.splitext(filename_only)[1].lower()

        # 1) .bin 파일은 요약 스킵
        if ext == ".bin":
            print(f"[INFO] Skipping .bin file for rId={img.get('rId')}: {filename_only}")
            continue

        # 2) PNG 만 요약 대상으로 사용 (.png 외 확장자는 스킵)
        if ext != ".png":
            print(f"[INFO] Skipping non-PNG image for rId={img.get('rId')}: {filename_only}")
            continue

        candidates.append((img, filename_only))

    total_png = len(candidates)
    print(f"[INFO] Total PNG images to summarize: {total_png}")

    # --- 2) 실제 요약 처리 루프 (진행도 출력 포함) ---
    for idx, (img, filename_only) in enumerate(candidates, start=1):
        rId = img.get("rId")
        full_image_path = os.path.join(assets_dir, filename_only)

        if not os.path.exists(full_image_path):
            print(f"[WARN] Image file not found at: {full_image_path}", file=sys.stderr)
            continue

        print(f"[INFO] ({idx}/{total_png}) Summarizing image: rId={rId}, path={full_image_path}")
        summary = summarize_image_with_qwen(full_image_path)

        results.append(
            {
                "rId": rId,
                "filename": filename_only,
                "saved_path": full_image_path,
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
        print("Usage: python summarize_images_with_qwen.py <sanitized_json_path>")
        sys.exit(1)

    input_json = sys.argv[1]
    process_inline_images_to_chunked(input_json)

#python img_summary_gen.py /root/ecm-preprocess-1/output/processed/1장_v3.1/v0/_sanitized/1장_v3.1_sanitized.json
