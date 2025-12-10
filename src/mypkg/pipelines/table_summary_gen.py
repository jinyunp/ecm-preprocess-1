# src/mypkg/pipelines/table_summary_gen.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ------------------------------------------------------------
# 1) Sanitized JSON에서 테이블 HTML 뽑기
# ------------------------------------------------------------
def extract_tables_from_sanitized(sanitized_path: str) -> List[Tuple[str, str]]:
    """
    sanitized JSON에서 (table_id, table_text_for_llm) 리스트를 추출한다.

    sanitized 구조 예:
    {
      "paragraphs": [...],
      "tables": [
        {
          "tid": "t25",
          "doc_index": 25,
          "preceding_text": "...",
          "rows": 5,
          "cols": 4,
          "data": [...],
          "table_html": "<table>...</table>",
          ...
        },
        ...
      ],
      "images": [...]
    }

    반환: [(tid, 테이블 요약용 텍스트), ...]
    """

    from pathlib import Path
    import json

    with open(sanitized_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tables: List[Tuple[str, str]] = []

    # top-level이 dict이고 "tables" 키를 가지고 있는 구조
    if isinstance(data, dict) and "tables" in data:
        for tbl in data.get("tables", []):
            if not isinstance(tbl, dict):
                continue

            # 테이블 ID: tid 필드를 우선 사용
            table_id = tbl.get("tid") or tbl.get("table_id") or tbl.get("id")
            if not table_id:
                continue

            # table_html 우선 사용, 없으면 data/anchors 기반으로 fallback 텍스트 생성
            table_html = tbl.get("table_html") or tbl.get("html")

            # preceding_text도 요약에 도움이 되므로 LLM 입력에 함께 포함
            preceding = tbl.get("preceding_text") or ""

            if table_html:
                # LLM에 넘길 때 약간의 컨텍스트를 붙여줌
                llm_input_text = (
                    "다음은 ECMiner 매뉴얼의 한 표입니다.\n"
                    "표 앞에 나오는 설명(preceding_text)과 HTML 구조를 함께 제공합니다.\n\n"
                )
                if preceding:
                    llm_input_text += f"[PRECEDING_TEXT]\n{preceding}\n\n"
                llm_input_text += f"[TABLE_HTML]\n{table_html}"
            else:
                # table_html이 없을 때는 data 셀 텍스트들을 단순 텍스트로 직렬화
                rows = tbl.get("data") or []
                lines: List[str] = []
                for r in rows:
                    if not isinstance(r, list):
                        continue
                    cell_texts = []
                    for cell in r:
                        if isinstance(cell, dict):
                            cell_texts.append(str(cell.get("text") or "").strip())
                        else:
                            cell_texts.append(str(cell))
                    lines.append(" | ".join(cell_texts))

                table_text = "\n".join(lines).strip()

                llm_input_text = "다음은 ECMiner 매뉴얼의 한 표의 셀 텍스트를 행 단위로 나열한 것입니다.\n"
                if preceding:
                    llm_input_text += f"[PRECEDING_TEXT]\n{preceding}\n\n"
                llm_input_text += f"[TABLE_TEXT]\n{table_text}"

            tables.append((str(table_id), llm_input_text))

    else:
        print(f"[WARN][table_summary_gen] sanitized 구조에 'tables' 키가 없습니다: {sanitized_path}")

    print(f"[INFO][table_summary_gen] Extracted {len(tables)} tables from {sanitized_path}")
    return tables


# ------------------------------------------------------------
# 2) Qwen 텍스트 LLM 로딩
# ------------------------------------------------------------
def load_qwen_text_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Qwen 텍스트 모델 로드.
    - QWEN_TEXT_MODEL_PATH 환경변수가 있으면 그걸 쓰고,
    - 없으면 기본값으로 'Qwen/Qwen2.5-7B-Instruct' 사용.
    """
    model_name = os.environ.get("QWEN_TEXT_MODEL_PATH", "/workspace/qwen/VL") #"Qwen/Qwen2.5-7B-Instruct")
    print(f"[INFO][table_summary_gen] Loading text model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model


# ------------------------------------------------------------
# 3) 테이블 HTML을 요약 텍스트로 변환
# ------------------------------------------------------------
def summarize_table_html(
    table_html: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int = 256,
) -> str:
    """
    테이블 HTML을 LLM에 넣어 한국어 요약 생성.
    프롬프트는 필요에 맞게 수정 가능.
    """
    prompt = (
        "The following is an HTML table extracted from the ECMiner manual.\n"
        "Please summarize its key meaning in clear and concise English.\n"
        "If possible, also describe the roles of its rows and columns.\n\n"
        f"[TABLE HTML]\n{table_html}\n\n"
        "Summary:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # "요약:" 뒤 부분만 잘라내기 (프롬프트 그대로 반환되는 경우 대응)
    idx = text.rfind("요약:")
    if idx != -1:
        summary = text[idx + len("요약:"):].strip()
    else:
        summary = text.strip()

    return summary


# ------------------------------------------------------------
# 4) 전체 파이프라인: sanitized → table_llm.json
# ------------------------------------------------------------
def process_tables_to_chunked(sanitized_path: str) -> str:
    """
    sanitized JSON 파일에서 테이블들을 뽑아 LLM 요약을 생성하고,
    {docname}_table_llm.json 파일을 생성한다.

    반환값은 생성된 JSON 경로.
    """
    sanitized_path = str(Path(sanitized_path).resolve())
    tables = extract_tables_from_sanitized(sanitized_path)

    if not tables:
        print("[INFO][table_summary_gen] No tables found. Skipping.")
        # 그래도 빈 리스트 파일은 만들어 둔다.
        out_path = _get_default_output_path(sanitized_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return out_path

    tokenizer, model = load_qwen_text_model()

    results: List[Dict[str, Any]] = []
    for idx, (tbl_id, html) in enumerate(tables, start=1):
        print(f"[INFO][table_summary_gen] Summarizing table {idx}/{len(tables)} (id={tbl_id})")
        summary = summarize_table_html(html, tokenizer, model)
        results.append({
            "table_id": tbl_id,
            "llm_text": summary,
        })

    out_path = _get_default_output_path(sanitized_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[INFO][table_summary_gen] Saved table LLM summaries to {out_path}")
    return out_path


def _get_default_output_path(sanitized_path: str) -> str:
    """
    sanitized 경로에서 docname을 추출하여
    /root/ecm-preprocess-1/output/processed/{docname}/v0/_chunked/{docname}_table_llm.json
    형태의 경로를 반환.
    """
    # sanitized_path 예: /root/ecm-preprocess-1/output/processed/1장_v3.1/v0/_sanitized/1장_v3.1_sanitized.json
    p = Path(sanitized_path)
    docname = p.stem.replace("_sanitized", "")  # 1장_v3.1_sanitized -> 1장_v3.1
    # processed/<docname>/v0/_chunked 아래에 저장
    base = Path("/root/ecm-preprocess-1/output/processed") / docname / "v0" / "_chunked"
    return str(base / f"{docname}_table_llm.json")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate table summaries from sanitized JSON using Qwen text model."
    )
    parser.add_argument(
        "sanitized_json",
        help="sanitized JSON 경로 (/root/ecm-preprocess-1/output/processed/.../_sanitized/..._sanitized.json)",
    )
    args = parser.parse_args()

    process_tables_to_chunked(args.sanitized_json)
