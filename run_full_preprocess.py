#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
End-to-end preprocessing pipeline for ECMiner manual:

1) .docx -> sanitized JSON + assets (images)
2) sanitized JSON -> image LLM summaries
3) sanitized JSON -> text chunks
4) chunks + image summaries -> chunks_with_imgsum JSON

사용 예:

  # 단일 파일
  python run_full_preprocess.py /root/ecm-preprocess-1/1장_v3.1.docx
  python run_full_pipeline.py /path/to/manual.docx -o output/processed

  # 폴더(내부 모든 .docx)
  python run_full_pipeline.py /path/to/docx_folder
  python run_full_pipeline.py /path/to/docx_folder -o output/processed
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional

# -------------------- sys.path 설정 --------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# -------------------- 내부 모듈 import --------------------
from mypkg.pipelines.docx_parsing_pipeline import DocxParsingPipeline
from mypkg.pipelines.img_summary_gen import process_inline_images_to_chunked
from mypkg.pipelines import chunking
from mypkg.pipelines.chunking import get_default_output_path
from mypkg.pipelines.merge_image_summaries_into_chunks import merge_image_summaries


# -------------------- 유틸: asyncio 러너 --------------------
def _run_async(coro):
    """노트북/일반 스크립트 환경 모두 대응."""
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        raise


# -------------------- STEP 1: DOCX 파싱 + sanitize --------------------
async def _run_docx_parsing(docx_path: Path, output_root: Path) -> Path:
    """
    DocxParsingPipeline을 실행하고 sanitized JSON 경로를 반환.
    output_root 아래에 _sanitized 디렉터리가 자동으로 만들어짐.
    """
    pipeline = DocxParsingPipeline()
    result = await pipeline.run(docx_path, output_root)

    sanitized = result.get("sanitized_output")
    if not sanitized:
        raise RuntimeError("DocxParsingPipeline.run() 결과에 'sanitized_output' 키가 없습니다.")
    return Path(sanitized)


# -------------------- 파일 단위 파이프라인 --------------------
def run_full_pipeline_for_file(
    docx_path: Path,
    base_output_root: Optional[Path] = None,
) -> Dict[str, str]:
    """
    단일 DOCX 파일에 대해 전체 파이프라인 수행.

    base_output_root:
      - None:  ./output/processed/<doc_name>/v0
      - 지정:  <base_output_root>/<doc_name>/v0
    """
    docx_path = docx_path.expanduser().resolve()

    if base_output_root is None:
        output_root_path = (
            PROJECT_ROOT / "output" / "processed" / docx_path.stem / "v0"
        )
    else:
        # base_output_root 아래에 <doc_name>/v0 구조로 생성
        output_root_path = base_output_root.expanduser().resolve() / docx_path.stem / "v0"

    output_root_path.mkdir(parents=True, exist_ok=True)

    print(f"\n===== Processing file: {docx_path.name} =====")
    print(f"  Output root: {output_root_path}")

    # STEP 1) DOCX -> sanitized
    print(f"[STEP 1] DOCX parsing & sanitizing ...")
    sanitized_path = _run_async(_run_docx_parsing(docx_path, output_root_path))
    print(f"  -> sanitized JSON: {sanitized_path}")

    # STEP 2) 이미지 요약 (Qwen2-VL)
    print(f"[STEP 2] Image semantic summary generation ...")
    image_llm_path = process_inline_images_to_chunked(str(sanitized_path))
    print(f"  -> image LLM summaries: {image_llm_path}")

    # STEP 3) 텍스트 chunking
    print(f"[STEP 3] Chunking sanitized JSON ...")
    # chunking.main은 output_path 인자를 생략하면 get_default_output_path 규칙을 사용
    chunking.main(str(sanitized_path))
    chunked_path = Path(get_default_output_path(str(sanitized_path)))
    print(f"  -> chunked JSON: {chunked_path}")

    # STEP 4) chunk에 이미지 요약 주입
    print(f"[STEP 4] Inject image summaries into chunks ...")
    merged_path = Path(
        merge_image_summaries(str(chunked_path), str(image_llm_path))
    )
    print(f"  -> chunked_with_imgsum JSON: {merged_path}")

    return {
        "sanitized_json": str(sanitized_path),
        "image_llm_json": str(image_llm_path),
        "chunked_json": str(chunked_path),
        "chunked_with_imgsum_json": str(merged_path),
    }


# -------------------- 폴더 단위 파이프라인 --------------------
def run_full_pipeline_for_dir(
    input_dir: Path,
    base_output_root: Optional[Path] = None,
    pattern: str = "*.docx",
) -> List[Dict[str, str]]:
    """
    디렉터리 내부의 모든 DOCX 파일에 대해 파이프라인 수행.

    base_output_root:
      - None: 각 파일별로 ./output/processed/<doc_name>/v0
      - 지정: 각 파일별로 <base_output_root>/<doc_name>/v0
    pattern:
      - 기본: "*.docx"
    """
    input_dir = input_dir.expanduser().resolve()
    if base_output_root is not None:
        base_output_root = base_output_root.expanduser().resolve()

    docx_files = sorted(input_dir.glob(pattern))

    if not docx_files:
        print(f"[WARN] 디렉터리({input_dir})에 '{pattern}' 패턴에 맞는 파일이 없습니다.")
        return []

    print(f"총 {len(docx_files)}개 DOCX 파일을 처리합니다.")
    results: List[Dict[str, str]] = []

    for idx, docx_file in enumerate(docx_files, start=1):
        print(f"\n==== [{idx}/{len(docx_files)}] {docx_file.name} ====")
        try:
            res = run_full_pipeline_for_file(docx_file, base_output_root)
            results.append(res)
        except Exception as e:
            # 개별 파일 실패 시 로그만 찍고 계속 진행
            print(f"[ERROR] 파일 처리 중 오류 발생: {docx_file} -> {e}")

    return results


# -------------------- CLI --------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full ECMiner manual preprocessing pipeline "
            "(docx -> sanitized -> image summaries -> chunks -> chunks_with_imgsum).\n"
            "입력 경로가 파일이면 해당 파일만, 디렉터리면 내부의 모든 .docx 파일을 처리합니다."
        )
    )
    parser.add_argument(
        "input_path",
        help="입력 경로 (DOCX 파일 또는 DOCX 파일들이 들어있는 디렉터리)",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        help=(
            "출력 루트 디렉터리.\n"
            "  - 단일 파일: <output_root>/<doc_name>/v0\n"
            "  - 디렉터리:   <output_root>/<doc_name>/v0 (파일별 생성)\n"
            "지정하지 않으면 ./output/processed/<doc_name>/v0 구조 사용."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.docx",
        help="디렉터리 모드에서 사용할 파일 패턴 (기본: '*.docx')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)

    base_output_root: Optional[Path]
    if args.output_root is None:
        base_output_root = None
    else:
        base_output_root = Path(args.output_root)

    if input_path.is_file():
        # 파일 모드
        outputs = run_full_pipeline_for_file(input_path, base_output_root)

        print("\n[RESULT PATHS - single file]")
        for k, v in outputs.items():
            print(f"{k}: {v}")

    elif input_path.is_dir():
        # 디렉터리 모드
        results = run_full_pipeline_for_dir(
            input_dir=input_path,
            base_output_root=base_output_root,
            pattern=args.pattern,
        )

        print("\n[RESULT PATHS - directory]")
        for i, res in enumerate(results, start=1):
            print(f"\n--- File #{i} ---")
            for k, v in res.items():
                print(f"{k}: {v}")
    else:
        raise SystemExit(f"[ERROR] input_path가 존재하지 않습니다: {input_path}")


if __name__ == "__main__":
    main()
