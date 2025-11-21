#!/usr/bin/env python3
"""
Pick a handful of random images and run ImageSanitizer's summary prompt.

Usage:
    python scripts/sample_image_summaries.py --images-dir ../sut-test-images/data/images --count 10

The script expects an Ollama-compatible endpoint at http://localhost:11434/api/generate,
because it reuses ImageSanitizer._get_image_summary().
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from mypkg.components.sanitizer.image_sanitizer import ImageSanitizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample N images and print ImageSanitizer summaries."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing images (recursively)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="How many images to sample (default: 10).",
    )
    parser.add_argument(
        "--suffixes",
        nargs="+",
        default=[".png", ".jpg", ".jpeg"],
        help="List of file extensions to include (case-insensitive).",
    )
    return parser.parse_args()


def collect_images(root: Path, suffixes: list[str]) -> list[Path]:
    normalized = {s.lower() for s in suffixes}
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in normalized
    )


def main() -> None:
    args = parse_args()
    images = collect_images(args.images_dir, args.suffixes)
    if not images:
        raise SystemExit(f"No images found under {args.images_dir}")

    sample_size = min(args.count, len(images))
    picked = random.sample(images, sample_size)

    sanitizer = ImageSanitizer()

    print(f"[info] Sampled {sample_size} image(s) out of {len(images)} available.\n")
    for idx, image_path in enumerate(picked, 1):
        summary_raw = sanitizer._get_image_summary(image_path)  # noqa: SLF001 (test helper)
        summary = sanitizer._normalize_summary(summary_raw)
        print(f"[{idx}] {image_path}")
        if summary:
            print(f"    summary: image description: {summary}")
        else:
            print("    summary: <no response>")
        print()


if __name__ == "__main__":
    main()
