#!/usr/bin/env python3
"""
프로젝트 배포용 소스 번들(zip) 생성 스크립트

포함 규칙
- src/ 디렉터리 전체
- data_store/ 디렉터리 중 processed/ 를 제외한 나머지 전부
- 저장소 루트에 있는 모든 파일(감춰진 파일 포함: .dockerignore, .gitignore, 등)

기본 출력 경로
- scripts/dist/src_bundle_YYYYmmdd_HHMMSS.zip (디렉터리가 없으면 생성)

사용 예
- uv run python scripts/src_make.py
- uv run python scripts/src_make.py -o /path/to/my_bundle.zip
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import sys
import zipfile


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def add_file(zf: zipfile.ZipFile, root: Path, path: Path) -> None:
    arcname = path.resolve().relative_to(root.resolve()).as_posix()
    zf.write(path, arcname)


def add_dir(zf: zipfile.ZipFile, root: Path, dir_path: Path, exclude_dirs: list[Path] | None = None) -> int:
    count = 0
    exclude_dirs = [p.resolve() for p in (exclude_dirs or [])]
    dir_abs = dir_path.resolve()
    if not dir_abs.exists():
        return 0

    for current_dir, dirnames, filenames in os.walk(dir_abs):
        cur_path = Path(current_dir).resolve()

        # 하위 탐색에서 제외할 디렉터리들을 미리 가지치기
        pruned = []
        for d in dirnames:
            cand = (cur_path / d).resolve()
            if any(_is_relative_to(cand, ex) for ex in exclude_dirs):
                continue
            pruned.append(d)
        dirnames[:] = pruned

        # 현재 경로 자체가 제외 대상이면 스킵
        if any(_is_relative_to(cur_path, ex) for ex in exclude_dirs):
            continue

        for fname in filenames:
            fpath = (cur_path / fname)
            add_file(zf, root, fpath)
            count += 1
    return count


def build_zip(output: Path) -> Path:
    script_path = Path(__file__).resolve()
    root = script_path.parent.parent.resolve()  # 프로젝트 루트(스크립트가 scripts/ 하위에 있다고 가정)

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    data_store = root / "data_store"
    data_store_excludes = [(data_store / "processed").resolve()]
    src_dir = root / "src"

    included_files = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1) 루트의 모든 파일(감춰진 파일 포함)
        for p in root.iterdir():
            if p.is_file():
                # 출력 zip 자신은 포함하지 않음
                if p.resolve() == output:
                    continue
                add_file(zf, root, p)
                included_files += 1

        # 2) src/ 전체
        included_files += add_dir(zf, root, src_dir)

        # 3) data_store/ (processed/ 제외)
        if data_store.exists():
            included_files += add_dir(zf, root, data_store, exclude_dirs=data_store_excludes)

    print(f"[ok] wrote: {output} (files: {included_files})")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="src + data_store(except processed) + root files → zip 번들 생성")
    default_name = f"src_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    default_out = Path(__file__).resolve().parent.parent / "scripts" / "dist" / default_name
    ap.add_argument("-o", "--output", type=Path, default=default_out, help="출력 zip 경로 (기본: dist/src_bundle_YYYYmmdd_HHMMSS.zip)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_zip(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

