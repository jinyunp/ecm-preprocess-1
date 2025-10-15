#!/usr/bin/env python3
"""
오프라인 설치용 wheels 다운로드 스크립트 (기본: linux/amd64)

기본값
- requirements: requirements.txt
- dest: wheels/amd64
- platform: manylinux2014_x86_64 (linux/amd64)
- python-version: 3.11
- abi: cp311
- impl: cp (CPython)

사용 예시
- 기본(amd64):
  python scripts/wheel_save.py
- arm64 아키텍처로 저장:
  python scripts/wheel_save.py --platform manylinux2014_aarch64 --dest wheels/arm64
- 파이썬 버전/ABI 커스터마이즈:
  python scripts/wheel_save.py --python-version 3.11 --abi cp311
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Download manylinux wheels for offline installs")
    ap.add_argument("--requirements", "-r", type=Path, default=root / "requirements.txt", help="요구사항 파일 경로")
    ap.add_argument("--dest", "-d", type=Path, default=root / "wheels" / "amd64", help="다운로드 대상 디렉터리")
    ap.add_argument("--platform", type=str, default="manylinux2014_x86_64", help="대상 플랫폼 태그 (manylinux2014_x86_64 / manylinux2014_aarch64 등)")
    ap.add_argument("--python-version", type=str, default="3.11", help="대상 파이썬 버전 (예: 3.11)")
    ap.add_argument("--abi", type=str, default="cp311", help="ABI 태그 (예: cp311)")
    ap.add_argument("--impl", type=str, default="cp", help="파이썬 구현 (cp/pp 등, 기본 cp)")
    ap.add_argument("--extra", action="append", default=[], help="추가로 다운로드할 패키지명 (여러 번 지정 가능)")
    ap.add_argument("--no-only-binary", action="store_true", help="--only-binary=:all: 비활성화")
    ap.add_argument("--dry-run", action="store_true", help="명령만 출력하고 실행하지 않음")
    return ap.parse_args(argv)


def ensure_pip_available() -> None:
    # sys.executable -m pip 로 호출하므로 별도 which 필요 없음
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, stdout=subprocess.DEVNULL)
    except Exception:
        print("[error] pip 을 찾을 수 없거나 실행할 수 없습니다. Python 환경에 pip를 설치하세요.", file=sys.stderr)
        sys.exit(2)


def build_cmd(args: argparse.Namespace) -> list[str]:
    cmd: list[str] = [
        sys.executable, "-m", "pip", "download",
        "--dest", str(args.dest),
        "--platform", args.platform,
        "--python-version", args.python_version,
        "--abi", args.abi,
        "--implementation", args.impl,
    ]
    if not args.no_only_binary:
        cmd += ["--only-binary", ":all:"]

    # requirements 파일 우선
    if args.requirements and Path(args.requirements).exists():
        cmd += ["-r", str(args.requirements)]

    # 추가 패키지
    for pkg in (args.extra or []):
        cmd.append(pkg)
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_pip_available()

    # 디렉터리 준비
    args.dest.mkdir(parents=True, exist_ok=True)

    cmd = build_cmd(args)
    print("[cmd]", " ".join(cmd))
    if args.dry_run:
        return 0

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[error] pip download 실패 (exit={e.returncode})", file=sys.stderr)
        return e.returncode

    print(f"[ok] wheels saved to: {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
