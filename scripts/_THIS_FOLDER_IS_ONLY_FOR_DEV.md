개요

- 이 폴더는 개발 편의용 스크립트를 보관합니다.
- 두 가지 스크립트를 제공합니다:
  - `src_make.py`: 소스 번들(zip) 생성
  - `wheel_save.py`: 오프라인 설치용 wheels 다운로드(기본: linux/amd64)

src_make.py 사용법

  - 기본: `uv run python scripts/src_make.py`
      - 생성 위치: `scripts/dist/src_bundle_YYYYmmdd_HHMMSS.zip`
  - 출력 파일 지정: `uv run python scripts/src_make.py -o /path/to/my_bundle.zip`

  포함 규칙

  - `src/` 디렉터리 전체
  - `data_store/` 디렉터리에서 `processed/`만 제외하고 나머지 전부
  - 저장소 루트의 모든 파일(숨김 파일 포함: `.dockerignore`, `.gitignore`, `Dockerfile*`, `README.md`, 등)

wheel_save.py 사용법(amd64 wheels 다운로드)

  - 기본(amd64, Python 3.11/cp311):
    - `python scripts/wheel_save.py`
    - 결과 저장 위치: `wheels/amd64/`
  - arm64로 저장하고 싶을 때:
    - `python scripts/wheel_save.py --platform manylinux2014_aarch64 --dest wheels/arm64`
  - 옵션 요약:
    - `--requirements`: 요구사항 파일(기본: `requirements.txt`)
    - `--dest`: 저장 디렉터리(기본: `wheels/amd64`)
    - `--platform`: 대상 플랫폼 태그(기본: `manylinux2014_x86_64`)
    - `--python-version`: 대상 파이썬 버전(기본: `3.11`)
    - `--abi`: ABI 태그(기본: `cp311`)
    - `--impl`: 구현(cp/pp 등, 기본: `cp`)
    - `--extra`: 추가 패키지 지정 (여러 번 지정 가능)
    - `--no-only-binary`: 소스 배포 허용 (기본은 wheels만)
    - `--dry-run`: 다운로드 명령만 출력

설명

- `wheel_save.py`는 온라인 환경에서 PyPI로부터 사전 빌드된 manylinux wheels만 내려받아 `wheels/<arch>/`에 저장합니다.
- 프로젝트 자체는 소스 설치(`-e .`)로 처리되므로 별도의 프로젝트 wheel은 만들지 않습니다.
- `Dockerfile.wsl2`/`Dockerfile.mac`은 `/opt/wheels`에서만 설치하므로, 해당 디렉터리에 필요한 wheel들이 모두 있어야 오프라인 설치가 성공합니다.
