# config/__init__.py
from .settings import get_settings

settings = get_settings()

try:
    from . import override

    for key in dir(override):
        # 설정 변수(대문자)만 덮어쓰도록 수정
        if key.isupper():
            value = getattr(override, key)
            setattr(settings, key, value)

except ImportError:
    pass