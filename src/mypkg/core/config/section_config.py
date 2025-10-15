"""섹션 분석을 위한 설정 객체.

테넌트마다 다른 섹션 구분 규칙을 적용할 수 있도록 설정 기반으로 래핑한다.
기본값은 숫자 헤딩 기반 분석이며, `heading_styles`를 지정하면 스타일 목록에
맞춰 섹션을 나눈다. 더 특수한 요구 사항은 `builder` 콜백으로 커스터마이징한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Dict, Any

from mypkg.core.docjson_types import Section
from mypkg.components.analyzer.section_analyzer import build_sections


BuilderFn = Callable[[List[dict]], List[Section]]


DEFAULT_HEADING_STYLES: Sequence[str] = (
    "Heading 1",
    "Heading 2",
    "Heading 3",
    "Heading 4",
    "Heading 5",
    "Heading 6",
    "Heading 7",
    "Heading 8",
    "Heading 9",
)

_MISSING = object()


def _normalize_styles(styles: Optional[Sequence[str]]) -> tuple[str, ...]:
    if styles is None:
        return tuple()
    if isinstance(styles, str):
        return (styles,)
    return tuple(styles)


@dataclass
class SectionConfig:
    """섹션 분석 옵션.

    - heading_styles: python-docx가 부여한 스타일 이름을 기준으로 섹션을 나누고 싶을 때 사용.
    - builder: 완전히 커스터마이징된 섹션 생성 로직이 필요할 때 주입.
    두 옵션이 모두 주어지면 builder가 우선한다.
    """

    heading_styles: Optional[Sequence[str]] = field(default_factory=lambda: DEFAULT_HEADING_STYLES)
    builder: Optional[BuilderFn] = None
    strict_heading_styles: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.strict_heading_styles is None:
            if self.heading_styles is None:
                self.strict_heading_styles = False
            else:
                normalized = _normalize_styles(self.heading_styles)
                self.strict_heading_styles = normalized != _normalize_styles(DEFAULT_HEADING_STYLES)

    def build(self, paragraphs: List[dict]) -> List[Section]:
        if self.builder:
            return self.builder(paragraphs)
        return build_sections(
            paragraphs,
            heading_styles=self.heading_styles,
            enforce_style_whitelist=bool(self.strict_heading_styles),
        )

    def merge(self, overrides: Optional[Dict[str, Any]]) -> "SectionConfig":
        if not overrides:
            return self
        heading_styles = overrides.get("heading_styles", self.heading_styles)
        strict_override = overrides.get("strict_heading_styles", _MISSING)

        strict_heading_styles: Optional[bool]
        if strict_override is not _MISSING:
            if strict_override is None:
                strict_heading_styles = None
            else:
                strict_heading_styles = bool(strict_override)
        else:
            tuple_current = _normalize_styles(self.heading_styles)
            tuple_new = _normalize_styles(heading_styles)
            heading_styles_changed = tuple_current != tuple_new
            if heading_styles_changed:
                strict_heading_styles = None
            else:
                strict_heading_styles = self.strict_heading_styles

        # builder는 코드에서만 주입 가능하도록 JSON 오버라이드는 무시한다.
        return SectionConfig(
            heading_styles=heading_styles,
            builder=self.builder,
            strict_heading_styles=strict_heading_styles,
        )


DEFAULT_SECTION_CONFIG = SectionConfig()
