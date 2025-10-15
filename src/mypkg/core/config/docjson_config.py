"""DocJSON 출력을 제어하기 위한 설정 객체."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from mypkg.core.docjson_types import Section
from .section_config import SectionConfig


@dataclass
class DocJsonConfig:
    """DocJSON 생성 시 공통 옵션 묶음."""

    include_metadata: bool = True
    section: SectionConfig = field(default_factory=SectionConfig)

    def build_sections(self, paragraphs: List[dict]) -> List[Section]:
        return self.section.build(paragraphs)

    def merge(self, overrides: Dict[str, Any] | None) -> "DocJsonConfig":
        if not overrides:
            return self
        include_metadata = overrides.get("include_metadata", self.include_metadata)
        section_overrides = overrides.get("section") if isinstance(overrides, dict) else None
        section_cfg = self.section.merge(section_overrides if isinstance(section_overrides, dict) else None)
        
        return DocJsonConfig(
            include_metadata=include_metadata,
            section=section_cfg,
        )

