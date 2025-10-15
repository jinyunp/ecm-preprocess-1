"""문서 메타데이터 분석기."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from mypkg.core.docjson_types import DocumentMetadata


@dataclass
class ParsedName:
    chapter: Optional[str]
    version: Optional[str]


def _parse_name(name: str) -> ParsedName:
    parts = name.split("_", 1)
    if len(parts) == 2:
        chapter, version = parts
    else:
        chapter, version = name, None
    chapter = chapter.strip() or None
    version = version.strip() if isinstance(version, str) else version
    version = version or None
    return ParsedName(chapter=chapter, version=version)


class DocumentMetadataAnalyzer:
    """sanitized JSON과 파일 이름/시간 정보를 기반으로 간단한 메타데이터를 구성한다."""

    CHAPTER_PATTERN = re.compile(r"^Chapter\s+(?P<chapter>\S+)\s+(?P<title>.+)$", re.IGNORECASE)

    def __init__(
        self,
        docjson: Dict[str, Any],
        doc_name: str,
        timestamp_path: Optional[Path] = None,
    ) -> None:
        self.docjson = docjson
        self.doc_name = doc_name
        self.timestamp_path = Path(timestamp_path) if timestamp_path else None
        self.parsed_name = _parse_name(self.doc_name)

    def _extract_chapter_title(self) -> tuple[Optional[str], Optional[str]]:
        paragraphs = self.docjson.get("paragraphs", [])
        for para in paragraphs:
            if not isinstance(para, dict):
                continue
            text = (para.get("text") or "").strip()
            if not text:
                continue
            match = self.CHAPTER_PATTERN.match(text)
            if match:
                chapter = match.group("chapter").strip() or None
                title = match.group("title").strip() or None
                return chapter, title
        return None, None

    def _last_modified(self) -> Optional[str]:
        if not self.timestamp_path or not self.timestamp_path.exists():
            return None
        return datetime.fromtimestamp(self.timestamp_path.stat().st_mtime).date().isoformat()

    def analyze(self) -> DocumentMetadata:
        chapter_from_name = self.parsed_name.chapter
        version = self.parsed_name.version

        chapter_from_content, title = self._extract_chapter_title()

        chapter = chapter_from_name or chapter_from_content

        return DocumentMetadata(
            chapter=chapter,
            version=version,
            title=title,
            last_modified=self._last_modified(),
        )
