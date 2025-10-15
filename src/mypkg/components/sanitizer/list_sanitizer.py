"""List Bullet 스타일 문단을 컴포넌트로 정리하는 sanitizer."""

from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Sequence
from mypkg.core.parser import ParagraphRecord

FormatFunc = Callable[[str, List[str]], str]

def _alpha_label(index: int) -> str:
    base = ord("a")
    label = []
    idx = index
    while True:
        idx, rem = divmod(idx, 26)
        label.append(chr(base + rem))
        if idx == 0:
            break
        idx -= 1
    return "".join(reversed(label))


def default_list_formatter(title: str, items: List[str]) -> str:
    """리스트 텍스트를 기본 포맷으로 정리한다."""

    lines: List[str] = []
    title_clean = title.strip()
    if title_clean:
        lines.append(f"제목: {title_clean}")
    for idx, item in enumerate(items):
        stripped = item.strip()
        if not stripped:
            continue
        label = _alpha_label(idx)
        lines.append(f"{label}. {stripped}")
    return "\n".join(lines)


class ListSanitizer:
    """List Bullet 문단을 묶어 컴포넌트 형태로 반환하는 sanitizer."""

    _LIST_STYLE_CANDIDATES = {
        "List Bullet",
        "List Number",
    }

    def __init__(self, formatter: FormatFunc | None = None) -> None:
        self.formatter: FormatFunc = formatter or default_list_formatter

    def _normalize_style(self, style: str | None) -> str:
        return (style or "").strip()

    def _is_list_style(self, style: str | None) -> bool:
        normalized = self._normalize_style(style)
        return normalized in self._LIST_STYLE_CANDIDATES

    @staticmethod
    def _sort_by_doc_index(paragraphs: Iterable[ParagraphRecord]) -> List[ParagraphRecord]:
        return sorted(paragraphs, key=lambda p: (p.doc_index is None, p.doc_index or 0))

    def _collect_bullet_groups(self, paragraphs: Sequence[ParagraphRecord]) -> List[List[ParagraphRecord]]:
        ordered = self._sort_by_doc_index(paragraphs)
        groups: List[List[ParagraphRecord]] = []
        i = 0
        while i < len(ordered):
            current = ordered[i]
            if not self._is_list_style(current.style):
                i += 1
                continue

            group: List[ParagraphRecord] = []
            last_doc_index = None

            while i < len(ordered):
                candidate = ordered[i]
                if not self._is_list_style(candidate.style):
                    break
                doc_index = candidate.doc_index
                if last_doc_index is not None and doc_index is not None:
                    if doc_index != last_doc_index + 1:
                        break
                group.append(candidate)
                if doc_index is not None:
                    last_doc_index = doc_index
                i += 1

            if group:
                groups.append(group)
            else:
                i += 1

        return groups

    def build_components(self, paragraphs: Sequence[ParagraphRecord]) -> Dict[str, List[Dict[str, object]] | List[int]]:
        """List Bullet 묶음을 컴포넌트 딕셔너리로 변환한다."""

        groups = self._collect_bullet_groups(paragraphs)
        by_doc_index = {p.doc_index: p for p in paragraphs if p.doc_index is not None}

        components: List[Dict[str, object]] = []
        consumed: List[int] = []

        for group in groups:
            first = group[0]
            first_doc_index = first.doc_index
            if first_doc_index is None:
                continue

            if len(group) <= 1:
                continue

            title_text = ""
            if isinstance(first_doc_index, int):
                candidates: List[ParagraphRecord] = []
                for offset in (1, 2):
                    prev = by_doc_index.get(first_doc_index - offset)
                    if not prev:
                        continue
                    if prev.image_included:
                        continue
                    text = (prev.text or "").strip()
                    if not text or text.startswith("[image:"):
                        continue
                    candidates.append(prev)
                subtitle = next(
                    (
                        c
                        for c in candidates
                        if self._normalize_style(c.style) == "소제목2"
                    ),
                    None,
                )
                chosen = subtitle or (candidates[0] if candidates else None)
                if chosen and chosen.text:
                    title_text = chosen.text.strip()

            items = [p.text.strip() for p in group if p.text]
            formatted = self.formatter(title_text, items)

            source_index_set = set()
            emphasized: List[str] = []
            math_texts: List[str] = []
            image_included = False
            for para in group:
                if para.doc_index is not None:
                    source_index_set.add(para.doc_index)
                for idx in getattr(para, "source_doc_indices", []) or []:
                    if isinstance(idx, int):
                        source_index_set.add(idx)
                emphasized.extend(para.emphasized or [])
                math_texts.extend(para.math_texts or [])
                if para.image_included:
                    image_included = True

            source_indices = sorted(source_index_set)

            for para in group:
                if para.doc_index is not None:
                    consumed.append(para.doc_index)

            components.append(
                {
                    "text": formatted,
                    "doc_index": first_doc_index,
                    "style": "List Bullet",
                    "source_doc_indices": source_indices,
                    "emphasized": emphasized,
                    "math_texts": math_texts,
                    "image_included": image_included,
                }
            )

        consumed_sorted = sorted(set(consumed))
        return {"lists": components, "consumed": consumed_sorted}
