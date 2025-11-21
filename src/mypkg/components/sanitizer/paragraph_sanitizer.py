"""
문단 컴포넌트 정제기 (Paragraph Sanitizer)

이 모듈은 두 소스(docx, xml)로부터 추출된 문단 목록을 병합하고 정제하여
일관성 있는 단일 문단 목록을 생성하는 역할을 합니다.

주요 기능:
- **문단 병합**: `python-docx`와 `xml.etree`로 각각 파싱된 두 문단 목록을
  내용 기반으로 비교하여 하나의 신뢰할 수 있는 목록으로 병합합니다.
- **Run 정리**: 문단 내의 텍스트 조각(Run)들을 서식에 따라 병합하고, 불필요한 공백을 제거합니다.
- **일반 문단 병합**: 스타일이 지정되지 않은 연속된 일반 문단들을 하나의 문단으로 합쳐
  의미 있는 본문 블록을 형성합니다.
"
"""
from __future__ import annotations
from collections import Counter
import copy
from typing import Iterable, List, Optional, Set, Dict, Any
from mypkg.core.parser import ParagraphRecord, RunRecord


class ParagraphSanitizer:
    """docx 문서로부터 추출된 문단 정보들을 병합하고 정제하는 클래스입니다."""

    _PUNCT_FOLLOW_NO_SPACE = set(".,;:!?)]}%")
    _PUNCT_PREV_NO_SPACE = set("([{/'\"")
    _SUBTITLE_STYLE_KEYS = {"소제목2", "소제목 2", "subtitle 2"}

    def sanitize(
        self,
        paragraphs_docx: List[ParagraphRecord],
        paragraphs_xml: List[ParagraphRecord],
    ) -> List[ParagraphRecord]:
        """
        두 소스(docx, xml)의 문단 목록을 병합하고 정제하여 단일화된 목록을 반환합니다.

        정제 과정:
        1. docx와 xml에서 추출된 문단 목록을 내용 기반으로 병합합니다. (`_merge_source_paragraphs`)
        2. 문단 내의 텍스트 조각(Run)들을 서식에 따라 정리하고 병합합니다. (`_clean_runs`)
        3. 스타일이 없는 연속된 일반 문단들을 하나의 의미 있는 블록으로 합칩니다. (`_merge_plain_paragraphs`)
        
        Args:
            paragraphs_docx: `python-docx`로 파싱된 문단 목록.
            paragraphs_xml: `xml.etree`로 파싱된 문단 목록.

        Returns:
            정제되고 병합된 최종 문단 레코드 목록.
        """
        merged = self._merge_source_paragraphs(paragraphs_docx, paragraphs_xml)
        cleaned = self._clean_runs(merged)
        final_paragraphs = self._merge_plain_paragraphs(cleaned)
        return final_paragraphs

    def build_components(self, sanitized_paragraphs: List[ParagraphRecord]) -> Dict[str, List[Dict[str, Any]]]:
        """정제된 문단 목록을 JSON 직렬화 가능한 딕셔너리로 변환합니다."""
        payload = []
        for p in sanitized_paragraphs:
            p_dict = {
                "text": p.text,
                "doc_index": p.doc_index,
                "style": p.style,
                "emphasized": p.emphasized,
                "math_texts": p.math_texts,
                "image_included": p.image_included,
                "source_doc_indices": p.source_doc_indices,
            }
            payload.append(p_dict)
        return {"paragraphs": payload}

    # --- Run 및 텍스트 처리 메서드 ---

    def _concat_text(self, left: str, right: str) -> str:
        """두 텍스트를 자연스럽게 연결합니다. 구두점, 숫자 등을 고려하여 공백을 추가할지 결정합니다."""
        if not left: return right
        if not right: return left

        left, right = left.rstrip(), right.lstrip()
        if not left: return right
        if not right: return left

        last, first = left[-1], right[0]

        join_without_space = (
            last in self._PUNCT_PREV_NO_SPACE or
            first in self._PUNCT_FOLLOW_NO_SPACE or
            (last.isdigit() and (first.isdigit() or first in "%)]")) or
            (last in ".-/" and first.isdigit()) or
            (last == '-' and first.isalpha())
        )

        return f"{left}{right}" if join_without_space else f"{left} {right}"

    def _clean_runs(self, paragraphs: List[ParagraphRecord]) -> List[ParagraphRecord]:
        """문단 내의 Run(텍스트 조각)들을 정리하고, 서식이 같은 연속된 Run을 병합합니다."""
        for p in paragraphs:
            if not p.runs: continue

            # 1. 각 Run의 텍스트 정규화 (특수 공백 제거 등)
            for r in p.runs:
                t = r.text.replace("\u00a0"," ").replace("\u200b","")
                r.text = " ".join(t.replace("\r"," ").replace("\t"," ").split())

            # 2. 서식이 동일한 연속 Run 병합
            merged_runs: List[RunRecord] = []
            for r in p.runs:
                if not r.text: continue
                
                can_merge = (
                    merged_runs and 
                    merged_runs[-1].b == r.b and
                    merged_runs[-1].i == r.i and
                    merged_runs[-1].u == r.u and
                    merged_runs[-1].rStyle == r.rStyle and
                    merged_runs[-1].sz == r.sz and
                    merged_runs[-1].color == r.color
                )
                if can_merge:
                    merged_runs[-1].text = self._concat_text(merged_runs[-1].text, r.text)
                    merged_runs[-1].image_rids.extend(r.image_rids)
                    if merged_runs[-1].image_rids:
                        merged_runs[-1].image_rids = list(dict.fromkeys(merged_runs[-1].image_rids))
                else:
                    merged_runs.append(r)
            
            p.runs = merged_runs
            
            # 3. 정리된 Run을 바탕으로 문단 전체 텍스트 및 속성 업데이트
            combined_text = ""
            for run in p.runs:
                combined_text = self._concat_text(combined_text, run.text)
            p.text = combined_text.strip()
            p.emphasized = [r.text.strip() for r in p.runs if getattr(r, "b", False) and r.text.strip()]
            p.image_included = any(r.image_rids for r in p.runs if r.image_rids)
            
            if not p.source_doc_indices and p.doc_index is not None:
                p.source_doc_indices = [p.doc_index]
            
        return paragraphs

    # --- 문단 병합 메서드 ---

    def _normalize_for_compare(self, text: Optional[str]) -> str:
        """비교를 위해 문단 텍스트에서 모든 공백을 제거하고 소문자로 변환합니다."""
        if not text:
            return ""
        return "".join(text.split()).lower()

    def _merge_source_paragraphs(
        self,
        paragraphs_docx: List[ParagraphRecord],
        paragraphs_xml: List[ParagraphRecord],
    ) -> List[ParagraphRecord]:
        """xml_parser의 문단 순서를 기준으로 두 소스의 문단 목록을 병합합니다."""
        docx_index, docx_len = 0, len(paragraphs_docx)
        merged_paragraphs: List[ParagraphRecord] = []

        xml_norm_cache = [self._normalize_for_compare(p.text) for p in paragraphs_xml]
        xml_norm_counts = Counter(norm for norm in xml_norm_cache if norm)
        remaining_xml_counts = xml_norm_counts.copy()

        for p_xml, xml_text_norm in zip(paragraphs_xml, xml_norm_cache):
            if not xml_text_norm:
                merged_paragraphs.append(p_xml)
                continue

            p_docx = None
            while docx_index < docx_len:
                candidate = paragraphs_docx[docx_index]
                candidate_text_norm = self._normalize_for_compare(candidate.text)

                if not candidate_text_norm:
                    docx_index += 1
                    continue

                # 이미지 포함, 또는 텍스트가 같거나 서로 포함하는 경우 매칭으로 간주
                texts_match = (
                    '[image:' in candidate.text or
                    candidate_text_norm == xml_text_norm or
                    (candidate_text_norm and xml_text_norm and 
                     (candidate_text_norm in xml_text_norm or xml_text_norm in candidate_text_norm))
                )

                if texts_match:
                    docx_index += 1
                    p_docx = candidate
                    break

                if remaining_xml_counts.get(candidate_text_norm, 0) == 0:
                    docx_index += 1
                    continue
                break

            if p_docx is None:
                merged_paragraphs.append(p_xml)
            else:
                p_docx.doc_index = p_xml.doc_index
                p_docx.math_texts = list(getattr(p_xml, "math_texts", []) or [])
                if p_xml.doc_index is not None:
                    self._extend_unique_ints(p_docx.source_doc_indices, [p_xml.doc_index])
                merged_paragraphs.append(p_docx)

            if xml_text_norm:
                remaining = remaining_xml_counts.get(xml_text_norm, 0)
                if remaining:
                    remaining_xml_counts[xml_text_norm] = remaining - 1

        return merged_paragraphs

    def _merge_plain_paragraphs(self, paragraphs: List[ParagraphRecord]) -> List[ParagraphRecord]:
        """연속된 기본 문단(style=None)을 하나의 본문 블록으로 합칩니다."""
        merged: List[ParagraphRecord] = []
        buffer: Optional[ParagraphRecord] = None

        for para in paragraphs:
            if not para.source_doc_indices and para.doc_index is not None:
                para.source_doc_indices = [para.doc_index]

            style_norm = self._normalize_style(para.style)
            is_heading_like = style_norm.startswith("heading") or style_norm in self._SUBTITLE_STYLE_KEYS
            text_has_content = bool((para.text or "").strip()) or is_heading_like
            is_plain = para.style is None and text_has_content

            if is_plain:
                if buffer is None:
                    buffer = copy.deepcopy(para)
                else:
                    self._append_paragraph(buffer, para)
                continue

            if buffer is not None:
                merged.append(buffer)
                buffer = None

            merged.append(para)

        if buffer is not None:
            merged.append(buffer)

        return merged

    # --- 병합 헬퍼 메서드 ---

    def _normalize_style(self, style: Optional[str]) -> str:
        """스타일 문자열에서 공백을 제거하고 소문자로 변환합니다."""
        return (style or "").strip().lower()

    def _join_paragraph_text(self, left: Optional[str], right: Optional[str]) -> str:
        """두 문단 텍스트를 줄바꿈으로 합칩니다."""
        left_norm = (left or "").strip()
        right_norm = (right or "").strip()
        if not left_norm: return right_norm
        if not right_norm: return left_norm
        return f"{left_norm}\n{right_norm}"

    def _extend_unique_ints(self, target: List[int], values: Iterable[Optional[int]]) -> None:
        """정수 리스트에 중복되지 않는 새 정수들을 추가합니다."""
        seen = set(target)
        for value in values:
            if value is not None and value not in seen:
                target.append(value)
                seen.add(value)

    def _append_paragraph(self, target: ParagraphRecord, source: ParagraphRecord) -> None:
        """source 문단을 target 문단에 병합(append)합니다."""
        target.text = self._join_paragraph_text(target.text, source.text)
        target.emphasized.extend(seg for seg in source.emphasized if seg)
        target.math_texts.extend(expr for expr in source.math_texts if expr)
        target.runs.extend(copy.deepcopy(source.runs))

        indices = source.source_doc_indices or ([source.doc_index] if source.doc_index is not None else [])
        self._extend_unique_ints(target.source_doc_indices, indices)
        target.image_included = bool(target.image_included or source.image_included)
