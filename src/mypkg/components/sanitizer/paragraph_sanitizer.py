from collections import Counter
from typing import Iterable, List, Optional, Set
from mypkg.core.parser import ParagraphRecord, RunRecord


_PUNCT_FOLLOW_NO_SPACE = set(".,;:!?)]}%")
_PUNCT_PREV_NO_SPACE = set("([{/'\"")


def _normalize_for_compare(text: Optional[str]) -> str:
    """Normalize text for comparing docx/xml paragraphs by stripping whitespace."""
    if not text:
        return ""
    return "".join(text.split()).lower()


def _concat_text(left: str, right: str) -> str:
    """Run 텍스트를 연결할 때 불필요한 공백 삽입을 피한다."""
    if not left:
        return right
    if not right:
        return left

    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left

    last = left[-1]
    first = right[0]

    join_without_space = False
    if last in _PUNCT_PREV_NO_SPACE:
        join_without_space = True
    elif first in _PUNCT_FOLLOW_NO_SPACE:
        join_without_space = True
    elif last.isdigit() and first.isdigit():
        join_without_space = True
    elif last.isdigit() and first in "%)]":
        join_without_space = True
    elif last in ".-/" and first.isdigit():
        join_without_space = True
    elif last == '-' and first.isalpha():
        join_without_space = True

    if join_without_space:
        return f"{left}{right}"
    return f"{left} {right}"


def _merge_paragraphs(paragraphs_docx: List[ParagraphRecord], paragraphs_xml: List[ParagraphRecord]) -> List[ParagraphRecord]:
    """
    [모듈 내부 함수] xml_parser의 문단 순서를 기준으로 두 문단 목록을 병합합니다.
    docx에서만 등장하는 문단은 제거하되, 공백이 끼어든 xml 텍스트도 대응되도록 비교합니다.
    """
    docx_index = 0
    docx_len = len(paragraphs_docx)
    merged_paragraphs: List[ParagraphRecord] = []

    xml_norm_cache = []
    xml_norm_counts: Counter[str] = Counter()
    for p_xml in paragraphs_xml:
        xml_text_norm = _normalize_for_compare(p_xml.xml_text or p_xml.text or "")
        xml_norm_cache.append(xml_text_norm)
        if xml_text_norm:
            xml_norm_counts[xml_text_norm] += 1

    remaining_xml_counts = xml_norm_counts.copy()

    for p_xml, xml_text_norm in zip(paragraphs_xml, xml_norm_cache):
        if not xml_text_norm:
            merged_paragraphs.append(p_xml)
            continue

        p_docx = None
        while docx_index < docx_len:
            candidate = paragraphs_docx[docx_index]
            candidate_text_norm = _normalize_for_compare(candidate.text)

            if not candidate_text_norm:
                docx_index += 1
                continue

            texts_match = False
            if '[image:' in candidate.text:
                texts_match = True
            elif candidate_text_norm == xml_text_norm:
                texts_match = True
            elif candidate_text_norm and xml_text_norm:
                if candidate_text_norm in xml_text_norm or xml_text_norm in candidate_text_norm:
                    texts_match = True

            if texts_match:
                docx_index += 1
                p_docx = candidate
                break

            # docx 문단이 더 이상 어떤 xml 문단과도 매칭되지 않는다면 폐기한다.
            if remaining_xml_counts.get(candidate_text_norm, 0) == 0:
                docx_index += 1
                continue

            # 이후 xml 문단에서 사용할 수 있으므로 현재 xml과는 결합하지 않는다.
            break

        if p_docx is None:
            merged_paragraphs.append(p_xml)
        else:
            p_docx.doc_index = p_xml.doc_index
            p_docx.numId = p_xml.numId
            p_docx.ilvl = p_xml.ilvl
            p_docx.numFmt = p_xml.numFmt
            p_docx.list_type = p_xml.list_type
            p_docx.math_texts = list(getattr(p_xml, "math_texts", []) or [])
            p_docx.xml_text = getattr(p_xml, "xml_text", p_xml.text)
            merged_paragraphs.append(p_docx)

        if xml_text_norm:
            remaining = remaining_xml_counts.get(xml_text_norm, 0)
            if remaining:
                remaining_xml_counts[xml_text_norm] = remaining - 1

    return merged_paragraphs


def exclude_paragraphs_by_doc_index(paragraphs: List[ParagraphRecord], doc_indices: Iterable[Optional[int]]) -> List[ParagraphRecord]:
    """Return paragraphs excluding those whose doc_index overlaps with any provided doc_index."""
    doc_index_set: Set[int] = set()
    for di in doc_indices:
        if di is None:
            continue
        if isinstance(di, bool):
            continue
        if isinstance(di, int):
            doc_index_set.add(di)
            continue
        try:
            doc_index_set.add(int(di))
        except (TypeError, ValueError):
            continue
    if not doc_index_set:
        return paragraphs
    return [p for p in paragraphs if p.doc_index not in doc_index_set]

def _clean_runs(paragraphs: List[ParagraphRecord]) -> List[ParagraphRecord]:
    """
    [모듈 내부 함수] 문단 내의 Run(텍스트 조각)들을 정리합니다.
    """
    for p in paragraphs:
        original_text = (p.text or "").strip()
        if not p.runs:
            xml_text = getattr(p, "xml_text", None)
            if xml_text and not original_text:
                p.text = xml_text.strip()
            continue

        for r in p.runs:
            t = r.text.replace("\u00A0"," ").replace("\u200B","")
            t = " ".join(t.replace("\r"," ").replace("\t"," ").split())
            r.text = t

        merged_runs: List[RunRecord] = []
        for r in p.runs:
            if not r.text: continue
            
            if (merged_runs and 
                merged_runs[-1].b == r.b and
                merged_runs[-1].i == r.i and
                merged_runs[-1].u == r.u and
                merged_runs[-1].rStyle == r.rStyle and
                merged_runs[-1].sz == r.sz and
                merged_runs[-1].color == r.color):
                merged_runs[-1].text = _concat_text(merged_runs[-1].text, r.text)
                merged_runs[-1].image_rids.extend(r.image_rids)
                if merged_runs[-1].image_rids:
                    merged_runs[-1].image_rids = list(dict.fromkeys(merged_runs[-1].image_rids))
            else:
                merged_runs.append(r)
        
        p.runs = merged_runs
        combined = ""
        for r in p.runs:
            combined = r.text if not combined else _concat_text(combined, r.text)
        combined = combined.strip()
        emphasized_segments = []
        for r in p.runs:
            if getattr(r, "b", False):
                txt = (r.text or "").strip()
                if txt:
                    emphasized_segments.append(txt)
        xml_text = getattr(p, "xml_text", None)

        preferred = original_text or combined
        preferred = preferred.strip()
        xml_candidate = (xml_text or "").strip()

        if xml_candidate:
            if preferred:
                if '[image:' in preferred:
                    final_text = preferred
                elif _normalize_for_compare(preferred) == _normalize_for_compare(xml_candidate):
                    final_text = preferred
                else:
                    final_text = xml_candidate
            else:
                final_text = xml_candidate
        else:
            final_text = preferred

        p.text = final_text
        p.emphasized = emphasized_segments
        
    return paragraphs

class ParagraphSanitizer:
    """
    docx 문서로부터 추출된 다양한 문단 정보들을 병합하고 정제하는 클래스입니다.
    """
    def apply(self, paragraphs_docx: List[ParagraphRecord], paragraphs_xml: List[ParagraphRecord]) -> List[ParagraphRecord]:
        """
        두 종류의 문단 목록을 입력받아 병합하고 정제한 후, 단일화된 문단 목록을 반환합니다.
        """
        merged = _merge_paragraphs(paragraphs_docx, paragraphs_xml)
        cleaned = _clean_runs(merged)
        return cleaned
