"""
리스트/테이블 블록 분석기

개요
- 문단(paragraphs)과 테이블(tables) 원본/정제 데이터를 입력 받아
  · 리스트 묶음을 ContentBlock(dict)으로 추출
  · 표를 ContentBlock(dict)으로 래핑
  · 필요 시 `_comp/`(list/table) JSON으로 저장

리스트 판정 전략(우선순위)
1) 파서가 이미 표시한 list_type이 있으면 그대로 신뢰
2) Word numbering(numPr)의 numId가 있으면 리스트로 간주
3) 스타일 기반: 스타일 이름 토큰에 list 관련 토큰(listparagraph 등)이 있으면 리스트로 추정
4) 텍스트 마커 기반: "1.", "(a)", "•" 등 전형적 마커 패턴 매칭 시 리스트로 추정

리스트 묶음 규칙
- numId 기반인 경우: 같은 numId이고 같은 레벨(ilvl)인 연속 문단을 하나의 리스트로 묶음
- 스타일 기반인 경우: numId가 없고, 레벨이 같고, 스타일 토큰 교집합이 있으며
  ordering(ordered/bullet) 분류가 일치하는 연속 문단을 묶음

출력 형태
- 리스트 블록: { id, type="list", doc_index(시작), level, list_data:{ordered, scheme, level, items[]} }
- 테이블 블록: { id, type="table", doc_index, table:{rows, cols, data} }

참고 문서
- docs/LIST_AND_TABLE.md: 탐지 로직/필드 설명서
"""

# mypkg/components/analyzer/list_table_analyzer.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from pathlib import Path
from mypkg.core.io import save_list_components_from_sanitized, save_table_components_from_sanitized
# --- add near the top
import re
from typing import Any, Dict, List, Tuple

ORDERED_NUMFMTS = {
    "decimal","decimalzero","decimalleadingzero","decimalfullwidth","decimalhalfwidth",
    "decimalenclosedcircle","decimalenclosedfullstop","decimalenclosedparen","decimalenclosedcirclechinese",
    "lowerroman","upperroman",
    "lowerletter","upperletter","alpha","alphabetic",
    "arabicalpha","arabicabjad",
    "hebrew1","hebrew2",
    "chinesecounting","chinesecountingthousand","chinesecountingtenthousand",
    "ideographtraditional","ideographzodiac","ideographdigital","ideographenclosedcircle",
    "japanesecounting","japanesedigitaltenthousand",
    "aiueo","aiueofullwidth","iroha","irohafullwidth",
    "thainumbers","thaicounting","hindinumbers",
    "ganada","ganadakr",
}
UNORDERED_NUMFMTS = {"bullet","none","dingbat","picture","disc","circle","square"}

# 다양한 스타일 키에서 후보를 뽑고, 토큰화(공백/하이픈/언더스코어 제거, 소문자화)
STYLE_KEYS = ("style", "style_name", "styleId", "style_id", "para_style", "pStyle")
STYLE_LIST_TOKENS = {
    "listparagraph",  # Word 기본
    "listbullet", "listnumber", "listcontinue",
    # 팀 템플릿 추가 가능: "목록단락", "불릿", "번호매기기" 등을 토큰화 형태로 넣기
}

def _style_tokens_from_para(p: Dict[str, Any]) -> List[str]:
    """여러 스타일 키를 모아 공백/하이픈/언더스코어 제거 후 소문자 토큰으로 표준화"""
    toks: List[str] = []
    for k in STYLE_KEYS:
        v = p.get(k)
        if not v:
            continue
        t = str(v).strip().lower().replace("_","").replace("-","").replace(" ","")
        if t:
            toks.append(t)
    return toks

def _looks_like_marker(text: str) -> bool:
    """텍스트가 리스트 마커(숫자/로마/알파+.,) 또는 글머리 기호처럼 보이는지 간단 판정"""
    if not text:
        return False
    t = text.strip()
    # 숫자/로마/알파 + . or )  , 또는 대표 bullet 문자
    if re.match(r"^(\(?\d+|\(?[ivxlcdm]+|\(?[a-zA-Z])[\.\)]\s+", t):
        return True
    if t[:1] in {"•","·","▪","◦","–","—","-","*"}:
        return True
    return False

def _collapse_spaces(s: str) -> str:
    return re.sub(r"\s{2,}", " ", s).strip()

def _is_list_para(p: Dict[str, Any]) -> bool:
    """리스트 후보 문단인지 종합 판정(파서 힌트 → numbering → 스타일 → 텍스트 마커)"""
    # 1) 파서가 이미 리스트로 표시했으면 그대로 신뢰
    if p.get("list_type"):
        return True
    # 2) Word numbering(numPr)이 있으면 리스트
    if p.get("numId") is not None:
        return True
    # 3) 스타일 기반 탐지 (여러 키, 다양한 철자/공백 변형 허용)
    toks = _style_tokens_from_para(p)
    if any(t in STYLE_LIST_TOKENS for t in toks):
        return True
    # 4) 최후의 폴백: 텍스트 마커 패턴으로 추정
    return _looks_like_marker((p.get("text") or ""))

def _classify_ordering(p: Dict[str, Any]) -> Tuple[bool, str]:
    """넘버링/스타일/텍스트 힌트로 ordered 여부와 scheme 분류"""
    # 기존 매핑 로직은 유지하되, 결과를 scheme으로도 보존
    def _norm(s: Any) -> str:
        return (str(s).strip().lower().replace("_","").replace("-","").replace(" ","")
                if s is not None else "")
    numfmt = _norm(p.get("numFmt"))
    if numfmt:
        if numfmt in {"bullet","none","dingbat","picture","disc","circle","square"}:
            return False, "bullet"
        if numfmt in ORDERED_NUMFMTS:
            if "roman" in numfmt: return True, "roman"
            if "letter" in numfmt or "alpha" in numfmt: return True, "alpha"
            if "ganada" in numfmt: return True, "ganada"
            if "arabic" in numfmt: return True, "arabicAlpha"
            if "hebrew" in numfmt: return True, "hebrew"
            if "thai" in numfmt: return True, "thai"
            if any(k in numfmt for k in ["chinese","ideograph"]): return True, "cjk"
            if any(k in numfmt for k in ["aiueo","iroha","japanese"]): return True, "kana"
            return True, "decimal"
    lt = _norm(p.get("list_type"))
    if lt in {"bullet","unordered"}:
        return False, "bullet"
    if lt in {"number","decimal","roman","alpha","ganada"}:
        if lt=="roman": return True, "roman"
        if lt=="alpha": return True, "alpha"
        if lt=="ganada": return True, "ganada"
        return True, "decimal"
    # 텍스트 마커 기반 힌트
    return (True, "decimal") if _looks_like_marker(p.get("text") or "") else (True, "decimal")

def analyze_lists(paragraphs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], set]:
    """연속 문단들을 규칙에 따라 묶어 리스트 블록 배열과 소비된 doc_index 집합을 반환"""
    paras = sorted(paragraphs, key=lambda x: x.get("doc_index", 0))
    list_blocks: List[Dict[str, Any]] = []
    consumed = set()
    i = 0
    while i < len(paras):
        p = paras[i]
        if not _is_list_para(p):
            i += 1; continue

        base_numId = p.get("numId")
        base_lvl   = _list_level(p)
        ordered, scheme = _classify_ordering(p)
        base_styles = set(_style_tokens_from_para(p))

        start = i
        items = []

        while i < len(paras):
            q = paras[i]
            if not _is_list_para(q):
                break

            same = True
            if base_numId is not None:
                # numId 기반 리스트는 numId와 레벨이 같아야 같은 묶음
                if q.get("numId") != base_numId or _list_level(q) != base_lvl:
                    same = False
            else:
                # 스타일 기반: numId 없어야 하고, 레벨 같고, 스타일 토큰이 교집합이어야 함
                if q.get("numId") is not None:
                    same = False
                elif _list_level(q) != base_lvl:
                    same = False
                else:
                    qstyles = set(_style_tokens_from_para(q))
                    if not (base_styles and qstyles and (base_styles & qstyles)):
                        same = False
                    else:
                        qo, qs = _classify_ordering(q)
                        if qo != ordered:
                            same = False

            if not same:
                break

            items.append({"doc_index": q["doc_index"], "text": _collapse_spaces(q.get("text") or "")})
            consumed.add(q["doc_index"])
            i += 1

        if items:
            list_blocks.append({
                "id": f"list_{items[0]['doc_index']}",
                "type": "list",
                "doc_index": items[0]["doc_index"],
                "text": None,
                "level": base_lvl,
                "page": None, "semantic": None,
                "table": None,
                "list_data": {
                    "ordered": ordered,
                    "scheme": scheme,   # ← 디버깅에 유용
                    "level": base_lvl,
                    "items": [it["text"] for it in items],
                },
            })
        if start == i:
            i += 1
    return list_blocks, consumed

def _norm(s: Any) -> str:
    return (str(s).strip().lower().replace("_","").replace("-","").replace(" ","") if s is not None else "")

def _classify_ordering(p: Dict[str, Any]) -> Tuple[bool, str]:
    numfmt = _norm(p.get("numFmt"))
    if numfmt:
        if numfmt in UNORDERED_NUMFMTS: return (False,"bullet")
        if numfmt in ORDERED_NUMFMTS:
            if "roman" in numfmt: return (True,"roman")
            if "letter" in numfmt or "alpha" in numfmt: return (True,"alpha")
            if "ganada" in numfmt: return (True,"ganada")
            if "arabic" in numfmt: return (True,"arabicAlpha")
            if "hebrew" in numfmt: return (True,"hebrew")
            if "thai" in numfmt: return (True,"thai")
            if any(k in numfmt for k in ["chinese","ideograph"]): return (True,"cjk")
            if any(k in numfmt for k in ["aiueo","iroha","japanese"]): return (True,"kana")
            return (True,"decimal")
    lt = _norm(p.get("list_type"))
    if lt in {"bullet","unordered"}: return (False,"bullet")
    if lt in {"number","decimal","roman","alpha","ganada"}:
        if lt=="roman": return (True,"roman")
        if lt=="alpha": return (True,"alpha")
        if lt=="ganada": return (True,"ganada")
        return (True,"decimal")
    return (True,"decimal")

def _is_list_para(p: Dict[str, Any]) -> bool:
    if p.get("list_type"):
        return True
    if p.get("numId") is not None:
        return True

    tokens = _style_tokens_from_para(p)
    if any(tok in STYLE_LIST_TOKENS for tok in tokens):
        return True

    return _looks_like_marker((p.get("text") or ""))

def _list_level(p: Dict[str, Any]) -> int:
    try: return int(p.get("ilvl")) if p.get("ilvl") is not None else 0
    except Exception: return 0

def analyze_lists(paragraphs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], set]:
    paras = sorted(paragraphs, key=lambda x: x.get("doc_index", 0))
    list_blocks: List[Dict[str, Any]] = []
    consumed = set()
    i = 0
    while i < len(paras):
        p = paras[i]
        if not _is_list_para(p):
            i += 1; continue
        numId = p.get("numId")
        lvl = _list_level(p)
        ordered,_ = _classify_ordering(p)
        base_styles = set(_style_tokens_from_para(p))

        start = i
        items = []
        while i < len(paras):
            q = paras[i]
            if not _is_list_para(q): break
            same = True
            q_numId = q.get("numId")
            if numId is not None:
                if q_numId != numId:
                    same = False
            else:
                if q_numId is not None:
                    same = False
            if same and _list_level(q) != lvl:
                same = False
            if same and numId is None and base_styles:
                q_styles = set(_style_tokens_from_para(q))
                if q_styles and not (base_styles & q_styles):
                    same = False
            if not same: break
            items.append({"doc_index": q["doc_index"], "text": (q.get("text") or "").strip()})
            consumed.add(q["doc_index"]); i += 1
        if items:
            list_blocks.append({
                "id": f"list_{items[0]['doc_index']}",
                "type": "list",
                "doc_index": items[0]["doc_index"],
                "text": None, "level": lvl, "page": None,
                "semantic": None, "table": None,
                "list_data": {"ordered": ordered, "level": lvl, "items": [it['text'] for it in items]},
            })
        if start == i: i += 1
    return list_blocks, consumed

def analyze_tables(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """정제된 테이블 구조를 ContentBlock(dict)로 래핑"""
    blks = []
    for t in tables or []:
        di = t.get("doc_index", -1)
        blks.append({
            "id": f"table_{t.get('tid', di)}",
            "type": "table",
            "doc_index": di,
            "text": None, "level": None, "page": None,
            "semantic": None,
            "table": {"doc_index": di, "rows": t.get("rows"), "cols": t.get("cols"), "data": t.get("data")},
            "list_data": None,
        })
    return blks

def emit_list_table_components_from_sanitized(paragraphs: List[Dict[str, Any]], tables: List[Dict[str, Any]],
                                              sanitized_path: str | Path, basename: str):
    """sanitized JSON을 기반으로 리스트/테이블 컴포넌트를 산출해 저장한다.

    저장 위치 규칙: `_comp/{basename}_{list|table}_components.json`
    반환값은 저장 경로 요약(dict)
    """
    lists, consumed = analyze_lists(paragraphs)
    tables_blk = analyze_tables(tables)
    lp = save_list_components_from_sanitized({"lists": lists, "consumed": sorted(consumed)}, sanitized_path, basename)
    tp = save_table_components_from_sanitized({"tables": tables_blk}, sanitized_path, basename)
    return {"list_path": str(lp), "table_path": str(tp)}
