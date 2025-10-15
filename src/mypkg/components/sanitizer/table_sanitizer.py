from typing import List, Dict, Any
from mypkg.core.parser import TableRecord, TableCellRecord, ParagraphRecord

class TableSanitizer:
    """
    - gridSpan/vMerge 보정 (None/0 -> 1)
    - 병합 전체 영역을 앵커 텍스트로 채움 (rowspan x colspan 사각형)
    - 2D matrix(data) 생성 + rows/cols 채우기
    - 테이블 바로 앞 문단 텍스트(preceding_text) 추가
    """
    def __init__(self, blank_placeholder: str = ""):
        self.blank = blank_placeholder

    def apply(self, tables: List[TableRecord], paragraphs: List[ParagraphRecord]) -> List[Dict[str, Any]]:
        out = []
        all_paragraphs = {p.doc_index: p for p in paragraphs if p.doc_index is not None}

        for t in tables or []:
            # 테이블의 이전 문단 찾기
            preceding_text = None
            if t.doc_index is not None and t.doc_index > 0:
                preceding_para = all_paragraphs.get(t.doc_index - 1)
                if preceding_para:
                    preceding_text = preceding_para.text

            rows = t.rows
            col_counts = [sum(max(1, c.gridSpan or 1) for c in row) for row in rows] or [0]
            C = max(col_counts) if col_counts else 0
            R = len(rows)

            matrix = [[self.blank for _ in range(C)] for _ in range(R)]
            image_matrix = [[[] for _ in range(C)] for _ in range(R)]
            color_matrix = [[None for _ in range(C)] for _ in range(R)]
            vflags = [[None for _ in range(C)] for _ in range(R)]

            anchors = []

            for r_idx, row in enumerate(rows):
                c_idx = 0
                while c_idx < C and vflags[r_idx][c_idx] == "continue":
                    c_idx += 1

                for cell in row:
                    colspan = max(1, cell.gridSpan or 1)
                    vmerge = cell.vMerge
                    txt = (cell.text or "").strip()

                    while c_idx < C and vflags[r_idx][c_idx] == "continue":
                        c_idx += 1

                    if c_idx >= C:
                        break

                    matrix[r_idx][c_idx] = txt
                    image_matrix[r_idx][c_idx] = list(getattr(cell, "inline_images", []) or [])
                    color_matrix[r_idx][c_idx] = getattr(cell, "bg_color", None)

                    for off in range(colspan):
                        cc = c_idx + off
                        if cc >= C:
                            break
                        if off > 0:
                            matrix[r_idx][cc] = self.blank
                            image_matrix[r_idx][cc] = list(getattr(cell, "inline_images", []) or [])
                            color_matrix[r_idx][cc] = getattr(cell, "bg_color", None)
                        if vmerge:
                            vflags[r_idx][cc] = vmerge

                    anchors.append({
                        "r": r_idx, "c": c_idx,
                        "text": txt, "colspan": colspan,
                        "vmerge": vmerge
                    })
                    c_idx += colspan

            for a in anchors:
                r0, c0 = a["r"], a["c"]
                text = a["text"]
                colspan = max(1, a["colspan"] or 1)
                vmerge = a["vmerge"]

                rowspan = 1
                if vmerge == "restart":
                    rr = r0 + 1
                    while rr < R:
                        all_cont = True
                        for off in range(colspan):
                            cc = c0 + off
                            if cc >= C or vflags[rr][cc] != "continue":
                                all_cont = False
                                break
                        if not all_cont:
                            break
                        rowspan += 1
                        rr += 1
                elif vmerge == "continue":
                    rowspan = 1
                else:
                    rowspan = 1

                base_images = list(image_matrix[r0][c0])
                base_color = color_matrix[r0][c0]
                for rr in range(r0, min(R, r0 + rowspan)):
                    for off in range(colspan):
                        cc = c0 + off
                        if cc < C:
                            matrix[rr][cc] = text
                            image_matrix[rr][cc] = list(base_images)
                            color_matrix[rr][cc] = base_color

            cleaned_rows = []
            cleaned_images = []
            cleaned_colors = []
            for r in range(R):
                cleaned_rows.append([str(x) if x is not None else self.blank for x in matrix[r]])
                cleaned_images.append([list(imgs) for imgs in image_matrix[r]])
                cleaned_colors.append([color_matrix[r][c] for c in range(C)])

            t_out = {
                "tid": t.tid,
                "doc_index": t.doc_index,
                "preceding_text": preceding_text,
                "rows": R,
                "cols": C,
                "data": cleaned_rows,
                "cell_images": cleaned_images,
                "cell_colors": cleaned_colors,
            }
            out.append(t_out)

        return out
