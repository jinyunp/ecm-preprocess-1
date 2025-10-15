# src/mypkg/pipelines/docx_parsing_pipeline.py

import asyncio
import json
import logging
import zipfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# 파서
from mypkg.components.parser.xml_parser import DocxXmlParser

# 정제기
from mypkg.components.sanitizer.paragraph_sanitizer import ParagraphSanitizer
from mypkg.components.sanitizer.table_sanitizer import TableSanitizer

# 코어 유틸리티
from mypkg.core.io import write_json_output, save_inline_image_components_from_sanitized


log = logging.getLogger(__name__)

class DocxParsingPipeline:
    """
    A pipeline that takes a DOCX file, runs it through parsers and sanitizers,
    and saves the intermediate and final outputs to a specified directory.
    """

    def __init__(
        self,
        xml_parser: Optional[DocxXmlParser] = None,
        para_sanitizer: Optional[ParagraphSanitizer] = None,
        table_sanitizer: Optional[TableSanitizer] = None,
    ) -> None:

        self.xml_parser = xml_parser or DocxXmlParser()
        self.para_sanitizer = para_sanitizer or ParagraphSanitizer()
        self.table_sanitizer = table_sanitizer or TableSanitizer()

    def _enhance_inline_images(
        self,
        inline_images: List[Any],
        xml_content_data: Dict[str, Any],
        docx_path: Path,
        assets_dir: Path,
    ) -> None:
        """inline 이미지 정보를 이용해 원본 이미지를 추출하고 저장 경로를 기록한다."""
        if not inline_images:
            return

        name_counts: Dict[str, int] = defaultdict(int)

        try:
            with zipfile.ZipFile(docx_path, "r") as zf:
                for record in inline_images:
                    rid = getattr(record, "rId", None)
                    saved_path = getattr(record, "saved_path", None)
                    if not rid or not saved_path:
                        continue
                    part_path = saved_path.lstrip("/")
                    try:
                        binary = zf.read(part_path)
                    except KeyError:
                        continue

                    filename = getattr(record, "filename", None) or Path(part_path).name or f"{rid}"
                    if not filename:
                        filename = f"{rid}.bin"

                    base_name = filename
                    count = name_counts[base_name]
                    if count:
                        stem = Path(base_name).stem or base_name
                        suffix = Path(base_name).suffix
                        filename = f"{stem}_{count}{suffix}"
                    name_counts[base_name] += 1

                    assets_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = assets_dir / filename
                    dest_path.write_bytes(binary)
                    record.filename = filename
                    record.saved_path = str(Path("_assets") / filename)

        except FileNotFoundError:
            return

        for record in inline_images:
            indices = getattr(record, "doc_indices", []) or []
            if indices:
                unique = sorted(set(indices))
                record.doc_indices = unique
                if record.doc_index is None:
                    record.doc_index = unique[0]

    async def run(self, file_path: Path, output_base_dir: Path) -> Dict[str, str]:
        """
        Executes the parsing and sanitizing pipeline and saves the outputs.
        Returns a dictionary of the paths to the saved files.
        """
        doc_name = file_path.stem
        # 새 디렉터리 규칙: base_dir/_sanitized
        output_dir = output_base_dir / "_sanitized"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Parsing stage (XML as primary source)
        res_xml = await self.xml_parser.parse(file_path)
        if not res_xml.success:
            raise RuntimeError(f"XML parsing failed for {file_path}: {res_xml.error}")

        xml_content_data = res_xml.content or {}

        docx_content_data = {
            "paragraphs": xml_content_data.get("paragraphs", []),
            "inline_images": xml_content_data.get("inline_images", []),
        }

        # Save raw parser outputs
        xml_output_path = output_dir / f"{doc_name}_output_xml.json"
        write_json_output(asdict(res_xml), xml_output_path)

        # Extract content for sanitizers
        paragraphs_for_context = xml_content_data.get("paragraphs", [])

        # 인라인 이미지 보강 및 저장
        inline_images = docx_content_data.get("inline_images", [])
        if inline_images:
            assets_dir = output_dir / "_assets"
            self._enhance_inline_images(
                inline_images,
                xml_content_data,
                file_path,
                assets_dir
            )

        # 2. Sanitizing stage
        sanitized_paragraphs = self.para_sanitizer.apply(
            docx_content_data.get("paragraphs", []),
            paragraphs_for_context
        )

        sanitized_tables = self.table_sanitizer.apply(xml_content_data.get("tables", []), paragraphs_for_context)

        # 3. Assembling and saving sanitized result
        final_result = {
            "paragraphs": [asdict(p) for p in sanitized_paragraphs],
            "tables": sanitized_tables,
            "relationships": {
                "map": {k: asdict(v) for k, v in xml_content_data.get("relationships", {}).get("map", {}).items()}
            },
            "inline_images": [asdict(i) for i in docx_content_data.get("inline_images", [])],
        }
        
        sanitized_output_path = output_dir / f"{doc_name}_sanitized.json"
        write_json_output(final_result, sanitized_output_path)

        # Inline image 컴포넌트 저장
        inline_image_payload = {"inline_images": final_result["inline_images"]}
        save_inline_image_components_from_sanitized(inline_image_payload, sanitized_output_path, doc_name)

        return {
            "xml_parser_output": str(xml_output_path),
            "sanitized_output": str(sanitized_output_path),
        }
