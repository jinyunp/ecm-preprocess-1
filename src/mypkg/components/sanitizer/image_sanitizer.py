from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import logging

try:
    import pytesseract
    from PIL import Image
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

from mypkg.core.parser import InlineImageRecord, TableRecord, ParagraphRecord
from mypkg.core.docjson_types import ImageComponentData

log = logging.getLogger(__name__)

class ImageSanitizer:
    """Performs OCR and location analysis for inline images."""

    def __init__(self, lang: str = 'kor+eng'):
        self.lang = lang
        if not PYTESSERACT_AVAILABLE:
            log.warning("pytesseract or Pillow is not installed. OCR functionality will be disabled.")

    def _perform_ocr(self, image_path: Path) -> str | None:
        if not PYTESSERACT_AVAILABLE or not image_path.exists() or image_path.suffix.lower() != '.png':
            return None
        try:
            with Image.open(image_path) as img:
                text = pytesseract.image_to_string(img, lang=self.lang)
                return text.strip()
        except Exception as e:
            log.error(f"OCR failed for image {image_path}: {e}")
            # Inform the user that Tesseract might not be installed or in PATH
            if "tesseract is not installed or it's not in your PATH" in str(e).lower():
                log.error("Tesseract OCR engine is likely not installed or not in the system's PATH.")
            return None

    def sanitize(self, inline_images: List[InlineImageRecord], tables: List[TableRecord], sanitized_paragraphs: List[ParagraphRecord], base_dir: Path) -> List[ImageComponentData]:
        sanitized_images: List[ImageComponentData] = []
        
        rids_in_tables = set()
        for table in tables or []:
            for row in table.rows:
                for cell in row:
                    for rId in cell.inline_images:
                        rids_in_tables.add(rId)

        para_text_map = {p.doc_index: p.text for p in sanitized_paragraphs if p.doc_index is not None}

        for image_record in inline_images or []:
            is_in_table = image_record.rId in rids_in_tables
            
            is_in_paragraph = False
            if image_record.doc_index in para_text_map:
                para_text = para_text_map[image_record.doc_index]
                if f"[image:{image_record.rId}]" in para_text:
                    is_in_paragraph = True

            ocr_text = None
            full_image_path = None
            if image_record.saved_path:
                full_image_path = base_dir / image_record.saved_path
                ocr_text = self._perform_ocr(full_image_path)

            component = ImageComponentData(
                rId=image_record.rId,
                filename=image_record.filename,
                doc_index=image_record.doc_index,
                saved_path=str(full_image_path) if full_image_path else image_record.saved_path,
                ocr_text=ocr_text,
                is_in_table=is_in_table,
                is_in_paragraph=is_in_paragraph,
            )
            sanitized_images.append(component)

        return sanitized_images

    def build_components(self, sanitized_images: List[ImageComponentData]) -> Dict[str, List[Dict[str, Any]]]:
        """Wraps the list of sanitized images in a dictionary for JSON output."""
        return {"images": [img.to_dict() for img in sanitized_images]}
