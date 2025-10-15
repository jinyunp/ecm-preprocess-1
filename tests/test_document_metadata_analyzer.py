import os
from datetime import datetime
from pathlib import Path

from mypkg.components.analyzer.document_metadata_analyzer import DocumentMetadataAnalyzer


def test_metadata_analyzer_extracts_fields(tmp_path: Path) -> None:
    doc_path = tmp_path / "1장_v3.1.docx"
    doc_path.write_text("dummy", encoding="utf-8")
    known_time = datetime(2024, 1, 2, 3, 4, 5)
    os.utime(doc_path, (known_time.timestamp(), known_time.timestamp()))

    sanitized = {
        "paragraphs": [
            {"text": "Chapter 1장 품질관리"},
            {"text": "기타 문단"},
        ]
    }

    metadata = DocumentMetadataAnalyzer(sanitized, doc_path.stem, doc_path).analyze()

    assert metadata.chapter == "1장"
    assert metadata.version == "v3.1"
    assert metadata.title == "품질관리"
    assert metadata.last_modified == "2024-01-02"


def test_metadata_analyzer_handles_missing_parts(tmp_path: Path) -> None:
    doc_path = tmp_path / "manual.docx"
    doc_path.write_text("dummy", encoding="utf-8")

    sanitized = {"paragraphs": [{"text": "서론"}]}

    metadata = DocumentMetadataAnalyzer(sanitized, doc_path.stem, doc_path).analyze()

    assert metadata.chapter == "manual"
    assert metadata.version is None
    assert metadata.title is None
    assert metadata.last_modified is not None
