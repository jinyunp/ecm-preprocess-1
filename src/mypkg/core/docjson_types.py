from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class ContentBlockType(Enum):
    """콘텐츠 블록 타입."""

    PARAGRAPH = "paragraph"
    TITLE = "title"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"
    LIST = "list"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    FORMULA = "formula"


@dataclass
class SemanticInfo:
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    cross_refs: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TableData:
    doc_index: int
    rows: int
    cols: int
    data: List[List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InlineImageData:
    rid: str
    filename: Optional[str] = None
    doc_index: Optional[int] = None
    doc_indices: List[int] = field(default_factory=list)
    saved_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "InlineImageData":
        return InlineImageData(
            rid=data.get("rid") or data.get("rId"),
            filename=data.get("filename"),
            doc_index=data.get("doc_index"),
            doc_indices=list(data.get("doc_indices") or []),
            saved_path=data.get("saved_path"),
        )


@dataclass
class ListData:
    ordered: bool
    level: int
    items: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContentBlock:
    id: str
    type: ContentBlockType | str
    doc_index: Optional[int] = None
    text: Optional[str] = None
    level: Optional[int] = None
    page: Optional[int] = None
    semantic: Optional[SemanticInfo] = None
    table: Optional[TableData] = None
    list_data: Optional[ListData] = None
    inline_image: Optional[InlineImageData] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        block_type = self.type
        data["type"] = block_type.value if hasattr(block_type, "value") else block_type
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ContentBlock":
        block_type = data.get("type")
        if isinstance(block_type, str) and block_type in {e.value for e in ContentBlockType}:
            block_type = ContentBlockType(block_type)
        inline_image = data.get("inline_image")
        if inline_image and not isinstance(inline_image, InlineImageData):
            inline_image = InlineImageData.from_dict(inline_image)
        return ContentBlock(
            id=data.get("id"),
            type=block_type if block_type is not None else ContentBlockType.PARAGRAPH,
            doc_index=data.get("doc_index"),
            text=data.get("text"),
            level=data.get("level"),
            page=data.get("page"),
            semantic=data.get("semantic"),
            table=data.get("table"),
            list_data=data.get("list_data"),
            inline_image=inline_image,
        )


@dataclass
class DocumentMetadata:
    chapter: Optional[str] = None
    version: Optional[str] = None
    title: Optional[str] = None
    last_modified: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Section:
    id: str
    number: str
    title: str
    level: int
    doc_index: int
    span: List[int] = field(default_factory=lambda: [0, 0])
    path: List[str] = field(default_factory=list)
    block_ids: List[str] = field(default_factory=list)
    blocks: List["ContentBlock"] = field(default_factory=list)
    subsections: List["Section"] = field(default_factory=list)
    heading_block_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        spans = [(sub.span[0], sub.span[1]) for sub in self.subsections if sub.span and len(sub.span) == 2]

        def _covered(doc_index: Optional[int]) -> bool:
            if doc_index is None:
                return False
            return any(start <= doc_index < end for start, end in spans)

        filtered_block_objs = [b for b in self.blocks if not _covered(b.doc_index)]
        filtered_blocks = [b.to_dict() for b in filtered_block_objs]
        filtered_block_ids = [b.id for b in filtered_block_objs]

        return {
            "id": self.id,
            "number": self.number,
            "title": self.title,
            "level": self.level,
            "doc_index": self.doc_index,
            "span": list(self.span),
            "path": list(self.path),
            "block_ids": filtered_block_ids,
            "blocks": filtered_blocks,
            "subsections": [s.to_dict() for s in self.subsections],
            "heading_block_id": self.heading_block_id,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Section":
        section = Section(
            id=data.get("id"),
            number=data.get("number"),
            title=data.get("title"),
            level=int(data.get("level")) if data.get("level") is not None else 1,
            doc_index=int(data.get("doc_index")) if data.get("doc_index") is not None else 0,
        )
        section.span = list(data.get("span") or [section.doc_index, section.doc_index + 1])
        section.path = list(data.get("path") or [])
        section.block_ids = list(data.get("block_ids") or [])
        blocks = data.get("blocks") or []
        section.blocks = [ContentBlock.from_dict(b) if not isinstance(b, ContentBlock) else b for b in blocks]
        subs = data.get("subsections") or []
        section.subsections = [Section.from_dict(s) if not isinstance(s, Section) else s for s in subs]
        section.heading_block_id = data.get("heading_block_id")
        return section


@dataclass
class DocumentDocJSON:
    version: str
    metadata: DocumentMetadata | Dict[str, Any]
    blocks: List[ContentBlock] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)
    inline_images: List[InlineImageData] = field(default_factory=list)
    include_metadata: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.metadata, dict):
            self.metadata = DocumentMetadata(**self.metadata)
        self.blocks = [b if isinstance(b, ContentBlock) else ContentBlock.from_dict(b) for b in (self.blocks or [])]
        self.sections = [s if isinstance(s, Section) else Section.from_dict(s) for s in (self.sections or [])]
        self.inline_images = [img if isinstance(img, InlineImageData) else InlineImageData.from_dict(img) for img in (self.inline_images or [])]

    def to_dict(self) -> Dict[str, Any]:
        payload = {"version": self.version}
        if self.include_metadata and self.metadata is not None:
            payload["metadata"] = self.metadata.to_dict()
        payload["sections"] = [s.to_dict() for s in self.sections]
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DocumentDocJSON":
        return cls(
            version=payload.get("version", "0.0.0"),
            metadata=payload.get("metadata", {}),
            blocks=payload.get("blocks", []),
            sections=payload.get("sections", []),
            inline_images=payload.get("inline_images", []),
        )
