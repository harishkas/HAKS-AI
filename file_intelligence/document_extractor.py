"""Multi-Format Document Content Extractor for HAKS File Intelligence (Phase 9.2).

Extracts structured text and metadata from PDF, DOCX, XLSX, PPTX, TXT, MD, CSV, and HTML
with explicit page, slide, and sheet provenance tracking.
"""

import csv
from html.parser import HTMLParser
import io
import logging
import os
from pathlib import Path
from typing import Generator, Optional, Tuple

logger = logging.getLogger("haks.file_intelligence.extractor")


class HTMLTextExtractor(HTMLParser):
    """Safe streaming parser extracting visible text while discarding scripts and styles."""

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []
        self._ignore: bool = False

    def handle_starttag(self, tag: str, attrs: list[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() in ("script", "style", "noscript"):
            self._ignore = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in ("script", "style", "noscript"):
            self._ignore = False
        elif tag.lower() in ("p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._pieces.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignore:
            text = data.strip()
            if text:
                self._pieces.append(text + " ")

    def get_text(self) -> str:
        return "".join(self._pieces).strip()


class DocumentExtractor:
    """Extracts text sections with structural coordinates from supported document formats."""

    SUPPORTED_EXTENSIONS = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".xlsx": "xlsx",
        ".pptx": "pptx",
        ".txt": "txt",
        ".md": "md",
        ".markdown": "md",
        ".csv": "csv",
        ".html": "html",
        ".htm": "html",
        ".json": "txt",
        ".xml": "txt",
        ".py": "txt",
        ".bat": "txt",
        ".ps1": "txt",
    }

    def __init__(self, max_file_size_bytes: int = 50 * 1024 * 1024) -> None:
        self.max_file_size_bytes = max_file_size_bytes

    def is_supported(self, file_path: str | Path) -> bool:
        """Check if file extension is supported for text extraction."""
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS

    def extract_sections(self, file_path: str | Path) -> Generator[Tuple[str, Optional[str]], None, None]:
        """Yield (text_block, section_or_page_ref) from document.

        Section references:
        - PDF: "Page 1", "Page 2"
        - PPTX: "Slide 1", "Slide 2"
        - XLSX: "Sheet: Summary", "Sheet: Q3_Data"
        - DOCX: "Paragraphs 1-10", "Table 1"
        - TXT/MD/CSV/HTML: None or "Lines 1-50"
        """
        p = Path(file_path).resolve()
        if not p.exists() or not p.is_file():
            logger.warning("Target document does not exist: %s", p)
            return

        size = p.stat().st_size
        if size > self.max_file_size_bytes:
            logger.warning("Skipping oversized file (%d MB > %d MB limit): %s", size // (1024*1024), self.max_file_size_bytes // (1024*1024), p)
            return

        ext = p.suffix.lower()
        file_type = self.SUPPORTED_EXTENSIONS.get(ext)
        if not file_type:
            logger.debug("Unsupported extension '%s' for %s", ext, p.name)
            return

        try:
            if file_type == "pdf":
                yield from self._extract_pdf(p)
            elif file_type == "docx":
                yield from self._extract_docx(p)
            elif file_type == "xlsx":
                yield from self._extract_xlsx(p)
            elif file_type == "pptx":
                yield from self._extract_pptx(p)
            elif file_type == "csv":
                yield from self._extract_csv(p)
            elif file_type == "html":
                yield from self._extract_html(p)
            else:
                yield from self._extract_text(p)
        except Exception as exc:
            logger.error("Failed to extract content from '%s': %s", p.name, exc)

    def extract_full_text(self, file_path: str | Path) -> str:
        """Convenience method returning entire extracted document text concatenated."""
        blocks = [text for text, _ in self.extract_sections(file_path)]
        return "\n\n".join(blocks).strip()

    # --------------------------------------------------------------------------
    # Format-Specific Extractors
    # --------------------------------------------------------------------------

    def _extract_pdf(self, path: Path) -> Generator[Tuple[str, str], None, None]:
        """Extract text page-by-page using PyMuPDF."""
        try:
            import pymupdf
        except ImportError:
            logger.error("pymupdf not installed; skipping PDF extraction.")
            return

        doc = None
        try:
            doc = pymupdf.open(str(path))
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text("text").strip()
                if text:
                    yield text, f"Page {page_idx + 1}"
        finally:
            if doc is not None:
                doc.close()

    def _extract_docx(self, path: Path) -> Generator[Tuple[str, str], None, None]:
        """Extract text from Word documents using python-docx."""
        try:
            import docx
        except ImportError:
            logger.error("python-docx not installed; skipping DOCX extraction.")
            return

        doc = docx.Document(str(path))

        # Paragraphs in batches
        para_batch = []
        batch_idx = 1
        for p in doc.paragraphs:
            t = p.text.strip()
            if t:
                para_batch.append(t)
                if len(para_batch) >= 15:
                    yield "\n".join(para_batch), f"Section {batch_idx}"
                    para_batch = []
                    batch_idx += 1

        if para_batch:
            yield "\n".join(para_batch), f"Section {batch_idx}"

        # Tables
        for t_idx, table in enumerate(doc.tables, 1):
            table_lines = []
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    table_lines.append(" | ".join(cells))
            if table_lines:
                yield "\n".join(table_lines), f"Table {t_idx}"

    def _extract_xlsx(self, path: Path) -> Generator[Tuple[str, str], None, None]:
        """Extract spreadsheet cells sheet-by-sheet using openpyxl."""
        try:
            import openpyxl
        except ImportError:
            logger.error("openpyxl not installed; skipping XLSX extraction.")
            return

        wb = None
        try:
            wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                rows_text = []
                for row in sheet.iter_rows(values_only=True):
                    row_vals = [str(v).strip() for v in row if v is not None and str(v).strip()]
                    if row_vals:
                        rows_text.append(" | ".join(row_vals))

                if rows_text:
                    # Yield in chunks of 50 rows to keep chunks bounded
                    for i in range(0, len(rows_text), 50):
                        chunk_rows = rows_text[i : i + 50]
                        yield "\n".join(chunk_rows), f"Sheet: {sheet_name}"
        finally:
            if wb is not None:
                wb.close()

    def _extract_pptx(self, path: Path) -> Generator[Tuple[str, str], None, None]:
        """Extract slide presentations slide-by-slide using python-pptx."""
        try:
            import pptx
        except ImportError:
            logger.error("python-pptx not installed; skipping PPTX extraction.")
            return

        prs = pptx.Presentation(str(path))
        for slide_idx, slide in enumerate(prs.slides, 1):
            slide_texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        t = para.text.strip()
                        if t:
                            slide_texts.append(t)

            # Slide notes
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    slide_texts.append(f"[Notes: {notes}]")

            if slide_texts:
                yield "\n".join(slide_texts), f"Slide {slide_idx}"

    def _extract_csv(self, path: Path) -> Generator[Tuple[str, str], None, None]:
        """Extract delimited CSV data in batches of rows."""
        lines = []
        encoding = self._detect_text_encoding(path)
        with open(path, "r", encoding=encoding, errors="replace") as f:
            reader = csv.reader(f)
            header = None
            try:
                header = next(reader, None)
            except Exception:
                pass

            if header:
                lines.append("Header: " + ", ".join(header))

            batch = []
            for row in reader:
                cleaned = [c.strip() for c in row if c.strip()]
                if cleaned:
                    batch.append(", ".join(cleaned))
                    if len(batch) >= 50:
                        yield "\n".join(lines + batch), "CSV Data"
                        batch = []

            if batch:
                yield "\n".join(lines + batch), "CSV Data"

    def _extract_html(self, path: Path) -> Generator[Tuple[str, str], None, None]:
        """Extract clean visible text from HTML files."""
        encoding = self._detect_text_encoding(path)
        with open(path, "r", encoding=encoding, errors="replace") as f:
            content = f.read()

        parser = HTMLTextExtractor()
        parser.feed(content)
        extracted = parser.get_text()
        if extracted:
            yield extracted, "HTML Content"

    def _extract_text(self, path: Path) -> Generator[Tuple[str, str], None, None]:
        """Extract plain text / markdown files with encoding detection."""
        encoding = self._detect_text_encoding(path)
        with open(path, "r", encoding=encoding, errors="replace") as f:
            lines = []
            line_start = 1
            current_line = 1
            for line in f:
                lines.append(line)
                if len(lines) >= 80:
                    text_block = "".join(lines).strip()
                    if text_block:
                        yield text_block, f"Lines {line_start}-{current_line}"
                    lines = []
                    line_start = current_line + 1
                current_line += 1

            if lines:
                text_block = "".join(lines).strip()
                if text_block:
                    yield text_block, f"Lines {line_start}-{current_line}"

    @staticmethod
    def _detect_text_encoding(path: Path) -> str:
        """Quickly detect UTF-8 vs local Windows ANSI encoding."""
        try:
            with open(path, "rb") as f:
                raw = f.read(4096)
            raw.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            return "cp1252"

