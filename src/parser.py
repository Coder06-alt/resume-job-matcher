"""Text extraction from PDF and DOCX resumes."""
from __future__ import annotations

import io
from typing import BinaryIO, Union

import PyPDF2
import docx


class ResumeParseError(Exception):
    """Raised when a resume file cannot be read into usable text."""


def extract_text_from_pdf(file: Union[BinaryIO, io.BytesIO]) -> str:
    """Return all text from a PDF. Scanned (image-only) PDFs give empty text."""
    try:
        reader = PyPDF2.PdfReader(file)
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # corrupted / encrypted file
        raise ResumeParseError(f"Could not read PDF: {exc}") from exc
    text = "\n".join(pages).strip()
    if not text:
        raise ResumeParseError(
            "No text found. This may be a scanned PDF (image only); OCR is needed."
        )
    return text


def extract_text_from_docx(file: Union[BinaryIO, io.BytesIO]) -> str:
    """Return all paragraph and table text from a DOCX file."""
    try:
        document = docx.Document(file)
    except Exception as exc:
        raise ResumeParseError(f"Could not read DOCX: {exc}") from exc
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    text = "\n".join(parts).strip()
    if not text:
        raise ResumeParseError("The DOCX file contains no text.")
    return text


def extract_text(file: BinaryIO, filename: str) -> str:
    """Pick the right extractor from the file extension."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    if name.endswith(".docx"):
        return extract_text_from_docx(file)
    if name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")
    raise ResumeParseError("Unsupported file type. Upload a PDF, DOCX or TXT file.")
