"""Token-Aware Document Chunker for HAKS File Intelligence (Phase 9.2).

Splits extracted document text into semantic chunks of 400-500 tokens with 50-75 token overlap,
preserving sentence and paragraph boundaries.
"""

import hashlib
import logging
from pathlib import Path
import re
from typing import Optional

from file_intelligence.models import DocumentChunk

logger = logging.getLogger("haks.file_intelligence.chunker")


class DocumentChunker:
    """Chunks text into bounded segments preserving sentence boundaries and section coordinates."""

    def __init__(
        self,
        target_chunk_tokens: int = 450,
        overlap_tokens: int = 60,
        approx_chars_per_token: float = 4.0,
    ) -> None:
        self.target_chunk_tokens = target_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.approx_chars_per_token = approx_chars_per_token

        self.max_chunk_chars = int(target_chunk_tokens * approx_chars_per_token)
        self.overlap_chars = int(overlap_tokens * approx_chars_per_token)

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count based on whitespace and punctuation splitting."""
        words = re.findall(r"\b\w+\b|[^\w\s]", text)
        return max(1, int(len(words) * 1.15))

    def split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences while retaining punctuation and whitespace."""
        # Split on sentence terminals followed by space or newline, or double newlines
        parts = re.split(r"(?<=[.!?\n])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def chunk_section(
        self,
        text: str,
        file_path: str,
        file_name: str,
        file_type: str,
        page_or_section: Optional[str] = None,
        starting_chunk_index: int = 0,
        mtime: float = 0.0,
    ) -> list[DocumentChunk]:
        """Chunk a single section of text into one or more DocumentChunk objects."""
        clean = text.strip()
        if not clean:
            return []

        # If section is already within target token limit, return single chunk
        est_tokens = self.estimate_token_count(clean)
        if est_tokens <= self.target_chunk_tokens:
            chunk_id = self._generate_chunk_id(file_path, starting_chunk_index, clean)
            return [
                DocumentChunk(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    file_name=file_name,
                    file_type=file_type,
                    chunk_index=starting_chunk_index,
                    text=clean,
                    token_count=est_tokens,
                    char_start=0,
                    char_end=len(clean),
                    page_or_section=page_or_section,
                    mtime=mtime,
                )
            ]

        # Break section into sentences and assemble overlapping chunks
        sentences = self.split_into_sentences(clean)
        chunks: list[DocumentChunk] = []
        current_chunk_sentences: list[str] = []
        current_char_len = 0
        char_offset = 0
        current_idx = starting_chunk_index

        for sent in sentences:
            sent_len = len(sent) + 1  # + space
            if current_char_len + sent_len > self.max_chunk_chars and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences).strip()
                tokens = self.estimate_token_count(chunk_text)
                chunk_id = self._generate_chunk_id(file_path, current_idx, chunk_text)

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        file_path=file_path,
                        file_name=file_name,
                        file_type=file_type,
                        chunk_index=current_idx,
                        text=chunk_text,
                        token_count=tokens,
                        char_start=char_offset,
                        char_end=char_offset + len(chunk_text),
                        page_or_section=page_or_section,
                        mtime=mtime,
                    )
                )
                current_idx += 1
                char_offset += len(chunk_text)

                # Maintain overlap by retaining trailing sentences
                overlap_accum = 0
                overlap_sentences = []
                for s in reversed(current_chunk_sentences):
                    overlap_accum += len(s) + 1
                    overlap_sentences.insert(0, s)
                    if overlap_accum >= self.overlap_chars:
                        break

                current_chunk_sentences = overlap_sentences
                current_char_len = sum(len(s) + 1 for s in current_chunk_sentences)

            current_chunk_sentences.append(sent)
            current_char_len += sent_len

        # Remaining trailing sentences
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()
            tokens = self.estimate_token_count(chunk_text)
            chunk_id = self._generate_chunk_id(file_path, current_idx, chunk_text)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    file_name=file_name,
                    file_type=file_type,
                    chunk_index=current_idx,
                    text=chunk_text,
                    token_count=tokens,
                    char_start=char_offset,
                    char_end=char_offset + len(chunk_text),
                    page_or_section=page_or_section,
                    mtime=mtime,
                )
            )

        return chunks

    @staticmethod
    def _generate_chunk_id(file_path: str, index: int, text: str) -> str:
        """Create deterministic stable chunk ID."""
        h = hashlib.sha256(f"{file_path}_{index}_{text[:64]}".encode("utf-8")).hexdigest()[:16]
        return f"chk_{index}_{h}"

