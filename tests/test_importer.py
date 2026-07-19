"""Tests for the import service."""

from __future__ import annotations

import pytest

from mnemosyne.services.importer import ImportService


class TestChunkText:
    """Test text chunking logic."""

    def test_short_text_no_split(self):
        chunks = ImportService._chunk_text("Hello world.")
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_paragraph_split(self):
        text = "A. " * 200 + "\n\n" + "B. " * 200
        chunks = ImportService._chunk_text(text)
        assert len(chunks) >= 2

    def test_sentence_split_long_paragraph(self):
        long_para = "Sent one. " * 60
        chunks = ImportService._chunk_text(long_para)
        for chunk in chunks:
            assert len(chunk) <= 500

    def test_very_long_sentence(self):
        long_sent = "word " * 200
        chunks = ImportService._chunk_text(long_sent)
        for chunk in chunks:
            assert len(chunk) <= 500

    def test_empty_text(self):
        chunks = ImportService._chunk_text("")
        assert len(chunks) >= 1

    def test_skips_empty_paragraphs(self):
        text = ("A. " * 200) + "\n\n\n\n" + ("B. " * 200)
        chunks = ImportService._chunk_text(text)
        assert len(chunks) >= 2

    def test_merges_small_paragraphs(self):
        text = "A.\n\nB.\n\nC."
        chunks = ImportService._chunk_text(text)
        assert len(chunks) == 1
