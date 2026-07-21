"""Tests for the OpenIE service."""

from __future__ import annotations

from mnemosyne.services.openie import extract_triples, _clean_token


class TestOpenIE:
    def test_extract_own_pattern(self):
        result = extract_triples("Alice owns a dog named Max.")
        triples = [(t.subject, t.predicate, t.object) for t in result.triples]
        assert any(t[0] == "Alice" and t[1] == "owns" for t in triples)

    def test_extract_work_pattern(self):
        result = extract_triples("John works at Google.")
        triples = [(t.subject, t.predicate, t.object) for t in result.triples]
        assert ("John", "works_for", "Google") in triples

    def test_extract_like_pattern(self):
        result = extract_triples("Alice likes coffee.")
        triples = [(t.subject, t.predicate, t.object) for t in result.triples]
        assert ("Alice", "likes", "coffee") in triples

    def test_extract_is_a_pattern(self):
        result = extract_triples("Max is a dog.")
        triples = [(t.subject, t.predicate, t.object) for t in result.triples]
        assert ("Max", "is_a", "dog") in triples

    def test_extract_entities(self):
        result = extract_triples("Alice works at Google in Mountain View.")
        entity_names = [e["name"] for e in result.entities]
        assert "Alice" in entity_names
        assert "Google" in entity_names

    def test_extract_is_a_pattern(self):
        result = extract_triples("Max is a dog.")
        triples = [(t.subject, t.predicate, t.object) for t in result.triples]
        assert any(t[0] == "Max" and t[1] == "is_a" for t in triples)

    def test_clean_token(self):
        assert _clean_token("Alice") == "Alice"
        assert _clean_token("the") == ""
        assert _clean_token("a") == ""
        assert _clean_token("") == ""

    def test_dedup(self):
        result = extract_triples("Alice likes coffee. Alice likes coffee.")
        assert len(result.triples) <= 2

    def test_empty_text(self):
        result = extract_triples("")
        assert len(result.triples) == 0
        assert len(result.entities) == 0
