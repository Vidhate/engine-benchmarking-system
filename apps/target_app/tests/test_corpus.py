"""Corpus loading: frontmatter parsing, body/archived split, determinism."""

from target_app.corpus import CORPUS_DIR, Document, load_corpus


def test_loads_every_markdown_file():
    docs = load_corpus()
    md_files = sorted(CORPUS_DIR.glob("*.md"))
    assert len(docs) == len(md_files)
    assert 8 <= len(docs) <= 12


def test_documents_have_required_metadata():
    for doc in load_corpus():
        assert isinstance(doc, Document)
        assert doc.doc_id and " " not in doc.doc_id
        assert doc.title
        assert doc.tags, f"{doc.doc_id} has no tags"
        assert doc.updated
        assert doc.text.strip()


def test_doc_ids_are_unique():
    ids = [d.doc_id for d in load_corpus()]
    assert len(ids) == len(set(ids))


def test_body_excludes_the_archived_revision():
    doc = {d.doc_id: d for d in load_corpus()}["refund-policy"]
    assert "30 days" in doc.text
    assert "90 days" not in doc.text  # lives only in the archived revision
    assert "ARCHIVED" not in doc.text


def test_every_doc_carries_a_stale_revision():
    for doc in load_corpus():
        assert doc.stale_text.strip(), f"{doc.doc_id} has no archived revision"
        assert doc.stale_updated, f"{doc.doc_id} has no archived date"
        assert doc.stale_text != doc.text


def test_stale_revision_contradicts_current_text():
    doc = {d.doc_id: d for d in load_corpus()}["refund-policy"]
    assert "90 days" in doc.stale_text
    assert doc.stale_updated == "2019-03-02"


def test_loading_is_deterministic_and_sorted():
    first = load_corpus()
    second = load_corpus()
    assert [d.doc_id for d in first] == [d.doc_id for d in second]
    assert [d.doc_id for d in first] == sorted(d.doc_id for d in first)


def test_load_corpus_accepts_an_explicit_directory(tmp_path):
    (tmp_path / "solo.md").write_text(
        "---\n"
        "doc_id: solo\n"
        "title: Solo Doc\n"
        "tags: alpha, beta\n"
        "updated: 2026-01-01\n"
        "---\n\n"
        "Current body.\n\n"
        "<!-- ARCHIVED 2001-01-01 -->\n\n"
        "Archived body.\n"
    )
    docs = load_corpus(tmp_path)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.doc_id == "solo"
    assert doc.tags == ["alpha", "beta"]
    assert doc.text.strip() == "Current body."
    assert doc.stale_text.strip() == "Archived body."
    assert doc.stale_updated == "2001-01-01"
