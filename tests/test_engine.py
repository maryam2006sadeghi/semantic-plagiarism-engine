import pytest
from plagiarism_engine.dataset import Dataset
from plagiarism_engine.minhash import MinHash, MinHashConfig
from plagiarism_engine.lsh import LSH, LSHConfig
from plagiarism_engine.preprocessing import (
    clean_text,
    tokenize,
    remove_stopwords,
    generate_word_shingles,
    generate_char_shingles,
    preprocess_text,
    shingle_preprocess,
    is_english_text,
)
import sys
from pathlib import Path
from typing import Any, List, Set

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))


def test_dataset_loading():
    dataset = Dataset("data/sample_corpus")
    dataset.load()
    assert len(dataset.source_docs) > 0
    assert len(dataset.suspicious_docs) > 0
    assert len(dataset.ground_truth) > 0


def test_read_source_document():
    dataset = Dataset("data/sample_corpus")
    dataset.load()
    filename = next(iter(dataset.source_docs))
    text = dataset.get_source_text(filename)
    assert isinstance(text, str)
    assert len(text) > 0


def test_read_suspicious_document():
    dataset = Dataset("data/sample_corpus")
    dataset.load()
    filename = next(iter(dataset.suspicious_docs))
    text = dataset.get_suspicious_text(filename)
    assert isinstance(text, str)
    assert len(text) > 0


def test_ground_truth_exists():
    dataset = Dataset("data/sample_corpus")
    dataset.load()
    suspicious = next(iter(dataset.ground_truth))
    truth = dataset.get_ground_truth(suspicious)
    assert isinstance(truth, set)
    assert len(truth) > 0


def test_source_documents_iterator():
    dataset = Dataset("data/sample_corpus")
    dataset.load()
    docs = list(dataset.source_documents())
    assert len(docs) == len(dataset.source_docs)


def test_suspicious_documents_iterator():
    dataset = Dataset("data/sample_corpus")
    dataset.load()
    docs = list(dataset.suspicious_documents())
    assert len(docs) == len(dataset.suspicious_docs)


def test_clean_text_basic():
    text = "Hello!!! This IS a TEST."
    cleaned = clean_text(text)
    assert isinstance(cleaned, str)
    assert cleaned.islower()
    assert "hello" in cleaned
    assert "test" in cleaned


def test_tokenize_basic():
    text = "hello world! 123"
    tokens = tokenize(text)
    assert isinstance(tokens, list)
    assert "hello" in tokens
    assert "world" in tokens
    assert "123" in tokens


def test_remove_stopwords():
    tokens = ["this", "is", "a", "test"]
    filtered = remove_stopwords(tokens, stopwords={"this", "is", "a"})
    assert "this" not in filtered
    assert "is" not in filtered
    assert "test" in filtered


def test_generate_word_shingles():
    tokens = ["this", "is", "a", "test"]
    shingles = generate_word_shingles(tokens, k=2)
    assert isinstance(shingles, set)
    assert "this is" in shingles
    assert "is a" in shingles
    assert "a test" in shingles


def test_generate_char_shingles():
    text = "abcd"
    shingles = generate_char_shingles(text, k=2)
    assert isinstance(shingles, set)
    assert "ab" in shingles
    assert "bc" in shingles
    assert "cd" in shingles


def test_is_english_text_true():
    assert is_english_text("This is a normal English sentence.")


def test_is_english_text_false():
    assert not is_english_text("سلام")


def test_preprocess_text():
    text = "This is a simple simple test document for testing."
    tokens = preprocess_text(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_shingle_preprocess():
    tokens = ["simpl", "test", "document", "test"]
    shingles = shingle_preprocess(tokens, k=2)
    assert isinstance(shingles, set)
    assert len(shingles) > 0


def test_preprocess_short_text():
    assert preprocess_text("hi") == []


def test_empty_shingles():
    assert shingle_preprocess([]) == set()


@pytest.fixture(scope="session")
def dataset():
    ds = Dataset("data/sample_corpus")
    ds.load()
    return ds


@pytest.fixture(scope="session")
def shingles(dataset: Dataset):
    all_docs = {}
    for name in dataset.source_docs:
        text = dataset.get_source_text(name)
        tokens = preprocess_text(text)
        all_docs[name] = shingle_preprocess(tokens)
    for name in dataset.suspicious_docs:
        text = dataset.get_suspicious_text(name)
        tokens = preprocess_text(text)
        all_docs[name] = shingle_preprocess(tokens)
    return all_docs


@pytest.fixture(scope="session")
def signatures(shingles: dict):
    mh = MinHash(MinHashConfig(num_perm=64, seed=42))
    return {
        doc_id: mh.signature(sh)
        for doc_id, sh in shingles.items()
    }


def test_minhash_signatures_exist(signatures: dict[Any, List[int]]):
    assert len(signatures) > 0
    for sig in signatures.values():
        assert isinstance(sig, list)
        assert len(sig) == 64


def test_lsh_runs_on_real_dataset(signatures: dict[Any, List[int]]):
    lsh = LSH(LSHConfig(signature_size=64, bands=32, rows=2))
    for doc_id, sig in signatures.items():
        lsh.index(doc_id, sig)
    candidates = lsh.get_candidate_pairs()
    assert isinstance(candidates, set)


def test_lsh_query_returns_results(signatures: dict[Any, List[int]]):
    lsh = LSH(LSHConfig(signature_size=64, bands=32, rows=2))
    doc_id, sig = next(iter(signatures.items()))
    for d, s in signatures.items():
        lsh.index(d, s)
    results = lsh.query(sig)
    assert isinstance(results, set)
    assert doc_id in results
