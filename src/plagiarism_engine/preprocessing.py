import re
import ssl
import nltk
import os
from typing import List, Set, Optional


nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
STOPWORDS = set(nltk.corpus.stopwords.words('english'))


def is_english_text(text: str, threshold: float = 0.70) -> bool:
    if not text or len(text.strip()) < 3:
        return False

    allowed_pattern = re.compile(r'[a-zA-Z0-9\s\.\,\!\?\;\:\"\'\-\(\)]')
    allowed_count = len(allowed_pattern.findall(text))
    total_count = len(text.strip())

    if total_count == 0:
        return False

    return (allowed_count / total_count) >= threshold


RE_CLEAN = re.compile(r'[^a-zA-Z0-9\s\-\']')
RE_SPACE = re.compile(r'\s+')


def clean_text(text: str) -> str:
    text = text.lower()
    text = RE_CLEAN.sub(' ', text)
    text = RE_SPACE.sub(' ', text).strip()
    return text


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = nltk.word_tokenize(text)
    return [t for t in tokens if any(c.isalnum() for c in t)]


def remove_stopwords(tokens: List[str], stopwords: Optional[Set[str]] = None) -> List[str]:
    if stopwords is None:
        stopwords = STOPWORDS
    return [t for t in tokens if t not in stopwords]


def generate_word_shingles(tokens: List[str], k: int) -> Set[str]:
    n = len(tokens)
    if n < k:
        return set()
    return {" ".join(tokens[i:i+k]) for i in range(n - k + 1)}


def generate_char_shingles(text: str, k: int) -> Set[str]:
    if not text or len(text) < k:
        return set()
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def preprocess_document(text: str, k: int = 3) -> Set[str]:
    if not text or len(text.strip()) < 3:
        return set()

    if not is_english_text(text):
        return set()

    cleaned = clean_text(text)
    if not cleaned:
        return set()

    tokens = tokenize(cleaned)
    if not tokens:
        return set()

    tokens = remove_stopwords(tokens)
    if not tokens:
        return set()

    if len(tokens) >= k:
        return generate_word_shingles(tokens, k)
    else:
        return generate_char_shingles(cleaned, k)
