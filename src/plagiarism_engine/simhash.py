from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import Counter
import hashlib
import math


@dataclass
class SimHashConfig:

    hash_bits: int = 64


class SimHash:

    def __init__(self,
                 config: Optional[SimHashConfig] = None):
        if config is None:
            config = SimHashConfig()

        if config.hash_bits > 64:
            raise ValueError(
                f"hash_bits must be <= 64, got {config.hash_bits}")
        self.config = config
        self.hash_bits = config.hash_bits

    @staticmethod
    def term_frequency(tokens: List[str]) -> Dict[str, float]:
        counts = Counter(tokens)
        tf = {}

        for token, count in counts.items():
            tf[token] = 1.0 + math.log10(count)

        return tf

    @staticmethod
    def inverse_document_frequency(
            corpus_tokens: List[List[str]]
    ) -> Dict[str, float]:
        N = len(corpus_tokens)
        df = Counter()

        for doc in corpus_tokens:

            for token in set(doc):
                df[token] += 1

        idf = {}
        for token, freq in df.items():

            idf[token] = math.log10(
                N / freq
            )

        return idf

    @staticmethod
    def tfidf(
            tokens: List[str],
            idf: Dict[str, float]
    ) -> Dict[str, float]:

        tf = SimHash.term_frequency(tokens)
        weights = {}

        for token, tf_value in tf.items():
            weights[token] = tf_value * idf.get(token, 0.0)

        return weights

    @staticmethod
    def hash64(token: str) -> int:
        digest = hashlib.md5(
            token.encode("utf-8")
        ).digest()

        return int.from_bytes(
            digest[:8],
            "big"
        )

    def signature(
            self,
            tokens: List[str],
            idf: Dict[str, float]
    ) -> int:
        weights = self.tfidf(tokens, idf)

        vector = [0.0] * self.hash_bits

        for token, weight in weights.items():
            h = self.hash64(token)
            for bit in range(self.hash_bits):
                if (h >> bit) & 1:
                    vector[bit] += weight
                else:
                    vector[bit] -= weight

        fingerprint = 0
        for bit in range(self.hash_bits):
            if vector[bit] > 0:
                fingerprint |= (1 << bit)
        return fingerprint

    @staticmethod
    def hamming_distance(
            hash1: int,
            hash2: int
    ) -> int:

        return (hash1 ^ hash2).bit_count()

    def similarity(
            self,
            hash1: int,
            hash2: int
    ) -> float:

        distance = self.hamming_distance(
            hash1,
            hash2
        )

        return 1.0 - (
            distance /
            self.hash_bits
        )
