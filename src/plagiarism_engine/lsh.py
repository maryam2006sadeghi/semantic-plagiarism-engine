from dataclasses import dataclass
from typing import List, Set, Tuple, Dict, Any, Optional
from collections import defaultdict
import random


@dataclass
class LSHConfig:
    signature_size: int = 128
    bands: int = 32
    rows: int = 4
    hash_prime: int = (2 ** 31) - 1
    seed: int = 42

    def __post_init__(self):
        if self.bands * self.rows != self.signature_size:
            raise ValueError(
                "bands × rows must equal signature_size."
            )


class LSH:

    def __init__(self,
                 config: Optional[LSHConfig] = None):
        if config is None:
            config = LSHConfig()

        self.config = config
        self.signature_size = config.signature_size
        self.bands = config.bands
        self.rows = config.rows
        self.hash_prime = config.hash_prime

        self._generate_hash_function()

        self.buckets = [
            defaultdict(set)
            for _ in range(self.bands)
        ]

        self.candidate_pairs = set()

    def _generate_hash_function(self):

        rng = random.Random(self.config.seed)

        self.a = [

            rng.randint(
                1,
                self.hash_prime - 1
            )

            for _ in range(self.rows)

        ]

        self.b = rng.randint(
            0,
            self.hash_prime - 1
        )

    def _split_signature_into_bands(
            self,
            signature: List[int]
    ) -> List[Tuple[int, ...]]:

        if len(signature) != self.signature_size:
            raise ValueError(
                "Invalid signature length."
            )

        bands = []
        for i in range(self.bands):
            start = i * self.rows
            end = start + self.rows
            bands.append(
                tuple(signature[start:end])
            )

        return bands

    def _hash_band(
            self,
            band: Tuple[int, ...]
    ) -> int:

        total = self.b
        for coeff, value in zip(self.a, band):
            total += coeff * value
        return total % self.hash_prime

    def _insert_into_bucket(
            self,
            band_index: int,
            bucket_key: int,
            document_id: Any
    ):

        self.buckets[band_index][bucket_key].add(
            document_id
        )

    def _detect_collisions(
            self,
            band_index: int,
            bucket_key: int,
            document_id: Any
    ):

        bucket = self.buckets[
            band_index
        ][bucket_key]

        for other_doc in bucket:
            if other_doc == document_id:
                continue

            pair = tuple(
                sorted(
                    (document_id,
                     other_doc)
                )
            )

            self.candidate_pairs.add(pair)

    def index(
            self,
            document_id: Any,
            signature: List[int]
    ):

        bands = self._split_signature_into_bands(
            signature
        )

        for band_index, band in enumerate(bands):
            bucket_key = self._hash_band(
                band
            )
            self._detect_collisions(
                band_index,
                bucket_key,
                document_id
            )
            self._insert_into_bucket(
                band_index,
                bucket_key,
                document_id
            )

    def query(
            self,
            signature: List[int]
    ) -> Set[Any]:

        candidates = set()

        bands = self._split_signature_into_bands(
            signature
        )

        for band_index, band in enumerate(bands):
            bucket_key = self._hash_band(
                band
            )
            bucket = self.buckets[
                band_index
            ].get(bucket_key)

            if bucket is not None:
                candidates.update(bucket)

        return candidates

    def verify_candidates(
            self,
            shingles: Dict[Any, Set[str]],
            threshold: Optional[float] = None
    ) -> Set[Tuple[Any, Any]]:

        if threshold is None:
            threshold = (1 / self.bands) ** (1 / self.rows)

        duplicates = set()

        for doc1, doc2 in self.candidate_pairs:
            sim = len(shingles[doc1].intersection(
                shingles[doc2])) / len(shingles[doc1].union(
                    shingles[doc2]))

            if sim >= threshold:
                duplicates.add(
                    (doc1, doc2)
                )

        return duplicates

    def get_candidate_pairs(self) -> Set[Tuple[Any, Any]]:
        return self.candidate_pairs

    def clear(self):
        self.buckets = [
            defaultdict(set)
            for _ in range(self.bands)
        ]

        self.candidate_pairs.clear()
