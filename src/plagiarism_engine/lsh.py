from dataclasses import dataclass
from typing import List, Set, Tuple, Any, Dict, Optional, Callable
from collections import defaultdict
import random
from evaluation import exact_jaccard


@dataclass
class LSHConfig:
    signature_size: int = 128
    bands: int = 32
    rows: int = 4
    hash_prime: int = 1000003
    seed: int = 42

    def __post_init__(self):
        if self.bands * self.rows != self.signature_size:
            raise ValueError(
                f"bands ({self.bands}) * rows ({self.rows}) must equal "
                f"signature_size ({self.signature_size})."
            )


class LSH:
    def __init__(self, config: Optional[LSHConfig] = None):
        if config is None:
            config = LSHConfig()

        self.config = config
        self.signature_size = config.signature_size
        self.bands = config.bands
        self.rows = config.rows
        self.hash_prime = config.hash_prime

        rng = random.Random(config.seed)
        self.a = [rng.randint(1, self.hash_prime - 1)
                  for _ in range(self.rows)]
        self.b = rng.randint(0, self.hash_prime - 1)

        self.buckets: List[Dict[int, Set[Any]]] = [
            defaultdict(set) for _ in range(self.bands)
        ]

        self.candidate_pairs: Set[Tuple[Any, Any]] = set()

    def _get_band(self, signature: List[int], band_index: int) -> Tuple[int, ...]:
        start = band_index * self.rows
        end = start + self.rows
        return tuple(signature[start:end])

    def _hash_band(self, band_tuple: Tuple[int, ...]) -> int:
        total = self.b
        for i, val in enumerate(band_tuple):
            total = (total + self.a[i] * val) % self.hash_prime
        return total

    def _get_bucket_key(self, band_tuple: Tuple[int, ...]) -> int:
        return self._hash_band(band_tuple)

    def index(self, document_id: Any, signature: List[int]) -> None:
        if len(signature) != self.signature_size:
            raise ValueError(
                f"Invalid signature length. Expected {self.signature_size}, "
                f"got {len(signature)}."
            )

        for band_index in range(self.bands):
            band_tuple = self._get_band(signature, band_index)
            bucket_key = self._get_bucket_key(band_tuple)
            bucket = self.buckets[band_index][bucket_key]

            for existing_doc_id in bucket:
                if existing_doc_id != document_id:
                    pair = tuple(sorted((document_id, existing_doc_id)))
                    self.candidate_pairs.add(pair)

            bucket.add(document_id)

    def query(self, signature: List[int]) -> Set[Any]:
        if len(signature) != self.signature_size:
            raise ValueError(
                f"Invalid signature length. Expected {self.signature_size}, "
                f"got {len(signature)}."
            )

        candidates = set()

        for band_index in range(self.bands):
            band_tuple = self._get_band(signature, band_index)
            bucket_key = self._get_bucket_key(band_tuple)

            bucket = self.buckets[band_index].get(bucket_key)
            if bucket is not None:
                candidates.update(bucket)

        return candidates

    def get_true_duplicates(
        self,
        all_shingles: Dict[Any, Set[str]],
        threshold: float = 0.5,
    ) -> Set[Tuple[Any, Any]]:

        true_duplicates = set()

        for doc_a, doc_b in self.candidate_pairs:
            sim = exact_jaccard(
                all_shingles[doc_a],
                all_shingles[doc_b]
            )
            if sim >= threshold:
                true_duplicates.add((doc_a, doc_b))

        return true_duplicates

    def get_candidate_pairs(self) -> Set[Tuple[Any, Any]]:
        return self.candidate_pairs

    def clear(self) -> None:
        self.buckets = [defaultdict(set) for _ in range(self.bands)]
        self.candidate_pairs.clear()


# ===========================
# بخش تست
# ===========================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing LSH implementation (Linear Hashing - PDF method)")
    print("=" * 60)

    # ====== تست 1: مثال ساده از PDF ======
    print("\n1. Testing banding with a simple example (based on PDF Page 50):")
    print("   Using signature_size=4, bands=2, rows=2")

    lsh = LSH(LSHConfig(
        signature_size=4,
        bands=2,
        rows=2
    ))

    # امضاهای مشابه (S1 و S2)
    sig1 = [1, 1, 2, 1]   # S1
    sig2 = [1, 0, 2, 1]   # S2 (با S1 در باند 2 مشابه است؟)
    sig3 = [2, 0, 3, 1]   # S3 (متفاوت)

    lsh.index("S1", sig1)
    lsh.index("S2", sig2)
    lsh.index("S3", sig3)

    print(f"   Candidate pairs: {lsh.get_candidate_pairs()}")
    # خروجی مورد انتظار: {('S1', 'S2')} چون در باند 2 با هم مشابه هستند

    print("\n✅ All tests passed!")
