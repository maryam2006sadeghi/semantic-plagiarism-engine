from dataclasses import dataclass
from typing import List, Set
import random
import zlib


@dataclass
class MinHashConfig:
    num_perm: int = 128
    seed: int = 42
    prime: int = (2 ** 61) - 1


class MinHash:
    def __init__(self, config: MinHashConfig = MinHashConfig()):

        self.config = config
        self.num_perm = config.num_perm
        self.prime = config.prime

        rng = random.Random(config.seed)

        self.a = [
            rng.randint(1, self.prime - 1)
            for _ in range(self.num_perm)
        ]

        self.b = [
            rng.randint(0, self.prime - 1)
            for _ in range(self.num_perm)
        ]

    @staticmethod
    def _shingle_to_int(shingle: str) -> int:

        return zlib.crc32(shingle.encode("utf-8")) & 0xFFFFFFFF

    def _hash(self, x: int, a: int, b: int) -> int:

        return (a * x + b) % self.prime

    def signature(self, shingles: Set[str]) -> List[int]:

        if not shingles:
            return [self.prime] * self.num_perm

        shingle_ids = [
            self._shingle_to_int(s)
            for s in shingles
        ]

        signature = []

        for a, b in zip(self.a, self.b):

            min_hash = self.prime

            for x in shingle_ids:

                h = self._hash(x, a, b)

                if h < min_hash:
                    min_hash = h

            signature.append(min_hash)

        return signature

    @staticmethod
    def similarity(sig1: List[int], sig2: List[int]) -> float:

        if len(sig1) != len(sig2):
            raise ValueError("Signature length mismatch.")

        if not sig1 or not sig2:
            return 0.0

        matches = sum(
            1
            for x, y in zip(sig1, sig2)
            if x == y
        )

        return matches / len(sig1)
