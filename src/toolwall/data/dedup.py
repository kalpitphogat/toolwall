"""Exact and near-duplicate detection with MinHash + LSH.

Public prompt-injection corpora are full of copies: the same payload resubmitted with a
changed name, or copied from one dataset into another. If one copy lands in train and
another in test, test accuracy measures memorisation, not detection. This module finds
those copies so the splitter can keep each cluster inside one split and the leak audit
can prove no test item has a near-twin in train.

Similarity is Jaccard over word 3-gram shingles (char 5-grams for very short texts).
MinHash estimates it with `num_perm` hash functions; LSH banding turns that into
candidate pairs, which are then verified against the exact Jaccard.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Sequence

import numpy as np

from .schema import normalize

_MERSENNE = np.uint64((1 << 61) - 1)
_MAX32 = np.uint64(0xFFFFFFFF)
_TOKEN = re.compile(r"\w+", re.UNICODE)


def shingles(text: str, k_words: int = 3, k_chars: int = 5) -> set[str]:
    words = _TOKEN.findall(text.lower())
    if len(words) >= k_words:
        return {" ".join(words[i : i + k_words]) for i in range(len(words) - k_words + 1)}
    s = normalize(text)
    if len(s) <= k_chars:
        return {s} if s else set()
    return {s[i : i + k_chars] for i in range(len(s) - k_chars + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _hash64(items: set[str]) -> np.ndarray:
    return np.fromiter(
        (int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") >> 3 for s in items),
        dtype=np.uint64,
        count=len(items),
    )


class MinHasher:
    def __init__(self, num_perm: int = 64, seed: int = 13):
        rng = np.random.default_rng(seed)
        # Universal hashing h(x) = (a*x + b) mod p, with a, b < 2^32 so a*x cannot overflow uint64 for x < 2^61.
        self.a = rng.integers(1, 1 << 32, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 32, size=num_perm, dtype=np.uint64)
        self.num_perm = num_perm

    def signature(self, sh: set[str]) -> np.ndarray:
        if not sh:
            return np.full(self.num_perm, _MAX32, dtype=np.uint64)
        x = _hash64(sh) & _MAX32                                   # 32-bit inputs keep a*x inside uint64
        hv = (np.outer(x, self.a) + self.b) % _MERSENNE           # (n_shingles, num_perm)
        return hv.min(axis=0)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def _lsh_buckets(sh: Sequence[set[str]], num_perm: int, bands: int) -> list[list[int]]:
    if num_perm % bands:
        raise ValueError("num_perm must be divisible by bands")
    rows = num_perm // bands
    mh = MinHasher(num_perm)
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    for i, s in enumerate(sh):
        sig = mh.signature(s)
        for b in range(bands):
            buckets[(b, sig[b * rows : (b + 1) * rows].tobytes())].append(i)
    return [m for m in buckets.values() if len(m) > 1]


def near_duplicate_pairs(
    texts: Sequence[str], threshold: float = 0.8, num_perm: int = 64, bands: int = 16
) -> list[tuple[int, int]]:
    """Index pairs whose exact shingle Jaccard is >= threshold (found via LSH candidates)."""
    sh = [shingles(t) for t in texts]
    pairs: set[tuple[int, int]] = set()
    checked: set[tuple[int, int]] = set()
    for members in _lsh_buckets(sh, num_perm, bands):
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                p = (members[x], members[y])
                if p in checked:
                    continue
                checked.add(p)
                if jaccard(sh[p[0]], sh[p[1]]) >= threshold:
                    pairs.add(p)
    return sorted(pairs)


def cluster_ids(texts: Sequence[str], threshold: float = 0.8) -> list[int]:
    """Assign each text a cluster id; exact duplicates (after normalisation) and near-duplicates share one."""
    uf = _UnionFind(len(texts))
    first_seen: dict[str, int] = {}
    for i, t in enumerate(texts):
        key = normalize(t)
        if key in first_seen:
            uf.union(first_seen[key], i)
        else:
            first_seen[key] = i
    for i, j in near_duplicate_pairs(texts, threshold=threshold):
        uf.union(i, j)
    return [uf.find(i) for i in range(len(texts))]


def drop_exact_duplicates(texts: Sequence[str]) -> list[int]:
    """Indices of the first occurrence of each normalised text."""
    seen: set[str] = set()
    keep = []
    for i, t in enumerate(texts):
        key = normalize(t)
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return keep


def leaking_indices(candidates: Sequence[str], reference: Sequence[str], threshold: float = 0.8) -> set[int]:
    """Indices into `candidates` that are near-duplicates of anything in `reference`."""
    ref_norm = {normalize(t) for t in reference}
    hits = {i for i, t in enumerate(candidates) if normalize(t) in ref_norm}
    offset = len(reference)
    sh = [shingles(t) for t in reference] + [shingles(t) for t in candidates]
    checked: set[tuple[int, int]] = set()
    for members in _lsh_buckets(sh, num_perm=64, bands=16):
        refs = [m for m in members if m < offset]
        cands = [m for m in members if m >= offset and (m - offset) not in hits]
        for j in cands:                       # only reference-vs-candidate pairs matter here
            for i in refs:
                if (i, j) in checked:
                    continue
                checked.add((i, j))
                if jaccard(sh[i], sh[j]) >= threshold:
                    hits.add(j - offset)
                    break
    return hits
