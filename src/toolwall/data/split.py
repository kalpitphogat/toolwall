"""Group-aware splitting and the train/test leak audit."""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Sequence

from .dedup import leaking_indices
from .schema import Record

SPLITS = ("train", "val", "test")


def split_for_group(group: str, seed: int = 0, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> str:
    """Deterministic split for a group id: same group -> same split, regardless of row order."""
    h = int.from_bytes(hashlib.blake2b(f"{seed}:{group}".encode(), digest_size=8).digest(), "little")
    u = (h % 1_000_000) / 1_000_000
    if u < ratios[0]:
        return "train"
    if u < ratios[0] + ratios[1]:
        return "val"
    return "test"


def assign_splits(records: Sequence[Record], seed: int = 0) -> None:
    for r in records:
        if not r.split:
            r.split = split_for_group(r.group, seed)


def purge_leaks(by_split: dict[str, list[Record]], threshold: float = 0.8) -> dict[str, int]:
    """Remove near-duplicates of held-out items from earlier splits.

    Order of precedence: test and every ood_* split are untouchable; val is purged against
    them; train is purged against val + test + ood. Returns how many records each split lost.
    """
    held_out = [r.text for name, rs in by_split.items() if name == "test" or name.startswith("ood") for r in rs]
    removed: dict[str, int] = {}

    if "val" in by_split:
        bad = leaking_indices([r.text for r in by_split["val"]], held_out, threshold)
        by_split["val"] = [r for i, r in enumerate(by_split["val"]) if i not in bad]
        removed["val"] = len(bad)

    reference = held_out + [r.text for r in by_split.get("val", [])]
    bad = leaking_indices([r.text for r in by_split["train"]], reference, threshold)
    by_split["train"] = [r for i, r in enumerate(by_split["train"]) if i not in bad]
    removed["train"] = len(bad)
    return removed


def audit(by_split: dict[str, list[Record]], threshold: float = 0.8) -> dict:
    """Independent re-check after purging; every number here should be zero."""
    train = [r.text for r in by_split["train"]]
    report: dict = {"threshold": threshold}
    for name, rs in by_split.items():
        if name == "train":
            continue
        report[f"{name}_items_with_train_near_duplicate"] = len(leaking_indices([r.text for r in rs], train, threshold))
    # A payload pasted into a different carrier has low whole-text similarity, so synthetic
    # records are also compared on the inserted text itself.
    train_inserts = [r.meta["insert"] for r in by_split["train"] if r.meta.get("insert")]
    for name, rs in by_split.items():
        if name == "train":
            continue
        # held-out sets have no inserts; compare their whole texts against training inserts instead
        if name.startswith("ood"):
            ins = [r.text for r in rs]
        else:
            ins = [r.meta["insert"] for r in rs if r.meta.get("insert")]
        if ins:
            report[f"{name}_inserts_with_train_near_duplicate"] = len(leaking_indices(ins, train_inserts, threshold))
    train_groups = {(r.source, r.group) for r in by_split["train"]}
    for name in ("val", "test"):
        if name in by_split:
            report[f"{name}_groups_shared_with_train"] = sum(
                1 for g in {(r.source, r.group) for r in by_split[name]} if g in train_groups
            )
    return report


def summarize(by_split: dict[str, list[Record]]) -> dict:
    out = {}
    for name, rs in by_split.items():
        out[name] = {
            "n": len(rs),
            "positives": sum(r.label for r in rs),
            "by_kind": dict(Counter(r.kind for r in rs)),
            "by_source": dict(Counter(r.source for r in rs)),
        }
    return out
