"""`toolwall data build`: assemble the leak-free train/val/test/ood splits.

Steps
  1. Load held-out OOD sets first; everything else is later purged against them.
  2. Cluster all short prompts (payloads + benign) by near-duplicate similarity. A cluster
     is one group, so copies of a payload can never straddle splits. Clusters whose members
     disagree on the label are dropped as ambiguous.
  3. Split groups deterministically (70/15/15). Carriers are split the same way.
  4. Within each split only, synthesise indirect examples (payload or benign insert in a
     carrier), add benign emails to balance the all-email LLMail positives, and add the
     short prompts themselves as direct examples.
  5. Purge near-duplicates of held-out items from val/train, run an independent audit,
     write jsonl files plus a manifest with counts and audit numbers.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import sources
from .dedup import cluster_ids, leaking_indices
from .schema import BENIGN_DOC, DIRECT, INDIRECT_SYNTH, Record, normalize, write_jsonl
from .split import SPLITS, assign_splits, audit, purge_leaks, summarize
from .synth import compose


@dataclass
class BuildConfig:
    out_dir: str = "data/processed"
    seed: int = 0
    near_dup_threshold: float = 0.8
    synth_per_split: dict = field(default_factory=lambda: {"train": 20_000, "val": 2_500, "test": 2_500})
    direct_cap_per_label: dict = field(default_factory=lambda: {"train": 6_000, "val": 800, "test": 800})
    empty_negative_fraction: float = 0.3      # negatives that are a bare carrier with no insert
    llmail_per_team_cap: int = 200
    llmail_total_cap: int = 10_000


def _cluster_and_split(records: list[Record], cfg: BuildConfig) -> list[Record]:
    ids = cluster_ids([r.text for r in records], threshold=cfg.near_dup_threshold)
    labels: dict[int, set[int]] = defaultdict(set)
    members: dict[int, list[str]] = defaultdict(list)
    for cid, r in zip(ids, records):
        labels[cid].add(r.label)
        members[cid].append(normalize(r.text))
    # Content-based group ids: a cluster is named after its smallest member, so adding a new
    # source in a later build does not move existing items between train and test.
    name = {cid: "c" + hashlib.blake2b(min(m).encode(), digest_size=8).hexdigest() for cid, m in members.items()}
    kept = []
    for cid, r in zip(ids, records):
        if len(labels[cid]) == 1:            # drop clusters with conflicting labels
            r.group = name[cid]
            kept.append(r)
    # one representative per cluster: near-copies add no information, only leakage risk
    seen: set[str] = set()
    uniq = []
    for r in kept:
        if r.group not in seen:
            seen.add(r.group)
            uniq.append(r)
    assign_splits(uniq, cfg.seed)
    return uniq


def _synthesize(split: str, n: int, payloads: list[Record], inserts: list[Record], carriers: list[Record],
                cfg: BuildConfig, rng: random.Random, force_wrapper: str | None = None) -> list[Record]:
    out = []
    for i in range(n):
        carrier = rng.choice(carriers)
        positive = (i % 2 == 0) if force_wrapper is None else False
        if positive:
            ins = rng.choice(payloads)
        elif rng.random() < cfg.empty_negative_fraction:
            ins = None
        else:
            ins = rng.choice(inserts)
        if ins is None:
            c = compose(carrier.text, carrier.meta.get("title", ""), "", rng, wrapper=force_wrapper)
            text = c.text.replace("\n\n\n\n", "\n\n")
            out.append(Record(text, 0, f"synth/{carrier.source}", carrier.group, kind=BENIGN_DOC, split=split,
                              meta={"wrapper": c.wrapper, "carrier": carrier.id}))
            continue
        c = compose(carrier.text, carrier.meta.get("title", ""), ins.text, rng, wrapper=force_wrapper)
        out.append(Record(
            c.text, ins.label, f"synth/{ins.source}", ins.group,
            kind=INDIRECT_SYNTH if ins.label else BENIGN_DOC, split=split, span=c.span,
            meta={"insert": ins.text, "wrapper": c.wrapper, "concealment": c.concealment,
                  "position": c.position, "carrier": carrier.id},
        ))
    return out


def build(cfg: BuildConfig) -> dict:
    rng = random.Random(cfg.seed)
    out_dir = Path(cfg.out_dir)

    ood = sources.load_ood()
    ood_texts = [r.text for rs in ood.values() for r in rs]
    prompts = _cluster_and_split(sources.load_payloads() + sources.load_benign_prompts(seed=cfg.seed), cfg)
    # A prompt that is a near-copy of a held-out item must not reach training, even pasted into a carrier.
    bad = leaking_indices([r.text for r in prompts], ood_texts, cfg.near_dup_threshold)
    dropped_vs_ood = {"prompts": len(bad)}
    prompts = [r for i, r in enumerate(prompts) if i not in bad]

    carriers = sources.load_carriers()
    for c in carriers:
        c.group = f"carrier:{c.id}"
    assign_splits(carriers, cfg.seed + 1)   # independent of the prompt split

    llmail = sources.load_llmail(cfg.llmail_per_team_cap, cfg.llmail_total_cap, seed=cfg.seed)
    bad = leaking_indices([r.meta["insert"] for r in llmail], ood_texts, cfg.near_dup_threshold)
    dropped_vs_ood["llmail"] = len(bad)
    llmail = [r for i, r in enumerate(llmail) if i not in bad]
    assign_splits(llmail, cfg.seed + 2)

    by_split: dict[str, list[Record]] = {s: [] for s in SPLITS}
    for split in SPLITS:
        pay = [r for r in prompts if r.split == split and r.label == 1]
        ben = [r for r in prompts if r.split == split and r.label == 0]
        car = [r for r in carriers if r.split == split]
        mail = [r for r in llmail if r.split == split]

        # payloads drawn for synthesis also include real attack emails from this split's teams
        mail_payloads = [Record(r.meta["insert"], 1, "llmail", r.group) for r in mail]
        synth = _synthesize(split, cfg.synth_per_split[split], pay + mail_payloads, ben, car, cfg, rng)
        # LLMail positives are all emails; add as many benign emails so the format carries no label signal
        benign_mail = _synthesize(split, len(mail), pay, ben, car, cfg, rng, force_wrapper="email")

        cap = cfg.direct_cap_per_label[split]
        direct = rng.sample(pay, min(cap, len(pay))) + rng.sample(ben, min(cap, len(ben)))
        for r in direct:
            r.kind = DIRECT

        rows = synth + benign_mail + mail + direct
        rng.shuffle(rows)
        by_split[split] = rows

    by_split.update(ood)
    removed = purge_leaks(by_split, cfg.near_dup_threshold)
    report = audit(by_split, cfg.near_dup_threshold)

    for name, rows in by_split.items():
        write_jsonl(out_dir / f"{name}.jsonl", rows)
    manifest = {
        "config": cfg.__dict__,
        "splits": summarize(by_split),
        "dropped_before_synthesis_as_near_duplicates_of_ood": dropped_vs_ood,
        "purged_as_near_duplicates_of_heldout": removed,
        "leak_audit": report,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
