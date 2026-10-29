"""Loaders for the public corpora. Each returns plain Records; grouping and splitting happen in build.py.

| source         | HF id                                   | licence      | used as                                  |
|----------------|-----------------------------------------|--------------|------------------------------------------|
| llmail         | microsoft/llmail-inject-challenge       | MIT          | real indirect injections (emails)        |
| gandalf        | Lakera/gandalf_ignore_instructions      | MIT          | injection payloads                       |
| deepset        | deepset/prompt-injections               | Apache-2.0   | payloads + benign prompts                |
| spml           | reshabhs/SPML_Chatbot_Prompt_Injection  | MIT          | payloads + benign prompts                |
| dolly          | databricks/databricks-dolly-15k         | CC BY-SA 3.0 | benign inserts (instructions) + carriers |
| cnn            | abisee/cnn_dailymail (3.0.0)            | Apache-2.0   | carriers                                 |
| ood_safeguard  | xTRam1/safe-guard-prompt-injection test | (none given) | held-out evaluation only                 |
| ood_jailbreak  | jackhhao/jailbreak-classification test  | Apache-2.0   | held-out evaluation only                 |

The two ood_* sets are never used for training or model selection. They are the same
2,322 prompts used to show that a widely used public injection classifier falls to
chance on general injections, so results here are directly comparable.
"""
from __future__ import annotations

import itertools
import random
from typing import Iterable

from .schema import DIRECT, INDIRECT_REAL, Record


def _ds(name: str, split: str, config: str | None = None, streaming: bool = False):
    from datasets import load_dataset  # optional dependency: pip install toolwall[data]

    return load_dataset(name, config, split=split, streaming=streaming)


def _clean(s) -> str:
    return (s or "").strip()


def _render_email(subject: str, body: str, rng: random.Random) -> str:
    # Same layout as synth._render("email"), so benign synthetic emails share the format.
    from .synth import _render

    return _render("email", subject or "(no subject)", body, rng)


# ---------------------------------------------------------------- real indirect injections

def load_llmail(per_team_cap: int = 200, total_cap: int = 10_000, max_chars: int = 2_500, seed: int = 0) -> list[Record]:
    """Attacker-written emails from the LLMail-Inject challenge; every row is an attack attempt.

    group = attacker team, so held-out splits contain only attackers never seen in training.
    Bodies longer than `max_chars` are skipped rather than truncated, because truncation
    could cut the injection off and leave a positive label on benign-looking text.
    """
    rng = random.Random(seed)
    by_team: dict[str, list[tuple[str, str]]] = {}
    seen: set[str] = set()
    for phase in ("Phase1", "Phase2"):
        for row in _ds("microsoft/llmail-inject-challenge", phase):
            body, subject = _clean(row["body"]), _clean(row["subject"])
            if not body or len(body) > max_chars:
                continue
            key = " ".join(body.lower().split())
            if key in seen:
                continue
            seen.add(key)
            by_team.setdefault(_clean(row["team_id"]) or "unknown", []).append((subject, body))

    rows: list[tuple[str, str, str]] = []
    for team, items in by_team.items():
        rng.shuffle(items)
        rows.extend((team, s, b) for s, b in items[:per_team_cap])
    rng.shuffle(rows)
    return [
        Record(text=_render_email(s, b, rng), label=1, source="llmail", group=f"team:{t}", kind=INDIRECT_REAL,
               meta={"insert": b})
        for t, s, b in rows[:total_cap]
    ]


# ---------------------------------------------------------------- short prompts (payloads / benign)

def load_payloads() -> list[Record]:
    out: list[Record] = []
    for split in ("train", "validation", "test"):
        out += [Record(_clean(r["text"]), 1, "gandalf", "") for r in _ds("Lakera/gandalf_ignore_instructions", split)]
    for split in ("train", "test"):
        out += [Record(_clean(r["text"]), 1, "deepset", "") for r in _ds("deepset/prompt-injections", split)
                if int(r["label"]) == 1]
    out += [Record(_clean(r["User Prompt"]), 1, "spml", "") for r in _ds("reshabhs/SPML_Chatbot_Prompt_Injection", "train")
            if int(r["Prompt injection"]) == 1]
    return [r for r in out if r.text]


def load_benign_prompts(dolly_cap: int = 8_000, seed: int = 0) -> list[Record]:
    out: list[Record] = []
    for split in ("train", "test"):
        out += [Record(_clean(r["text"]), 0, "deepset", "") for r in _ds("deepset/prompt-injections", split)
                if int(r["label"]) == 0]
    out += [Record(_clean(r["User Prompt"]), 0, "spml", "") for r in _ds("reshabhs/SPML_Chatbot_Prompt_Injection", "train")
            if int(r["Prompt injection"]) == 0]
    dolly = [r for r in _ds("databricks/databricks-dolly-15k", "train") if _clean(r["instruction"])]
    random.Random(seed).shuffle(dolly)
    out += [Record(_clean(r["instruction"]), 0, "dolly", "", meta={"category": r["category"]}) for r in dolly[:dolly_cap]]
    return [r for r in out if r.text]


# ---------------------------------------------------------------- carriers

def _cut(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind(".\n"))
    return cut[: end + 1] if end > max_chars // 2 else cut


def load_carriers(cnn_cap: int = 6_000, max_chars: int = 1_400) -> list[Record]:
    """Benign documents that inserts get embedded into. label/group are filled in later."""
    out: list[Record] = []
    for r in _ds("databricks/databricks-dolly-15k", "train"):
        ctx = _clean(r["context"])
        if 200 <= len(ctx):
            out.append(Record(_cut(ctx, max_chars), 0, "dolly_ctx", "", kind="carrier",
                              meta={"title": " ".join(_clean(r["instruction"]).split()[:8])}))
    stream = _ds("abisee/cnn_dailymail", "train", config="3.0.0", streaming=True)
    for r in itertools.islice(stream, cnn_cap):
        art = _clean(r["article"]).replace("(CNN) -- ", "")
        if len(art) >= 300:
            title = _clean(r["highlights"]).split("\n")[0][:90]
            out.append(Record(_cut(art, max_chars), 0, "cnn", "", kind="carrier", meta={"title": title}))
    return out


# ---------------------------------------------------------------- held-out evaluation sets

def load_ood() -> dict[str, list[Record]]:
    sg = [Record(_clean(r["text"]), int(r["label"]), "safeguard", "ood", split="ood_safeguard")
          for r in _ds("xTRam1/safe-guard-prompt-injection", "test")]
    jb = [Record(_clean(r["prompt"]), int(r["type"] == "jailbreak"), "jailbreak_cls", "ood", split="ood_jailbreak")
          for r in _ds("jackhhao/jailbreak-classification", "test")]
    return {"ood_safeguard": [r for r in sg if r.text], "ood_jailbreak": [r for r in jb if r.text]}


def flatten(groups: Iterable[list[Record]]) -> list[Record]:
    return [r for g in groups for r in g]
