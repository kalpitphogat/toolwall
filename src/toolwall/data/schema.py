"""The single record type every stage of the data pipeline passes around."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# kind values
DIRECT = "direct"                  # a prompt typed straight at the model
INDIRECT_REAL = "indirect_real"    # attacker-written content delivered through a tool (LLMail emails)
INDIRECT_SYNTH = "indirect_synth"  # payload embedded into a carrier document by synth.py
BENIGN_DOC = "benign_doc"          # carrier document, optionally with a benign insert


@dataclass
class Record:
    text: str
    label: int                     # 1 = contains an injection, 0 = benign
    source: str                    # dataset of origin, e.g. "llmail", "dolly"
    group: str                     # unit that must never straddle splits (attacker team, payload cluster, ...)
    kind: str = DIRECT
    split: str = ""                # train / val / test / ood_*
    span: tuple[int, int] | None = None   # char offsets of the inserted payload (synthetic records only)
    meta: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return hashlib.blake2b(f"{self.source}\x00{self.text}".encode(), digest_size=8).hexdigest()

    def to_json(self) -> str:
        d = asdict(self)
        d["id"] = self.id
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        d = {k: v for k, v in d.items() if k != "id"}
        if d.get("span") is not None:
            d["span"] = tuple(d["span"])
        return cls(**d)


_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Canonical form used for exact-duplicate detection (case and whitespace insensitive)."""
    return _WS.sub(" ", text).strip().lower()


def write_jsonl(path: str | Path, records: Iterable[Record]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Record]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Record.from_dict(json.loads(line))
