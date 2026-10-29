"""Scorers: map a text to an injection probability plus the window that produced it.

Every consumer (eval harness, FastAPI service, MCP gateway) goes through `Scorer.scan`,
so reported benchmark numbers describe exactly the code path that runs in production.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .calibrate import sigmoid


@dataclass
class Scan:
    score: float                       # calibrated probability that the text contains an injection
    span: tuple[int, int] | None       # characters of the highest-scoring window
    windows: int                       # how many windows were scored
    latency_ms: float


class Scorer(Protocol):
    name: str

    def scan_many(self, texts: Sequence[str]) -> list[Scan]: ...


def scan(scorer: Scorer, text: str) -> Scan:
    return scorer.scan_many([text])[0]


class TransformerScorer:
    """Fine-tuned sequence classifier with overlapping windows and temperature calibration.

    Also loads public checkpoints for comparison: pass `positive_label` (e.g. "INJECTION")
    when the checkpoint has no calibration.json of its own.
    """

    def __init__(self, model_dir: str, device: str | None = None, batch_size: int = 16,
                 max_length: int = 512, stride: int = 128, positive_label: str | None = None, name: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # explicit fp32: transformers >= 5 would otherwise load fp16 weights, which are slow or unsupported on CPU
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32).to(self.device).eval()
        self.batch_size, self.max_length, self.stride = batch_size, max_length, stride
        self.name = name or Path(model_dir).name

        cal_path = Path(model_dir) / "calibration.json"
        self.calibration = json.loads(cal_path.read_text()) if cal_path.exists() else {}
        self.temperature = float(self.calibration.get("temperature", 1.0))
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        if positive_label is not None and positive_label.isdigit():
            self.pos = int(positive_label)
        elif positive_label is not None:
            matches = [i for i, lab in id2label.items() if lab.upper() == positive_label.upper()]
            if not matches:
                raise ValueError(f"label {positive_label!r} not in {id2label}")
            self.pos = matches[0]
        else:
            self.pos = int(self.calibration.get("positive_index", 1))

    @property
    def thresholds(self) -> dict[str, float]:
        return self.calibration.get("thresholds", {})

    def scan_many(self, texts: Sequence[str]) -> list[Scan]:
        from .chunking import chunk_texts

        t0 = time.perf_counter()
        texts = [t if t.strip() else " " for t in texts]
        chunks = chunk_texts(self.tokenizer, list(texts), self.max_length, self.stride)
        margins = np.empty(len(chunks))
        pad = self.tokenizer.pad_token_id or 0
        order = sorted(range(len(chunks)), key=lambda i: len(chunks[i].input_ids))  # less padding per batch
        with self.torch.inference_mode():
            for b in range(0, len(order), self.batch_size):
                idx = order[b : b + self.batch_size]
                width = max(len(chunks[i].input_ids) for i in idx)
                ids = [chunks[i].input_ids + [pad] * (width - len(chunks[i].input_ids)) for i in idx]
                mask = [chunks[i].attention_mask + [0] * (width - len(chunks[i].attention_mask)) for i in idx]
                with self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16, enabled=self.device.startswith("cuda")):
                    logits = self.model(
                        input_ids=self.torch.tensor(ids, device=self.device),
                        attention_mask=self.torch.tensor(mask, device=self.device),
                    ).logits.float()
                neg = 1 - self.pos if logits.shape[-1] == 2 else None
                m = logits[:, self.pos] - (logits[:, neg] if neg is not None else 0)
                margins[idx] = m.cpu().numpy()
        probs = sigmoid(margins / self.temperature)
        per_doc_ms = (time.perf_counter() - t0) * 1000 / max(1, len(texts))

        best: dict[int, int] = {}
        counts: dict[int, int] = {}
        for i, c in enumerate(chunks):
            counts[c.doc] = counts.get(c.doc, 0) + 1
            if c.doc not in best or probs[i] > probs[best[c.doc]]:
                best[c.doc] = i
        return [Scan(float(probs[best[d]]), chunks[best[d]].char_span, counts[d], per_doc_ms) for d in range(len(texts))]


class KeywordScorer:
    """Deliberately simple lexical baseline, used only as a reference point in evaluation."""

    name = "keyword-baseline"
    _PATTERNS = [re.compile(p, re.I) for p in (
        r"\bignore\b.{0,30}\b(instruction|prompt|rule)s?\b",
        r"\b(disregard|forget)\b.{0,30}\b(above|previous|prior|earlier)\b",
        r"\bsystem prompt\b",
        r"\byou are now\b",
        r"\b(reveal|print|output|leak)\b.{0,40}\b(password|secret|key|token|prompt)\b",
        r"\b(send|forward|email)\b.{0,60}\bto\b.{0,40}@",
        r"\bdo not (tell|inform|mention)\b",
        r"\bnew (instructions|task|role)\b",
    )]

    def scan_many(self, texts: Sequence[str]) -> list[Scan]:
        out = []
        for t in texts:
            t0 = time.perf_counter()
            hits = [m.span() for p in self._PATTERNS if (m := p.search(t))]
            score = 1 - 0.5 ** len(hits)          # 0, .5, .75, ... saturating with more independent cues
            out.append(Scan(score, hits[0] if hits else None, 1, (time.perf_counter() - t0) * 1000))
        return out


class StubScorer:
    """Deterministic scorer for tests and for running the gateway without model weights."""

    name = "stub"

    def __init__(self, triggers: Sequence[str] = ("ignore previous instructions",), score: float = 0.99):
        self.triggers = [t.lower() for t in triggers]
        self.hit_score = score

    def scan_many(self, texts: Sequence[str]) -> list[Scan]:
        out = []
        for t in texts:
            low = t.lower()
            pos = next((low.find(trig) for trig in self.triggers if trig in low), -1)
            span = (pos, pos + 20) if pos >= 0 else None
            out.append(Scan(self.hit_score if pos >= 0 else 0.01, span, 1, 0.0))
        return out


def load_scorer(spec: dict) -> Scorer:
    """Build a scorer from a config dict: {"type": "transformer"|"keyword"|"stub", ...}."""
    kind = spec.get("type", "transformer")
    if kind == "transformer":
        return TransformerScorer(spec["model_dir"], device=spec.get("device"), positive_label=spec.get("positive_label"))
    if kind == "keyword":
        return KeywordScorer()
    if kind == "stub":
        return StubScorer(spec.get("triggers", ("ignore previous instructions",)))
    raise ValueError(f"unknown scorer type {kind!r}")
