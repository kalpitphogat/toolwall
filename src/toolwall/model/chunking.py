"""Split long texts into overlapping token windows, shared by training and inference.

Tool outputs (web pages, emails, search results) are often longer than the encoder's
512-token window, and an injection can sit anywhere. Truncating would silently drop
the tail, so both training and scoring see the same overlapping windows; a document's
score is the max over its windows, and that window's character range is the evidence span.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    doc: int                   # index of the source text
    input_ids: list[int]
    attention_mask: list[int]
    char_span: tuple[int, int]  # characters of the source text this window covers


def chunk_texts(tokenizer, texts: list[str], max_length: int = 512, stride: int = 128) -> list[Chunk]:
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )
    out = []
    for i, doc in enumerate(enc["overflow_to_sample_mapping"]):
        offsets = [o for o in enc["offset_mapping"][i] if o[1] > o[0]]   # special tokens have (0, 0)
        span = (offsets[0][0], offsets[-1][1]) if offsets else (0, 0)
        out.append(Chunk(doc, enc["input_ids"][i], enc["attention_mask"][i], span))
    return out


def chunk_label(doc_label: int, doc_span: tuple[int, int] | None, chunk_span: tuple[int, int]) -> int | None:
    """Training label for one window, or None if it cannot be known.

    Benign document -> every window is benign. Injected document with a known payload span
    -> windows overlapping the payload are positive, the rest are benign context. Injected
    document without a span (real attack emails) -> only usable if it fits in one window.
    """
    if doc_label == 0:
        return 0
    if doc_span is None:
        return None
    s, e = doc_span
    return int(chunk_span[0] < e and s < chunk_span[1])
