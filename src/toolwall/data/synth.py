"""Build indirect-injection examples by placing an insert inside a carrier document.

The rule that matters: every wrapper (web page, email, search JSON, ...) and every
concealment style (HTML comment, footer, quoted note, ...) is used for BOTH labels.
A positive is a carrier + injection payload; a negative is the same kind of carrier +
a benign insert (often itself an imperative, e.g. "Please send the report by Friday").
So formatting and position carry no label signal, and the classifier has to look at
what the inserted text asks for and whom it addresses.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

WRAPPERS = ("plain", "web", "markdown", "email", "search_json", "chat", "ticket")
CONCEALMENTS = ("paragraph", "html_comment", "footer", "quote", "inline", "bracket_note")
POSITIONS = ("start", "middle", "end")

_NAMES = ("Priya Shah", "Daniel Okafor", "Mei Lin", "Arjun Rao", "Sofia Rossi", "Tom Becker",
          "Aisha Khan", "Lucas Martin", "Nora Ali", "Ravi Menon", "Emma Clarke", "Kenji Sato")
_DOMAINS = ("example.com", "mail.example.org", "corp.example.net", "uni.example.edu")
_CHANNELS = ("#general", "#eng-infra", "#support", "#random", "#finance")
_NOTE_LABELS = ("Note", "Reminder", "FYI", "Update", "Editor's note", "PS")


@dataclass(frozen=True)
class Composed:
    text: str
    span: tuple[int, int]        # where the rendered insert sits inside `text`
    wrapper: str
    concealment: str
    position: str


def _conceal(insert: str, style: str, rng: random.Random) -> str:
    if style == "paragraph":
        return insert
    if style == "html_comment":
        return f"<!-- {insert} -->"
    if style == "footer":
        return f"--\n{insert}"
    if style == "quote":
        return "\n".join("> " + line for line in insert.splitlines() or [""])
    if style == "inline":
        return insert
    if style == "bracket_note":
        return f"[{rng.choice(_NOTE_LABELS)}: {insert}]"
    raise ValueError(style)


def _split_sentences(text: str) -> list[str]:
    parts, cur = [], []
    for tok in text.replace("\n", " \n ").split(" "):
        cur.append(tok)
        if tok.endswith((".", "!", "?")) and len(" ".join(cur)) > 40:
            parts.append(" ".join(cur).strip())
            cur = []
    if cur and " ".join(cur).strip():
        parts.append(" ".join(cur).strip())
    return parts or [text]


def _place(body: str, piece: str, position: str, inline: bool, rng: random.Random) -> str:
    sents = _split_sentences(body)
    if position == "start":
        idx = 0
    elif position == "end":
        idx = len(sents)
    else:
        idx = rng.randint(1, max(1, len(sents) - 1)) if len(sents) > 1 else 1
    sep = " " if inline else "\n\n"
    before, after = " ".join(sents[:idx]).strip(), " ".join(sents[idx:]).strip()
    return sep.join(p for p in (before, piece, after) if p)


def _render(wrapper: str, title: str, body: str, rng: random.Random) -> str:
    if wrapper == "plain":
        return body
    if wrapper == "web":
        paras = "\n".join(f"<p>{p}</p>" for p in body.split("\n\n"))
        return f"<html><head><title>{title}</title></head>\n<body>\n<h1>{title}</h1>\n{paras}\n</body></html>"
    if wrapper == "markdown":
        return f"# {title}\n\n{body}\n"
    if wrapper == "email":
        sender = rng.choice(_NAMES)
        addr = sender.lower().replace(" ", ".") + "@" + rng.choice(_DOMAINS)
        return f"From: {sender} <{addr}>\nSubject: {title}\n\n{body}\n\nBest,\n{sender.split()[0]}"
    if wrapper == "search_json":
        results = [{"title": title, "url": f"https://{rng.choice(_DOMAINS)}/{rng.randint(100, 999)}", "snippet": body}]
        return json.dumps({"results": results}, ensure_ascii=False, indent=2)
    if wrapper == "chat":
        lines = [f"{rng.choice(_NAMES)}: {p}" for p in body.split("\n\n")]
        return f"Messages from {rng.choice(_CHANNELS)}\n" + "\n".join(lines)
    if wrapper == "ticket":
        return (f"Ticket #{rng.randint(1000, 99999)}: {title}\nStatus: open\nReporter: {rng.choice(_NAMES)}\n\n"
                f"Description:\n{body}")
    raise ValueError(wrapper)


def _locate(text: str, piece: str, wrapper: str) -> tuple[int, int]:
    needle = json.dumps(piece, ensure_ascii=False)[1:-1] if wrapper == "search_json" else piece
    if wrapper == "web":
        needle = needle.split("\n\n")[0]
    start = text.find(needle)
    if start < 0:  # fall back to the first line of the piece
        needle = needle.splitlines()[0] if needle.splitlines() else needle
        start = text.find(needle)
    return (start, start + len(needle)) if start >= 0 else (0, 0)


def compose(carrier: str, title: str, insert: str, rng: random.Random,
            wrapper: str | None = None, concealment: str | None = None, position: str | None = None) -> Composed:
    wrapper = wrapper or rng.choice(WRAPPERS)
    concealment = concealment or rng.choice(CONCEALMENTS)
    position = position or rng.choice(POSITIONS)
    piece = _conceal(insert.strip(), concealment, rng)
    body = _place(carrier.strip(), piece, position, inline=(concealment == "inline"), rng=rng)
    text = _render(wrapper, title.strip() or "Untitled", body, rng)
    return Composed(text, _locate(text, piece, wrapper), wrapper, concealment, position)
