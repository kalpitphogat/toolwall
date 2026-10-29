"""FastAPI service: scan text on demand and read the audit log the gateway writes.

    POST /v1/scan            {"text": ..., "source": ...}  -> score, verdict, evidence span
    POST /v1/scan/batch      {"texts": [...]}               -> one result per text
    GET  /v1/events          ?limit=&verdict=&tool=          -> recent audit events
    GET  /v1/stats                                          -> counts by verdict/tool, latency percentiles
    GET  /healthz
"""
from __future__ import annotations

import threading

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .config import Settings, load_settings
from .model.scorer import Scorer, load_scorer
from .policy import decide, resolve_thresholds
from .store import AuditStore, Event

MAX_CHARS = 200_000


class ScanRequest(BaseModel):
    text: str = Field(max_length=MAX_CHARS)
    source: str | None = Field(default=None, description="free-form label stored with the audit event, e.g. a URL")


class BatchRequest(BaseModel):
    texts: list[str] = Field(max_length=64)


class ScanResponse(BaseModel):
    score: float
    verdict: str
    evidence: str | None
    span: tuple[int, int] | None
    thresholds: dict[str, float]
    model: str
    latency_ms: float
    event_id: int | None = None


def create_app(config: str | Settings = "toolwall.yaml", scorer: Scorer | None = None,
               store: AuditStore | None = None) -> FastAPI:
    settings = config if isinstance(config, Settings) else load_settings(config)
    scorer = scorer or load_scorer(settings.scorer)
    thresholds = resolve_thresholds(settings.policy, getattr(scorer, "thresholds", {}) or {})
    store = store or AuditStore(settings.audit_db, settings.store_excerpts)
    lock = threading.Lock()
    app = FastAPI(title="toolwall", version="0.1.0",
                  description="Prompt-injection scanning for agent tool outputs.")

    def run(texts: list[str]):
        with lock:
            return scorer.scan_many(texts)

    def respond(text: str, s, source: str | None) -> ScanResponse:
        verdict = decide(s.score, thresholds)
        eid = store.record(Event("api", verdict.value, s.score, False, s.latency_ms, text, s.span, tool=source))
        return ScanResponse(
            score=round(s.score, 6), verdict=verdict.value,
            evidence=text[s.span[0]: s.span[1]][:500] if s.span and verdict.value != "allow" else None,
            span=s.span, thresholds={"warn": thresholds.warn, "block": thresholds.block},
            model=scorer.name, latency_ms=round(s.latency_ms, 2), event_id=eid,
        )

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "model": scorer.name}

    @app.post("/v1/scan", response_model=ScanResponse)
    def scan_one(req: ScanRequest):
        [s] = run([req.text])
        return respond(req.text, s, req.source)

    @app.post("/v1/scan/batch", response_model=list[ScanResponse])
    def scan_batch(req: BatchRequest):
        if any(len(t) > MAX_CHARS for t in req.texts):
            raise HTTPException(413, f"each text must be at most {MAX_CHARS} characters")
        return [respond(t, s, None) for t, s in zip(req.texts, run(req.texts))]

    @app.get("/v1/events")
    def events(limit: int = Query(100, ge=1, le=1000), verdict: str | None = None, tool: str | None = None):
        return store.events(limit=limit, verdict=verdict, tool=tool)

    @app.get("/v1/stats")
    def stats():
        return store.stats()

    app.state.store = store
    return app
