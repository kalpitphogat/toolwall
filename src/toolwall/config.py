"""Runtime configuration shared by the API service and the MCP gateway (one YAML file)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class UpstreamServer(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")   # becomes the tool-name prefix: <name>__<tool>
    command: str
    args: list[str] = []
    env: dict[str, str] | None = None
    cwd: str | None = None


class PolicyConfig(BaseModel):
    mode: Literal["enforce", "monitor"] = "enforce"   # monitor: log verdicts but never alter tool output
    warn_threshold: float | str = "fpr_0.05"          # number, or a key into the model's calibration thresholds
    block_threshold: float | str = "fpr_0.01"


class Settings(BaseModel):
    scorer: dict = {"type": "transformer", "model_dir": "models/toolwall-deberta-v3-small"}
    policy: PolicyConfig = PolicyConfig()
    audit_db: str = "toolwall_audit.db"
    store_excerpts: bool = True       # store a short excerpt of flagged text; off = hashes only
    upstreams: list[UpstreamServer] = []


def load_settings(path: str | Path) -> Settings:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    return Settings.model_validate(data or {})
