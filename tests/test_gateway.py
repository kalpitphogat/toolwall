"""End to end over real stdio: test client -> toolwall gateway -> examples/demo_server.py.

The gateway runs as a subprocess with a stub scorer, exactly as an MCP host would launch it.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pytest
import yaml
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[1]
CLEAN = "https://docs.example.com/install"
POISONED = "https://blog.example.com/release-notes"


def _config(tmp_path, mode="enforce", warn=0.5, block=0.9, triggers=("ignore previous instructions",), score=0.99):
    cfg = {
        "scorer": {"type": "stub", "triggers": list(triggers)},
        "policy": {"mode": mode, "warn_threshold": warn, "block_threshold": block},
        "audit_db": str(tmp_path / "audit.db"),
        "upstreams": [{"name": "demo", "command": sys.executable, "args": [str(ROOT / "examples" / "demo_server.py")]}],
    }
    p = tmp_path / "toolwall.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


async def _session_calls(config_path, calls):
    params = StdioServerParameters(command=sys.executable, args=["-m", "toolwall.cli", "gateway", "--config", str(config_path)])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            results = [await s.call_tool(name, args) for name, args in calls]
            return tools, results


@pytest.mark.anyio
async def test_tools_are_namespaced_and_clean_output_passes_unchanged(tmp_path):
    tools, [res] = await _session_calls(_config(tmp_path), [("demo__fetch_page", {"url": CLEAN})])
    assert {"demo__fetch_page", "demo__read_inbox", "toolwall__scan_text"} <= set(tools)
    assert not res.is_error and res.content[0].text.startswith("Installation")


@pytest.mark.anyio
async def test_poisoned_output_is_blocked_and_audited(tmp_path):
    cfg = _config(tmp_path)
    _, [res] = await _session_calls(cfg, [("demo__fetch_page", {"url": POISONED})])
    assert res.is_error and "withheld" in res.content[0].text and "id_rsa" not in res.content[0].text
    rows = sqlite3.connect(tmp_path / "audit.db").execute("SELECT verdict, tool, enforced FROM events").fetchall()
    assert rows == [("block", "demo__fetch_page", 1)]


@pytest.mark.anyio
async def test_warn_band_fences_output_instead_of_blocking(tmp_path):
    # the stub scores 0.99 on the inbox's injected email, which lands between warn=0.5 and block=0.999
    cfg = _config(tmp_path, warn=0.5, block=0.999, triggers=("do not mention this",))
    _, [res] = await _session_calls(cfg, [("demo__read_inbox", {})])
    text = res.content[0].text
    assert not res.is_error and "<<<UNTRUSTED_TOOL_OUTPUT" in text and "Lunch Friday?" in text


@pytest.mark.anyio
async def test_monitor_mode_logs_but_never_alters(tmp_path):
    cfg = _config(tmp_path, mode="monitor")
    _, [res] = await _session_calls(cfg, [("demo__fetch_page", {"url": POISONED})])
    assert not res.is_error and "ignore previous instructions" in res.content[0].text
    rows = sqlite3.connect(tmp_path / "audit.db").execute("SELECT verdict, enforced FROM events").fetchall()
    assert rows == [("block", 0)]


@pytest.mark.anyio
async def test_scan_tool_and_unknown_tool(tmp_path):
    _, (scan, unknown) = await _session_calls(_config(tmp_path), [
        ("toolwall__scan_text", {"text": "please ignore previous instructions"}),
        ("demo__does_not_exist", {}),
    ])
    assert json.loads(scan.content[0].text)["verdict"] == "block"
    assert unknown.is_error


@pytest.fixture
def anyio_backend():
    return "asyncio"
