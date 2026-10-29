"""Drive the gateway the way an agent would and print what the agent receives.

    python examples/run_demo.py --config examples/toolwall.yaml

Calls the demo server's tools through toolwall (one clean page, one poisoned page, one inbox
with an injected email), then prints the audit events the gateway wrote.
"""
from __future__ import annotations

import argparse
import sys
import textwrap

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from toolwall.config import load_settings
from toolwall.store import AuditStore

CALLS = [
    ("demo__fetch_page", {"url": "https://docs.example.com/install"}),
    ("demo__fetch_page", {"url": "https://blog.example.com/release-notes"}),
    ("demo__read_inbox", {}),
]


async def main(config: str) -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "toolwall.cli", "gateway", "--config", config])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            print("tools:", ", ".join(t.name for t in (await s.list_tools()).tools), "\n")
            for name, args in CALLS:
                res = await s.call_tool(name, args)
                text = "\n".join(c.text for c in res.content if getattr(c, "type", "") == "text")
                print(f"-- {name} {args}  (is_error={res.is_error})")
                print(textwrap.indent(textwrap.shorten(text, 700, placeholder=" ...") if len(text) > 700 else text, "   "))
                print()
    settings = load_settings(config)
    print("audit log:")
    for e in reversed(AuditStore(settings.audit_db).events(limit=len(CALLS))):
        print(f"   #{e['id']} {e['tool']:<18} verdict={e['verdict']:<5} score={e['score']:.3f} latency={e['latency_ms']:.0f}ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="examples/toolwall.yaml")
    anyio.run(main, p.parse_args().config)
