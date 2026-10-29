"""The gateway: an MCP server to the agent, an MCP client to each upstream server.

    agent ──MCP──▶ toolwall gateway ──MCP──▶ upstream servers (web fetch, mail, files, ...)
                        │
                        └─ scans every text result, applies policy, writes the audit log

Upstream tools are re-exported as `<upstream>__<tool>`. Their output schemas are dropped,
because the gateway may rewrite (spotlight) or withhold a result, and a client validating
against the original schema would reject the rewritten content.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass

import anyio
from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from ..config import Settings, UpstreamServer, load_settings
from ..model.scorer import Scorer, load_scorer
from ..policy import Thresholds, Verdict, block_message, decide, resolve_thresholds, spotlight
from ..store import AuditStore, Event

log = logging.getLogger("toolwall.gateway")
SEP = "__"
SCAN_TOOL = "toolwall__scan_text"


@dataclass
class Route:
    upstream: str
    tool: str
    session: ClientSession


class Gateway:
    def __init__(self, settings: Settings, scorer: Scorer, store: AuditStore, thresholds: Thresholds):
        self.settings, self.scorer, self.store, self.thresholds = settings, scorer, store, thresholds
        self.routes: dict[str, Route] = {}
        self.tools: list[types.Tool] = [types.Tool(
            name=SCAN_TOOL,
            description="Score a piece of text for prompt-injection content. Returns score (0-1) and the verdict.",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        )]
        self._score_lock = threading.Lock()   # one model, many concurrent tool calls

    # ------------------------------------------------------------------ upstreams
    async def connect(self, stack: AsyncExitStack, upstream: UpstreamServer) -> None:
        params = StdioServerParameters(command=upstream.command, args=upstream.args, env=upstream.env, cwd=upstream.cwd)
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        cursor = None
        while True:
            res = await session.list_tools(params=types.PaginatedRequestParams(cursor=cursor) if cursor else None)
            for tool in res.tools:
                name = f"{upstream.name}{SEP}{tool.name}"
                self.routes[name] = Route(upstream.name, tool.name, session)
                self.tools.append(tool.model_copy(update={
                    "name": name,
                    "output_schema": None,
                    "description": f"[via {upstream.name}] {tool.description or ''}".strip(),
                }))
            cursor = res.next_cursor
            if not cursor:
                break
        log.info("upstream %s: %d tools", upstream.name, sum(1 for r in self.routes.values() if r.upstream == upstream.name))

    # ------------------------------------------------------------------ scoring
    async def _scan(self, texts: list[str]):
        def run():
            with self._score_lock:
                return self.scorer.scan_many(texts)
        return await anyio.to_thread.run_sync(run)

    # ------------------------------------------------------------------ MCP handlers
    async def on_list_tools(self, ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=self.tools)

    async def on_call_tool(self, ctx, params: types.CallToolRequestParams):
        if params.name == SCAN_TOOL:
            text = str((params.arguments or {}).get("text", ""))
            [s] = await self._scan([text])
            verdict = decide(s.score, self.thresholds)
            self.store.record(Event("gateway", verdict.value, s.score, False, s.latency_ms, text, s.span, tool=SCAN_TOOL))
            return _text_result(json.dumps({"score": round(s.score, 4), "verdict": verdict.value}))

        route = self.routes.get(params.name)
        if route is None:
            return _text_result(f"Unknown tool {params.name!r}", error=True)
        try:
            res = await route.session.call_tool(route.tool, params.arguments or {})
        except Exception as e:  # upstream crashed or timed out: report, never hang the agent
            log.exception("upstream call failed")
            return _text_result(f"Upstream {route.upstream} failed: {e}", error=True)
        if not isinstance(res, types.CallToolResult):   # e.g. an input-required interaction: nothing to scan yet
            return res
        return await self._filter(route, params, res)

    async def _filter(self, route: Route, params, res: types.CallToolResult) -> types.CallToolResult:
        items = [c.text for c in res.content if isinstance(c, types.TextContent)]
        if res.structured_content is not None:
            items.append(json.dumps(res.structured_content, ensure_ascii=False))
        if not items:
            return res
        t0 = time.perf_counter()
        scans = await self._scan(items)
        latency = (time.perf_counter() - t0) * 1000
        top = max(range(len(scans)), key=lambda i: scans[i].score)
        score = scans[top].score
        verdict = decide(score, self.thresholds)
        enforce = self.settings.policy.mode == "enforce"
        event_id = self.store.record(Event(
            "gateway", verdict.value, score, enforce and verdict != Verdict.ALLOW, latency, items[top], scans[top].span,
            upstream=route.upstream, tool=params.name, args_repr=json.dumps(params.arguments or {}, sort_keys=True, default=str),
        ))
        if verdict != Verdict.ALLOW:
            log.warning("%s on %s (score %.3f, event %d)", verdict.value, params.name, score, event_id)
        if not enforce or verdict == Verdict.ALLOW:
            return res
        if verdict == Verdict.BLOCK:
            return _text_result(block_message(params.name, score, event_id), error=True)

        # WARN: fence each flagged text item; structured content cannot be fenced, so it is dropped
        it = iter(scans)
        content = []
        for c in res.content:
            if isinstance(c, types.TextContent):
                s = next(it)
                content.append(c.model_copy(update={"text": spotlight(c.text, s.score)}) if s.score >= self.thresholds.warn else c)
            else:
                content.append(c)
        return types.CallToolResult(content=content, is_error=res.is_error)


def _text_result(text: str, error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=error)


def build_gateway(settings: Settings, scorer: Scorer | None = None, store: AuditStore | None = None) -> Gateway:
    scorer = scorer or load_scorer(settings.scorer)
    thresholds = resolve_thresholds(settings.policy, getattr(scorer, "thresholds", {}) or {})
    store = store or AuditStore(settings.audit_db, settings.store_excerpts)
    return Gateway(settings, scorer, store, thresholds)


async def serve(gateway: Gateway) -> None:
    async with AsyncExitStack() as stack:
        for up in gateway.settings.upstreams:
            await gateway.connect(stack, up)
        server = Server("toolwall", version="0.1.0", on_list_tools=gateway.on_list_tools, on_call_tool=gateway.on_call_tool,
                        instructions="Tools are proxied through toolwall, which scans tool outputs for prompt injection.")
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())


async def run_gateway(config_path: str) -> None:
    # stdout carries the MCP protocol, so every log line must go to stderr
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    await serve(build_gateway(load_settings(config_path)))
