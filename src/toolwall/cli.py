"""Command-line entry point. Imports are lazy so each subcommand only needs its own extras."""
from __future__ import annotations

import argparse
import json
import sys


def _data_build(args):
    from .data.build import BuildConfig, build

    cfg = BuildConfig(out_dir=args.out, seed=args.seed)
    manifest = build(cfg)
    print(json.dumps({"splits": manifest["splits"], "leak_audit": manifest["leak_audit"]}, indent=2))


def _serve(args):
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(args.config), host=args.host, port=args.port)


def _gateway(args):
    import anyio

    from .gateway.proxy import run_gateway

    anyio.run(run_gateway, args.config)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # train / eval keep their own argparse so `toolwall train --help` shows every option
    if argv[:1] == ["train"]:
        from .model.train import main as train_main

        return train_main(argv[1:])
    if argv[:1] == ["eval"]:
        from .eval.run import main as eval_main

        return eval_main(argv[1:])

    p = argparse.ArgumentParser(prog="toolwall", description="Scan agent tool outputs for indirect prompt injection.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="fine-tune the classifier (see `toolwall train --help`)")
    sub.add_parser("eval", help="evaluate scorers on held-out splits (see `toolwall eval --help`)")

    d = sub.add_parser("data", help="dataset commands")
    dsub = d.add_subparsers(dest="data_cmd", required=True)
    b = dsub.add_parser("build", help="download corpora and build leak-free splits")
    b.add_argument("--out", default="data/processed")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=_data_build)

    s = sub.add_parser("serve", help="run the FastAPI scan/audit service")
    s.add_argument("--config", default="toolwall.yaml")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8787)
    s.set_defaults(func=_serve)

    g = sub.add_parser("gateway", help="run the MCP gateway over stdio")
    g.add_argument("--config", default="toolwall.yaml")
    g.set_defaults(func=_gateway)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
