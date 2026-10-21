"""CLI: python3 -m ablation {plan,run,score,draft-items,build-items} ...

Without --live only the offline mock provider runs. Live runs need --live,
a finite --max-calls and the key in the named environment variable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from ablation.conditions import CONDITIONS, TASKS, build_stimuli, load_items
from ablation.items import DEFAULT_GRAPH, DEFAULT_SCOPES, build_items, draft_review, load_gaming_rows, scope_counts
from ablation.prompts import build_prompt
from ablation.providers import make_provider
from ablation.runner import LiveCallRefused, Paced, run
from ablation.scoring import load_predictions, score_rows, summarize


def _common(parser):
    parser.add_argument("--items", required=True, type=Path, help="JSONL items file")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--task", choices=TASKS, default="mcq")
    parser.add_argument("--seed", type=int, default=42)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m ablation", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Print the stimuli and planned call count; no provider is called")
    _common(plan)
    plan.add_argument("--show-prompts", action="store_true")
    go = sub.add_parser("run", help="Run stimuli through a provider")
    _common(go)
    go.add_argument("--provider", choices=("mock", "openai", "gemini"), default="mock")
    go.add_argument("--model", default="mock")
    go.add_argument("--base-url", help="OpenAI-compatible base URL, e.g. https://api.openai.com/v1")
    go.add_argument("--api-key-env", help="Name of the environment variable holding the key")
    go.add_argument("--max-calls", type=int, required=True, help="Hard cap on provider calls, repairs included")
    go.add_argument("--max-output-tokens", type=int, default=512)
    go.add_argument("--timeout", type=float, default=120.0)
    go.add_argument("--transient-retries", type=int, default=0,
                    help="Retries per call after HTTP 429/5xx or a timeout; each retry uses --max-calls")
    go.add_argument("--min-interval", type=float, default=0.0, help="Minimum seconds between provider calls")
    go.add_argument("--live", action="store_true", help="Allow network calls to a real provider")
    go.add_argument("--out", required=True, type=Path, help="Fresh output directory")
    score = sub.add_parser("score", help="Score saved predictions and compute paired contrasts")
    score.add_argument("--items", required=True, type=Path)
    score.add_argument("--predictions", required=True, type=Path)
    score.add_argument("--seed", type=int, default=42)
    score.add_argument("--out", type=Path)
    draft = sub.add_parser("draft-items", help="Write a review CSV of candidate items from the usage graph; no provider is called")
    draft.add_argument("--graph", type=Path, default=DEFAULT_GRAPH, help="game_context_space.json")
    draft.add_argument("--scopes", nargs="+", default=list(DEFAULT_SCOPES), help="Exact scope labels, quoted")
    draft.add_argument("--list-scopes", action="store_true", help="Print scope labels with gaming term counts and exit")
    draft.add_argument("--seed", type=int, default=42)
    draft.add_argument("--out", type=Path, help="Fresh review CSV path")
    build = sub.add_parser("build-items", help="Validate a reviewed sheet and write items JSONL")
    build.add_argument("--review", required=True, type=Path)
    build.add_argument("--images", required=True, type=Path, help="Directory that image_file entries resolve against")
    build.add_argument("--out", required=True, type=Path, help="Fresh items JSONL path")
    args = parser.parse_args(argv)

    if args.command == "draft-items":
        if args.list_scopes:
            for label, count in scope_counts(load_gaming_rows(args.graph)).most_common():
                print(f"{count:4d}  {label}")
            return 0
        if args.out is None:
            parser.error("draft-items needs --out (or --list-scopes)")
        try:
            print(json.dumps(draft_review(args.graph, args.scopes, args.out, args.seed), indent=2))
        except (ValueError, FileExistsError) as exc:
            print(exc, file=sys.stderr)
            return 2
        return 0
    if args.command == "build-items":
        try:
            print(json.dumps(build_items(args.review, args.images, args.out), indent=2))
        except (ValueError, FileExistsError) as exc:
            print(exc, file=sys.stderr)
            return 2
        return 0
    if args.command == "score":
        summary = summarize(score_rows(load_predictions(args.predictions), load_items(args.items), args.seed),
                            args.seed)
        text = json.dumps(summary, indent=2)
        if args.out:
            args.out.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0

    items = load_items(args.items)
    stimuli = build_stimuli(items, args.conditions, args.task, args.seed)
    if args.command == "plan":
        print(json.dumps({"items": len(items), "stimuli": len(stimuli), "conditions": args.conditions,
                          "task": args.task, "max_calls_with_repairs": 2 * len(stimuli)}, indent=2))
        if args.show_prompts:
            for s in stimuli:
                print(f"\n--- {s.example_id} {s.condition} images={len(s.image_paths)}\n{build_prompt(s, args.task)}")
        return 0
    if args.provider != "mock" and not args.live:
        print("Refusing a live provider without --live.", file=sys.stderr)
        return 2
    provider = make_provider(args.provider, args.model, base_url=args.base_url, api_key_env=args.api_key_env,
                             timeout=args.timeout, max_output_tokens=args.max_output_tokens)
    if args.min_interval > 0:
        provider = Paced(provider, args.min_interval)
    try:
        manifest = run(provider, stimuli, task=args.task, out_dir=args.out, max_calls=args.max_calls,
                       allow_live=args.live, transient_retries=args.transient_retries,
                       items_sha256=hashlib.sha256(args.items.read_bytes()).hexdigest())
    except LiveCallRefused as exc:
        print(exc, file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
