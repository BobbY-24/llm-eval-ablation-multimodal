"""CLI: python3 -m ablation {plan,run,score} ...

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
from ablation.prompts import build_prompt
from ablation.providers import make_provider
from ablation.runner import LiveCallRefused, run
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
    go.add_argument("--live", action="store_true", help="Allow network calls to a real provider")
    go.add_argument("--out", required=True, type=Path, help="Fresh output directory")
    score = sub.add_parser("score", help="Score saved predictions and compute paired contrasts")
    score.add_argument("--items", required=True, type=Path)
    score.add_argument("--predictions", required=True, type=Path)
    score.add_argument("--seed", type=int, default=42)
    score.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

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
    try:
        manifest = run(provider, stimuli, task=args.task, out_dir=args.out, max_calls=args.max_calls,
                       allow_live=args.live,
                       items_sha256=hashlib.sha256(args.items.read_bytes()).hexdigest())
    except LiveCallRefused as exc:
        print(exc, file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
