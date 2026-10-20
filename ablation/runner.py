"""Bounded execution: one attempt plus at most one JSON repair per stimulus.

Every attempt is appended to ``attempts.jsonl`` before the next call, so a
crash cannot silently lose paid outputs. Live providers run only with
``allow_live=True`` and a finite ``max_calls``; unrun stimuli stay visible as
``not_run_budget`` instead of disappearing.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ablation.conditions import Stimulus
from ablation.prompts import build_prompt, build_repair_prompt, parse_response


class LiveCallRefused(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prediction_id(model: str, stimulus: Stimulus, task: str) -> str:
    key = f"{model}|{stimulus.example_id}|{stimulus.condition}|{task}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class Budget:
    def __init__(self, max_calls: int):
        if not isinstance(max_calls, int) or max_calls < 0:
            raise ValueError("max_calls must be a nonnegative integer")
        self.max_calls = max_calls
        self.used = 0

    def take(self) -> bool:
        if self.used >= self.max_calls:
            return False
        self.used += 1
        return True


def run_one(provider, stimulus: Stimulus, task: str, budget: Budget, log) -> dict:
    original = build_prompt(stimulus, task)
    prompt = original
    result = {"status": "not_run_budget", "attempts": 0, "error": None}
    for attempt in range(2):
        if not budget.take():
            if attempt:
                result["status"] = "parsing_failure"
            return result
        record = {"example_id": stimulus.example_id, "condition": stimulus.condition, "task": task,
                  "model": provider.name, "attempt": attempt, "started_at": _now(),
                  "raw_output": None, "usage": None, "parsing_status": "not_attempted", "error": None}
        started = time.perf_counter()
        parsed = None
        try:
            generated = provider.generate(prompt, stimulus.image_paths)
            record["raw_output"], record["usage"] = generated.text, generated.usage
        except Exception as exc:  # recorded, never retried as a repair
            record["error"] = f"{type(exc).__name__}: {exc}"
            result["status"] = "inference_failure"
        else:
            try:
                parsed = parse_response(generated.text, task)
                record["parsing_status"] = "valid"
                result["status"] = "ok"
            except (ValueError, TypeError, RecursionError) as exc:
                record["parsing_status"] = "invalid"
                record["error"] = f"{type(exc).__name__}: {exc}"
                result["status"] = "parsing_failure"
        record["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        log(record)
        result["attempts"] = attempt + 1
        result["error"] = record["error"]
        if parsed is not None:
            result.update(parsed)
            return result
        if record["parsing_status"] != "invalid":
            return result
        prompt = build_repair_prompt(original, record["raw_output"], record["error"], task)
    return result


def run(provider, stimuli: list[Stimulus], *, task: str, out_dir: str | Path, max_calls: int,
        allow_live: bool = False, items_sha256: str | None = None) -> dict:
    if getattr(provider, "live", True) and not allow_live:
        raise LiveCallRefused("Live providers need allow_live=True (CLI: --live) and a finite --max-calls")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(max_calls)
    counts: dict[str, int] = {}
    with (out / "attempts.jsonl").open("w", encoding="utf-8") as attempts, \
            (out / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
        def log(record):
            attempts.write(json.dumps(record, ensure_ascii=False) + "\n")
            attempts.flush()

        for stimulus in stimuli:
            result = run_one(provider, stimulus, task, budget, log)
            row = {"prediction_id": prediction_id(provider.name, stimulus, task),
                   "example_id": stimulus.example_id, "condition": stimulus.condition, "task": task,
                   "model": provider.name, "image_source_ids": list(stimulus.image_source_ids),
                   "choices": stimulus.choices, **result}
            predictions.write(json.dumps(row, ensure_ascii=False) + "\n")
            predictions.flush()
            counts[result["status"]] = counts.get(result["status"], 0) + 1
    manifest = {"provider": provider.name, "live": bool(getattr(provider, "live", True)), "task": task,
                "stimuli": len(stimuli), "max_calls": max_calls, "calls_used": budget.used,
                "status_counts": counts, "items_sha256": items_sha256,
                "conditions": sorted({s.condition for s in stimuli}), "finished_at": _now()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
