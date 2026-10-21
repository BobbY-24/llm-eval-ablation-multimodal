"""Bounded execution: one attempt plus at most one JSON repair per stimulus.

Every provider call is appended to ``attempts.jsonl`` before the next call, so
a crash cannot silently lose paid outputs. Live providers run only with
``allow_live=True`` and a finite ``max_calls``; unrun stimuli stay visible as
``not_run_budget`` instead of disappearing. Transient failures (HTTP 429/5xx,
timeouts) may be retried a bounded number of times; each retry is a call and
uses the budget. A provider error is never treated as a JSON repair.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from ablation.conditions import Stimulus
from ablation.prompts import build_prompt, build_repair_prompt, parse_response
from ablation.providers import ProviderError

TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})


class LiveCallRefused(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 6)


def prediction_id(model: str, stimulus: Stimulus, task: str) -> str:
    key = f"{model}|{stimulus.example_id}|{stimulus.condition}|{task}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def is_transient(exc: Exception) -> bool:
    if isinstance(exc, ProviderError):
        return exc.status in TRANSIENT_STATUS
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in TRANSIENT_STATUS
    return isinstance(exc, (TimeoutError, urllib.error.URLError))


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


class Paced:
    """Delegates to a provider; each call starts at least ``min_interval`` seconds after the previous one."""

    def __init__(self, provider, min_interval: float, clock=time.monotonic, sleep=time.sleep):
        if min_interval < 0:
            raise ValueError("min_interval must be nonnegative")
        self.provider = provider
        self.min_interval = min_interval
        self.clock = clock
        self.sleep = sleep
        self.name = provider.name
        self.live = getattr(provider, "live", True)
        self._last = None

    def generate(self, prompt: str, image_paths):
        if self._last is not None:
            remaining = self.min_interval - (self.clock() - self._last)
            if remaining > 0:
                self.sleep(remaining)
        self._last = self.clock()
        return self.provider.generate(prompt, image_paths)


def _record(provider, stimulus: Stimulus, task: str, attempt: int, transient: int) -> dict:
    return {"example_id": stimulus.example_id, "condition": stimulus.condition, "task": task,
            "model": provider.name, "attempt": attempt, "transient_try": transient,
            "started_at": _now(), "raw_output": None, "usage": None,
            "parsing_status": "not_attempted", "error": None}


def run_one(provider, stimulus: Stimulus, task: str, budget: Budget, log, *,
            transient_retries: int = 0, sleep=time.sleep, backoff_seconds: float = 2.0) -> dict:
    original = build_prompt(stimulus, task)
    prompt = original
    result = {"status": "not_run_budget", "attempts": 0, "error": None}
    for attempt in range(2):
        error = None
        for transient in range(transient_retries + 1):
            if not budget.take():
                if transient:
                    result.update(status="inference_failure", attempts=attempt + 1, error=error)
                elif attempt:
                    result["status"] = "parsing_failure"
                return result
            record = _record(provider, stimulus, task, attempt, transient)
            started = time.perf_counter()
            try:
                generated = provider.generate(prompt, stimulus.image_paths)
            except Exception as exc:  # recorded, never retried as a repair
                error = f"{type(exc).__name__}: {exc}"
                record["error"] = error
                record["elapsed_seconds"] = _elapsed(started)
                log(record)
                if transient < transient_retries and is_transient(exc):
                    sleep(backoff_seconds * 2 ** transient)
                    continue
                result.update(status="inference_failure", attempts=attempt + 1, error=error)
                return result
            record["raw_output"], record["usage"] = generated.text, generated.usage
            record["elapsed_seconds"] = _elapsed(started)
            break
        parsed = None
        try:
            parsed = parse_response(generated.text, task)
            record["parsing_status"] = "valid"
            result["status"] = "ok"
        except (ValueError, TypeError, RecursionError) as exc:
            record["parsing_status"] = "invalid"
            record["error"] = f"{type(exc).__name__}: {exc}"
            result["status"] = "parsing_failure"
        log(record)
        result["attempts"] = attempt + 1
        result["error"] = record["error"]
        if parsed is not None:
            result.update(parsed)
            return result
        prompt = build_repair_prompt(original, generated.text, record["error"], task)
    return result


def run(provider, stimuli: list[Stimulus], *, task: str, out_dir: str | Path, max_calls: int,
        allow_live: bool = False, items_sha256: str | None = None,
        transient_retries: int = 0, sleep=time.sleep) -> dict:
    if getattr(provider, "live", True) and not allow_live:
        raise LiveCallRefused("Live providers need allow_live=True (CLI: --live) and a finite --max-calls")
    if not isinstance(transient_retries, int) or transient_retries < 0:
        raise ValueError("transient_retries must be a nonnegative integer")
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
            result = run_one(provider, stimulus, task, budget, log,
                             transient_retries=transient_retries, sleep=sleep)
            row = {"prediction_id": prediction_id(provider.name, stimulus, task),
                   "example_id": stimulus.example_id, "condition": stimulus.condition, "task": task,
                   "model": provider.name, "image_source_ids": list(stimulus.image_source_ids),
                   "choices": stimulus.choices, **result}
            predictions.write(json.dumps(row, ensure_ascii=False) + "\n")
            predictions.flush()
            counts[result["status"]] = counts.get(result["status"], 0) + 1
    manifest = {"provider": provider.name, "live": bool(getattr(provider, "live", True)), "task": task,
                "stimuli": len(stimuli), "max_calls": max_calls, "calls_used": budget.used,
                "transient_retries": transient_retries, "status_counts": counts,
                "items_sha256": items_sha256, "conditions": sorted({s.condition for s in stimuli}),
                "finished_at": _now()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
