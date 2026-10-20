"""Score saved predictions and compute paired condition contrasts.

MCQ accuracy is exact. Freeform responses get a lexical token-F1 against the
gold meaning, which is only a crude proxy until a rubric or judge scores them.
Failed or unrun predictions are never imputed: they count against
``accuracy_all_attempts`` and drop out of paired contrasts.
"""
from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

from ablation.conditions import Item, mcq_options

# name: (left, right); gain = score(left) - score(right)
CONTRASTS = {
    "context_gain": ("A1", "A0"),
    "multimodal_gain": ("A3", "A1"),
    "text_gain": ("A3", "A2"),
    "image_only_gain": ("A2", "A0"),
    "distractor_penalty": ("A1", "A4"),
}
_TOKEN = re.compile(r"[a-z0-9]+")


def token_f1(prediction: str, gold: str) -> float:
    p, g = Counter(_TOKEN.findall(prediction.lower())), Counter(_TOKEN.findall(gold.lower()))
    overlap = sum((p & g).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(p.values()), overlap / sum(g.values())
    return 2 * precision * recall / (precision + recall)


def load_predictions(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    for row in rows:
        unit = (row["model"], row["example_id"], row["condition"], row["task"])
        if unit in seen:
            raise ValueError(f"Duplicate prediction for {unit}")
        seen.add(unit)
    return rows


def score_rows(rows: list[dict], items: list[Item], seed: int = 42) -> list[dict]:
    by_id = {item.id: item for item in items}
    scored = []
    for row in rows:
        item = by_id.get(row["example_id"])
        score, status = None, "scored"
        if item is None or not item.gold_meaning:
            status = "missing_gold"
        elif row.get("status") != "ok":
            status = "inference_unavailable"
        elif row["task"] == "mcq":
            score = float(row["selected_option"] == mcq_options(item, seed)[1])
        else:
            score = token_f1(row["predicted_meaning"], item.gold_meaning)
        scored.append({**row, "score": score, "scoring_status": status, "community": item.community if item else None})
    return scored


def bootstrap_ci(values: list[float], seed: int = 42, n: int = 2000) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    return [means[int(0.025 * (n - 1))], means[int(0.975 * (n - 1))]]


def summarize(scored: list[dict], seed: int = 42, n_bootstrap: int = 2000) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for row in scored:
        groups.setdefault((row["model"], row["task"]), []).append(row)
    conditions, contrasts = [], []
    for (model, task), rows in sorted(groups.items()):
        table = {(r["example_id"], r["condition"]): r for r in rows}
        for condition in sorted({r["condition"] for r in rows}):
            selected = [r for r in rows if r["condition"] == condition and r["scoring_status"] != "missing_gold"]
            valid = [r["score"] for r in selected if r["score"] is not None]
            conditions.append({"model": model, "task": task, "condition": condition, "n_labeled": len(selected),
                               "n_scored": len(valid),
                               "mean_score_scored": statistics.fmean(valid) if valid else None,
                               "mean_score_all_attempts": sum(valid) / len(selected) if selected else None})
        examples = sorted({r["example_id"] for r in rows})
        for name, (left, right) in CONTRASTS.items():
            gains = []
            possible = 0
            for example in examples:
                a, b = table.get((example, left)), table.get((example, right))
                if a is None and b is None:
                    continue
                possible += 1
                if a and b and a["score"] is not None and b["score"] is not None:
                    gains.append(a["score"] - b["score"])
            if not possible:
                continue
            contrasts.append({"model": model, "task": task, "contrast": name, "left": left, "right": right,
                              "n_pairs": len(gains), "n_possible": possible,
                              "mean_gain": statistics.fmean(gains) if gains else None,
                              "ci95": bootstrap_ci(gains, seed, n_bootstrap),
                              "share_positive": sum(g > 0 for g in gains) / len(gains) if gains else None,
                              "small_sample": len(gains) < 30})
    return {"conditions": conditions, "contrasts": contrasts, "seed": seed, "n_bootstrap": n_bootstrap,
            "freeform_metric": "token_f1_lexical_proxy"}
