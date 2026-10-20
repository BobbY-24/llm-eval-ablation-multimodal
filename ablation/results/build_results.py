"""Derive the public provider-comparison tables from saved aggregate ledgers.

This script reads only aggregate summary JSON files from the historical
50-source term-collection campaigns (2026-10-01 .. 2026-10-03). It never reads
page snapshots, quotes, candidate rows or proposals, and it writes no term,
URL or source identifier. It makes no network or model calls.

    python3 -I -B ablation/results/build_results.py --outputs-root /path/to/outputs

Figures are written only when matplotlib is importable; the tables need only
the standard library.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

INPUTS = {
    "cost_latency": "all-model-comparison-2026-10-03/evidence/cost-latency.json",
    "glm_glm_comparison": "glm-glm-50-comparison-2026-10-02/reporting/comparison.json",
    "deepseek_glm_summary": "deepseek-glm-50-comparison-2026-10-01/reporting/summary.json",
    "glm_nemotron_summary": "glm-nemotron-50-comparison-2026-10-02/run/summary.json",
    "glm_qwen_summary": "glm-qwen-judge50-2026-10-02/run/summary.json",
}

# Aggregate counts that the saved reports state in prose tables rather than JSON.
# Transcribed from glm-glm-50-comparison-2026-10-02/extraction-comparison.md and
# glm-qwen-judge50-2026-10-02/result-report.md.
TRANSCRIBED = {
    "deepseek_glm": {"distinct_normalized_raw_forms": 6578, "local_grounding_rejects": 212,
                      "gaming_raw": 9100, "cyber_raw": 4205, "gaming_retained": 7186, "cyber_retained": 3903},
    "glm_glm": {"distinct_normalized_raw_forms": 8230, "local_grounding_rejects": 647,
                "gaming_raw": 9329, "cyber_raw": 6387, "gaming_retained": 7282, "cyber_retained": 5481},
    "shared_forms": 6098, "deepseek_only_forms": 2555, "glm_only_forms": 4331,
    "qwen_smoke_active_proposals": 508, "guarded_raw_proposals": 17349,
}

MODEL_NAMES = {
    "accounts/fireworks/models/deepseek-v4p1-flash": "DeepSeek V4.1 Flash",
    "accounts/fireworks/models/glm-5p3-flash": "GLM 5.3 Flash",
    "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b": "Nemotron 3.5 Lightning 30B A3B",
    "qwen3.7-flash-2026-07-15": "Qwen 3.7 Flash",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def money(value) -> float:
    return round(float(value), 8)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def record(cost_latency: dict, root: str) -> dict:
    return next(r for r in cost_latency["records"] if r["artifact_root"] == root)


def pipeline_rows(data: dict) -> list[dict]:
    rows = []
    for key, root in (("deepseek_glm", "deepseek-glm-50-comparison-2026-10-01"),
                      ("glm_glm", "glm-glm-50-comparison-2026-10-02")):
        r = record(data["cost_latency"], root)
        e, j = r["stages"]["niche_extraction"], r["stages"]["niche_judgment"]
        eligible = (data["deepseek_glm_summary"]["quality_audit"]["screening_status"]["cleanup_checked"] if key == "deepseek_glm"
                    else data["glm_nemotron_summary"]["judge_eligible_ids"])
        t = TRANSCRIBED[key]
        decisions = r["decisions"]
        rows.append({
            "pipeline": f"{MODEL_NAMES[r['extractor']]} -> {MODEL_NAMES[r['judge']]}",
            "extractor": r["extractor"], "judge": r["judge"],
            "sources": r["processed_source_count"],
            "sources_complete": r["source_statuses"].get("complete", 0),
            "sources_partial": r["source_statuses"].get("partial", 0),
            "sources_failed": r["source_statuses"].get("failed", 0),
            "raw_proposals": r["candidates"],
            "distinct_normalized_raw_forms": t["distinct_normalized_raw_forms"],
            "local_grounding_rejects": t["local_grounding_rejects"],
            "local_grounding_reject_rate": round(t["local_grounding_rejects"] / r["candidates"], 4),
            "judge_eligible_occurrences": eligible,
            "keep": decisions.get("KEEP", 0), "review": decisions.get("REVIEW", 0),
            "reject": decisions.get("REJECT", 0), "duplicate": decisions.get("DUPLICATE", 0),
            "retained_keep_plus_review": r["retained_candidates"],
            "extraction_calls": e["physical_calls"], "judgment_calls": j["physical_calls"],
            "unknown_outcome_calls": sum(s.get("states", {}).get("outcome_unknown", 0) for s in (e, j)),
            "extraction_input_tokens": e["reported_input_tokens"],
            "extraction_output_tokens": e["reported_visible_output_tokens"],
            "extraction_reasoning_tokens": e["reported_reasoning_tokens"],
            "judgment_input_tokens": j["reported_input_tokens"],
            "judgment_output_tokens": j["reported_visible_output_tokens"],
            "judgment_reasoning_tokens": j["reported_reasoning_tokens"],
            "known_extraction_usd": money(e["known_standard_rate_usd"]),
            "known_judgment_usd": money(j["known_standard_rate_usd"]),
            "known_total_usd": money(float(e["known_standard_rate_usd"]) + float(j["known_standard_rate_usd"])),
            "unknown_hold_usd": money(float(e["held_unknown_ceiling_usd"]) + float(j["held_unknown_ceiling_usd"])),
        })
    guarded = record(data["cost_latency"], "glm-nemotron-glm-guarded50-2026-10-02")
    roles = guarded["role_costs"]
    rows.append({
        "pipeline": "GLM 5.3 Flash -> Nemotron screen + GLM fallback (guarded)",
        "extractor": roles["extraction"]["model"],
        "judge": f"{roles['primary']['model']} + {roles['strong']['model']}",
        "sources": 50,
        "sources_complete": guarded["source_statuses"]["complete"],
        "sources_partial": guarded["source_statuses"]["partial"],
        "sources_failed": guarded["source_statuses"]["failed"],
        "raw_proposals": TRANSCRIBED["guarded_raw_proposals"], "distinct_normalized_raw_forms": "", "local_grounding_rejects": "",
        "local_grounding_reject_rate": "", "judge_eligible_occurrences": "",
        "keep": "", "review": "", "reject": "", "duplicate": "", "retained_keep_plus_review": "",
        "extraction_calls": roles["extraction"]["calls"],
        "judgment_calls": roles["primary"]["calls"] + roles["strong"]["calls"],
        "unknown_outcome_calls": sum(v["states"].get("outcome_unknown", 0) for v in roles.values()),
        "extraction_input_tokens": roles["extraction"]["input_tokens"],
        "extraction_output_tokens": roles["extraction"]["visible_output_tokens"],
        "extraction_reasoning_tokens": roles["extraction"]["reasoning_tokens"],
        "judgment_input_tokens": roles["primary"]["input_tokens"] + roles["strong"]["input_tokens"],
        "judgment_output_tokens": roles["primary"]["visible_output_tokens"] + roles["strong"]["visible_output_tokens"],
        "judgment_reasoning_tokens": roles["primary"]["reasoning_tokens"] + roles["strong"]["reasoning_tokens"],
        "known_extraction_usd": money(roles["extraction"]["known_usd"]),
        "known_judgment_usd": money(roles["primary"]["known_usd"] + roles["strong"]["known_usd"]),
        "known_total_usd": money(guarded["known_standard_rate_usd"]),
        "unknown_hold_usd": money(guarded["held_unknown_ceiling_usd"]),
    })
    return rows


def community_rows() -> list[dict]:
    rows = []
    for key, name in (("deepseek_glm", "DeepSeek V4.1 Flash"), ("glm_glm", "GLM 5.3 Flash")):
        t = TRANSCRIBED[key]
        for community, prefix in (("Gaming", "gaming"), ("Cybersecurity", "cyber")):
            rows.append({"extractor": name, "judge": "GLM 5.3 Flash", "community": community, "sources": 25,
                         "raw_proposals": t[f"{prefix}_raw"], "retained_keep_plus_review": t[f"{prefix}_retained"]})
    return rows


def replay_rows(data: dict) -> list[dict]:
    nemo, qwen = data["glm_nemotron_summary"], data["glm_qwen_summary"]
    nemo_cost = record(data["cost_latency"], "glm-nemotron-50-comparison-2026-10-02")
    glm_glm = record(data["cost_latency"], "glm-glm-50-comparison-2026-10-02")
    glm_j = glm_glm["stages"]["niche_judgment"]
    qwen_usable = qwen["qwen_keep"] + qwen["qwen_review"]
    return [
        {"judge": "GLM 5.3 Flash (baseline)", "candidate_pool": "GLM extraction, 15716 proposals",
         "scope": "all 50 sources", "eligible_ids": nemo["judge_eligible_ids"],
         "validated_ids": nemo["judge_eligible_ids"], "coverage": 1.0,
         "calls": glm_j["physical_calls"], "unknown_outcome_calls": 0,
         "known_judgment_usd": money(glm_j["known_standard_rate_usd"]), "unknown_hold_usd": 0.0,
         "status": "completed"},
        {"judge": "Nemotron 3.5 Lightning 30B A3B", "candidate_pool": "GLM extraction, 15716 proposals",
         "scope": "all 50 sources", "eligible_ids": nemo["judge_eligible_ids"],
         "validated_ids": nemo["validated_judgment_ids"],
         "coverage": round(nemo["validated_judgment_ids"] / nemo["judge_eligible_ids"], 4),
         "calls": nemo_cost["stages"]["niche_judgment"]["physical_calls"],
         "unknown_outcome_calls": nemo["budget"]["unknown_usage_or_outcome_calls"],
         "known_judgment_usd": money(nemo["budget"]["known_estimated_cost_usd"]),
         "unknown_hold_usd": money(nemo["budget"]["outstanding_reserved_usd"]),
         "status": "halted after unknown-outcome threshold"},
        {"judge": "Qwen 3.7 Flash", "candidate_pool": "GLM extraction, 15716 proposals",
         "scope": f"{qwen['active_sources']}-source smoke", "eligible_ids": TRANSCRIBED["qwen_smoke_active_proposals"],
         "validated_ids": qwen_usable,
         "coverage": round(qwen_usable / TRANSCRIBED["qwen_smoke_active_proposals"], 4),
         "calls": qwen["budget"]["calls"], "unknown_outcome_calls": qwen["budget"]["unknown"],
         "known_judgment_usd": money(qwen["budget"]["known_usd"]),
         "unknown_hold_usd": money(qwen["budget"]["held_usd"]),
         "status": "halted: provider rejected request (HTTP 400)"},
    ]


def latency_rows(data: dict) -> list[dict]:
    guarded = record(data["cost_latency"], "glm-nemotron-glm-guarded50-2026-10-02")
    rows = []
    for role, label in (("extraction", "GLM 5.3 Flash extraction"), ("strong", "GLM 5.3 Flash judgment"),
                        ("primary", "Nemotron 3.5 Lightning judgment")):
        t = guarded["independent_telemetry_latency"][role]
        rows.append({"stage": label, "returned_n": t["returned"]["n"],
                     "returned_mean_s": round(t["returned"]["mean_seconds"], 3),
                     "returned_median_s": round(t["returned"]["median_seconds"], 3),
                     "returned_p95_s": round(t["returned"]["p95_nearest_rank_seconds"], 3),
                     "unknown_or_timeout_n": t["failure"]["n"],
                     "all_attempt_mean_s": round(t["all_terminal"]["mean_seconds"], 3)})
    return rows


def figures(pipelines: list[dict], replays: list[dict], out: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    plt.rcParams.update({"svg.hashsalt": "ablation-results", "font.size": 9})
    out.mkdir(exist_ok=True)
    labels = ["DeepSeek→GLM", "GLM→GLM", "GLM→Nemotron+GLM"]
    written = []

    fig, ax = plt.subplots(figsize=(6, 3.2))
    e = [r["known_extraction_usd"] for r in pipelines]
    j = [r["known_judgment_usd"] for r in pipelines]
    h = [r["unknown_hold_usd"] for r in pipelines]
    ax.bar(labels, e, color="#4C78A8", label="extraction (known)")
    ax.bar(labels, j, bottom=e, color="#F58518", label="judgment (known)")
    ax.bar(labels, h, bottom=[a + b for a, b in zip(e, j)], color="none", edgecolor="#888",
           hatch="//", label="unknown-outcome hold (not a charge)")
    ax.set_ylabel("USD, 50 sources")
    ax.set_title("Known token cost by stage, same 50 frozen sources")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "cost_by_stage.svg", metadata={"Date": None})
    plt.close(fig)
    written.append("figures/cost_by_stage.svg")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    two = pipelines[:2]
    x = range(len(two))
    ax.bar([i - 0.2 for i in x], [r["raw_proposals"] for r in two], 0.4, color="#4C78A8", label="raw proposals")
    ax.bar([i + 0.2 for i in x], [r["retained_keep_plus_review"] for r in two], 0.4, color="#54A24B",
           label="retained (KEEP+REVIEW)")
    ax.set_xticks(list(x), labels[:2])
    ax.set_ylabel("occurrences")
    ax.set_title("Candidate pool size (not recall)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "candidate_pool.svg", metadata={"Date": None})
    plt.close(fig)
    written.append("figures/candidate_pool.svg")

    fig, ax = plt.subplots(figsize=(6, 2.6))
    names = ["GLM (baseline)", "Nemotron (replay)", "Qwen (2-source smoke)"]
    ax.barh(names, [r["coverage"] * 100 for r in replays], color=["#4C78A8", "#E45756", "#B279A2"])
    for i, r in enumerate(replays):
        ax.text(r["coverage"] * 100 + 1, i, f"{r['validated_ids']}/{r['eligible_ids']}", va="center", fontsize=8)
    ax.set_xlim(0, 115)
    ax.invert_yaxis()
    ax.set_xlabel("% of eligible IDs with a valid judgment")
    ax.set_title("Judge replay coverage on the GLM candidate pool")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "judgment_coverage.svg", metadata={"Date": None})
    plt.close(fig)
    written.append("figures/judgment_coverage.svg")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outputs-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    paths = {key: args.outputs_root / rel for key, rel in INPUTS.items()}
    data = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}

    pipelines = pipeline_rows(data)
    replays = replay_rows(data)
    tables = {"matched50_pipelines.csv": pipelines, "matched50_by_community.csv": community_rows(),
              "judgment_replays.csv": replays, "latency_guarded50.csv": latency_rows(data)}
    for name, rows in tables.items():
        write_csv(args.out / name, rows)
    written = figures(pipelines, replays, args.out / "figures")

    ds, glm = pipelines[0], pipelines[1]
    summary = {
        "evidence_type": "historical_api_campaign_aggregates",
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "task": ("Collect community terms from 50 frozen web sources (25 Gaming, 25 Cybersecurity): "
                 "LLM extraction of candidate occurrences, local literal-grounding checks, LLM judgment."),
        "cohort": {"sources": 50, "frozen_characters": 921230, "root_chunks": 352,
                   "chunk_chars": 3000, "chunk_overlap": 250, "max_candidates_per_chunk": 64,
                   "judgment_batch_size": 16, "same_bytes_across_pipelines": True},
        "providers": {"fireworks": ["DeepSeek V4.1 Flash", "GLM 5.3 Flash", "Nemotron 3.5 Lightning 30B A3B"],
                      "alibaba_model_studio_singapore": ["Qwen 3.7 Flash"]},
        "headline": {
            "glm_vs_deepseek_extraction_cost_change": round(glm["known_extraction_usd"] / ds["known_extraction_usd"] - 1, 4),
            "glm_vs_deepseek_raw_proposal_change": round(glm["raw_proposals"] / ds["raw_proposals"] - 1, 4),
            "glm_vs_deepseek_judge_eligible_change": round(glm["judge_eligible_occurrences"] / ds["judge_eligible_occurrences"] - 1, 4),
            "glm_vs_deepseek_judgment_cost_change": round(glm["known_judgment_usd"] / ds["known_judgment_usd"] - 1, 4),
            "glm_vs_deepseek_total_known_cost_change": round(glm["known_total_usd"] / ds["known_total_usd"] - 1, 4),
            "normalized_forms_shared_deepseek_only_glm_only": [TRANSCRIBED["shared_forms"],
                                                               TRANSCRIBED["deepseek_only_forms"],
                                                               TRANSCRIBED["glm_only_forms"]],
        },
        "caveats": [
            "Costs are provider-reported tokens times the saved list price, not invoices.",
            "Unknown-outcome holds are conservative reservations for requests whose billing is unresolved, not charges.",
            "KEEP is a model verdict, not a correctness label. No occurrence-level gold set exists, so no precision or recall is reported.",
            "More proposals do not imply higher recall; distinct forms include aliases, segmentation differences and errors.",
            "DeepSeek extraction ran without reasoning, GLM with a 1,024-token reasoning budget: a model-plus-setting comparison.",
            "The guarded GLM/Nemotron run changed extraction recovery (split depth 2) and judging policy; it is not a judge-only ablation.",
            "Nemotron and Qwen judge replays halted early; their costs are not completed-pipeline costs.",
        ],
        "excluded": "Page snapshots, quotes, candidate terms, proposals, URLs and source identifiers (third-party content).",
        "inputs_sha256": {rel: sha256(paths[key]) for key, rel in INPUTS.items()},
        "transcribed_from_reports": TRANSCRIBED,
        "files": sorted(tables) + written,
        "network_calls": 0,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
