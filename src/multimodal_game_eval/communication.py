"""Bounded offline descriptions of explicit communication and reviewed probes.

String production, delivery, read exposure, and later reuse are different logged
events. None proves semantic understanding, adoption, novelty, or causal influence.
Only seeded_forms and explicitly recorded message.coined_forms are tracked. Text
matching casefolds Unicode, respects word boundaries, and keeps multiword forms
as phrases (whitespace between words may vary). No arbitrary vocabulary mining,
tokenizer, model API, downloader, or generated-code execution is involved.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .core import read_jsonl, sha256_file
from .experiments import validate_events


MAX_BYTES = 64 * 1024 * 1024
MAX_ROWS = 100_000
MAX_FORMS = 256
MAX_FORM_CHARACTERS = 256
MAX_MATCH_CHARACTER_WORK = 512 * 1024 * 1024
MAX_FORM_OBSERVATIONS = 1_000_000


def _rows(path):
    result = []
    for row in read_jsonl(path, max_bytes=MAX_BYTES):
        if len(result) >= MAX_ROWS:
            raise ValueError("communication_row_limit_exceeded")
        result.append(row)
    return result


def _form_key(value):
    return " ".join(value.casefold().split())


def _form(value):
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_FORM_CHARACTERS:
        raise ValueError("form_must_be_nonempty_bounded_string")
    return " ".join(value.split())


def _finite_nonnegative(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def _text(row):
    for field in ("text", "content"):
        if isinstance(row.get(field), str):
            return row[field]
    return None


def _actor(row):
    return row.get("agent_id", row.get("recipient", row.get("sender")))


def _point(row, index, community, episode_order):
    sequence = row.get("sequence")
    return {"event_id": row["event_id"], "event_index": index,
            "sequence": sequence if isinstance(sequence, int) and not isinstance(sequence, bool) else None,
            "episode_id": row["episode_id"], "community_episode_order": episode_order[row["episode_id"]],
            "community_id": community}


def _account(records):
    """Sum only recorded fields, retaining absent/invalid accounting explicitly."""
    sums = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
    present = Counter()
    invalid, calls = [], Counter()
    for row in records:
        usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
        values = {
            "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
            "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
            "total_tokens": usage.get("total_tokens"),
            "cost_usd": row.get("cost_usd", row.get("estimated_usd", usage.get("cost_usd"))),
        }
        for key, value in values.items():
            if value is None:
                continue
            if not _finite_nonnegative(value) or key != "cost_usd" and not isinstance(value, int):
                invalid.append({"event_id": row["event_id"], "field": key})
                continue
            present[key] += 1
            sums[key] += value
        call_id = row.get("model_call_id", row.get("call_id"))
        if isinstance(call_id, str) and call_id and any(value is not None for value in values.values()):
            calls[call_id] += 1
    return {"events": len(records),
            "recorded_sums": {key: sums[key] if present[key] else None for key in sums},
            "events_with_field": dict(present),
            "events_missing_field": {key: len(records) - present[key] for key in sums},
            "invalid_values": invalid,
            "repeated_accounted_call_ids": sorted(key for key, count in calls.items() if count > 1),
            "interpretation": "Recorded event fields only; stage totals may overlap and are not a deduplicated provider invoice. No absent tokens or costs are estimated."}


def _ci(records, field):
    clusters = defaultdict(list)
    for row in records:
        if field in row:
            clusters[row["community_id"]].append(int(row[field]))
    if len(clusters) < 2:
        return None
    keys = sorted(clusters)
    rng, values = random.Random(11), []
    for _ in range(1000):
        sample = [value for key in rng.choices(keys, k=len(keys)) for value in clusters[key]]
        values.append(sum(sample) / len(sample))
    values.sort()
    return [values[25], values[975]]


def _probes(records, known_communities):
    if records is None:
        return {"status": "not_supplied", "accepted": 0, "results": [], "excluded": [],
                "label_accuracy": None, "action_accuracy": None,
                "label_community_bootstrap_95_ci": None, "action_community_bootstrap_95_ci": None}
    if not isinstance(records, list) or len(records) > MAX_ROWS:
        raise ValueError("probe_results_must_be_bounded_list")
    probe_counts = Counter(row["probe_id"] for row in records if isinstance(row, dict) and isinstance(row.get("probe_id"), str))
    accepted, excluded = [], []
    for index, row in enumerate(records):
        reasons = []
        if not isinstance(row, dict):
            excluded.append({"row": index, "reasons": ["probe_object_required"]})
            continue
        for key in ("probe_id", "community_id", "agent_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                reasons.append("missing_" + key)
        probe_id = row.get("probe_id")
        if isinstance(probe_id, str) and probe_counts[probe_id] > 1:
            reasons.append("duplicate_probe_id")
        if row.get("reviewer_status") != "adjudicated":
            reasons.append("probe_not_adjudicated")
        if row.get("fresh") is not True:
            reasons.append("freshness_not_verified")
        if not isinstance(row.get("label_correct"), bool):
            reasons.append("label_correct_must_be_boolean")
        if "action_correct" in row and not isinstance(row["action_correct"], bool):
            reasons.append("action_correct_must_be_boolean_when_supplied")
        if not isinstance(row.get("community_id"), str) or row["community_id"] not in known_communities:
            reasons.append("community_not_in_validated_event_log")
        if reasons:
            excluded.append({"row": index, "probe_id": probe_id, "reasons": reasons})
        else:
            accepted.append({key: row[key] for key in ("probe_id", "community_id", "agent_id", "label_correct", "action_correct") if key in row})
    labels = [row["label_correct"] for row in accepted]
    actions = [row["action_correct"] for row in accepted if "action_correct" in row]
    communities = sorted({row["community_id"] for row in accepted})
    action_communities = sorted({row["community_id"] for row in accepted if "action_correct" in row})
    return {"status": "reviewed_fresh_probes_only", "accepted": len(accepted), "results": accepted,
            "excluded": excluded, "communities": communities,
            "label_accuracy": sum(labels) / len(labels) if labels else None,
            "action_accuracy": sum(actions) / len(actions) if actions else None,
            "action_denominator": len(actions),
            "action_communities": action_communities,
            "label_community_bootstrap_95_ci": _ci(accepted, "label_correct"),
            "action_community_bootstrap_95_ci": _ci(accepted, "action_correct"),
            "bootstrap_iterations": 1000,
            "uncertainty_warning": "fewer_than_two_communities" if len(communities) < 2 else None,
            "action_uncertainty_warning": "fewer_than_two_action_communities" if len(action_communities) < 2 else None,
            "interpretation": "Separate supplied adjudicated fresh probe outcomes; string reuse is not used as semantic gold. These descriptive outcomes do not identify a causal effect."}


def analyze_communication(events_path: Path, seeded_forms: list[str],
                          reviewed_probe_results: list[dict] | None = None) -> dict:
    """Analyze a complete valid log, never fabricate missing delivery or meaning.

    Additional optional event fields: text (or string content), coined_forms:list,
    usage:{input_tokens/output_tokens/total_tokens} (prompt/completion aliases),
    cost_usd or estimated_usd, model_call_id, status/error/failure, round and turn.
    episode_start may declare rounds/max_rounds/round_cap and
    max_turns/turn_cap/turns. Round/turn indices are one-based when supplied.
    Probe rows need probe_id, community_id, agent_id, reviewer_status:'adjudicated',
    fresh:true, label_correct:bool, optional action_correct:bool.
    """
    if not isinstance(seeded_forms, list) or len(seeded_forms) > MAX_FORMS:
        raise ValueError("seeded_forms_must_be_bounded_list")
    forms = {}
    for value in seeded_forms:
        canonical = _form(value)
        key = _form_key(canonical)
        forms.setdefault(key, {"form": canonical, "seeded": True, "first_coinage_marker": None})
    audit = validate_events(events_path)
    base = {"schema_version": "communication-analysis-v1", "mode": "offline_descriptive_analysis",
            "events_sha256": sha256_file(events_path, max_bytes=MAX_BYTES), "network_calls": 0,
            "event_audit_findings": audit,
            "limitations": ["First logged production is not proof of invention or absence from model pretraining.",
                            "A read event documents logged exposure, not comprehension or attention.",
                            "Matching strings do not establish shared meaning, adoption, or causal influence.",
                            "Missing deliveries, usage, probes, or counters remain missing; no synthetic results are filled in."]}
    if audit:
        return {**base, "status": "invalid", "analysis_performed": False,
                "reason": "event_access_audit_failed", "form_records": [],
                "reviewed_probes": {"status": "not_analyzed_due_to_invalid_event_log"}}
    rows = _rows(events_path)
    episode_order, community_counts, community_by_episode = {}, Counter(), {}
    missing_community = []
    for row in rows:
        if row.get("type", row.get("event_type")) == "episode_start":
            community = row.get("community_id")
            if not isinstance(community, str) or not community:
                community = "unassigned-episode:" + row["episode_id"]
                missing_community.append(row["episode_id"])
            community_by_episode[row["episode_id"]] = community
            community_counts[community] += 1
            episode_order[row["episode_id"]] = community_counts[community]
    aliases = {"message": "message_sent", "delivery": "message_delivered", "read": "message_read"}
    messages, warnings, stage_rows, counters, ceilings, failures = {}, [], defaultdict(list), {}, [], []
    for index, row in enumerate(rows, 1):
        kind = aliases.get(row.get("type", row.get("event_type")), row.get("type", row.get("event_type")))
        episode = row["episode_id"]
        counter = counters.setdefault(episode, {"episode_id": episode, "max_recorded_round": None,
            "max_recorded_turn": None, "round_cap": None, "turn_cap": None, "decision_events": 0})
        if kind == "episode_start":
            for cap, names in (("round_cap", ("round_cap", "max_rounds", "rounds")),
                               ("turn_cap", ("turn_cap", "max_turns", "turns"))):
                supplied = [row[key] for key in names if key in row]
                if supplied:
                    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in supplied) or len(set(supplied)) != 1:
                        ceilings.append({"event_id": row["event_id"], "code": "invalid_or_conflicting_" + cap})
                    else:
                        counter[cap] = supplied[0]
        for field, alternate, maximum, cap in (("round", "round_index", "max_recorded_round", "round_cap"),
                                                ("turn", "turn_index", "max_recorded_turn", "turn_cap")):
            value = row.get(field, row.get(alternate))
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    ceilings.append({"event_id": row["event_id"], "code": "invalid_" + field + "_index"})
                else:
                    counter[maximum] = max(counter[maximum] or 0, value)
                    if counter[cap] is not None and value > counter[cap]:
                        ceilings.append({"event_id": row["event_id"], "code": field + "_ceiling_exceeded"})
        if kind == "decision":
            counter["decision_events"] += 1
        status = row.get("status")
        if status in ("failed", "failure", "error", "timeout", "parse_failed", "transport_failed", "cancelled", "unknown") or row.get("error") or row.get("failure"):
            failures.append({"event_id": row["event_id"], "episode_id": episode,
                             "status": status if isinstance(status, str) else "recorded_error_or_failure"})
        stage = ("sent" if kind in ("message_sent", "forum_post") else "delivered" if kind == "message_delivered"
                 else "read" if kind in ("message_read", "forum_read") else "decision" if kind == "decision" else "other")
        stage_rows[stage].append(row)
        if kind in ("message_sent", "forum_post"):
            text = _text(row)
            point = _point(row, index, community_by_episode[episode], episode_order)
            messages[row["message_id"]] = {"row": row, "text": text, "folded_text": text.casefold() if text is not None else None,
                                           "point": point, "actor": _actor(row)}
            declared = row.get("coined_forms", [])
            if not isinstance(declared, list):
                warnings.append({"event_id": row["event_id"], "code": "invalid_coined_forms_list"})
                declared = []
            for value in declared:
                try:
                    canonical = _form(value)
                except ValueError:
                    warnings.append({"event_id": row["event_id"], "code": "invalid_explicit_coined_form"})
                    continue
                key = _form_key(canonical)
                forms.setdefault(key, {"form": canonical, "seeded": False, "first_coinage_marker": None})
                if forms[key]["first_coinage_marker"] is None:
                    forms[key]["first_coinage_marker"] = {**point, "agent_id": _actor(row), "message_id": row["message_id"]}
    accounting = {stage: _account(stage_rows[stage]) for stage in ("sent", "delivered", "read", "decision", "other")}
    if ceilings:
        return {**base, "status": "invalid", "analysis_performed": False,
                "reason": "recorded_ceiling_violation", "ceiling_findings": ceilings,
                "episode_counters": list(counters.values()), "recorded_accounting_by_stage": accounting,
                "failures": failures, "failure_count": len(failures),
                "form_records": [], "reviewed_probes": {"status": "not_analyzed_due_to_invalid_budget_log"}}
    if len(forms) > MAX_FORMS:
        raise ValueError("explicit_form_registry_limit_exceeded")
    if sum(len(message["folded_text"] or "") for message in messages.values()) * len(forms) > MAX_MATCH_CHARACTER_WORK:
        raise ValueError("form_matching_work_limit_exceeded")
    patterns = {key: re.compile(r"(?<!\w)" + r"\s+".join(re.escape(part) for part in key.split()) + r"(?!\w)")
                for key in forms}
    states = {}
    observation_count = 0

    def state(community, actor, key):
        identity = (community, actor, key)
        if identity not in states:
            states[identity] = {"community_id": community, "agent_id": actor, "form": forms[key]["form"],
                "form_key": key, "seeded": forms[key]["seeded"], "first_coinage_marker": forms[key]["first_coinage_marker"],
                "productions": [], "reads": [], "production_occurrences": 0, "read_occurrences": 0}
        return states[identity]

    for message in messages.values():
        text = message["text"]
        found = {} if text is None else {key: sum(1 for _ in pattern.finditer(message["folded_text"]))
                                        for key, pattern in patterns.items()}
        message["matches"] = {key: count for key, count in found.items() if count}
        for key, count in message["matches"].items():
            record = state(message["point"]["community_id"], message["actor"], key)
            record["productions"].append(message["point"])
            record["production_occurrences"] += count
            observation_count += 1
            if observation_count > MAX_FORM_OBSERVATIONS:
                raise ValueError("form_observation_limit_exceeded")
    stage_characters = {stage: {"events": len(stage_rows[stage]), "recorded_text_characters": 0,
                                "events_missing_message_text": 0} for stage in ("sent", "delivered", "read")}
    for index, row in enumerate(rows, 1):
        kind = aliases.get(row.get("type", row.get("event_type")), row.get("type", row.get("event_type")))
        stage = "sent" if kind in ("message_sent", "forum_post") else "delivered" if kind == "message_delivered" else "read" if kind in ("message_read", "forum_read") else None
        if stage is None:
            continue
        message = messages[row["message_id"]]
        if message["text"] is None:
            stage_characters[stage]["events_missing_message_text"] += 1
        else:
            stage_characters[stage]["recorded_text_characters"] += len(message["text"])
        if stage == "read":
            point = _point(row, index, community_by_episode[row["episode_id"]], episode_order)
            point["source_agent_id"] = message["actor"]
            point["self_authored_read"] = _actor(row) == message["actor"]
            for key, count in message["matches"].items():
                record = state(point["community_id"], _actor(row), key)
                record["reads"].append(point)
                record["read_occurrences"] += count
                observation_count += 1
        if observation_count > MAX_FORM_OBSERVATIONS:
            raise ValueError("form_observation_limit_exceeded")
    form_records = []
    for identity in sorted(states):
        record = states[identity]
        production = min(record["productions"], key=lambda point: point["event_index"], default=None)
        exposure = min(record["reads"], key=lambda point: point["event_index"], default=None)
        reuse = min((point for point in record["productions"] if exposure and point["event_index"] > exposure["event_index"]),
                    key=lambda point: point["event_index"], default=None)
        lag = None
        if reuse and exposure:
            lag = {"event_index": reuse["event_index"] - exposure["event_index"],
                   "recorded_sequence": reuse["sequence"] - exposure["sequence"] if reuse["sequence"] is not None and exposure["sequence"] is not None else None,
                   "community_episode_order": reuse["community_episode_order"] - exposure["community_episode_order"]}
        form_records.append({key: value for key, value in record.items() if key not in ("productions", "reads")})
        form_records[-1].update({"first_observed_production": production, "first_actual_read_exposure": exposure,
            "first_reuse_after_read": reuse, "exposure_to_first_reuse_lag": lag,
            "produced_before_first_read": bool(production and (not exposure or production["event_index"] < exposure["event_index"])),
            "produced_before_coinage_marker": bool(production and record["first_coinage_marker"]
                 and production["event_index"] < record["first_coinage_marker"]["event_index"]),
            "prior_exposure_unobserved": exposure is None,
            "semantic_adoption": "not_inferred"})
    known_communities = {value for episode, value in community_by_episode.items() if episode not in missing_community}
    for key, value in forms.items():
        marker = value["first_coinage_marker"]
        if marker and key not in messages[marker["message_id"]]["matches"]:
            warnings.append({"event_id": marker["event_id"], "form": value["form"], "code": "coinage_marker_without_matching_message_text"})
    if not rows:
        warnings.append({"code": "empty_event_log"})
    return {**base, "status": "valid_with_missing_metadata" if missing_community or warnings else "valid",
        "analysis_performed": True, "event_count": len(rows), "message_count": len(messages),
        "form_registry": [{**value, "form_key": key,
                           "classification": "seeded_baseline" if value["seeded"] else "explicit_coinage_marker_logged_in_run"}
                          for key, value in sorted(forms.items())],
        "form_records": form_records, "character_counts_by_stage": stage_characters,
        "recorded_accounting_by_stage": accounting, "episode_counters": list(counters.values()),
        "ceiling_findings": [], "failures": failures, "failure_count": len(failures),
        "missing_community_episode_ids": missing_community, "warnings": warnings,
        "reviewed_probes": _probes(reviewed_probe_results, known_communities),
        "lag_interpretation": "Ordered log event/sequence and community episode gaps, not elapsed time or a causal adoption rate.",
        "character_interpretation": "Unicode code points in original message text, counted once per logged send/delivery/read event; repeated reads count again. Characters are not model tokens."}
