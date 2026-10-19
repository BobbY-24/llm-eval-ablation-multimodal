"""Offline preparation, planning, imported-response scoring, and event auditing.

No function in this module calls a provider, downloads an asset, renders a replay,
or executes generated code. JSON/JSONL is data throughout.

Annotation JSONL contract (one row per proposed term/sample pairing)::

    {"episode_id": "e1", "sample_id": "s1", "sense_id": "sense1",
     "term_group_id": "g1", "scope_id": "scope1", "occurrence_ids": ["o1"],
     "term": "short hop", "aliases": [], "game": "Melee", "version": "...",
     "source_ids": ["source-component"], "replay_id": "r1",
     "session_id": "session1", "parent_ids": [],
     "clip": {"start_seconds": 0, "end_seconds": 2}, "category": "positive",
     "public": {"evidence_text": "...", "visual_refs": ["asset:frame1"],
                "plain_description": "a short-duration jump",
                "action_options": [{"action_id": "jump", "text": "Jump"}]},
     "information_match_verified": true,
     "reviews": [{"reviewer_id": "a", "independent": true,
                  "decision": "approve", "information_match_verified": true,
                  "condition_labels": {"jargon_text": "PRESENT", ...}}, ...],
     "adjudication": null,
     "gold": {"label": "PRESENT", "critical_properties": {"duration": "short"},
              "event_interval_seconds": [0.2, 0.5], "valid_action_ids": null}}

Two distinct independent approving reviewers must supply all six condition_labels
matching their protected rubrics. Disagreement instead requires a third distinct
approving adjudicator's complete ``resolved_condition_labels``. Shared gold also
requires annotation and both reviewers' information_match_verified:true. Optional
gold_by_condition contains six full arm-specific rubrics, allowing reviewed
INSUFFICIENT labels for unidentifiable arms. Jargon/plain gold within a modality
must agree. An empty export template is unscored.
Critical-property scoring is exact schema agreement, NOT a semantic/paraphrase
judge: use a dict of canonical property keys to scalar values, or a list of
canonical strings. Annotation instructions must define this response convention.

``export_annotations`` writes the supplied output JSONL. ``prepare_episodes``
writes ``episodes.jsonl`` (public registry), ``inputs.jsonl``, ``gold.jsonl``,
``unscored.jsonl`` and ``manifest.json``. The manifest hashes every JSONL file.
Inference projects ONLY each input row's ``input`` object. Its explicit allowlist
is task, response_schema, game, version, terminology, evidence, clip, and optional
action_options; source/index metadata and gold never automatically enter it.
Six conditions are jargon/plain crossed with text/visual/both. Split labels are
``dev`` and ``eval``. Connected source/replay/session/parent/episode identities
remain together; terms intentionally recur as known senses across partitions.

``plan_runs`` consumes episodes.jsonl and returns a finite, dry-run-only plan.
``score_responses`` consumes one replication's JSONL (one response per item_id),
and optionally writes scores.json and score_details.jsonl. Missing, duplicate,
and invalid responses score as errors. Replications use separate response files.
``validate_events`` returns findings without changing any file. Its reference
checks depend on complete truthful logs; they cannot prove absence of unlogged
information leakage or establish that a natural-language message is grounded.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from .core import MAX_JSONL_LINE_BYTES, DatasetError, read_json, read_jsonl, sha256_file


LABELS = ("PRESENT", "ABSENT", "INSUFFICIENT")
CONDITIONS = tuple(f"{term}_{mode}" for term in ("jargon", "plain")
                   for mode in ("text", "visual", "both"))
CATEGORIES = ("positive", "near_miss", "wrong_context")
REGIMES = ("independent", "fixed_workflow", "peer")
SCHEMA_VERSION = "multimodal-game-eval-v1"
MAX_EXPERIMENT_BYTES = 64 * 1024 * 1024
MAX_EXPERIMENT_ROWS = 100_000


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_new(path, json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False, allow_nan=False) + "\n")


def _write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) > MAX_EXPERIMENT_ROWS:
        raise ValueError("experiment_row_limit_exceeded")
    _write_new(path, "".join(_canonical(row) + "\n" for row in rows))


def _write_new(path, text):
    if path.exists() or path.is_symlink():
        raise ValueError("refusing_to_overwrite_existing_output:" + str(path))
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_EXPERIMENT_BYTES:
        raise ValueError("experiment_output_byte_limit_exceeded")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".experiment-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link creation is atomic and cannot replace a concurrently created file.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fresh_output_directory(path):
    path = Path(path)
    if path.is_symlink() or path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("output_directory_must_be_new_or_empty:" + str(path))


def _hash_file(path):
    return sha256_file(path, max_bytes=MAX_EXPERIMENT_BYTES)


def _read_rows(path):
    rows = []
    for value in read_jsonl(path, max_bytes=MAX_EXPERIMENT_BYTES):
        if len(rows) >= MAX_EXPERIMENT_ROWS:
            raise ValueError("experiment_row_limit_exceeded")
        rows.append(value)
    return rows


def _tolerant_rows(path):
    """Bounded decoder preserving malformed response/event lines as findings."""
    consumed = count = 0
    with Path(path).open("rb") as stream:
        line_number = 0
        while raw := stream.readline(MAX_JSONL_LINE_BYTES + 1):
            line_number += 1
            consumed += len(raw)
            if len(raw) > MAX_JSONL_LINE_BYTES or consumed > MAX_EXPERIMENT_BYTES:
                raise DatasetError("Experiment JSONL exceeds line/file byte limit")
            if not raw.strip():
                continue
            count += 1
            if count > MAX_EXPERIMENT_ROWS:
                raise ValueError("experiment_row_limit_exceeded")
            try:
                row = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite")))
            except (ValueError, UnicodeError):
                row = None
            yield line_number, row


def _strings(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return list(dict.fromkeys(v for v in value if isinstance(v, str) and v))
    return []


def _ids(row, singular, plural):
    return list(dict.fromkeys(_strings(row.get(plural)) + _strings(row.get(singular))))


def _finite_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _interval(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(_finite_number(v) for v in value)
            and 0 <= value[0] < value[1])


def _property_map(value):
    if isinstance(value, list):
        if not all(isinstance(v, str) and v for v in value) or len(set(value)) != len(value):
            raise ValueError("invalid_critical_properties")
        return {v: True for v in value}
    if isinstance(value, dict):
        if any(not isinstance(k, str) or not k for k in value):
            raise ValueError("invalid_property_key")
        for v in value.values():
            if not (v is None or isinstance(v, (str, bool)) or _finite_number(v)):
                raise ValueError("property_values_must_be_finite_scalars")
        return dict(value)
    raise ValueError("critical_properties_must_be_object_or_string_list")


def _public_fields(row):
    public = row.get("public") if isinstance(row.get("public"), dict) else {}
    # Never copy arbitrary adapter/inventory metadata into public model input.
    result = {key: public.get(key, row.get(key, default)) for key, default in (
        ("evidence_text", ""), ("visual_refs", []), ("plain_description", ""),
        ("action_options", []))}
    if isinstance(result["visual_refs"], list):
        result["visual_refs"] = list(result["visual_refs"])
    if not result["visual_refs"] and isinstance(row.get("assets"), dict):
        spatial = any("spatial" in value for value in _strings(row.get("modalities")))
        for key, ref in row["assets"].items():
            if key in ("video", "image", "spatial", "encoded_map", "encoded_png") and isinstance(ref, str) and ref:
                result["visual_refs"].append({"ref": ref, "media_type": "encoded_spatial" if spatial or key in ("spatial", "encoded_map", "encoded_png") else key})
    return result


def _normalized_lineage(sample):
    """Use media origins/hashes, never the generic dataset documentation URL."""
    lineage = sample.get("lineage") if isinstance(sample.get("lineage"), dict) else {}
    namespace = str(sample.get("dataset") or "local")
    source_ids = _ids(sample, "source_id", "source_ids")
    source_ids += _ids(lineage, "source_id", "source_ids")
    replay_id = sample.get("replay_id")
    session_id = sample.get("session_id")
    for key in ("replay_id", "source_replay_id", "replay_name", "trajectory"):
        value = lineage.get(key)
        if not replay_id and isinstance(value, str) and value:
            replay_id = namespace + ":replay:" + value
    for key in ("session_id", "session_hint", "recording_file"):
        value = lineage.get(key)
        if not session_id and isinstance(value, str) and value:
            session_id = namespace + ":session:" + value
    for key in ("source_video_id", "source_video", "source_image", "recording_file", "trajectory", "segment_stem"):
        value = lineage.get(key)
        if isinstance(value, str) and value:
            source_ids.append(namespace + ":media_origin:" + value)
    media_hashes = {}
    hashes = sample.get("hashes") if isinstance(sample.get("hashes"), dict) else {}
    for key, value in hashes.items():
        if (isinstance(key, str) and not any(v in key.lower() for v in ("metadata", "manifest", "label"))
                and isinstance(value, str) and len(value) == 64
                and all(c in "0123456789abcdefABCDEF" for c in value)):
            media_hashes[key] = value.lower()
            source_ids.append("media-sha256:" + value.lower())
    parent_ids = _ids(sample, "parent_id", "parent_ids") + _ids(lineage, "parent_id", "parent_ids")
    parent_ids += _strings(lineage.get("derived_from")) + _strings(lineage.get("parent_sample_id"))
    parent_replay = lineage.get("parent_replay_id")
    if isinstance(parent_replay, str) and parent_replay:
        parent_ids.append(namespace + ":replay:" + parent_replay)
    if replay_id:
        source_ids.append(replay_id)
    if session_id:
        source_ids.append(session_id)
    return {"source_ids": list(dict.fromkeys(source_ids)), "replay_id": replay_id,
            "session_id": session_id, "parent_ids": list(dict.fromkeys(parent_ids)),
            "media_hashes": media_hashes}


def export_annotations(samples: list[dict], inventory: dict, output: Path,
                       selections: dict | None = None) -> dict:
    """Export explicitly selected sense/sample candidates, never fabricated gold.

    selections maps sample_id to a sense_id or an object containing sense_id and
    optional episode_id/category. Without a selection, an explicit sample sense_id
    is used; otherwise a single unresolved template is retained for that sample.
    """
    if not isinstance(samples, list) or not all(isinstance(s, dict) for s in samples):
        raise ValueError("samples_must_be_object_list")
    if len(samples) > MAX_EXPERIMENT_ROWS:
        raise ValueError("experiment_row_limit_exceeded")
    if not isinstance(inventory, dict) or (selections is not None and not isinstance(selections, dict)):
        raise ValueError("inventory_and_selections_must_be_objects")
    rows, seen = [], set()
    for sample in samples:
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample_id_required")
        if sample_id in seen:
            raise ValueError("duplicate_sample_id:" + sample_id)
        seen.add(sample_id)
        selected = (selections or {}).get(sample_id, sample.get("sense_id", ""))
        selection = selected if isinstance(selected, dict) else {"sense_id": selected}
        sense_id = selection.get("sense_id", "")
        if not isinstance(sense_id, str):
            raise ValueError("selected_sense_id_must_be_string")
        sense = inventory.get(sense_id, {}) if isinstance(sense_id, str) else {}
        if sense_id and sense_id not in inventory:
            raise ValueError("unknown_selected_sense:" + str(sense_id))
        if not isinstance(sense, dict):
            raise ValueError("inventory_sense_must_be_object")
        group_ids = _ids(sense, "term_group_id", "term_group_ids") or _strings(sense.get("original_group_ids"))
        occurrences = _ids(sense, "occurrence_id", "occurrence_ids") or _strings(sense.get("all_occurrence_ids"))
        row = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": selection.get("episode_id", sample.get("episode_id")) or
                          "episode-" + _digest([sample_id, sense_id])[:16],
            "sample_id": sample_id, "sense_id": sense_id or "",
            "term_group_id": sample.get("term_group_id", group_ids[0] if len(group_ids) == 1 else ""),
            "scope_id": sample.get("scope_id", sense.get("scope_id", "")),
            "occurrence_ids": _strings(sample.get("occurrence_ids")) or occurrences,
            "term": sense.get("term", sense.get("form", sample.get("term", ""))),
            "aliases": _strings(sense.get("aliases")),
            "game": sample.get("game", sense.get("game", "")),
            "version": sample.get("version", sense.get("version", "unknown")),
            **_normalized_lineage(sample),
            "clip": sample.get("clip"),
            "category": selection.get("category", sample.get("category", "positive")),
            "public": _public_fields(sample), "reviews": [], "adjudication": None,
            "information_match_verified": False,
            "gold": None, "annotation_status": "candidate_unreviewed",
        }
        rows.append(row)
    _write_rows(output, rows)
    return {"schema_version": SCHEMA_VERSION, "output": str(output),
            "annotations_path": str(output), "samples": len(rows),
            "selected_senses": sum(bool(row["sense_id"]) for row in rows),
            "scored": 0, "status": "candidate_templates_only",
            "sha256": _hash_file(output)}


def _inventory_reasons(row, inventory):
    if inventory is None:
        return []
    sense_id = row.get("sense_id")
    sense = inventory.get(sense_id) if isinstance(sense_id, str) else None
    if not isinstance(sense, dict):
        return ["sense_not_in_inventory"]
    reasons = []
    if row.get("term_group_id") != sense.get("term_group_id") or not sense.get("term_group_id"):
        reasons.append("term_group_inventory_join_failed")
    if row.get("scope_id") != sense.get("scope_id") or not sense.get("scope_id"):
        reasons.append("scope_inventory_join_failed")
    if row.get("game") != sense.get("game") or not sense.get("game"):
        reasons.append("game_inventory_join_failed")
    occurrences = _strings(sense.get("occurrence_ids"))
    given = _strings(row.get("occurrence_ids"))
    if sense.get("occurrence_join_verified") is not True or not occurrences or not given or not set(given) <= set(occurrences):
        reasons.append("occurrence_inventory_join_failed")
    return reasons


def _rubrics(row):
    arms = row.get("gold_by_condition")
    if isinstance(arms, dict):
        return {condition: arms.get(condition) for condition in CONDITIONS}
    return {condition: row.get("gold") for condition in CONDITIONS}


def _rubric_reasons(gold, clip, option_ids, visibility):
    reasons = []
    if not isinstance(gold, dict) or gold.get("label") not in LABELS:
        return ["missing_reviewed_label"]
    try:
        _property_map(gold.get("critical_properties", {}))
    except ValueError:
        reasons.append("invalid_gold_critical_properties")
    event = gold.get("event_interval_seconds")
    if event is not None:
        if not _interval(event) or gold.get("label") != "PRESENT":
            reasons.append("invalid_gold_event_interval")
        elif isinstance(clip, dict) and _interval([clip.get("start_seconds"), clip.get("end_seconds")]) and not (
                clip["start_seconds"] <= event[0] < event[1] <= clip["end_seconds"]):
            reasons.append("gold_interval_outside_clip")
    valid_actions = gold.get("valid_action_ids")
    if valid_actions is not None:
        if not isinstance(valid_actions, list) or not valid_actions or any(
                not isinstance(v, str) or not v for v in valid_actions) or len(set(valid_actions)) != len(valid_actions):
            reasons.append("invalid_gold_action_ids")
        elif not set(valid_actions) <= set(option_ids):
            reasons.append("gold_actions_not_in_public_options")
    if gold.get("evidence_status", visibility) in ("unknown", "out_of_view") and gold.get("label") != "INSUFFICIENT":
        reasons.append("unknown_visibility_requires_reviewed_insufficient_label")
    return reasons


def _eligibility(row, inventory):
    reasons = []
    for field in ("episode_id", "sample_id", "term_group_id", "sense_id", "scope_id", "term", "game", "version"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            reasons.append("missing_" + field)
    occurrence_ids = row.get("occurrence_ids")
    if not isinstance(occurrence_ids, list) or not occurrence_ids:
        reasons.append("missing_occurrence_ids")
    elif any(not isinstance(v, str) or not v for v in occurrence_ids) or len(set(occurrence_ids)) != len(occurrence_ids):
        reasons.append("invalid_occurrence_ids")
    if not _ids(row, "source_id", "source_ids"):
        reasons.append("missing_source_component")
    if row.get("category") not in CATEGORIES:
        reasons.append("invalid_category")
    clip = row.get("clip")
    if not isinstance(clip, dict) or not _interval([clip.get("start_seconds"), clip.get("end_seconds")]):
        reasons.append("invalid_clip_range")
    public = _public_fields(row)
    if not isinstance(public["evidence_text"], str) or not public["evidence_text"].strip():
        reasons.append("missing_public_text_evidence")
    visuals = public["visual_refs"]
    if not isinstance(visuals, list) or not visuals or any(
            not (isinstance(v, str) and v or isinstance(v, dict) and isinstance(v.get("ref"), str) and v["ref"])
            for v in visuals):
        reasons.append("missing_or_invalid_visual_refs")
    if not isinstance(public["plain_description"], str) or not public["plain_description"].strip():
        reasons.append("missing_nonanswer_plain_description")
    options = public["action_options"]
    if not isinstance(options, list) or any(not isinstance(v, dict)
            or not isinstance(v.get("action_id"), str) or not v["action_id"]
            or not isinstance(v.get("text"), str) for v in options):
        reasons.append("invalid_public_action_options")
        options = []
    option_ids = [v["action_id"] for v in options]
    if len(option_ids) != len(set(option_ids)):
        reasons.append("duplicate_public_action_options")
    rubrics = _rubrics(row)
    for rubric in rubrics.values():
        reasons.extend(_rubric_reasons(rubric, clip, option_ids, row.get("visibility_status")))
    condition_labels = {condition: value.get("label") if isinstance(value, dict) else None
                        for condition, value in rubrics.items()}
    explicit_arms = "gold_by_condition" in row
    if explicit_arms and (not isinstance(row["gold_by_condition"], dict)
                          or set(row["gold_by_condition"]) != set(CONDITIONS)):
        reasons.append("all_six_condition_rubrics_required")
    for modality in ("text", "visual", "both"):
        a, b = rubrics["jargon_" + modality], rubrics["plain_" + modality]
        if isinstance(a, dict) and isinstance(b, dict):
            if any(a.get(key) != b.get(key) for key in (
                    "label", "critical_properties", "event_interval_seconds", "valid_action_ids")):
                reasons.append("terminology_pair_gold_mismatch")
    reviews = row.get("reviews")
    approving = [r for r in reviews if isinstance(r, dict)
                 and r.get("independent") is True and r.get("decision") == "approve"
                 and isinstance(r.get("reviewer_id"), str) and r["reviewer_id"]
                 and isinstance(r.get("condition_labels"), dict)
                 and set(r["condition_labels"]) == set(CONDITIONS)
                 and all(v in LABELS for v in r["condition_labels"].values())] if isinstance(reviews, list) else []
    reviewers = [r["reviewer_id"] for r in approving]
    if len(approving) < 2 or len(set(reviewers)) != len(reviewers):
        reasons.append("two_distinct_independent_reviews_required")
    elif all(r["condition_labels"] == approving[0]["condition_labels"] for r in approving):
        if approving[0]["condition_labels"] != condition_labels:
            reasons.append("gold_disagrees_with_reviewers")
    else:
        adjudication = row.get("adjudication")
        if (not isinstance(adjudication, dict) or adjudication.get("decision") != "approve"
                or not isinstance(adjudication.get("reviewer_id"), str)
                or not adjudication["reviewer_id"] or adjudication["reviewer_id"] in reviewers
                or adjudication.get("resolved_condition_labels") != condition_labels):
            reasons.append("distinct_adjudication_matching_gold_required")
    match_verified = (row.get("information_match_verified") is True
                      and len(approving) >= 2
                      and all(r.get("information_match_verified") is True for r in approving))
    if not explicit_arms and not match_verified:
        reasons.append("shared_gold_requires_verified_information_match")
    if row.get("information_match_verified") is True and not match_verified:
        reasons.append("information_match_not_affirmed_by_reviewers")
    reasons.extend(_inventory_reasons(row, inventory))
    return sorted(set(reasons))


def _components(rows):
    parents = {}

    def find(value):
        parents.setdefault(value, value)
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    row_nodes = []
    for index, row in enumerate(rows):
        node = "row:" + str(index)
        find(node)
        identifiers = _normalized_lineage(row)["source_ids"] + _normalized_lineage(row)["parent_ids"]
        for singular, plural in (("episode_id", "episode_ids"), ("sample_id", "sample_ids"),
                                 ("replay_id", "replay_ids"), ("session_id", "session_ids"),
                                 ("source_id", "source_ids"), ("parent_id", "parent_ids")):
            identifiers.extend(_ids(row, singular, plural))
        # One identity namespace lets a derived parent reference any original ID.
        for identifier in identifiers:
            union(node, "identity:" + identifier)
        row_nodes.append(node)
    groups = defaultdict(list)
    for index, node in enumerate(row_nodes):
        groups[find(node)].append(index)
    return list(groups.values())


def _project_input(row, condition):
    terminology, modality = condition.split("_")
    public = _public_fields(row)
    payload = {
        "task": "Decide whether the named mechanic is evidenced in the supplied clip/context. "
                "Use INSUFFICIENT when the supplied evidence cannot settle the decision. "
                "Report critical properties using the annotation protocol's canonical keys/values. "
                "Report a seconds interval only for an observable event; choose an action only when options are supplied.",
        "response_schema": {"label": list(LABELS), "critical_properties": "object_or_canonical_string_list",
                            "event_interval_seconds": "[start,end]_or_null", "action_id": "string_or_null",
                            "confidence": "optional_finite_probability"},
        "game": row["game"], "version": row["version"],
        "terminology": {"kind": terminology, "expression": row["term"] if terminology == "jargon" else public["plain_description"]},
        "evidence": {},
        "clip": {"start_seconds": row["clip"]["start_seconds"], "end_seconds": row["clip"]["end_seconds"]},
    }
    if modality in ("text", "both"):
        payload["evidence"]["text"] = public["evidence_text"]
    if modality in ("visual", "both"):
        # Keep an allowlisted media reference/timestamp projection, not arbitrary metadata.
        payload["evidence"]["visual_refs"] = [v if isinstance(v, str) else {
            k: v[k] for k in ("ref", "timestamp_seconds", "media_type") if k in v
        } for v in public["visual_refs"]]
    if public["action_options"]:
        payload["action_options"] = [{"action_id": v["action_id"], "text": v["text"]}
                                     for v in public["action_options"]]
    return payload


def prepare_episodes(annotation_path: Path, output_dir: Path, seed: int = 42,
                     dev_fraction: float = .25, inventory: dict | None = None) -> dict:
    """Gate annotations, split connected lineage, and project six blinded inputs.

    An absent inventory is explicitly disclosed, never silently called validated.
    All rows (including unscored rows) participate in component construction so an
    unreviewed intermediate parent cannot bridge development and evaluation.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed_must_be_integer")
    if not _finite_number(dev_fraction) or not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction_must_be_between_zero_and_one")
    if inventory is not None and not isinstance(inventory, dict):
        raise ValueError("inventory_must_be_object")
    rows = _read_rows(annotation_path)
    _fresh_output_directory(output_dir)
    episode_counts = Counter(r.get("episode_id") for r in rows if isinstance(r.get("episode_id"), str))
    eligibility = [_eligibility(row, inventory) for row in rows]
    for index, row in enumerate(rows):
        if isinstance(row.get("episode_id"), str) and episode_counts.get(row["episode_id"], 0) > 1:
            eligibility[index].append("duplicate_episode_id")
    components = _components(rows)
    eligible_components = [indices for indices in components if any(not eligibility[i] for i in indices)]
    family_by_row = {}
    families = []
    for indices in eligible_components:
        identity = sorted({identifier for i in indices for singular, plural in (
            ("episode_id", "episode_ids"), ("sample_id", "sample_ids"), ("source_id", "source_ids"),
            ("replay_id", "replay_ids"), ("session_id", "session_ids"), ("parent_id", "parent_ids"))
                           for identifier in _ids(rows[i], singular, plural)})
        family_id = "family-" + _digest(identity)[:20]
        families.append(family_id)
        for i in indices:
            family_by_row[i] = family_id
    shuffled = sorted(families)
    random.Random(seed).shuffle(shuffled)
    n_dev = round(len(shuffled) * dev_fraction)
    if len(shuffled) > 1:
        n_dev = max(1, min(len(shuffled) - 1, n_dev))
    elif shuffled:
        n_dev = 1
    development = set(shuffled[:n_dev])
    inputs, gold_rows, episodes, unscored = [], [], [], []
    for index, row in enumerate(rows):
        if eligibility[index]:
            unscored.append({"episode_id": row.get("episode_id"), "sample_id": row.get("sample_id"),
                             "sense_id": row.get("sense_id"), "reasons": sorted(set(eligibility[index]))})
            continue
        family_id = family_by_row[index]
        split = "dev" if family_id in development else "eval"
        item_ids = {}
        for condition in CONDITIONS:
            item_id = "item-" + _digest([row["episode_id"], condition])[:24]
            item_ids[condition] = item_id
            common = {"item_id": item_id, "episode_id": row["episode_id"],
                      "family_id": family_id, "split": split, "condition": condition}
            inputs.append({**common, "input": _project_input(row, condition)})
            protected = _rubrics(row)[condition]
            gold_rows.append({**common, "sample_id": row["sample_id"],
                "sense_id": row["sense_id"], "term_group_id": row["term_group_id"],
                "scope_id": row["scope_id"], "game": row["game"], "version": row["version"],
                "category": row["category"], "label": protected["label"],
                "information_match_verified": row.get("information_match_verified") is True,
                "critical_properties": _property_map(protected.get("critical_properties", {})),
                "event_interval_seconds": protected.get("event_interval_seconds"),
                "valid_action_ids": protected.get("valid_action_ids"),
                "public_action_ids": [a["action_id"] for a in _public_fields(row)["action_options"]]})
        episodes.append({"episode_id": row["episode_id"], "family_id": family_id,
                         "split": split, "game": row["game"], "version": row["version"],
                         "item_ids": item_ids})
    if len(inputs) > MAX_EXPERIMENT_ROWS:
        raise ValueError("six_condition_output_exceeds_row_limit")
    output_dir = Path(output_dir)
    filenames = {"episodes": "episodes.jsonl", "inputs": "inputs.jsonl", "gold": "gold.jsonl",
                 "unscored": "unscored.jsonl", "manifest": "manifest.json"}
    for key, value in (("episodes", episodes), ("inputs", inputs), ("gold", gold_rows), ("unscored", unscored)):
        _write_rows(output_dir / filenames[key], value)
    warnings = []
    if inventory is None:
        warnings.append("inventory_joins_not_validated")
    if len(families) < 2:
        warnings.append("fewer_than_two_independent_lineage_components_no_locked_comparison")
    if not episodes:
        warnings.append("no_reviewed_eligible_episodes")
    manifest = {"schema_version": SCHEMA_VERSION, "mode": "offline_preparation",
        "annotation_sha256": _hash_file(annotation_path),
        "inventory_validation": "performed" if inventory is not None else "not_performed",
        "inventory_sha256": _digest(inventory) if inventory is not None else None,
        "seed": seed, "dev_fraction": dev_fraction, "conditions": list(CONDITIONS),
        "filenames": filenames, "files": {key: {"path": str(output_dir / filename),
            "sha256": _hash_file(output_dir / filename)}
            for key, filename in filenames.items() if key != "manifest"},
        "annotation_count": len(rows), "eligible_episodes": len(episodes), "unscored_count": len(unscored),
        "input_count": len(inputs), "family_count": len(families),
        "episode_splits": dict(Counter(r["split"] for r in episodes)),
        "unscored_reasons": dict(Counter(reason for r in unscored for reason in r["reasons"])),
        "warnings": warnings, "projection_contract": "model receives inputs.jsonl row.input only",
        "limitations": ["Known senses may recur; this is not unseen-sense generalization.",
                         "Assets and nonanswer public descriptions require human leakage review.",
                         "No simulator, renderer, agent interaction, or model performance is produced."]}
    _write_json(output_dir / filenames["manifest"], manifest)
    return {**manifest, **{key + "_path": str(output_dir / filename) for key, filename in filenames.items()}}


def _bounded_int(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError("invalid_" + name)
    return value


def _grounding_cost_scenarios(spec, requests, output_cap):
    """Per-model grounding scenarios from stated token assumptions; null prices stay unknown."""
    if spec is None:
        return None
    if not isinstance(spec, dict) or not isinstance(spec.get("models"), list) or not spec["models"]:
        raise ValueError("cost_scenarios_require_models")
    by_condition = spec.get("input_tokens_by_condition")
    if not isinstance(by_condition, dict) or any(not _finite_number(v) or v < 0 for v in by_condition.values()):
        raise ValueError("invalid_input_tokens_by_condition")
    if any(r["condition"] not in by_condition for r in requests):
        raise ValueError("input_tokens_missing_for_planned_condition")
    assumed_output = spec.get("output_tokens_per_request")
    discount = spec.get("batch_discount", 0)
    if not _finite_number(assumed_output) or not 0 <= assumed_output <= output_cap:
        raise ValueError("invalid_output_tokens_per_request")
    if not _finite_number(discount) or not 0 <= discount < 1:
        raise ValueError("invalid_batch_discount")
    input_tokens = sum(by_condition[r["condition"]] for r in requests)
    output_assumed, output_upper = len(requests) * assumed_output, len(requests) * output_cap
    rows = []
    for model in spec["models"]:
        if not isinstance(model, dict) or not isinstance(model.get("model"), str):
            raise ValueError("invalid_cost_model")
        prices = [model.get("input_usd_per_million"), model.get("output_usd_per_million")]
        if any(p is not None and (not _finite_number(p) or p < 0) for p in prices):
            raise ValueError("invalid_cost_model_price")
        known = None not in prices

        def usd(output):
            return round((input_tokens * prices[0] + output * prices[1]) / 1e6 * (1 - discount), 2) if known else None
        rows.append({"model": model["model"], "price_source": model.get("price_source"),
                     "input_usd_per_million": prices[0], "output_usd_per_million": prices[1],
                     "estimated_usd_assumed_output": usd(output_assumed),
                     "estimated_usd_output_cap": usd(output_upper),
                     "status": "configured_scenario_not_provider_quote" if known else "unknown_prices"})
    return {"requests": len(requests), "input_tokens": input_tokens,
            "output_tokens_assumed": output_assumed, "output_tokens_cap": output_upper,
            "batch_discount_applied": discount, "assumptions": spec.get("assumptions", []), "models": rows}


def plan_runs(episodes_path: Path, config: dict) -> dict:
    """Return counts and community schedules only; estimates are not invoices.

    Flat config: split, seeds, regimes, conditions, agents, rounds, max_turns,
    episodes_per_community, newcomer_after, and optional cost_estimates containing
    input_tokens_per_turn, output_tokens_per_turn, input_usd_per_million,
    output_usd_per_million. Missing prices yield null estimates, never free calls.
    Stage 2 options: grounding_only skips community schedules;
    grounding_max_output_tokens caps grounding requests; property_probe_modalities
    adds separate neutral-wording property requests; cost_scenarios gives
    per-condition input-token assumptions and per-model prices.
    """
    if not isinstance(config, dict) or config.get("live") is True or config.get("execute") is True:
        raise ValueError("offline_plan_config_required")
    rows = _read_rows(episodes_path)
    split = config.get("split", "eval")
    if split not in ("dev", "eval", "all"):
        raise ValueError("invalid_split")
    seeds = config.get("seeds", [11, 23, 37])
    if (not isinstance(seeds, list) or not seeds or len(seeds) > 100
            or any(isinstance(v, bool) or not isinstance(v, int) for v in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("invalid_seeds")
    regimes = config.get("regimes", config.get("communication_regimes", list(REGIMES)))
    conditions = config.get("conditions", config.get("grounding_conditions", list(CONDITIONS)))
    for value, allowed, name in ((regimes, REGIMES, "regimes"), (conditions, CONDITIONS, "conditions")):
        if not isinstance(value, list) or not value or any(v not in allowed for v in value) or len(set(value)) != len(value):
            raise ValueError("invalid_" + name)
    agents = _bounded_int(config.get("agents", config.get("agent_count", 3)), "agents", 1, 18)
    rounds = _bounded_int(config.get("rounds", 6), "rounds", 1, 18)
    max_turns = _bounded_int(config.get("max_turns", config.get("turns", 18)), "max_turns", 1, 18)
    if agents * rounds > max_turns:
        raise ValueError("agents_times_rounds_exceeds_turn_cap")
    community_size = _bounded_int(config.get("episodes_per_community", config.get("community_episodes", 10)), "episodes_per_community", 2, 1000)
    newcomer_after = _bounded_int(config.get("newcomer_after", config.get("newcomer_after_episode", 5)), "newcomer_after", 1, community_size - 1)
    max_tokens = _bounded_int(config.get("max_tokens_per_turn", 256), "max_tokens_per_turn", 1, 4096)
    max_frames = _bounded_int(config.get("max_media_frames", 64), "max_media_frames", 1, 256)
    exposures = config.get("forum_exposures", ["enabled", "withheld"])
    if (not isinstance(exposures, list) or not exposures or len(exposures) > 2
            or any(value not in ("enabled", "withheld") for value in exposures)
            or len(set(exposures)) != len(exposures)):
        raise ValueError("invalid_forum_exposures")
    selected = [r for r in rows if split == "all" or r.get("split") == split]
    if any(not isinstance(r.get("episode_id"), str) or not r["episode_id"] for r in selected):
        raise ValueError("invalid_episode_registry")
    if len({r.get("episode_id") for r in selected}) != len(selected):
        raise ValueError("duplicate_episode_in_registry")
    if len(selected) * len(seeds) * len(regimes) * len(conditions) * len(exposures) > MAX_EXPERIMENT_ROWS:
        raise ValueError("planned_episode_trial_limit_exceeded")
    groups, family_splits = defaultdict(list), defaultdict(set)
    for row in selected:
        if row.get("split") not in ("dev", "eval") or not isinstance(row.get("item_ids"), dict) or any(
                not isinstance(row.get(key), str) or not row[key] for key in ("game", "version", "episode_id", "family_id")):
            raise ValueError("invalid_episode_registry")
        if any(not isinstance(row["item_ids"].get(condition), str) for condition in conditions):
            raise ValueError("missing_condition_item_id")
        groups[(row.get("game"), row.get("version"), row["split"])].append(row)
        family_splits[row["family_id"]].add(row["split"])
    if any(len(splits) > 1 for splits in family_splits.values()):
        raise ValueError("family_crosses_splits")
    grounding = [{"request_id": "grounding-" + _digest([row["episode_id"], condition, seed])[:20],
                  "episode_id": row["episode_id"], "item_id": row["item_ids"][condition],
                  "split": row["split"], "condition": condition, "seed": seed,
                  "max_output_tokens": max_tokens, "max_media_frames": max_frames}
                 for row in sorted(selected, key=lambda value: value["episode_id"])
                 for condition, seed in itertools.product(conditions, seeds)]
    grounding_only = config.get("grounding_only", False)
    if not isinstance(grounding_only, bool):
        raise ValueError("invalid_grounding_only")
    grounding_output = _bounded_int(config.get("grounding_max_output_tokens", max_tokens),
                                    "grounding_max_output_tokens", 1, 32768)
    for request in grounding:
        request["max_output_tokens"] = grounding_output
    modalities = config.get("property_probe_modalities", [])
    if (not isinstance(modalities, list) or len(set(modalities)) != len(modalities)
            or any(m not in ("text", "visual", "both") for m in modalities)):
        raise ValueError("invalid_property_probe_modalities")
    if any(not isinstance(row["item_ids"].get("plain_" + m), str) for row in selected for m in modalities):
        raise ValueError("missing_condition_item_id")
    # Property questions use neutral wording shared by both terminology arms, in
    # a separate request, so property keys cannot define the jargon term.
    property_probes = [{"request_id": "property-probe-" + _digest([row["episode_id"], mode, seed])[:20],
                        "episode_id": row["episode_id"], "evidence_item_id": row["item_ids"]["plain_" + mode],
                        "split": row["split"], "condition": "property_probe_" + mode,
                        "wording": "neutral_property_questions", "seed": seed,
                        "max_output_tokens": grounding_output, "max_media_frames": max_frames}
                       for row in sorted(selected, key=lambda value: value["episode_id"])
                       for mode, seed in itertools.product(modalities, seeds)]
    communities, trials, probe_calls = [], 0, 0
    for (game, version, row_split), group in sorted(groups.items(), key=lambda value: str(value[0])):
        if grounding_only:
            break
        for seed, regime, condition in itertools.product(seeds, regimes, conditions):
            ordered = sorted(group, key=lambda row: row["episode_id"])
            random.Random(_digest([seed, game, version, row_split])).shuffle(ordered)
            for start in range(0, len(ordered), community_size):
                chunk = ordered[start:start + community_size]
                identity = [game, version, row_split, seed, regime, condition, start]
                assigned = list(exposures)
                random.Random(_digest(identity + ["forum_assignment"])).shuffle(assigned)
                for slot, exposure in enumerate(assigned):
                    probes = ([{"after_completed_episodes": newcomer_after, "kind": "arrival_meaning_and_action", "input_item_id": None,
                                "freshness_status": "requires_reviewed_fresh_examples_outside_community_exposure"},
                               {"after_completed_episodes": len(chunk), "kind": "retention_meaning_and_action", "input_item_id": None,
                                "freshness_status": "requires_reviewed_fresh_examples_outside_community_exposure"}]
                              if len(chunk) > newcomer_after else [])
                    communities.append({"community_id": "community-" + _digest(identity + [slot])[:20],
                        "matched_group_id": "matched-" + _digest(identity)[:20],
                        "forum_assignment": exposure, "forum_exposure": exposure == "enabled",
                        "game": game, "version": version, "split": row_split, "seed": seed,
                        "regime": regime, "condition": condition,
                        "episode_ids": [r["episode_id"] for r in chunk],
                        "item_ids": [r["item_ids"][condition] for r in chunk],
                        "complete_ten_episode_window": len(chunk) == community_size,
                        "agents": agents, "rounds": rounds, "turn_cap_per_episode": max_turns,
                        "newcomer": {"after_completed_episodes": newcomer_after,
                                     "replace_agent_slot": agents,
                                     "occurs_in_this_window": len(chunk) > newcomer_after},
                        "newcomer_probes": probes})
                    trials += len(chunk)
                    probe_calls += len(probes)
    turn_upper = trials * agents * rounds
    request_upper = turn_upper + len(grounding) + len(property_probes) + probe_calls
    scenarios = _grounding_cost_scenarios(config.get("cost_scenarios"), grounding + property_probes,
                                          grounding_output)
    estimate_config = config.get("cost_estimates", {})
    if not isinstance(estimate_config, dict):
        raise ValueError("cost_estimates_must_be_object")
    cost_keys = ("input_tokens_per_turn", "output_tokens_per_turn", "input_usd_per_million", "output_usd_per_million")
    for name in cost_keys:
        if name in estimate_config and (not _finite_number(estimate_config[name]) or estimate_config[name] < 0):
            raise ValueError("invalid_cost_estimate_" + name)
    available = all(name in estimate_config for name in cost_keys)
    dollars = (request_upper * (estimate_config["input_tokens_per_turn"] * estimate_config["input_usd_per_million"]
               + estimate_config["output_tokens_per_turn"] * estimate_config["output_usd_per_million"]) / 1e6
               if available else None)
    return {"schema_version": SCHEMA_VERSION, "mode": "dry_run_only", "network_calls": 0,
        "episodes_path": str(episodes_path), "episodes_sha256": _hash_file(episodes_path),
        "selected_episodes": len(selected), "seeds": seeds, "regimes": regimes,
        "conditions": conditions, "agents": agents, "rounds": rounds, "max_turns": max_turns,
        "episodes_per_community": community_size, "newcomer_after": newcomer_after,
        "forum_exposures": exposures, "max_tokens_per_turn": max_tokens, "max_media_frames": max_frames,
        "grounding_only": grounding_only, "grounding_max_output_tokens": grounding_output,
        "grounding_request_count": len(grounding), "grounding_requests": grounding,
        "property_probe_request_count": len(property_probes), "property_probe_requests": property_probes,
        "grounding_cost_scenarios": scenarios,
        "episode_trials": trials, "community_count": len(communities),
        "team_model_turn_upper_bound": turn_upper, "model_turn_upper_bound": turn_upper,
        "newcomer_probe_request_count": probe_calls, "model_request_upper_bound": request_upper,
        "output_token_upper_bound": request_upper * max_tokens, "communities": communities,
        "resource_estimate": {"model_request_upper_bound": request_upper,
            "output_token_upper_bound": request_upper * max_tokens,
            "media_frame_exposure_upper_bound": request_upper * max_frames,
            "input_tokens_per_request_assumption": estimate_config.get("input_tokens_per_turn"),
            "input_token_upper_bound": None,
            "status": "request_output_media_ceilings_known_input_context_and_provider_costs_require_verification",
            "live_run_gate": "resolve_probe_items_and_bound_input_context_and_cost_before_live_execution"},
        "cost_estimates": {"assumptions": estimate_config, "estimated_usd": dollars,
                           "status": "configured_scenario_not_provider_quote" if available else "unknown_prices_or_token_estimates"},
        "limitations": ["Schedules are plans, not simulated or observed interactions.",
                         "Each regime receives the same maximum model-turn allowance.",
                         "Forum assignment is randomized within isolated matched pairs; messages and outcomes are not fabricated.",
                         "Media tokenization and actual provider usage need separate verification.",
                         "Fresh newcomer probe item IDs remain unassigned; select reviewed examples not previously exposed in the community.",
                         "Incomplete community windows cannot establish ten-episode newcomer retention."]}


def _parse_response(record, target):
    answer = record.get("answer", record)
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except ValueError as exc:
            raise ValueError("answer_not_json_object") from exc
    if not isinstance(answer, dict) or answer.get("label") not in LABELS:
        raise ValueError("invalid_label_or_answer")
    properties = _property_map(answer.get("critical_properties", {}))
    event = answer.get("event_interval_seconds")
    if event is not None and not _interval(event):
        raise ValueError("invalid_event_interval")
    action = answer.get("action_id")
    if action is not None and (not isinstance(action, str) or action not in target.get("public_action_ids", [])):
        raise ValueError("action_not_in_public_options")
    if "confidence" in answer and (not _finite_number(answer["confidence"]) or not 0 <= answer["confidence"] <= 1):
        raise ValueError("invalid_confidence")
    return {"label": answer["label"], "critical_properties": properties,
            "event_interval_seconds": event, "action_id": action}


def _iou(a, b):
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return overlap / (max(a[1], b[1]) - min(a[0], b[0]))


def _mean(values):
    return sum(values) / len(values) if values else None


def _bootstrap(records, key, iterations=1000):
    groups = defaultdict(list)
    for row in records:
        if row.get(key) is not None:
            groups[row["family_id"]].append(row[key])
    if not groups:
        return None
    families = sorted(groups)
    rng = random.Random(20261008)
    samples = []
    for _ in range(iterations):
        values = [value for family in rng.choices(families, k=len(families)) for value in groups[family]]
        samples.append(sum(values) / len(values))
    samples.sort()
    return [samples[int(.025 * iterations)], samples[min(iterations - 1, int(.975 * iterations))]]


def _metrics(records, bootstrap_iterations=1000):
    properties_n = sum(r["property_required_n"] for r in records)
    predicted_n = sum(r["property_predicted_n"] for r in records)
    property_correct = sum(r["property_correct_n"] for r in records)
    return {"items": len(records), "source_families": len({r["family_id"] for r in records}),
        "response_statuses": dict(Counter(r["status"] for r in records)),
        "interpretation_accuracy": _mean([r["interpretation_correct"] for r in records]),
        "interpretation_family_bootstrap_95_ci": _bootstrap(records, "interpretation_correct", bootstrap_iterations),
        "uncertainty_warning": "fewer_than_two_source_families" if len({r["family_id"] for r in records}) < 2 else None,
        "critical_property_required_n": properties_n,
        "critical_property_exact_recall": property_correct / properties_n if properties_n else None,
        "critical_property_predicted_n": predicted_n,
        "critical_property_exact_precision": property_correct / predicted_n if predicted_n else (0.0 if properties_n else None),
        "property_set_exact_accuracy": _mean([r["property_set_exact"] for r in records if r["property_required_n"]]),
        "temporal_items": sum(r["temporal_iou"] is not None for r in records),
        "temporal_mean_iou": _mean([r["temporal_iou"] for r in records if r["temporal_iou"] is not None]),
        "temporal_iou_ge_05": _mean([float(r["temporal_iou"] >= .5) for r in records if r["temporal_iou"] is not None]),
        "action_items": sum(r["action_correct"] is not None for r in records),
        "valid_action_accuracy": _mean([r["action_correct"] for r in records if r["action_correct"] is not None])}


def score_responses(gold_path: Path, responses_path: Path,
                    output_dir: Path | None = None, bootstrap_iterations: int = 1000) -> dict:
    """Score one imported replication; duplicates invalidate their target item.

    Response row: {item_id, label, critical_properties, event_interval_seconds,
    action_id, confidence?}; the answer fields may instead be inside ``answer``.
    Missing properties/actions/intervals count as misses when a reviewed gold
    target exists. No temporal/action target is invented for unannotated cases.
    Per-game/version scores are separate; no unlike-game leaderboard is produced.
    """
    _bounded_int(bootstrap_iterations, "bootstrap_iterations", 100, 10_000)
    gold = _read_rows(gold_path)
    if output_dir is not None:
        _fresh_output_directory(output_dir)
    manifest_path = Path(gold_path).parent / "manifest.json"
    manifest_status = "not_available"
    if manifest_path.exists():
        manifest = read_json(manifest_path, max_bytes=MAX_EXPERIMENT_BYTES)
        files = manifest.get("files") if isinstance(manifest, dict) else None
        gold_file = files.get("gold") if isinstance(files, dict) else None
        if isinstance(gold_file, dict) and isinstance(gold_file.get("sha256"), str):
            if _hash_file(gold_path) != gold_file["sha256"]:
                raise ValueError("gold_manifest_hash_mismatch")
            manifest_status = "gold_hash_verified"
    targets = {}
    for row in gold:
        if not isinstance(row.get("item_id"), str) or not row["item_id"] or row["item_id"] in targets:
            raise ValueError("invalid_or_duplicate_gold_item_id")
        if row.get("label") not in LABELS or row.get("condition") not in CONDITIONS:
            raise ValueError("invalid_gold_label_or_condition")
        for field in ("family_id", "episode_id", "sense_id", "game", "version"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError("invalid_gold_" + field)
        _property_map(row.get("critical_properties", {}))
        event = row.get("event_interval_seconds")
        if event is not None and (not _interval(event) or row["label"] != "PRESENT"):
            raise ValueError("invalid_gold_interval")
        actions = row.get("valid_action_ids")
        if actions is not None:
            if (not isinstance(actions, list) or not actions or any(not isinstance(v, str) or not v for v in actions)
                    or len(set(actions)) != len(actions) or not set(actions) <= set(_strings(row.get("public_action_ids")))):
                raise ValueError("invalid_gold_action_ids")
        targets[row["item_id"]] = row
    findings, received = [], defaultdict(list)
    for line_number, row in _tolerant_rows(responses_path):
        if row is None:
            findings.append({"line": line_number, "code": "malformed_response_json"})
            continue
        if not isinstance(row, dict) or not isinstance(row.get("item_id"), str):
            findings.append({"line": line_number, "code": "response_item_id_required"})
            continue
        if row["item_id"] not in targets:
            findings.append({"line": line_number, "item_id": row["item_id"], "code": "unknown_response_item"})
            continue
        received[row["item_id"]].append(row)
    details = []
    for item_id, target in targets.items():
        records = received[item_id]
        answer, reason = None, None
        if not records:
            status = "missing"
        elif len(records) > 1:
            status, reason = "duplicate", "multiple_responses_for_one_replication"
        else:
            try:
                answer = _parse_response(records[0], target)
                status = "valid"
            except ValueError as exc:
                status, reason = "invalid", str(exc)
        if reason:
            findings.append({"item_id": item_id, "code": reason})
        expected = _property_map(target.get("critical_properties", {}))
        predicted = answer["critical_properties"] if answer else {}
        matches = sum(key in predicted and type(predicted[key]) is type(value)
                      and predicted[key] == value for key, value in expected.items())
        label_correct = float(bool(answer and answer["label"] == target["label"]))
        event = target.get("event_interval_seconds")
        temporal = (_iou(event, answer["event_interval_seconds"]) if answer and label_correct
                    and answer["event_interval_seconds"] is not None else 0.0) if event is not None else None
        actions = target.get("valid_action_ids")
        action_correct = float(bool(answer and answer["action_id"] in actions)) if actions is not None else None
        details.append({"item_id": item_id, "episode_id": target["episode_id"],
            "family_id": target["family_id"], "split": target.get("split"),
            "game": target["game"], "version": target["version"], "sense_id": target["sense_id"],
            "information_match_verified": target.get("information_match_verified") is True,
            "condition": target["condition"], "status": status, "reason": reason,
            "interpretation_correct": label_correct, "property_required_n": len(expected),
            "property_predicted_n": len(predicted), "property_correct_n": matches,
            "property_set_exact": float(bool(answer and predicted == expected
                 and all(type(predicted[k]) is type(v) for k, v in expected.items()))),
            "temporal_iou": temporal, "action_correct": action_correct})
    by_game, by_sense, by_condition = defaultdict(list), defaultdict(list), defaultdict(list)
    paired_lookup = defaultdict(dict)
    for row in details:
        game_key = _canonical([row["game"], row["version"]])
        by_game[game_key].append(row)
        by_sense[_canonical([row["game"], row["version"], row["sense_id"]])].append(row)
        by_condition[_canonical([row["game"], row["version"], row["condition"]])].append(row)
        available = paired_lookup[(game_key, row["episode_id"])]
        if row["condition"] in available:
            raise ValueError("duplicate_gold_episode_condition")
        available[row["condition"]] = row
    pairs = {}
    for game_key in sorted(by_game):
        for first, second in itertools.combinations(CONDITIONS, 2):
            records = []
            complete = 0
            for (candidate_game, _), available in paired_lookup.items():
                if candidate_game != game_key or first not in available or second not in available:
                    continue
                a, b = available[first], available[second]
                complete += a["status"] == "valid" and b["status"] == "valid"
                records.append({"family_id": a["family_id"],
                                "information_match_verified": a["information_match_verified"] and b["information_match_verified"],
                                "difference": b["interpretation_correct"] - a["interpretation_correct"]})
            if records:
                pairs[_canonical([game_key, second, "minus", first])] = {
                    "game_version": json.loads(game_key), "second": second, "first": first,
                    "planned_pair_items": len(records), "both_responses_valid": complete,
                    "source_families": len({r["family_id"] for r in records}),
                    "information_match_verified": all(r["information_match_verified"] for r in records),
                    "interpretation": ("controlled_terminology_contrast" if first.split("_")[1] == second.split("_")[1]
                        and all(r["information_match_verified"] for r in records) else "diagnostic_not_pure_lexical_effect"),
                    "interpretation_accuracy_difference": _mean([r["difference"] for r in records]),
                    "family_bootstrap_95_ci": _bootstrap(records, "difference", bootstrap_iterations),
                    "denominator_policy": "all_gold_pairs_missing_and_invalid_count_as_wrong"}
    result = {"schema_version": SCHEMA_VERSION, "evidence_type": "imported_responses_not_execution",
        "gold_sha256": _hash_file(gold_path), "gold_manifest_validation": manifest_status,
        "responses_sha256": _hash_file(responses_path), "bootstrap_iterations": bootstrap_iterations,
        "expected_items": len(targets), "response_statuses": dict(Counter(r["status"] for r in details)),
        "valid_response_coverage": sum(r["status"] == "valid" for r in details) / len(details) if details else None,
        "by_game_version": {key: _metrics(value, bootstrap_iterations) for key, value in sorted(by_game.items())},
        "by_sense": {key: _metrics(value, bootstrap_iterations) for key, value in sorted(by_sense.items())},
        "by_condition": {key: _metrics(value, bootstrap_iterations) for key, value in sorted(by_condition.items())},
        "paired_condition_differences": pairs, "findings": findings,
        "limitations": ["Property metrics measure exact canonical schema agreement, not semantic paraphrase quality.",
                         "Temporal/action denominators contain only reviewed targets; missing outputs count as misses.",
                         "Source-family bootstrap does not make a small or convenience corpus representative.",
                         "This scorer does not establish gameplay optimality, language emergence, or causal diffusion."]}
    if output_dir is not None:
        result["scores_path"] = str(Path(output_dir) / "scores.json")
        result["details_path"] = str(Path(output_dir) / "score_details.jsonl")
        _write_json(Path(output_dir) / "scores.json", result)
        _write_rows(Path(output_dir) / "score_details.jsonl", details)
    return result


def validate_events(events_path: Path) -> list[dict]:
    """Audit explicit information access in an ordered JSONL interaction log.

    Types: episode_start (agent_ids), observation (agent_id, observation_id),
    message_sent (message_id, sender, recipients, observation_ids),
    message_delivered/message_read (message_id, recipient or agent_id),
    decision (agent_id, observation_ids, message_ids), newcomer_join,
    forum_post/forum_read (message_id, community_id, recipients or
    visibility:"community"), episode_start forum_exposure:boolean,
    memory_update, episode_end. Each row needs event_id and episode_id.
    A read must follow an actual permitted delivery. Reading grants access to the
    observation references conveyed in that message; membership alone does not.
    Forum reads may cross episodes only within their logged community. Decision
    and memory observation/message references are checked against actual access.
    """
    findings, events = [], []
    for line_number, row in _tolerant_rows(events_path):
        if row is None:
            findings.append({"line": line_number, "code": "malformed_event_json"})
            continue
        if not isinstance(row, dict):
            findings.append({"line": line_number, "code": "event_object_required"})
            continue
        events.append((line_number, row))
    event_ids, messages, observed, granted, delivered, read = set(), {}, {}, defaultdict(set), set(), set()
    teams, communities, forum_exposure, closed, last_sequence = {}, {}, {}, set(), None
    aliases = {"delivery": "message_delivered", "read": "message_read", "message": "message_sent",
               "private_observation": "observation", "join": "newcomer_join"}
    known = {"episode_start", "observation", "message_sent", "message_delivered", "message_read",
             "decision", "memory_update", "forum_post", "forum_read", "newcomer_join", "episode_end"}

    def issue(line, row, code, **extra):
        findings.append({"line": line, "event_id": row.get("event_id"), "code": code, **extra})

    for line, row in events:
        event_id, episode = row.get("event_id"), row.get("episode_id")
        if not isinstance(event_id, str) or not event_id or not isinstance(episode, str) or not episode:
            issue(line, row, "event_and_episode_ids_required")
            continue
        if event_id in event_ids:
            issue(line, row, "duplicate_event_id")
            continue
        event_ids.add(event_id)
        sequence = row.get("sequence")
        if sequence is not None:
            if isinstance(sequence, bool) or not isinstance(sequence, int) or (last_sequence is not None and sequence <= last_sequence):
                issue(line, row, "nonmonotonic_or_invalid_sequence")
            elif last_sequence is None or sequence > last_sequence:
                last_sequence = sequence
        raw_kind = row.get("type", row.get("event_type"))
        kind = aliases.get(raw_kind, raw_kind) if isinstance(raw_kind, str) else None
        if kind not in known:
            issue(line, row, "unknown_event_type")
            continue
        malformed_references = False
        for field in ("observation_ids", "used_observation_ids", "message_ids", "depends_on_message_ids", "recipients", "visible_to"):
            if field in row:
                value = row[field]
                if (not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value)
                        or len(set(value)) != len(value)):
                    issue(line, row, "invalid_reference_list", field=field)
                    malformed_references = True
        if malformed_references:
            continue
        if kind == "episode_start":
            if episode in teams:
                issue(line, row, "duplicate_episode_start")
                continue
            members = _strings(row.get("agent_ids"))
            if not members:
                issue(line, row, "episode_agent_ids_required")
            teams[episode] = set(members)
            communities[episode] = row.get("community_id")
            if "forum_exposure" in row and not isinstance(row["forum_exposure"], bool):
                issue(line, row, "forum_exposure_must_be_boolean")
            forum_exposure[episode] = row.get("forum_exposure") is True
            continue
        if episode not in teams:
            issue(line, row, "episode_start_missing")
            continue
        if episode in closed:
            issue(line, row, "event_after_episode_end")
            continue
        actor = row.get("agent_id", row.get("recipient", row.get("sender")))
        if kind == "newcomer_join":
            if not isinstance(actor, str) or not actor:
                issue(line, row, "newcomer_agent_id_required")
            elif actor in teams[episode]:
                issue(line, row, "newcomer_already_member")
            else:
                replacement = row.get("replace_agent_id")
                if replacement is not None and (not isinstance(replacement, str) or replacement not in teams[episode]):
                    issue(line, row, "replacement_agent_not_in_team")
                else:
                    if replacement is not None:
                        teams[episode].remove(replacement)
                    teams[episode].add(actor)
            continue
        if kind == "episode_end":
            closed.add(episode)
            continue
        if not isinstance(actor, str) or actor not in teams[episode]:
            issue(line, row, "agent_not_in_episode_team")
            continue
        access_key = (episode, actor)
        if row.get("sender") is not None and kind in ("message_sent", "forum_post") and row["sender"] != actor:
            issue(line, row, "sender_agent_mismatch")
            continue
        if row.get("recipient") is not None and kind in ("message_delivered", "message_read", "forum_read") and row["recipient"] != actor:
            issue(line, row, "recipient_agent_mismatch")
            continue
        if kind == "observation":
            observation = row.get("observation_id")
            if not isinstance(observation, str) or not observation or observation in observed:
                issue(line, row, "invalid_or_duplicate_observation_id")
                continue
            audience = _strings(row.get("visible_to")) or [actor]
            if actor not in audience or not set(audience) <= teams[episode]:
                issue(line, row, "invalid_observation_audience")
                continue
            observed[observation] = episode
            for member in audience:
                granted[(episode, member)].add(observation)
            continue
        if kind in ("message_sent", "forum_post"):
            message = row.get("message_id")
            if not isinstance(message, str) or not message or message in messages:
                issue(line, row, "invalid_or_duplicate_message_id")
                continue
            recipients = _strings(row.get("recipients", row.get("audience")))
            community_visible = kind == "forum_post" and row.get("visibility") == "community"
            if not community_visible and (not recipients or not set(recipients) <= teams[episode]):
                issue(line, row, "invalid_message_audience")
                continue
            refs = _strings(row.get("observation_ids"))
            bad = set(refs) - granted[access_key]
            if bad:
                issue(line, row, "sender_uses_unseen_observation", observation_ids=sorted(bad))
                continue
            community = row.get("community_id", communities.get(episode))
            if kind == "forum_post" and (not isinstance(community, str) or not community
                                         or community != communities.get(episode)):
                issue(line, row, "forum_community_required_or_mismatch")
                continue
            messages[message] = {"episode": episode, "sender": actor, "recipients": set(recipients),
                                 "observations": refs, "forum": kind == "forum_post", "community": community,
                                 "community_visible": community_visible}
            continue
        if kind in ("message_delivered", "message_read", "forum_read"):
            message_id = row.get("message_id")
            message = messages.get(message_id) if isinstance(message_id, str) else None
            if message is None:
                issue(line, row, "message_not_sent")
                continue
            same_scope = message["episode"] == episode
            if message["forum"]:
                same_scope = (message["community"] == communities.get(episode)
                              and row.get("community_id", communities.get(episode)) == message["community"])
                if forum_exposure[episode] is not True:
                    issue(line, row, "forum_access_withheld")
                    continue
            permitted_recipient = actor in message["recipients"] or message["forum"] and message["community_visible"]
            if not same_scope or not permitted_recipient:
                issue(line, row, "message_outside_permitted_audience_or_scope")
                continue
            key = (episode, actor, message_id)
            if kind == "message_delivered":
                if key in delivered:
                    issue(line, row, "duplicate_delivery")
                delivered.add(key)
                continue
            if kind == "forum_read" and not message["forum"]:
                issue(line, row, "forum_read_requires_forum_post")
                continue
            if kind == "message_read" and key not in delivered:
                issue(line, row, "read_without_actual_delivery")
                continue
            read.add(key)
            granted[access_key].update(message["observations"])
            continue
        if kind in ("decision", "memory_update"):
            bad_observations = set(_strings(row.get("observation_ids", row.get("used_observation_ids")))) - granted[access_key]
            if bad_observations:
                issue(line, row, "decision_or_memory_uses_unseen_observation", observation_ids=sorted(bad_observations))
            referenced = _strings(row.get("message_ids", row.get("depends_on_message_ids")))
            for message_id in referenced:
                message = messages.get(message_id)
                if (episode, actor, message_id) not in read and not (
                        message and message["episode"] == episode and message["sender"] == actor):
                    issue(line, row, "decision_or_memory_uses_unread_message", message_id=message_id)
    return findings
