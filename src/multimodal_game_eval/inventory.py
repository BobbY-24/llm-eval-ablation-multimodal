"""Read-only joins into the existing community-term inventory and source graph."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .core import DatasetError, read_json, safe_path


def load_inventory(path: Path | str, graph_path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    path = Path(path)
    raw = read_json(path)
    if not isinstance(raw, dict) or not isinstance(raw.get("scopes"), list):
        raise DatasetError("Expected the existing inventory's scopes array")
    if graph_path is None:
        source = raw.get("audit", {}).get("source")
        candidate = safe_path(path.parent.parent, source) if source else None
        if candidate is not None and candidate.is_file():
            graph_path = candidate
    graph = read_json(graph_path) if graph_path else {}
    usages = graph.get("usages", [])
    if isinstance(usages, dict):
        usages = list(usages.values())
    by_sense: dict[str, list[str]] = {}
    evidence_by_sense: dict[str, list[dict[str, Any]]] = {}
    by_group: dict[str, list[str]] = {}
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        occurrence_id = usage.get("occurrence_id") or usage.get("usage_id") or usage.get("id")
        if not occurrence_id:
            continue
        sense_ids = usage.get("sense_ids", [])
        if usage.get("sense_id"):
            sense_ids = [*sense_ids, usage["sense_id"]]
        for sense_id in sense_ids:
            by_sense.setdefault(sense_id, []).append(occurrence_id)
            evidence_by_sense.setdefault(sense_id, []).append({key: usage.get(key) for key in (
                "occurrence_id", "scope_id", "source_id", "source_url", "content_hash", "exact_quote", "context", "assignment_status")})
        group_id = usage.get("term_group_id") or usage.get("original_group_id")
        if group_id:
            by_group.setdefault(group_id, []).append(occurrence_id)
    result: dict[str, dict[str, Any]] = {}
    for scope in raw["scopes"]:
        for term in scope.get("terms", []):
            sense_id = term["sense_id"]
            if sense_id in result:
                raise DatasetError(f"Duplicate sense ID: {sense_id}")
            # A group's usages in another game are not evidence for this sense.
            occurrence_ids = sorted(set(by_sense.get(sense_id, [])))
            result[sense_id] = {**term, "scope_id": scope["scope_id"], "scope_type": scope.get("scope_type"), "game": scope["label"], "occurrence_ids": occurrence_ids, "evidence": evidence_by_sense.get(sense_id, []), "occurrence_join_verified": bool(occurrence_ids), "inventory_path": str(path.resolve())}
    return result


def validate_annotation_join(annotation: dict[str, Any], inventory: dict[str, dict[str, Any]]) -> list[str]:
    sense = inventory.get(annotation.get("sense_id", ""))
    if sense is None:
        return ["unknown_sense_id"]
    errors = [f"mismatched_{field}" for field in ("term_group_id", "scope_id", "game") if annotation.get(field) != sense.get(field)]
    occurrence_ids = annotation.get("occurrence_ids", [])
    if not isinstance(occurrence_ids, list) or not occurrence_ids:
        errors.append("missing_occurrence_ids")
    elif not sense.get("occurrence_join_verified"):
        errors.append("occurrence_join_unverified")
    elif not set(occurrence_ids).issubset(set(sense["occurrence_ids"])):
        errors.append("unknown_occurrence_id")
    return errors
