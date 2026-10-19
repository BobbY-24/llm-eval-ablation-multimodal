"""Provider-neutral Stage 2 request rendering from blinded inputs only. No provider is called.

Rendering accepts exactly the allowlisted keys that ``prepare`` projects into
``inputs.jsonl`` ``input`` objects and refuses anything else, so gold, reviews,
candidate metadata and inventory definitions cannot reach a request through
this path. Frames come first (a shared, cacheable prefix across the wording
arms of one example), then text evidence, then the worded question.
"""
from __future__ import annotations

import random
from typing import Any

from .core import DatasetError

INPUT_KEYS = frozenset({"task", "response_schema", "game", "version", "terminology", "evidence", "clip", "action_options"})
LABELS = ("PRESENT", "ABSENT", "INSUFFICIENT")


def _check_input(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or not set(payload) <= INPUT_KEYS:
        extra = sorted(set(payload) - INPUT_KEYS) if isinstance(payload, dict) else ["<not an object>"]
        raise DatasetError("Refusing non-allowlisted input keys: " + ", ".join(extra))
    terminology = payload.get("terminology") or {}
    if terminology.get("kind") not in ("jargon", "plain") or not isinstance(terminology.get("expression"), str) \
            or not terminology["expression"].strip():
        raise DatasetError("Input terminology must be jargon/plain with a nonempty expression")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or not set(evidence) <= {"text", "visual_refs"}:
        raise DatasetError("Evidence may contain only text and visual_refs")


def _frames(evidence: dict[str, Any], max_frames: int) -> list[dict[str, Any]]:
    refs = evidence.get("visual_refs")
    if refs is None:
        return []
    if not isinstance(refs, list) or len(refs) > max_frames:
        raise DatasetError(f"Visual evidence must be a list of at most {max_frames} references")
    frames = []
    for ref in refs:
        ref = {"ref": ref} if isinstance(ref, str) else ref
        if not isinstance(ref, dict) or not isinstance(ref.get("ref"), str):
            raise DatasetError("Visual reference requires a string ref")
        frames.append({"type": "image_ref", "ref": ref["ref"], "timestamp_seconds": ref.get("timestamp_seconds"),
                       "media_type": ref.get("media_type")})
    return frames


def render_label_request(payload: dict[str, Any], templates: dict[str, Any], seed: int,
                         max_frames: int = 64) -> dict[str, Any]:
    """Render one grounding request; the seed only permutes the listed label order."""
    _check_input(payload)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DatasetError("Seed must be an integer")
    terminology, evidence = payload["terminology"], payload["evidence"]
    options = list(LABELS)
    random.Random(seed).shuffle(options)
    frames = _frames(evidence, max_frames)
    content: list[dict[str, Any]] = []
    if frames:
        content.append({"type": "text", "text": templates["frames_preamble"]})
        for frame in frames:
            content.append({"type": "text", "text": f"t = {frame['timestamp_seconds']} s"})
            content.append(frame)
    if "text" in evidence:
        content.append({"type": "text", "text": templates["text_evidence_preamble"] + "\n" + str(evidence["text"])})
    question = templates["label_question"][terminology["kind"]].format(expression=terminology["expression"], game=payload["game"])
    clip = payload.get("clip") or {}
    instructions = templates["label_instructions"].format(start_seconds=clip.get("start_seconds"), end_seconds=clip.get("end_seconds"),
                                                          label_options=", ".join(options))
    content.append({"type": "text", "text": question + "\n" + instructions})
    return {"template_id": templates["template_id"], "seed": seed, "label_order": options,
            "system": templates["system"].format(game=payload["game"]), "content": content,
            "response_schema": templates["label_response_schema"], "frame_count": len(frames)}


def render_property_request(payload: dict[str, Any], templates: dict[str, Any], properties: list[dict[str, Any]],
                            max_frames: int = 64) -> dict[str, Any]:
    """Neutral-wording property probe on the same evidence; the term and plain description are omitted."""
    _check_input(payload)
    lines = []
    for item in properties:
        if not isinstance(item.get("key"), str) or not isinstance(item.get("values"), list):
            raise DatasetError("Property items need key and values")
        if item.get("visibility") == "not_observable":
            continue
        lines.append(f"- {item['key']}: one of {', '.join(map(str, item['values']))}, insufficient")
    frames = _frames(payload["evidence"], max_frames)
    content: list[dict[str, Any]] = []
    if frames:
        content.append({"type": "text", "text": templates["frames_preamble"]})
        for frame in frames:
            content.append({"type": "text", "text": f"t = {frame['timestamp_seconds']} s"})
            content.append(frame)
    if "text" in payload["evidence"]:
        content.append({"type": "text", "text": templates["text_evidence_preamble"] + "\n" + str(payload["evidence"]["text"])})
    content.append({"type": "text", "text": templates["property_question"] + "\n" + "\n".join(lines)})
    return {"template_id": templates["template_id"], "system": templates["system"].format(game=payload["game"]),
            "content": content, "response_schema": templates["property_response_schema"], "frame_count": len(frames)}
