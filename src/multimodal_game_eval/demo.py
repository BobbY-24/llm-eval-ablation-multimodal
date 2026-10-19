"""Synthetic end-to-end plumbing controls, kept separate from corpus candidates."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .core import DatasetError, read_json, read_jsonl
from .experiments import CONDITIONS, prepare_episodes, plan_runs, score_responses, validate_events
from .io import write_json, write_jsonl


def build_demo(output: Path, inventory: dict[str, dict[str, Any]], config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    candidates = []
    for target in config.get("target_sense_families", []):
        matches = [sense for sense in inventory.values() if sense["scope_id"] == target["scope_id"] and sense["term"] == target["family"]]
        if len(matches) != 1:
            raise DatasetError(f"Pilot target does not resolve uniquely: {target['family']}")
        sense = matches[0]
        candidates.append({**sense, "annotation_status": "candidate_requires_human_review", "media_status": "not_collected", "scorable": False})
    write_json(output / "pilot_sense_candidates.json", {"origin": "existing_inventory_candidates_not_gold", "senses": candidates})
    image = output / "synthetic_blue_square.ppm"
    image.write_bytes(b"P6\n2 2\n255\n" + bytes([0, 0, 255]) * 4)
    hidden_image = output / "synthetic_hidden_area.ppm"
    hidden_image.write_bytes(b"P6\n2 2\n255\n" + bytes([0, 0, 0]) * 4)
    fixture_inventory: dict[str, dict[str, Any]] = {}
    annotations = []
    for game_index in range(2):
        for case_index, label in enumerate(("PRESENT", "ABSENT", "INSUFFICIENT")):
            sense_id = f"fixture:sense:{game_index}:{case_index}"
            scope_id, group_id, usage_id = f"fixture:scope:{game_index}", f"fixture:term:{case_index}", f"fixture:usage:{game_index}:{case_index}"
            game = f"Synthetic fixture world {game_index + 1}"
            meaning = ("a visible blue square", "a visible red square", "a blue square outside the observed area")[case_index]
            fixture_inventory[sense_id] = {"scope_id": scope_id, "sense_id": sense_id, "term_group_id": group_id,
                "term": f"fixture-symbol-{case_index}", "game": game, "occurrence_ids": [usage_id], "occurrence_join_verified": True}
            for repetition in range(2):
                episode_id = f"fixture-episode-{game_index}-{case_index}-{repetition}"
                annotations.append({"synthetic_fixture": True, "episode_id": episode_id, "sample_id": f"fixture-sample-{episode_id}",
                    "term_group_id": group_id, "sense_id": sense_id, "scope_id": scope_id, "occurrence_ids": [usage_id],
                    "term": f"fixture-symbol-{case_index}", "aliases": [], "game": game, "version": "synthetic-v1",
                    "source_ids": [f"fixture-source-{episode_id}"], "replay_id": None, "session_id": None, "parent_ids": [],
                    "clip": {"start_seconds": 0, "end_seconds": 1},
                    "category": ("positive", "near_miss", "wrong_context")[case_index],
                    "information_match_verified": True,
                    "public": {"plain_description": meaning,
                        "evidence_text": "Synthetic control: the visible image is a blue square." if case_index < 2 else "Synthetic control: the target area is outside the observation; its color is unknown.",
                        "visual_refs": [{"ref": str(image if case_index < 2 else hidden_image), "timestamp_seconds": 0}], "action_options": []},
                    "reviews": [{"reviewer_id": f"synthetic-control-{reviewer}", "independent": True, "decision": "approve", "label": label,
                        "information_match_verified": True, "condition_labels": {condition: label for condition in CONDITIONS}}
                        for reviewer in ("a", "b")], "adjudication": None,
                    "gold": {"label": label, "critical_properties": {}, "event_interval_seconds": None, "valid_action_ids": None}})
    write_jsonl(output / "synthetic_annotations.jsonl", annotations)
    prepared = prepare_episodes(output / "synthetic_annotations.jsonl", output / "prepared", inventory=fixture_inventory)
    # Explicitly mark all derived files as synthetic in the enclosing provenance.
    gold = list(read_jsonl(output / "prepared/gold.jsonl"))
    correct = [{"item_id": row["item_id"], "label": row["label"], "critical_properties": row["critical_properties"]} for row in gold]
    incorrect = [{"item_id": row["item_id"], "label": "ABSENT" if row["label"] == "PRESENT" else "PRESENT", "critical_properties": {}} for row in gold]
    write_jsonl(output / "synthetic_responses_correct.jsonl", correct)
    write_jsonl(output / "synthetic_responses_incorrect.jsonl", incorrect)
    correct_scores = score_responses(output / "prepared/gold.jsonl", output / "synthetic_responses_correct.jsonl", output / "scores_correct")
    incorrect_scores = score_responses(output / "prepared/gold.jsonl", output / "synthetic_responses_incorrect.jsonl", output / "scores_incorrect")
    plan = plan_runs(output / "prepared/episodes.jsonl", {**config, "split": "all"})
    write_json(output / "dry_run_plan.json", {"synthetic_fixture": True, **plan})
    events = [
        {"event_id": "fixture-log-1", "episode_id": "fixture-log-episode", "type": "episode_start", "agent_ids": ["a", "b", "c"], "community_id": "fixture-community", "forum_exposure": True},
        {"event_id": "fixture-log-2", "episode_id": "fixture-log-episode", "type": "observation", "agent_id": "a", "observation_id": "fixture-private-view"},
        {"event_id": "fixture-log-3", "episode_id": "fixture-log-episode", "type": "message_sent", "sender": "a", "recipients": ["b"], "message_id": "fixture-message", "observation_ids": ["fixture-private-view"], "text": "Synthetic fixture message"},
        {"event_id": "fixture-log-4", "episode_id": "fixture-log-episode", "type": "message_delivered", "recipient": "b", "message_id": "fixture-message"},
        {"event_id": "fixture-log-5", "episode_id": "fixture-log-episode", "type": "message_read", "agent_id": "b", "message_id": "fixture-message"},
        {"event_id": "fixture-log-6", "episode_id": "fixture-log-episode", "type": "decision", "agent_id": "b", "observation_ids": ["fixture-private-view"], "message_ids": ["fixture-message"]},
        {"event_id": "fixture-log-7", "episode_id": "fixture-log-episode", "type": "episode_end"},
    ]
    write_jsonl(output / "synthetic_events.jsonl", events)
    event_findings = validate_events(output / "synthetic_events.jsonl")
    summary = {"evidence_type": "synthetic_plumbing_controls_not_llm_or_gameplay_results", "network_calls": 0,
        "real_media_samples_validated": 0, "real_term_candidates": len(candidates),
        "synthetic_eligible_episodes": prepared["eligible_episodes"], "synthetic_input_items": prepared["input_count"],
        "synthetic_correct_control_by_game": correct_scores["by_game_version"],
        "synthetic_incorrect_control_by_game": incorrect_scores["by_game_version"],
        "synthetic_event_findings": event_findings,
        "limitations": ["The candidate senses are source-linked but unreviewed and have no collected media.",
            "The synthetic reviewer IDs and labels are testing fixtures, not human annotation.",
            "This demo validates exports and scoring mechanics; it measures no LLM competence."]}
    write_json(output / "SUMMARY.json", summary)
    return {"summary_path": str(output / "SUMMARY.json"), "evidence_type": summary["evidence_type"],
            "real_term_candidates": len(candidates), "synthetic_eligible_episodes": prepared["eligible_episodes"],
            "synthetic_input_items": prepared["input_count"], "event_findings": event_findings, "network_calls": 0}
