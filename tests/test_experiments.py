"""Small offline fixtures for the experiment contracts; no model or media calls."""

import json
import tempfile
import unittest
from pathlib import Path

from multimodal_game_eval.experiments import (
    CONDITIONS, export_annotations, plan_runs, prepare_episodes,
    score_responses, validate_events,
)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def annotation(index=1, **overrides):
    row = {
        "episode_id": f"episode{index}", "sample_id": f"sample{index}",
        "sense_id": "sense", "term_group_id": "group", "scope_id": "scope",
        "occurrence_ids": ["occurrence"], "term": "hop", "aliases": ["short hop"],
        "game": "fixture-game", "version": "fixture-v1",
        "source_ids": [f"source{index}"], "replay_id": f"replay{index}",
        "session_id": f"session{index}", "parent_ids": [],
        "clip": {"start_seconds": 0, "end_seconds": 2}, "category": "positive",
        "information_match_verified": True,
        "public": {"evidence_text": "A fictional movement rule.",
                   "visual_refs": [{"ref": f"asset{index}", "timestamp_seconds": .5,
                                    "private_label": "PRESENT"}],
                   "plain_description": "a short-duration jump",
                   "action_options": [{"action_id": "jump", "text": "Jump"},
                                      {"action_id": "wait", "text": "Wait"}]},
        "reviews": [{"reviewer_id": "reviewer-a", "independent": True,
                     "decision": "approve", "label": "PRESENT", "information_match_verified": True,
                     "condition_labels": {condition: "PRESENT" for condition in CONDITIONS}},
                    {"reviewer_id": "reviewer-b", "independent": True,
                     "decision": "approve", "label": "PRESENT", "information_match_verified": True,
                     "condition_labels": {condition: "PRESENT" for condition in CONDITIONS}}],
        "adjudication": None,
        "gold": {"label": "PRESENT", "critical_properties": {"duration": "short"},
                 "event_interval_seconds": [.2, .8], "valid_action_ids": ["jump"]},
        "private_note": "do not expose", "correct_action": "jump",
    }
    row.update(overrides)
    return row


def inventory():
    return {"sense": {"term": "hop", "term_group_id": "group", "scope_id": "scope",
                      "game": "fixture-game", "occurrence_join_verified": True,
                      "occurrence_ids": ["occurrence"], "aliases": ["short hop"]}}


class ExperimentFixtures(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.preparation_count = 0

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, rows, **kwargs):
        path = self.root / "annotations.jsonl"
        write_rows(path, rows)
        self.preparation_count += 1
        return prepare_episodes(path, self.root / f"prepared-{self.preparation_count}", inventory=inventory(), **kwargs)


class PreparationTests(ExperimentFixtures):

    def test_export_never_fabricates_reviews_or_gold(self):
        sample = annotation()
        result = export_annotations([sample], inventory(), self.root / "export.jsonl")
        row = read_rows(self.root / "export.jsonl")[0]
        self.assertEqual(result["scored"], 0)
        self.assertIsNone(row["gold"])
        self.assertEqual(row["reviews"], [])
        self.assertNotIn("private_note", row)

    def test_export_prefills_nested_media_lineage_without_generic_url_merging(self):
        samples = [{"sample_id": "first", "dataset": "dataset", "game": "fixture-game",
                    "source": {"url": "https://example.invalid/dataset-docs"},
                    "lineage": {"source_replay_id": "match1"}, "assets": {"video": "first.mp4"},
                    "hashes": {"video_sha256": "a" * 64}},
                   {"sample_id": "second", "dataset": "dataset", "game": "fixture-game",
                    "source": {"url": "https://example.invalid/dataset-docs"},
                    "lineage": {"source_replay_id": "match2"}, "assets": {"image": "second.png"},
                    "hashes": {"image_sha256": "b" * 64}}]
        path = self.root / "export.jsonl"
        export_annotations(samples, inventory(), path)
        first, second = read_rows(path)
        self.assertIn("dataset:replay:match1", first["source_ids"])
        self.assertIn("media-sha256:" + "a" * 64, first["source_ids"])
        self.assertFalse(set(first["source_ids"]) & set(second["source_ids"]))
        self.assertEqual(first["public"]["visual_refs"], [{"ref": "first.mp4", "media_type": "video"}])
        self.assertEqual(second["public"]["visual_refs"], [{"ref": "second.png", "media_type": "image"}])

    def test_exact_game_inventory_join_is_required(self):
        result = self.prepare([annotation(game="different-game")])
        self.assertEqual(result["eligible_episodes"], 0)
        self.assertIn("game_inventory_join_failed", result["unscored_reasons"])

    def test_six_conditions_and_explicit_projection(self):
        result = self.prepare([annotation(1), annotation(2)])
        inputs = read_rows(Path(result["inputs_path"]))
        self.assertEqual(len(inputs), 12)
        self.assertEqual({row["condition"] for row in inputs}, set(CONDITIONS))
        for row in inputs:
            payload = row["input"]
            self.assertNotIn("gold", payload)
            self.assertNotIn("sense_id", payload)
            self.assertNotIn("correct_action", payload)
            self.assertNotIn("private_note", payload)
            self.assertNotIn("private_label", json.dumps(payload))
            self.assertEqual(len(payload["action_options"]), 2)
            mode = row["condition"].split("_")[1]
            self.assertEqual("text" in payload["evidence"], mode in ("text", "both"))
            self.assertEqual("visual_refs" in payload["evidence"], mode in ("visual", "both"))

    def test_duplicate_reviewer_and_unreviewed_candidates_are_unscored(self):
        one = annotation(1)
        one["reviews"][1]["reviewer_id"] = "reviewer-a"
        two = annotation(2, reviews=[], gold=None)
        result = self.prepare([one, two])
        self.assertEqual(result["eligible_episodes"], 0)
        self.assertEqual(result["unscored_count"], 2)
        self.assertIn("two_distinct_independent_reviews_required", result["unscored_reasons"])

    def test_disagreement_requires_distinct_matching_adjudicator(self):
        row = annotation()
        row["reviews"][1]["label"] = "ABSENT"
        row["reviews"][1]["condition_labels"] = {condition: "ABSENT" for condition in CONDITIONS}
        row["adjudication"] = {"reviewer_id": "reviewer-a", "decision": "approve",
                               "resolved_condition_labels": {condition: "PRESENT" for condition in CONDITIONS}}
        result = self.prepare([row])
        self.assertEqual(result["eligible_episodes"], 0)
        row["adjudication"]["reviewer_id"] = "reviewer-c"
        result = self.prepare([row])
        self.assertEqual(result["eligible_episodes"], 1)

    def test_inventory_joins_and_unknown_visibility_gate(self):
        first = annotation(1, occurrence_ids=["not-in-inventory"])
        second = annotation(2, visibility_status="out_of_view")
        result = self.prepare([first, second])
        self.assertEqual(result["eligible_episodes"], 0)
        self.assertIn("occurrence_inventory_join_failed", result["unscored_reasons"])
        self.assertIn("unknown_visibility_requires_reviewed_insufficient_label", result["unscored_reasons"])

    def test_lineage_bridge_through_unscored_row_cannot_split(self):
        first, bridge, third, independent = [annotation(i) for i in range(1, 5)]
        bridge["reviews"] = []
        bridge["parent_ids"] = [first["replay_id"], third["session_id"]]
        result = self.prepare([first, bridge, third, independent])
        episodes = {row["episode_id"]: row for row in read_rows(Path(result["episodes_path"]))}
        self.assertEqual(episodes["episode1"]["family_id"], episodes["episode3"]["family_id"])
        self.assertEqual(episodes["episode1"]["split"], episodes["episode3"]["split"])
        self.assertNotEqual(episodes["episode1"]["family_id"], episodes["episode4"]["family_id"])
        self.assertEqual({row["split"] for row in episodes.values()}, {"dev", "eval"})

    def test_replay_source_and_session_components_are_all_respected(self):
        rows = [annotation(i) for i in range(1, 6)]
        rows[1]["source_ids"] = rows[0]["source_ids"]
        rows[2]["replay_id"] = rows[1]["replay_id"]
        rows[3]["session_id"] = rows[2]["session_id"]
        result = self.prepare(rows)
        episodes = read_rows(Path(result["episodes_path"]))
        self.assertEqual(len({r["family_id"] for r in episodes[:4]}), 1)
        self.assertEqual(len({r["family_id"] for r in episodes}), 2)

    def test_missing_inventory_validation_is_disclosed(self):
        path = self.root / "annotations.jsonl"
        write_rows(path, [annotation()])
        result = prepare_episodes(path, self.root / "no-inventory")
        self.assertEqual(result["inventory_validation"], "not_performed")
        self.assertIn("inventory_joins_not_validated", result["warnings"])

    def test_manifest_hashes_and_deterministic_split(self):
        rows = [annotation(i) for i in range(1, 5)]
        first = self.prepare(rows)
        before = Path(first["inputs_path"]).read_bytes()
        second = self.prepare(rows)
        self.assertEqual(before, Path(second["inputs_path"]).read_bytes())
        self.assertEqual(first["files"]["gold"]["sha256"], second["files"]["gold"]["sha256"])

    def test_duplicate_episode_ids_and_hidden_gold_action_are_rejected(self):
        first, second = annotation(1), annotation(2, episode_id="episode1")
        result = self.prepare([first, second])
        self.assertEqual(result["eligible_episodes"], 0)
        first = annotation(1)
        first["gold"]["valid_action_ids"] = ["hidden-action"]
        result = self.prepare([first])
        self.assertIn("gold_actions_not_in_public_options", result["unscored_reasons"])

    def test_plan_counts_defaults_and_cost_unknown(self):
        result = self.prepare([annotation(i) for i in range(1, 13)])
        plan = plan_runs(Path(result["episodes_path"]), {"split": "all"})
        self.assertEqual(plan["seeds"], [11, 23, 37])
        self.assertEqual(plan["grounding_request_count"], 12 * 3 * 6)
        self.assertEqual(plan["episode_trials"], 12 * 3 * 3 * 6 * 2)
        self.assertEqual(plan["model_turn_upper_bound"], plan["episode_trials"] * 18)
        self.assertEqual(plan["network_calls"], 0)
        self.assertIsNone(plan["cost_estimates"]["estimated_usd"])
        self.assertTrue(any(c["newcomer"]["occurs_in_this_window"] for c in plan["communities"]))
        grouped = {}
        for community in plan["communities"]:
            grouped.setdefault(community["matched_group_id"], []).append(community)
        for pair in grouped.values():
            self.assertEqual({r["forum_assignment"] for r in pair}, {"enabled", "withheld"})
            self.assertEqual(pair[0]["episode_ids"], pair[1]["episode_ids"])

    def test_shared_hidden_worldtruth_is_unscored_without_information_review(self):
        row = annotation(information_match_verified=False)
        result = self.prepare([row])
        self.assertEqual(result["eligible_episodes"], 0)
        self.assertIn("shared_gold_requires_verified_information_match", result["unscored_reasons"])

    def test_arm_specific_insufficient_gold_is_scored_without_invented_worldtruth(self):
        row = annotation(information_match_verified=False)
        row["gold_by_condition"] = {}
        for condition in CONDITIONS:
            rubric = dict(row["gold"])
            if condition.endswith("visual"):
                rubric = {"label": "INSUFFICIENT", "critical_properties": {},
                          "event_interval_seconds": None, "valid_action_ids": None,
                          "evidence_status": "out_of_view"}
            row["gold_by_condition"][condition] = rubric
        for review in row["reviews"]:
            review["information_match_verified"] = False
            review["condition_labels"] = {condition: rubric["label"] for condition, rubric in row["gold_by_condition"].items()}
        result = self.prepare([row])
        self.assertEqual(result["eligible_episodes"], 1)
        gold = {g["condition"]: g for g in read_rows(Path(result["gold_path"]))}
        self.assertEqual(gold["jargon_visual"]["label"], "INSUFFICIENT")
        self.assertIsNone(gold["jargon_visual"]["event_interval_seconds"])
        row["gold_by_condition"]["plain_visual"]["label"] = "PRESENT"
        result = self.prepare([row])
        self.assertEqual(result["eligible_episodes"], 0)

    def test_existing_outputs_are_never_overwritten(self):
        result = self.prepare([annotation()])
        with self.assertRaises(ValueError):
            prepare_episodes(self.root / "annotations.jsonl", Path(result["manifest_path"]).parent, inventory=inventory())
        path = self.root / "candidate.jsonl"
        export_annotations([annotation()], inventory(), path)
        with self.assertRaises(ValueError):
            export_annotations([annotation()], inventory(), path)

    def test_plan_rejects_nonfinite_costs_and_turn_overrun(self):
        result = self.prepare([annotation()])
        path = Path(result["episodes_path"])
        for config in ({"rounds": 7}, {"live": True}, {"cost_estimates": {"input_tokens_per_turn": float("nan")}}):
            with self.assertRaises(ValueError):
                plan_runs(path, config)


class ScoringTests(ExperimentFixtures):
    def test_missing_duplicate_invalid_and_unknown_responses(self):
        prepared = self.prepare([annotation()])
        gold = read_rows(Path(prepared["gold_path"]))
        good = {"item_id": gold[0]["item_id"], "label": "PRESENT", "critical_properties": {"duration": "short"},
                "event_interval_seconds": [.2, .8], "action_id": "jump"}
        responses = [good, {**good, "item_id": gold[1]["item_id"]},
                     {**good, "item_id": gold[1]["item_id"]},
                     {**good, "item_id": gold[2]["item_id"], "label": "NOT_A_LABEL"},
                     {**good, "item_id": "unknown"}]
        path = self.root / "responses.jsonl"
        write_rows(path, responses)
        result = score_responses(Path(prepared["gold_path"]), path, self.root / "scores")
        self.assertEqual(result["response_statuses"], {"valid": 1, "duplicate": 1, "invalid": 1, "missing": 3})
        metric = next(iter(result["by_game_version"].values()))
        self.assertAlmostEqual(metric["interpretation_accuracy"], 1 / 6)
        self.assertAlmostEqual(metric["critical_property_exact_recall"], 1 / 6)
        self.assertAlmostEqual(metric["temporal_mean_iou"], 1 / 6)
        self.assertAlmostEqual(metric["valid_action_accuracy"], 1 / 6)
        self.assertTrue(any(f["code"] == "unknown_response_item" for f in result["findings"]))

    def test_no_temporal_or_action_target_is_invented(self):
        row = annotation()
        row["gold"].update(event_interval_seconds=None, valid_action_ids=None, critical_properties={})
        prepared = self.prepare([row])
        path = self.root / "responses.jsonl"
        write_rows(path, [])
        result = score_responses(Path(prepared["gold_path"]), path)
        metric = next(iter(result["by_game_version"].values()))
        self.assertIsNone(metric["temporal_mean_iou"])
        self.assertIsNone(metric["valid_action_accuracy"])
        self.assertEqual(metric["interpretation_accuracy"], 0)

    def test_exact_properties_do_not_treat_bool_as_integer(self):
        row = annotation()
        row["gold"]["critical_properties"] = {"grounded": True}
        prepared = self.prepare([row])
        gold = read_rows(Path(prepared["gold_path"]))
        path = self.root / "responses.jsonl"
        write_rows(path, [{"item_id": g["item_id"], "label": "PRESENT", "critical_properties": {"grounded": 1}}
                          for g in gold])
        result = score_responses(Path(prepared["gold_path"]), path)
        self.assertEqual(next(iter(result["by_game_version"].values()))["critical_property_exact_recall"], 0)

    def test_paired_missing_responses_use_all_gold_pairs_and_bootstrap_is_stable(self):
        prepared = self.prepare([annotation(1), annotation(2)])
        gold = read_rows(Path(prepared["gold_path"]))
        path = self.root / "responses.jsonl"
        write_rows(path, [{"item_id": g["item_id"], "label": "PRESENT"}
                          for g in gold if g["condition"] == "plain_both"])
        first = score_responses(Path(prepared["gold_path"]), path)
        second = score_responses(Path(prepared["gold_path"]), path)
        self.assertEqual(first, second)
        pair = next(v for v in first["paired_condition_differences"].values()
                    if v["first"] == "jargon_text" and v["second"] == "plain_both")
        self.assertEqual(pair["planned_pair_items"], 2)
        self.assertEqual(pair["both_responses_valid"], 0)
        self.assertEqual(pair["interpretation_accuracy_difference"], 1)

    def test_malformed_json_and_nonfinite_confidence_remain_errors(self):
        prepared = self.prepare([annotation()])
        gold = read_rows(Path(prepared["gold_path"]))
        path = self.root / "responses.jsonl"
        path.write_text("not-json\n" + json.dumps({"item_id": gold[0]["item_id"], "label": "PRESENT", "confidence": float("inf")}) + "\n")
        result = score_responses(Path(prepared["gold_path"]), path)
        self.assertEqual(result["response_statuses"], {"missing": 6})
        self.assertEqual(len(result["findings"]), 2)

    def test_gold_manifest_corruption_and_output_overwrite_are_rejected(self):
        prepared = self.prepare([annotation()])
        path = self.root / "responses.jsonl"
        write_rows(path, [])
        output = self.root / "scores"
        score_responses(Path(prepared["gold_path"]), path, output)
        with self.assertRaises(ValueError):
            score_responses(Path(prepared["gold_path"]), path, output)
        gold_path = Path(prepared["gold_path"])
        gold_path.write_text(gold_path.read_text() + "\n")
        with self.assertRaises(ValueError):
            score_responses(gold_path, path)


class EventTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def audit(self, records):
        path = self.root / "events.jsonl"
        rows = [{"event_id": f"event{i}", "episode_id": "episode", **row}
                for i, row in enumerate(records)]
        write_rows(path, rows)
        return validate_events(path)

    def transcript(self):
        return [{"type": "episode_start", "agent_ids": ["a", "b", "c"], "community_id": "community", "forum_exposure": True},
                {"type": "observation", "agent_id": "a", "observation_id": "private-a"},
                {"type": "message_sent", "sender": "a", "message_id": "m", "recipients": ["b"], "observation_ids": ["private-a"]},
                {"type": "message_delivered", "recipient": "b", "message_id": "m"},
                {"type": "message_read", "recipient": "b", "message_id": "m"},
                {"type": "decision", "agent_id": "b", "message_ids": ["m"], "observation_ids": ["private-a"]}]

    def test_actual_read_grants_shared_observation(self):
        self.assertEqual(self.audit(self.transcript()), [])

    def test_delivery_does_not_imply_read(self):
        records = self.transcript()
        del records[4]
        codes = {r["code"] for r in self.audit(records)}
        self.assertIn("decision_or_memory_uses_unseen_observation", codes)
        self.assertIn("decision_or_memory_uses_unread_message", codes)

    def test_read_without_delivery_or_wrong_audience_is_rejected(self):
        records = self.transcript()
        del records[3]
        self.assertIn("read_without_actual_delivery", {r["code"] for r in self.audit(records)})
        records = self.transcript()
        records[3]["recipient"] = "c"
        self.assertIn("message_outside_permitted_audience_or_scope", {r["code"] for r in self.audit(records)})

    def test_sender_cannot_cite_another_agents_private_observation(self):
        records = self.transcript()
        records[2]["sender"] = "c"
        self.assertIn("sender_uses_unseen_observation", {r["code"] for r in self.audit(records)})

    def test_newcomer_does_not_inherit_private_observations(self):
        records = self.transcript()
        records.extend([{"type": "newcomer_join", "agent_id": "new"},
                        {"type": "decision", "agent_id": "new", "observation_ids": ["private-a"]}])
        self.assertIn("decision_or_memory_uses_unseen_observation", {r["code"] for r in self.audit(records)})

    def test_persistent_forum_requires_community_and_explicit_read(self):
        records = self.transcript()[:2]
        records.extend([
            {"type": "forum_post", "agent_id": "a", "message_id": "forum1", "recipients": ["a", "b", "c"],
             "community_id": "community", "observation_ids": ["private-a"]},
            {"type": "episode_start", "episode_id": "next", "agent_ids": ["a", "b", "c"], "community_id": "community", "forum_exposure": True},
            {"type": "forum_read", "episode_id": "next", "agent_id": "b", "message_id": "forum1", "community_id": "community"},
            {"type": "decision", "episode_id": "next", "agent_id": "b", "message_ids": ["forum1"], "observation_ids": ["private-a"]},
        ])
        self.assertEqual(self.audit(records), [])
        records[-2]["community_id"] = "other"
        self.assertIn("message_outside_permitted_audience_or_scope", {r["code"] for r in self.audit(records)})

    def test_community_forum_newcomer_access_respects_exposure_and_read(self):
        records = self.transcript()[:2]
        records.extend([
            {"type": "forum_post", "agent_id": "a", "message_id": "forum1", "visibility": "community",
             "community_id": "community", "observation_ids": ["private-a"]},
            {"type": "episode_start", "episode_id": "next", "agent_ids": ["a", "b", "new"],
             "community_id": "community", "forum_exposure": True},
            {"type": "forum_read", "episode_id": "next", "agent_id": "new", "message_id": "forum1"},
            {"type": "decision", "episode_id": "next", "agent_id": "new", "message_ids": ["forum1"], "observation_ids": ["private-a"]},
        ])
        self.assertEqual(self.audit(records), [])
        records[3]["forum_exposure"] = False
        self.assertIn("forum_access_withheld", {r["code"] for r in self.audit(records)})
        records[3]["forum_exposure"] = True
        del records[4]
        self.assertIn("decision_or_memory_uses_unseen_observation", {r["code"] for r in self.audit(records)})

    def test_replaced_agents_and_closed_episodes_cannot_act(self):
        records = self.transcript()
        records.extend([{"type": "newcomer_join", "agent_id": "new", "replace_agent_id": "c"},
                        {"type": "decision", "agent_id": "c"},
                        {"type": "episode_end"}, {"type": "decision", "agent_id": "a"}])
        codes = {r["code"] for r in self.audit(records)}
        self.assertIn("agent_not_in_episode_team", codes)
        self.assertIn("event_after_episode_end", codes)


if __name__ == "__main__":
    unittest.main()
