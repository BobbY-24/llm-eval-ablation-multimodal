"""Offline lexical/exposure bookkeeping fixtures, not emergent-language results."""

import json
import tempfile
import unittest
from pathlib import Path

from multimodal_game_eval.communication import analyze_communication


class CommunicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def analyze(self, rows, seeded=None, probes=None):
        path = self.root / "events.jsonl"
        rows = [{"event_id": f"event{i}", "episode_id": "episode", "sequence": i + 1, **row}
                for i, row in enumerate(rows)]
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return analyze_communication(path, seeded or [], probes)

    def transcript(self):
        return [{"type": "episode_start", "agent_ids": ["a", "b", "c"], "community_id": "community", "forum_exposure": True},
                {"type": "message_sent", "sender": "a", "message_id": "m1", "recipients": ["b"],
                 "text": "Try zig zag now.", "coined_forms": ["zig zag"],
                 "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}, "estimated_usd": .01},
                {"type": "message_delivered", "recipient": "b", "message_id": "m1"},
                {"type": "message_read", "recipient": "b", "message_id": "m1"},
                {"type": "message_sent", "sender": "b", "message_id": "m2", "recipients": ["a"], "text": "ZIG\nZAG worked."}]

    def test_actual_exposure_then_reuse_has_logged_lag(self):
        result = self.analyze(self.transcript())
        record = next(row for row in result["form_records"] if row["agent_id"] == "b")
        self.assertEqual(result["status"], "valid")
        self.assertEqual(record["form"], "zig zag")
        self.assertEqual(record["exposure_to_first_reuse_lag"]["recorded_sequence"], 1)
        self.assertFalse(record["produced_before_first_read"])
        self.assertEqual(record["semantic_adoption"], "not_inferred")

    def test_unicode_casefold_token_boundaries_and_no_arbitrary_novelty(self):
        rows = self.transcript()[:1] + [{"type": "message_sent", "sender": "a", "message_id": "m", "recipients": ["b"],
                                      "text": "STRASSE GG bigger unlistedword"}]
        result = self.analyze(rows, ["Straße", "gg", "big"])
        self.assertEqual({row["form_key"] for row in result["form_records"]}, {"strasse", "gg"})
        self.assertNotIn("unlistedword", json.dumps(result["form_registry"]))

    def test_delivery_is_not_actual_read_exposure(self):
        rows = self.transcript()
        del rows[3]
        result = self.analyze(rows)
        record = next(row for row in result["form_records"] if row["agent_id"] == "b")
        self.assertIsNone(record["first_actual_read_exposure"])
        self.assertIsNone(record["exposure_to_first_reuse_lag"])
        self.assertTrue(record["produced_before_first_read"])

    def test_invalid_event_audit_stops_analysis(self):
        rows = self.transcript()
        del rows[2]
        result = self.analyze(rows)
        self.assertEqual(result["status"], "invalid")
        self.assertFalse(result["analysis_performed"])
        self.assertEqual(result["form_records"], [])

    def test_recorded_costs_and_missing_usage_are_separate_from_characters(self):
        result = self.analyze(self.transcript())
        sent = result["recorded_accounting_by_stage"]["sent"]
        read = result["recorded_accounting_by_stage"]["read"]
        self.assertEqual(sent["recorded_sums"]["input_tokens"], 3)
        self.assertEqual(sent["events_missing_field"]["input_tokens"], 1)
        self.assertEqual(sent["recorded_sums"]["cost_usd"], .01)
        self.assertIsNone(read["recorded_sums"]["input_tokens"])
        self.assertEqual(result["character_counts_by_stage"]["read"]["recorded_text_characters"], len("Try zig zag now."))

    def test_recorded_ceiling_violation_invalidates_analysis(self):
        rows = self.transcript()
        rows[0].update(rounds=6, max_turns=18)
        rows.append({"type": "decision", "agent_id": "a", "round": 7, "turn": 19, "status": "failed"})
        result = self.analyze(rows)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual({row["code"] for row in result["ceiling_findings"]}, {"round_ceiling_exceeded", "turn_ceiling_exceeded"})

    def test_unreviewed_or_nonfresh_probes_do_not_become_semantic_gold(self):
        probes = [{"probe_id": "p1", "community_id": "community", "agent_id": "b", "reviewer_status": "candidate",
                   "fresh": True, "label_correct": True},
                  {"probe_id": "p2", "community_id": "community", "agent_id": "b", "reviewer_status": "adjudicated",
                   "label_correct": True}]
        result = self.analyze(self.transcript(), probes=probes)
        self.assertEqual(result["reviewed_probes"]["accepted"], 0)
        self.assertEqual(len(result["reviewed_probes"]["excluded"]), 2)

    def test_two_community_probe_uncertainty_and_action_denominator(self):
        rows = self.transcript() + [{"type": "episode_start", "episode_id": "second", "agent_ids": ["x"], "community_id": "other"}]
        probes = [{"probe_id": "p1", "community_id": "community", "agent_id": "b", "reviewer_status": "adjudicated",
                   "fresh": True, "label_correct": True, "action_correct": True},
                  {"probe_id": "p2", "community_id": "other", "agent_id": "x", "reviewer_status": "adjudicated",
                   "fresh": True, "label_correct": False}]
        result = self.analyze(rows, probes=probes)["reviewed_probes"]
        self.assertEqual(result["label_accuracy"], .5)
        self.assertEqual(result["action_accuracy"], 1)
        self.assertEqual(result["action_denominator"], 1)
        self.assertIsNotNone(result["label_community_bootstrap_95_ci"])
        self.assertIsNone(result["action_community_bootstrap_95_ci"])

    def test_failures_and_missing_counters_are_honest(self):
        rows = self.transcript() + [{"type": "decision", "agent_id": "b", "status": "timeout"}]
        result = self.analyze(rows)
        self.assertEqual(result["failure_count"], 1)
        self.assertIsNone(result["episode_counters"][0]["turn_cap"])
        self.assertIsNone(result["episode_counters"][0]["max_recorded_turn"])

    def test_persistent_community_exposure_lag_uses_episode_order(self):
        rows = self.transcript()[:1] + [
            {"type": "forum_post", "agent_id": "a", "message_id": "forum", "visibility": "community",
             "text": "zig zag", "coined_forms": ["zig zag"]},
            {"type": "forum_read", "agent_id": "b", "message_id": "forum"},
            {"type": "episode_end"},
            {"type": "episode_start", "episode_id": "next", "agent_ids": ["a", "b", "c"], "community_id": "community", "forum_exposure": True},
            {"type": "message_sent", "episode_id": "next", "sender": "b", "message_id": "reuse", "recipients": ["a"], "text": "zig zag"},
        ]
        result = self.analyze(rows)
        record = next(row for row in result["form_records"] if row["agent_id"] == "b")
        self.assertEqual(record["exposure_to_first_reuse_lag"]["community_episode_order"], 1)
        self.assertFalse(record["first_actual_read_exposure"]["self_authored_read"])

    def test_duplicate_probes_are_all_excluded(self):
        probe = {"probe_id": "same", "community_id": "community", "agent_id": "b", "reviewer_status": "adjudicated",
                 "fresh": True, "label_correct": True}
        result = self.analyze(self.transcript(), probes=[probe, dict(probe)])["reviewed_probes"]
        self.assertEqual(result["accepted"], 0)
        self.assertEqual(len(result["excluded"]), 2)


if __name__ == "__main__":
    unittest.main()
