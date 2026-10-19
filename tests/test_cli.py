"""Exercise real CLI dispatch and filesystem exports, using declared fixtures."""
import contextlib
import io
import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from multimodal_game_eval.cli import PACKAGE_ROOT, DEFAULT_INVENTORY, main
from multimodal_game_eval.core import read_json, read_jsonl
from multimodal_game_eval.demo import build_demo
from multimodal_game_eval.inventory import load_inventory
from multimodal_game_eval.io import write_json, write_jsonl


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def command(self, *arguments):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            code = main(list(arguments))
        return code, json.loads(captured.getvalue())

    def test_aoe_index_annotation_prepare_pipeline_keeps_candidates_unscored(self):
        media = self.root / "data"
        media.mkdir()
        (media / "fixture.mp4").write_bytes(b"fixture-placeholder-not-a-real-video")
        (media / "fixture.aoe2record").write_bytes(b"fixture-placeholder-not-a-real-replay")
        write_jsonl(media / "manifest.jsonl", [{"video_path": "fixture.mp4", "replay_path": "fixture.aoe2record",
            "source_replay_id": "fixture-replay", "game": "Fixture game", "game_version": "fixture-v1", "fps": 30}])
        code, probe = self.command("inspect", "--dataset", "aoe2", "--root", str(media))
        self.assertEqual((code, probe["sample_count"]), (0, 1))
        index = self.root / "index.jsonl"
        code, result = self.command("index", "--dataset", "aoe2", "--root", str(media), "--output", str(index))
        self.assertEqual(code, 0)
        sample = next(read_jsonl(index))
        self.assertEqual(len(sample["hashes"]["replay"]), 64)
        self.assertTrue(result["asset_hashes_complete"])
        code, result = self.command("validate", "--dataset", "aoe2", "--root", str(media))
        self.assertEqual((code, result["validation_status"]), (0, "schema_checks_only"))
        write_json(self.root / "inventory.json", {"scopes": [{"scope_id": "scope", "label": "Fixture game", "scope_type": "game",
            "terms": [{"sense_id": "sense", "term_group_id": "group", "term": "fixture-term"}]}]})
        write_json(self.root / "graph.json", {"usages": [{"sense_id": "sense", "term_group_id": "group", "scope_id": "scope", "occurrence_id": "usage"}]})
        write_json(self.root / "selections.json", {sample["sample_id"]: "sense"})
        annotations = self.root / "annotations.jsonl"
        source_args = ("--inventory", str(self.root / "inventory.json"), "--graph", str(self.root / "graph.json"))
        code, result = self.command("export-annotations", "--index", str(index), "--selections", str(self.root / "selections.json"),
            "--output", str(annotations), *source_args)
        self.assertEqual((code, result["scored"]), (0, 0))
        row = next(read_jsonl(annotations))
        self.assertEqual(row["occurrence_ids"], ["usage"])
        self.assertEqual(row["reviews"], [])
        self.assertIsNone(row["gold"])
        self.assertIn("aoe2:replay:fixture-replay", row["source_ids"])
        code, prepared = self.command("prepare", "--annotations", str(annotations), "--output", str(self.root / "prepared"), *source_args)
        self.assertEqual((code, prepared["eligible_episodes"]), (0, 0))
        self.assertEqual(prepared["unscored_count"], 1)
        code, _ = self.command("export-annotations", "--index", str(index), "--output", str(annotations), "--force", *source_args)
        self.assertEqual(code, 0)

    def test_index_companion_collision_does_not_create_partial_index(self):
        output = self.root / "index.jsonl"
        write_json(Path(str(output) + ".manifest.json"), {"existing": True})
        code, _ = self.command("index", "--dataset", "vpt", "--root", str(self.root), "--output", str(output))
        self.assertEqual(code, 2)
        self.assertFalse(output.exists())

    def test_communication_cli_tracks_logged_reuse_without_semantic_gold(self):
        events = self.root / "events.jsonl"
        write_jsonl(events, [
            {"event_id": "start", "episode_id": "e", "type": "episode_start", "agent_ids": ["a", "b"], "community_id": "c"},
            {"event_id": "send", "episode_id": "e", "type": "message_sent", "message_id": "m", "sender": "a", "recipients": ["b"], "text": "fixture-term"},
            {"event_id": "deliver", "episode_id": "e", "type": "message_delivered", "message_id": "m", "recipient": "b"},
            {"event_id": "read", "episode_id": "e", "type": "message_read", "message_id": "m", "agent_id": "b"},
            {"event_id": "reuse", "episode_id": "e", "type": "message_sent", "message_id": "n", "sender": "b", "recipients": ["a"], "text": "fixture-term"}])
        write_json(self.root / "seeded.json", ["fixture-term"])
        code, result = self.command("analyze-communication", "--events", str(events), "--seeded-forms", str(self.root / "seeded.json"))
        self.assertEqual((code, result["status"]), (0, "valid"))
        recipient = next(row for row in result["form_records"] if row["agent_id"] == "b")
        self.assertEqual(recipient["exposure_to_first_reuse_lag"]["event_index"], 1)
        self.assertEqual(recipient["semantic_adoption"], "not_inferred")
        self.assertEqual(result["reviewed_probes"]["status"], "not_supplied")

    def test_configs_and_package_metadata_parse_without_dependencies(self):
        for path in (PACKAGE_ROOT / "configs").glob("*.json"):
            self.assertIsInstance(read_json(path), dict)
        metadata = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())
        self.assertEqual(metadata["project"]["dependencies"], [])
        config = read_json(PACKAGE_ROOT / "configs/pilot.json")
        self.assertEqual(config["seeds"], [11, 23, 37])
        self.assertEqual(config["agents"] * config["rounds"], 18)

    def test_synthetic_demo_correct_wrong_and_insufficient_controls(self):
        inventory = load_inventory(DEFAULT_INVENTORY)
        result = build_demo(self.root, inventory, PACKAGE_ROOT / "configs/pilot.json")
        self.assertEqual(result["real_term_candidates"], 10)
        self.assertEqual(result["synthetic_eligible_episodes"], 12)
        self.assertEqual(result["synthetic_input_items"], 72)
        self.assertEqual(result["event_findings"], [])
        candidates = read_json(self.root / "pilot_sense_candidates.json")["senses"]
        self.assertTrue(all(not row["scorable"] and row["evidence"] for row in candidates))
        for subfolder, accuracy in (("scores_correct", 1), ("scores_incorrect", 0)):
            scores = read_json(self.root / subfolder / "scores.json")
            self.assertTrue(all(row["interpretation_accuracy"] == accuracy for row in scores["by_game_version"].values()))
        labels = {row["label"] for row in read_jsonl(self.root / "prepared/gold.jsonl")}
        self.assertEqual(labels, {"PRESENT", "ABSENT", "INSUFFICIENT"})
        for row in read_jsonl(self.root / "prepared/inputs.jsonl"):
            self.assertFalse(set(row["input"]) & {"gold", "label", "reviews", "sense_id"})


if __name__ == "__main__":
    unittest.main()
