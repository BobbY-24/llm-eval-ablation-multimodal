"""Stage 1 mining CLI, Stage 2 grounding-only plans and gold-free prompt rendering (fixtures only)."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from multimodal_game_eval.cli import DEFAULT_INVENTORY, PACKAGE_ROOT, main
from multimodal_game_eval.core import DatasetError, Sample, Window, read_json, read_jsonl
from multimodal_game_eval.experiments import plan_runs
from multimodal_game_eval.melee_mining import mine_adapter
from multimodal_game_eval.vlm_prompts import render_label_request, render_property_request

from test_melee_mining import dash, series, seg, wavedash_rows, window, WAIT

DEMO = PACKAGE_ROOT / "artifacts/demo-final/prepared"


def replay_window():
    p1 = series(*([seg(WAIT, 40, **{"position.x": 0.0, "direction": 1.0}), wavedash_rows()] * 3))
    p2 = series(seg(WAIT, 7), dash(1.0), dash(-1.0), dash(1.0), seg(WAIT, 30))
    return window({"1": p1, "2": p2})


class FakeSlippi:
    def __init__(self, data):
        self.data, self.root = data, Path("/nonexistent")
        self.sample = Sample("replay-sample", "slippi", "Super Smash Bros. Melee", {"replay": "a.slp"},
                             ["actions", "states"], {"unit": "frames", "nominal_hz": 60},
                             hashes={"replay_sha256": "ab" * 32})

    def iter_samples(self):
        yield self.sample

    def read_window(self, sample, span):
        pick = lambda rows: [r for r in rows if span.start <= r["frame_id"] < span.end]
        return Window(sample.sample_id, states=pick(self.data["states"]), actions=pick(self.data["actions"]))


class MiningCLITests(unittest.TestCase):
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

    def test_mine_then_export_keeps_candidates_unreviewed_and_lineage_connected(self):
        with patch("multimodal_game_eval.adapters.create_adapter", return_value=FakeSlippi(replay_window())):
            code, manifest = self.command("mine-candidates", "--dataset", "slippi", "--root", str(self.root),
                                          "--output", str(self.root / "mined"))
        self.assertEqual(code, 0, manifest)
        self.assertEqual(manifest["status"], "review_candidates_not_labels")
        self.assertEqual(manifest["by_family_stratum"], {"dashdancing|positive_candidate": 1, "wavedash|positive_candidate": 3})
        index = list(read_jsonl(self.root / "mined/candidate_index.jsonl"))
        self.assertTrue(all(row["lineage"]["replay_id"] == "ab" * 32 for row in index))
        self.assertTrue(all(row["metadata"]["frame_selection"]["output_frames"] <= 64 for row in index))
        code, exported = self.command("export-annotations", "--index", str(self.root / "mined/candidate_index.jsonl"),
                                      "--selections", str(self.root / "mined/selections.json"),
                                      "--output", str(self.root / "candidates.jsonl"))
        self.assertEqual((code, exported["status"], exported["scored"]), (0, "candidate_templates_only", 0))
        rows = list(read_jsonl(self.root / "candidates.jsonl"))
        self.assertEqual({row["sense_id"] for row in rows},
                         {"sense:d4d6ca6b86c65bf9714de62c:050", "sense:4a62a8139aa4f83e7da0da03:050"})
        self.assertTrue(all(row["gold"] is None and row["reviews"] == [] for row in rows))
        self.assertTrue(all("slippi:replay:" + "ab" * 32 == row["replay_id"] for row in rows))
        code, prepared = self.command("prepare", "--annotations", str(self.root / "candidates.jsonl"),
                                      "--output", str(self.root / "prepared"))
        self.assertEqual((code, prepared["eligible_episodes"]), (0, 0))

    def test_unknown_sense_in_config_is_refused(self):
        config = read_json(PACKAGE_ROOT / "configs/pilot_families.json")
        config["families"]["wavedash"]["sense_id"] = "sense:not-real"
        (self.root / "families.json").write_text(json.dumps(config))
        with patch("multimodal_game_eval.adapters.create_adapter", return_value=FakeSlippi(replay_window())):
            code, result = self.command("mine-candidates", "--dataset", "slippi", "--root", str(self.root),
                                        "--families-config", str(self.root / "families.json"),
                                        "--output", str(self.root / "mined"))
        self.assertEqual(code, 2)
        self.assertIn("sense:not-real", result["message"])

    def test_replay_ending_on_chunk_edge_keeps_tail_events(self):
        data = replay_window()
        last = data["states"][-1]["frame_id"]  # trim so the data ends exactly at a chunk edge
        chunk, overlap = 120, 60
        end = ((last + 1 - chunk) // (chunk - overlap)) * (chunk - overlap) + chunk
        data = {key: [r for r in rows if r["frame_id"] < end] for key, rows in data.items()}
        from multimodal_game_eval.melee_mining import mine_window
        whole = {c["candidate_id"] for c in mine_window(data, "replay-sample")}
        fake = FakeSlippi(data)
        found, report = mine_adapter(fake, [fake.sample], chunk_frames=chunk, overlap_frames=overlap,
                                     margin_frames=20, first_frame=0)
        self.assertEqual({c["candidate_id"] for c in found}, whole)
        self.assertEqual(report[0]["dropped_at_window_edge"], 0)


class SlippiFieldTests(unittest.TestCase):
    def test_optional_l_cancel_and_velocity_fields_are_read_when_present(self):
        from multimodal_game_eval.adapters.slippi import SlippiAdapter
        sample = Sample("t", "slippi", "Super Smash Bros. Melee", {}, ["actions", "states"], {"unit": "frames"})
        post = SimpleNamespace(state=[70, 70], l_cancel=[1, 0],
                               velocities=SimpleNamespace(self_x_ground=[0.5, 0.25], self_x_air=[0.0, 0.0], self_y=[0.0, 0.0]))
        game = SimpleNamespace(frames=SimpleNamespace(id=[0, 1], ports=[SimpleNamespace(
                                   leader=SimpleNamespace(pre=SimpleNamespace(), post=post), follower=None)]),
                               start=SimpleNamespace(players=[SimpleNamespace(port=0)], stage=31, is_pal=False))
        with tempfile.TemporaryDirectory() as root:
            output = SlippiAdapter(root)._window(game, sample, "frames", 0, 2, {"finalized_through_frame": 1})
        leader = output.states[0]["ports"]["1"]["leader"]
        self.assertEqual((leader["l_cancel"], leader["velocities.self_x_ground"]), (1, 0.5))
        self.assertIsNone(leader["percent"])


class GroundingPlanTests(unittest.TestCase):
    def config(self, **extra):
        stage2 = read_json(PACKAGE_ROOT / "configs/stage2_vlm.json")
        return {**stage2["plan_config"], "split": "all", **extra}

    def test_grounding_only_plan_has_no_communities_and_counts_requests(self):
        plan = plan_runs(DEMO / "episodes.jsonl", self.config())
        episodes = plan["selected_episodes"]
        self.assertEqual(plan["community_count"], 0)
        self.assertEqual(plan["grounding_request_count"], episodes * 6 * 3)
        self.assertEqual(plan["property_probe_request_count"], episodes * 3 * 3)
        self.assertEqual(plan["model_request_upper_bound"], episodes * 27)
        self.assertTrue(all(r["max_output_tokens"] == 2048 for r in plan["grounding_requests"]))
        probe = plan["property_probe_requests"][0]
        self.assertTrue(probe["condition"].startswith("property_probe_") and probe["wording"] == "neutral_property_questions")
        scenarios = plan["grounding_cost_scenarios"]
        self.assertEqual(scenarios["requests"], episodes * 27)
        self.assertIsNone([m for m in scenarios["models"] if m["input_usd_per_million"] is None][0]["estimated_usd_output_cap"])
        priced = [m for m in scenarios["models"] if m["input_usd_per_million"] is not None]
        self.assertTrue(priced and all(m["estimated_usd_output_cap"] >= m["estimated_usd_assumed_output"] for m in priced))

    def test_cli_plan_reads_nested_stage2_config(self):
        with tempfile.TemporaryDirectory() as root:
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                code = main(["plan", "--episodes", str(DEMO / "episodes.jsonl"),
                             "--config", str(PACKAGE_ROOT / "configs/stage2_vlm.json"),
                             "--output", str(Path(root) / "plan.json")])
            plan = read_json(Path(root) / "plan.json")
        self.assertEqual(code, 0)
        self.assertTrue(plan["grounding_only"])
        self.assertEqual(plan["community_count"], 0)
        self.assertTrue(all(r["split"] == "eval" for r in plan["grounding_requests"]))

    def test_invalid_stage2_options_are_rejected(self):
        for extra in ({"grounding_only": "yes"}, {"property_probe_modalities": ["audio"]},
                      {"cost_scenarios": {"models": []}}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                plan_runs(DEMO / "episodes.jsonl", self.config(**extra))
        missing = self.config()
        missing["cost_scenarios"] = {**missing["cost_scenarios"], "input_tokens_by_condition": {"jargon_text": 1}}
        with self.assertRaises(ValueError):
            plan_runs(DEMO / "episodes.jsonl", missing)


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.templates = read_json(PACKAGE_ROOT / "configs/stage2_prompts.json")
        self.inputs = list(read_jsonl(DEMO / "inputs.jsonl"))

    def test_demo_inputs_render_without_gold_and_wording_differs_only_in_question(self):
        gold = {row["item_id"]: row for row in read_jsonl(DEMO / "gold.jsonl")}
        by_episode = {}
        for row in self.inputs:
            request = render_label_request(row["input"], self.templates, seed=11)
            text = json.dumps(request)
            for secret in ("critical_properties\": {\"", "reviewer", "occurrence", "sense:"):
                self.assertNotIn(secret, text)
            self.assertEqual(request["frame_count"] > 0, row["condition"].split("_")[1] in ("visual", "both")
                             and bool(row["input"]["evidence"].get("visual_refs")))
            self.assertIn(row["item_id"], gold)
            by_episode.setdefault((row["episode_id"], row["condition"].split("_")[1]), {})[row["condition"].split("_")[0]] = request
        for pair in by_episode.values():
            self.assertEqual(pair["jargon"]["content"][:-1], pair["plain"]["content"][:-1])
            self.assertNotEqual(pair["jargon"]["content"][-1], pair["plain"]["content"][-1])

    def test_refuses_extra_keys_and_seed_only_permutes_label_order(self):
        payload = dict(self.inputs[0]["input"])
        with self.assertRaises(DatasetError):
            render_label_request({**payload, "gold": {"label": "PRESENT"}}, self.templates, seed=11)
        orders = {tuple(render_label_request(payload, self.templates, seed=s)["label_order"]) for s in (11, 23, 37)}
        self.assertEqual(render_label_request(payload, self.templates, seed=23),
                         render_label_request(payload, self.templates, seed=23))
        self.assertTrue(all(sorted(order) == ["ABSENT", "INSUFFICIENT", "PRESENT"] for order in orders))

    def test_property_probe_omits_term_and_unobservable_properties(self):
        families = read_json(PACKAGE_ROOT / "configs/pilot_families.json")["families"]
        row = next(r for r in self.inputs if r["condition"] == "jargon_both")
        request = render_property_request(row["input"], self.templates, families["edgehogging"]["critical_properties"])
        text = json.dumps(request)
        self.assertNotIn(row["input"]["terminology"]["expression"], text)
        self.assertNotIn("deliberate_denial", text)
        self.assertIn("holder_on_ledge", text)


if __name__ == "__main__":
    unittest.main()
