"""Offline tests for the A0-A4 API ablation runner. Network access is blocked."""
import base64
import contextlib
import csv
import io
import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablation import providers  # noqa: E402
from ablation.__main__ import main  # noqa: E402
from ablation.conditions import Item, build_stimuli, load_items, mcq_options  # noqa: E402
from ablation.items import DEFAULT_GRAPH, DEFAULT_SCOPES, build_items, draft_review, load_gaming_rows, scope_counts  # noqa: E402
from ablation.prompts import build_prompt, parse_response  # noqa: E402
from ablation.runner import Budget, LiveCallRefused, Paced, run, run_one  # noqa: E402
from ablation.scoring import score_rows, summarize, token_f1  # noqa: E402

ITEMS = ROOT / "ablation" / "fixtures" / "synthetic_items.jsonl"
RESULTS = ROOT / "ablation" / "results"
VALID = json.dumps({"selected_option": "B", "confidence": 0.5, "evidence_used": {"text": True, "image": False}})


def _no_network(*args, **kwargs):
    raise AssertionError("network access attempted during an offline test")


def setUpModule():
    global _patches
    _patches = [mock.patch.object(socket, "create_connection", _no_network),
                mock.patch.object(socket.socket, "connect", _no_network)]
    for patch in _patches:
        patch.start()


def tearDownModule():
    for patch in _patches:
        patch.stop()


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": headers, "body": json.loads(body), "timeout": timeout})
        status, payload = self.responses.pop(0)
        return status, json.dumps(payload).encode() if not isinstance(payload, bytes) else payload


class ScriptedProvider:
    live = False
    name = "scripted"

    def __init__(self, *texts):
        self.texts = list(texts)
        self.prompts = []

    def generate(self, prompt, image_paths):
        self.prompts.append(prompt)
        value = self.texts.pop(0)
        if isinstance(value, Exception):
            raise value
        return providers.Generation(value)


class ConditionTests(unittest.TestCase):
    def setUp(self):
        self.items = load_items(ITEMS)

    def test_conditions_expose_only_allowed_evidence(self):
        stimuli = {s.condition: s for s in build_stimuli(self.items[:1] + self.items[2:3], task="freeform")
                   if s.example_id == "syn-001"}
        self.assertIsNone(stimuli["A0"].community)
        self.assertEqual((stimuli["A0"].text_context, stimuli["A0"].image_paths), (None, ()))
        self.assertIsNotNone(stimuli["A1"].text_context)
        self.assertEqual(stimuli["A1"].image_paths, ())
        self.assertIsNone(stimuli["A2"].text_context)
        self.assertEqual(stimuli["A2"].image_source_ids, ("syn-001",))
        self.assertEqual(stimuli["A3"].image_source_ids, ("syn-001",))
        self.assertIsNotNone(stimuli["A3"].text_context)
        self.assertEqual(stimuli["A4"].image_source_ids, ("syn-003",))  # other community preferred
        self.assertIsNotNone(stimuli["A4"].text_context)

    def test_prompt_never_contains_gold_ids_or_condition(self):
        for task in ("freeform", "mcq"):
            for s in build_stimuli(self.items, task=task):
                prompt = build_prompt(s, task)
                item = next(i for i in self.items if i.id == s.example_id)
                self.assertNotIn(s.example_id, prompt)
                self.assertNotIn(f'"{s.condition}"', prompt)
                self.assertNotIn(item.image_path, prompt)
                if task == "freeform":
                    self.assertNotIn(item.gold_meaning, prompt)
                if s.condition == "A0":
                    self.assertNotIn(item.community, prompt)

    def test_mcq_options_are_deterministic_and_contain_gold_once(self):
        choices, key = mcq_options(self.items[0], seed=7)
        self.assertEqual((choices, key), mcq_options(self.items[0], seed=7))
        self.assertEqual(choices[key], self.items[0].gold_meaning)
        self.assertEqual(sorted(choices), ["A", "B", "C", "D"])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            build_stimuli(self.items, conditions=("A1", "A1"))
        with self.assertRaises(ValueError):
            build_stimuli([Item("x", "t", "c")], conditions=("A1",))
        with self.assertRaises(ValueError):
            build_stimuli([Item("x", "t", "c", "ctx", "/a.png")], conditions=("A4",))
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "items.jsonl"
            bad.write_text(json.dumps({"id": "a", "term": "t", "community": "c", "answer_key": "A"}) + "\n")
            with self.assertRaises(ValueError):
                load_items(bad)


class ParseTests(unittest.TestCase):
    def test_strict_contract(self):
        self.assertEqual(parse_response(VALID, "mcq")["selected_option"], "B")
        bad = [
            "```json\n" + VALID + "\n```",
            '{"selected_option": "E", "confidence": 0.5, "evidence_used": {"text": true, "image": false}}',
            '{"selected_option": "A", "confidence": 2, "evidence_used": {"text": true, "image": false}}',
            '{"selected_option": "A", "confidence": NaN, "evidence_used": {"text": true, "image": false}}',
            '{"selected_option": "A", "selected_option": "B", "confidence": 0, "evidence_used": {"text": true, "image": false}}',
            '{"selected_option": "A", "confidence": 0, "evidence_used": {"text": 1, "image": false}}',
            '{"selected_option": "A", "confidence": 0, "evidence_used": {"text": true, "image": false}, "x": 1}',
        ]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_response(raw, "mcq")
        with self.assertRaises(ValueError):
            parse_response('{"predicted_meaning": " ", "confidence": 0, "evidence_used": {"text": true, "image": true}}')


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.image = Path(self.tmp.name) / "one.png"
        self.image.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    def tearDown(self):
        self.tmp.cleanup()

    def test_openai_compatible_request_and_response(self):
        transport = FakeTransport((200, {"choices": [{"message": {"content": VALID}}], "usage": {"total_tokens": 9}}))
        provider = providers.OpenAICompatibleProvider("model-x", base_url="https://example.test/v1",
                                                      api_key_env="TEST_OPENAI_KEY", transport=transport)
        with mock.patch.dict(os.environ, {"TEST_OPENAI_KEY": "sk-test-123"}):
            out = provider.generate("hello", [str(self.image)])
        call = transport.calls[0]
        self.assertEqual(call["url"], "https://example.test/v1/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer sk-test-123")
        self.assertNotIn("sk-test-123", call["url"])
        parts = call["body"]["messages"][0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "hello"})
        self.assertEqual(parts[1]["image_url"]["url"],
                         "data:image/png;base64," + base64.b64encode(self.image.read_bytes()).decode())
        self.assertEqual(call["body"]["temperature"], 0)
        self.assertEqual(call["body"]["response_format"], {"type": "json_object"})
        self.assertEqual((out.text, out.usage), (VALID, {"total_tokens": 9}))
        self.assertNotIn("sk-test-123", repr(vars(provider)))

    def test_gemini_request_uses_header_key_and_inline_image(self):
        body = {"candidates": [{"content": {"parts": [{"text": VALID}]}}], "usageMetadata": {"totalTokenCount": 5}}
        transport = FakeTransport((200, body))
        provider = providers.GeminiProvider("gemini-test", api_key_env="TEST_GEMINI_KEY", transport=transport)
        with mock.patch.dict(os.environ, {"TEST_GEMINI_KEY": "g-test-456"}):
            out = provider.generate("hi", [str(self.image)])
        call = transport.calls[0]
        self.assertEqual(call["url"],
                         "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent")
        self.assertEqual(call["headers"]["x-goog-api-key"], "g-test-456")
        self.assertNotIn("g-test-456", call["url"])
        parts = call["body"]["contents"][0]["parts"]
        self.assertEqual(parts[1]["inline_data"]["mime_type"], "image/png")
        self.assertEqual(call["body"]["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual((out.text, out.usage), (VALID, {"totalTokenCount": 5}))

    def test_errors_are_redacted_and_missing_key_makes_no_call(self):
        transport = FakeTransport((401, b'{"error": "bad key sk-secret-9"}'))
        provider = providers.OpenAICompatibleProvider("m", api_key_env="TEST_OPENAI_KEY", transport=transport)
        with mock.patch.dict(os.environ, {"TEST_OPENAI_KEY": "sk-secret-9"}):
            with self.assertRaises(providers.ProviderError) as caught:
                provider.generate("x", [])
        self.assertIn("HTTP 401", str(caught.exception))
        self.assertNotIn("sk-secret-9", str(caught.exception))
        empty = FakeTransport()
        provider = providers.GeminiProvider("m", api_key_env="TEST_MISSING_KEY", transport=empty)
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(providers.ProviderError):
            provider.generate("x", [])
        self.assertEqual(empty.calls, [])

    def test_unsafe_configuration_is_rejected(self):
        for url in ("http://example.test/v1", "https://user:pw@example.test", "https://example.test/v1?key=1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                providers.OpenAICompatibleProvider("m", base_url=url)
        providers.OpenAICompatibleProvider("m", base_url="http://localhost:8000/v1")
        with self.assertRaises(ValueError):
            providers.OpenAICompatibleProvider("m", api_key_env="not a name")
        with self.assertRaises(ValueError):
            providers.encode_image(str(Path(self.tmp.name) / "x.bmp"))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.items = load_items(ITEMS)
        self.stimulus = build_stimuli(self.items[:2], conditions=("A1",), task="mcq")[0]

    def test_one_repair_then_success(self):
        provider = ScriptedProvider("not json", VALID)
        log = []
        result = run_one(provider, self.stimulus, "mcq", Budget(5), log.append)
        self.assertEqual((result["status"], result["attempts"], result["selected_option"]), ("ok", 2, "B"))
        self.assertEqual([r["parsing_status"] for r in log], ["invalid", "valid"])
        self.assertIn("previous response failed", provider.prompts[1])

    def test_inference_errors_are_not_repaired(self):
        provider = ScriptedProvider(RuntimeError("timeout"), VALID)
        result = run_one(provider, self.stimulus, "mcq", Budget(5), lambda r: None)
        self.assertEqual((result["status"], result["attempts"]), ("inference_failure", 1))

    def test_budget_caps_calls_and_keeps_unrun_rows(self):
        stimuli = build_stimuli(self.items, task="mcq")
        with tempfile.TemporaryDirectory() as tmp:
            manifest = run(providers.MockProvider(), stimuli, task="mcq", out_dir=Path(tmp) / "r", max_calls=3)
            rows = [json.loads(line) for line in (Path(tmp) / "r" / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual(manifest["calls_used"], 3)
        self.assertEqual(len(rows), len(stimuli))
        self.assertEqual(sum(r["status"] == "not_run_budget" for r in rows), len(stimuli) - 3)

    def test_live_provider_requires_explicit_permission(self):
        provider = providers.OpenAICompatibleProvider("m", transport=FakeTransport())
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(LiveCallRefused):
            run(provider, [self.stimulus], task="mcq", out_dir=Path(tmp) / "r", max_calls=1)
        with contextlib.redirect_stderr(io.StringIO()), tempfile.TemporaryDirectory() as tmp:
            code = main(["run", "--items", str(ITEMS), "--provider", "gemini", "--model", "m",
                         "--max-calls", "1", "--out", str(Path(tmp) / "r")])
        self.assertEqual(code, 2)

    def test_cli_mock_run_and_score(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp) / "run"
            self.assertEqual(main(["run", "--items", str(ITEMS), "--max-calls", "50", "--out", str(out)]), 0)
            self.assertEqual(main(["score", "--items", str(ITEMS), "--predictions", str(out / "predictions.jsonl"),
                                   "--out", str(Path(tmp) / "score.json")]), 0)
            summary = json.loads((Path(tmp) / "score.json").read_text())
            attempts = (out / "attempts.jsonl").read_text()
        self.assertEqual(len(summary["conditions"]), 5)
        self.assertEqual({c["contrast"] for c in summary["contrasts"]},
                         {"context_gain", "multimodal_gain", "text_gain", "image_only_gain", "distractor_penalty"})
        self.assertEqual(attempts.count("\n"), 20)


class ScoringTests(unittest.TestCase):
    def test_paired_gains_and_failures(self):
        items = load_items(ITEMS)
        keys = {item.id: mcq_options(item)[1] for item in items}
        rows = []
        for item in items:
            for condition in ("A1", "A3"):
                right = condition == "A3" or item.id == "syn-001"
                status = "inference_failure" if (item.id, condition) == ("syn-004", "A1") else "ok"
                wrong = next(label for label in "ABCD" if label != keys[item.id])
                rows.append({"model": "m", "task": "mcq", "example_id": item.id, "condition": condition,
                             "status": status, "selected_option": keys[item.id] if right else wrong})
        summary = summarize(score_rows(rows, items), n_bootstrap=200)
        gain = next(c for c in summary["contrasts"] if c["contrast"] == "multimodal_gain")
        self.assertEqual((gain["n_pairs"], gain["n_possible"]), (3, 4))
        self.assertAlmostEqual(gain["mean_gain"], 2 / 3)
        a1 = next(c for c in summary["conditions"] if c["condition"] == "A1")
        self.assertEqual((a1["n_scored"], a1["mean_score_all_attempts"]), (3, 0.25))
        self.assertEqual(summarize(score_rows(rows, items), n_bootstrap=200), summary)

    def test_token_f1(self):
        self.assertEqual(token_f1("a b", "a b"), 1.0)
        self.assertEqual(token_f1("x", "y"), 0.0)
        self.assertAlmostEqual(token_f1("a b c", "a"), 0.5)


class RetryAndPacingTests(unittest.TestCase):
    def setUp(self):
        self.stimulus = build_stimuli(load_items(ITEMS)[:1], conditions=("A1",), task="mcq")[0]

    def test_transient_error_is_retried_then_succeeds(self):
        provider = ScriptedProvider(providers.ProviderError("HTTP 429: slow down", status=429), VALID)
        log, sleeps = [], []
        result = run_one(provider, self.stimulus, "mcq", Budget(5), log.append,
                         transient_retries=2, sleep=sleeps.append)
        self.assertEqual((result["status"], result["attempts"], result["selected_option"]), ("ok", 1, "B"))
        self.assertEqual([r["transient_try"] for r in log], [0, 1])
        self.assertEqual(sleeps, [2.0])

    def test_retries_consume_budget(self):
        provider = ScriptedProvider(providers.ProviderError("HTTP 503", status=503), VALID)
        result = run_one(provider, self.stimulus, "mcq", Budget(1), lambda r: None,
                         transient_retries=2, sleep=lambda seconds: None)
        self.assertEqual((result["status"], result["attempts"]), ("inference_failure", 1))
        self.assertIn("HTTP 503", result["error"])
        self.assertEqual(len(provider.prompts), 1)

    def test_non_transient_errors_are_not_retried(self):
        provider = ScriptedProvider(providers.ProviderError("HTTP 401", status=401), VALID)
        log, sleeps = [], []
        result = run_one(provider, self.stimulus, "mcq", Budget(5), log.append,
                         transient_retries=2, sleep=sleeps.append)
        self.assertEqual((result["status"], len(log), sleeps), ("inference_failure", 1, []))

    def test_paced_provider_waits_between_calls(self):
        clock, sleeps = [0.0], []

        def sleep(seconds):
            clock[0] += seconds
            sleeps.append(seconds)

        paced = Paced(ScriptedProvider(VALID, VALID), 5.0, clock=lambda: clock[0], sleep=sleep)
        self.assertEqual((paced.name, paced.live), ("scripted", False))
        paced.generate("a", [])
        clock[0] = 2.0
        paced.generate("b", [])
        self.assertEqual(sleeps, [3.0])

    def test_negative_retries_are_rejected_before_any_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r"
            with self.assertRaises(ValueError):
                run(providers.MockProvider(), [self.stimulus], task="mcq", out_dir=out, max_calls=5,
                    transient_retries=-1)
            self.assertFalse(out.exists())


class ItemDraftTests(unittest.TestCase):
    @staticmethod
    def _row(index, term, meaning, popular, usage, scope):
        return {"Term": term, "Meaning": meaning, "Subject area": "Gaming", "Popular meaning": popular,
                "Example usage": usage, "Source URL": "https://example.test/post",
                "Occurrence ID": f"occ-{index}", "Term group ID": f"{index:024x}",
                "original_row_index": index, "Selection basis": "TECHNIQUE",
                "game_or_context_scopes": [scope]}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rows = [
            self._row(1, "drift stack", "Boost charge from chained drifts.", "A pile of karts.",
                      "Hold the drift to build a drift stack.", "Arena Alpha"),
            self._row(2, "pebble call", "A callout that a shortcut is blocked.",
                      "No established general sense", "Make the pebble call early.", "Arena Alpha"),
            self._row(3, "glimmer dash", "A brief invulnerable dash.", "A sparkle effect.",
                      "Glimmer dash off the ledge.", "Arena Alpha"),
            self._row(4, "tidelock", "Trapped in water and unable to cast.", "Locking a skin.",
                      "Tidelocked under the water zone.", "Arena Alpha"),
            self._row(5, "ember sweep", "A low kick that resets a stance.", "A cleaning method.",
                      "Ember sweep resets them.", "Arena Beta"),
            self._row(6, "rune ping", "A signal marking a rune spawn.", "A sound effect.",
                      "Ping the rune before it spawns.", "Arena Beta"),
            self._row(7, "lane freeze", "Holding creeps in place near the tower.", "Cold weather.",
                      "Lane freeze until minute five.", "Arena Beta"),
            self._row(8, "bait swap", "Trading a cheap ability for a costly one.", "Fishing term.",
                      "Bait swap before the fight.", "Arena Beta"),
        ]
        self.graph = self.root / "graph.json"
        self.graph.write_text(json.dumps({"original_terms": rows}), encoding="utf-8")
        self.review = self.root / "review.csv"

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _read(path):
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return reader.fieldnames, list(reader)

    def test_draft_is_deterministic_and_flags_missing_popular_meaning(self):
        other = self.root / "again.csv"
        summary = draft_review(self.graph, ["Arena Alpha", "Arena Beta"], self.review, seed=3)
        draft_review(self.graph, ["Arena Alpha", "Arena Beta"], other, seed=3)
        self.assertEqual(self.review.read_bytes(), other.read_bytes())
        _, rows = self._read(self.review)
        by_term = {r["term"]: r for r in rows}
        self.assertEqual(summary["rows"], 8)
        self.assertIn("popular_meaning_missing", by_term["pebble call"]["flags"])
        self.assertEqual(by_term["pebble call"]["distractor_1_source"], "same_scope_meaning")
        self.assertTrue(all(r["keep"] == "" for r in rows))

    def test_reviewed_sheet_builds_loadable_items(self):
        draft_review(self.graph, ["Arena Alpha", "Arena Beta"], self.review, seed=3)
        images = self.root / "images"
        images.mkdir()
        keep = {"drift stack": ("explicit", "high"), "tidelock": ("implicit", "low"),
                "ember sweep": ("explicit", "high"), "rune ping": ("implicit", "low")}
        fieldnames, rows = self._read(self.review)
        for row in rows:
            if row["term"] in keep:
                row.update(keep="y", text_explicitness=keep[row["term"]][0],
                           visual_groundability=keep[row["term"]][1],
                           image_file=f"{row['term'].replace(' ', '_')}.png", image_source="own screenshot")
                (images / row["image_file"]).write_bytes(b"\x89PNG " + row["term"].encode())
        with self.review.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        out = self.root / "items.jsonl"
        summary = build_items(self.review, images, out)
        self.assertEqual((summary["items"], sorted(summary["by_community"])), (4, ["Arena Alpha", "Arena Beta"]))
        items = load_items(out)
        self.assertEqual(sorted(i.term for i in items), sorted(keep))
        self.assertTrue(all(len(i.distractor_meanings) == 3 for i in items))
        self.assertEqual(items[0].metadata["gold_status"], "provisional_single_reviewer")
        with self.assertRaises(FileExistsError):
            build_items(self.review, images, out)

    def test_invalid_sheet_reports_every_problem_and_writes_nothing(self):
        draft_review(self.graph, ["Arena Alpha"], self.review, seed=3)
        fieldnames, rows = self._read(self.review)
        rows[0].update(keep="y", text_explicitness="maybe", visual_groundability="high",
                       image_file="missing.png")
        with self.review.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        out = self.root / "items.jsonl"
        with self.assertRaises(ValueError) as caught:
            build_items(self.review, self.root, out)
        message = str(caught.exception)
        self.assertIn("text_explicitness must be one of", message)
        self.assertIn("image not found", message)
        self.assertIn("at least two communities", message)
        self.assertFalse(out.exists())

    def test_unknown_scope_and_overwrite_are_refused(self):
        with self.assertRaises(ValueError):
            draft_review(self.graph, ["No Such Game"], self.root / "x.csv")
        draft_review(self.graph, ["Arena Alpha"], self.review)
        with self.assertRaises(FileExistsError):
            draft_review(self.graph, ["Arena Alpha"], self.review)


class BundledGraphTests(unittest.TestCase):
    def test_default_scopes_have_enough_terms(self):
        counts = scope_counts(load_gaming_rows(DEFAULT_GRAPH))
        for scope in DEFAULT_SCOPES:
            self.assertGreaterEqual(counts[scope], 10, scope)

    def test_cli_lists_scopes_without_writing(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["draft-items", "--list-scopes"]), 0)
        self.assertIn("Super Smash Bros. Melee", out.getvalue())


class HistoricalResultsTests(unittest.TestCase):
    def rows(self, name):
        with (RESULTS / name).open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_tables_are_internally_consistent(self):
        for row in self.rows("matched50_pipelines.csv"):
            total = float(row["known_extraction_usd"]) + float(row["known_judgment_usd"])
            self.assertAlmostEqual(total, float(row["known_total_usd"]), places=6)
            self.assertEqual(int(row["sources_complete"]) + int(row["sources_partial"]) + int(row["sources_failed"]), 50)
        for row in self.rows("judgment_replays.csv"):
            self.assertAlmostEqual(int(row["validated_ids"]) / int(row["eligible_ids"]), float(row["coverage"]), places=3)
        by_extractor = {}
        for row in self.rows("matched50_by_community.csv"):
            by_extractor[row["extractor"]] = by_extractor.get(row["extractor"], 0) + int(row["raw_proposals"])
        pipelines = {r["pipeline"].split(" -> ")[0]: int(r["raw_proposals"]) for r in self.rows("matched50_pipelines.csv")[:2]}
        self.assertEqual(by_extractor, pipelines)
        summary = json.loads((RESULTS / "summary.json").read_text())
        self.assertEqual(summary["network_calls"], 0)
        for name in summary["files"]:
            self.assertTrue((RESULTS / name).is_file(), name)

    def test_results_hold_no_urls_source_ids_or_local_paths(self):
        for path in RESULTS.rglob("*"):
            if path.suffix not in {".csv", ".json", ".svg"}:
                continue
            text = path.read_text(encoding="utf-8")
            for marker in ("http://", "https://", "lv-", "nic_", "/Users/"):
                if path.suffix == ".svg" and marker in ("http://", "https://"):
                    continue  # SVG namespace declarations
                self.assertNotIn(marker, text, f"{marker!r} in {path.name}")


if __name__ == "__main__":
    unittest.main()
