"""Synthetic schema/alignment tests. No remote datasets or game clients needed."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import io
import struct
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from multimodal_game_eval.adapters import ADAPTERS, create_adapter
from multimodal_game_eval.adapters._common import numpy_zip_guard
from multimodal_game_eval.adapters.dota2vis import parse_yolo
from multimodal_game_eval.adapters.starcraftimage import decode_unit_bags
from multimodal_game_eval.core import TimeRange, MissingDependency


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_PIL = importlib.util.find_spec("PIL") is not None
HAS_H5PY = importlib.util.find_spec("h5py") is not None


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def jsonl(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_registry_and_empty_probes_need_no_optional_libraries(self):
        self.assertEqual(set(ADAPTERS), {"vpt", "minerl", "csgo", "terraria", "proact", "starcraftimage", "slippi", "dota2vis", "aoe2"})
        for name in ADAPTERS:
            result = create_adapter(name, self.root).probe()
            self.assertEqual(result.sample_count, 0)
            self.assertIn(result.status, {"missing", "gated_missing_assets"})

    def test_vpt_preserves_idle_actions_and_recorder_clock(self):
        (self.root / "session.mp4").touch()
        rows = [{"mouse": {"buttons": [], "newButtons": []}, "keyboard": {"keys": []},
                 "tick": i, "milli": 5000 + i * 50, "hotbar": i} for i in range(4)]
        self.jsonl(self.root / "session.jsonl", rows)
        adapter = create_adapter("vpt", self.root)
        sample = next(adapter.iter_samples())
        with patch("multimodal_game_eval.adapters.vpt.decode_video", return_value=[{"index": 1}, {"index": 2}]):
            output = adapter.read_window(sample, TimeRange(1, 3, "frames"))
        self.assertEqual([r["index"] for r in output.actions], [1, 2])
        self.assertEqual(output.actions[0]["milli"], 5050)
        self.assertTrue(output.metadata["idle_frames_preserved"])
        self.assertFalse(output.metadata["action_corrections"])

    def test_vpt_mismatched_frame_action_stream_is_rejected(self):
        (self.root / "session.mp4").touch()
        self.jsonl(self.root / "session.jsonl", [{"mouse": {}, "keyboard": {}}])
        adapter = create_adapter("vpt", self.root)
        with patch("multimodal_game_eval.adapters.vpt.decode_video", return_value=[]):
            with self.assertRaisesRegex(ValueError, "alignment"):
                adapter.read_window(next(adapter.iter_samples()), TimeRange(0, 1, "frames"))

    def test_terraria_retains_source_and_absent_labels(self):
        (self.root / "videos").mkdir()
        (self.root / "videos" / "abc.mp4").touch()
        (self.root / "metadata.tsv").write_text("id\tsource_url\tchannel\tduration_s\nabc\thttps://example.test/source\tcreator\t40\n")
        adapter = create_adapter("terraria", self.root)
        sample = next(adapter.iter_samples())
        self.assertEqual(sample.source["channel"], "creator")
        self.assertEqual(sample.lineage["source_video_id"], "abc")
        with patch("multimodal_game_eval.adapters.terraria.decode_video", return_value=[]):
            output = adapter.read_window(sample, TimeRange(0, 1))
        self.assertIsNone(output.actions)
        self.assertIsNone(output.audio)

    def test_terraria_retains_and_filters_publisher_provenance(self):
        (self.root / "abc.mp4").touch()
        (self.root / "provenance").mkdir()
        (self.root / "provenance/reviews.tsv").write_text(
            "video_id\tdecision\nabc\tkeep\nother\treject\n")
        adapter = create_adapter("terraria", self.root)
        sample = next(adapter.iter_samples())
        self.assertEqual(sample.assets["provenance_0"], "provenance/reviews.tsv")
        with patch("multimodal_game_eval.adapters.terraria.decode_video", return_value=[]):
            output = adapter.read_window(sample, TimeRange(0, 1))
        rows = output.metadata["filtering_provenance"]["provenance_0"]["rows"]
        self.assertEqual(rows, [{"video_id": "abc", "decision": "keep"}])
        with patch("multimodal_game_eval.adapters.terraria.MAX_PROVENANCE_ROWS", 0):
            with self.assertRaisesRegex(ValueError, "aggregate return budget"):
                adapter.read_window(sample, TimeRange(0, 1))

    def test_proact_missing_video_is_partial_and_eval_schema_is_used(self):
        row = {"idx": 7, "dataset_name": "csgo", "tag": "solo", "video_path": "game_commentary/CSGO/videos/x.mp4",
               "video_begin": 10, "video_end": 20, "video_duration": 10, "history": "earlier context",
               "active_speaker": {"persona": []},
               "annotations": [{"role": "assistant", "speaker": "a", "start": 11, "end": 12, "text": "event"}]}
        self.jsonl(self.root / "DATA/anns/main_games.jsonl", [row])
        adapter = create_adapter("proact", self.root)
        self.assertEqual(adapter.probe().status, "partial_available")
        sample = next(adapter.iter_samples())
        self.assertEqual(sample.game, "Counter-Strike: Global Offensive")
        with patch("multimodal_game_eval.adapters.proact.decode_video", return_value=[]):
            output = adapter.read_window(sample, TimeRange(11, 12))
        self.assertEqual(output.annotations[0]["text"], "event")
        self.assertEqual(output.metadata["time_origin"], "source_video")

    def test_proact_rejects_external_video_reference(self):
        self.jsonl(self.root / "main_games.jsonl", [{"video_path": "../outside.mp4"}])
        self.assertEqual(create_adapter("proact", self.root).probe().status, "invalid")

    def test_proact_discovers_nested_training_metadata_and_keeps_split(self):
        video = "DATA/game_commentary/Minecraft/videos/source.mp4"
        (self.root / video).parent.mkdir(parents=True)
        (self.root / video).touch()
        row = {"video_path": "game_commentary/Minecraft/videos/source.mp4", "video_begin": 3, "video_end": 6,
               "metadata": {"dataset_name": "minecraft", "tag": "solo"},
               "annotations": [{"role": "assistant", "start": 4, "end": 5, "text": "mining"}]}
        self.jsonl(self.root / "DATA/anns/minecraft_final_train.jsonl", [row])
        self.jsonl(self.root / "DATA/anns/minecraft_final_val.jsonl", [row])
        adapter = create_adapter("proact", self.root)
        samples = list(adapter.iter_samples())
        self.assertEqual(len(samples), 2)
        self.assertEqual({s.lineage["annotation_split"] for s in samples}, {"train", "val"})
        self.assertTrue(all(s.game == "Minecraft" and s.assets["video"] == video for s in samples))
        self.assertEqual(samples[0].metadata["record"]["tag"], "solo")

    def test_proact_explicit_nested_local_clip_fields(self):
        (self.root / "clip.mp4").touch()
        self.jsonl(self.root / "anns/csgo_final_train_t.jsonl", [{"metadata": {
            "video_path": "clip.mp4", "dataset_name": "csgo", "video_begin": 1, "video_end": 2,
            "tag": "co-commentary"}, "annotations": []}])
        sample = next(create_adapter("proact", self.root).iter_samples())
        self.assertEqual(sample.game, "Counter-Strike: Global Offensive")
        self.assertEqual(sample.metadata["record"]["video_begin"], 1)
        self.assertEqual(sample.lineage["annotation_split"], "train")

    def test_aoe_raw_replays_are_not_videos(self):
        (self.root / "match.aoe2record").touch()
        adapter = create_adapter("aoe2", self.root)
        self.assertEqual(adapter.probe().status, "replay_only")
        self.assertEqual(list(adapter.iter_samples()), [])

    def test_aoe_explicit_export_sidecar_clock(self):
        (self.root / "rendered.mp4").touch()
        self.jsonl(self.root / "manifest.jsonl", [{"video_path": "rendered.mp4", "events_path": "events.jsonl",
                                                   "time_unit": "seconds", "source_replay_id": "r1"}])
        self.jsonl(self.root / "events.jsonl", [{"timestamp": 1, "type": "build"}, {"timestamp": 4, "type": "attack"}])
        adapter = create_adapter("aoe2", self.root)
        sample = next(adapter.iter_samples())
        with patch("multimodal_game_eval.adapters.aoe2.decode_video", return_value=[]):
            output = adapter.read_window(sample, TimeRange(0, 3))
        self.assertEqual([r["type"] for r in output.annotations], ["build"])
        self.assertEqual(sample.lineage["source_replay_id"], "r1")

    def test_aoe_sidecar_limit_is_shared_across_modalities(self):
        (self.root / "rendered.mp4").touch()
        self.jsonl(self.root / "manifest.jsonl", [{"video_path": "rendered.mp4", "events_path": "events.jsonl",
                                                  "states_path": "states.jsonl", "time_unit": "seconds"}])
        for filename in ("events.jsonl", "states.jsonl"):
            self.jsonl(self.root / filename, [{"timestamp": 0}, {"timestamp": 1}])
        adapter = create_adapter("aoe2", self.root)
        sample = next(adapter.iter_samples())
        with patch("multimodal_game_eval.adapters.aoe2.MAX_RETURNED_SIDECAR_ROWS", 3):
            with patch("multimodal_game_eval.adapters.aoe2.decode_video") as decoder:
                with self.assertRaisesRegex(ValueError, "aggregate sidecar row limit"):
                    adapter.read_window(sample, TimeRange(0, 2))
                decoder.assert_not_called()

    def test_csgo_aggregate_budget_precedes_second_image_allocation(self):
        # Stub the optional readers to exercise the actual window path using
        # tiny arrays even when h5py/NumPy are unavailable.
        class Array:
            size = 3
            shape = (1, 1, 3)
            dtype = SimpleNamespace(name="uint8", kind="u", itemsize=1)
            def __getitem__(self, key):
                return self
            def copy(self):
                return self

        class Handle(dict):
            def get(self, key, default=None, *, getlink=False):
                return type("HardLink", (), {})() if getlink else super().get(key, default)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False

        handle = Handle(frame_0_x=Array(), frame_1_x=Array())
        (self.root / "tiny.hdf5").touch()
        adapter = create_adapter("csgo", self.root)
        sample = next(adapter.iter_samples())
        modules = {"h5py": SimpleNamespace(File=lambda *args: handle),
                   "numpy": SimpleNamespace(asarray=lambda value: value)}
        with patch("multimodal_game_eval.adapters.csgo.require_dependency", side_effect=modules.__getitem__):
            with patch("multimodal_game_eval.adapters.csgo.MAX_WINDOW_BYTES", 500):
                with self.assertRaisesRegex(ValueError, "aggregate materialization budget"):
                    adapter.read_window(sample, TimeRange(0, 2, "frames"))

    def test_csgo_python_container_estimate_includes_empty_nested_lists(self):
        from multimodal_game_eval.adapters.csgo import materialization_bytes
        array = SimpleNamespace(size=0, shape=(1_000_000, 0), dtype=SimpleNamespace(itemsize=1))
        self.assertGreater(materialization_bytes(array, python_lists=True), 60_000_000)

    def test_yolo_normalized_box_conversion_and_bad_values(self):
        labels = self.root / "labels.txt"
        labels.write_text("2 0.5 0.5 0.2 0.4\n")
        row = parse_yolo(labels, 200, 100)[0]
        self.assertEqual(row["bbox_xyxy"], [80.0, 30.0, 120.0, 70.0])
        labels.write_text("0 nan 0.5 0.2 0.4\n")
        with self.assertRaises(ValueError):
            parse_yolo(labels, 200, 100)

    def test_minerl_missing_dependency_is_not_invalid_data(self):
        folder = self.root / "trajectory"
        folder.mkdir()
        for name in ("rendered.npz", "recording.mp4", "metadata.json"):
            (folder / name).touch()
        adapter = create_adapter("minerl", self.root)
        sample = next(adapter.iter_samples())
        with patch("multimodal_game_eval.adapters.minerl.require_dependency", side_effect=MissingDependency("numpy absent")):
            findings = adapter.validate(sample)
        self.assertEqual(findings[0].code, "missing_dependency")
        self.assertEqual(findings[0].severity, "warning")

    def test_slippi_archived_replays_deduplicate_and_keep_member_out_of_assets(self):
        # A minimal documented raw envelope suffices for safe inventory testing;
        # it is not passed to a third-party parser.
        from multimodal_game_eval.adapters.slippi import _PREFIX
        raw = bytes([0x35, 4, 0x39, 0, 1, 0x39, 3])
        replay = _PREFIX + struct.pack(">i", len(raw)) + raw
        (self.root / "local.slp").write_bytes(replay)
        with tarfile.open(self.root / "shard.tar.gz", "w:gz") as archive:
            member = tarfile.TarInfo("nested/duplicate.slp")
            member.size = len(replay)
            archive.addfile(member, io.BytesIO(replay))
        adapter = create_adapter("slippi", self.root)
        samples = list(adapter.iter_samples())
        self.assertEqual(len(samples), 1)
        self.assertEqual(len(samples[0].lineage["duplicates"]), 1)
        self.assertNotIn("member", samples[0].assets)
        self.assertEqual(adapter.probe().status, "replay_only")

    def test_slippi_archive_traversal_is_rejected_without_extraction(self):
        with tarfile.open(self.root / "bad.tar", "w") as archive:
            member = tarfile.TarInfo("../outside.slp")
            member.size = 4
            archive.addfile(member, io.BytesIO(b"nope"))
        adapter = create_adapter("slippi", self.root)
        self.assertEqual(adapter.probe().status, "invalid")
        self.assertFalse((self.root.parent / "outside.slp").exists())

    def slippi_fixture(self):
        from multimodal_game_eval.adapters.slippi import _PREFIX
        raw = bytes([0x35, 4, 0x39, 0, 1, 0x39, 3])
        replay = _PREFIX + struct.pack(">i", len(raw)) + raw
        (self.root / "local.slp").write_bytes(replay)
        game = SimpleNamespace(frames=SimpleNamespace(id=[0, 1, 2, 3], ports=[SimpleNamespace(
            leader=SimpleNamespace(pre=SimpleNamespace(buttons=[0, 1, 2, 3]),
                                   post=SimpleNamespace(percent=[0, 1, 2, 3])), follower=None)]),
            start=SimpleNamespace(players=[SimpleNamespace(port=0)], stage=31, is_pal=False))
        return hashlib.sha256(replay).hexdigest(), game

    def test_slippi_rendered_video_is_hash_linked_and_uses_native_frame_origin(self):
        digest, game = self.slippi_fixture()
        (self.root / "render.mp4").touch()
        row = {"replay_sha256": digest, "video_path": "render.mp4", "video_start_frame": -2,
               "game_version": "NTSC 1.02", "fps": 60, "renderer": "reviewed local capture",
               "provenance": {"alignment_review": "start/middle/end events checked"}}
        self.jsonl(self.root / "rendered_manifest.jsonl", [row])
        adapter = create_adapter("slippi", self.root)
        sample = next(adapter.iter_samples())
        self.assertIn("video", sample.modalities)
        self.assertEqual(sample.version, "NTSC 1.02")
        self.assertEqual(sample.assets["rendered_manifest"], "rendered_manifest.jsonl")
        self.assertEqual(adapter.probe().status, "available")
        pixels = [{"index": 2, "timestamp": 2 / 60, "pts": 20, "rgb": "pixel-a"},
                  {"index": 3, "timestamp": 3 / 60, "pts": 30, "rgb": "pixel-b"}]
        with patch("multimodal_game_eval.adapters.slippi.decode_video", return_value=pixels) as decoder:
            output = adapter._window(game, sample, "frames", 0, 2, {"finalized_through_frame": 3})
        self.assertEqual(decoder.call_args.args[1], TimeRange(2, 4, "frames"))
        self.assertEqual([f["frame_id"] for f in output.frames], [0, 1])
        self.assertEqual([f["video_index"] for f in output.frames], [2, 3])
        self.assertEqual([f["index"] for f in output.frames], [2, 3])
        self.assertEqual(output.frames[0]["pts"], 20)
        self.assertEqual(output.frames[0]["timestamp"], 0)
        self.assertTrue(output.metadata["rgb_available"])

    def test_slippi_unmatched_rendered_hash_does_not_attach_pixels(self):
        self.slippi_fixture()
        (self.root / "render.mp4").touch()
        self.jsonl(self.root / "rendered_manifest.jsonl", [{"replay_sha256": "f" * 64,
            "video_path": "render.mp4", "video_start_frame": 0, "game_version": "NTSC 1.02", "fps": 60}])
        adapter = create_adapter("slippi", self.root)
        sample = next(adapter.iter_samples())
        self.assertNotIn("video", sample.assets)
        self.assertEqual(adapter.probe().status, "replay_only")
        self.assertIn("slippi_rendered_unmatched", {f.code for f in adapter.validate(sample)})

    def test_slippi_duplicate_and_conflicting_rendered_mappings_are_findings(self):
        digest, _ = self.slippi_fixture()
        for filename in ("one.mp4", "two.mp4"):
            (self.root / filename).touch()
        row = {"replay_sha256": digest, "video_path": "one.mp4", "video_start_frame": 0,
               "game_version": "NTSC 1.02", "fps": 60}
        self.jsonl(self.root / "rendered_manifest.jsonl", [row, row, {**row, "video_path": "two.mp4"}])
        adapter = create_adapter("slippi", self.root)
        sample = next(adapter.iter_samples())
        self.assertNotIn("video", sample.assets)
        codes = {f.code for f in adapter.validate(sample)}
        self.assertTrue({"slippi_rendered_duplicate", "slippi_rendered_conflict"} <= codes)

    def test_slippi_pal_capture_requires_and_uses_explicit_frame_mapping(self):
        digest, game = self.slippi_fixture()
        game.start.is_pal = True
        (self.root / "render.mp4").touch()
        row = {"replay_sha256": digest, "video_path": "render.mp4", "video_start_frame": 0,
               "game_version": "PAL 1.00", "fps": 100, "frame_map": [
                   {"video_index": 0, "native_frame": 0}, {"video_index": 1, "native_frame": 0},
                   {"video_index": 2, "native_frame": 1}]}
        self.jsonl(self.root / "rendered_manifest.jsonl", [row])
        adapter = create_adapter("slippi", self.root)
        sample = next(adapter.iter_samples())
        with patch("multimodal_game_eval.adapters.slippi.decode_video", return_value=[
                {"index": i, "timestamp": i / 100} for i in range(3)]):
            output = adapter._window(game, sample, "frames", 0, 2, {"finalized_through_frame": 3})
        self.assertEqual([f["frame_id"] for f in output.frames], [0, 0, 1])
        self.assertEqual([f["index"] for f in output.frames], [0, 1, 2])
        self.assertEqual(output.frames[2]["timestamp"], 1 / 50)

    def test_slippi_rollback_resolution_and_separate_finalized_watermark(self):
        from multimodal_game_eval.adapters.slippi import SlippiAdapter
        from multimodal_game_eval.core import Sample
        sample = Sample("test", "slippi", "Super Smash Bros. Melee", {}, ["actions", "states"], {"unit": "frames"})
        pre = SimpleNamespace(buttons=[0, 1, 2, 101, 102, 103])
        post = SimpleNamespace(percent=[0, 1, 2, 11, 12, 13])
        game = SimpleNamespace(frames=SimpleNamespace(id=[0, 1, 2, 1, 2, 3],
                               ports=[SimpleNamespace(leader=SimpleNamespace(pre=pre, post=post), follower=None)]),
                               start=SimpleNamespace(players=[SimpleNamespace(port=0)], stage=31, is_pal=False))
        info = {"finalized_through_frame": 2, "has_game_end": True}
        adapter = SlippiAdapter(self.root)
        output = adapter._window(game, sample, "frames", 0, 4, info)
        self.assertEqual([r["frame_id"] for r in output.states], [0, 1, 2, 3])
        self.assertEqual([r["ports"]["1"]["leader"]["percent"] for r in output.states], [0, 11, 12, 13])
        self.assertEqual([r["finalized"] for r in output.states], [True, True, True, False])
        unknown = adapter._window(game, sample, "frames", 0, 4, {"finalized_through_frame": None})
        self.assertTrue(all(r["finalized"] is None for r in unknown.states))
        self.assertIsNone(output.frames)

    def test_implicit_terraria_metadata_symlink_is_rejected(self):
        outside = self.root.parent / (self.root.name + "-outside.tsv")
        outside.write_text("id\tchannel\na\tprivate\n")
        try:
            (self.root / "metadata.tsv").symlink_to(outside)
            self.assertEqual(create_adapter("terraria", self.root).probe().status, "invalid")
        finally:
            outside.unlink()

    @unittest.skipUnless(HAS_NUMPY, "NumPy optional reader dependency absent")
    def test_minerl_terminal_observation_and_padding(self):
        import numpy as np
        folder = self.root / "trajectory"
        folder.mkdir()
        np.savez(folder / "rendered.npz", reward=np.array([1, 2]), **{"action$jump": np.array([0, 1]),
                  "observation$inventory$wood": np.array([0, 1, 2])})
        (folder / "recording.mp4").touch()
        (folder / "metadata.json").write_text(json.dumps({"true_video_frame_count": 5}))
        adapter = create_adapter("minerl", self.root)
        sample = next(adapter.iter_samples())
        self.assertEqual(adapter.validate(sample), [])
        with patch("multimodal_game_eval.adapters.minerl.decode_video", return_value=[]) as decode:
            output = adapter.read_window(sample, TimeRange(0, 2, "frames"))
        self.assertEqual(decode.call_args.kwargs["leading_padding"], 2)
        self.assertEqual(output.actions[1]["action"]["jump"], 1)
        self.assertEqual(output.states[-1]["next_observation"]["inventory"]["wood"], 2)
        self.assertTrue(output.annotations[-1]["done"])
        with self.assertRaises(ValueError):
            adapter.read_window(sample, TimeRange(0, 1, "seconds"))

    @unittest.skipUnless(HAS_NUMPY, "NumPy optional reader dependency absent")
    def test_minerl_object_arrays_never_unpickle(self):
        import numpy as np
        folder = self.root / "trajectory"
        folder.mkdir()
        np.savez(folder / "rendered.npz", reward=np.array([0]), **{"action$bad": np.array([{}], dtype=object)})
        (folder / "recording.mp4").touch()
        (folder / "metadata.json").write_text("{}")
        adapter = create_adapter("minerl", self.root)
        findings = adapter.validate(next(adapter.iter_samples()))
        self.assertTrue(any(f.code == "unsafe_or_invalid_npz" for f in findings))

    @unittest.skipUnless(HAS_NUMPY, "NumPy optional reader dependency absent")
    def test_starcraft_exact_band_and_storage_channel_mapping(self):
        import numpy as np
        image = np.zeros((192, 128, 3), dtype=np.uint8)
        # Player1 is storage channel2. First horizontal plane stores one unit.
        image[2, 3, 2] = 25
        image[66, 3, 2] = 7
        image[128:192, :64, 2] = 1
        # Player2 is channel0; neutral is channel1, independently decoded.
        image[4, 5, 0] = 9
        image[68, 5, 0] = 3
        image[6, 7, 1] = 12
        image[70, 7, 1] = 4
        bags = decode_unit_bags(image)
        self.assertEqual(bags["player_1_unit_ids"].shape, (1, 64, 64))
        self.assertEqual(bags["player_1_unit_ids"][0, 2, 3], 7)
        self.assertEqual(bags["player_1_unit_values"][0, 2, 3], 25)
        self.assertEqual(bags["neutral_unit_ids"][0, 6, 7], 4)
        self.assertNotIn("neutral_map_state", bags)

    @unittest.skipUnless(HAS_NUMPY, "NumPy optional reader dependency absent")
    def test_starcraft_memory_limit_covers_all_decoded_bands(self):
        import numpy as np
        image = np.ones((192, 64, 3), dtype=np.uint8)
        # The encoded input plus any one band fits; all eight bands do not.
        with patch("multimodal_game_eval.adapters.starcraftimage.MAX_SPATIAL_BYTES", 90_000):
            with self.assertRaisesRegex(ValueError, "aggregate memory budget"):
                decode_unit_bags(image)

    @unittest.skipUnless(HAS_NUMPY and HAS_H5PY, "NumPy/h5py optional dependencies absent")
    def test_csgo_native_bgr_to_rgb_and_numeric_frame_order(self):
        import h5py
        import numpy as np
        with h5py.File(self.root / "expert.hdf5", "w") as stream:
            for i in (0, 1, 2):
                stream[f"frame_{i}_x"] = np.array([[[10, 20, 30]]], dtype=np.uint8)
                stream[f"frame_{i}_y"] = np.array([i])
        adapter = create_adapter("csgo", self.root)
        output = adapter.read_window(next(adapter.iter_samples()), TimeRange(1, 3, "frames"))
        self.assertEqual([frame["index"] for frame in output.frames], [1, 2])
        self.assertEqual(output.frames[0]["rgb"].tolist(), [[[30, 20, 10]]])

    @unittest.skipUnless(HAS_NUMPY and HAS_H5PY, "NumPy/h5py optional dependencies absent")
    def test_csgo_gap_selection_uses_native_ids(self):
        import h5py
        import numpy as np
        with h5py.File(self.root / "gap.hdf5", "w") as stream:
            for i in (0, 5):
                stream[f"frame_{i}_x"] = np.zeros((1, 1, 3), dtype=np.uint8)
                stream[f"frame_{i}_y"] = np.array([i])
        adapter = create_adapter("csgo", self.root)
        output = adapter.read_window(next(adapter.iter_samples()), TimeRange(1, 3, "frames"))
        self.assertEqual(output.frames, [])

    @unittest.skipUnless(HAS_NUMPY and HAS_H5PY, "NumPy/h5py optional dependencies absent")
    def test_csgo_rejects_external_hdf5_links(self):
        import h5py
        with h5py.File(self.root / "links.hdf5", "w") as stream:
            stream["frame_0_x"] = h5py.ExternalLink("outside.hdf5", "image")
            stream["frame_0_y"] = [0]
        adapter = create_adapter("csgo", self.root)
        findings = adapter.validate(next(adapter.iter_samples()))
        self.assertTrue(any(f.code == "invalid_hdf5" for f in findings))

    @unittest.skipUnless(HAS_NUMPY and HAS_PIL, "NumPy/Pillow optional dependencies absent")
    def test_dota_image_has_no_invented_timestamp(self):
        from PIL import Image
        (self.root / "YOLO/train/images").mkdir(parents=True)
        (self.root / "YOLO/train/labels").mkdir()
        Image.new("RGB", (100, 100)).save(self.root / "YOLO/train/images/a.png")
        (self.root / "YOLO/train/labels/a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        adapter = create_adapter("dota2vis", self.root)
        self.assertEqual(adapter.probe().status, "partial_available")
        output = adapter.read_window(next(adapter.iter_samples()), TimeRange(0, 1, "frames"))
        self.assertIsNone(output.frames[0]["timestamp"])


if __name__ == "__main__":
    unittest.main()
