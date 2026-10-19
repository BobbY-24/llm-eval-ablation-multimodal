import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

from multimodal_game_eval.core import DatasetError, TimeRange, check_archive, read_json, read_jsonl, safe_path, sha256_file, stable_id
from multimodal_game_eval.inventory import load_inventory, validate_annotation_join
from multimodal_game_eval.io import write_json, write_jsonl
from multimodal_game_eval.media import decode_video


class CoreTests(unittest.TestCase):
    def test_range_is_end_exclusive_and_finite(self):
        self.assertEqual(TimeRange(-123, 0, "frames").start, -123)
        for args in ((0, 0), (float("nan"), 2), (0, float("inf")), (.5, 2, "frames")):
            with self.assertRaises(DatasetError):
                TimeRange(*args)

    def test_path_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for path in ("../private", "/etc/passwd", "C:\\secret", "a\\b", "bad\x00name"):
                with self.assertRaises(DatasetError):
                    safe_path(root, path)
            (root / "link").symlink_to("/etc")
            with self.assertRaises(DatasetError):
                safe_path(root, "link/passwd")

    def test_json_bounds_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.json"
            path.write_text('{"a": NaN}')
            with self.assertRaises(DatasetError):
                read_json(path)
            path.write_text('{"a": 1}')
            with self.assertRaises(DatasetError):
                read_json(path, max_bytes=3)
            path.write_text('[]\n')
            with self.assertRaises(DatasetError):
                list(read_jsonl(path))

    def test_archive_rejects_tar_and_zip_traversal_and_links(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with tarfile.open(root / "bad.tar", "w") as archive:
                member = tarfile.TarInfo("../escape")
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))
                link = tarfile.TarInfo("link")
                link.type = tarfile.SYMTYPE
                link.linkname = "/private"
                archive.addfile(link)
            self.assertEqual(len(check_archive(root / "bad.tar")), 2)
            with zipfile.ZipFile(root / "bad.zip", "w") as archive:
                archive.writestr("../escape", "x")
            self.assertTrue(check_archive(root / "bad.zip"))
            with tarfile.open(root / "good.tar", "w") as archive:
                member = tarfile.TarInfo("data/a")
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))
            self.assertEqual(check_archive(root / "good.tar"), [])

    def test_atomic_export_and_hash_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            write_jsonl(path, [{"n": 1}, {"n": 2}])
            self.assertEqual(len(list(read_jsonl(path))), 2)
            with self.assertRaises(DatasetError):
                write_json(path, {})
            with self.assertRaises(DatasetError):
                sha256_file(path, max_bytes=1)
            self.assertEqual(stable_id("x", {"a": 1}), stable_id("x", {"a": 1}))

    def test_inventory_join_uses_exact_sense(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "inventory.json", {"scopes": [{"scope_id": "g", "label": "Game", "scope_type": "game", "terms": [{"sense_id": "s", "term_group_id": "t", "term": "term"}]}]})
            write_json(root / "graph.json", {"usages": [{"sense_id": "s", "term_group_id": "t", "occurrence_id": "u"}, {"sense_id": "other", "term_group_id": "t", "occurrence_id": "other-u"}]})
            inventory = load_inventory(root / "inventory.json", root / "graph.json")
            annotation = {"sense_id": "s", "scope_id": "g", "term_group_id": "t", "game": "Game", "occurrence_ids": ["u"]}
            self.assertEqual(validate_annotation_join(annotation, inventory), [])
            annotation["occurrence_ids"] = ["other-u"]
            self.assertIn("unknown_occurrence_id", validate_annotation_join(annotation, inventory))


class MediaClockTests(unittest.TestCase):
    def fake_av(self):
        frames = [SimpleNamespace(pts=i * 2, time_base=Fraction(1, 120), width=2, height=2,
                                  to_ndarray=lambda format: [["RGB"]]) for i in range(7)]

        class Container:
            streams = SimpleNamespace(video=[SimpleNamespace(average_rate=60)])
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def decode(self, stream): return iter(frames)

        return SimpleNamespace(open=lambda path: Container())

    def test_padding_does_not_change_native_frame_alignment(self):
        with patch("multimodal_game_eval.media.require_dependency", return_value=self.fake_av()):
            frames = decode_video("fixture.mp4", TimeRange(1, 3, "frames"), native_fps=60, leading_padding=2)
        self.assertEqual([frame["index"] for frame in frames], [1, 2])
        self.assertEqual([frame["pts"] for frame in frames], [6, 8])
        self.assertAlmostEqual(frames[0]["timestamp"], 1 / 60)
        self.assertAlmostEqual(frames[0]["pts_seconds"], 3 / 60)

    def test_time_window_uses_pts(self):
        with patch("multimodal_game_eval.media.require_dependency", return_value=self.fake_av()):
            frames = decode_video("fixture.mp4", TimeRange(2 / 60, 4 / 60))
        self.assertEqual([frame["index"] for frame in frames], [2, 3])


if __name__ == "__main__":
    unittest.main()
