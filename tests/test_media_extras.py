"""Real local NumPy/Pillow I/O fixtures, not downloaded game media."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from multimodal_game_eval.media import decode_image
from multimodal_game_eval.core import TimeRange
from multimodal_game_eval.adapters import create_adapter


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL"), "Optional NumPy/Pillow not installed in this runtime")
class ImageExtraTests(unittest.TestCase):
    def test_real_ppm_decodes_to_rgb_without_resizing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.ppm"
            path.write_bytes(b"P6\n2 1\n255\n" + bytes([255, 0, 0, 0, 128, 255]))
            image = decode_image(path)
            self.assertEqual(image.shape, (1, 2, 3))
            self.assertEqual(image.tolist(), [[[255, 0, 0], [0, 128, 255]]])

    def test_real_packed_png_is_spatial_evidence_without_gameplay_pixels(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "png_files").mkdir()
            (root / "metadata.csv").write_text("global_idx,replay_name,window_idx,map_name\n0,fixture,1,fixture-map\n")
            encoded = np.zeros((192, 64, 3), dtype=np.uint8)
            encoded[0, 0, 2] = 7
            encoded[64, 0, 2] = 12
            Image.fromarray(encoded).save(root / "png_files/idx_0__replay_fixture__window_1.png")
            reader = create_adapter("starcraftimage", root)
            sample = next(reader.iter_samples())
            self.assertEqual(reader.validate(sample), [])
            window = reader.read_window(sample, TimeRange(0, 1, "frames"))
            self.assertIsNone(window.frames)
            self.assertEqual(window.spatial["player_1_unit_values"][0, 0, 0], 7)
            self.assertEqual(window.spatial["player_1_unit_ids"][0, 0, 0], 12)
            self.assertFalse(window.metadata["rgb_gameplay"])

    def test_csgo_real_numpy_color_conversion_with_mocked_hdf_reader(self):
        import numpy as np
        class Handle(dict):
            def get(self, key, default=None, *, getlink=False):
                return type("HardLink", (), {})() if getlink else super().get(key, default)
            def __enter__(self): return self
            def __exit__(self, *args): pass
        original = np.array([[[3, 2, 1]]], dtype=np.uint8)
        handle = Handle({"frame_0_x": original, "frame_0_y": np.array([1])})
        fake_hdf = SimpleNamespace(File=lambda *args: handle)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "expert_fixture.h5").touch()
            reader = create_adapter("csgo", root)
            sample = next(reader.iter_samples())
            with patch("multimodal_game_eval.adapters.csgo.require_dependency", side_effect=lambda module: fake_hdf if module == "h5py" else np):
                window = reader.read_window(sample, TimeRange(0, 1, "frames"))
            self.assertEqual(window.frames[0]["rgb"].tolist(), [[[1, 2, 3]]])
            self.assertEqual(original.tolist(), [[[3, 2, 1]]])
            self.assertEqual(sample.metadata["action_provenance"], "recorded")
