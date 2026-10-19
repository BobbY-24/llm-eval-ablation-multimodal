import unittest

from multimodal_game_eval.core import DatasetError
from multimodal_game_eval.sampling import select_frames


class SamplingTests(unittest.TestCase):
    def test_motion_frames_cannot_be_silently_dropped(self):
        frames = [{"index": i, "timestamp": i / 60} for i in range(100)]
        with self.assertRaises(DatasetError):
            select_frames(frames, 64)
        selected, record = select_frames(frames, 8, "uniform", [49, 50])
        self.assertEqual(len(selected), 8)
        self.assertTrue({49, 50} <= {frame["index"] for frame in selected})
        self.assertEqual(record["pixel_transform"], "none")
        self.assertFalse(record["temporal_interpolation"])
        self.assertEqual(len(frames), 100)

    def test_missing_required_frame_is_an_error(self):
        with self.assertRaises(DatasetError):
            select_frames([{"index": 0}], 1, "uniform", [1])

    def test_small_window_is_retained(self):
        frames = [{"index": i, "timestamp": i / 60} for i in range(4)]
        selected, record = select_frames(frames)
        self.assertEqual(selected, frames)
        self.assertEqual(record["selected_timestamps"], [i / 60 for i in range(4)])
