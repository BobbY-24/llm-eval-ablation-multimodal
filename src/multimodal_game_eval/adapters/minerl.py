"""Legacy MineRL transition streams; object-array pickle loading is forbidden."""
import math
from ..core import Sample, Window, Finding, MissingDependency, stable_id, require_dependency
from ..media import decode_video
from ._common import LocalAdapter, bounded_paths, relative, json_object, numpy_zip_guard, selected_indices


def unflatten(values):
    output = {}
    for key, value in values.items():
        current = output
        parts = key.split("$")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return output


class MineRLAdapter(LocalAdapter):
    dataset = "minerl"
    game = "Minecraft"
    declared_modalities = ["video", "actions", "state", "reward"]

    def iter_samples(self):
        for array in bounded_paths(self.root, ("rendered.npz",)):
            if array.name != "rendered.npz":
                continue
            folder = array.parent
            assets = {"arrays": relative(self.root, array),
                      "video": relative(self.root, folder / "recording.mp4"),
                      "metadata": relative(self.root, folder / "metadata.json")}
            rel = relative(self.root, folder)
            yield Sample(sample_id=stable_id(self.dataset, rel), dataset=self.dataset, game=self.game,
                         assets=assets, modalities=list(self.declared_modalities),
                         native_clock={"unit": "steps", "fps": None, "origin": "trajectory", "timing": "step_index"},
                         source={"url": "https://zenodo.org/records/12659939"}, dataset_revision="legacy-v0",
                         lineage={"trajectory": folder.name, "task": folder.parent.name})

    def read_window(self, sample, range):
        np = require_dependency("numpy")
        array_path = self.asset(sample, "arrays")
        numpy_zip_guard(array_path)
        meta = json_object(self.asset(sample, "metadata"))
        # The legacy format has no guaranteed timestamp column. Seconds require
        # an explicit trusted local fps; media PTS can still be decoded separately.
        fps = meta.get("fps")
        fps = float(fps) if fps is not None else None
        if fps is not None and (not math.isfinite(fps) or fps <= 0):
            raise ValueError("MineRL metadata fps must be finite and positive")
        actions, states, annotations = [], [], []
        with np.load(array_path, allow_pickle=False) as arrays:
            rewards = arrays["reward"]
            count = len(rewards)
            indices = selected_indices(range, count, fps)
            action_keys = [k for k in arrays.files if k.startswith("action$")]
            state_keys = [k for k in arrays.files if k.startswith("observation$")]
            # Access all selected numeric columns before decoding video. An object
            # dtype raises here, rather than silently enabling unpickling.
            action_arrays = {k: arrays[k] for k in action_keys}
            state_arrays = {k: arrays[k] for k in state_keys}
            if any(len(a) != count for a in action_arrays.values()) or any(len(a) != count + 1 for a in state_arrays.values()):
                raise ValueError("MineRL transition arrays do not align with reward and terminal observation")
            for i in indices:
                common = {"index": i, "timestamp": i / fps if fps else None}
                actions.append({**common, "action": unflatten({k.split("$", 1)[1]: a[i].tolist() for k, a in action_arrays.items()})})
                states.append({**common, "observation": unflatten({k.split("$", 1)[1]: a[i].tolist() for k, a in state_arrays.items()}),
                               "next_observation": unflatten({k.split("$", 1)[1]: a[i + 1].tolist() for k, a in state_arrays.items()})})
                annotations.append({**common, "reward": rewards[i].tolist(), "done": i == count - 1})
            padding = int(meta.get("true_video_frame_count", count + 1)) - (count + 1)
            if padding < 0:
                raise ValueError("Video frame count is shorter than transition count + terminal state")
        frames = decode_video(self.asset(sample, "video"), range, native_fps=fps, leading_padding=padding)
        return Window(sample_id=sample.sample_id, frames=frames, actions=actions, states=states,
                      annotations=annotations,
                      metadata={"terminal_observation_extra": True, "leading_padding": padding,
                                "pickle_allowed": False, "fps": fps,
                                "implicit_window_limit": 4096 if range.end is None else None})

    def validate(self, sample):
        findings = super().validate(sample)
        if findings:
            return findings
        try:
            np = require_dependency("numpy")
            numpy_zip_guard(self.asset(sample, "arrays"))
            with np.load(self.asset(sample, "arrays"), allow_pickle=False) as arrays:
                count = len(arrays["reward"])
                for key in arrays.files:
                    array = arrays[key]
                    expected = count if key.startswith("action$") or key == "reward" else count + 1
                    if key.startswith(("action$", "observation$")) and len(array) != expected:
                        findings.append(Finding("length_mismatch", f"{key}: {len(array)} != {expected}", sample_id=sample.sample_id))
        except (ImportError, MissingDependency) as error:
            findings.append(Finding("missing_dependency", str(error), "warning", sample.sample_id))
        except (ValueError, OSError, KeyError) as error:
            findings.append(Finding("unsafe_or_invalid_npz", str(error), sample_id=sample.sample_id))
        return findings
