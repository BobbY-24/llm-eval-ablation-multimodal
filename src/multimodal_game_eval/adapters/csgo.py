"""CS:GO screenshot/action HDF5 files. Native images are BGR."""
import re

from ..core import Sample, Window, Finding, MissingDependency, stable_id, require_dependency
from ._common import LocalAdapter, bounded_paths, relative, selected_indices, MAX_ARRAY_BYTES

MAX_WINDOW_BYTES = 512 * 1024 * 1024


def materialization_bytes(array, *, python_lists=False):
    """Conservative allocation estimate, including conversion temporaries.

    Numeric sidecars become Python lists, whose element/container overhead can
    greatly exceed the HDF5 storage size (including empty nested dimensions).
    """
    raw = int(array.size) * int(array.dtype.itemsize)
    if not python_lists:
        return 2 * raw + 256  # native BGR read plus retained RGB copy
    containers, prefix = 1, 1
    for dimension in array.shape[:-1]:
        prefix *= int(dimension)
        containers += prefix
    return raw + 64 * (int(array.size) + containers) + 256


class CSGOAdapter(LocalAdapter):
    dataset = "csgo"
    game = "Counter-Strike: Global Offensive"
    declared_modalities = ["video", "actions", "state", "events"]

    def iter_samples(self):
        for path in bounded_paths(self.root, (".hdf5", ".h5")):
            rel = relative(self.root, path)
            expert = "expert" in rel.lower()
            yield Sample(sample_id=stable_id(self.dataset, rel), dataset=self.dataset, game=self.game,
                         assets={"hdf5": rel}, modalities=list(self.declared_modalities),
                         native_clock={"unit": "frames", "fps": 16, "timing": "nominal", "origin": "file"},
                         source={"url": "https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch"},
                         lineage={"recording_file": path.stem},
                         metadata={"native_color": "BGR", "action_provenance": "recorded" if expert else "inferred",
                                   "mouse_targets": "quantized_one_hot"})

    @staticmethod
    def frame_ids(handle):
        ids = []
        for count, key in enumerate(handle.keys()):
            if count >= 100_000:
                raise ValueError("HDF5 key inventory exceeds bound")
            if m := re.fullmatch(r"frame_(\d+)_x", key):
                ids.append(int(m.group(1)))
        return sorted(ids)

    @staticmethod
    def numeric_array(handle, key):
        # External/soft links and virtual/external storage could read unrelated
        # files. This adapter accepts self-contained numeric datasets only.
        link = handle.get(key, getlink=True)
        if type(link).__name__ != "HardLink":
            raise ValueError("HDF5 links outside ordinary internal datasets are unsupported")
        array = handle[key]
        if getattr(array, "is_virtual", False) or getattr(array, "external", None):
            raise ValueError("HDF5 external/virtual dataset storage is unsupported")
        if array.dtype.kind not in "biuf" or array.size * array.dtype.itemsize > MAX_ARRAY_BYTES:
            raise ValueError("HDF5 dataset must be bounded numeric storage")
        return array

    def read_window(self, sample, range):
        h5py = require_dependency("h5py")
        np = require_dependency("numpy")
        frames, actions, states, annotations = [], [], [], []
        materialized = 0

        def reserve(array, *, python_lists=False):
            nonlocal materialized
            materialized += materialization_bytes(array, python_lists=python_lists)
            if materialized > MAX_WINDOW_BYTES:
                raise ValueError("CS:GO window exceeds aggregate materialization budget")

        with h5py.File(self.asset(sample, "hdf5"), "r") as handle:
            ids = self.frame_ids(handle)
            selection = selected_indices(range, ids[-1] + 1 if ids else 0, 16)
            for i in ids:
                if i not in selection:
                    continue
                common = {"index": i, "timestamp": i / 16, "timing": "nominal"}
                image = self.numeric_array(handle, f"frame_{i}_x")
                if image.dtype.name != "uint8" or image.size * image.dtype.itemsize > MAX_ARRAY_BYTES or len(image.shape) != 3 or image.shape[-1] != 3:
                    raise ValueError("Invalid or oversized CS:GO image")
                reserve(image)
                frames.append({**common, "rgb": np.asarray(image)[..., ::-1].copy()})
                for suffix, target, field in [("y", actions, "target"), ("xaux", states, "aux"), ("helperarr", annotations, "kill_death")]:
                    key = f"frame_{i}_{suffix}"
                    if key in handle:
                        array = self.numeric_array(handle, key)
                        if array.size * array.dtype.itemsize > MAX_ARRAY_BYTES:
                            raise ValueError("Oversized HDF5 array")
                        reserve(array, python_lists=True)
                        target.append({**common, field: np.asarray(array).tolist()})
        return Window(sample_id=sample.sample_id, frames=frames, actions=actions, states=states,
                      annotations=annotations, metadata={"native_color": "BGR", "output_color": "RGB",
                                                         "timebase": sample.native_clock,
                                                         "implicit_window_limit": 4096 if range.end is None else None})

    def validate(self, sample):
        findings = super().validate(sample)
        if findings:
            return findings
        try:
            h5py = require_dependency("h5py")
            with h5py.File(self.asset(sample, "hdf5"), "r") as handle:
                ids = self.frame_ids(handle)
                if not ids:
                    raise ValueError("No frame_i_x keys")
                if ids != list(range(ids[0], ids[-1] + 1)):
                    findings.append(Finding("frame_gaps", "HDF5 native frame indices contain gaps", "warning", sample.sample_id))
                if any(f"frame_{i}_y" not in handle for i in ids):
                    findings.append(Finding("missing_action", "Some frames lack action targets", sample_id=sample.sample_id))
                for i in ids:
                    image = self.numeric_array(handle, f"frame_{i}_x")
                    if image.dtype.name != "uint8" or len(image.shape) != 3 or image.shape[-1] != 3:
                        raise ValueError("CS:GO images must be uint8 HWC with three native BGR channels")
        except (ImportError, MissingDependency) as error:
            findings.append(Finding("missing_dependency", str(error), "warning", sample.sample_id))
        except (ValueError, OSError, KeyError) as error:
            findings.append(Finding("invalid_hdf5", str(error), sample_id=sample.sample_id))
        return findings
