"""Dota2-Vis partial YOLO minimap image release; full videos remain separate."""
import math

from ..core import Sample, Window, Probe, Finding, MissingDependency, stable_id, require_dependency
from ..media import decode_image
from ._common import LocalAdapter, bounded_paths, relative, text_file, MAX_RECORDS


def parse_yolo(path, width, height):
    annotations = []
    with text_file(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number > MAX_RECORDS:
                raise ValueError("YOLO annotation count exceeds bound")
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"YOLO line {line_number} requires class cx cy width height")
            class_id = int(fields[0])
            cx, cy, bw, bh = map(float, fields[1:])
            if class_id < 0 or not all(math.isfinite(x) and 0 <= x <= 1 for x in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
                raise ValueError(f"Invalid normalized YOLO box on line {line_number}")
            annotations.append({"class_id": class_id, "bbox_cxcywh_normalized": [cx, cy, bw, bh],
                                "bbox_xyxy": [(cx - bw / 2) * width, (cy - bh / 2) * height,
                                              (cx + bw / 2) * width, (cy + bh / 2) * height]})
    return annotations


class Dota2VisAdapter(LocalAdapter):
    dataset = "dota2vis"
    game = "Dota 2"
    declared_modalities = ["image", "boxes"]
    empty_note = "Only local YOLO images/labels are supported; full match-video access is unresolved."

    def iter_samples(self):
        for image in bounded_paths(self.root, (".png", ".jpg", ".jpeg")):
            if image.parent.name != "images" or image.parent.parent.name not in {"train", "valid", "val", "test"}:
                continue
            label = image.parent.parent / "labels" / (image.stem + ".txt")
            rel = relative(self.root, image)
            yield Sample(sample_id=stable_id(self.dataset, rel), dataset=self.dataset, game=self.game,
                         assets={"image": rel, "labels": relative(self.root, label)},
                         modalities=list(self.declared_modalities), native_clock={"unit": "items", "timestamp": None},
                         source={"url": "https://github.com/RicardoRCarvalho/dota2-vis"},
                         lineage={"source_image": image.stem}, dataset_revision=None,
                         metadata={"split": image.parent.parent.name, "continuous_video": False,
                                   "label_format": "YOLO detection", "license_notice": "BY-NC-SA 4.0"})

    def probe(self):
        result = super().probe()
        if result.status == "available":
            result.status = "partial_available"
            result.notes.append("Minimap annotation images only; full videos are not established by this release.")
        return result

    def _names(self):
        candidates = list(bounded_paths(self.root, ("data.yaml",)))
        if not candidates:
            return {}
        yaml = require_dependency("yaml")
        data = yaml.safe_load(text_file(candidates[0]).read_text(encoding="utf-8"))
        names = data.get("names", {}) if isinstance(data, dict) else {}
        if isinstance(names, list):
            return dict(enumerate(names))
        return {int(k): str(v) for k, v in names.items()}

    def read_window(self, sample, range):
        if range.unit != "frames" or range.start != 0 or (range.end is not None and range.end != 1):
            raise ValueError("Still-image samples use TimeRange(0,1,'frames'); timestamps are unavailable")
        rgb = decode_image(self.asset(sample, "image"))
        annotations = parse_yolo(self.asset(sample, "labels"), rgb.shape[1], rgb.shape[0])
        names = self._names()
        for annotation in annotations:
            annotation["class_name"] = names.get(annotation["class_id"])
            if names and annotation["class_id"] not in names:
                raise ValueError("YOLO class index absent from data.yaml")
        return Window(sample_id=sample.sample_id, frames=[{"index": 0, "timestamp": None, "rgb": rgb}],
                      annotations=annotations, metadata={"continuous_video": False, "split": sample.metadata["split"]})

    def validate(self, sample):
        findings = super().validate(sample)
        if findings:
            return findings
        try:
            from ..core import TimeRange
            self.read_window(sample, TimeRange(0, 1, "frames"))
        except (ImportError, MissingDependency) as error:
            findings.append(Finding("missing_dependency", str(error), "warning", sample.sample_id))
        except (ValueError, OSError, KeyError, TypeError) as error:
            findings.append(Finding("invalid_yolo", str(error), sample_id=sample.sample_id))
        return findings
