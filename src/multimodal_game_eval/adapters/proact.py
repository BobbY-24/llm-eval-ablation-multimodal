"""Local Proact-VL evaluation/training JSONL and video references; no gates."""
import math
import re

from ..core import Sample, Window, Probe, Finding, TimeRange, stable_id, safe_path
from ..media import decode_video
from ._common import LocalAdapter, bounded_paths, relative, json_rows

GAMES = {"minecraft": "Minecraft", "starcraft2": "StarCraft II", "csgo": "Counter-Strike: Global Offensive",
         "lol": "League of Legends", "cyberpunk_2077": "Cyberpunk 2077", "elden_ring": "Elden Ring",
         "baldurs_gate_3": "Baldur's Gate 3", "streetfighter6": "Street Fighter 6",
         "tears_of_the_kingdom": "The Legend of Zelda: Tears of the Kingdom",
         "yu_gi_oh": "Yu-Gi-Oh!", "black_myth_wukong": "Black Myth: Wukong", "wukong": "Black Myth: Wukong"}


class ProactAdapter(LocalAdapter):
    dataset = "proact"
    declared_modalities = ["video", "text"]
    empty_note = "Proact-VL files are gated; provide an authorized local export. This adapter never accepts conditions."

    def _video_reference(self, raw):
        raw = str(raw)
        direct = safe_path(self.root, raw)
        if direct.is_file():
            return relative(self.root, direct)
        alternatives = ["DATA/" + raw]
        if raw.startswith("DATA/"):
            alternatives.append(raw[5:])
        for candidate in alternatives:
            path = safe_path(self.root, candidate)
            if path.is_file():
                return relative(self.root, path)
        return relative(self.root, direct)

    def iter_samples(self):
        files = list(bounded_paths(self.root, (".jsonl",)))
        recognized = {"main_games.jsonl", "streaming_games.jsonl", "wukong.jsonl"}
        final_pattern = re.compile(r".+_final_(train|val)(?:_[A-Za-z0-9_-]+)?\.jsonl$")
        preferred = [p for p in files if p.name in recognized or final_pattern.fullmatch(p.name)]
        if not preferred:
            preferred = [p for p in files if p.name in {"all_in_one.jsonl", "manifest.jsonl"}]
        seen = set()
        for manifest in preferred:
            match = final_pattern.fullmatch(manifest.name)
            split = match.group(1) if match else "evaluation" if manifest.name in recognized or manifest.name == "all_in_one.jsonl" else "unspecified"
            for line, row in enumerate(json_rows(manifest)):
                info = row.get("metadata", {})
                if not isinstance(info, dict):
                    raise ValueError("Proact metadata must be an object")
                # Published training rows put dataset_name/tag under metadata.
                # Also retain explicit nested clip fields in authorized local
                # exports, with present top-level fields taking precedence.
                record = {**info, **row}
                if not record.get("video_path"):
                    continue
                if not isinstance(record["video_path"], str):
                    raise ValueError("Proact video_path must be a string")
                name = str(record.get("dataset_name", "unknown"))
                if name.lower() == "ego4d":
                    continue
                video = self._video_reference(record["video_path"])
                begin, end = record.get("video_begin", 0), record.get("video_end", record.get("video_duration"))
                tag = record.get("tag")
                # Preserve training/evaluation and annotation-file provenance;
                # identical clips in different splits must remain auditable.
                key = (video, begin, end, tag, split, manifest.stem)
                if key in seen:
                    continue
                seen.add(key)
                yield Sample(sample_id=stable_id(self.dataset, *key), dataset=self.dataset,
                             game=GAMES.get(name.lower(), name), assets={"video": video, "manifest": relative(self.root, manifest)},
                             modalities=list(self.declared_modalities),
                             native_clock={"unit": "seconds", "annotation_resolution_seconds": 1, "origin": "source_video"},
                             source={"url": ("https://huggingface.co/datasets/oaaoaa/game_commentary_sft" if split in {"train", "val"}
                                              else "https://huggingface.co/datasets/oaaoaa/LiveGamingBenchmark"), "line": line + 1},
                             lineage={"source_video": video, "source_idx": record.get("idx"), "annotation_split": split},
                             metadata={"record": record, "raw_record": row, "annotation_split": split,
                                       "audio_unverified": True})

    def probe(self):
        result = super().probe()
        if result.status == "missing":
            result.status = "gated_missing_assets"
        elif result.status == "available":
            from itertools import islice
            if any(not self.asset(s, "video").is_file() for s in islice(self.iter_samples(), 1000)):
                result.status = "partial_available"
                result.notes.append("Annotations exist but some referenced videos are absent.")
        return result

    def read_window(self, sample, range):
        if range.unit != "seconds":
            raise ValueError("Proact annotation windows use source-video seconds")
        row = sample.metadata["record"]
        begin = float(row.get("video_begin", 0))
        end = float(row.get("video_end", begin + row.get("video_duration", 0)))
        start = max(begin, range.start)
        stop = min(end, range.end) if range.end is not None else end
        if stop <= start:
            return Window(sample_id=sample.sample_id, frames=[], annotations=[], metadata={"outside_clip": True})
        annotations = []
        for ann in row.get("annotations", []):
            ann_start = float(ann["start"])
            ann_end = float(ann.get("end", ann_start))
            if ann_start < stop and (ann_end > start or ann_start >= start):
                annotations.append(dict(ann))
        return Window(sample_id=sample.sample_id,
                      frames=decode_video(self.asset(sample, "video"), TimeRange(start, stop)), annotations=annotations,
                      metadata={"history": row.get("history", ""), "active_speaker": row.get("active_speaker"),
                                "time_origin": "source_video", "audio_unverified": True,
                                "annotation_split": sample.metadata.get("annotation_split"),
                                "annotation_provenance": "ASR and model-processed labels; not independent semantic gold"})

    def validate(self, sample):
        findings = super().validate(sample)
        row = sample.metadata.get("record", {})
        try:
            begin = float(row.get("video_begin", 0))
            end = float(row.get("video_end", begin + row.get("video_duration", 0)))
            if not (math.isfinite(begin) and math.isfinite(end) and 0 <= begin < end):
                raise ValueError("Invalid Proact clip interval")
            for ann in row.get("annotations", []):
                start = float(ann["start"])
                stop = float(ann.get("end", start))
                if not (math.isfinite(start) and math.isfinite(stop) and begin <= start <= stop <= end):
                    raise ValueError("Annotation lies outside clip interval")
                if ann.get("role") not in {"assistant", "user"}:
                    raise ValueError("Annotation role must be assistant or user")
                if "text" not in ann and "query" not in ann:
                    raise ValueError("Annotation requires text or query")
        except (ValueError, KeyError, TypeError) as error:
            findings.append(Finding("invalid_proact", str(error), sample_id=sample.sample_id))
        return findings
