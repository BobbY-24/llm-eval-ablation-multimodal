"""Explicit local rendered exports; raw Age of Empires II replays are not videos."""
import math

from ..core import Sample, Window, Finding, stable_id
from ..media import decode_video
from ._common import LocalAdapter, bounded_paths, relative, json_rows

MAX_RETURNED_SIDECAR_ROWS = 4096


class AoE2Adapter(LocalAdapter):
    dataset = "aoe2"
    game = "Age of Empires II"
    declared_modalities = ["video"]
    empty_note = "Provide a local rendered-export manifest.jsonl; no game client or replay renderer is invoked."

    def iter_samples(self):
        for manifest in bounded_paths(self.root, ("manifest.jsonl",)):
            if manifest.name != "manifest.jsonl":
                continue
            for row in json_rows(manifest):
                if "video_path" not in row:
                    raise ValueError("AoE export row requires video_path")
                assets = {"video": row["video_path"], "manifest": relative(self.root, manifest)}
                if row.get("replay_path"):
                    assets["replay"] = row["replay_path"]
                modalities = ["video"]
                for kind in ("events", "states", "actions"):
                    if row.get(kind + "_path"):
                        assets[kind] = row[kind + "_path"]
                        modalities.append(kind)
                yield Sample(sample_id=stable_id(self.dataset, row.get("source_replay_id"), row["video_path"]),
                             dataset=self.dataset, game=row.get("game", self.game), assets=assets, modalities=modalities,
                             native_clock={"unit": row.get("time_unit", "seconds"), "fps": row.get("fps"),
                                           "origin": "rendered_export", "timing": "encoded_pts"},
                             source=row.get("source", {}), lineage={"source_replay_id": row.get("source_replay_id"),
                                 "session_id": row.get("session_id"), "parent_ids": row.get("parent_ids", [])},
                             version=row.get("game_version"), dataset_revision=row.get("dataset_revision"),
                             metadata={"record": row, "rendered_export": True})

    def probe(self):
        result = super().probe()
        if result.status == "missing" and list(bounded_paths(self.root, (".aoe2record", ".mgz", ".mgx", ".mgl"))):
            result.status = "replay_only"
            result.notes.append("Raw replay files contain events/state, not RGB. Supply separately rendered local exports.")
        return result

    def read_window(self, sample, range):
        output = {}
        returned_rows = 0
        for kind in ("events", "states", "actions"):
            if kind not in sample.assets:
                continue
            sidecar_unit = sample.native_clock["unit"]
            if sidecar_unit != range.unit:
                raise ValueError("Sidecar time_unit must match requested range; no implicit clock conversion")
            rows = []
            for row in json_rows(self.asset(sample, kind)):
                key = "timestamp" if range.unit == "seconds" else "index"
                if key not in row:
                    raise ValueError(f"{kind} sidecar requires {key}")
                position = float(row[key])
                if not math.isfinite(position):
                    raise ValueError("Sidecar timestamp/index must be finite")
                if position >= range.start and (range.end is None or position < range.end):
                    returned_rows += 1
                    if returned_rows > MAX_RETURNED_SIDECAR_ROWS:
                        raise ValueError("AoE window exceeds aggregate sidecar row limit")
                    rows.append(row)
            output[kind] = rows
        return Window(sample_id=sample.sample_id,
                      frames=decode_video(self.asset(sample, "video"), range, native_fps=sample.native_clock.get("fps")),
                      actions=output.get("actions"), states=output.get("states"), annotations=output.get("events"),
                      metadata={"rendered_export": True, "source_replay_id": sample.lineage.get("source_replay_id"),
                                "game_version": sample.version, "original_record": sample.metadata["record"],
                                "absent_modalities": [key for key in ("audio", "actions", "states", "events") if key not in sample.assets]})

    def validate(self, sample):
        findings = super().validate(sample)
        if sample.native_clock["unit"] not in {"seconds", "frames"}:
            findings.append(Finding("unsupported_time_unit", "Export time_unit must be seconds or frames", sample_id=sample.sample_id))
        fps = sample.native_clock.get("fps")
        if fps is not None and (not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0):
            findings.append(Finding("invalid_fps", "Export fps must be finite and positive", sample_id=sample.sample_id))
        if not sample.version:
            findings.append(Finding("unknown_game_version", "Record the replay and playback game versions before semantic evaluation", "warning", sample.sample_id))
        row = sample.metadata["record"]
        recorded = row.get("recorded_game_version")
        rendered = row.get("rendered_game_version")
        if recorded and rendered and recorded != rendered:
            findings.append(Finding("playback_version_mismatch", "Recorded and playback game versions differ", sample_id=sample.sample_id))
        return findings
