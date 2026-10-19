"""OpenAI VPT contractor MP4/JSONL pairs, preserving raw steps."""
from ..core import Sample, Window, Finding, TimeRange, stable_id
from ..media import decode_video
from ._common import LocalAdapter, bounded_paths, relative, json_rows, json_object, selected_indices


class VPTAdapter(LocalAdapter):
    dataset = "vpt"
    game = "Minecraft"
    declared_modalities = ["video", "actions", "state"]

    def iter_samples(self):
        for video in bounded_paths(self.root, (".mp4",)):
            actions = video.with_suffix(".jsonl")
            if not actions.is_file():
                continue
            rel = relative(self.root, video)
            version = video.parent.name if video.parent.name[:1].isdigit() else None
            options = video.with_name(video.stem + "-options.json")
            assets = {"video": rel, "actions": relative(self.root, actions)}
            if options.is_file():
                assets["options"] = relative(self.root, options)
            yield Sample(sample_id=stable_id(self.dataset, rel), dataset=self.dataset, game=self.game,
                         assets=assets,
                         modalities=list(self.declared_modalities),
                         native_clock={"unit": "steps", "fps": 20, "timing": "nominal", "origin": "segment"},
                         source={"url": "https://github.com/openai/Video-Pre-Training"}, dataset_revision=version,
                         lineage={"segment_stem": video.stem, "session_hint": video.stem.rsplit("-", 2)[0]},
                         metadata={"transformations": [], "missing_segments_possible": True,
                                   "recorder_version_from_directory": version})

    def read_window(self, sample, range):
        rows = list(json_rows(self.asset(sample, "actions")))
        indices = selected_indices(range, len(rows), 20)
        frames = decode_video(self.asset(sample, "video"), TimeRange(indices.start, indices.stop, "frames"), native_fps=20) if indices else []
        if [frame["index"] for frame in frames] != list(indices):
            raise ValueError("VPT video/action alignment mismatch in selected window")
        actions, states = [], []
        for i in indices:
            row = rows[i]
            common = {"index": i, "timestamp": i / 20, "tick": row.get("tick"), "milli": row.get("milli")}
            actions.append({**common, "mouse": row.get("mouse"), "keyboard": row.get("keyboard")})
            states.append({**common, **{k: v for k, v in row.items() if k not in {"mouse", "keyboard", "tick", "milli"}}})
        return Window(sample_id=sample.sample_id, frames=frames, actions=actions, states=states,
                      metadata={"raw_alignment": True, "idle_frames_preserved": True,
                                "cursor_compositing": False, "action_corrections": False,
                                "timebase": sample.native_clock,
                                "options": json_object(self.asset(sample, "options")) if "options" in sample.assets else None,
                                "alignment_checked_frames": len(frames),
                                "implicit_window_limit": 4096 if range.end is None else None})

    def validate(self, sample):
        findings = super().validate(sample)
        if findings:
            return findings
        try:
            rows = list(json_rows(self.asset(sample, "actions")))
            if not rows:
                raise ValueError("Empty action stream")
            ticks = []
            for row in rows:
                if not isinstance(row.get("mouse"), dict) or not isinstance(row.get("keyboard"), dict):
                    raise ValueError("VPT action row requires mouse and keyboard objects")
                if isinstance(row.get("tick"), (int, float)):
                    ticks.append(row["tick"])
            if any(b < a for a, b in zip(ticks, ticks[1:])):
                findings.append(Finding("nonmonotonic_tick", "Recorder tick decreases within segment", sample_id=sample.sample_id))
        except (ValueError, OSError, TypeError) as error:
            findings.append(Finding("invalid_vpt", str(error), sample_id=sample.sample_id))
        return findings
