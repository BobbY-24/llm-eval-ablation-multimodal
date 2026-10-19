"""Silent Terraria MP4 corpus and its retained provenance tables."""
from pathlib import PurePosixPath
from urllib.parse import parse_qs, urlparse

from ..core import Sample, Window, stable_id
from ..media import decode_video
from ._common import LocalAdapter, bounded_paths, relative, csv_rows

MAX_PROVENANCE_TABLES = 64
MAX_PROVENANCE_ROWS = 4096
MAX_PROVENANCE_BYTES = 32 * 1024 * 1024


def source_matches(row, source_id, source_url):
    """Join only explicit IDs, exact URLs, or named media-file stems."""
    if not isinstance(source_id, str) or not source_id:
        return False
    if any(row.get(key) == source_id for key in ("id", "video_id", "youtube_id", "source_video_id")):
        return True
    for key in ("source_url", "url", "video_url"):
        value = row.get(key)
        if not value:
            continue
        if source_url and value == source_url:
            return True
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower()
        if host in {"youtube.com", "www.youtube.com", "m.youtube.com"} and parse_qs(parsed.query).get("v") == [source_id]:
            return True
        if host == "youtu.be" and parsed.path.strip("/") == source_id:
            return True
    return any(PurePosixPath(str(row.get(key, "")).replace("\\", "/")).stem == source_id
               for key in ("filename", "video_path", "path") if row.get(key))


class TerrariaAdapter(LocalAdapter):
    dataset = "terraria"
    game = "Terraria"
    declared_modalities = ["video"]

    def iter_samples(self):
        metadata = self.root / "metadata.tsv"
        by_id = {r.get("id", r.get("video_id", "")): r for r in csv_rows(metadata, "\t")} if metadata.is_file() else {}
        provenance = {}
        for table in bounded_paths(self.root, (".tsv",)):
            rel = relative(self.root, table)
            if not rel.startswith("provenance/"):
                continue
            if len(provenance) >= MAX_PROVENANCE_TABLES:
                raise ValueError("Terraria provenance table inventory exceeds bound")
            provenance[f"provenance_{len(provenance)}"] = rel
        for video in bounded_paths(self.root, (".mp4",)):
            rel = relative(self.root, video)
            row = by_id.get(video.stem, {})
            assets = {"video": rel, **provenance}
            if metadata.is_file():
                assets["metadata_table"] = relative(self.root, metadata)
            yield Sample(sample_id=stable_id(self.dataset, rel), dataset=self.dataset, game=self.game,
                         assets=assets, modalities=["video"],
                         native_clock={"unit": "frames", "fps": 15, "origin": "video", "timing": "encoded_pts"},
                         source={"url": row.get("source_url", row.get("url")), "channel": row.get("channel")},
                         lineage={"source_video_id": video.stem},
                         metadata={"source_metadata": row, "audio": False, "action_labels": False,
                                   "provenance_asset_keys": list(provenance),
                                   "license_notice": "youtube-creator-content"})

    def read_window(self, sample, range):
        provenance = {}
        returned_rows = returned_bytes = 0
        source_id = sample.lineage.get("source_video_id")
        for key in sample.metadata.get("provenance_asset_keys", []):
            matches = []
            for row in csv_rows(self.asset(sample, key), "\t"):
                if not source_matches(row, source_id, sample.source.get("url")):
                    continue
                returned_rows += 1
                returned_bytes += sum(len(str(k).encode("utf-8")) + len(str(v).encode("utf-8")) + 128
                                      for k, v in row.items())
                if returned_rows > MAX_PROVENANCE_ROWS or returned_bytes > MAX_PROVENANCE_BYTES:
                    raise ValueError("Terraria matched provenance exceeds aggregate return budget")
                matches.append(row)
            provenance[key] = {"asset": sample.assets[key], "rows": matches,
                               "join": "explicit source ID/URL or media filename; no semantic inference"}
        return Window(sample_id=sample.sample_id,
                      frames=decode_video(self.asset(sample, "video"), range, native_fps=15),
                      metadata={"audio_available": False, "actions_available": False,
                                "source_video_id": source_id, "filtering_provenance": provenance,
                                "provenance_semantics": "Publisher table fields retained as data; no filtering models executed"})
