"""Bounded local Slippi telemetry and explicitly linked local rendered video.

Parser contract: peppi-py 0.8.6 ``read_slippi(path, skip_frames=False)`` returns
``game.frames.id`` and ``game.frames.ports[i].{leader,follower}.{pre,post}``
Arrow arrays.  The adapter resolves observed rollback revisions itself; it does
not assume that peppi has discarded them.  Finalization is read from the raw
Slippi frame-bookend watermark, independently of peppi's Python schema.

Sources: https://peppi-py.readthedocs.io/en/latest/source/peppi_py.html
https://github.com/project-slippi/slippi-wiki/blob/master/SPEC.md
"""

from __future__ import annotations

from contextlib import contextmanager
import gzip
import hashlib
from importlib import metadata as package_metadata
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import tarfile
import tempfile
from typing import Any, BinaryIO, Iterator

from ..core import DatasetAdapter, DatasetError, Finding, MissingDependency, Probe, Sample, TimeRange, Window
from ..core import require_dependency, safe_path, stable_id
from ..media import decode_video
from ._common import json_rows


_PREFIX = b"{U\x03raw[$U#l"
_DATASET = "slippi"
_SOURCE = "https://huggingface.co/datasets/erickfm/melee-ranked-replays"
_PARSER_VERSION = "0.8.6"
_RANK = re.compile(r"^(diamond|master|platinum)-(diamond|master|platinum)-")


class _Budget:
    def __init__(self, limit: int):
        self.remaining = limit

    def charge(self, size: int) -> None:
        self.remaining -= size
        if self.remaining < 0:
            raise ValueError("Slippi scan byte limit exceeded; inventory is incomplete")


class _BudgetReader:
    """Cap decompressed TAR bytes, including skipped members and TAR metadata."""

    def __init__(self, stream: BinaryIO, budget: _Budget, max_read: int):
        self.stream, self.budget, self.max_read = stream, budget, max_read

    def read(self, size: int = -1) -> bytes:
        if size < 0 or size > self.max_read:
            raise ValueError("Unbounded or oversized archive read rejected")
        data = self.stream.read(min(size, self.budget.remaining + 1))
        self.budget.charge(len(data))
        return data


def _member_name(name: str) -> str:
    # TAR paths are POSIX even on Windows. Reject Windows spellings as well.
    path = PurePosixPath(name)
    if (not name or "\x00" in name or "\\" in name or path.is_absolute()
            or ".." in path.parts or re.match(r"^[A-Za-z]:", name)):
        raise ValueError(f"Unsafe TAR member path: {name!r}")
    return path.as_posix()


def _scalar(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite value in replay telemetry")
    return value


def _field(value: Any, name: str, index: int) -> Any:
    for part in name.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return _scalar(value[index])


def _replay_info(data: bytes, max_rows: int) -> dict[str, Any]:
    """Inspect the documented raw event envelope without executing a parser."""
    if not data.startswith(_PREFIX) or len(data) < 17:
        raise ValueError("Invalid or truncated Slippi UBJSON raw header")
    raw_size = struct.unpack_from(">i", data, len(_PREFIX))[0]
    raw_start = len(_PREFIX) + 4
    if raw_size <= 0 or raw_start + raw_size > len(data):
        raise ValueError("Incomplete or invalid Slippi raw byte count")
    raw = memoryview(data)[raw_start:raw_start + raw_size]
    if len(raw) < 2 or raw[0] != 0x35 or raw[1] < 1 or (raw[1] - 1) % 3:
        raise ValueError("Invalid Slippi event payload table")
    cursor = 1 + raw[1]
    if cursor > len(raw):
        raise ValueError("Truncated Slippi event payload table")
    sizes = {raw[i]: struct.unpack_from(">H", raw, i + 1)[0]
             for i in range(2, cursor, 3)}
    version = None
    watermark = None
    game_end = False
    event_count = 0
    frame_rows = 0
    previous_frame = None
    while cursor < len(raw):
        code = raw[cursor]
        size = sizes.get(code)
        if size is None or cursor + size + 1 > len(raw):
            raise ValueError("Unknown or truncated Slippi raw event")
        event_count += 1
        if event_count > max_rows * 32:
            raise ValueError("Slippi raw event count exceeds resource limit")
        if code == 0x36 and size >= 4:
            version = ".".join(str(int(v)) for v in raw[cursor + 1:cursor + 4])
        if code == 0x38 and size >= 6:
            frame = struct.unpack_from(">i", raw, cursor + 1)[0]
            if frame != previous_frame:
                frame_rows += 1
                previous_frame = frame
                if frame_rows > max_rows:
                    raise ValueError("Slippi frame revision count exceeds resource limit")
        if code == 0x3C and size >= 8:
            finalized = struct.unpack_from(">i", raw, cursor + 5)[0]
            watermark = finalized if watermark is None else max(watermark, finalized)
        game_end = game_end or code == 0x39
        cursor += size + 1
    return {"slippi_version": version, "finalized_through_frame": watermark,
            "has_game_end": game_end, "raw_event_count": event_count}


class SlippiAdapter(DatasetAdapter):
    """Discover .slp files and TAR/TAR.GZ/TGZ shards with content deduplication.

    Bounds are explicit constructor options. Defaults permit a small evaluation
    subset; increasing them is a caller decision. ``probe`` and ``validate`` do
    not import peppi. Archived replays use a private temporary file only during
    ``read_window`` because peppi requires a filesystem path. No TAR extraction
    function is used. Cached discovery represents an inventory snapshot; reads
    verify its SHA-256 to reject files changed since discovery.
    """

    dataset = _DATASET

    def __init__(self, root: str | Path, *, max_files: int = 100_000,
                 max_asset_bytes: int = 64 * 1024 * 1024,
                 max_scan_bytes: int = 4 * 1024 * 1024 * 1024,
                 max_frame_rows: int = 100_000):
        self.root = Path(root).expanduser().resolve()
        for limit in (max_files, max_asset_bytes, max_scan_bytes, max_frame_rows):
            if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                raise ValueError("Slippi resource limits must be positive integers")
        self.max_files = max_files
        self.max_asset_bytes = max_asset_bytes
        self.max_scan_bytes = max_scan_bytes
        self.max_frame_rows = max_frame_rows
        self._samples: list[Sample] | None = None
        self._findings: list[Finding] = []

    def _paths(self) -> Iterator[Path]:
        count = 0
        stack = [self.root]
        while stack:
            folder = stack.pop()
            entries = []
            with os.scandir(folder) as scan:
                for entry in scan:
                    count += 1
                    if count > self.max_files:
                        raise ValueError("Slippi filesystem entry limit exceeded")
                    entries.append(entry)
            for entry in sorted(entries, key=lambda item: item.name):
                if entry.is_symlink():
                    self._findings.append(Finding("slippi_symlink", f"Skipped symlink: {entry.name}", "warning"))
                elif entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    lower = entry.name.lower()
                    if lower.endswith((".slp", ".tar", ".tar.gz", ".tgz")):
                        yield safe_path(self.root, str(Path(entry.path).relative_to(self.root)))

    def _read_bytes(self, stream: BinaryIO, size: int) -> bytes:
        if size <= 0 or size > self.max_asset_bytes:
            raise ValueError("Replay size is empty or exceeds max_asset_bytes")
        data = stream.read(size + 1)
        if len(data) != size:
            raise ValueError("Replay changed size or is truncated")
        return data

    @contextmanager
    def _archive(self, path: Path, budget: _Budget):
        with path.open("rb") as source:
            compressed = path.name.lower().endswith((".tar.gz", ".tgz"))
            with (gzip.GzipFile(fileobj=source) if compressed else _Passthrough(source)) as stream:
                reader = _BudgetReader(stream, budget, self.max_asset_bytes + 1)
                with tarfile.open(fileobj=reader, mode="r|") as archive:
                    yield archive

    def _scan(self) -> None:
        if self._samples is not None:
            return
        self._samples = []
        if not self.root.is_dir():
            self._findings.append(Finding("slippi_root", "Dataset root must be an existing directory"))
            return
        by_hash: dict[str, Sample] = {}
        budget = _Budget(self.max_scan_bytes)
        member_count = 0

        def add(data: bytes, assets: dict[str, Any], label: str, archive_member: str | None = None) -> None:
            info = _replay_info(data, self.max_frame_rows)
            digest = hashlib.sha256(data).hexdigest()
            if digest in by_hash:
                by_hash[digest].lineage["duplicates"].append(label)
                return
            match = _RANK.match(PurePosixPath(label).name)
            sample = Sample(
                sample_id=stable_id(_DATASET, digest), dataset=_DATASET,
                game="Super Smash Bros. Melee", assets=assets,
                modalities=["actions", "states"],
                native_clock={"unit": "frames", "nominal_hz": 60,
                              "origin": "frame 0 is match timer start; pregame IDs are negative"},
                source={"url": _SOURCE, "kind": "raw_slippi_replay"},
                lineage={"duplicates": []}, dataset_revision=info["slippi_version"],
                metadata={**info, "bytes": len(data),
                          "archive_member": archive_member,
                          "rank_pair": "-".join(match.groups()) if match else None,
                          "rgb_available": False},
                hashes={"replay_sha256": digest},
            )
            by_hash[digest] = sample
            if not info["has_game_end"]:
                self._findings.append(Finding("slippi_game_end", "Replay lacks a Game End event", "warning", sample.sample_id))

        try:
            for path in self._paths():
                relative = path.relative_to(self.root).as_posix()
                try:
                    if path.suffix.lower() == ".slp":
                        size = path.stat().st_size
                        if size > self.max_asset_bytes:
                            raise ValueError("Replay exceeds max_asset_bytes")
                        budget.charge(size)
                        with path.open("rb") as stream:
                            add(self._read_bytes(stream, size), {"replay": relative}, relative)
                    else:
                        with self._archive(path, budget) as archive:
                            for member in archive:
                                member_count += 1
                                if member_count > self.max_files:
                                    raise ValueError("Slippi archive member limit exceeded")
                                name = _member_name(member.name)
                                if member.isdir():
                                    continue
                                if not member.isfile() or member.issparse():
                                    raise ValueError(f"Non-regular TAR member rejected: {name}")
                                if not name.lower().endswith(".slp"):
                                    continue
                                stream = archive.extractfile(member)
                                if stream is None:
                                    raise ValueError(f"Cannot read TAR member: {name}")
                                with stream:
                                    data = self._read_bytes(stream, member.size)
                                add(data, {"archive": relative}, f"{relative}::{name}", archive_member=name)
                except (OSError, ValueError, EOFError, tarfile.TarError) as exc:
                    self._findings.append(Finding("slippi_asset", f"{relative}: {exc}"))
                    if budget.remaining < 0 or member_count > self.max_files:
                        break
        except (OSError, ValueError) as exc:
            self._findings.append(Finding("slippi_inventory", str(exc)))
        self._register_rendered(by_hash)
        self._samples = sorted(by_hash.values(), key=lambda item: item.sample_id)

    def _rendered_mapping(self, row: dict[str, Any]) -> dict[str, Any]:
        digest = row.get("replay_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("Rendered mapping requires a 64-digit replay_sha256")
        video = row.get("video_path")
        if not isinstance(video, str) or not video:
            raise ValueError("Rendered mapping requires a local video_path")
        path = safe_path(self.root, video)
        if not path.is_file():
            raise ValueError("Rendered mapping video_path is missing")
        origin = row.get("video_start_frame")
        if not isinstance(origin, int) or isinstance(origin, bool) or not -(2**31) <= origin < 2**31:
            raise ValueError("video_start_frame must be a native signed 32-bit frame ID")
        version = row.get("game_version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Rendered mapping requires an explicit game_version")
        fps = row.get("fps")
        if not isinstance(fps, (int, float)) or isinstance(fps, bool) or not math.isfinite(fps) or not 0 < fps <= 1000:
            raise ValueError("Rendered mapping requires finite positive fps <= 1000")
        frame_map = row.get("frame_map")
        if frame_map is None:
            if fps != 60:
                raise ValueError("Non-60-fps capture requires an explicit frame_map")
        else:
            if not isinstance(frame_map, list) or not frame_map or len(frame_map) > self.max_frame_rows:
                raise ValueError("frame_map must be a bounded nonempty list")
            previous_index, previous_native = -1, -(2**31)
            for entry in frame_map:
                if not isinstance(entry, dict):
                    raise ValueError("frame_map entries must be objects")
                index, native = entry.get("video_index"), entry.get("native_frame")
                if (not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 2_000_000
                        or not isinstance(native, int) or isinstance(native, bool) or not -(2**31) <= native < 2**31
                        or index <= previous_index or native < previous_native):
                    raise ValueError("frame_map must increase video indices and never decrease native frames")
                previous_index, previous_native = index, native
            if frame_map[0]["video_index"] != 0 or frame_map[0]["native_frame"] != origin:
                raise ValueError("frame_map index 0 must match video_start_frame")
        return {"replay_sha256": digest.lower(), "video_path": path.relative_to(self.root).as_posix(),
                "video_start_frame": origin, "game_version": version, "fps": fps,
                "frame_map": frame_map, "renderer": row.get("renderer"), "provenance": row.get("provenance")}

    def _register_rendered(self, by_hash: dict[str, Sample]) -> None:
        """Attach only unambiguous reviewed index mappings; never render a replay."""
        mappings, conflicted = {}, set()
        try:
            manifest = safe_path(self.root, "rendered_manifest.jsonl")
            if not manifest.is_file():
                return
            for line, row in enumerate(json_rows(manifest), 1):
                raw_digest = row.get("replay_sha256", "")
                digest = raw_digest.lower() if isinstance(raw_digest, str) else ""
                try:
                    mapping = self._rendered_mapping(row)
                except (OSError, ValueError, TypeError) as exc:
                    self._findings.append(Finding("slippi_rendered_mapping", f"Rendered manifest line {line}: {exc}"))
                    if digest in by_hash:
                        conflicted.add(digest)
                    continue
                digest = mapping["replay_sha256"]
                if digest not in by_hash:
                    self._findings.append(Finding("slippi_rendered_unmatched", f"Rendered manifest line {line} has no matching local replay hash", "warning"))
                    continue
                if digest in mappings:
                    if mappings[digest] == mapping:
                        self._findings.append(Finding("slippi_rendered_duplicate", f"Duplicate rendered mapping at line {line}", "warning", by_hash[digest].sample_id))
                    else:
                        conflicted.add(digest)
                        self._findings.append(Finding("slippi_rendered_conflict", f"Conflicting rendered mappings for replay at line {line}; RGB attachment disabled", "error", by_hash[digest].sample_id))
                else:
                    mappings[digest] = mapping
        except (OSError, ValueError, TypeError) as exc:
            self._findings.append(Finding("slippi_rendered_manifest", str(exc)))
            return  # An incomplete or malformed manifest cannot register assets.
        for digest, mapping in mappings.items():
            if digest in conflicted:
                continue
            sample = by_hash[digest]
            sample.assets.update(video=mapping["video_path"], rendered_manifest="rendered_manifest.jsonl")
            sample.modalities.append("video")
            sample.version = mapping["game_version"]
            sample.metadata.update(rendered_mapping=mapping, rgb_available=True)

    def probe(self) -> Probe:
        self._scan()
        status = "replay_only" if self._samples else "missing"
        mapped = sum("video" in s.assets for s in self._samples or [])
        if mapped:
            status = "available" if mapped == len(self._samples) else "partial_available"
        if any(f.severity == "error" for f in self._findings):
            status = "partial_available" if self._samples else "invalid"
        return Probe(_DATASET, str(self.root), status, len(self._samples or []),
                     ["actions", "states"] + (["video"] if mapped else []),
                     [f"{mapped} replay(s) linked to existing RGB through rendered_manifest.jsonl; no rendering performed.",
                      "Inventory is bounded and content-deduplicated; peppi is loaded only on read."]
                     + [f"{f.code}: {f.message}" for f in self._findings[:16]])

    def iter_samples(self) -> Iterator[Sample]:
        self._scan()
        yield from self._samples or []

    def validate(self, sample: Sample | None = None) -> list[Finding]:
        self._scan()
        findings = list(self._findings)
        if not self._samples:
            findings.append(Finding("slippi_empty", "No valid local .slp replays found"))
        if sample is not None:
            try:
                data = self._sample_bytes(sample)
                if hashlib.sha256(data).hexdigest() != sample.hashes.get("replay_sha256"):
                    raise ValueError("Replay SHA-256 differs from inventory")
                _replay_info(data, self.max_frame_rows)
                if "rendered_mapping" in sample.metadata:
                    mapping = self._rendered_mapping(sample.metadata["rendered_mapping"])
                    if mapping["replay_sha256"] != sample.hashes.get("replay_sha256") or mapping["video_path"] != sample.assets.get("video"):
                        raise ValueError("Rendered mapping hash/path differs from registered sample")
            except (OSError, ValueError, EOFError, tarfile.TarError) as exc:
                findings.append(Finding("slippi_sample", str(exc), "error", sample.sample_id))
        try:
            installed = package_metadata.version("peppi-py")
        except package_metadata.PackageNotFoundError:
            findings.append(Finding("slippi_parser_missing", "Window reads require optional peppi-py==0.8.6", "warning"))
        else:
            if installed != _PARSER_VERSION:
                findings.append(Finding("slippi_parser_version", f"Expected peppi-py==0.8.6; found {installed}", "warning"))
        return findings

    def _sample_bytes(self, sample: Sample) -> bytes:
        if sample.dataset != _DATASET:
            raise ValueError("Sample does not belong to the Slippi adapter")
        if "replay" in sample.assets:
            path = safe_path(self.root, sample.assets["replay"])
            with path.open("rb") as stream:
                return self._read_bytes(stream, path.stat().st_size)
        path = safe_path(self.root, sample.assets["archive"])
        target = _member_name(sample.metadata["archive_member"])
        with self._archive(path, _Budget(self.max_scan_bytes)) as archive:
            for index, member in enumerate(archive):
                if index >= self.max_files:
                    raise ValueError("Slippi archive member limit exceeded")
                name = _member_name(member.name)
                if not member.isdir() and (not member.isfile() or member.issparse()):
                    raise ValueError(f"Non-regular TAR member rejected: {name}")
                if name == target:
                    if not member.isfile():
                        raise ValueError("Requested replay is not a regular TAR member")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError("Cannot read requested replay member")
                    with stream:
                        return self._read_bytes(stream, member.size)
        raise FileNotFoundError(f"Replay member not found: {target}")

    def read_window(self, sample: Sample, range: TimeRange) -> Window:
        try:
            return self._read_window(sample, range)
        except MissingDependency:
            raise
        except (OSError, ValueError, EOFError, tarfile.TarError, KeyError,
                AttributeError, IndexError, TypeError, struct.error) as exc:
            raise DatasetError(f"Cannot read Slippi window: {exc}") from exc

    def _read_window(self, sample: Sample, range: TimeRange) -> Window:
        start, end = float(range.start), None if range.end is None else float(range.end)
        if (not math.isfinite(start) or (end is not None and not math.isfinite(end))
                or (end is not None and end < start)):
            raise ValueError("Window bounds must be finite and end must be >= start")
        if range.unit not in ("seconds", "frames"):
            raise ValueError("Slippi windows support seconds or native frames")
        data = self._sample_bytes(sample)
        expected = sample.hashes.get("replay_sha256")
        if not expected or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Replay SHA-256 differs from its inventory snapshot")
        info = _replay_info(data, self.max_frame_rows)
        try:
            version = package_metadata.version("peppi-py")
        except package_metadata.PackageNotFoundError as exc:
            raise MissingDependency("Install optional peppi-py==0.8.6 to read Slippi windows") from exc
        if version != _PARSER_VERSION:
            raise MissingDependency(f"Slippi windows require peppi-py==0.8.6; found {version}")
        parser = require_dependency("peppi_py")
        # Always parse a verified private snapshot, including standalone files.
        with tempfile.TemporaryDirectory(prefix="game-eval-slippi-") as temporary:
            replay_path = Path(temporary) / "replay.slp"
            replay_path.write_bytes(data)
            game = parser.read_slippi(str(replay_path), skip_frames=False)
        return self._window(game, sample, range.unit, start, end, info)

    def _window(self, game: Any, sample: Sample, unit: str, start: float,
                end: float | None, info: dict[str, Any]) -> Window:
        frames = game.frames
        if frames is None:
            raise ValueError("Parser returned no frame telemetry")
        if len(frames.id) > self.max_frame_rows:
            raise ValueError("Parsed frame count exceeds resource limit")
        timeline: list[int] = []
        latest: dict[int, int] = {}
        for index in range(len(frames.id)):
            frame_id = int(_scalar(frames.id[index]))
            # A rollback invalidates the previously observed future, including
            # an unfinished tail. Equal IDs replace one revision, not a frame.
            while timeline and timeline[-1] >= frame_id:
                latest.pop(timeline.pop())
            timeline.append(frame_id)
            latest[frame_id] = index
        players = list(game.start.players)
        if len(players) != len(frames.ports) or len(players) > 4:
            raise ValueError("Parser player/port mapping is inconsistent")
        hz = 50 if getattr(game.start, "is_pal", False) else 60
        watermark = info["finalized_through_frame"]
        actions, states = [], []
        for frame_id in timeline:
            timestamp = frame_id / hz
            clock_value = timestamp if unit == "seconds" else frame_id
            if clock_value < start or (end is not None and clock_value >= end):
                continue
            if len(actions) >= 4096:
                raise ValueError("Slippi selected window exceeds 4096 frames; request a smaller interval")
            index = latest[frame_id]
            action_ports, state_ports = {}, {}
            for player, port in zip(players, frames.ports):
                key = str(int(player.port) + 1)
                action_units, state_units = {}, {}
                for role in ("leader", "follower"):
                    actor = getattr(port, role, None)
                    if actor is None:
                        continue
                    action_units[role] = {name: _field(actor.pre, name, index) for name in
                                          ("buttons", "buttons_physical", "joystick.x", "joystick.y",
                                           "cstick.x", "cstick.y", "triggers", "triggers_physical.l",
                                           "triggers_physical.r", "raw_analog_x", "raw_analog_y")}
                    state_units[role] = {name: _field(actor.post, name, index) for name in
                                         ("character", "state", "position.x", "position.y", "direction",
                                          "percent", "shield", "stocks", "state_age", "airborne",
                                          "ground", "jumps", "hurtbox_state",
                                          # Optional in older replays; peppi field names unverified.
                                          "l_cancel", "velocities.self_x_ground",
                                          "velocities.self_x_air", "velocities.self_y")}
                action_ports[key], state_ports[key] = action_units, state_units
            row = {"frame_id": frame_id, "timestamp": timestamp,
                   "finalized": None if watermark is None else frame_id <= watermark}
            actions.append({**row, "ports": action_ports})
            states.append({**row, "ports": state_ports})
        pixels, render_metadata = self._rendered_frames(sample, unit, start, end, hz, timeline)
        return Window(sample_id=sample.sample_id, frames=pixels, actions=actions, states=states,
                      metadata={"parser": f"peppi-py=={_PARSER_VERSION}", "hz": hz,
                                "window_unit": unit, "start": start, "end": end,
                                "interval": "[start, end)", **info,
                                "rollback_policy": "truncate on decreasing frame IDs; last observed revision",
                                "finalization_policy": "bookend watermark when present; otherwise unknown",
                                "rgb_available": pixels is not None, **render_metadata, "stage_id": int(game.start.stage),
                                "observed_rows": len(frames.id), "resolved_rows": len(timeline)})

    def _rendered_frames(self, sample: Sample, unit: str, start: float, end: float | None,
                         hz: int, timeline: list[int]) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
        mapping = sample.metadata.get("rendered_mapping")
        if mapping is None:
            return None, {}
        mapping = self._rendered_mapping(mapping)
        if mapping["replay_sha256"] != sample.hashes.get("replay_sha256") or mapping["video_path"] != sample.assets.get("video"):
            raise ValueError("Rendered mapping does not match verified replay hash/path")
        lower = math.ceil(start * hz) if unit == "seconds" else int(start)
        upper = (math.ceil(end * hz) if unit == "seconds" else int(end)) if end is not None else (timeline[-1] + 1 if timeline else lower)
        origin = mapping["video_start_frame"]
        explicit = mapping["frame_map"]
        metadata = {"rendered_mapping": mapping, "rgb_requested_native_interval": [lower, upper],
                    "rgb_index_space": "unique video capture index; frame_id is the native Slippi frame",
                    "rgb_alignment_basis": "user-supplied replay-hash and native-frame/index mapping; not inferred from lengths"}
        if explicit is None:
            if hz != 60:
                raise ValueError("PAL/non-60-Hz telemetry requires an explicit rendered frame_map")
            first, stop = max(0, lower - origin), max(0, upper - origin)
            if stop - first > 4096:
                raise ValueError("Slippi rendered window exceeds 4096 video frames")
            selected = {index: origin + index for index in range(first, stop)}
            metadata["rgb_mapping_covers_requested_interval"] = lower >= origin
        else:
            selected = {entry["video_index"]: entry["native_frame"] for entry in explicit
                        if lower <= entry["native_frame"] < upper}
            metadata["rgb_mapping_covers_requested_interval"] = None  # Explicit sparse mapping declares only its listed frames.
        if not selected:
            return [], metadata
        first, stop = min(selected), max(selected) + 1
        if stop - first > 4096:
            raise ValueError("Slippi explicit mapping spans more than 4096 decoded video frames")
        decoded = decode_video(safe_path(self.root, mapping["video_path"]), TimeRange(first, stop, "frames"), native_fps=mapping["fps"])
        result, observed = [], set()
        for frame in decoded:
            index = frame["index"]
            if index not in selected:
                continue
            native = selected[index]
            observed.add(index)
            result.append({**frame, "video_index": index, "video_timestamp": frame.get("timestamp"),
                           "index": index, "frame_id": native, "timestamp": native / hz,
                           "timestamp_basis": "native Slippi frame ID mapped from video index"})
        if observed != set(selected):
            raise ValueError("Rendered video lacks frame indices required by its manifest mapping")
        return result, metadata


class _Passthrough:
    def __init__(self, stream: BinaryIO):
        self.stream = stream

    def __enter__(self) -> BinaryIO:
        return self.stream

    def __exit__(self, *args: Any) -> None:
        return None
