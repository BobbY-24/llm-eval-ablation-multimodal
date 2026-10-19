"""Local-only adapter utilities. No acquisition or model imports belong here."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import zipfile

from ..core import DatasetAdapter, Finding, Probe, safe_path, read_json, read_jsonl, check_archive

MAX_FILES = 100_000
MAX_RECORDS = 100_000
MAX_TEXT_BYTES = 32 * 1024 * 1024
MAX_ARRAY_BYTES = 512 * 1024 * 1024
MAX_WINDOW_ITEMS = 4096
MAX_PROBE_SAMPLES = 1000
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".zip")


def bounded_paths(root, suffixes):
    """Walk without following directory links; reject links escaping root."""
    root = Path(root).resolve()
    count = 0
    if not root.is_dir():
        return
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not Path(directory, d).is_symlink())
        for name in sorted(names):
            count += 1
            if count > MAX_FILES:
                raise ValueError(f"Inventory exceeds {MAX_FILES} files")
            path = Path(directory, name)
            if name.lower().endswith(tuple(suffixes)):
                # Resolve and check even if a file is a symlink.
                safe_path(root, str(path.relative_to(root)))
                yield path


def relative(root, path):
    return str(Path(path).relative_to(Path(root)))


def text_file(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Symlink metadata is unsupported: {path.name}")
    if path.stat().st_size > MAX_TEXT_BYTES:
        raise ValueError(f"Metadata exceeds {MAX_TEXT_BYTES} bytes: {path.name}")
    return path


def json_rows(path):
    for i, row in enumerate(read_jsonl(text_file(path))):
        if i >= MAX_RECORDS:
            raise ValueError(f"JSONL exceeds {MAX_RECORDS} records")
        if not isinstance(row, dict):
            raise ValueError("JSONL records must be objects")
        yield row


def csv_rows(path, delimiter=","):
    with text_file(path).open(encoding="utf-8", newline="") as stream:
        for i, row in enumerate(csv.DictReader(stream, delimiter=delimiter)):
            if i >= MAX_RECORDS:
                raise ValueError(f"Table exceeds {MAX_RECORDS} records")
            yield row


def json_object(path):
    row = read_json(text_file(path))
    if not isinstance(row, dict):
        raise ValueError("Expected a JSON object")
    return row


def selected_indices(interval, count, fps=None):
    if not math.isfinite(interval.start) or interval.start < 0:
        raise ValueError("Range start must be finite and nonnegative")
    if interval.end is not None and (not math.isfinite(interval.end) or interval.end < interval.start):
        raise ValueError("Range end must be finite and at least start")
    if interval.unit == "seconds":
        if fps is None or fps <= 0:
            raise ValueError("Seconds are unavailable without an established timebase")
        scale = fps
    elif interval.unit in {"frames", "steps", "ticks", "items", "windows"}:
        scale = 1
    else:
        raise ValueError(f"Unsupported range unit: {interval.unit}")
    start = min(count, math.ceil(interval.start * scale))
    end = min(count, math.ceil(interval.end * scale)) if interval.end is not None else min(count, start + MAX_WINDOW_ITEMS)
    if end - start > MAX_WINDOW_ITEMS:
        raise ValueError(f"Requested window exceeds {MAX_WINDOW_ITEMS} items")
    return range(start, end)


def numpy_zip_guard(path):
    """NPZ is a ZIP; check uncompressed sizes before NumPy allocation."""
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_RECORDS or sum(info.file_size for info in infos) > MAX_ARRAY_BYTES:
            raise ValueError("NPZ exceeds resource bounds")
        for info in infos:
            name = Path(info.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
                raise ValueError("Unsafe NPZ member path")


class LocalAdapter(DatasetAdapter):
    dataset = "local"
    game = "unknown"
    declared_modalities = []
    empty_note = "No supported local assets found; this adapter never downloads data."

    def __init__(self, root):
        self.root = Path(root).resolve()

    def asset(self, sample, name):
        return safe_path(self.root, sample.assets[name])

    def probe(self):
        try:
            count = 0
            limited = False
            for sample in self.iter_samples():
                count += 1
                if count >= MAX_PROBE_SAMPLES:
                    limited = True
                    break
            archives = [] if count else list(bounded_paths(self.root, ARCHIVE_SUFFIXES))
            status = "partial_available" if limited else "available" if count else "archives_only" if archives else "missing"
            notes = [] if count else [self.empty_note]
            if limited:
                notes.append(f"Probe stopped at {MAX_PROBE_SAMPLES} samples; sample_count is a lower bound.")
            if archives:
                for archive in archives[:16]:
                    issues = check_archive(archive)
                    if issues:
                        status = "invalid"
                        notes.extend(f"{archive.name}: {issue.code}: {issue.message}" for issue in issues[:8])
                notes.append("Archive shards detected; no extraction performed. Header checks cover at most 16 shards.")
            return Probe(dataset=self.dataset, root=str(self.root), status=status,
                         sample_count=count, modalities=list(self.declared_modalities), notes=notes)
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
            return Probe(dataset=self.dataset, root=str(self.root), status="invalid", sample_count=0,
                         modalities=list(self.declared_modalities), notes=[str(error)])

    def validate(self, sample):
        findings = []
        for name, path in sample.assets.items():
            try:
                asset = safe_path(self.root, path)
                if not asset.is_file():
                    findings.append(Finding("missing_asset", f"Missing {name}: {path}", sample_id=sample.sample_id))
            except (OSError, ValueError) as error:
                findings.append(Finding("unsafe_asset", str(error), sample_id=sample.sample_id))
        return findings
