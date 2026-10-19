"""Common records and bounded readers. This module never downloads or executes data."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import tarfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_JSONL_BYTES = 256 * 1024 * 1024
MAX_JSONL_LINE_BYTES = 4 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 100_000
MAX_ARCHIVE_MEMBER_BYTES = 8 * 1024**3
MAX_ARCHIVE_TOTAL_BYTES = 64 * 1024**3


class DatasetError(ValueError):
    """Invalid, unsupported or unsafe data; no network fallback is attempted."""


class MissingDependency(DatasetError):
    pass


@dataclass
class TimeRange:
    start: float = 0
    end: float | None = None
    unit: str = "seconds"

    def __post_init__(self) -> None:
        if self.unit not in {"seconds", "frames"}:
            raise DatasetError("TimeRange.unit must be seconds or frames")
        if not math.isfinite(self.start):
            raise DatasetError("Range start must be finite")
        if self.end is not None and (not math.isfinite(self.end) or self.end <= self.start):
            raise DatasetError("Range end must be finite and greater than start")
        if self.unit == "frames" and (self.start != int(self.start) or (self.end is not None and self.end != int(self.end))):
            raise DatasetError("Frame ranges require integer indices")


@dataclass
class Sample:
    sample_id: str
    dataset: str
    game: str
    assets: dict[str, str]
    modalities: list[str]
    native_clock: dict[str, Any]
    source: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    version: str | None = None
    dataset_revision: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    hashes: dict[str, str | None] = field(default_factory=dict)


@dataclass
class Window:
    sample_id: str
    frames: list[dict[str, Any]] | None = None
    actions: list[dict[str, Any]] | None = None
    states: list[dict[str, Any]] | None = None
    annotations: list[dict[str, Any]] | None = None
    spatial: Any = None
    audio: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Probe:
    dataset: str
    root: str
    status: str
    sample_count: int
    modalities: list[str]
    notes: list[str] = field(default_factory=list)


@dataclass
class Finding:
    code: str
    message: str
    severity: str = "error"
    sample_id: str | None = None


class DatasetAdapter:
    """Local-only dataset reader protocol; ranges are end-exclusive."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()

    def probe(self) -> Probe:
        raise NotImplementedError

    def iter_samples(self) -> Iterator[Sample]:
        raise NotImplementedError

    def read_window(self, sample: Sample, range: TimeRange) -> Window:
        raise NotImplementedError

    def validate(self, sample: Sample) -> list[Finding]:
        raise NotImplementedError


def stable_id(*parts: Any) -> str:
    encoded = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def safe_path(root: Path | str, relative: Path | str) -> Path:
    """Resolve a reference, including symlinks, and require it to remain under root."""
    root = Path(root).resolve()
    raw = str(relative)
    if "\x00" in raw or "\\" in raw or re.match(r"^[A-Za-z]:", raw):
        raise DatasetError("Unsafe asset path")
    target = (root / Path(relative)).resolve()
    if not target.is_relative_to(root):
        raise DatasetError("Asset reference escapes dataset root")
    return target


def sha256_file(path: Path | str, max_bytes: int | None = None) -> str:
    path = Path(path)
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise DatasetError(f"Asset exceeds hash byte limit: {path.name}")
    digest = hashlib.sha256()
    consumed = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            consumed += len(chunk)
            if max_bytes is not None and consumed > max_bytes:
                raise DatasetError("Asset grew past hash byte limit")
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise DatasetError(f"Non-finite JSON number: {value}")


def read_json(path: Path | str, max_bytes: int = MAX_JSON_BYTES) -> Any:
    with Path(path).open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise DatasetError("JSON exceeds configured byte limit")
    return json.loads(raw, parse_constant=_reject_constant)


def read_jsonl(path: Path | str, max_bytes: int = MAX_JSONL_BYTES) -> Iterator[dict[str, Any]]:
    consumed = 0
    with Path(path).open("rb") as stream:
        while raw := stream.readline(MAX_JSONL_LINE_BYTES + 1):
            consumed += len(raw)
            if len(raw) > MAX_JSONL_LINE_BYTES or consumed > max_bytes:
                raise DatasetError("JSONL exceeds configured line/file byte limit")
            if not raw.strip():
                continue
            row = json.loads(raw, parse_constant=_reject_constant)
            if not isinstance(row, dict):
                raise DatasetError("JSONL records must be objects")
            yield row


def require_dependency(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependency(f"Optional reader dependency '{module}' is unavailable; see README extras") from exc


def _archive_name_safe(name: str) -> bool:
    return bool(name) and not name.startswith("/") and "\\" not in name and "\x00" not in name and not re.match(r"^[A-Za-z]:", name) and ".." not in PurePosixPath(name).parts


def check_archive(path: Path | str) -> list[Finding]:
    """Check headers without extraction; reject links, traversal and excessive expansion."""
    path = Path(path)
    findings: list[Finding] = []
    count = total = 0

    def inspect(name: str, size: int, link: bool) -> None:
        nonlocal count, total
        count += 1
        total += size
        if not _archive_name_safe(name) or link:
            findings.append(Finding("unsafe_archive_member", f"Unsafe archive entry: {name[:200]}"))
        if size < 0 or size > MAX_ARCHIVE_MEMBER_BYTES or total > MAX_ARCHIVE_TOTAL_BYTES or count > MAX_ARCHIVE_MEMBERS:
            raise DatasetError("Archive exceeds bounded member/expansion limits")

    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for entry in archive.infolist():
                    inspect(entry.filename, entry.file_size, ((entry.external_attr >> 16) & 0o170000) == 0o120000)
                    if entry.flag_bits & 1:
                        findings.append(Finding("encrypted_archive", "Encrypted archives are unsupported"))
        else:
            with tarfile.open(path, "r|*") as archive:
                for entry in archive:
                    inspect(entry.name, entry.size, not (entry.isfile() or entry.isdir()))
        return findings
    except (tarfile.TarError, zipfile.BadZipFile, OSError, DatasetError) as exc:
        return findings + [Finding("invalid_archive", str(exc))]


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
