"""Atomic, finite JSON exports. Existing outputs require explicit replacement."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .core import DatasetError, to_jsonable


def write_jsonl(path: Path | str, rows: Iterable[Any], force: bool = False) -> int:
    path = Path(path)
    if (path.exists() or path.is_symlink()) and not force:
        raise DatasetError(f"Output already exists: {path}. Choose a new output or --force.")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            for row in rows:
                stream.write(json.dumps(to_jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        if force:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise DatasetError("Output appeared during export; refusing replacement") from error
            Path(temporary).unlink()
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return count


def write_json(path: Path | str, value: Any, force: bool = False) -> None:
    # A one-record JSONL is also a valid standalone JSON document.
    write_jsonl(path, [value], force=force)


def new_output_directory(path: Path | str) -> Path:
    path = Path(path)
    if path.is_symlink() or path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise DatasetError(f"Output directory is not empty: {path}. Choose a fresh run directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path
