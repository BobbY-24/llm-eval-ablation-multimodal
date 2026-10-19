"""Lazy CPU media decoding with native timestamps and finite decode limits."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .core import DatasetError, TimeRange, require_dependency

MAX_DECODED_FRAMES = 4096
MAX_SCANNED_FRAMES = 2_000_000
MAX_FRAME_PIXELS = 16_777_216
MAX_WINDOW_BYTES = 512 * 1024**2


def decode_image(path: Path | str) -> Any:
    image_module = require_dependency("PIL.Image")
    np = require_dependency("numpy")
    with image_module.open(path) as image:
        if image.width * image.height > MAX_FRAME_PIXELS:
            raise DatasetError("Image exceeds pixel limit")
        return np.asarray(image.convert("RGB")).copy()


def decode_video(path: Path | str, range: TimeRange, native_fps: float | None = None, leading_padding: int = 0) -> list[dict[str, Any]]:
    """Read an end-exclusive interval. PTS is retained independently of nominal time.

    Native frame index zero follows explicitly identified codec padding. No idle
    frames are discarded, cursor painted, or implicit resampling performed.
    """
    av = require_dependency("av")
    if leading_padding < 0:
        raise DatasetError("Negative leading padding")
    if native_fps is not None and (not math.isfinite(native_fps) or native_fps <= 0):
        raise DatasetError("Invalid native FPS")
    result: list[dict[str, Any]] = []
    decoded_bytes = 0
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise DatasetError("Asset has no video stream")
        stream = container.streams.video[0]
        width, height = getattr(stream, "width", 0), getattr(stream, "height", 0)
        if width and height and width * height > MAX_FRAME_PIXELS:
            raise DatasetError("Video stream exceeds pixel limit")
        inferred_fps = float(stream.average_rate) if stream.average_rate else None
        fps = native_fps or inferred_fps
        base_pts: float | None = None
        for raw_index, frame in enumerate(container.decode(stream)):
            if raw_index >= MAX_SCANNED_FRAMES:
                raise DatasetError("Video scan exceeds frame limit; provide pre-cut local assets")
            if raw_index < leading_padding:
                continue
            index = raw_index - leading_padding
            pts_seconds = float(frame.pts * frame.time_base) if frame.pts is not None and frame.time_base else None
            if base_pts is None and pts_seconds is not None:
                base_pts = pts_seconds
            timestamp = pts_seconds - base_pts if pts_seconds is not None and base_pts is not None else (index / fps if fps else None)
            selector = float(index) if range.unit == "frames" else timestamp
            if selector is None:
                raise DatasetError("Video lacks timestamps and a usable frame rate")
            if selector < range.start:
                continue
            if range.end is not None and selector >= range.end:
                break
            decoded_bytes += frame.width * frame.height * 3
            if frame.width * frame.height > MAX_FRAME_PIXELS or len(result) >= MAX_DECODED_FRAMES or decoded_bytes > MAX_WINDOW_BYTES:
                raise DatasetError("Window exceeds decode bounds; request a smaller window")
            result.append({"index": index, "timestamp": timestamp, "pts": frame.pts, "pts_seconds": pts_seconds, "time_base": str(frame.time_base), "timestamp_basis": "pts" if pts_seconds is not None else "nominal_fps", "rgb": frame.to_ndarray(format="rgb24")})
    return result
