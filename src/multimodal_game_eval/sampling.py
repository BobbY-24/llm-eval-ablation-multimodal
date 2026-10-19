"""Explicit model-facing selection; native decoded data is never changed in place."""
from __future__ import annotations

from typing import Any

from .core import DatasetError


def select_frames(frames: list[dict[str, Any]], max_frames: int = 64,
                  policy: str = "preserve", required_indices: list[int] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select existing native frames and return an auditable transformation record.

    ``preserve`` refuses an over-budget window. ``uniform`` subsamples by ordinal
    position without interpolating time or pixels. Required native indices are
    always retained, and must fit the budget. Expert review must establish that
    the selected evidence can identify the target mechanic.
    """
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or not 1 <= max_frames <= 4096:
        raise DatasetError("max_frames must be an integer from 1 to 4096")
    if policy not in {"preserve", "uniform"}:
        raise DatasetError("Unknown frame selection policy")
    indices = [frame.get("index") for frame in frames]
    if len(indices) != len(set(indices)) or any(not isinstance(i, int) or isinstance(i, bool) for i in indices):
        raise DatasetError("Native frame indices must be unique integers")
    required = set(required_indices or [])
    if not required <= set(indices) or len(required) > max_frames:
        raise DatasetError("Required native indices are missing or exceed budget")
    selected_positions = set(range(len(frames)))
    if len(frames) > max_frames:
        if policy == "preserve":
            raise DatasetError("Native window exceeds frame budget; explicitly choose a selection policy or shorter window")
        selected_positions = {i for i, frame in enumerate(frames) if frame["index"] in required}
        # Build a deterministic full ordering, then retain evenly spaced native
        # positions until the budget is filled. Never silently drop required frames.
        candidates = ([0] if max_frames == 1 else
                      [round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)])
        for position in candidates + list(range(len(frames))):
            if len(selected_positions) == max_frames:
                break
            selected_positions.add(position)
    selected = [frame for position, frame in enumerate(frames) if position in selected_positions]
    transform = {"operation": "native_frame_selection", "policy": policy, "input_frames": len(frames),
                 "output_frames": len(selected), "max_frames": max_frames, "required_indices": sorted(required),
                 "selected_native_indices": [frame["index"] for frame in selected],
                 "selected_timestamps": [frame.get("timestamp") for frame in selected],
                 "pixel_transform": "none", "temporal_interpolation": False,
                 "term_identifiability": "requires_review"}
    return selected, transform
