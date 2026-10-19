"""Deterministic Melee telemetry rules that propose review windows, never labels.

Each detector reads the Slippi adapter's ``Window.states``/``Window.actions``
rows (native frame IDs, rollback already resolved) and returns candidate
records with ``status: review_candidate_not_label``. Strata are sampling
proposals for annotation (``positive_candidate`` / ``near_miss_candidate``);
human review decides the label. Rule outputs also serve later as a
telemetry-rule reference arm, so annotators must not see a candidate's kind,
stratum or rule evidence while labelling.

Action-state IDs and button masks below are Melee common action states and
Slippi pre-frame physical-button bits as documented in the Slippi ecosystem
(project-slippi SPEC.md; slippi-js ``State`` enum). They were transcribed
offline and are UNVERIFIED against a real replay in this project: check a
replay's state histogram (``state_histogram``) before mining real data.
"""
from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Callable, Iterable

from .core import DatasetError, TimeRange, stable_id
from .sampling import select_frames

RULESET_VERSION = "melee-mining-v1"
STATUS = "review_candidate_not_label"
CONSTANTS_STATUS = "transcribed_offline_unverified_against_real_replay"

DEAD = frozenset(range(0, 11))
TURN, DASH, RUN = 18, 20, 21
KNEE_BEND = 24
GROUND_JUMP = frozenset({25, 26})
LANDING, LANDING_FALL_SPECIAL = 42, 43
ATTACK_AIR = frozenset(range(65, 70))
LANDING_AIR = frozenset(range(70, 75))
ESCAPE_AIR = 236
CLIFF = frozenset({252, 253})
BUTTON_Z, BUTTON_R, BUTTON_L = 0x0010, 0x0020, 0x0040
BUTTON_X, BUTTON_Y = 0x0400, 0x0800

FAMILIES = ("wavedash", "L-cancel", "dashdancing", "short hop", "edgehogging")

DEFAULT_PARAMS: dict[str, Any] = {
    "max_frames_jump_to_airdodge": 3,
    "max_airdodge_frames_before_landing": 8,
    "waveland_lookback_frames": 10,
    "min_slide_units": 1.0,
    "slide_measure_frames": 10,
    "l_cancel_window_frames": 7,
    "trigger_analog_threshold": 0.3,
    "l_cancel_flag_lookahead_frames": 3,
    "dash_max_segment_frames": 20,
    "dashdance_min_reversals": 2,
    "tap_jump_threshold": 0.66,
    "jump_max_air_frames": 90,
    "edgehog_min_overlap_frames": 5,
    "edgehog_merge_gap_frames": 2,
    "edgehog_outcome_frames": 180,
}
DEFAULT_PADDING = {"wavedash": (20, 20), "L-cancel": (12, 20), "dashdancing": (12, 12),
                   "short hop": (8, 12), "edgehogging": (30, 30)}


def _params(params: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_PARAMS)
    for key, value in (params or {}).items():
        if key not in DEFAULT_PARAMS:
            raise DatasetError(f"Unknown mining parameter: {key}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise DatasetError(f"Mining parameter must be a nonnegative number: {key}")
        merged[key] = value
    return merged


def port_series(window: dict[str, Any], port: str, role: str = "leader") -> list[dict[str, Any]]:
    """Merge one port's post-frame state with its pre-frame inputs, in frame order."""
    inputs = {row.get("frame_id"): row for row in window.get("actions") or []}
    series = []
    for row in window.get("states") or []:
        state = ((row.get("ports") or {}).get(port) or {}).get(role)
        if state is None:
            continue
        action = (((inputs.get(row.get("frame_id")) or {}).get("ports") or {}).get(port) or {}).get(role) or {}
        series.append({**state, "frame": row["frame_id"], "finalized": row.get("finalized"), "input": action})
    return series


def ports(window: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    for row in window.get("states") or []:
        found.update((row.get("ports") or {}).keys())
    return sorted(found)


def state_histogram(window: dict[str, Any]) -> dict[str, dict[int, int]]:
    """Per-port action-state counts, for checking transcribed constants on real data."""
    return {port: dict(sorted(Counter(row.get("state") for row in port_series(window, port)
                                      if isinstance(row.get("state"), int)).items()))
            for port in ports(window)}


class _Series:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.n = len(rows)

    def state(self, i: int) -> int | None:
        value = self.rows[i].get("state") if 0 <= i < self.n else None
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def frame(self, i: int) -> int:
        return self.rows[i]["frame"]

    def joined(self, i: int) -> bool:
        """True when row i directly follows row i-1 in native time."""
        return 0 < i < self.n and self.rows[i]["frame"] == self.rows[i - 1]["frame"] + 1

    def num(self, i: int, key: str) -> float | None:
        value = self.rows[i].get(key)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def run_end(self, i: int, predicate: Callable[[int | None], bool], limit: int | None = None) -> int:
        j = i + 1
        while j < self.n and self.joined(j) and predicate(self.state(j)) and (limit is None or j - i < limit):
            j += 1
        return j

    def entry(self, i: int, states: Iterable[int] | int) -> bool:
        wanted = {states} if isinstance(states, int) else set(states)
        return self.state(i) in wanted and not (self.joined(i) and self.state(i - 1) in wanted)

    def buttons(self, i: int) -> int | None:
        value = self.rows[i]["input"].get("buttons_physical")
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def finalized(self, start: int, end: int) -> bool | None:
        flags = [self.rows[i].get("finalized") for i in range(start, end)]
        return None if any(flag is None for flag in flags) else all(flags)


def _candidate(sample_id: str, family: str, stratum: str, kind: str, port: str, s: _Series,
               start: int, end: int, key: list[int], evidence: dict[str, Any], rule: str,
               params: dict[str, Any], used: tuple[str, ...], **extra: Any) -> dict[str, Any]:
    frames = [s.frame(i) for i in key]
    return {"candidate_id": "cand-" + stable_id(RULESET_VERSION, sample_id, family, kind, port, frames),
            "status": STATUS, "sample_id": sample_id, "family": family, "stratum": stratum,
            "kind": kind, "port": port, **extra,
            "event_frames": [s.frame(start), s.frame(end - 1) + 1], "key_frames": frames,
            # A run that reached the window edge never saw its terminating frame.
            "examined_through_frame": s.frame(min(max(end, key[-1] + 1), s.n - 1)),
            "finalized": s.finalized(start, end), "evidence": evidence,
            "rule": {"id": rule, "ruleset": RULESET_VERSION, "constants": CONSTANTS_STATUS,
                     "params": {name: params[name] for name in used}}}


def detect_wavedash(sample_id: str, port: str, rows: list[dict[str, Any]], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Jumpsquat → (≤N jump frames) → airdodge → special landing; near-misses keep the context."""
    p, s, out, used_airdodges = _params(params), _Series(rows), [], set()
    names = ("max_frames_jump_to_airdodge", "max_airdodge_frames_before_landing", "min_slide_units",
             "slide_measure_frames", "waveland_lookback_frames")
    for i in range(s.n):
        if not s.entry(i, KNEE_BEND):
            continue
        j = s.run_end(i, lambda v: v == KNEE_BEND)
        k = j
        while k < s.n and s.joined(k) and s.state(k) in GROUND_JUMP and k - j < p["max_frames_jump_to_airdodge"]:
            k += 1
        if not (k < s.n and s.joined(k) and s.state(k) == ESCAPE_AIR):
            continue
        used_airdodges.add(k)
        m = s.run_end(k, lambda v: v == ESCAPE_AIR)
        landed = m < s.n and s.joined(m) and s.state(m) == LANDING_FALL_SPECIAL and m - k <= p["max_airdodge_frames_before_landing"]
        evidence = {"jumpsquat_frames": j - i, "jump_frames_before_airdodge": k - j, "airdodge_frames": m - k}
        if landed:
            end = s.run_end(m, lambda v: v == LANDING_FALL_SPECIAL)
            measure = min(end, m + int(p["slide_measure_frames"])) - 1
            x0, x1, facing = s.num(m, "position.x"), s.num(measure, "position.x"), s.num(i, "direction")
            dx = None if x0 is None or x1 is None else round(x1 - x0, 4)
            slide = None if dx is None else ("none" if abs(dx) < p["min_slide_units"] else
                                             "unknown" if facing is None else
                                             "forward" if dx * facing > 0 else "backward")
            evidence.update(landing_frames=end - m, slide_dx=dx, slide_direction=slide)
            out.append(_candidate(sample_id, "wavedash", "positive_candidate", "wavedash", port, s, i, end,
                                  [i, k, m], evidence, "wavedash.v1", p, names))
        else:
            evidence.update(state_after_airdodge=s.state(m) if m < s.n else None)
            out.append(_candidate(sample_id, "wavedash", "near_miss_candidate", "jump_airdodge_no_landing", port,
                                  s, i, min(m + 1, s.n), [i, k], evidence, "wavedash.v1", p, names))
    for k in range(s.n):
        if k in used_airdodges or not s.entry(k, ESCAPE_AIR):
            continue
        lookback = range(max(0, k - int(p["waveland_lookback_frames"])), k)
        if any(s.state(i) == KNEE_BEND for i in lookback) or k - int(p["waveland_lookback_frames"]) < 0:
            continue  # a jumpsquat nearby, or too little history to exclude one
        m = s.run_end(k, lambda v: v == ESCAPE_AIR)
        if m < s.n and s.joined(m) and s.state(m) == LANDING_FALL_SPECIAL and m - k <= p["max_airdodge_frames_before_landing"]:
            end = s.run_end(m, lambda v: v == LANDING_FALL_SPECIAL)
            out.append(_candidate(sample_id, "wavedash", "near_miss_candidate", "waveland_without_jumpsquat", port,
                                  s, k, end, [k, m], {"airdodge_frames": m - k, "landing_frames": end - m,
                                                      "prior_state": s.state(k - 1)}, "wavedash.v1", p, names))
    return out


def detect_l_cancel(sample_id: str, port: str, rows: list[dict[str, Any]], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Aerial → aerial landing. Game L-cancel flag first; L/R/Z input window as a cross-check."""
    p, s, out = _params(params), _Series(rows), []
    names = ("l_cancel_window_frames", "trigger_analog_threshold", "l_cancel_flag_lookahead_frames")
    window = int(p["l_cancel_window_frames"])
    for i in range(1, s.n):
        if not (s.joined(i) and s.state(i - 1) in ATTACK_AIR and s.state(i) in LANDING_AIR | {LANDING}):
            continue
        start = max(0, i - window)
        if s.state(i) == LANDING:
            end = s.run_end(i, lambda v: v == LANDING)
            out.append(_candidate(sample_id, "L-cancel", "near_miss_candidate", "aerial_autocancel_landing", port,
                                  s, start, end, [i], {"landing_frames": end - i, "aerial_state": s.state(i - 1)},
                                  "l_cancel.v1", p, names))
            continue
        end = s.run_end(i, lambda v: v in LANDING_AIR)
        flag = None
        for f in range(i, min(s.n, i + int(p["l_cancel_flag_lookahead_frames"]))):
            value = s.rows[f].get("l_cancel")
            if isinstance(value, int) and value in (1, 2):
                flag = value
                break
        contiguous = all(s.joined(f) for f in range(start + 1, i + 1)) and i - start == window
        hit = None
        if contiguous:
            observed = []
            for f in range(start, i):
                buttons = s.buttons(f)
                left, right = (s.rows[f]["input"].get("triggers_physical." + side) for side in "lr")
                if buttons is None and left is None and right is None:
                    observed.append(None)
                    continue
                analog = any(isinstance(v, (int, float)) and v >= p["trigger_analog_threshold"] for v in (left, right))
                observed.append(bool(buttons and buttons & (BUTTON_L | BUTTON_R | BUTTON_Z)) or analog)
            hit = True if any(observed) else (None if None in observed else False)
        success = {1: True, 2: False}.get(flag, hit)
        if success is None:
            continue  # neither validator available: no proposal rather than a guess
        evidence = {"game_l_cancel_flag": flag, "input_window_hit": hit,
                    "validator": "game_l_cancel_flag" if flag else "input_window_only",
                    "sources_agree": None if flag is None or hit is None else (flag == 1) == hit,
                    "landing_lag_frames": end - i, "aerial_state": s.state(i - 1)}
        out.append(_candidate(sample_id, "L-cancel", "positive_candidate" if success else "near_miss_candidate",
                              "l_cancel_success" if success else "l_cancel_missed", port, s, start, end, [i],
                              evidence, "l_cancel.v1", p, names))
    return out


def detect_dashdance(sample_id: str, port: str, rows: list[dict[str, Any]], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Chains of dash entries; alternating directions are proposals, same-direction chains are foxtrots."""
    p, s, out = _params(params), _Series(rows), []
    names = ("dash_max_segment_frames", "dashdance_min_reversals")

    def dash_entry(i: int) -> bool:
        if s.state(i) != DASH:
            return False
        if not s.joined(i) or s.state(i - 1) != DASH:
            return True
        age, previous = s.num(i, "state_age"), s.num(i - 1, "state_age")
        return s.num(i, "direction") != s.num(i - 1, "direction") or (age is not None and previous is not None and age < previous)

    entries = [i for i in range(s.n) if dash_entry(i) and not s.rows[i].get("airborne")]
    chains: list[list[int]] = []
    for i in entries:
        if chains and i - chains[-1][-1] <= p["dash_max_segment_frames"] and all(
                s.joined(f) and s.state(f) in (DASH, TURN) for f in range(chains[-1][-1] + 1, i)):
            chains[-1].append(i)
        else:
            chains.append([i])
    for chain in chains:
        if len(chain) < 2:
            continue
        directions = [s.num(i, "direction") for i in chain]
        if None in directions:
            continue
        reversals = sum(a != b for a, b in zip(directions, directions[1:]))
        end = s.run_end(chain[-1], lambda v: v in (DASH, TURN))
        if reversals >= p["dashdance_min_reversals"]:
            stratum, kind = "positive_candidate", "dashdance"
        elif reversals == 1 and len(chain) == 2:
            stratum, kind = "near_miss_candidate", "single_dashback"
        elif reversals == 0:
            stratum, kind = "near_miss_candidate", "foxtrot_same_direction"
        else:
            continue
        evidence = {"dash_entries": len(chain), "reversals": reversals, "directions": directions,
                    "segment_frames": [b - a for a, b in zip(chain, chain[1:])],
                    "exit_state": s.state(end) if end < s.n and s.joined(end) else None}
        out.append(_candidate(sample_id, "dashdancing", stratum, kind, port, s, chain[0], end, chain,
                              evidence, "dashdance.v1", p, names))
    return out


def detect_short_hop(sample_id: str, port: str, rows: list[dict[str, Any]], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Ground jumps classified by jump input on the last jumpsquat frame; apex kept for a consistency check."""
    p, s, out = _params(params), _Series(rows), []
    names = ("tap_jump_threshold", "jump_max_air_frames", "max_frames_jump_to_airdodge")
    for i in range(s.n):
        if not s.entry(i, KNEE_BEND):
            continue
        j = s.run_end(i, lambda v: v == KNEE_BEND)
        if not (j < s.n and s.joined(j) and s.state(j) in GROUND_JUMP):
            continue
        soon = range(j, min(s.n, j + int(p["max_frames_jump_to_airdodge"]) + 1))
        if any(s.state(f) == ESCAPE_AIR for f in soon):
            continue  # jump-cancelled airdodge belongs to the wavedash family
        last = j - 1
        buttons, stick = s.buttons(last), s.rows[last]["input"].get("joystick.y")
        if buttons is None and stick is None:
            continue
        held_button = bool(buttons and buttons & (BUTTON_X | BUTTON_Y))
        held_stick = isinstance(stick, (int, float)) and stick >= p["tap_jump_threshold"]
        end, y0 = j, s.num(i, "position.y")
        while end < s.n and s.joined(end) and s.rows[end].get("airborne") and end - j < p["jump_max_air_frames"]:
            end += 1
        heights = [s.num(f, "position.y") for f in range(j, end)]
        apex = None if y0 is None or not heights or None in heights else round(max(heights) - y0, 4)
        full = held_button or held_stick
        evidence = {"jump_input_at_takeoff": {"button": held_button, "tap_stick": held_stick},
                    "jumpsquat_frames": j - i, "air_frames": end - j, "apex_height": apex,
                    "landed_within_limit": end < s.n and s.joined(end) and not s.rows[end].get("airborne"),
                    "character": s.rows[i].get("character")}
        out.append(_candidate(sample_id, "short hop", "near_miss_candidate" if full else "positive_candidate",
                              "full_hop" if full else "short_hop", port, s, i, max(end, j + 1), [i, j],
                              evidence, "short_hop.v1", p, names))
    return out


def detect_edgehog(sample_id: str, holder: str, rows: list[dict[str, Any]], opponent: str,
                   opponent_rows: list[dict[str, Any]], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Holder on ledge while the opponent is airborne beyond/below it on the same side; outcome decides stratum."""
    p, a = _params(params), _Series(rows)
    names = ("edgehog_min_overlap_frames", "edgehog_merge_gap_frames", "edgehog_outcome_frames")
    other = {row["frame"]: index for index, row in enumerate(opponent_rows)}
    b = _Series(opponent_rows)

    def hogging(i: int) -> bool:
        k = other.get(a.frame(i))
        if k is None or a.state(i) not in CLIFF or b.state(k) in CLIFF | DEAD or not b.rows[k].get("airborne"):
            return False
        ax, ay, bx, by = a.num(i, "position.x"), a.num(i, "position.y"), b.num(k, "position.x"), b.num(k, "position.y")
        if None in (ax, ay, bx, by) or ax == 0 or ax * bx <= 0:
            return False
        return abs(bx) > abs(ax) or by < ay

    spans: list[list[int]] = []
    for i in range(a.n):
        if not hogging(i):
            continue
        if spans and a.frame(i) - a.frame(spans[-1][1] - 1) <= p["edgehog_merge_gap_frames"] + 1:
            spans[-1][1] = i + 1
        else:
            spans.append([i, i + 1])
    out = []
    for start, end in spans:
        if a.frame(end - 1) - a.frame(start) + 1 < p["edgehog_min_overlap_frames"]:
            continue
        k0 = other[a.frame(end - 1)]
        stocks = b.num(k0, "stocks")
        outcome, k = None, k0 + 1
        while k < b.n and b.joined(k) and b.frame(k) - b.frame(k0) <= p["edgehog_outcome_frames"]:
            now = b.num(k, "stocks")
            if b.state(k) in DEAD or (stocks is not None and now is not None and now < stocks):
                outcome = "opponent_lost_stock"
            elif b.state(k) in CLIFF:
                outcome = "opponent_grabbed_ledge"
            elif b.rows[k].get("airborne") is False:
                outcome = "opponent_returned_to_stage"
            if outcome:
                break
            k += 1
        if outcome is None:
            continue  # unresolved within the lookahead, or the window ended
        holder_end = a.run_end(end - 1, lambda v: v in CLIFF)
        evidence = {"overlap_frames": a.frame(end - 1) - a.frame(start) + 1, "outcome": outcome,
                    "frames_span_end_to_outcome": b.frame(k) - a.frame(end - 1),
                    "holder_left_ledge_before_outcome": a.frame(holder_end - 1) < b.frame(k) if holder_end > 0 else None,
                    "opponent_character": b.rows[k0].get("character")}
        positive = outcome == "opponent_lost_stock"
        candidate = _candidate(sample_id, "edgehogging", "positive_candidate" if positive else "near_miss_candidate",
                               "ledge_denial_stock_loss" if positive else "ledge_held_opponent_recovered",
                               holder, a, start, end, [start, end - 1], evidence, "edgehog.v1", p, names,
                               opponent_port=opponent)
        candidate["event_frames"][1] = b.frame(k) + 1
        candidate["key_frames"].append(b.frame(k))
        candidate["examined_through_frame"] = b.frame(k)
        out.append(candidate)
    return out


def mine_window(window: dict[str, Any], sample_id: str, params: dict[str, Any] | None = None,
                families: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Run selected detectors on every port's leader; deterministic order by frame then ID."""
    selected = set(families or FAMILIES)
    if not selected <= set(FAMILIES):
        raise DatasetError("Unknown Melee family: " + ", ".join(sorted(selected - set(FAMILIES))))
    found: list[dict[str, Any]] = []
    series = {port: port_series(window, port) for port in ports(window)}
    for port, rows in series.items():
        for family, detector in (("wavedash", detect_wavedash), ("L-cancel", detect_l_cancel),
                                 ("dashdancing", detect_dashdance), ("short hop", detect_short_hop)):
            if family in selected:
                found.extend(detector(sample_id, port, rows, params))
    if "edgehogging" in selected and len(series) == 2:
        first, second = sorted(series)
        found.extend(detect_edgehog(sample_id, first, series[first], second, series[second], params))
        found.extend(detect_edgehog(sample_id, second, series[second], first, series[first], params))
    return sorted(found, key=lambda c: (c["event_frames"][0], c["family"], c["candidate_id"]))


def clip_and_frame_plan(candidate: dict[str, Any], hz: int = 60, max_frames: int = 64,
                        padding: tuple[int, int] | None = None) -> dict[str, Any]:
    """End-exclusive native clip with a ≤max_frames selection that always keeps defining transitions.

    Each key frame and its predecessor are required, so a short transition is
    never removed by uniform subsampling. Translating native frames to rendered
    video indices is a later step that needs the reviewed rendered manifest.
    """
    before, after = padding or DEFAULT_PADDING.get(candidate["family"], (15, 15))
    start = candidate["event_frames"][0] - before
    end = candidate["event_frames"][1] + after
    required = sorted({f for key in candidate["key_frames"] for f in (key - 1, key) if start <= f < end})
    frames = [{"index": f, "timestamp": f / hz} for f in range(start, end)]
    policy = "preserve" if len(frames) <= max_frames else "uniform"
    _, transform = select_frames(frames, max_frames=max_frames, policy=policy, required_indices=required)
    transform["index_space"] = "native_slippi_frame"
    return {"clip": {"start_seconds": start / hz, "end_seconds": end / hz},
            "clip_native_frames": [start, end], "frame_selection": transform}


def mine_adapter(adapter: Any, samples: list[Any], *, params: dict[str, Any] | None = None,
                 families: Iterable[str] | None = None, chunk_frames: int = 3600, overlap_frames: int = 600,
                 margin_frames: int = 30, first_frame: int = -123, max_chunks: int = 64,
                 max_candidates: int = 50_000) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mine replays in bounded overlapping windows; each event is owned by exactly one chunk.

    Chunk k owns events starting in [s_k + margin, s_k+1 + margin), so lookbehind
    and lookahead stay inside the chunk that reports the event. An owned event
    that still reaches a non-final chunk's edge (longer than overlap - margin
    frames) is dropped and counted rather than truncated.
    """
    if not 0 < margin_frames < overlap_frames < chunk_frames <= 4096:
        raise DatasetError("Require 0 < margin < overlap < chunk <= 4096 frames")
    step = chunk_frames - overlap_frames
    candidates: dict[str, dict[str, Any]] = {}
    report = []

    def keep(candidate: dict[str, Any]) -> None:
        candidates.setdefault(candidate["candidate_id"], candidate)
        if len(candidates) > max_candidates:
            raise DatasetError("Candidate limit exceeded; mine fewer replays")

    for sample in samples:
        chunks, start, dropped, finished = 0, first_frame, 0, False
        histogram: Counter = Counter()
        while chunks < max_chunks:
            window = adapter.read_window(sample, TimeRange(start, start + chunk_frames, "frames"))
            data = {"states": window.states or [], "actions": window.actions or []}
            chunks += 1
            if not data["states"]:
                finished = True  # overlapping chunks make this reachable only on the first read
                break
            last_frame = data["states"][-1]["frame_id"]
            final = last_frame < start + chunk_frames - 1
            owned_low = float("-inf") if start == first_frame else start + margin_frames
            owned_high = float("inf") if final else start + step + margin_frames
            for port, counts in state_histogram(data).items():
                for state, count in counts.items():
                    histogram[(port, state)] += count
            for candidate in mine_window(data, sample.sample_id, params, families):
                if not owned_low <= candidate["event_frames"][0] < owned_high:
                    continue
                if not final and candidate["examined_through_frame"] >= last_frame:
                    dropped += 1
                else:
                    keep(candidate)
            if final:
                finished = True
                break
            start += step
        report.append({"sample_id": sample.sample_id, "chunks_read": chunks,
                       "truncated": not finished, "dropped_at_window_edge": dropped,
                       "state_histogram": {f"{port}:{state}": count for (port, state), count in sorted(histogram.items())}})
    ordered = sorted(candidates.values(), key=lambda c: (c["sample_id"], c["event_frames"][0], c["family"], c["candidate_id"]))
    return ordered, report


def candidate_samples(candidates: list[dict[str, Any]], replays: dict[str, Any], sense_ids: dict[str, str],
                      plain_descriptions: dict[str, str] | None = None, sessions: dict[str, str] | None = None,
                      hz: int = 60, max_frames: int = 64) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Derived per-candidate sample rows plus export selections, keeping replay lineage connected."""
    rows, selections = [], {}
    for candidate in candidates:
        replay = replays[candidate["sample_id"]]
        digest = (replay.get("hashes") or {}).get("replay_sha256")
        plan = clip_and_frame_plan(candidate, hz=hz, max_frames=max_frames)
        lineage = {"parent_sample_id": candidate["sample_id"], "replay_id": digest or candidate["sample_id"]}
        if sessions and digest in sessions:
            lineage["session_id"] = sessions[digest]
        rows.append({"sample_id": candidate["candidate_id"], "dataset": replay.get("dataset", "slippi"),
                     "game": replay.get("game", "Super Smash Bros. Melee"), "version": replay.get("version") or "unknown",
                     "assets": copy.deepcopy(replay.get("assets") or {}), "modalities": ["actions", "states"],
                     "native_clock": copy.deepcopy(replay.get("native_clock") or {}),
                     "source": copy.deepcopy(replay.get("source") or {}), "lineage": lineage,
                     "hashes": {"replay_sha256": digest} if digest else {},
                     "clip": plan["clip"],
                     "public": {"evidence_text": "", "visual_refs": [],
                                "plain_description": (plain_descriptions or {}).get(candidate["family"], ""),
                                "action_options": []},
                     "metadata": {"candidate_id": candidate["candidate_id"], "family": candidate["family"],
                                  "clip_native_frames": plan["clip_native_frames"],
                                  "frame_selection": plan["frame_selection"],
                                  "visual_refs_status": "pending_reviewed_render",
                                  "annotation_status": "candidate_unreviewed"}})
        selections[candidate["candidate_id"]] = {
            "sense_id": sense_ids[candidate["family"]], "episode_id": "episode:" + candidate["candidate_id"],
            "category": "positive" if candidate["stratum"] == "positive_candidate" else "near_miss"}
    return rows, selections
