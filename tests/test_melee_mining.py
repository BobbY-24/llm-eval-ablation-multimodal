"""Synthetic telemetry fixtures for the Melee review-candidate rules (not real replays)."""
import unittest
from types import SimpleNamespace

from multimodal_game_eval.core import DatasetError, Window
from multimodal_game_eval.melee_mining import (
    STATUS, clip_and_frame_plan, detect_dashdance, detect_edgehog, detect_l_cancel, detect_short_hop,
    detect_wavedash, mine_adapter, mine_window, port_series)

WAIT, DASH, KNEE, JUMP_F, FALL, ESC, LFS, NAIR, LNAIR, LANDING = 14, 20, 24, 25, 29, 236, 43, 65, 70, 42


def seg(state, count, **fields):
    """Rows for one action state; callables receive the offset within the segment."""
    return [{"state": state, **{k: (v(i) if callable(v) else v) for k, v in fields.items()}} for i in range(count)]


def series(*segments, start=0):
    rows = [dict(row) for part in segments for row in part]
    for offset, row in enumerate(rows):
        row.setdefault("input", {})
        row.update(frame=start + offset, finalized=True)
    return rows


def window(by_port, start=0):
    length = max(len(rows) for rows in by_port.values())
    states, actions = [], []
    for k in range(length):
        state_ports, action_ports = {}, {}
        for port, rows in by_port.items():
            if k < len(rows):
                row = {key: value for key, value in rows[k].items() if key not in ("frame", "finalized")}
                action_ports[port] = {"leader": row.pop("input", {})}
                state_ports[port] = {"leader": row}
        common = {"frame_id": start + k, "timestamp": (start + k) / 60, "finalized": True}
        states.append({**common, "ports": state_ports})
        actions.append({**common, "ports": action_ports})
    return {"states": states, "actions": actions}


def wavedash_rows():
    return series(seg(WAIT, 5, **{"position.x": 0.0, "direction": 1.0}),
                  seg(KNEE, 4, **{"position.x": 0.0, "direction": 1.0}),
                  seg(JUMP_F, 1, **{"position.x": 0.0}), seg(ESC, 2, **{"position.x": 0.0}),
                  seg(LFS, 10, **{"position.x": lambda i: 2.0 * i}), seg(WAIT, 5, **{"position.x": 18.0}))


class WavedashTests(unittest.TestCase):
    def test_positive_has_key_transitions_and_slide(self):
        found = detect_wavedash("s", "1", wavedash_rows())
        self.assertEqual(len(found), 1)
        c = found[0]
        self.assertEqual((c["stratum"], c["kind"], c["status"]), ("positive_candidate", "wavedash", STATUS))
        self.assertEqual(c["key_frames"], [5, 10, 12])
        self.assertEqual(c["event_frames"], [5, 22])
        self.assertEqual(c["evidence"]["slide_direction"], "forward")

    def test_airdodge_without_landing_is_near_miss(self):
        rows = series(seg(WAIT, 3), seg(KNEE, 4), seg(JUMP_F, 1), seg(ESC, 15), seg(FALL, 5))
        (c,) = detect_wavedash("s", "1", rows)
        self.assertEqual((c["stratum"], c["kind"]), ("near_miss_candidate", "jump_airdodge_no_landing"))

    def test_waveland_without_jumpsquat_is_near_miss(self):
        rows = series(seg(FALL, 15), seg(ESC, 2), seg(LFS, 10), seg(WAIT, 3))
        (c,) = detect_wavedash("s", "1", rows)
        self.assertEqual(c["kind"], "waveland_without_jumpsquat")

    def test_rollback_gap_breaks_a_sequence(self):
        rows = wavedash_rows()
        for row in rows[11:]:
            row["frame"] += 5  # missing native frames between airdodge entry and landing
        self.assertEqual([c["kind"] for c in detect_wavedash("s", "1", rows)], ["jump_airdodge_no_landing"])


class LCancelTests(unittest.TestCase):
    def rows(self, flag=None, press_at=None, inputs=True):
        rows = series(seg(FALL, 10), seg(NAIR, 10), seg(LNAIR, 8, l_cancel=flag), seg(WAIT, 3))
        for row in rows:
            row["input"] = {"buttons_physical": 0, "triggers_physical.l": 0.0, "triggers_physical.r": 0.0} if inputs else {}
        if press_at is not None:
            rows[press_at]["input"]["buttons_physical"] = 0x0040
        return rows

    def test_game_flag_decides_and_input_cross_checks(self):
        (ok,) = detect_l_cancel("s", "1", self.rows(flag=1, press_at=17))
        self.assertEqual((ok["stratum"], ok["evidence"]["validator"], ok["evidence"]["sources_agree"]),
                         ("positive_candidate", "game_l_cancel_flag", True))
        (missed,) = detect_l_cancel("s", "1", self.rows(flag=2))
        self.assertEqual(missed["kind"], "l_cancel_missed")
        (disagree,) = detect_l_cancel("s", "1", self.rows(flag=1))
        self.assertFalse(disagree["evidence"]["sources_agree"])

    def test_input_window_only_and_no_guess_without_validators(self):
        (c,) = detect_l_cancel("s", "1", self.rows(press_at=15))
        self.assertEqual((c["stratum"], c["evidence"]["validator"]), ("positive_candidate", "input_window_only"))
        late = self.rows(press_at=12)  # 8 frames before landing: outside the 7-frame window
        self.assertEqual(detect_l_cancel("s", "1", late)[0]["stratum"], "near_miss_candidate")
        self.assertEqual(detect_l_cancel("s", "1", self.rows(inputs=False)), [])

    def test_autocancel_landing_is_near_miss(self):
        rows = series(seg(FALL, 10), seg(NAIR, 5), seg(LANDING, 4), seg(WAIT, 3))
        (c,) = detect_l_cancel("s", "1", rows)
        self.assertEqual(c["kind"], "aerial_autocancel_landing")


def dash(direction, count=6):
    return seg(DASH, count, direction=direction, state_age=lambda i: float(i + 1), airborne=False)


def ledge_pair(recover):
    holder = series(seg(WAIT, 5, **{"position.x": -60.0, "position.y": 0.0}),
                    seg(253, 40, **{"position.x": -85.0, "position.y": -10.0}))
    tail = (seg(WAIT, 10, airborne=False, stocks=4, **{"position.x": -70.0, "position.y": 0.0}) if recover
            else seg(3, 10, airborne=True, stocks=3, **{"position.x": -110.0, "position.y": -200.0}))
    opponent = series(seg(FALL, 5, airborne=True, stocks=4, **{"position.x": -100.0, "position.y": 5.0}),
                      seg(FALL, 20, airborne=True, stocks=4, **{"position.x": -110.0, "position.y": -40.0}), tail)
    return holder, opponent


class DashTests(unittest.TestCase):
    def dash(self, direction, count=6):
        return dash(direction, count)

    def test_alternating_dashes_propose_dashdance(self):
        rows = series(seg(WAIT, 3), self.dash(1.0), self.dash(-1.0), self.dash(1.0), seg(WAIT, 3))
        (c,) = detect_dashdance("s", "1", rows)
        self.assertEqual((c["kind"], c["evidence"]["reversals"], c["key_frames"]), ("dashdance", 2, [3, 9, 15]))

    def test_same_direction_and_single_reversal_are_near_misses(self):
        fox = detect_dashdance("s", "1", series(seg(WAIT, 3), self.dash(1.0), self.dash(1.0), seg(WAIT, 3)))
        back = detect_dashdance("s", "1", series(seg(WAIT, 3), self.dash(1.0), self.dash(-1.0), seg(WAIT, 3)))
        self.assertEqual([c["kind"] for c in fox + back], ["foxtrot_same_direction", "single_dashback"])

    def test_long_segment_splits_chain(self):
        rows = series(self.dash(1.0, 25), self.dash(-1.0), seg(WAIT, 2))
        self.assertEqual(detect_dashdance("s", "1", rows), [])


class ShortHopTests(unittest.TestCase):
    def rows(self, held_frames, airdodge=False):
        knee = seg(KNEE, 4, **{"position.y": 0.0, "character": 2})
        for i, row in enumerate(knee):
            row["input"] = {"buttons_physical": 0x0400 if i < held_frames else 0, "joystick.y": 0.0}
        air = seg(ESC if airdodge else JUMP_F, 20, airborne=True, **{"position.y": lambda i: float(min(i, 10))})
        return series(seg(WAIT, 3, airborne=False, **{"position.y": 0.0}), knee, air,
                      seg(WAIT, 3, airborne=False, **{"position.y": 0.0}))

    def test_release_before_takeoff_proposes_short_hop(self):
        (short,) = detect_short_hop("s", "1", self.rows(held_frames=2))
        (full,) = detect_short_hop("s", "1", self.rows(held_frames=4))
        self.assertEqual((short["kind"], full["kind"]), ("short_hop", "full_hop"))
        self.assertEqual(short["evidence"]["apex_height"], 10.0)
        self.assertTrue(short["evidence"]["landed_within_limit"])

    def test_jump_cancelled_airdodge_is_left_to_wavedash(self):
        self.assertEqual(detect_short_hop("s", "1", self.rows(held_frames=2, airdodge=True)), [])


class EdgehogTests(unittest.TestCase):
    def pair(self, recover):
        return ledge_pair(recover)

    def test_stock_loss_while_ledge_held_is_positive(self):
        holder, opponent = self.pair(recover=False)
        (c,) = detect_edgehog("s", "1", holder, "2", opponent)
        self.assertEqual((c["kind"], c["opponent_port"], c["evidence"]["outcome"]),
                         ("ledge_denial_stock_loss", "2", "opponent_lost_stock"))
        self.assertEqual(detect_edgehog("s", "2", opponent, "1", holder), [])

    def test_recovery_is_near_miss(self):
        holder, opponent = self.pair(recover=True)
        (c,) = detect_edgehog("s", "1", holder, "2", opponent)
        self.assertEqual(c["kind"], "ledge_held_opponent_recovered")


class MiningPipelineTests(unittest.TestCase):
    def long_replay(self):
        filler = seg(WAIT, 40, **{"position.x": 0.0, "direction": 1.0})
        p1 = series(*(([dict(r) for r in filler]) + [dict(r) for r in wavedash_rows()] for _ in range(12)))
        p2 = series(*([seg(WAIT, 7)] + [dash(1.0), dash(-1.0), dash(1.0), seg(WAIT, 30)]) * 6)
        return window({"1": p1, "2": p2})

    def test_mine_window_is_deterministic_and_rejects_unknown_family(self):
        data = self.long_replay()
        first, second = mine_window(data, "s"), mine_window(data, "s")
        self.assertEqual(first, second)
        self.assertEqual({c["family"] for c in first}, {"wavedash", "dashdancing"})
        self.assertTrue(all(c["status"] == STATUS for c in first))
        with self.assertRaises(DatasetError):
            mine_window(data, "s", families=["wombo"])
        with self.assertRaises(DatasetError):
            mine_window(data, "s", params={"unknown_threshold": 1})

    def test_chunked_mining_matches_whole_window(self):
        data = self.long_replay()

        class Adapter:
            def read_window(self, sample, span):
                pick = lambda rows: [r for r in rows if span.start <= r["frame_id"] < span.end]
                return Window(sample.sample_id, states=pick(data["states"]), actions=pick(data["actions"]))

        whole = {c["candidate_id"] for c in mine_window(data, "s")}
        chunked, report = mine_adapter(Adapter(), [SimpleNamespace(sample_id="s")], chunk_frames=200,
                                       overlap_frames=100, margin_frames=30, first_frame=0)
        self.assertEqual({c["candidate_id"] for c in chunked}, whole)
        self.assertGreater(report[0]["chunks_read"], 3)
        self.assertIn("1:24", report[0]["state_histogram"])

    def test_frame_plan_keeps_transitions_under_budget(self):
        (short,) = detect_wavedash("s", "1", wavedash_rows())
        plan = clip_and_frame_plan(short)
        self.assertEqual(plan["frame_selection"]["policy"], "preserve")
        self.assertLessEqual(plan["frame_selection"]["output_frames"], 64)
        holder, opponent = ledge_pair(recover=False)
        (long,) = detect_edgehog("s", "1", holder, "2", opponent)
        plan = clip_and_frame_plan(long, padding=(60, 60))
        kept = set(plan["frame_selection"]["selected_native_indices"])
        self.assertEqual(plan["frame_selection"]["policy"], "uniform")
        self.assertEqual(plan["frame_selection"]["output_frames"], 64)
        self.assertTrue({f for k in long["key_frames"] for f in (k - 1, k)} <= kept)
        self.assertEqual(plan["clip"]["start_seconds"], plan["clip_native_frames"][0] / 60)

    def test_port_series_merges_inputs(self):
        data = window({"1": series(seg(WAIT, 2, input={"buttons_physical": 16}))})
        self.assertEqual([r["input"]["buttons_physical"] for r in port_series(data, "1")], [16, 16])


if __name__ == "__main__":
    unittest.main()
