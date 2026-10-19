# Project memory

## Origin and state — 2026-10-19

- Extracted from the Community Task workspace (`multimodal_game_eval/` at Community Task commit `8f7aa7d` plus uncommitted state of 2026-10-19). That workspace's `memory.md` holds the full history of the CV/VLM decisions; the relevant parts are summarized here.
- Changes made during extraction:
  - `DEFAULT_INVENTORY` now points at the bundled `data/research/GAME_TERM_INVENTORY_2026-10-06.json` (byte-identical to the Community Task v1.0.0 release copy); its usage graph is auto-resolved from `data/outputs/01a10cf7-…/game_context_space.json`.
  - `run_offline.sh`/`offline.sb`/`offline_entry.py` are the portable launcher from the v1.0.0 release, with the later `--data-root` read-only grant ported in. The Codex-owned `--bundled-python` option was dropped; use `--python`.
  - CV/VLM planning notes copied into `research/cv_vlm_community_terms/` (agent prompt and sandbox scratch excluded).
- Verified 2026-10-19: 121 tests, 118 passed, 3 h5py skips (Python 3.13.5, NumPy 2.3.1, Pillow 11.1.0, PyYAML 6.0.2), directly and through `run_offline.sh`. The sandbox denied an ungranted media read, allowed it with `--data-root`, and refused a home-directory grant.
- Saved artifacts and the usage graph contain absolute local provenance paths (`/Users/bobyan/...`); kept as recorded.

## CV/VLM grounding decisions (from 2026-10-10/11)

- Melee/Slippi first; first family wavedash, L-cancel as telemetry-only control. AoE deferred.
- Provisional single-annotator labels (the user), clearly marked. Closed API VLMs only. Budget $0, so plan only.
- Preferred first Melee sample: Slippi Public Dataset v3.7 on HF (`erickfm/slippi-public-dataset-v3.7`, CC0).
- Unverified: Melee action-state IDs and button masks, peppi field names. Check mining-manifest state histograms on the first real replay.
- Blocked on the user: replays plus approval of the peppi-py install, a renderer with a user-owned ISO, a second annotator or a provisional-mode decision (`prepare` needs two reviewers), and any spend.
