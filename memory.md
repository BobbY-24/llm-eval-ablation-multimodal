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

## Public release — 2026-10-19
- Made public as the P4 portfolio repo; added MIT LICENSE, `.github/workflows/tests.yml` (test_runner on 3.11/3.13, green) and README badges. Planned rename to `llm-eval-ablation-multimodal` is deferred until the API ablation module (part A) lands.
- The refined terminology dataset (PURA, `community-terminology-llm-evaluation`) must stay private: do not copy it or derived items into this public repo.

## Part A: API ablation study — 2026-10-19
- Added `ablation/` (A0–A4 context-ablation runner ported from local TermSight: conditions, strict JSON contract with one repair, mock/OpenAI-compatible/Gemini adapters via injectable transport, budgeted runner, MCQ + token-F1 scoring, paired contrasts with bootstrap CI), `ablation/fixtures/` (4 invented terms, fictional communities, placeholder PNGs; checked: no overlap with the private dataset), `docs/research_design.md`, README Part A section, `tests/test_ablation_api.py` (18 offline tests).
- `ablation/results/`: aggregates from deepseek-glm-50, glm-glm-50, glm-nemotron-50, glm-qwen-judge50 and all-model-comparison (which supplies the guarded GLM/Nemotron run). Some counts (distinct forms, grounding rejects, community splits, Qwen smoke 508 proposals, guarded 17,349 proposals) are transcribed from report prose; listed in `TRANSCRIBED` in `build_results.py`. Headline: GLM extraction −28% cost, +18% proposals vs DeepSeek; total known cost only −1.3%; Nemotron/Qwen judge replays halted at 17% / 50% coverage.
- No live A0–A4 run exists (budget $0). Verified 2026-10-19: 139 tests, 136 passed, 3 h5py skips, directly and via `run_offline.sh`.
- Repo rename to `llm-eval-ablation-multimodal` still awaits the user's go-ahead.
