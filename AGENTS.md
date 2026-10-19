# multimodal-game-eval

Read `memory.md` at the start of each task and update it after substantive work.

Offline toolkit for grounding community gaming terms in game media: dataset adapters (loaders), annotation scaffolding, episode preparation, VLM prompt rendering, run planning and scoring of saved responses. Extracted on 2026-10-19 from `~/Documents/ChatGPT/Community Task/multimodal_game_eval/`, which remains the historical source; this repository is now the development copy.

- Tests: `python3 -I -B test_runner.py`, or sandboxed on macOS: `zsh run_offline.sh test`. Both must pass before commits.
- The package has no runtime dependencies. Optional readers (`av`, `numpy`, `Pillow`, `h5py`, `peppi-py==0.8.6`, `PyYAML`) are installed only in a deliberately prepared environment, never automatically.
- Demos and stored results are synthetic controls. Never describe them as model, human or real-gameplay results.
- The bundled inventory under `data/` is model-generated candidate data, not adjudicated gold. Do not edit it in place.
- `research/cv_vlm_community_terms/stage2_vlm_run_plan_v1.json` freezes file hashes of the Stage 2 configs and modules it lists. Editing any of those files means writing plan v2, not changing v1.
- Acquired media, annotations and run outputs stay untracked (`data/*`, `runs/`, `artifacts/*` except the saved demo/plan/verification records).
- No live provider calls, downloads, paid compute or publication without explicit user approval.
- Scratch work goes under `work/<task>/` (untracked).
