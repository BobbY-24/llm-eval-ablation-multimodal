# multimodal-game-eval

Read `memory.md` at the start of each task and update it after substantive work.

Offline toolkit for grounding community gaming terms in game media: dataset adapters (loaders), annotation scaffolding, episode preparation, VLM prompt rendering, run planning and scoring of saved responses. Extracted on 2026-10-19 from `~/Documents/ChatGPT/Community Task/multimodal_game_eval/`, which remains the historical source; this repository is now the development copy.

- Tests: `python3 -I -B test_runner.py`, or sandboxed on macOS: `zsh run_offline.sh test`. Both must pass before commits.
- The package has no runtime dependencies. Optional readers (`av`, `numpy`, `Pillow`, `h5py`, `peppi-py==0.8.6`, `PyYAML`) are installed only in a deliberately prepared environment, never automatically.
- Demos and stored results are synthetic controls. Never describe them as model, human or real-gameplay results.
- The bundled inventory under `data/` is model-generated candidate data, not adjudicated gold. Do not edit it in place.
- `research/cv_vlm_community_terms/stage2_vlm_run_plan_v1.json` freezes file hashes of the Stage 2 configs and modules it lists. Editing any of those files means writing plan v2, not changing v1.
- Acquired media, annotations and run outputs stay untracked (`data/*`, `runs/`, `artifacts/*` except the saved demo/plan/verification records).
- Part A (`ablation/`): `python3 -m ablation {plan,run,score}`; default provider is an offline mock. `--live` runs (OpenAI-compatible, Gemini) need explicit user approval each time, a finite `--max-calls`, and keys from environment variables only. Tests must stay offline (`tests/test_ablation_api.py` blocks sockets).
- `ablation/results/` holds only aggregate tables/figures derived from the 2026-10-01..03 50-source runs under `~/Documents/ChatGPT/agentic terms/outputs/` (regenerate with `ablation/results/build_results.py --outputs-root DIR`). Never add page snapshots, quotes, candidate terms, proposals, URLs or source IDs.
- No live provider calls, downloads, paid compute or publication without explicit user approval.
- Scratch work goes under `work/<task>/` (untracked).
