# LLM evaluation: API ablation study and multimodal game-term loaders

[![tests](https://github.com/BobbY-24/llm-eval-ablation-multimodal/actions/workflows/tests.yml/badge.svg)](https://github.com/BobbY-24/llm-eval-ablation-multimodal/actions/workflows/tests.yml) ![python](https://img.shields.io/badge/python-3.11%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

Python 3.11 or later. Neither part needs third-party packages for its core commands.

This repository has two parts:

| Part | What it is | Where |
|---|---|---|
| **A. API ablation study** | Provider/model comparison from completed API runs, plus a context-ablation runner (A0–A4) for OpenAI-compatible and Gemini APIs | `ablation/`, [research design](docs/research_design.md) |
| **B. Multimodal dataset loaders** | Nine offline game-media dataset adapters, annotation scaffolding, episode preparation and scoring | `src/multimodal_game_eval/` |

## Part A: API ablation study

### Provider comparison (completed runs)

Five extraction/judgment configurations were run on the same 50 frozen web sources (25 Gaming, 25 Cybersecurity). The tables in `ablation/results/` are derived only from aggregate ledgers. Page text, quotes, candidate terms, URLs and source IDs are excluded.

| Pipeline (extractor → judge) | Sources complete / partial / failed | Raw proposals | Grounding rejects | Known cost (USD) | Unknown hold (USD) |
|---|---|---:|---:|---:|---:|
| DeepSeek V4.1 Flash → GLM 5.3 Flash | 46 / 4 / 0 | 13,305 | 1.6% | 1.976 | 1.389 |
| GLM 5.3 Flash → GLM 5.3 Flash | 46 / 4 / 0 | 15,716 | 4.1% | 1.951 | 0 |
| GLM 5.3 Flash → Nemotron screen + GLM fallback | 40 / 5 / 5 | 17,349 | — | 1.841 | 0.486 |

- GLM extraction cost 28% less and proposed 18% more candidates than DeepSeek. That sent 38% more occurrences to the judge, so total known cost fell only 1.3%.
- The extractors were complementary: 2,555 normalized forms appeared only in DeepSeek's output and 4,331 only in GLM's.
- On the same GLM candidate pool, Nemotron and Qwen judge replays halted early, validating 17% and 50% of eligible IDs. Their low cost is not a completed-pipeline cost (`judgment_replays.csv`).

Known cost means reported tokens times the saved list price; it is not an invoice. An unknown hold is a reservation for a request whose billing was never resolved; it is not a charge. KEEP is a model verdict and no gold set exists, so no precision or recall is claimed.

| File | Contents |
|---|---|
| `matched50_pipelines.csv` | Per-pipeline sources, proposals, decisions, calls, tokens, cost |
| `matched50_by_community.csv` | Gaming / Cybersecurity split |
| `judgment_replays.csv` | Judge coverage on a fixed candidate pool |
| `latency_guarded50.csv` | Per-request latency where telemetry exists |
| `summary.json` | Headline ratios, caveats, SHA-256 pins of the input ledgers |
| `figures/*.svg` | Cost by stage, candidate pool, judge coverage |

`ablation/results/build_results.py --outputs-root DIR` regenerates the tables from the local run directories, which are not distributed with this repository.

### Context ablation runner (A0–A4)

Each item is asked about under five paired conditions:

| Condition | Evidence given to the model |
|---|---|
| A0 | Term only |
| A1 | Term, community and usage text |
| A2 | Term, community and image |
| A3 | Term, community, usage text and image |
| A4 | Term, community, usage text and a distractor image |

Scoring reports per-condition accuracy and paired contrasts with bootstrap confidence intervals: context gain (A1 − A0), multimodal gain (A3 − A1), text gain (A3 − A2), image-only gain (A2 − A0) and distractor penalty (A1 − A4). The design is ported from the local TermSight prototype.

```sh
python3 -m ablation plan --items ablation/fixtures/synthetic_items.jsonl --task mcq
python3 -m ablation run --items ablation/fixtures/synthetic_items.jsonl --max-calls 50 --out runs/ablation-mock
python3 -m ablation score --items ablation/fixtures/synthetic_items.jsonl --predictions runs/ablation-mock/predictions.jsonl
```

The default provider is an offline mock, so its scores are meaningless by design. A live run needs `--live`, a hard `--max-calls` and the key in an environment variable:

```sh
OPENAI_API_KEY=... python3 -m ablation run --provider openai --model MODEL --live --max-calls 40 --items ITEMS.jsonl --out runs/ablation-openai
GEMINI_API_KEY=... python3 -m ablation run --provider gemini --model MODEL --live --max-calls 40 --items ITEMS.jsonl --out runs/ablation-gemini
```

- **Other OpenAI-compatible hosts:** `--base-url` targets any OpenAI-compatible host (https, or http on localhost only), and `--api-key-env NAME` reads the key from a different variable.
- **Key handling:** keys are sent only as request headers, never in URLs, and are never logged.
- **Failure handling:** each stimulus gets at most one JSON repair, and every attempt is saved. Stimuli left unrun by the budget stay visible as `not_run_budget`.
- **Status:** no live A0–A4 run has been made. The bundled fixture has four fictional terms with placeholder images. The tests (`tests/test_ablation_api.py`) block sockets and exercise both API adapters through fake transports.

## Part B: multimodal dataset loaders

Local tools to join the existing community-term inventory to game media, annotate grounded meanings, prepare evaluation episodes, and score saved responses. Python 3.11 or later is required. This package has no runtime dependencies, downloads, model calls, or automatic installs.

The existing inventory contains candidate scope memberships and source usages. Its model-generated definitions and scope mappings are not adjudicated semantic gold. This project reads that inventory without changing it. Real media and expert annotations must be supplied separately; the demo creates synthetic fixtures and mock responses only.

## Repository layout

| Path | Contents |
|---|---|
| `ablation/` | Part A: A0–A4 runner, providers, scoring, synthetic fixture, `results/` from completed API runs |
| `src/multimodal_game_eval/` | Package: dataset adapters (loaders), media readers, inventory join, sampling, episode preparation, VLM prompt rendering, scoring, Melee candidate mining |
| `src/multimodal_game_eval/adapters/` | Per-dataset loaders: `vpt`, `minerl`, `csgo`, `terraria`, `proact`, `starcraftimage`, `slippi`, `dota2vis`, `aoe2` |
| `tests/` | Fixture-only unit tests (`python -I -B test_runner.py`) |
| `configs/` | Pilot, family checklist, Stage 2 VLM plan and prompt configs |
| `data/` | Bundled community-term inventory (`research/`) and its usage graph (`outputs/`); acquired media placed elsewhere under `data/` stays untracked |
| `artifacts/` | Saved synthetic demo (`demo-final/`), frozen Stage 2 plan projections (`stage2-plan-v1/`) and test verification records; new run outputs stay untracked |
| `research/cv_vlm_community_terms/` | CV/VLM grounding notes: frozen Stage 2 run plan, session decisions, vision-data sweep for the 1000-term list |
| `docs/` | Part A research design; datasets, annotation, protocol, API, communication and verification guides |

## Start offline

Run commands from this directory. The source-checkout entry point works without installation and ignores Python environment variables and user site packages:

```sh
python -I -B run.py --help
python -I -B run.py inventory
python -I -B run.py demo --config configs/pilot.json --output runs/demo-001
```

Choose a fresh output directory for each run. Demo scores verify plumbing, not model ability or real gameplay. No first real sample has been validated: no dataset media was acquired for this implementation.

On macOS, the wrapper adds an OS sandbox denying network access and unrelated file reads, strips the environment, bounds CPU/file output, and confines writes to `artifacts/`. It uses `python3` from `PATH` unless `--python /absolute/path/to/python3` is given:

```sh
zsh run_offline.sh test
zsh run_offline.sh inventory
zsh run_offline.sh demo --config configs/pilot.json --output artifacts/demo-001
```

To read one local media directory, name it explicitly as a read-only grant before the command. Pass absolute paths inside the wrapper; relative paths outside the package are denied:

```sh
zsh run_offline.sh --data-root /absolute/path/to/melee inspect --dataset slippi --root /absolute/path/to/melee
```

The wrapper refuses `/`, hidden paths, and any directory that contains the package or the home directory. `--runtime DIR` overrides the Python runtime directory the sandbox may read (default: two levels above the interpreter). The normal `python -I -B` entry does not itself provide an OS filesystem/network sandbox. Local media outside the wrapper's allowlist requires a separately reviewed isolation policy, rather than disabling isolation implicitly.

## Work with acquired local data

```sh
python -I -B run.py inspect --dataset vpt --root /absolute/path/to/vpt
python -I -B run.py index --dataset vpt --root /absolute/path/to/vpt --limit 20 --output runs/vpt-index.jsonl
python -I -B run.py validate --dataset vpt --root /absolute/path/to/vpt --limit 2 --decode --unit frames --start 0 --end 4 --output runs/vpt-check.json
python -I -B run.py export-annotations --index runs/vpt-index.jsonl --output runs/annotations-to-review.jsonl
python -I -B run.py prepare --annotations runs/adjudicated-annotations.jsonl --seed 11 --dev-fraction 0.25 --output runs/pilot-001
python -I -B run.py plan --episodes runs/pilot-001/episodes.jsonl --config configs/pilot.json --output runs/pilot-001-plan.json
python -I -B run.py score --gold runs/pilot-001/gold.jsonl --responses runs/responses.jsonl --output runs/score-001
python -I -B run.py validate-events --events runs/community-events.jsonl
python -I -B run.py analyze-communication --events runs/community-events.jsonl --seeded-forms runs/seeded-forms.json --probes runs/reviewed-probes.jsonl --output runs/communication-analysis.json
```

For Slippi telemetry, `mine-candidates` proposes per-family review windows (positives and near-misses) with deterministic rules, then writes a derived index and selections for `export-annotations`. Its outputs are review candidates, not labels; annotators must not see the candidate kind, stratum or rule evidence:

```sh
python -I -B run.py mine-candidates --dataset slippi --root /absolute/path/to/melee --limit 5 --output runs/mined-001
python -I -B run.py export-annotations --index runs/mined-001/candidate_index.jsonl --selections runs/mined-001/selections.json --output runs/mined-001/candidates-to-review.jsonl
```

Per-family property checklists and visibility classes are in `configs/pilot_families.json`. The Stage 2 grounding-only plan uses `configs/stage2_vlm.json` (`plan` reads its nested `plan_config`); prompt templates are in `configs/stage2_prompts.json` and render only from blinded `inputs.jsonl` inputs (`vlm_prompts.py`). No provider is called.

Replace the VPT paths with authorized local assets; this sequence does not acquire them. `export-annotations` creates review scaffolding, not accepted labels. See the annotation guide before `prepare`. `plan` writes a finite run plan; it does not invoke agents or providers. `score` imports saved JSONL responses. The CLI also accepts `--inventory` and `--graph` on inventory, annotation, preparation and demo commands for explicit source joins.

Supported adapter names are `vpt`, `minerl`, `csgo`, `terraria`, `proact`, `starcraftimage`, `slippi`, `dota2vis`, and `aoe2`. Availability depends on local files. `configs/datasets.example.json` is a path/provenance worksheet; copy it to `configs/datasets.local.json`, fill roots deliberately, and pass a chosen root through `--root`. The CLI does not download URLs or automatically consume that worksheet.

## Optional readers

Install only the extras needed in an intentionally prepared environment. No command here installs them automatically.

| Extra | Package | Used for |
|---|---|---|
| `video` | `av`, `numpy` | Local video decoding |
| `arrays` | `numpy` | Numeric arrays and RGB conversion |
| `images` | `Pillow`, `numpy` | Still and packed PNG images |
| `hdf5` | `h5py`, `numpy` | CS:GO HDF5 |
| `slippi` | `peppi-py==0.8.6` | Melee telemetry only |
| `yaml` | `PyYAML` | Optional Dota2-Vis class-name mapping |

For example, a manually reviewed install of `.[video,arrays]` enables video readers; `.[images,arrays,yaml]` enables Dota2-Vis images and class names. A missing reader produces a finding instead of a network fallback. Python installation and dependency management are outside the offline commands.

## Pilot and limits

The default pilot targets ten candidate families: Melee wavedash, L-cancel, dashdancing, short hop and edgehogging; AoE II DE FC, Trush, deer pushing, drush and 3 TC booming. It requests 20 expert-reviewed examples per family: ten positives, five near-misses and five wrong-context examples. The split targets 25% of lineage groups for development and 75% for evaluation; example proportions depend on group sizes. This is not a held-out-sense test.

Grounding crosses jargon/plain wording with text/visual/both input. Community experiments then test communication and newcomer transfer. No base model, paid provider, live game controller, replay renderer, model fine-tuning or audio decoder is included. Observed player actions are evidence about what happened, not labels for optimal action. See [experiment protocol](docs/PROTOCOL.md) for gates and causal controls.

Readers retain native clocks, provenance and transformation metadata. Windows are end-exclusive. Indexing is not full media validation; decoding a few windows is not validation of the entire corpus. Limits can leave an index truncated or an inventory partial. Unsafe paths, unsafe archive entries and pickle-dependent object arrays are rejected; do not bypass those findings to load unknown files.

- [Acquisition, formats, rights and rendering](docs/DATASETS.md)
- [Annotation and evidence guide](docs/ANNOTATION.md)
- [Staged experiment protocol](docs/PROTOCOL.md)
- [Python reader API and time ranges](docs/API.md)
- [Communication costs, exposure, reuse and reviewed probes](docs/COMMUNICATION.md)
- [Verification results and remaining real-data checks](docs/VERIFICATION.md)

## Verified locally

On 2026-10-10 the 121-test suite passed 110 / 11 optional-reader skips under base Python 3.14.6 (the 2026-10-08 93-test results were 90 passed / 3 skips under the bundled NumPy/Pillow runtime). The Melee mining rules are tested on synthetic telemetry only; their action-state constants are unverified on real replays. The final synthetic demo is saved in `artifacts/demo-final/`: 12 fictional episodes, 72 blinded inputs, correct/wrong/insufficient controls, and ten separate source-linked real pilot candidates. See the verification guide for exact native-reader limitations; these are software checks, not real gameplay or LLM results.

On 2026-10-19, after extraction into this standalone repository (bundled inventory, portable launcher), the suite ran 121 tests: 118 passed, 3 h5py skips, under Python 3.13.5 with NumPy 2.3.1, Pillow 11.1.0 and PyYAML 6.0.2, both directly and through `run_offline.sh` (`artifacts/verification-python-3.13.5.json`).
