# Multimodal game-term evaluation

Local tools to join the existing community-term inventory to game media, annotate grounded meanings, prepare evaluation episodes, and score saved responses. Python 3.11 or later is required. The base package has no runtime dependencies, downloads, model calls, or automatic installs.

The existing inventory contains candidate scope memberships and source usages. Its model-generated definitions and scope mappings are not adjudicated semantic gold. This project reads that inventory without changing it. Real media and expert annotations must be supplied separately; the demo creates synthetic fixtures and mock responses only.

## Repository layout

| Path | Contents |
|---|---|
| `src/multimodal_game_eval/` | Package: dataset adapters (loaders), media readers, inventory join, sampling, episode preparation, VLM prompt rendering, scoring, Melee candidate mining |
| `src/multimodal_game_eval/adapters/` | Per-dataset loaders: `vpt`, `minerl`, `csgo`, `terraria`, `proact`, `starcraftimage`, `slippi`, `dota2vis`, `aoe2` |
| `tests/` | Fixture-only unit tests (`python -I -B test_runner.py`) |
| `configs/` | Pilot, family checklist, Stage 2 VLM plan and prompt configs |
| `data/` | Bundled community-term inventory (`research/`) and its usage graph (`outputs/`); acquired media placed elsewhere under `data/` stays untracked |
| `artifacts/` | Saved synthetic demo (`demo-final/`), frozen Stage 2 plan projections (`stage2-plan-v1/`) and test verification records; new run outputs stay untracked |
| `research/cv_vlm_community_terms/` | CV/VLM grounding notes: frozen Stage 2 run plan, session decisions, vision-data sweep for the 1000-term list |
| `docs/` | Datasets, annotation, protocol, API, communication and verification guides |

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
