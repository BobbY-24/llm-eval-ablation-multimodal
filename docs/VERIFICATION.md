# Implementation verification — 2026-10-08

All nine local adapters, CLI exports, annotation gates, partitioning, imported-response scoring, communication analysis and synthetic demonstration are implemented. These checks establish fixture conformance. No real dataset media was acquired or validated, and no model experiment was run.

| Isolated runtime | Tests discovered | Passed | Skipped | Failed/errors |
|---|---:|---:|---:|---:|
| Python 3.14.6, base standard library (2026-10-10, with Melee mining, Stage 2 plan and prompt tests) | 121 | 110 | 11 | 0 |
| Python 3.14.6, base standard library (2026-10-08) | 93 | 82 | 11 | 0 |
| Python 3.12.14, NumPy 2.3.5 / Pillow 12.3.0 | 93 | 90 | 3 | 0 |

Machine-readable results are saved in `artifacts/verification-python-3.14.6.json` and `artifacts/verification-python-3.12.14.json`. Each records exact skips, runtime/dependency versions and fixture-only status. Tests ran through `run_offline.sh` with an empty environment, denied network, restricted filesystem access and finite CPU/file output limits. No dependencies were installed.

The three skips in the bundled runtime are native HDF5 fixtures because h5py is absent. CS:GO's BGR-to-RGB conversion was separately checked with real NumPy arrays and a mocked HDF reader; that does not validate h5py compatibility. PyAV, peppi-py and PyYAML are also absent. Video alignment, Slippi parser shape/rollback and rendering mappings have controlled mocked-reader fixtures; native video decoding, the pinned parser and YAML class mapping still need checks in a prepared reader environment. Do not interpret absent native-reader coverage as a pass.

Actual local fixture I/O includes numeric MineRL NPZ arrays, refusal of object-array pickle loading, RGB images, a packed StarCraftImage PNG and Dota image/YOLO annotations. Archive/path checks, missing modalities, recorded/inferred action provenance, source joins, unscored disagreement, gold projection, component leakage, response errors, timing, exposure order and resource bounds have dedicated cases.

The final `artifacts/demo-final/` run contains ten real inventory candidates with saved occurrence evidence, plus **separate fictional controls**. It prepares twelve synthetic episodes and 72 blinded condition inputs, with correct-control interpretation accuracy 1.0, wrong-control accuracy 0.0, and explicit insufficient-evidence cases. Its event access audit has no findings. Synthetic reviewer IDs are fixtures; they are not expert annotations. See `artifacts/demo-final/SUMMARY.json` and the protected/public exports separately.

The canonical inventory has 903 scope-specific candidate senses, each with an exact structural occurrence join. Their meanings remain unadjudicated. Both original source files retain their pre-implementation SHA-256s:

| Read-only source | SHA-256 |
|---|---|
| `research/GAME_TERM_INVENTORY_2026-10-06.json` | `a9df5e368d774de2d357d04724efbe47c4f57e23a441e19c26e3d6c09d4ef0c6` |
| `outputs/01a10cf7-a4dc-7cd2-b0d9-e2e69a76155a/game_context_space.json` | `88a54b303b467f4d5dc630bd658aa969dd77f21f10e06664ec16bed8faaf7c30` |

2026-10-10 additions are fixture-checked only: `melee_mining.py` rules (synthetic frame sequences), chunked mining equivalence, `mine-candidates` → `export-annotations` → `prepare` (zero eligible episodes, as intended for unreviewed candidates), grounding-only `plan`, and gold-free prompt rendering of the synthetic demo inputs. The `--data-root` read grant was checked with canaries: the named directory was readable, while a sibling directory and the no-flag case were denied. No real replay was parsed.

For the real pilot, acquire reviewed Melee/AoE media, verify native readers and frame mappings, complete independent annotations/adjudication, freeze source components, assign fresh newcomer probe items, and bound provider input context/cost before live execution. Collection/rendering, inference and parameter training remain separate stages as specified in the plan.
