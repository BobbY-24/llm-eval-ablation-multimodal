# Local reader API

All adapters use the same configured-root interface. Construction, discovery and schema inspection need no game installation or model weights. Media and native parser dependencies load only when needed.

```python
from multimodal_game_eval.adapters import create_adapter
from multimodal_game_eval.core import TimeRange

reader = create_adapter("aoe2", "/absolute/path/to/local/export")
availability = reader.probe()
sample = next(reader.iter_samples())
findings = reader.validate(sample)
window = reader.read_window(sample, TimeRange(0, 2, "seconds"))
```

`probe()` inspects the configured root and returns availability, bounded sample count, declared modalities and notes. `iter_samples()` yields normalized `Sample` records without decoding video. `validate(sample)` returns typed errors/warnings; it never acquires a missing asset. `read_window(sample, range)` preserves native data in an end-exclusive window. A missing optional reader raises `MissingDependency`; invalid or unsupported data raises `DatasetError`/`ValueError`.

`Sample.version` is the game build/edition; `dataset_revision` identifies the recording/export format. `source`, `lineage`, `hashes`, `native_clock` and the original metadata remain separate. The CLI index computes bounded SHA-256s of referenced local assets; discovery alone does not imply hashes or validation. IDs are stable within the declared dataset layout; content hashes provide identity across duplicate media/replay files.

`Window` has `frames`, `actions`, `states`, `annotations`, `spatial`, `audio`, and `metadata`. Absent channels are `None`; present channels with no matching events are empty lists. Video frames retain native indices, encoded PTS, time base, seconds and RGB pixels. Game state or spatial summaries are never silently converted into gameplay screenshots. Audio decoding is outside v1. Keep original source annotations as evidence; they do not become reviewed term gold automatically.

| Reader | Native range contract |
|---|---|
| VPT | Frame indices or seconds; action records retain recorder tick/milliseconds. |
| MineRL | Frame/transition indices; seconds require explicit trusted FPS metadata. Terminal next observation is retained separately. |
| CS:GO | Native file frame IDs or nominal 16 Hz seconds; gaps remain gaps. |
| Terraria | Encoded frame indices or video seconds; source/filter provenance retained. |
| Proact-VL | Source-video seconds, intersected with the annotation clip. |
| StarCraftImage | Exactly `TimeRange(0, 1, "frames")`; returns one packed spatial summary. |
| Slippi | Native game frame IDs, including negative pregame frames; seconds use the declared regional timebase. Separately registered capture mapping is required for pixels. |
| Dota2-Vis | Exactly `TimeRange(0, 1, "frames")`; timestamp is unknown. |
| AoE II | Video range plus sidecars in their explicitly declared matching `time_unit`. No implicit game-clock conversion. |

Readers apply finite inventory, text, array and window bounds. Request short windows rather than disabling limits. Full media remains in the supplied external root; JSON exports contain references, provenance and labels, not copied large videos.

For model-facing media, call `sampling.select_frames` explicitly and save its transformation ledger. The default preserves every supplied frame and refuses an over-budget window. A selected visual reference must then be resolved into actual pixels by the future inference harness. The preparation CLI supplies no vision-model transport.

For reviewed data, `experiments.prepare_episodes` exports public inputs and protected gold separately. The inventory argument is optional at the Python API level for controlled fixtures, and its absence is disclosed. The CLI always supplies the canonical inventory/occurrence graph. Pass only the `input` object of an input row to a model; retain outer IDs for joining saved responses afterward.
