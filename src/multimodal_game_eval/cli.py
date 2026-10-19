"""Offline CLI: inspect → index → annotate → prepare → plan → import and score."""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .core import DatasetError, Finding, MissingDependency, TimeRange, read_json, read_jsonl, safe_path, sha256_file, to_jsonable
from .inventory import load_inventory, validate_annotation_join
from .io import new_output_directory, write_json, write_jsonl

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = PACKAGE_ROOT / "data/research/GAME_TERM_INVENTORY_2026-10-06.json"


def _print(value: Any) -> None:
    print(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


def _adapter(args: argparse.Namespace) -> Any:
    from .adapters import create_adapter
    return create_adapter(args.dataset, Path(args.root))


def _limited(adapter: Any, limit: int) -> tuple[list[Any], bool]:
    if not 1 <= limit <= 100_000:
        raise DatasetError("Limit must be between 1 and 100000")
    rows = list(itertools.islice(adapter.iter_samples(), limit + 1))
    return rows[:limit], len(rows) > limit


def _inspect(args: argparse.Namespace) -> int:
    _print(_adapter(args).probe())
    return 0


def _index(args: argparse.Namespace) -> int:
    outputs = (Path(args.output), Path(str(args.output) + ".manifest.json"))
    if not args.force and any(path.exists() or path.is_symlink() for path in outputs):
        raise DatasetError("Index or companion manifest exists; select new outputs or --force")
    adapter = _adapter(args)
    samples, truncated = _limited(adapter, args.limit)
    findings: list[Finding] = []
    for sample in samples:
        for kind, reference in sample.assets.items():
            try:
                asset = safe_path(adapter.root, reference)
                if not asset.is_file():
                    sample.hashes[kind] = None
                    findings.append(Finding("asset_not_hashable", f"No local file for {kind}", "warning", sample.sample_id))
                elif args.skip_asset_hashes:
                    sample.hashes[kind] = None
                else:
                    sample.hashes[kind] = sha256_file(asset, args.max_hash_bytes)
            except (OSError, ValueError) as error:
                sample.hashes[kind] = None
                findings.append(Finding("asset_not_hashed", str(error), "warning", sample.sample_id))
    count = write_jsonl(args.output, samples, args.force)
    manifest = {"dataset": args.dataset, "count": count, "truncated": truncated, "limit": args.limit,
                "index_sha256": sha256_file(args.output), "asset_hashes_complete": bool(samples) and all(all(value is not None for value in sample.hashes.values()) for sample in samples),
                "validation_status": "indexed_not_media_validated", "findings": findings}
    write_json(str(args.output) + ".manifest.json", manifest, args.force)
    _print(manifest)
    return 0


def _validate(args: argparse.Namespace) -> int:
    adapter = _adapter(args)
    samples, truncated = _limited(adapter, args.limit)
    findings: list[Any] = []
    decoded = 0
    if not samples:
        findings.append(Finding("no_samples", "No supported local samples are available"))
    for sample in samples:
        local_findings = adapter.validate(sample)
        findings.extend(local_findings)
        if args.decode and not any(getattr(item, "severity", "error") == "error" for item in local_findings):
            try:
                adapter.read_window(sample, TimeRange(args.start, args.end, args.unit))
                decoded += 1
            except MissingDependency as error:
                findings.append(Finding("missing_dependency", str(error), "warning", sample.sample_id))
            except (ValueError, OSError, KeyError, TypeError) as error:
                findings.append(Finding("decode_failed", str(error), "error", sample.sample_id))
    result = {"dataset": args.dataset, "samples_checked": len(samples), "truncated": truncated,
              "decoded_windows": decoded, "requested_decode": args.decode, "findings": findings,
              "validation_status": "local_window_checks" if decoded else "schema_checks_only"}
    if args.output:
        write_json(args.output, result, args.force)
    _print(result)
    return 2 if any(getattr(item, "severity", "error") == "error" for item in findings) else 0


def _export_annotations(args: argparse.Namespace) -> int:
    from .experiments import export_annotations
    inventory = load_inventory(args.inventory, args.graph)
    samples = list(read_jsonl(args.index)) if args.index else []
    selections = read_json(args.selections) if args.selections else None
    if Path(args.output).exists() and not args.force:
        raise DatasetError("Annotation output exists; select a new file or --force")
    if args.force and Path(args.output).exists():
        # The API is deliberately create-only. Replace through a sibling file so
        # failed validation leaves the original annotation export intact.
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(dir=Path(args.output).parent) as temporary:
            staged = Path(temporary) / "annotations.jsonl"
            result = export_annotations(samples, inventory, staged, selections)
            staged.replace(args.output)
        result.update(output=str(args.output), annotations_path=str(args.output))
        _print(result)
    else:
        _print(export_annotations(samples, inventory, Path(args.output), selections))
    return 0


def _prepare(args: argparse.Namespace) -> int:
    from .experiments import prepare_episodes
    inventory = load_inventory(args.inventory, args.graph)
    new_output_directory(args.output)
    _print(prepare_episodes(Path(args.annotations), Path(args.output), seed=args.seed, dev_fraction=args.dev_fraction, inventory=inventory))
    return 0


def _plan(args: argparse.Namespace) -> int:
    from .experiments import plan_runs
    config = read_json(args.config)
    if isinstance(config, dict) and isinstance(config.get("plan_config"), dict):
        config = config["plan_config"]  # e.g. configs/stage2_vlm.json
    result = plan_runs(Path(args.episodes), config)
    if args.output:
        write_json(args.output, result, args.force)
    _print(result)
    return 0


def _score(args: argparse.Namespace) -> int:
    from .experiments import score_responses
    output = new_output_directory(args.output) if args.output else None
    _print(score_responses(Path(args.gold), Path(args.responses), output))
    return 0


def _events(args: argparse.Namespace) -> int:
    from .experiments import validate_events
    findings = validate_events(Path(args.events))
    result = {"findings": findings, "valid": not any(item.get("severity", "error") == "error" for item in findings)}
    if args.output:
        write_json(args.output, result, args.force)
    _print(result)
    return 0 if result["valid"] else 2


def _communication(args: argparse.Namespace) -> int:
    from .communication import analyze_communication
    forms = read_json(args.seeded_forms) if args.seeded_forms else []
    probes = list(read_jsonl(args.probes, max_bytes=64 * 1024**2)) if args.probes else None
    result = analyze_communication(Path(args.events), forms, probes)
    if args.output:
        write_json(args.output, result, args.force)
    _print(result)
    return 2 if result["status"] == "invalid" else 0


def _mine(args: argparse.Namespace) -> int:
    from collections import Counter
    from .melee_mining import CONSTANTS_STATUS, FAMILIES, RULESET_VERSION, candidate_samples, mine_adapter
    if args.dataset != "slippi":
        raise DatasetError("Telemetry candidate mining is implemented for slippi only")
    config = read_json(args.families_config)
    inventory = load_inventory(args.inventory, args.graph)
    families = args.families.split(",") if args.families else list(FAMILIES)
    sense_ids = {name: config["families"][name]["sense_id"] for name in families}
    unknown = sorted(sense for sense in sense_ids.values() if sense not in inventory)
    if unknown:
        raise DatasetError("Configured sense IDs are absent from the inventory: " + ", ".join(unknown))
    params = read_json(args.params) if args.params else None
    sessions = read_json(args.sessions) if args.sessions else None
    output = new_output_directory(args.output)
    adapter = _adapter(args)
    samples, truncated = _limited(adapter, args.limit)
    candidates, report = mine_adapter(adapter, samples, params=params, families=families)
    replays = {sample.sample_id: to_jsonable(sample) for sample in samples}
    rows, selections = candidate_samples(candidates, replays, sense_ids,
                                         {name: config["families"][name]["plain_description"] for name in families},
                                         sessions)
    paths = {"candidates": output / "candidates.jsonl", "candidate_index": output / "candidate_index.jsonl",
             "selections": output / "selections.json"}
    write_jsonl(paths["candidates"], candidates)
    write_jsonl(paths["candidate_index"], rows)
    write_json(paths["selections"], selections)
    manifest = {"status": "review_candidates_not_labels", "dataset": args.dataset, "ruleset": RULESET_VERSION,
                "constants_status": CONSTANTS_STATUS, "families": families, "params_override": params,
                "replays_mined": len(samples), "replay_scan_truncated": truncated,
                "candidates": len(candidates),
                "by_family_stratum": {f"{f}|{s}": n for (f, s), n in sorted(Counter((c["family"], c["stratum"]) for c in candidates).items())},
                "by_kind": dict(sorted(Counter(c["kind"] for c in candidates).items())),
                "replay_reports": report, "families_config_sha256": sha256_file(args.families_config),
                "files": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()},
                "next_step": "export-annotations --index candidate_index.jsonl --selections selections.json",
                "limitations": ["Rule outputs propose review windows; they are not labels or human gold.",
                                "Action-state IDs and button masks are unverified against real replays; inspect replay_reports state histograms first.",
                                "Annotators must not see candidate kind, stratum or rule evidence (blinding for the telemetry-rule reference arm).",
                                "visual_refs stay empty until frames are rendered and alignment is reviewed."]}
    write_json(output / "manifest.json", manifest)
    _print(manifest)
    return 0


def _inventory(args: argparse.Namespace) -> int:
    inventory = load_inventory(args.inventory, args.graph)
    counts: dict[str, int] = {}
    for sense in inventory.values():
        counts[sense["game"]] = counts.get(sense["game"], 0) + 1
    _print({"senses": len(inventory), "verified_occurrence_joins": sum(bool(sense["occurrence_ids"]) for sense in inventory.values()), "scopes": counts})
    return 0


def _demo(args: argparse.Namespace) -> int:
    from .demo import build_demo
    output = new_output_directory(args.output)
    _print(build_demo(output, load_inventory(args.inventory, args.graph), Path(args.config)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("inspect", _inspect), ("index", _index), ("validate", _validate)):
        command = commands.add_parser(name)
        command.add_argument("--dataset", required=True, choices=["vpt", "minerl", "csgo", "terraria", "proact", "starcraftimage", "slippi", "dota2vis", "aoe2"])
        command.add_argument("--root", required=True)
        command.set_defaults(function=function)
        if name != "inspect":
            command.add_argument("--limit", type=int, default=100)
            command.add_argument("--output", required=name == "index")
            command.add_argument("--force", action="store_true")
        if name == "index":
            command.add_argument("--skip-asset-hashes", action="store_true")
            command.add_argument("--max-hash-bytes", type=int, default=2 * 1024**3)
        if name == "validate":
            command.add_argument("--decode", action="store_true")
            command.add_argument("--unit", choices=["frames", "seconds"], default="frames")
            command.add_argument("--start", type=float, default=0)
            command.add_argument("--end", type=float, default=4)
    for name, function in (("export-annotations", _export_annotations), ("prepare", _prepare), ("inventory", _inventory), ("demo", _demo)):
        command = commands.add_parser(name)
        command.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
        command.add_argument("--graph")
        command.set_defaults(function=function)
        if name != "inventory":
            command.add_argument("--output", required=True)
        if name == "export-annotations":
            command.add_argument("--index")
            command.add_argument("--selections")
            command.add_argument("--force", action="store_true")
        if name == "prepare":
            command.add_argument("--annotations", required=True)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--dev-fraction", type=float, default=.25)
        if name == "demo":
            command.add_argument("--config", default=str(PACKAGE_ROOT / "configs/pilot.json"))
    command = commands.add_parser("mine-candidates", help="Propose telemetry review windows (not labels)")
    command.add_argument("--dataset", required=True, choices=["slippi"])
    command.add_argument("--root", required=True)
    command.add_argument("--limit", type=int, default=5)
    command.add_argument("--output", required=True)
    command.add_argument("--families", help="Comma-separated subset of the Melee pilot families")
    command.add_argument("--families-config", default=str(PACKAGE_ROOT / "configs/pilot_families.json"))
    command.add_argument("--params", help="JSON object overriding detector thresholds")
    command.add_argument("--sessions", help="JSON object mapping replay SHA-256 to a session/set ID")
    command.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    command.add_argument("--graph")
    command.set_defaults(function=_mine)
    command = commands.add_parser("plan")
    command.add_argument("--episodes", required=True)
    command.add_argument("--config", default=str(PACKAGE_ROOT / "configs/pilot.json"))
    command.add_argument("--output")
    command.add_argument("--force", action="store_true")
    command.set_defaults(function=_plan)
    command = commands.add_parser("score")
    command.add_argument("--gold", required=True)
    command.add_argument("--responses", required=True)
    command.add_argument("--output")
    command.set_defaults(function=_score)
    command = commands.add_parser("validate-events")
    command.add_argument("--events", required=True)
    command.add_argument("--output")
    command.add_argument("--force", action="store_true")
    command.set_defaults(function=_events)
    command = commands.add_parser("analyze-communication")
    command.add_argument("--events", required=True)
    command.add_argument("--seeded-forms", help="JSON list of seeded strings; absent means no seeded forms")
    command.add_argument("--probes", help="Separate adjudicated fresh probe outcome JSONL")
    command.add_argument("--output")
    command.add_argument("--force", action="store_true")
    command.set_defaults(function=_communication)
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (DatasetError, OSError, KeyError, TypeError, ValueError) as error:
        _print({"error": type(error).__name__, "message": str(error)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
