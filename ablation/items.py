"""Draft a review sheet of real items from the bundled usage graph, then build items JSONL.

``draft_review`` reads gaming rows of the graph (term, scope, example usage,
community meaning, popular meaning) and writes one CSV row per term for human
review. Distractors are the popular meaning plus meanings of other terms in
the same scope, picked deterministically. Saved meanings are model proposals,
not gold, so a reviewer must keep, correct and tag each row and attach an
image. ``build_items`` validates the reviewed sheet and writes the JSONL that
``conditions.load_items`` reads. Neither function calls a provider.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from ablation.conditions import CONDITIONS, build_stimuli, load_items
from ablation.providers import MAX_IMAGE_BYTES, MIME_TYPES

DEFAULT_GRAPH = (Path(__file__).resolve().parents[1] / "data" / "outputs"
                 / "01a10cf7-a4dc-7cd2-b0d9-e2e69a76155a" / "game_context_space.json")
DEFAULT_SCOPES = ("Super Smash Bros. Melee", "Minecraft", "Dota 2", "Age of Empires II")
REVIEW_COLUMNS = ("id", "keep", "term", "community", "text_context", "gold_meaning",
                  "distractor_1", "distractor_2", "distractor_3", "distractor_1_source",
                  "image_file", "image_source", "text_explicitness", "visual_groundability",
                  "flags", "notes", "term_group_id", "occurrence_id", "source_url", "selection_basis")
KEEP_VALUES = frozenset({"y", "yes", "x", "1", "true", "keep"})
STRATA = {"text_explicitness": ("explicit", "implicit"), "visual_groundability": ("high", "low")}
SIMILAR = 0.6  # content-token F1 at which two options count as near-duplicates

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("""a an and are as at be by for from in into is it its of on or that the their them
this to used using when where which while who with without your you game games player players""".split())
_LABEL_STOP = frozenset({"the", "super", "bros", "series", "games", "game", "general", "legend",
                         "edition", "definitive", "family", "engine"})
_NO_POPULAR = re.compile(r"^\s*(no (established|general|common)|none\b|n/?a\b)", re.I)
_DEFINES = re.compile(r"\[\s*edit\s*\]|\bis a term\b|\brefers to\b|\bis defined as\b", re.I)


def _clean(text) -> str:
    return " ".join(str(text or "").split())


def _content(text: str, drop=frozenset()) -> Counter:
    return Counter(t for t in _TOKEN.findall(text.lower()) if t not in _STOP and t not in drop)


def overlap(a: str, b: str, drop=frozenset()) -> float:
    """Content-token F1 between two strings (stopwords and ``drop`` tokens ignored)."""
    p, g = _content(a, drop), _content(b, drop)
    common = sum((p & g).values())
    if not common:
        return 0.0
    precision, recall = common / sum(p.values()), common / sum(g.values())
    return 2 * precision * recall / (precision + recall)


def _rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def community_tokens(label: str) -> set[str]:
    return {t for t in _TOKEN.findall(label.lower()) if len(t) >= 4 and t not in _LABEL_STOP}


def load_gaming_rows(graph_path: str | Path) -> list[dict]:
    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    return [row for row in graph["original_terms"] if row.get("Subject area") == "Gaming"]


def scope_counts(rows: list[dict]) -> Counter:
    return Counter(scope for row in rows for scope in row.get("game_or_context_scopes") or [])


def _pick_distractors(row: dict, gold: str, pool: list[dict], needed: int, seed: int) -> list[str]:
    chosen: list[str] = []
    for other in sorted(pool, key=lambda o: _rank(seed, row["Term group ID"] + ":" + o["Term group ID"])):
        meaning = _clean(other.get("Meaning"))
        if not meaning or overlap(meaning, gold) >= SIMILAR:
            continue
        if any(overlap(meaning, c) >= SIMILAR for c in chosen):
            continue
        chosen.append(meaning)
        if len(chosen) == needed:
            break
    return chosen


def draft_row(row: dict, community: str, pool: list[dict], seed: int) -> dict:
    term, gold = _clean(row["Term"]), _clean(row["Meaning"])
    usage, popular = _clean(row["Example usage"]), _clean(row.get("Popular meaning"))
    flags = []
    if not popular or _NO_POPULAR.match(popular):
        flags.append("popular_meaning_missing")
        popular = ""
    elif overlap(popular, gold) >= 0.5:
        flags.append("popular_close_to_gold")
    others = _pick_distractors(row, gold, pool, 2 if popular else 3, seed)
    distractors = ([popular] if popular else []) + others
    if len(distractors) < 3:
        flags.append("too_few_distractors")
    distractors += [""] * (3 - len(distractors))
    options = [gold] + [d for d in distractors if d]
    if any(overlap(a, b) >= SIMILAR for i, a in enumerate(options) for b in options[i + 1:]):
        flags.append("options_too_similar")
    names = community_tokens(community)
    if names and any(names & set(_TOKEN.findall(option.lower())) for option in options):
        flags.append("options_name_community")
    if term.casefold() not in usage.casefold():
        flags.append("usage_missing_term")
    if _DEFINES.search(usage) or overlap(usage, gold, drop=set(_TOKEN.findall(term.lower()))) >= 0.35:
        flags.append("usage_may_define_term")
    return {"id": "p-" + row["Term group ID"], "keep": "", "term": term, "community": community,
            "text_context": usage, "gold_meaning": gold,
            "distractor_1": distractors[0], "distractor_2": distractors[1], "distractor_3": distractors[2],
            "distractor_1_source": "popular_meaning" if popular else "same_scope_meaning",
            "image_file": "", "image_source": "", "text_explicitness": "", "visual_groundability": "",
            "flags": ";".join(flags), "notes": "", "term_group_id": row["Term group ID"],
            "occurrence_id": row.get("Occurrence ID", ""), "source_url": row.get("Source URL", ""),
            "selection_basis": row.get("Selection basis", "")}


def draft_review(graph_path: str | Path, scopes, out_path: str | Path, seed: int = 42) -> dict:
    """Write a review CSV for every gaming term in ``scopes``; refuses to overwrite."""
    out = Path(out_path)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    rows = load_gaming_rows(graph_path)
    known = scope_counts(rows)
    unknown = [s for s in scopes if s not in known]
    if unknown:
        raise ValueError(f"Unknown scopes {unknown}; list them with --list-scopes")
    drafted, seen = [], set()
    for scope in scopes:
        members = [r for r in rows if scope in (r.get("game_or_context_scopes") or [])]
        for row in sorted(members, key=lambda r: r["original_row_index"]):
            if row["Term group ID"] in seen:  # multi-scope terms go to the first listed scope
                continue
            seen.add(row["Term group ID"])
            pool = [o for o in members if o["Term group ID"] != row["Term group ID"]]
            drafted.append(draft_row(row, scope, pool, seed))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(drafted)
    flags = Counter(f for d in drafted for f in d["flags"].split(";") if f)
    return {"rows": len(drafted), "by_community": dict(Counter(d["community"] for d in drafted)),
            "flags": dict(sorted(flags.items())), "seed": seed, "out": str(out)}


def _image(images_dir: Path, name: str) -> tuple[Path | None, str | None]:
    if not name:
        return None, "image_file is empty"
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        return None, f"image_file must be relative to --images without '..': {name!r}"
    path = (images_dir / relative).resolve()
    if images_dir not in path.parents:
        return None, f"image_file resolves outside --images: {name!r}"
    if not path.is_file():
        return None, f"image not found: {name!r}"
    if path.suffix.lower() not in MIME_TYPES:
        return None, f"unsupported image type {path.suffix!r}: {name!r}"
    size = path.stat().st_size
    if not 0 < size <= MAX_IMAGE_BYTES:
        return None, f"image must be 1..{MAX_IMAGE_BYTES} bytes: {name!r}"
    return path, None


def build_items(review_path: str | Path, images_dir: str | Path, out_path: str | Path) -> dict:
    """Validate kept review rows and write items JSONL; refuses to overwrite."""
    out, images = Path(out_path), Path(images_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    with Path(review_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REVIEW_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Review sheet lacks columns {sorted(missing)}")
        rows = list(reader)
    errors, items, hashes, kept_communities = [], [], {}, set()
    for number, row in enumerate(rows, 2):
        if _clean(row["keep"]).casefold() not in KEEP_VALUES:
            continue
        where = f"row {number} ({_clean(row['id']) or '?'})"
        values = {key: _clean(row[key]) for key in REVIEW_COLUMNS}
        kept_communities.add(values["community"])
        for key in ("id", "term", "community", "text_context", "gold_meaning",
                    "distractor_1", "distractor_2", "distractor_3"):
            if not values[key]:
                errors.append(f"{where}: {key} is empty")
        options = [values[k] for k in ("gold_meaning", "distractor_1", "distractor_2", "distractor_3")]
        if all(options) and len({o.casefold() for o in options}) != 4:
            errors.append(f"{where}: gold and distractors must be four distinct options")
        for key, allowed in STRATA.items():
            if values[key].casefold() not in allowed:
                errors.append(f"{where}: {key} must be one of {list(allowed)}")
        path, problem = _image(images, values["image_file"])
        if problem:
            errors.append(f"{where}: {problem}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            errors.append(f"{where}: same image as {hashes[digest]}")
        hashes[digest] = where
        items.append({"id": values["id"], "term": values["term"], "community": values["community"],
                      "text_context": values["text_context"],
                      "image_path": os.path.relpath(path, out.parent.resolve()),
                      "gold_meaning": values["gold_meaning"],
                      "distractor_meanings": options[1:],
                      "metadata": {"gold_status": "provisional_single_reviewer",
                                   "text_explicitness": values["text_explicitness"].casefold(),
                                   "visual_groundability": values["visual_groundability"].casefold(),
                                   "distractor_1_source": values["distractor_1_source"],
                                   "draft_flags": [f for f in values["flags"].split(";") if f],
                                   "image_source": values["image_source"], "image_sha256": digest,
                                   "term_group_id": values["term_group_id"],
                                   "occurrence_id": values["occurrence_id"],
                                   "source_url": values["source_url"],
                                   "selection_basis": values["selection_basis"]}})
    if not kept_communities:
        errors.append("no rows are marked keep")
    elif len(kept_communities) < 2:
        errors.append("kept rows need at least two communities (A4 uses other-community images)")
    if errors:
        raise ValueError(f"{len(errors)} problem(s) in the review sheet:\n" + "\n".join(errors))
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    partial.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    try:
        build_stimuli(load_items(partial), CONDITIONS, "mcq")  # MCQ options and A4 images resolve
    except Exception:
        partial.unlink()
        raise
    partial.rename(out)
    return {"items": len(items), "by_community": dict(Counter(i["community"] for i in items)),
            "strata": {key: dict(Counter(i["metadata"][key] for i in items)) for key in STRATA},
            "stimuli_per_task": len(items) * len(CONDITIONS), "out": str(out)}
