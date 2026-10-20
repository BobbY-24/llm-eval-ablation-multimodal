"""Items and paired A0-A4 stimuli. Gold labels never enter a Stimulus."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

CONDITIONS = ("A0", "A1", "A2", "A3", "A4")
CONDITION_NAMES = {
    "A0": "term only",
    "A1": "term + community + usage text",
    "A2": "term + community + image",
    "A3": "term + community + usage text + image",
    "A4": "term + community + usage text + distractor image",
}
TEXT_CONDITIONS = frozenset({"A1", "A3", "A4"})
IMAGE_CONDITIONS = frozenset({"A2", "A3", "A4"})
TASKS = ("freeform", "mcq")
_ITEM_FIELDS = {"id", "term", "community", "text_context", "image_path", "gold_meaning",
                "distractor_meanings", "metadata"}


@dataclass(frozen=True)
class Item:
    id: str
    term: str
    community: str
    text_context: str | None = None
    image_path: str | None = None
    gold_meaning: str | None = None
    distractor_meanings: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Stimulus:
    """Everything a provider may see, plus tracking fields that are never serialized."""
    example_id: str
    condition: str
    term: str
    community: str | None
    text_context: str | None = None
    image_paths: tuple[str, ...] = ()
    image_source_ids: tuple[str, ...] = ()
    choices: dict[str, str] | None = None


def _seed(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")


def load_items(path: str | Path) -> list[Item]:
    """Read JSONL items; image paths resolve relative to the items file."""
    path = Path(path)
    items, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict) or set(raw) - _ITEM_FIELDS:
            raise ValueError(f"{path}:{number}: unknown item fields {sorted(set(raw) - _ITEM_FIELDS)}")
        for key in ("id", "term", "community"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise ValueError(f"{path}:{number}: {key} must be a nonempty string")
        if raw["id"] in seen:
            raise ValueError(f"{path}:{number}: duplicate id {raw['id']!r}")
        seen.add(raw["id"])
        image = raw.get("image_path")
        if image is not None:
            image = str((path.parent / image).resolve()) if not Path(image).is_absolute() else image
        distractors = tuple(raw.get("distractor_meanings") or ())
        if distractors and len(distractors) != 3:
            raise ValueError(f"{path}:{number}: MCQ needs exactly three distractor_meanings")
        items.append(Item(id=raw["id"], term=raw["term"], community=raw["community"],
                          text_context=raw.get("text_context"), image_path=image,
                          gold_meaning=raw.get("gold_meaning"), distractor_meanings=distractors,
                          metadata=raw.get("metadata") or {}))
    return items


def mcq_options(item: Item, seed: int = 42) -> tuple[dict[str, str], str]:
    """Shuffle gold plus three distractors deterministically; return (choices, key)."""
    if not item.gold_meaning or len(item.distractor_meanings) != 3:
        raise ValueError(f"{item.id}: MCQ needs gold_meaning and three distractor_meanings")
    options = [(item.gold_meaning, True)] + [(d, False) for d in item.distractor_meanings]
    if len({text.casefold().strip() for text, _ in options}) != 4:
        raise ValueError(f"{item.id}: MCQ options must be distinct")
    random.Random(_seed(seed, item.id + ":mcq")).shuffle(options)
    choices = {label: text for label, (text, _) in zip("ABCD", options)}
    key = next(label for label, (_, correct) in zip("ABCD", options) if correct)
    return choices, key


def pick_distractor(item: Item, pool: list[Item], seed: int = 42) -> Item:
    """Another item's image, preferring a different community; deterministic per seed."""
    candidates = [o for o in pool if o.id != item.id and o.image_path
                  and o.image_path != item.image_path]
    other = [o for o in candidates if o.community != item.community]
    choices = sorted(other or candidates, key=lambda o: o.id)
    if not choices:
        raise ValueError(f"{item.id}: A4 needs another item with a different image")
    return choices[_seed(seed, item.id + ":distractor") % len(choices)]


def build_stimuli(items: list[Item], conditions=CONDITIONS, task: str = "freeform",
                  seed: int = 42) -> list[Stimulus]:
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - set(CONDITIONS):
        raise ValueError("conditions must be unique members of A0..A4")
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    stimuli = []
    for item in items:
        choices = mcq_options(item, seed)[0] if task == "mcq" else None
        for condition in conditions:
            uses_text = condition in TEXT_CONDITIONS
            uses_image = condition in IMAGE_CONDITIONS
            if uses_text and not item.text_context:
                raise ValueError(f"{item.id}: {condition} needs text_context")
            source = pick_distractor(item, items, seed) if condition == "A4" else item
            if uses_image and not source.image_path:
                raise ValueError(f"{item.id}: {condition} needs an image")
            stimuli.append(Stimulus(
                example_id=item.id, condition=condition, term=item.term,
                community=None if condition == "A0" else item.community,
                text_context=item.text_context if uses_text else None,
                image_paths=(source.image_path,) if uses_image else (),
                image_source_ids=(source.id,) if uses_image else (),
                choices=choices))
    return stimuli
