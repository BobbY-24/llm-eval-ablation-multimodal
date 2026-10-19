# Offline communication analysis

`analyze_communication(events_path, seeded_forms, reviewed_probe_results=None)`
reads a local JSONL event log. It makes no model calls and writes no output files.
The access validator runs first. Invalid delivery/read/private-observation logs
return `status: "invalid"`, with no downstream lexical or probe analysis. Explicit
round/turn ceiling violations also stop that analysis.

```python
from pathlib import Path
from multimodal_game_eval.communication import analyze_communication

report = analyze_communication(
    Path("events.jsonl"),
    seeded_forms=["short hop", "wavedash"],
    reviewed_probe_results=None,
)
```

The caller may serialize the returned plain JSON-compatible dictionary, or use:

```sh
python -I -B run.py analyze-communication --events runs/events.jsonl --seeded-forms runs/seeded-forms.json --probes runs/reviewed-probes.jsonl --output runs/communication-analysis.json
```

The seeded-forms file is a JSON string list such as `["short hop", "wavedash"]`.
The optional probes file uses the separate reviewed-outcome JSONL contract below.
Omit either flag when those records are absent. Existing output files require
`--force`; invalid logs return exit status 2 and retain their findings.

## Event fields

Use the event-access contract from `experiments.validate_events`: every event has
`event_id`, `episode_id`, and `type`. Start episodes with `agent_ids`,
`community_id`, and explicit `forum_exposure`. Messages need sender, message ID,
and recipients; deliveries and reads must be logged separately. Community-visible
forum posts use `visibility: "community"`. A future newcomer needs an actual
permitted `forum_read`; posting or joining alone does not establish exposure.

Additional fields used here:

| Field | Meaning |
|---|---|
| `text` or string `content` | Original message text; absent text stays missing. |
| `coined_forms: ["zig zag"]` | Explicit coinage markers on a sent message/forum post. A marker is not proof of invention. |
| `sequence` | Optional ordered sequence from the event log. Without it, sequence lag is unknown. |
| `usage.input_tokens`, `usage.output_tokens`, `usage.total_tokens` | Recorded nonnegative integer counts. `prompt_tokens`/`completion_tokens` aliases are accepted. |
| `cost_usd` or `estimated_usd` | Recorded nonnegative value; the report does not verify billing. |
| `model_call_id` or `call_id` | Optional identifier used to flag repeated accounting records. |
| `status`, `error`, `failure` | Explicit failure records on otherwise valid event types. |
| `round`, `turn` | Optional **one-based** indices. `round_index`/`turn_index` aliases are accepted. |
| Start-event `rounds`/`max_rounds`/`round_cap` | Optional round ceiling; aliases must agree when supplied. |
| Start-event `max_turns`/`turn_cap`/`turns` | Optional turn ceiling; aliases must agree when supplied. |

Only supplied seeded forms and explicit `coined_forms` enter the form registry.
Matching uses Unicode casefolding and word boundaries. Multiword forms remain
phrases, allowing whitespace differences. Surface overlap is counted literally;
it is not sense disambiguation. Characters are original Unicode code points,
not estimated model tokens. Counts are separated into sent, delivered and read
stages; repeated logged deliveries/reads count as recorded events.

`form_records` reports each observed community–agent–form combination's first
production, first actual read, first later production and their event/sequence/
community-episode gaps. It flags production before the first logged read. Missing
reads imply unobserved exposure, not proof that an agent had no prior knowledge.
Reads of an agent's own post are marked `self_authored_read`; they do not establish
peer exposure. Production before an explicit coinage marker is also flagged.
Unseeded forms are called **explicit coinage markers logged in this run**;
arbitrary unfamiliar strings are never automatically classified as novel.

## Separate reviewed probes

```json
{
  "probe_id": "newcomer-probe-1",
  "community_id": "community-1",
  "agent_id": "newcomer-1",
  "reviewer_status": "adjudicated",
  "fresh": true,
  "label_correct": true,
  "action_correct": false
}
```

Probe IDs must be unique. The community must appear in the validated log.
`fresh: true` and `reviewer_status: "adjudicated"` are required; action correctness
is optional and has its own denominator. These are supplied reviewed outcomes,
not labels generated from lexical repetition. With at least two communities
contributing the relevant outcome, deterministic 1,000-resample community-cluster
bootstrap intervals are reported. A single-community result has no such interval.

All analyses are descriptive. They do not establish semantic adoption, language
emergence, causal diffusion, or a communication treatment effect. An experiment
requires independently reviewed probes, randomized exposure and suitable controls.

## Bounds and missing data

Inputs are bounded to 64 MiB and 100,000 records; the registry permits at most
256 forms, each at most 256 characters. Lexical matching work and form-event
observations have additional finite ceilings; exceeding them raises an error
instead of silently truncating. No downloads, pickle loading, tokenizer execution
or external services are used. Token/cost sums contain only recorded values and
may overlap across event stages. Missing values are never replaced with zero-cost
assumptions or character-based estimates.
