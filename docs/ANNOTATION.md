# Annotation and evidence guide

One JSONL row links a proposed media interval to one existing inventory sense. An exported template is a candidate, never a label inferred from a term, filename or observed action. The saved inventory/source graph remains unchanged. Keep original spellings and usage quotes; do not silently replace them with a normalized glossary definition.

## Select and review

1. Identify the exact game/build, mechanic and original `sense_id`, `term_group_id`, `scope_id` and `occurrence_ids`. Review the saved usage, rather than transferring a franchise/general-genre label to every title. For example, AoE II DE FC and Tekken Full Crouch require separate senses. The ten pilot families in `configs/pilot.json` are collection targets, not accepted alias equivalences.
2. Index a small local media subset. Create a selections object mapping an indexed sample to an existing sense, then export a template:

```json
{"ACTUAL_SAMPLE_ID":{"sense_id":"ACTUAL_EXISTING_SENSE_ID","episode_id":"episode:local-001","category":"positive"}}
```

```sh
python -I -B run.py export-annotations --index runs/local-index.jsonl --selections runs/selections.json --output runs/candidate-annotations.jsonl
```

Use real IDs from the index and inventory, not these placeholders. Without an explicit selection or a sample's sense ID, export retains an unresolved template. It does not select senses by lexical similarity.

3. Choose a short, end-exclusive interval in seconds. Verify frame/tick-to-seconds conversion, event visibility and any replay/render offset. Keep both the native timebase and the encoded media provenance in the acquisition ledger. A still image or temporal summary cannot establish an unobserved action sequence.
4. Write the public observation and separate protected labels. Two distinct reviewers annotate independently before seeing each other's decisions. Retain disagreement and a distinct third adjudicator's resolution. Do not populate reviewer IDs for automated mock fixtures as if they were human judgments.

## Row fields

| Field | Meaning / review requirement |
|---|---|
| `episode_id`, `sample_id` | Stable, nonempty IDs; each annotation episode ID is unique. |
| `sense_id`, `term_group_id`, `scope_id`, `occurrence_ids` | Existing canonical inventory/source joins. At least one actual occurrence is required. Source membership is candidate evidence, not independently adjudicated meaning. |
| `term`, `aliases`, `game`, `version` | Actual scoped expression, reviewed aliases, exact title/build. Keep source wording; unknown provenance is not repaired by guessing. |
| `source_ids` | Nonempty provenance-component IDs used by the split. Preserve all source/derivation connections that could leak between partitions. |
| `replay_id`, `session_id`, `parent_ids` | Replay/session and derivative lineage; every crop/render of the same original must remain connected. Use globally unique namespaced IDs, such as `replay:001`, since identity strings are joined across fields. |
| `clip` | `{"start_seconds":0.0,"end_seconds":2.0}` in the recorded source clock. Gold event intervals must lie inside it. |
| `category` | `positive`, `near_miss`, or `wrong_context`. These are sampling strata, not substitutes for the reviewed label. |
| `visibility_status` | Optional default visibility assessment. `unknown` or `out_of_view` requires reviewed `INSUFFICIENT`; a condition's `gold.evidence_status` can give its specific assessment. Review the evidence actually supplied to each condition. |
| `public` | Allowlisted observation, visual references, plain description and optional action choices. No gold, reviewer notes or explanatory source answer. |
| `reviews`, `adjudication` | Independent decisions and any third-party resolution, as below. Human responsibility is not verified merely by passing the schema. |
| `gold` / `gold_by_condition` | Shared protected rubric after an information-match review, or six explicit condition rubrics. Keep outside agent-visible input. |

The split groups connected source/replay/session/parent/sample/episode identities, including unscored intermediate rows. Do not change shared IDs to manufacture independence. Large connected components can reduce the achievable development/evaluation balance. Usage `occurrence_ids` link lexical evidence; record media-source lineage independently and document shared textual-source exposure where it could leak answers.

Public fields have this shape:

```json
{
  "evidence_text": "REVIEWED OBSERVATION WITHOUT THE ANSWER",
  "visual_refs": [{"ref":"LOCAL_RESOLVABLE_MEDIA_REFERENCE","timestamp_seconds":0.5,"media_type":"image"}],
  "plain_description": "MEANING-MATCHED DESCRIPTION OF THE REQUESTED MECHANIC",
  "action_options": []
}
```

Visual references are prepared input references, not proof that bytes were decoded or supplied to a vision model. Verify that they resolve to the intended real pixels, fit the 64-frame window cap, and preserve the defining transition. The offline planner does not run a visual model. A plain description identifies the queried mechanic; it must not assert that the example contains it. Match specificity across jargon/plain prompts.

For decoded frame lists, the model-facing selection API is explicit:

```python
from multimodal_game_eval.sampling import select_frames

selected, transformation = select_frames(window.frames, max_frames=64, policy="preserve")
```

The default refuses a window over budget. If subsampling is justified, explicitly choose `policy="uniform"` and supply `required_indices` for reviewed defining native frames. The returned ledger records retained native indices/timestamps and applies no resizing or interpolation. Save it with the annotation/export provenance. This function does not determine which frames define the term; expert term-identifiability review remains required.

## Labels and evidence sufficiency

- `PRESENT`: the supplied evidence establishes the named mechanic under the agreed criteria.
- `ABSENT`: the supplied evidence establishes that the mechanic's criteria are not met.
- `INSUFFICIENT`: the supplied evidence cannot settle the decision, including an out-of-view or unobservable defining property.

A negative label requires evidence of absence under observable criteria; missing information is not an automatic negative. The underlying event can be present while a modality-specific answer is insufficient. Do not mark a text-only or visual-only model wrong for information its condition never received. Information matching and condition-specific gold are described below.

Define a canonical property vocabulary before scoring, for example a dictionary with finite scalar values or a list of canonical strings. Publish the response convention without exposing each example's protected values. The scorer uses exact canonical agreement: it does not recognize arbitrary paraphrases or validate natural-language semantics. Missing properties count as misses when gold properties exist.

Shared protected gold uses:

```json
{
  "label": "PRESENT",
  "critical_properties": {"CANONICAL_PROPERTY_KEY":"CANONICAL_VALUE"},
  "event_interval_seconds": [0.2,0.5],
  "valid_action_ids": null
}
```

This is a schema illustration, not a reviewed example. An event interval is optional and is meaningful only for an observable positive event. Set it to `null` where timing cannot be established. Set `valid_action_ids` to `null` when no independent action criterion exists. If provided, IDs must be among `public.action_options`, with explicit legal choices and a recorded rationale/validator. Replay commands or expert demonstrations alone do not prove optimal action.

Each approving review requires `reviewer_id`, `independent:true`, `decision:"approve"` and `condition_labels` mapping all six exact condition names to reviewed labels. Two distinct independent approving reviewers must agree with every protected arm. If they disagree, retain both judgments and provide `adjudication` with a distinct third `reviewer_id`, `decision:"approve"` and `resolved_condition_labels` mapping all six arms to labels matching gold. An empty review list remains unscored. Schema checks cannot establish actual reviewer independence or quality.

## Match information or supply condition-specific gold

Shared `gold` is permitted only when the annotation row has `information_match_verified:true` and both independent approving reviews also have `information_match_verified:true`. Their six `condition_labels` must match the shared label. Reviewers must actually inspect each agent-visible evidence arm, including visibility, text specificity, frame selection and omitted state. An identical event is not sufficient evidence of matched information.

If sufficiency differs, set row `information_match_verified:false` and supply `gold_by_condition` with **all six** complete rubric objects, using the same fields as the protected gold illustration. For example, an arm that cannot reveal a defining property can use `label:"INSUFFICIENT"`, an empty property dictionary, null timing/actions and `evidence_status:"out_of_view"`. Reviewers independently label all six arms; do not automatically assign insufficiency from missing metadata.

For each modality, jargon and plain rubrics must agree on label, critical properties, event interval and valid action IDs. The wording contrast cannot change the event's reference answer. Missing arm rubrics, unreviewed information matching and mismatched wording-pair gold fail preparation. Unequal-information cases remain diagnostic; they are not eligible evidence for the claimed lexical-effect contrasts. A row's boolean claim is no substitute for a documented review.

## Preparation and leakage boundaries

```sh
python -I -B run.py prepare --annotations runs/adjudicated-annotations.jsonl --seed 11 --dev-fraction 0.25 --output runs/pilot-001
```

Preparation writes `episodes.jsonl`, `inputs.jsonl`, protected `gold.jsonl`, `unscored.jsonl`, and `manifest.json` with hashes and reasons. Inspect actual eligible counts, independent components, split proportions and exclusions. Rejected/unreviewed rows do not receive guessed scores. A 25% split is over connected lineage groups, and terms intentionally recur in both partitions.

An inference implementation should pass **only each `inputs.jsonl` row's `input` object** to an agent. The outer registry includes evaluation metadata. Do not provide the annotation row, inventory definitions, occurrence quotes, source notes, protected gold or manifest with answers. Public preparation uses an explicit allowlist; it cannot detect answers embedded inside an apparently harmless image, transcript, filename, plain description or action option. Review those manually.

## Saved responses and events

One response JSONL file represents one replication, with at most one response per planned `item_id`:

```json
{"item_id":"ACTUAL_ITEM_ID","label":"INSUFFICIENT","critical_properties":{},"event_interval_seconds":null,"action_id":null,"confidence":0.5}
```

Answer fields can instead be nested under `answer`. Confidence is optional and must be a finite probability. Import separate files for separate seeds/regimes. Missing, duplicate and malformed responses are errors rather than silently discarded successful cases. `score` measures agreement with the reviewed reference, not a live simulator reward.

Interaction logs are ordered JSONL. Every row needs `event_id`, `episode_id`, and `type`; use a global increasing integer `sequence` when supplied. Core types are:

| Type | Required linkage |
|---|---|
| `episode_start` | `agent_ids`, stable `community_id`, and `forum_exposure` for the assigned forum condition |
| `observation` | `agent_id`, `observation_id`; optional explicit `visible_to` |
| `message_sent` | `sender`, `message_id`, `recipients`, and referenced `observation_ids` |
| `message_delivered`, `message_read` | `recipient`/`agent_id`, `message_id`; read follows actual delivery |
| `forum_post`, `forum_read` | `message_id`, `community_id`, permitted audience and explicit read exposure; `visibility:"community"` permits future members of that community |
| `decision`, `memory_update` | `agent_id`, referenced `observation_ids` and `message_ids` |
| `newcomer_join`, `episode_end` | Explicit membership transition / episode boundary; `replace_agent_id` identifies a removed incumbent when replacing one |

Also preserve actual content, channel, timing, visible payloads, actions and verified outcomes for analysis; reference checks alone cannot recover them. A newcomer sees only prescribed observations and logged deliveries/forum reads. Explicit-recipient posts remain private to their audience; community-visible posts can be read by future same-community members. Withheld-forum episodes must not log forum reads. `validate-events` checks documented access/order/reference consistency. It cannot discover unlogged leakage, authenticate records or establish semantic grounding/diffusion.
