# Staged experiment protocol

This protocol separates tooling checks, semantic grounding and dynamic community behavior. The current package implements offline preparation, finite plans, synthetic mock demonstration and scoring of imported results. It does not run a model provider, a live game, a renderer or parameter training.

## Stage 0: plumbing and provenance

Run the synthetic demo and inspect its generated records. Check unknown/duplicate IDs, unsafe paths, time units, missing modalities, source joins and refused overwrites. Label all synthetic fixtures and mock scores as synthetic. Passing this stage supports software integrity only.

For real data, preserve the original inventory, source graph and acquired media. Review each sense with saved usage quotes; record original and transformed text separately. Complete media rights/provenance and independently check timing. Split all derivative clips from the same replay/session together. Keep near-duplicate recordings and examples used to produce prompts in the same split.

## Stage 1: 200 annotated examples

Target these ten candidate sense families. Alias grouping requires expert review and preserves the original sense IDs:

| Exact title | Candidate families |
|---|---|
| Super Smash Bros. Melee | wavedash; L-cancel; dashdancing; short hop; edgehogging |
| Age of Empires II: Definitive Edition | FC; Trush; deer pushing; drush; 3 TC booming |

Collect 20 examples per family: ten positive, five near-miss and five wrong-context. A near-miss fails a defining property while sharing relevant context. A wrong-context example uses the wrong game, sense, actor or task. It is not evidence that the vocabulary itself is invalid. Uncertain examples are withheld until adjudicated.

Use independent annotations followed by adjudication, with annotator IDs and reasons retained. Ground critical properties in visible evidence or verified state, not a glossary definition alone. Two humans agreeing does not replace executable evidence where the task permits it. Observed player commands do not establish optimality; optimal-action labels require a stated objective, legal-action set and independent validator or expert rationale.

Reserve 25% of lineage groups for development and 75% for evaluation, not individual frames. Equal-sized groups would yield about 50 development and 150 evaluation examples; group sizes can prevent exact ratios or per-family balance. Report actual counts and ensure each intended family/label stratum is represented. This is a within-family generalization test, not held-out-sense or cross-game transfer. Freeze the evaluation annotations and prompt templates before importing model results.

## Stage 2: six grounding cells

| Condition | Wording | Evidence |
|---|---|---|
| `jargon_text` | Community term | Text-only observation |
| `jargon_visual` | Community term | Visual observation |
| `jargon_both` | Community term | Text and visual observation |
| `plain_text` | Meaning-matched plain description | Text-only observation |
| `plain_visual` | Meaning-matched plain description | Visual observation |
| `plain_both` | Meaning-matched plain description | Text and visual observation |

The visual cells still contain a question/instruction; “visual” specifies the observation channel. Plain descriptions must preserve the requested meaning, specificity and task difficulty without disclosing the gold label. Text observations must describe the same underlying event without adding privileged state or a definition unavailable visually. Keep gold, annotator notes, source explanations and critical-property answers outside agent-visible payloads.

Judge what each evidence arm can support. If a mechanic is visible only through telemetry, the image-only arm can correctly be `INSUFFICIENT` even when the underlying event happened. Use independently reviewed condition-specific gold for unequal-information cases; a shared gold label requires an explicit verified information match across conditions. Unequal-information cases are diagnostic, and must be excluded from paired lexical-effect claims. See the annotation guide for the exact review fields.

Pair all six cells on the same underlying example and use seeds 11, 23 and 37. Cap visible media at 64 frames per window; record the deterministic selection rule and retained native timestamps. Do not erase a short defining transition by uniform subsampling. If 64 frames cannot preserve the property, choose a shorter window or explicit key events and document that choice. Audio is outside v1.

The frozen grounding-only design is `configs/stage2_vlm.json` with `configs/stage2_prompts.json`. It asks for labels with the jargon or plain wording, then asks critical properties in a separate request with neutral wording shared by both arms, so property keys cannot define the jargon term.

Score sense/label recognition and critical-property agreement; score next-action success only where a real legal-action/outcome validator exists. Report coverage and missing responses. V1 validates optional confidence probabilities but does not compute calibration; analyze calibration separately if collected. Aggregate by family and lineage group, with paired differences for jargon/plain and modalities. Use 1,000 bootstrap resamples at the independent group level; correlated frames or turns are not independent observations. Inspect confusion cases before claiming semantic understanding.

## Stage 3: peer communication and newcomer transfer

Plan three agents, six rounds and 18 turns per community episode; each turn is capped at 256 output tokens. Input/context usage needs a separate recorded budget. Run ten episodes per community. After five completed episodes, introduce the newcomer before episode six. The protocol must state which incumbent is replaced, which observations/memory the newcomer receives, and whether a persistent forum is enabled. Keep local chat and forum exposure logs separately. The finite plan is not an execution of this experiment.

The planner itemizes standalone grounding requests separately from team schedules and newcomer probes. It pairs `enabled` and `withheld` forum assignments for the same episode sequence and assigns the `independent`, `fixed_workflow` and `peer` regimes a matched maximum turn allowance. Incomplete ten-episode windows are flagged. These assignments define a proposed contrast; a run must enforce them and record actual exposure and resource use before the contrast is observable.

Newcomer probe slots contain no invented inputs. Before a live run, assign expert-reviewed fresh examples whose source components were not exposed to that community. Resolve the planner's input-context and provider-price assumptions as well as its known request/output/media ceilings. A null dollar estimate means unknown cost, and is not a free-run estimate.

For an executed study, assign independent community/session IDs. Log sender, recipients, content, channel, sequence/time, model/build, observations, visible history, forum reads, actions and externally checked outcomes. Give agents asymmetric observations only in a declared condition. The newcomer should receive exactly the prescribed public/peer evidence rather than gold definitions or hidden incumbent memory.

Predeclare controls: no peer communication; equal-budget plain descriptions; shuffled or delayed exposure; community-local nonce terms; and newcomer with versus without the relevant forum history. Keep model, observation budget, turn budget and task distribution matched. Replicate across independent communities: adoption within one shared transcript is not a population-level diffusion result. Static video datasets supply task evidence, not the causal exposure history.

Measure first use, first exposure, reuse after exposure, referent consistency and action success separately. A new string is lexical coinage; repeated strings are reuse; successful referent-sensitive behavior on counterexamples and newcomer tasks is stronger grounding evidence. Neither reuse nor agreement proves emergent semantics. Exclude direct copying, prompt-provided shorthand and shared prior knowledge as alternative explanations before attributing conventions to interaction.

## Stage 4: adaptation after the community test

First compare frozen agents with explicit, logged memory/retrieval or prompt updates on development episodes. Call these changes memory or harness adaptation. Parameter self-training is a separate future stage requiring training provenance, frozen evaluation, independent reward/verification, retained old-task probes and contamination checks. Improved scores after viewing evaluation examples cannot establish generalization.

Stop each offline run at its planned finite limits. Missing real media, adjudication, provider responses, event logs or executable outcome checks remain missing evidence, not scores to impute. Report adapter readiness, synthetic smoke checks, real grounding results and community results separately.
