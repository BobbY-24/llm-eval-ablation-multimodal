# Research design: API ablation study (part A)

Part A asks how model choice and context change what LLM APIs get right about community-specific terms ("jargon" whose meaning depends on a game or technical community). It has two studies:

1. **Provider comparison (completed, historical).** Extraction-and-judgment pipelines on one fixed cohort of 50 frozen web sources. Aggregate results are in `ablation/results/`.
2. **Context ablation A0–A4 (implemented, not yet run live).** One term-meaning prompt, five evidence conditions, paired per item. Code is in `ablation/`. No live results exist yet; the repository ships only a synthetic fixture and offline tests.

## Research questions

| ID | Question | Study |
|---|---|---|
| RQ1 | On identical inputs, how do extractor and judge choices across providers trade known cost, candidate-pool size, literal-grounding fidelity and operational completeness? | Provider comparison |
| RQ2 | How much do community labels and usage text improve meaning identification over the bare term (A1 − A0)? | A0–A4 |
| RQ3 | Does an on-topic image add information beyond text (A3 − A1), and how much can an image alone recover (A2 − A0)? | A0–A4 |
| RQ4 | Does a misleading image hurt a model that already has the right text (A1 − A4)? Over-reliance on vision shows up as a positive penalty. | A0–A4 |
| RQ5 | Do these effects differ by model/provider and by community? | Both |

## Study 1: provider comparison on 50 frozen sources

**Cohort.** 25 Gaming and 25 Cybersecurity web pages, frozen once (921,230 characters, 352 root chunks) and verified byte-identical across runs. Chunks are 3,000 characters with 250 overlap. Extraction is capped at 64 candidates per chunk, and judgment runs in batches of 16.

**Pipeline.** An extractor model proposes candidate term occurrences with evidence. Local code checks that each expression and quote appear literally in the source ("grounding"). A judge model labels the surviving candidates KEEP / REVIEW / REJECT.

**Compared configurations.** All four models were tested on Fireworks, except Qwen, which was tested on Alibaba Model Studio (Singapore).

| Configuration | What varies |
|---|---|
| DeepSeek V4.1 Flash → GLM 5.3 Flash | Extractor (reasoning off) |
| GLM 5.3 Flash → GLM 5.3 Flash | Extractor (1,024-token reasoning budget) |
| GLM pool → Nemotron 3.5 Lightning judge | Judge only; GLM proposals reused byte-for-byte |
| GLM pool → Qwen 3.7 Flash judge | Judge only; two-source smoke test |
| GLM → Nemotron screen + GLM fallback | Routing policy; also changed extraction recovery |

**Measures.**
- Known cost: provider-reported tokens times the saved list price.
- Unknown-outcome holds: reserved but unresolved billing.
- Source completion: complete / partial / failed.
- Raw proposals and distinct normalized forms.
- Local grounding rejects.
- Judge coverage: the share of eligible IDs that received a valid judgment.
- Per-request latency, where telemetry exists.

**Main findings.**
- GLM extraction cost 28.0% less than DeepSeek and proposed 18.1% more occurrences, mostly in Cybersecurity. That sent 38.1% more occurrences to the judge, so judgment cost 37.2% more. The net saving was only 1.3% (about $1.95 against $1.98).
- DeepSeek had fewer literal-grounding rejects: 1.6% against 4.1%.
- Each extractor proposed forms the other missed: 6,098 shared normalized forms, 2,555 DeepSeek-only and 4,331 GLM-only.
- The cheaper judges (Nemotron and Qwen) halted early on unknown outcomes or provider rejections. They validated 17% and 50% of their eligible IDs, so their low costs are not completed-pipeline costs.

**What it cannot show.** There is no occurrence-level gold set, so no precision or recall is reported. More proposals do not imply better recall. KEEP is a model verdict, not a correctness label.

## Study 2: context ablation A0–A4

### Conditions

Each item is run under every condition, and contrasts are paired within item, model and task.

| Condition | Term | Community | Usage text | Image |
|---|:-:|:-:|:-:|---|
| A0 | ✓ | | | |
| A1 | ✓ | ✓ | ✓ | |
| A2 | ✓ | ✓ | | the item's image |
| A3 | ✓ | ✓ | ✓ | the item's image |
| A4 | ✓ | ✓ | ✓ | another item's image, preferably from a different community |

A0 omits the community on purpose, so it measures the model's prior for the bare string. The A4 distractor is chosen deterministically from the seed and item ID.

### Tasks

- **MCQ (primary).** The gold definition plus three distractor definitions, shuffled deterministically per item. The answer key never crosses the inference boundary. Scoring is exact.
- **Freeform (secondary).** The model writes a definition. The built-in score is token F1 against the gold definition, which is only a lexical proxy. Reportable freeform results need a rubric or a judge model calibrated against human labels on a held-out subset.

### Protocol controls

- One prompt template for all conditions. Only the allowlisted stimulus fields are serialized. Item IDs, condition names, file paths and gold labels never appear in prompts; tests enforce this.
- The prompt tells the model to treat stimulus text and images as data, not instructions.
- Temperature is 0. The response must be bare JSON with exactly `selected_option` or `predicted_meaning`, `confidence` and `evidence_used`.
- An invalid response gets at most one repair attempt with the same stimulus. A transport or provider error is recorded and never retried as a repair.
- Live calls need `--live` and a hard `--max-calls`, with repairs counted. Stimuli left unrun when the budget runs out stay in the output as `not_run_budget`.
- Every attempt is appended to `attempts.jsonl` before the next call. Keys come only from a named environment variable and are never logged or placed in URLs.

### Metrics

- **Per condition:** accuracy over scored predictions, and accuracy over all labeled attempts, where failures count as wrong. Report both, plus parse-failure, repair and inference-failure rates.
- **Paired contrasts:**

  | Contrast | Formula |
  |---|---|
  | context gain | A1 − A0 |
  | multimodal gain | A3 − A1 |
  | text gain | A3 − A2 |
  | image-only gain | A2 − A0 |
  | distractor penalty | A1 − A4 |

  Each contrast reports its mean, a percentile bootstrap 95% CI over items, the share of positive pairs and its pair coverage. Pairs with a missing score are dropped and counted, never imputed. Contrasts with n < 30 are flagged.
- **Operational:** calls, reported tokens, cost at the stated price, and latency per condition. Image conditions cost more input tokens.
- **Diagnostics only:** self-reported `evidence_used` and `confidence`. Calibration is reported only when n is large enough.

### Pilot run (descriptive, $0)

The first live run uses about 40 reviewed items across four communities (the default draft scopes: Super Smash Bros. Melee, Minecraft, Dota 2, Age of Empires II), one free-tier Gemini model and at most 260 calls. Items come from the bundled usage graph. Its community meaning is the gold label, its popular meaning is one distractor, and two other terms from the same scope supply the rest. These are model-generated proposals, so a single reviewer must confirm, correct or drop each row before it becomes an item (`gold_status: provisional_single_reviewer`). The reviewer also tags text explicitness and visual groundability for descriptive breakdowns.

Primary contrasts, fixed before the run: context gain (A1 − A0), multimodal gain (A3 − A1) and distractor penalty (A1 − A4), i.e. RQ2–RQ4. At n ≈ 40 their intervals will be wide. The pilot checks the pipeline and gives a first effect-size estimate; it does not support a power-based or confirmatory claim.

### Sample size

For paired binary outcomes with about 30% discordant pairs, detecting a 10-point accuracy difference at α = 0.05 (two-sided) with 80% power needs about 235 items per model (McNemar approximation). A 5-point difference needs roughly four times as many. Pilots smaller than this are descriptive.

### Data

The public repository contains only `ablation/fixtures/synthetic_items.jsonl`: four invented terms from two fictional communities, with solid-colour placeholder images. It tests plumbing, not models, and mock-provider scores are meaningless by design. Real items (term, community, usage text, image, gold and distractor definitions) are supplied locally and are not distributed here.

## Threats to validity

**Construct validity**
- MCQ difficulty depends on distractor quality. Weak distractors inflate every condition and compress the contrasts.
- Token F1 rewards lexical overlap, not meaning. Treat it as a smoke metric.
- `evidence_used` is a self-report, not an attribution of which evidence the model actually used.
- In the provider study, "candidate retained" is not "correct term". There is no gold set.

**Internal validity**
- Usage text can state the meaning outright. That makes A1 near ceiling and hides any image effect. Report items by how informative their text is, or include text that is deliberately less explicit.
- A4 distractors from another community may be obviously off-topic. This measures resistance to irrelevant images, not to plausible wrong ones. Hard negatives from the same community are a planned extension.
- Even at temperature 0, providers can be nondeterministic, and model versions behind an alias can change. Record the exact model ID and date, and rerun a fixed subset to estimate run-to-run variance.
- Content filters and HTTP errors can remove items non-randomly. Qwen's provider-side inspection rejection in Study 1 is an example. Report missingness by condition, and use accuracy over all attempts as the conservative figure.
- The repair prompt differs from the original prompt. Report repair rates per condition and check that conclusions hold when repaired responses are excluded.
- Study 1 compares model-plus-setting combinations: reasoning differed between extractors, and the guarded run changed recovery depth. Differences are not attributable to the model alone.

**External validity**
- Two communities, English text and a small item set: effects may not transfer to other communities, languages or rarer terms.
- The fixture is fictional, so it shows nothing about real terms.
- Costs and latencies are specific to the provider, region and price list on the stated dates.

**Statistical conclusion validity**
- Five contrasts per model invite multiple-comparison false positives. Pre-register RQ2–RQ4 as the primary contrasts and treat the rest as exploratory.
- Items cluster by community. Report community-level results and, where n allows, bootstrap clustered by community.

**Contamination**
- Public terms and definitions may be in pretraining data, which inflates A0. A0 is reported for exactly this reason. A rising A0 across model versions is a contamination signal, not a context effect.
