# Data-source check for "Community Term Pipeline.pptx", slides 3–10 (2026-10-10)

Deck: Community Term Pipeline.pptx (private deck, not in this repo) (9 slide files, numbered 1 and 3–10). Slides 3–6 are images of 16 retained terms; slide 7 lists the 9 loader datasets; slides 8–10 describe setup and the extraction pipeline.

## Slide 7 versus the code

All 9 rows match the adapters in multimodal_game_eval (vpt, minerl, csgo, terraria, proact, starcraftimage, slippi, dota2vis, aoe2). Slide 8's statement "real-media validation and model experiments remain pending" is accurate. Slide 8 lists the optional readers; none is installed on this host.

## Slides 3–6: terms versus inventory and multimodal sources

Every term is in GAME_TERM_INVENTORY_2026-10-06. Title mismatches:

- meaty: slide says SF6; inventory scope is Fighting games (general), scope:022.
- abare: slide says GG Strive; inventory scope is scope:022.
- QCF: slide says Tekken 7; inventory scope is scope:022.
- footsies: slide says Tekken 7; inventory scope is Tekken series (scope:021), plus scope:022.

| Term | Game | Loader | Public video + telemetry found | Visible in frames? | Validator route |
|---|---|---|---|---|---|
| meaty, abare, footsies, QCF | SF6 / GGST / Tekken 7 | none | None for these titles. Closest: GGST replay-derived controller logs (ACM, availability unverified); COMBAT Tekken 3 (1.2M frames + inputs, release promised, unverified; wrong title) | QCF: input motion, telemetry_only (input-display overlay = privileged). Meaty: native-rate timing. Abare: partly. Footsies: strategic, not observable in a clip | In-game replays only; no parser |
| BoltBeam, Spinblocker, RestTalk, VoltTurn | Pokémon (Smogon/Showdown) | none | Showdown text battle logs (documented protocol); no hosted dataset found | Weak: turn-based; move names appear as on-screen text, which leaks the answer | Battle log (text); fits text/action tasks, not CV |
| brez, snap threat, pugging, LFR Tank | WoW | none | No paired set. Warcraft Logs (combat logs, API) + Warcraft Recorder (open-source video capture) | brez: visible cast. Snap threat: internal threat, telemetry_only. Pugging/LFR Tank: social, not observable | Combat log |
| science sushi | Factorio | none | Factorio Learning Environment (NeurIPS 2025; programmatic state); piebro blueprint renders (<1K, HF) | Mixed-item loop belt is visible; the dynamic aspect needs video | Blueprint string / FLE state |
| deer push | AoE II | aoe2 | As in SESSION notes | context_dependent | Not integrated |
| hopper clocks | Minecraft | vpt/minerl (general play only) | No redstone-labelled data | Visible build; behaviour needs video | Schematic files (not integrated) |
| sideclimb | War Thunder | none | .wrpl proprietary; Rust `wrpl` crate decodes headers/chat, not positions; live localhost:8111 telemetry → Tacview ACMI (air battles); a Ground RB replay scrape is mentioned with no location | Altitude/heading on HUD; partial | Self-recorded live telemetry |

## Assessment

- Melee/Slippi remains the only source with public frame-exact state, a working loader and visual mechanics.
- Next candidates with an independent validator, all requiring self-recorded captures: WoW combat log + Warcraft Recorder (brez); War Thunder live telemetry + capture (sideclimb); Factorio blueprints/FLE (science sushi, static). Each needs a new adapter, and ToS/rights review for recording.
- Fighting-game terms have no public telemetry. QCF and meaty would be strong jargon tests, but they need a replay/input source we do not have.
- Pokémon terms suit the text and action-selection tasks; they are poor for the visual arm.
