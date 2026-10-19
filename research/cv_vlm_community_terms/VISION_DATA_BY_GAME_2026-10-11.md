# Vision data for the 37 game groups in "1000 Terms.xlsx" (2026-10-11)

> **Corrections (2026-10-11, after the literature pass):** StatePlay uses an abstracted 11-dimensional action space and five state variables from "emulator memory", not explicit controller bits or RAM; the "controller inputs + RAM state" wording and the CC BY-NC label are unverified. NitroGen is about 40,000 hours across more than 1,000 games with automatically extracted actions; the "about 30k videos, 164 GB, CC BY-NC" figures are unverified and conflict with the arXiv summary. See the "Corrections to earlier project notes" section of `work/neologism_gaming_visual_lit_2026-10-11/LITERATURE_ANALYSIS_2026-10-11.md` and `bibliography_2026-10-11.csv`.

## Method and status

- The 804 gaming terms were mapped to 37 game groups (`1000_terms_game_map_2026-10-10.csv`).
- A 37-group workflow (two searchers per game plus page-checkers) hit the account session limit twice. Of 465 agent runs, 22 finished, all of them searchers for 11 groups. Their candidates were recovered from the run journal.
- The other groups were covered with direct web searches. TF2 got no successful search because of the search rate limit.
- **Verification:** only two sources were page-checked (marked [checked]). Everything else is searcher- or search-result level. Full list: `vision_data_candidates_2026-10-11.csv` (240 rows, with URL, licence, access, telemetry and verification status).

## Tiers

- **A:** public frames, or replays that render to frames, paired with inputs/state that can validate a term.
- **B:** public frames or video without a validator.
- **C:** self-generation only (emulator, replay renderer or own capture; usually needs a game copy, ROM or ISO).
- **D:** nothing usable found.

| Game group (terms) | Tier | Best source(s) | Notes |
|---|---|---|---|
| Fighting games, general (234) | A (SF3 only) | StatePlay-Dataset: SF3 clips + controller inputs + RAM state (CC BY-NC); ReactiveGWM-v2 (SF3); self-generation via DIAMBRA / stable-retro / FightLadder | Generic motion/input terms (DP, QCB, 360) can be grounded in SF3; title-specific terms need their own game |
| Tekken (75) | B/C (A if released) | COMBAT Tekken 3 (frames + both players' inputs + HP; release claimed, licence unstated); TASVideos Tekken input movies + encodes; Wavu Wiki T8 clips with input display; TekkenBot memory reader | No confirmed public frames+inputs for Tekken 7/8 |
| Guilty Gear (13) | C | ggxrd_replay_mod (Xrd Rev 2 footage + per-frame inputs, MIT, needs game); Strive Replay Theater + Afterimage recorder | |
| Street Fighter (2) | A (SF3) / C (SF6) | StatePlay (SF3); SF6_Tools replay input logger (REFramework, MIT) | Drive Rush is SF6-only |
| Soulcalibur (9) | C | TASVideos SoulCalibur II (.dtm) / Soul Blade (.bk2); Dolphin, PCSX2, MAME capture; SCVI replays + HorseMod overlay | |
| MK / KI / BlazBlue (4) | C | DIAMBRA UMK3, stable-retro MK, MAME KI; BBCF Improvement Mod; MK11 Button Log | |
| Pokémon (81) | A for text logs / C visual | Showdown public replays; Metamon raw/parsed (parsed CC BY-NC); VGC-Bench logs (MIT); Showdown client renders battles; PyBoy/Orak/VideoGameBench for Red/Crystal frames + RAM | Weak visual arm: turn-based, move names appear on screen |
| Zelda OoT (57) | A-narrow / C | TASVideos publications: encoded videos + .bk2/.m64 input movies (site content CC BY 2.0; ROM needed for state); gz practice ROM; OoTR spoiler logs + Mido's House restreams paired with seeds | TAS play is a narrow distribution |
| Zelda MM (10) | A-narrow / C | 4 TASVideos publications; kz practice ROM; ZSR tutorials with input display | |
| Zelda ALttP (11) | A-narrow / C | TASVideos publications; ALttPR/Archipelago seeds and spoilers; racetime.gg race archive; zelda3 C reimplementation (MIT) | Randomizer terms are checkable from spoilers |
| Smash series (49) | C | Ultimate replay MP4 export; TASVideos SSB64/Brawl input movies; gym-mupen64plus SmashEnv; vods.co VOD index | |
| **Smash Melee (21)** | **A** | **Slippi Public Dataset v3.7 [checked]: CC0, about 95k tournament replays as individual .slp files, about 200+ GB**; melee-ranked-replays (MIT); slp-to-video (GPL-2.0, needs ISO) | CC0 tournament set is better than the ranked shard for the first sample |
| Super Metroid (44) | C | TASVideos publications; SNES9X movies + videos for 400-500 hacks (offered on a forum, on request); stable-retro (integration unconfirmed) | |
| Minecraft (34) | A (actions) / B | VPT (video + actions), MineRL; MineDojo (730K narrated YouTube videos + transcripts) | Redstone terms are rare in VPT; search MineDojo transcripts for natural uses |
| EVE Online (30) | D | Only CCP ship/structure renders | |
| osu! (21) | C | .osr replays + danser render (osu-ai pipeline); beatmaps validate pattern terms | Redistribution restricted |
| AoE I/II (28) | C | Own recordings + aoc-mgz parser (MIT) for state | No public footage dataset |
| Left 4 Dead 2 (18) | C | Speedrun demos (runner-hosted) + in-game playdemo render | |
| Tetris (24) | B / C | Roboflow piece-detection images (modern Tetris); NES via emulator; TETR.IO replays (seed + inputs) | |
| Quake / Source movement (31) | B / C | Quake 1 footage dataset (unlabeled); demos + HLAE / wolfcamql render | HL: Alyx VR dataset exists but is the wrong mechanics |
| Counter-Strike (3) | A / B | TeaPearce CS:GO Deathmatch (frames + actions; existing adapter); EgoCS-400K (new, unverified); ESTA trajectories | Bunnyhop visible + velocity |
| Team Fortress 2 (3) | not searched | Assumed route: demos + render | Search was rate-limited |
| Apex Legends (3) | B | apex_enemy_detect (4,921 frames, CC BY-NC); Roboflow projects | Images only |
| Dota 2 (14) | B / C | Dota2-Vis minimap stills (existing adapter); .dem parsing (gem-dota) + client render | |
| WoW (9) | C | Warcraft Recorder + combat logs | |
| War Thunder (7) | C | Live telemetry (localhost:8111) + own capture | |
| DCS World (6) | C | Tacview ACMI + own capture; tacview-duckdb | |
| Path of Exile (7) | D | Economy data only | |
| Terraria (7) | B | Terraria Gameplay Corpus (MP4 256x144, 15 fps; existing adapter) | 15 fps may miss fast tricks |
| Factorio (6) | B / C | FLE map renderings; piebro blueprint renders; factorio-render | |
| Satisfactory (2) | D | none | |
| StarCraft II (6) | A (replay + render) | SC2EGSet (CC BY 4.0) rendered in the SC2 client; StarCraftImage summaries | |
| Dead by Daylight (3) | D | none | |
| NetHack (2) | A | NetHack Learning Dataset: 1.5M human ttyrecs + 100K bot games (~200 GB, NetHack GPL) | Terminal frames; Elbereth engraving is visible |
| Noita (2) | D | none | |
| Grim Dawn (1) | D | none | |
| Racing sims (1) | B / A? | Assetto Corsa Gym human dataset (NeurIPS 2024; 2,200+ laps; video unconfirmed) | |

## Multi-game sources

- **NitroGen** (nvidia) [checked]: per-frame gamepad labels for about 30k public gameplay videos (164 GB, CC BY-NC). Videos are not included (source URLs only), and the card does not list games.
- **PlayMyData** (Zenodo, CC BY 4.0): IGDB screenshots + gameplay-video links.
- **Orak** (SF3, Pokémon Red, others; dataset CC BY-NC); **VideoGameBench** (Pokémon Red/Crystal).
- **TASVideos** (most retro titles, input movies + encodes); **speedrun.com API** (run video links); **DIAMBRA / stable-retro** (arcade and console fighting games).

## Rights red flags

- Non-commercial only: StatePlay, ReactiveGWM, NitroGen, Metamon parsed, Apex HF, Orak dataset.
- No stated licence: Showdown replays, Dustloop, UltimateFrameData, most mod tools.
- MineDojo's licence is not stated, and it is third-party YouTube content.
- Emulator and replay routes need a user-owned ROM/ISO/game.
- Game footage remains publisher copyright.
- Cleanest licences: Slippi Public Dataset (CC0), SC2EGSet (CC BY 4.0), VGC-Bench (MIT), PlayMyData (CC BY 4.0).

## Next additions after Melee

1. **Street Fighter III via StatePlay** (+ DIAMBRA self-generation): covers the generic fighting-game input terms (234-term group).
2. **TASVideos Nintendo set** (OoT, Super Metroid, MM, ALttP; ~120 terms): public videos + frame-exact inputs; RAM state needs user ROMs.
3. **Tekken** (75 terms): COMBAT if actually released; otherwise TekkenBot self-capture.
4. **StarCraft II / SC2EGSet:** clean licence, state validates build orders.
5. **Minecraft:** MineDojo transcripts to find natural redstone-term clips; VPT for action-paired frames.

Cheap bonus: NetHack NLD (perfect validator; only 2 terms).
