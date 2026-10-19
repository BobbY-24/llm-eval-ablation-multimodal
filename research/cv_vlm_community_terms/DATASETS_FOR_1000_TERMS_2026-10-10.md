# Datasets for the 804 gaming terms in "1000 Terms.xlsx" (2026-10-10)

Workbook: /Users/bobyan/Desktop/1000 Terms.xlsx, sheet "High Quality Terms", 1,040 rows (804 Gaming, 236 Cybersecurity). Games were assigned by joining the workbook's Term group ID to GAME_TERM_INVENTORY_2026-10-06 (all 804 gaming terms matched). The mapping is in `1000_terms_game_map_2026-10-10.csv`. A term can map to several scopes.

All findings below come from web searches on 2026-10-10. None were downloaded or verified locally. Licences are as reported by the source pages.

| Game (terms) | Dataset / source | What it has | Independent validator? | Visual arm? | Rights / notes |
|---|---|---|---|---|---|
| Fighting games: general 233, Tekken 71, GG 14, SC 11, SF/MK/others ~8 | StatePlay-Dataset (HF onepiece1999), SF3 | Frames + 5 RAM states (timer, HP, meters) + 11-dim actions | Partial (RAM + inputs) | Yes | Random/agent play, so few intentional techniques; ROM rights |
| | ReactiveGWM (arXiv 2605.15256), SF2:CE / SF3 | 5 s clips at 20 fps + per-frame button vectors via stable-retro | Inputs | Yes | Random agent; release unclear |
| | COMBAT (arXiv 2603.00825), Tekken 3 | 1.2M frames with both players' inputs, HP, pose, masks | Inputs | Yes | Release promised, unverified; Tekken 3 ≠ Tekken 7/8 |
| | GGST replay controller-log dataset (ACM DL) | Controller ops + battle progression from Strive replays | Inputs | No (logs) | Availability unverified |
| | Self-built via stable-retro (SF2/SF3 RAM) | Our own clips with exact inputs | **Yes**: QCF/QCB/DP motions are checkable from inputs | Yes | User-owned ROMs needed |
| Pokémon (81) | Metamon (UT-Austin-RPL; HF jakegrigsby/metamon-parsed-replays, -raw-replays) | 5M+ trajectories reconstructed from human Showdown battles | Battle log | No: text/turn-based | Good for text and action-selection arms only |
| Zelda OoT 57 / MM 10 / ALttP 10, Super Metroid 44, L4D2 18 | TASVideos input files (bk2/fm2) replayed in BizHawk with RAM logs | Deterministic inputs → exact state per frame | **Yes** (RAM) | Yes (emulator render) | No bulk export or licence; ROMs required; TAS play is narrow |
| Smash Melee 21 (+ series 49) | erickfm/melee-ranked-replays (MIT, 1.43 TB; smallest shard ~3 MB) | Slippi telemetry | **Yes** | Needs own render | Current Stage 1 path |
| Minecraft (34) | MineDojo YouTube (Zenodo 6643838) | 730K narrated videos, ~300K h, 2.2B transcript words | No | Yes (YouTube) | Licence not stated; third-party video. Best for finding in-the-wild uses (transcript search) |
| | VPT / MineRL (existing loaders) | Video + actions | Actions only | Yes | General play; redstone terms rare |
| StarCraft II (6) | SC2EGSet v2.1.0 (Zenodo 17829625, CC BY 4.0) | 17,930 tournament replays with game state (since 2016) | **Yes** (build orders: 12 pool, 4 gate, cheese) | Replays renderable in the SC2 client | Clear licence |
| CS / Source / Quake movement (CS 3, Source 15, Quake ~21, Apex 3) | ESTA (arXiv 2209.09861; awpy) | 1,558 pro CS:GO demos, 7.9M frames, 417K trajectories | **Yes** (velocity traces → bhop, air-strafe) | Demo playback needs a game client; existing csgo loader has frames + actions | Check repo licence |
| Dota 2 (14) | OpenDota data dumps (2015/2017; ~10M parsed matches); gem-dota .dem parser | Match/player stats; .dem parsing gives positions | Yes if .dem parsed (pulls, stacks) | Client render | Dumps are old; replays expire after ~10 days |
| AoE I/II (36) | aoc-mgz (MIT, PyPI `mgz`) parses .aoe2record; aoe2rec (Rust) | Parsed commands/state from recorded games | **Yes**: fills the AoE validator gap | Own capture | Need replays (own or shared); no public labelled set found |
| Tetris / TETR.IO / TGM (30) | TETR.IO replays (JSON: seed + inputs) | Deterministic replays | **Yes** (T-spin, DAS, finesse) | Re-simulate/render | No public bulk set found; player-exported |
| osu! (21) | .osr replays + .osu beatmaps | Cursor/key inputs; beatmap patterns | **Yes** (jumpstream, chordjack from beatmaps) | Render (e.g. danser, unverified) | ppy/authors restrict redistribution |
| EVE Online (30) | zKillboard per-killmail API / RedisQ; ESI | Killmails (no positions) | Weak | No | Most EVE terms (kite, scram kiting) are not derivable |
| WoW 9, War Thunder 7, DCS 6, Factorio 6, Terraria 7 | Warcraft Logs + Warcraft Recorder; WT localhost:8111 telemetry; Tacview ACMI; FLE; Terraria corpus (existing loader) | Self-recorded logs/telemetry | Varies | Varies | Self-capture; ToS review |
| Cross-game references | GameplayQA (9 games, 2.4K QA); VideoGameQA-Bench (800+ games); GameQA-140K (synthetic) | Gameplay QA formats | No | Yes | Benchmark design references, not jargon coverage |

## Recommendation (after Melee)

1. **StarCraft II / SC2EGSet:** clear CC BY 4.0 licence, game state validates build-order terms.
2. **Fighting-game input terms via stable-retro (SF2/SF3) or StatePlay:** the largest term group (300+), and input motions (QCF, QCB, DP) are exactly checkable. Needs user-owned ROMs; intentional techniques need scripted or human play.
3. **AoE II via aoc-mgz:** a parser exists (MIT), which unlocks a validator for the 5 existing AoE pilot families.
4. Mining source, not a benchmark: **MineDojo transcripts**, to find natural video uses of Minecraft terms.
