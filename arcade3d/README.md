# Flytris arcade (3D)

A 3D render of every recorded Flytris game. Each fly sits on a stool at its own cabinet and
replays its real game move by move. When a game ended, that screen flashes red and the fly
takes off. The last fly still playing is the 86-line V3 champion.

## What is real

- **The games.** `export_games.py` exports 7,172 recorded games: 7,136 V2 evolution games
  (`runs/train_after`, `runs/train_after_v2`) and the 36 V3 validation games
  (`runs/v3_modal/wide2048`). Each one is replayed with `flytris/tetris.py` and must reproduce
  its recorded lines and death. `check_engine.mjs` checks the browser engine the same way.
- **The order they drop out.** All games run on one shared clock counted in pieces placed. A
  fly leaves exactly when its game ended, so the survivors are the games that actually lasted
  longest. The clock runs faster in the middle of the video and slower at the start and end.
- **The winner.** Seed 1800006, 86 lines in 257 pieces. The video stops on the piece where it
  clears its 86th line (piece 249), 30 seconds in total. The runner-up died at piece 239.
- **The fly body.** NeuroMechFly, a micro-CT model of an adult *Drosophila*, from
  [flygym](https://github.com/NeLy-EPFL/flygym) 2.1.0 (Apache-2.0, license in `nmf/`). The hero
  flies use the full-resolution meshes from flygym's public asset bucket (`nmf_full/`).
  `build_fly.py` poses it with IK: left front foot on the joystick, right front foot on a button.

## What is not

- These games were not played at the same time or on the same pieces. V2 flies played one
  sequence per generation and each V3 game had its own seed. The video lines them up by piece
  count. Say "every cabinet replays a real recorded game", not "7,172 flies played a live
  tournament".
- The V2 flies came from an earlier, weaker pipeline (mostly 0 to 5 lines). The winner is V3.

## Sound

`soundtrack.py` synthesizes everything in code from `timeline.json`, the render's own event
times. The music is Korobeiniki, the Tetris melody, which is a public-domain 1861 folk song;
the chiptune arrangement is original. A buzzing fly voice doubles the melody an octave down,
since a Drosophila wing beat is about 200 Hz. The swarm buzz follows how many flies are in the air, and
the line-clear chimes fire on the real clears of the opening fly and the winner.

## Run it

- Finished video with music: double-click `RenderFull.bat`, or `python make_final.py`. It
  renders in three 10-second chunks and writes `media/flytris_arcade_3d_final.mp4` (about 5 minutes).
- Live preview: double-click `Preview.bat` (space pauses, arrow keys seek).
- Stills: `python capture.py stills --n 1500 --times 0 10 20 29`
- Rebuild the fly: `python build_fly.py` (needs trimesh, scipy, fast-simplification)

Rendering uses headless Chromium through Playwright with the NVIDIA GPU (`--force_high_performance_gpu`).
Frame i is time i / 30, so renders are deterministic.
