# OpenFrontMind

## Goal

Train a reinforcement-learning agent to play [OpenFront](https://openfront.io) well,
using a self-play approach in the spirit of DeepMind's AlphaStar (StarCraft II) — scaled
down to fit local/rented GPU budgets. This is primarily a learning project (RL
fundamentals), so scope grows iteratively: get a correct, fast environment and a dumb
baseline working first, then layer on self-play, a league, and bigger models.

Upstream game repo: [openfrontio/OpenFrontIO](https://github.com/openfrontio/OpenFrontIO)
(AGPLv3; see Licensing below).

## Why this game is workable for this approach

- `src/core` (the game simulation) is pure, dependency-free, **deterministic**
  TypeScript — seeded PRNG, no floats, no DOM. It runs headlessly in Node with no
  browser/canvas involved. Confirmed via the upstream repo's own headless perf harness
  (`tests/perf/fullgame/FullGamePerf.ts`), which drives thousands of ticks purely in
  Node.
- In production, the server is a dumb intent-relay — every client (including the
  upstream "Nation"/"Tribe" AI) simulates locally and submits `Intent` messages. That
  means we can drive the exact same simulation for training without ever touching a
  browser or the network stack.
- Actions are a clean, enumerable set of Zod-typed `Intent`s (attack, boat/transport,
  build_unit, move_warship, alliance/donate/embargo actions, etc.), each mapped 1:1 to
  an `Execution` class. There's already a legal-action-mask helper
  (`GameRunner.playerActions()` / `playerBuildables()`).
- The map is a flat integer-indexed tile array (`TileRef = number`), and per-tick state
  diffs (`PlayerUpdate`, tile/unit updates) are compact — a reasonable base for
  building observation tensors (raster tile planes + per-player scalar features,
  AlphaStar-style).
- Existing scripted bots ("Nation" full AI civilizations, "Tribe" simpler roaming AI,
  core logic in `src/core/execution/utils/AiAttackBehavior.ts`) are useful as
  non-trivial baseline opponents before self-play kicks in.

## Architecture

Two-language split: the game engine/environment stays in TypeScript (it's the upstream
language and the sim must stay bit-identical to real OpenFront); the learning stack is
Python/PyTorch (mature RL tooling, GPU support for the 3070). A small local bridge
process connects them — conceptually the same shape as AlphaStar's Python side driving
the SC2 binary via PySC2.

```
openfrontmind/
  engine/              # vendored copy of OpenFrontIO's src/core (deterministic sim)
  env/                 # Node.js headless training harness wrapping engine/
    src/
      server.ts        # per-process env server speaking the bridge protocol (stdio)
      protocol.ts       # message schema for reset/step/observation/action exchange
  training/             # Python side
    envs/               # Gym/PettingZoo-style wrapper; spawns N env/ subprocesses
    models/              # policy/value network definitions (PyTorch)
    train.py              # self-play PPO training loop
    league/                # checkpoint pool + opponent sampling (later phase)
  scripts/               # dev/setup scripts
  docs/
    Architecture.md       # deeper design notes as they firm up
  CLAUDE.md
```

### Parallelization ("headless client" for self-play)

Node is single-threaded per process and the simulation is CPU-bound, so parallelism
comes from running many independent OS processes rather than threads:

- Python spawns N instances of the `env/` server as subprocesses (one game each,
  players controlled either by the RL policy, frozen past checkpoints, or the upstream
  scripted bots).
- Each subprocess speaks a minimal newline-delimited JSON protocol over
  stdin/stdout to start: `reset -> observation`, `step(actions) -> observation, reward,
  done`. Optimize later (e.g. a binary/msgpack protocol over Unix domain sockets) only
  once correctness is established — premature to build that now.
- Python-side vectorized env wrapper fans actions out to all subprocesses and batches
  observations back for the policy forward pass on the GPU, same shape as a
  `SubprocVecEnv` in standard RL tooling, except the "env" is our own game server
  instead of a gym env directly.
- Practical parallelism ceiling is CPU core count on the training machine, not the
  3070 — the GPU only does policy/value net forward+backward passes, which are small
  relative to vision/LLM models.

## Roadmap (iterative — expect this to evolve)

0. **Vendor the engine.** Copy `src/core` into `engine/`, get it building and its
   existing tests passing standalone in this repo.
1. **Minimal env harness.** In-process (no subprocess yet) `reset`/`step` wrapper
   around `engine/` to validate the API shape and measure raw ticks/sec.
2. **Baseline agent.** Random or trivial heuristic policy against the vendored
   Nation/Tribe bots, just to validate the env end-to-end (rewards flowing, episodes
   terminating correctly, no desyncs).
3. **Bridge protocol.** Stand up the stdio JSON protocol in `env/server.ts`, drive one
   subprocess from Python, confirm parity with the in-process harness from step 1.
4. **First real learning.** Small map, single opponent type, simple algorithm (e.g.
   PPO via Stable-Baselines3) — get *something* learning before worrying about scale.
5. **Parallelize.** Multiple subprocesses, vectorized env, scale up self-play
   throughput.
6. **Self-play league (AlphaStar-lite).** Checkpoint pool, opponent sampling,
   matchmaking against past versions of itself + scripted bots.
7. **Stretch:** full map/ruleset, evaluation against scripted bots and possibly a live
   game via the real WebSocket client protocol (separate, harder path — see below).

## Key decisions log

- **Python + PyTorch for the learning stack**, not TensorFlow.js. Chosen for RL
  ecosystem maturity (Stable-Baselines3/RLlib or a custom PPO loop) and better GPU
  tooling on the 3070, at the cost of needing a cross-process bridge to the TS engine.
- **Vendor `src/core` into `engine/` rather than a git submodule.** We expect to
  modify the engine (e.g. adding a training-friendly step/reset API, possibly
  stripping unused subsystems) and don't want submodule friction while iterating. We
  lose easy upstream pull; revisit if upstream ships something we need later.
- **Self-play via OS subprocesses, not worker_threads.** The simulation is CPU-bound
  single-threaded JS; separate processes parallelize cleanly across cores without
  fighting Node's threading model.
- **Not** pursuing the live-networked-client path (real WebSocket `intent` messages
  against openfront.io) for now — it requires an auth token, reimplementing the
  client's worker-side simulation, and is strictly harder than headless self-play for
  no near-term benefit. Revisit only if/when we want the trained agent to play the
  actual live game.
- Historical human replays are **not available to us** — they're archived via a
  closed-source API. Self-play is the practical data source; no imitation-learning
  bootstrap from human games planned.

## Licensing note

Upstream OpenFrontIO is AGPLv3 (network copyleft — matters only if we run a modified
version as a public-facing network service; training locally is unaffected).
`/resources` assets are CC BY-SA 4.0. `/proprietary` assets (logos, fonts, music) are
all-rights-reserved and out of scope for this project (headless training needs none of
them). Not legal advice — revisit before any public/commercial release.
