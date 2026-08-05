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

## Training run log

Format: `update N` = training iteration count at that point. Stats, then verdict.

- **Baseline (inherited).** `bots=25`, raw `delta_tiles` reward, `WIN/LOSS=±10000`,
  `GAMMA=0.99`, `STEPS=500`, `ENTROPY_COEF=0.01`. Entropy 0.0000 from update 0.
  `value_loss` 1e5-1e6. Wins: 0/~290. **Bug**: reward scale swamped shared-backbone
  gradient via `VALUE_COEF`. Root-caused, not tuning.
- **Fix: reward rescale.** `WIN/LOSS` →`±100`, fresh net. `value_loss` sane. Entropy no
  longer permanently flat. Wins: 0/~900 (`bots=25`).
- **Fix: `ENTROPY_COEF`** `0.01→0.05`. Entropy still mostly ~0. Wins: 0/~1300 combined.
- **Fix: reward = fraction, curriculum start.** `delta_tiles`→`100×delta_tiles_fraction`.
  `bots` `25→10`. Reward flat/slightly negative. Wins: 0/~1200.
- **Fix: horizon match.** `GAMMA` `0.99→0.998` (horizon=`STEPS`=500). Entropy sustained
  high (0.3-1.3+), no longer collapsing. Reward still flat. Wins: 0/~1300.
- **Diagnostic (`bots=1`).** Greedy-policy episode: peak 62.5% map share at tick 95 →
  full elimination by tick 1951 (~1850-tick decline). **Finding: horizon (500) way
  short of decline arc (~1850) — value fn structurally can't connect peak to collapse.**
- **Fix: lengthen horizon.** `STEPS` `500→2000`, `GAMMA` `0.998→0.9995` (horizon=2000).
  Reward flipped ~100% positive, stable. Re-ran diagnostic: peak 62.8%@tick44 → 19.7%
  final, lost to opponent hitting 80% (not elimination — holding measurably improved).
- **Discovery: `nations=1` silently active all session** alongside every `bots=N`
  tested — every run secretly included one strong Nation opponent too.
- **Current: `bots=3, nations=1`.** Few weak Tribes (exploitable, low threat) + one
  real Nation. Same `STEPS=2000`/`GAMMA=0.9995`/`ENTROPY_COEF=0.05`. Reward positive,
  stable. Wins: 0 so far (early).

## Game mechanics notes (learned this session, not obvious from a skim)

- **Win condition**: control ≥80% of map tiles (`Config.percentageTilesOwnedToWin()`,
  FFA), or a 170-min hard time cap → highest tile-share wins. Not last-player-standing.
- **`bots` config param → Tribes** (weak: `TribeExecution.ts`, 129 lines, no navy/
  nukes/diplomacy, attacks only every 40-80 ticks). **`nations` config param →
  Nations** (strong: `NationExecution.ts` 372+ lines plus dedicated warship/nuke/
  MIRV/alliance/structure behavior files).
- **Attack troop cost**: `Config.attackAmount()` = 20% of current troops for
  `PlayerType.Human` (us), 5% for `PlayerType.Bot`. Fresh cut taken on *every* new
  attack intent — attacks never merge with an existing one toward the same target
  (`PlayerImpl.createAttack` always makes a new `Attack`).
- **Attack shape**: tile-claim order in `AttackExecution.addNeighbors` uses 4-
  connectivity plus a priority that prefers tiles more enclosed by owned territory
  (`numOwnedByMe`) — this smooths a *single sustained* attack into a circle over many
  ticks. Our action space re-issues a fresh attack intent nearly every tick an attack
  action is chosen, resetting this each time → jagged/square blobs, and risks
  compounding troop depletion (each new attack takes 20% of an already-reduced pool).
- **Gold**: passive per-tick income scaling with territory/workers
  (`goldAdditionRate`) — no kill bonus; an eliminated player's remaining gold is
  destroyed, not transferred to the attacker.
- Troops/gold are already in the network's observation (`observation.py` scalar);
  reward only uses tile fraction.

## Known issues / next steps (not yet implemented)

- **Redundant-attack masking.** Mask out re-attacking a target we already have an
  active outgoing attack toward. Needs `computeAttackMask` (`env/src/harness.ts`) to
  expose `agent.outgoingAttacks()` state, plus a `build_action_mask` change on the
  Python side. Motivated by the troop-depletion + attack-shape finding above.
- **Subprocess-level env parallelism.** N `OpenFrontBridge` subprocesses driven by a
  Python `ThreadPoolExecutor` (blocking I/O releases the GIL, so threads suffice —
  no need to rewrite `bridge.py` as asyncio), batching observations into one forward
  pass per tick. `N` should be a tunable config value, not hardcoded. Prerequisite:
  `OpenFrontBridge.__init__` currently reruns `node build.mjs` on *every*
  instantiation — needs to happen once, shared, before fanning out N subprocesses.
  Bundle the GPU migration with this step, not before — a batch-of-1 pipeline doesn't
  benefit from GPU (sync/kernel-launch overhead dominates a network this small).
- **Multi-horizon value heads** (idea, unimplemented): a second value head at a much
  higher `GAMMA`, computed via a second GAE backward pass over the same rollout data.
  Note: `GAMMA`'s effective reach is still capped by `STEPS` regardless of how high it
  is set, since GAE only looks back within one rollout buffer — a long-horizon
  `GAMMA` needs a correspondingly large `STEPS` to actually matter, not just a high
  discount factor.

## Licensing note

Upstream OpenFrontIO is AGPLv3 (network copyleft — matters only if we run a modified
version as a public-facing network service; training locally is unaffected).
`/resources` assets are CC BY-SA 4.0. `/proprietary` assets (logos, fonts, music) are
all-rights-reserved and out of scope for this project (headless training needs none of
them). Not legal advice — revisit before any public/commercial release.
