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
- **`bots=3, nations=1` run, 720 updates.** Reward positive/stable throughout. Final:
  26/13,154 episodes won (0.2%).
- **Diagnostic: dumb (non-learning) greedy policy vs. lobby size**, same seeds, same
  `attack wilderness else nearest target else do-nothing` heuristic:
  - `bots=0, nations=1` (pure 1v1): peak tile share 71-80% across 4 games, 1 outright
    WIN. Matches informal manual "left-click every few seconds" testing.
  - `bots=3, nations=1` (this run's actual config): peak only 29-42%, 4/4 LOSS.
  - `bots=9, nations=0` (10-player lobby): peak only 26-28%, **3/3 full elimination**
    (`winner=None` — no one hit 80%, agent just got ground down by numerous weak
    opponents). **Finding: more competing players caps the achievable ceiling AND
    raises elimination risk for a pure-offense policy**, independent of learning.
- **Diagnostic: trained policy (300 updates, ~2,700 episodes) vs. the same dumb
  policy, identical seeds + config (`bots=3, nations=1`).** Peak tile shares and even
  peak-tick numbers nearly identical (e.g. one seed: 36.8%@tick487 for both). **Finding:
  after 300 updates the trained policy is statistically indistinguishable from a
  policy with zero learning in it** — the flat win rate isn't (only) a task-difficulty
  or horizon problem, something in training itself isn't producing improvement over
  the naive baseline. Root cause not yet found; see optimization ideas below.
  Checkpoint preserved at
  `training/models/checkpoints_saved/agent_bots3_nations1_easy_attackonly_update300.pt`.
  **Validated, not a one-off:** re-ran the trained policy 5x on the same seed to rule
  out its stochastic action sampling (`get_action_and_value` always samples, no
  greedy-eval mode) explaining the match by coincidence. Peak tile share across the 5
  runs: 36.5-37.0% (tight), peak tick 486-499 (tight), **5/5 lost to the same
  opponent** — essentially zero variance from the dumb-policy comparison point. The
  finding is robust, not a lucky/unlucky single draw.
- **Gold logging added** (`mean_gold_at_done` in `train.py`'s per-update print).
  Confirmed at scale (2,700+ episodes): average **135,262 gold sitting unspent** at
  episode end (range 54k-255k) — the action space has no way to spend it
  (attack-only). Cross-checked against real build costs in `Config.ts`: `City` is
  125k→250k→500k→1M (doubling, capped), `DefensePost` is 50k→100k→150k... (capped at
  250k) — so a typical game's banked gold is worth roughly *one* City or *one-to-two*
  DefensePosts, not a real economy. Motivated deprioritizing buildings until richer
  observation lands (see below) — building one structure once per game isn't enough
  upside to chase yet.
- **`NUM_ENVS` throughput benchmark** (rollout-collection only, PPO phase excluded,
  16-core machine): 8→970 ticks/s, 16→1452 (+50%), 24→1705 (+17%), 32→1890 (+11%),
  40→1970 (+4%), 48→1952 (flat/regressed). **Plateaus around 32-40 envs.** But real
  full-update throughput only improved +13% from 8→16 (750→850 ticks/s) because the
  PPO minibatch phase scales with total data (`STEPS*n`) and gets zero benefit from
  more envs — don't conflate the two numbers when picking `NUM_ENVS`.

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

- ~~Redundant-attack masking~~ **Done.** `harness.ts` exposes `agent.outgoingAttacks()`
  state per candidate; `build_action_mask` filters already-attacking targets.
- ~~Subprocess-level env parallelism~~ **Done, differently than originally planned.**
  `VecOpenFrontBridge` uses real OS processes (`multiprocessing.Process` + `Pipe`),
  not a `ThreadPoolExecutor` on the hot path — profiling showed N threads all doing
  the CPU-bound base64/numpy decode work in the parent contended badly on the GIL, so
  decode happens inside each worker process instead. `ThreadPoolExecutor` is only
  used for occasional concurrent resets. `NUM_ENVS` is a tunable constant in
  `train.py`; `bridge.py`'s build step is now a shared `_ensure_built()`.

**Root-cause the stuck learning (see diagnostic above).** Trained policy after 300
updates was statistically indistinguishable from a non-learning dumb policy. Leading
hypothesis: the old `ATTACK_SLOTS` scheme had no stable identity — `Player.nearby()`'s
return order depends on `_borderTiles`'s `Set` iteration order, which reshuffles every
tick as territory shape changes, so slot `k` meant a different opponent tick to tick.
A feedforward policy scoring fixed slot indices structurally cannot learn "prefer
this kind of target" under that scheme. **Fixed — see "Entity encoder" below.**
Remaining candidate fixes:
- ~~Value-loss clipping~~ **Done.** Same clip-and-take-worse-case idea as the
  existing policy-ratio clip, applied to the value regression (clamp `new_value -
  old_value` to `±CLIP_EPS`, take the max of clipped/unclipped squared error).
  Motivated by something directly observed while verifying the entity encoder: a
  real training run's output showed `value_loss` sawtoothing sharply on every
  episode completion (0.0012 → 925.77 → decaying again) — clipping caps how hard one
  epoch can yank the value estimate from a single rare high-variance return.
- **Observation/reward normalization via running mean/std.** Scalars are currently
  just `log1p`-transformed, not normalized. Advantage normalization is already done
  (good) — extend the same idea to observations/rewards.

**Entity encoder — done.** Went straight to the "principled long-term version" from
the original plan (type-planes / top-K slots were considered but skipped) since one
mechanism fixes both the observation problem and the action-space problem at once:
- `action_space.py`: `build_entities()` turns `attackMask` candidates into a
  variable-length `(N, 7)` feature array (`is_tribe`, `is_nation`, `is_wilderness`,
  `is_do_nothing`, `tile_fraction`, `log_gold`, `log_troops`) + a parallel `targets`
  list. `do_nothing` is folded in as a fixed all-zero-stats entity so one code path
  covers every candidate. `pad_entities()` pads to `MAX_ENTITIES=16` — a buffer-size
  cap now, not an identity-bound one like the old `ATTACK_SLOTS` (truncation just
  drops least-relevant excess candidates for that one tick, not a permanent "no
  slot" for some opponent).
- `policy.py`: new `EntityEncoder` (shared per-entity MLP + learned-query attention
  pooling, Deep Sets / Set Transformer style) produces a fixed-size context vector
  from however many candidates exist, feeding the backbone alongside the spatial CNN
  output. The old flat `Linear(256, TOTAL_ACTIONS)` policy head is gone — action
  selection is pointer-network style: score each candidate's own embedding against a
  query derived from the backbone hidden state, softmax over exactly as many
  candidates as exist that tick. Fixes the slot-identity problem directly: the
  network now scores candidates by their features, not by a meaningless index.
- `train.py`: `EpisodeState` swaps `mask`/`slot_targets` for `entities`/
  `entity_mask`/`targets`; rollout buffers gained an `(STEPS, n, MAX_ENTITIES, 7)`
  entity tensor + mask. `watch.py` updated to match.
- Verified at both the single-episode level and the full batched `run_update()`
  level (ragged per-env entity counts padded/masked correctly) before trusting it.
- **Not yet re-validated**: whether this actually fixes the "trained ≈ dumb
  baseline" finding — that requires an actual training run + the same
  trained-vs-dumb comparison methodology used to catch the original problem.

**Architecture bottleneck — fixed.** The backbone used to flatten the full conv
output straight into a `Linear`, making that one layer ~97-99% of the network's
params (worse on bigger maps, since it scaled with `H*W`) and tying the network's
shape to one map resolution. Replaced with `nn.AdaptiveAvgPool2d((6, 6))` before the
flatten — fixed-size regardless of map dimensions, keeps coarse spatial layout
without the blowup. Params: 5.3M (original) → 11.9M (after adding the entity
encoder, on the bigger `mid_plains` map) → **413,889** (after pooling). Side effect:
`Backbone`/`Agent` no longer need `height`/`width` to be constructed at all.

**Buildings** (`DefensePost`, `City`) — still deliberately deprioritized, now
pending validation that the entity encoder actually moves the needle on the stuck-
learning problem before adding more surface area. Motivated by: (a) the gold-
hoarding finding above (typical banked gold is only worth 1-2 structures, so the
economic ceiling needs addressing too, not just the mechanic existing), and (b) the
dumb-policy lobby diagnostic shows the peak-then-collapse pattern even with zero
learning and zero buildings involved, meaning "no way to hold a lead" is a real gap
but info-about-threats was the more load-bearing missing piece first. `DefensePost`
gives a genuine defensive multiplier (`Config.ts` `defensePostDefenseBonus()`),
`City` uses the otherwise-wasted gold for more troops.

**Verify the `mid_plains` horizon before a real training run.** `STEPS=2000`/
`GAMMA=0.9995` were tuned for `big_plains`'s ~1400-2000-tick games. `mid_plains` is
2.25x bigger by land area; the dumb-policy timing test on the even-bigger `onion` map
didn't finish within 6000 ticks in 2 of 3 trials, suggesting bigger maps take
proportionally longer to resolve. Should re-run the dumb-policy timing check
specifically on `mid_plains` before trusting a long run — otherwise risks silently
reintroducing the exact horizon-mismatch bug the `bots=1` diagnostic caught earlier.

**New maps vendored.** `onion` (real production map, 512×512, 210,555 land tiles,
real named nation spawn points) and `mid_plains` (synthetic, generated by tiling
`big_plains`'s land texture to 300×300, 90,000 land tiles) both added to
`tests/testdata/maps/`. `mid_plains` is now the default map everywhere (`harness.ts`,
`cli.ts`, `train.py`, `watch.py`) — picked because `big_plains` (200×200) turned out
to be far more cramped than a typical real game: The Box, a real production map, is
2048×2048 (~104x `big_plains`'s area), and manual testing on it showed trivial wins
by clicking alone, which the same policy could not reproduce on `big_plains`'s denser
`bots=3, nations=1` setup.

**Throughput — async actor/learner overlap (IMPALA-style), likely bigger than more
`NUM_ENVS` tuning.** Rollout and PPO currently run strictly sequentially within one
update (`time=64s (rollout=38s ppo=26s)` at `NUM_ENVS=16`) — the GPU is fully idle
during rollout, all env workers are fully idle during PPO. Overlapping rollout
collection for update N+1 with the PPO phase for update N (even without full
IMPALA/V-trace off-policy correction) could reclaim a large chunk of that idle time.
Bigger lever than `NUM_ENVS`, which only helps the rollout half and plateaus around
32-40 envs on this machine (see benchmark above) while the PPO phase scales with
`STEPS*n` regardless of env count.

Smaller/cheaper throughput wins, same idea of not leaving free performance on the
table:
- `torch.compile` on the policy forward/backward — with 500+ small minibatches per
  update, kernel-launch overhead is a bigger fraction of PPO-phase time than compute
  is for a 5.3M-param network.
- Bigger `MINIBATCH_SIZE` (GPU util is ~60% at the current 256) — fewer, larger
  minibatches amortize per-call overhead better.
- Vectorize the per-env Python loop in `run_update()` (currently a plain
  `for k in range(n):` doing numpy work one env at a time under the GIL) — doesn't
  change the parallelism ceiling found in the benchmark, but lowers the constant
  per-tick cost, raising the whole throughput curve rather than shifting the plateau.

**Multi-horizon value heads** (idea, unimplemented): a second value head at a much
higher `GAMMA`, computed via a second GAE backward pass over the same rollout data.
Note: `GAMMA`'s effective reach is still capped by `STEPS` regardless of how high it
is set, since GAE only looks back within one rollout buffer — a long-horizon
`GAMMA` needs a correspondingly large `STEPS` to actually matter, not just a high
discount factor.

**Operational gotcha (learned the hard way this session):** killing `train.py`'s main
process does **not** cascade to its `VecOpenFrontBridge` worker processes or their
Node subprocesses — the `with` block's cleanup never runs on a bare `kill`, so they
become orphaned and keep running (and keep burning GPU/CPU) alongside a freshly
started run. Kill the full process group (`ps aux | grep -E "python -m
training.models.train|node.*dist/server.mjs"`, kill all matching PIDs) before
relaunching, not just the main PID.

## Licensing note

Upstream OpenFrontIO is AGPLv3 (network copyleft — matters only if we run a modified
version as a public-facing network service; training locally is unaffected).
`/resources` assets are CC BY-SA 4.0. `/proprietary` assets (logos, fonts, music) are
all-rights-reserved and out of scope for this project (headless training needs none of
them). Not legal advice — revisit before any public/commercial release.
