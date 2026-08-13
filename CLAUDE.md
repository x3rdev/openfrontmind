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

**Rollout buffer memory bug, found and fixed on first real run at full settings.**
`spatial_buf` (`STEPS, n, 3, H, W`) was allocated as `float32` despite holding purely
binary planes (`is_land`/`is_mine`/`is_enemy`) - at `STEPS=2000, NUM_ENVS=24` on
`mid_plains` this tried to allocate **48.3 GiB**, failing outright (31GB machine).
Nobody had actually run these exact settings together before - throughput
benchmarking used a separate small buffer-free script, and entity-encoder testing
used much smaller `STEPS`/`NUM_ENVS` for speed. Changed `spatial_buf`'s dtype to
`bool` (4x smaller, lossless - the values are always exactly 0.0/1.0), casting back
to float only for the per-minibatch GPU slice in the PPO loop. Brings the same
config down to ~12.1 GiB, comfortably within budget. First run of the entity-encoder
architecture was against `bots=0, nations=1` (pure 1v1, no Tribes) rather than the
earlier `bots=3` config.
**New maps vendored.** `onion` (real production map, 512×512, 210,555 land tiles,
real named nation spawn points) and `mid_plains` (synthetic, generated by tiling
`big_plains`'s land texture to 300×300, 90,000 land tiles) both added to
`tests/testdata/maps/`. `mid_plains` is now the default map everywhere (`harness.ts`,
`cli.ts`, `train.py`, `watch.py`) — picked because `big_plains` (200×200) turned out
to be far more cramped than a typical real game: The Box, a real production map, is
2048×2048 (~104x `big_plains`'s area), and manual testing on it showed trivial wins
by clicking alone, which the same policy could not reproduce on `big_plains`'s denser
`bots=3, nations=1` setup.

**Entity-encoder re-validation at 500 updates: still ≈ dumb baseline.** Ran the same
trained-vs-dumb comparison that caught the original bug, this time against the new
entity-encoder architecture on `bots=0, nations=1` (500 updates, `NUM_ENVS=8`).
Result: peak tile share matched within ~0.5 points on every seed, peak ticks matched
exactly on 3 of 4, every game lost to the identical opponent. **The entity encoder
fixed the action-slot-instability bug but did not fix the stuck-learning symptom** -
something else is the dominant cause.

**Value trace: the value head learned tick number, not game state.** Traced `V(s)`
tick-by-tick through one real game (trained policy, 500 updates). It declined
smoothly and almost monotonically for the *entire* episode - including during the
first ~140 ticks while tile share was still climbing from 39% to 78.5% (peak). It
kept declining at essentially the same rate through the peak, the collapse, and a
long flat-ish middle stretch where tile share was actually just oscillating, not
declining. That's the signature of a value function outputting "generic expected
return at this point in time" rather than "expected return given this board state" -
which explains why `value_loss` can drop cleanly (fitting a smooth time curve is
easy) while the *policy* shows zero improvement: PPO's advantage (`return - value`)
can't isolate whether a specific action was good if the baseline isn't state-aware.

**Missing feature found via this: troop ratio, not raw troop count.** Observation
only ever exposed raw (log-transformed) troop count - never `maxTroops`, so the
network had no way to compute the ratio the engine's *own* scripted AI gates on
(`AiAttackBehavior.ts`'s `hasReserveRatioTroops()` uses `troops()/maxTroops()`
directly). Raw troop count also grows with elapsed game time almost regardless of
strategy, which is exactly the kind of feature that lets a value function cheat
toward predicting via tick number instead of state. Fixed: `maxTroops` now travels
as a plain JSON side-channel every tick (`server.ts`, not the packed binary stream
used by the real game client), merged into `player_stats` via a new
`apply_max_troops()`, and both the agent's own scalar and each entity's gold field
now use `troops/maxTroops` instead of raw log-troops.

**Gold now normalized via a running mean/std** (`training/envs/running_stats.py`,
`RunningMeanStd` - Welford's online algorithm), not just `log1p`-transformed.
Updated from real observations only (wilderness/do-nothing placeholder rows carry a
fixed `0.0` in the gold column that would otherwise bias the mean). Persisted inside
the checkpoint (`gold_mean`/`gold_var`/`gold_count` alongside `model`) so a resumed
run or `watch.py` normalizes with the exact statistics the weights were trained
against, not a freshly-reset estimate - this also fixed a real ordering bug where
episodes were being built (and normalized) *before* a resumed checkpoint's stats
were restored.

**Decisive diagnostic: patience alone wins 6/6, greedy attack loses 6/6.** Built a
second hardcoded (non-learning) heuristic - do nothing until `troops/maxTroops`
crosses 50%, then attack - and ran it against the original always-attack-when-
possible heuristic on 6 identical seeds (`bots=0, nations=1`, `mid_plains`). Patient
heuristic: 6/6 WIN, peak ~80-81%. Greedy heuristic: 6/6 LOSS, peak 72-76% then
decline. Same action space, same map, same opponent, zero learning involved either
way. **This disproves the "action space has a structural ceiling, buildings are
required" hypothesis from earlier in the session** - patience-then-strike is fully
achievable with attack + do-nothing alone. It also hands a concrete, mechanically-
verified ceiling to hold the trained policy against (100% win rate, ~1700-2000
ticks, peak ~80%) using information (`troop_ratio`) now actually in the
observation. `50%` was chosen deliberately past the engine's actual troop-production
optimum: `Config.ts`'s `troopIncreaseRate()` is `(10 + troops^0.73/4) *
(1 - troops/max)`, which maximizes analytically at `ratio = 0.73/1.73 ≈ 0.422`
(42.2%) - banking a bit past peak production rate before striking.

**Deliberately not adding a reward term for patience/troop accumulation.** Tempting
given the diagnostic above, but risky: a shaped reward that directly rewards
"waiting" or "high troop ratio" would teach a fixed habit, not a conditional
strategy - and patience isn't universally correct (a weak/exposed opponent should be
hit immediately, not given time to recover). That's the same class of failure as the
original always-attack policy, just inverted. The principled fix is what already
landed above: give the network the observation it needs to *condition* the decision
(`troop_ratio`, opponent type/strength via the entity encoder) and let the existing
outcome-based reward (tile fraction + win/loss) teach it when each strategy applies.
The patient-heuristic result is a ceiling to validate against, not a target
behavior to bake into the reward function. If credit assignment is still too weak
to learn this after the fixes above, the principled fallback is potential-based
reward shaping (Ng et al. 1999) - provably policy-invariant, unlike an ad hoc
patience bonus. **Implemented since** (`reward.py`): `_troop_potential(ratio) =
-(ratio - TROOP_RATIO_TARGET)²`, shaping = `SHAPING_COEF * (gamma*Φ(curr) -
Φ(prev))`. `SHAPING_COEF` raised `0.1 → 0.3 → 1.0` over the session in response to
diagnostics (see below) still showing near-zero attack-time troop ratios - see the
"low-outcome-diversity stalemate" finding below for why coefficient tuning alone
turned out not to be the resolving lever.

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
- ~~Vectorize the per-env Python loop~~ **Done, real cause was sharper than
  suspected.** An overnight run bumped `NUM_ENVS` 24→32 expecting more throughput.
  Instead `ticks/sec` was identical (~550) at both 8 and 32 envs — zero parallelism
  benefit, rollout wall time scaling exactly linearly with env count. Cause:
  `encode_observation()` recomputed the full `(H,W)` `is_mine`/`is_enemy` planes from
  `owner_grid` every tick, per env, in the serial `for k in range(n):` loop — even
  though `apply_tile_updates` already knows exactly which tiles changed that tick.
  Fixed by maintaining `is_mine`/`is_enemy` incrementally inside `apply_tile_updates`
  (update only the changed indices), threaded through `EpisodeState` and
  `encode_observation`'s new signature. Verified bit-exact against the old full
  recompute over 400 real ticks before trusting it. Re-benchmarked `run_update()`
  itself post-fix (5 timed updates/config): `NUM_ENVS` 16→866, 20→906, **24→925
  (best)**, 28→919 (statistically tied with 24, higher variance), 32→629 (cliff —
  16-core machine oversubscribed 2x). `NUM_ENVS` set to 24.

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

**Another one: stdout buffering when logging to a file.** `train.py`'s prints are
plain `print()`, no `flush=True`. Redirected to a file (`> log.txt`, not a TTY),
Python fully buffers stdout - each `update` line is ~220 bytes, default buffer is
~8KB, so ~35+ updates queue up before anything actually appears on disk. Looks
exactly like a hang (a live process, growing CPU/GPU use, an empty-looking log) but
isn't one - the Node subprocesses' own stdout (game-end messages) is a separate,
unbuffered fd and shows up immediately, which is a useful tell: log growing on game
events but silent on `update` lines means it's buffering, not stuck. Launch with
`python -u -m training.models.train` (or `PYTHONUNBUFFERED=1`) to see progress live.

**Diagnostic tooling added to `watch.py`: `--diagnostic`, `--log-probs`,
`--log-value`, `--headless`.** None of the training-time metrics (win rate,
mean_reward) could distinguish *why* the policy was stuck, so three per-tick tracing
modes were added on top of the existing GUI replay:
- `--diagnostic`: tracks troop_ratio/tile_fraction at decision time, counts
  do_nothing vs. attack actions, first-attack tick/ratio, mean attack-time
  troop_ratio, peak tile share.
- `--log-probs`: exposes the actual softmax probability of `do_nothing` vs.
  attacking (`get_action_and_value(..., return_probs=True)`), bucketed by
  troop_ratio decile at the end of the game.
- `--log-value`: traces `V(s)` per tick plus its correlation with tick number,
  troop_ratio, and tile_fraction.
- `--headless`: skips the `cv2` window so these run unattended; `--seed` accepts a
  comma-separated list to aggregate across multiple games.

**Bug found via the diagnostics themselves: `watch.py` never applied the same
elimination-is-terminal rule `train.py` uses.** `train.py`'s `run_update()` already
treats `tilesOwned == 0` as done (the wider FFA game runs long after our agent is
wiped out) - `watch.py`'s standalone loop had no equivalent, so after the agent died
it kept "observing" a dead player whose `troops`/`maxTroops` go stale/degenerate
(`troop_ratio` showing as a nonsensical `9999.0`), polluting the diagnostic buckets
(one run's `[0.9,1.0)` bucket was ~1900 samples of this artifact, not real high-ratio
patience). Fixed by mirroring `train.py`'s check.

**Second real bug found: `watch.py` had silently drifted from what `train.py`
actually trains against.** It hardcoded `bridge.reset(..., bots=3, nations=1)` while
`train.py` was training on `bots=0, nations=1` - every diagnostic that session had
been run against a harder, different opponent mix than the checkpoint was ever
trained on. Fixed by importing `BOTS`/`NATIONS` (and later `DIFFICULTY`) from
`train.py` as defaults, overridable via new `--bots`/`--nations`/`--difficulty`
flags, so the two scripts can no longer silently diverge.

**Diagnostic result (update~224, `SHAPING_COEF=0.3`): greedy, not patient.** 5/5
games lost. Every game's first attack landed at tick 1-4, troop_ratio ≈ 0.21 (the
starting value) on every seed. `mean_attack_troop_ratio` 0.004-0.014 - the policy
spends troops as fast as they regenerate rather than letting the ratio climb toward
the 0.422 target. Peak tile share 11-39%. Bumped `SHAPING_COEF` `0.3→1.0` in
response.

**Re-diagnosed with `--log-probs` after the coefficient bump: more nuanced than "no
learning."** `p(do_nothing)` climbs from ~0.6-0.77 near troop_ratio=0 to ~0.93-1.00
above troop_ratio≈0.1-0.2, consistent across 3 seeds - a real, graded,
state-conditioned relationship, not the flat/greedy pattern the raw action counts
alone suggested. Two problems remained visible in the same data: (1) every game
still opens with an attack at tick 1-4 regardless of state - looks like a fixed
habit, not a decision; (2) even below troop_ratio 0.1, `p(attack)` is still 23-39% -
sampled every tick, that's enough to keep re-crashing the ratio before it can climb,
which is why `mean_attack_troop_ratio` stays pinned near zero and peak tile share
caps at 13-18%.

**`--log-value` result: still time-dominated, not state-aware, even after the
troop_ratio fix.** Across 3 seeds: `corr(V, tick)` ≈ -0.95 to -0.98 (strong,
consistent). `corr(V, troop_ratio)` weak and **sign-flips** between seeds (-0.46,
-0.57, +0.64) - troop_ratio isn't reliably driving V's shape, tick still is.
Concretely: in 2/3 seeds troop_ratio climbed to 0.85-0.92 by the end, yet V stayed
flat/pinned near its floor rather than recovering. Honest caveat: all 3 traced games
ended in losses, so a value function correctly predicting an already-doomed
trajectory would *also* show this pattern (discount horizon shrinking toward a fixed
terminal reward) - can't fully rule that out without a winning trajectory to compare
against, which is hard to get at ~3% win rate.

**Diagnosis: a low-outcome-diversity stalemate, not (primarily) a reward-magnitude
problem.** If ~97% of episodes end in a loss regardless of how troop_ratio/tile_
fraction fluctuate mid-game, the value function's training data genuinely doesn't
contain a strong troop_ratio→return relationship to learn - the one thing that *does*
reliably predict return is "closeness to the end," so V settles into tracking tick.
That starves PPO's advantage estimate (`return - V(s)`) of useful signal specifically
where troop_ratio should matter, so the policy gets weak gradient pressure to learn
patience even though the reward shaping is nominally pushing for it. Measured
confirmation that reward-tuning wasn't the lever: relative L2 weight distance
between checkpoints 32 updates apart was ~0.19-0.23 regardless of whether a
`SHAPING_COEF` change happened in that window - the coefficient bump didn't move the
network any more than ordinary continued training does.

**Curriculum experiment: `bots=1, nations=0` (weak Tribe instead of a Nation), tried
and reverted.** Rationale: get real winning trajectories flowing so the value
function has outcome-diverse data to learn from. Immediate result: 4/8 wins at
update 0 (vs. ~3% against a Nation) - confirms opponent strength, not task
difficulty, was gating outcome diversity. **Reverted** after reasoning it through:
Tribes barely fight back (`TribeExecution.ts` - no navy/nukes/diplomacy, attacks
every 40-80 ticks), so the existing greedy/spam-attack habit already wins against
them - PPO would reinforce spamming as "correct" rather than push toward the
patience actually needed, the opposite lesson from what's intended. Back to
`bots=0, nations=1`.

**Difficulty was always implicitly `Easy` - made explicit.** `train.py` never set
`difficulty` in `bridge.reset()`; `harness.ts`'s default (`Difficulty.Easy`, 0.5x
Nation troop strength) was silently in effect the whole session, including for the
6/6 patience-heuristic diagnostic. So the "hoard to 100%, never strike" and "spam
constantly" failure modes aren't a too-hard-opponent problem - patience-then-strike
is proven achievable at exactly this difficulty. Set `DIFFICULTY = "Easy"`
explicitly in both `train.py` and `watch.py` (same fix pattern as the `bots`/
`nations` drift bug above) so it can't silently change later.

**Root cause identified: the entangled action softmax.** `do_nothing` was scored as
just one row among however many attack candidates (`policy.py`'s old
`get_action_and_value` fed every entity, `do_nothing` included, into one
`Categorical`). `p(do_nothing) = exp(score_dn) / Σ exp(score_i)` over *every* live
candidate that tick - so the same learned preference for waiting produces a
different actual probability depending on how crowded the board happens to be,
entangling "should I act at all" with "which target" instead of keeping them as two
separable decisions. Gets *worse*, not better, once non-troop actions (buildings)
are added, since they'd just be more rows in the same shared softmax.

**Fix: gated action head.** `policy.py`'s `Agent` gained `gate_head: Linear(256,
2)` - a dedicated binary wait-vs-act decision, computed once per tick from the
backbone hidden state. Target selection (`query_proj` + the pointer-style
entity-embedding dot product, unchanged) now scores only real (non-`do_nothing`)
candidates; the gate is forced to "wait" when no legal target exists. Log-prob/
entropy of the joint action are computed via the exact hierarchical decomposition
(`H(joint) = H(gate) + P(act)·H(target|act)`, log-prob similarly gated by whether
the sampled/given action was the `do_nothing` index) - verified via a standalone
smoke test before touching real training: probabilities sum to 1, the forced-wait
edge case behaves correctly, log_prob is self-consistent between the just-sampled
and PPO-recompute code paths, and gradients flow through all parameters cleanly.
`get_action_and_value`'s external signature (and `return_probs`'s flattened
per-entity output, for `watch.py`'s `--log-probs`) is unchanged, so no other call
site needed to change.

**`train.py` checkpoint loading now supports a partial (warm) resume.**
`agent.load_state_dict(checkpoint["model"], strict=False)` instead of a strict
load, so an architecture change like adding `gate_head` can carry over every weight
that still matches (backbone, entity encoder, value head, query projection) instead
of forcing a full reset - verified bit-exact transfer of all matching layers against
a real checkpoint before trusting it. **Not currently in use**: reset to fresh
(cold-start) weights instead, deliberately, to isolate the effect of the new action
representation from any confound of a warm-started backbone trained under the old
entangled-softmax scheme. The warm-start path stays available for later.

**Current state: fresh training run, gated action head, `bots=0, nations=1,
difficulty=Easy`, cold-started weights.** Old checkpoints archived (not deleted) to
`training/models/checkpoints_pre_gatedaction_<timestamp>/`. **Not yet validated**:
whether the gated head actually fixes the patience/hoarding problem - that needs the
same `--diagnostic`/`--log-probs`/`--log-value` methodology re-run once this has had
enough updates to say something, plus the trained-vs-dumb-baseline comparison used
to catch the original stuck-learning bug.

**Rollout collection now overlaps with the PPO phase (`train.py`).** Split the old
`run_update()` into `collect_rollout()` and `ppo_update()`, connected by a `Rollout`
dataclass. `train()` runs a `ThreadPoolExecutor(max_workers=1)`: each update submits
the *next* update's `collect_rollout()` (against a separate `collector_agent` instance
holding its own weight copy, refreshed via `load_state_dict()` right before submit)
while `ppo_update()` trains `agent` on the *current* rollout on the main thread, then
joins before the next iteration. Threading (not multiprocessing) because collection is
CPU/pipe-I/O-bound against `VecOpenFrontBridge`, not GPU-bound. `collector_agent`
exists specifically so the collection thread never reads/writes the same parameter
tensors `ppo_update()`'s `backward()`/`optimizer.step()` is concurrently mutating -
`load_state_dict()` copies values into independent storage, verified via a standalone
smoke test (mutate one agent's params on the main thread while the other's forward
pass runs on a pool thread; no corruption). Target win: per-update wall time closer to
`max(rollout, ppo)` than `rollout + ppo`. **Known tradeoff, accepted for now**: rollout
N is collected using `agent`'s weights from *before* update N-1's PPO step, one update
more stale than the within-epoch drift `PPO_EPOCHS`/`CLIP_EPS` already tolerate. Not
adding V-trace-style importance correction unless diagnostics later show this
staleness actually matters.

**`torch.compile` and the bigger `MINIBATCH_SIZE`, tried and reverted - both were
regressions, not speedups.** Benchmarked all three throughput changes together
(`STEPS=500, NUM_ENVS=16`, fresh uncompiled/compiled agents, checkpoints untouched) by
running `collect_rollout()`/`ppo_update()` standalone outside a real `train()` session:

- **Baseline** (sequential, uncompiled, `MINIBATCH_SIZE=256`): 19.79s/update.
- **`torch.compile(dynamic=True)` + `MINIBATCH_SIZE=1024`, still sequential**: 91.85s/
  update - **4.6x slower**.
- **Same, plus the threaded overlap**: 135.56s/update - **6.8x slower than baseline**,
  and *degrading every update* (96s -> 120s -> 191s, rollout `ticks/sec` collapsing
  781 -> 67 across 4 updates).

**Follow-up isolated the two changes - it was `MINIBATCH_SIZE=1024` doing essentially
all of the damage, not `torch.compile`.** The numbers above never isolated which
change caused the regression; a same-shaped re-run with `MINIBATCH_SIZE=1024` alone
(no compile) hit 96.79s/update - **4.7x slower**, with the identical escalating-across-
updates pattern (76s -> 79s -> 74s -> 123s -> 111s) blamed on compile earlier. Since
the mb1024-alone number (96.79s) lands in the same range as the mb1024+compile number
(91.85s), `torch.compile` wasn't adding a meaningful additional penalty on top -
the original writeup's explanation (Inductor kernel-launch overhead) was a plausible-
sounding guess stated as fact without ever running the isolating experiment, and it
was wrong. Best current guess at the real mechanism: a 4x bigger minibatch means a 4x
bigger `(batch, 3, H, W)` spatial tensor per forward/backward pass through
`PPO_EPOCHS=4` - on `mid_plains` that's a meaningful chunk of the RTX 3070's 8GB VRAM
per minibatch, and the escalating-slower-each-update pattern looks like GPU allocator
fragmentation/thrashing building up across updates, not a compile artifact. Not yet
confirmed with a memory profiler - stated as a hypothesis, not a verified cause.

Separately, the threaded-overlap-specific regression (config C, on top of both of the
above) has its own explanation: the overlap design assumed `collect_rollout()` was
I/O-bound against `VecOpenFrontBridge`'s pipes and would release the GIL, letting it
run alongside the GPU-bound PPO thread on the main thread. In fact the per-tick
Python loop does real CPU-bound work between pipe calls (numpy stacking, tensor
construction) that holds the GIL and directly contends with the main thread - the
same class of GIL-contention problem `VecOpenFrontBridge` itself already hit and
solved by switching to `multiprocessing` instead of `ThreadPoolExecutor` for its hot
path (see the parallelization notes above). This part of the diagnosis held up: see
below, the overlap is a confirmed real win once measured without `MINIBATCH_SIZE=1024`
in the mix, and the GIL contention it does have is small enough not to erase that win.

**Reverted**: `MINIBATCH_SIZE` back to 256, `torch.compile` calls removed entirely.
**Kept, and confirmed to actually help once decoupled from `torch.compile`**: the
`collect_rollout()`/`ppo_update()`/`Rollout` split and the threaded overlap. Re-ran
the same `STEPS=500, NUM_ENVS=16` benchmark with `torch.compile` out of the picture:
sequential 21.73s/update vs. overlapped 16.31s/update - a real **24.9% win**, and
unlike the compiled version, *stable* across every measured update (16.0-16.5s, no
drift), confirming the earlier degradation was specific to `torch.compile`
compounding across the two threads, not a property of the threading design itself.
The GIL contention flagged above is real but small: the collector thread's own
rollout phase does measurably slow down under contention (~16-17s vs ~12-14s run
alone), it's just cheap enough that overlapping it with PPO's ~9-11s still nets a win
rather than erasing one. Not pursuing a process-based collector for now given the
thread version already delivers a solid win with much less complexity.

## Licensing note

Upstream OpenFrontIO is AGPLv3 (network copyleft — matters only if we run a modified
version as a public-facing network service; training locally is unaffected).
`/resources` assets are CC BY-SA 4.0. `/proprietary` assets (logos, fonts, music) are
all-rights-reserved and out of scope for this project (headless training needs none of
them). Not legal advice — revisit before any public/commercial release.
