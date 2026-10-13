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

**`STEPS=5000` bump swap-thrashed the machine - `NUM_ENVS` cut 16→8 to fix.**
`spatial_buf` is `STEPS * NUM_ENVS * 3 * H * W` bytes (bool dtype) - at
`STEPS=5000, NUM_ENVS=16` on `mid_plains` that's ~21.6GB, and this box is WSL2, which
caps the VM at 31GB regardless of the 64GB Windows host unless `.wslconfig` overrides
it. Pushed past available RAM and thrashed swap (disk I/O pinned, page faults in the
thousands) rather than failing cleanly. `NUM_ENVS=8` (~10.8GB) fixed it; `STEPS` stays
at 5000 since that's load-bearing for `nations=2` episode completion (see above).

**Multi-enemy representation gap diagnosed and fixed (2026-08-26).** Root cause of
suspected `nations=2` weakness: `is_enemy` was a single flat bit - every opponent's
territory looked identical to the CNN regardless of strength, and `owner_grid` (which
already carries per-tile smallID) was never surfaced beyond that. Fixed:
- `is_impassable` plane added - `GameMap.isImpassable()` was never sent over the
  bridge; `isLand()` alone can't distinguish ownable plains from a permanently-blocked
  wall/mountain, which reads as ordinary neutral land forever. Static per tile
  (`setWater()` refuses impassable tiles), so it's a one-time `impassableBitmap` dump
  at reset (`server.ts`), no per-tick upkeep like `land_mask`.
- `troop_ratio_plane`/`troop_advantage_plane` added (`observation.py`) -
  `owner_grid`-indexed lookup-array gather (`ratio_by_id[owner_grid]`) scatters each
  owner's `troops/maxTroops` onto their tiles anonymously (no per-opponent identity/
  slot, avoids the old `ATTACK_SLOTS` instability class). `troop_advantage_plane` is
  `agent_ratio - troop_ratio_plane`, bounded to roughly `[-1, 1]` - deliberately not a
  raw-troop-count ratio (unbounded, reintroduces the "raw count tracks game time" trap
  `troop_ratio` itself was built to avoid). `spatial` grew 4→6 channels.
- `Backbone`/`Agent` no longer hardcode the conv input channel count - `spatial_channels`
  is now sized from a real observation via `Agent.from_observation`, same pattern
  already used for `scalar_dim`/`entity_dim`. Closes off the "forgot to bump the Conv2d
  channel count" bug class that hit twice in one session (impassable plane, then the
  troop planes).
- **Deferred, not implemented**: `EntityEncoder`'s single-query attention pooling still
  blends every live opponent into one averaged context vector before the gate/value
  heads see it - multi-query pooling (a few learned queries instead of one) was
  designed but deliberately held back until a real `nations=2` run shows the specific
  blending symptom, rather than stacking an unvalidated architecture change on top of
  the representation fix and losing the ability to attribute results to either one.

**Refactor: `EpisodeState`/`apply_reset`/`apply_step` shared between `train.py` and
`watch.py` (`game_state.py`).** `watch.py` had drifted to a different tile/player
update order than `train.py`'s `collect_rollout` (harmless today since the two updates
don't read each other's output, but the same drift class already caused real bugs
here before - the `bots`/`nations` config drift, the missing-elimination-check gap).
`EpisodeState` moved out of `train.py` into `game_state.py` so both scripts import the
same type instead of `watch.py` maintaining ~9 loose local variables in parallel;
`apply_step(episode, step_result)` now does the tile/player-update application +
observation/entity rebuild in one shared call instead of being duplicated inline in
both scripts. `training/envs/bridge_test.py` deleted (superseded by `watch.py`, was
the last caller relying on the optional-`agent_small_id` path in `apply_tile_updates`,
which is now a required arg).

**`server.ts` now throws instead of silently sending `agentSmallID: null`.**
`playerByClientID(agentClientID)` returning `null` is structurally unreachable given
how `reset()` constructs the game (the agent's `PlayerInfo` is added synchronously in
`GameImpl`'s constructor with a hardcoded clientID) - it was defensive code guarding
an invariant that can't currently break, but silently propagating `null` would have
turned a future invariant break into a corrupted-but-not-crashing observation
(`owners == None` numpy comparisons just evaluate false everywhere) instead of a loud
error. Matches the project's existing "propagate env-server errors instead of dropping
them" precedent.

**Two real bugs caught only because a smoke test was run before trusting any of the
above**, both would have crashed or corrupted the very first real training step:
- `apply_reset`'s internal call to `apply_tile_updates` wasn't updated after
  `apply_tile_updates` gained a required `player_stats` first parameter (mid-session,
  for the troop-ratio-plane work) - `TypeError` on every `reset()`.
- `build_troop_advantage_plane` was missing its `return` entirely (implicitly
  returned `None`), and its formula divided raw troop counts rather than comparing
  two `troops/maxTroops` ratios - fixed to derive from `troop_ratio_plane` instead of
  re-scanning `player_stats`, so the two planes can't drift out of sync.

**Critical bug, caught after a live training run already looked suspiciously
cheap on memory: `spatial_buf` (`train.py`) and `SelfImitationBuffer.spatial`
(`self_imitation.py`) were both hardcoded `dtype=bool`.** Correct when every
`spatial` plane was truly binary, but `troop_ratio_plane`/`troop_advantage_plane` are
continuous floats - writing them into a bool array silently truncated every value to
`0`/`1`, destroying almost all the signal those planes were added for. Worse than a
missing feature: the live policy saw the real continuous values when picking each
action, while PPO's log-prob/value recompute during the update trained against the
truncated version - an actor/learner data mismatch, not just a weaker observation.
Fixed to `float16` for both (half the cost of `float32`, still exactly lossless for
the boolean planes). Verified post-fix: `rollout.spatial` dtype is `float16` and
contains real non-binary values, not just `{0, 1}`.

`spatial_buf` at `float16` with 6 channels is 2x the per-element cost of the old
3-4-channel `bool` version, so the existing `NUM_ENVS` sizing no longer holds. At
`STEPS=2000`, `NUM_ENVS=20` (what the corrupted-bool run had been using) computes to
~40GB for `spatial_buf` alone - would have pushed straight into the same
swap-thrashing failure mode as the `STEPS=5000/NUM_ENVS=16` incident above, just from
a different direction (more channels instead of more steps/envs). Cut to `NUM_ENVS=8`
(~16GB), consistent with that incident's resolution.

**Training restarted on `nations=2` (`NATIONS_MIN=NATIONS_MAX=2`), cold-started,
after the fixes above.** Not detouring through `nations=1` first: `nations=1` can't
exercise the new planes at all (no second opponent to disambiguate), so it wouldn't
validate the thing that changed - see the multi-enemy diagnosis above. The one
existing checkpoint (`nations=1`, gated-action-head run, only 32 updates in, and
shape-incompatible with the new 6-channel conv input anyway) archived to
`checkpoints_pre_multitroopplanes_20260826/`, not deleted.

**`watch.py` diagnostic at update192 (this `nations=2` run): still the same greedy/
spam-attack failure mode.** 0/5 wins, mean peak tile fraction 0.406. Every game's
first attack still lands at tick 1-3 (troop_ratio≈0.21, the spawn value) and
`mean_attack_troop_ratio` is 0.016-0.050 across all 5 seeds - far below the 0.422
target, unchanged from the pre-shaping-bump diagnostics. One real improvement though:
`corr(V, troop_ratio)` is now strongly positive in 4/5 seeds (0.45-0.97) - a reversal
from the earlier "value head learned tick number, not game state" finding. So the
critic has learned "high troop_ratio is good" but the actor hasn't converted that into
behavior yet (classic actor-lagging-critic) - `SHAPING_COEF=300` didn't fix it, so the
bottleneck looks like policy-side credit assignment, not the reward signal's existence
or magnitude.

**Idea (unimplemented, deliberately deferred): explicit troop-density entity feature.**
Confirmed the engine's combat resolution genuinely depends on troop density, not just
raw troop count - `AiAttackBehavior.ts:713`'s `defenderTroopLoss = defender.troops() /
defender.numTilesOwned()` feeds directly into combat outcomes, and the built-in
scripted AI already uses this exact density formula (`AiAttackBehavior.ts:447`) to
pick weak targets. There's also a separate large-territory debuff past 100k tiles
(`:692-711`, sigmoid-based) independent of density. The agent's current per-entity
features (`action_space.py`) have `tile_fraction` (`/total_land`) and `troop_ratio`
(`/maxTroops`) as two separately-normalized values - true density (`troops/
tilesOwned`) isn't directly derivable from them without reconstructing
`troop_ratio*maxTroops / (tile_fraction*total_land)`, a nonlinear combination unlikely
for a small per-entity MLP to discover from reward alone. Real missing signal, not a
redundant one. Relevant beyond targeting quality too: once `DefensePost`/`City`
structures land, effective defense-per-tile becomes even more load-bearing, so this is
worth building as the density feature naturally generalizes to "defended density"
later. Deliberately held off implementing now (would change `entity_dim`, forcing
another cold restart) - do alongside the next architecture change rather than
interrupting the current run.

**Recurring PPO collapse: a real winning streak destabilizes and reverts to ~0% win
rate.** `watch.py` diagnostics at update192/224 (this `nations=2` run) looked
identical to a stagnant plateau (0/5 wins both times, same first-attack-tick, same
low mean_attack_troop_ratio) - but the full per-update log told a different story:
the policy actually learned real patience and won regularly from update82-186 (many
updates with 2-7/10-15 episode wins, e.g. update160: 6/8, update164: 7/10), then
collapsed to ~0 wins for the 50+ updates since. Right before the collapse (updates
161-187), `kl_stopped` events clustered (`approx_kl` 0.02-0.06, well over
`TARGET_KL=0.02`) - 6 of 27 updates, vs. almost none before or after. Checkpoint
weight-distance check (update192 vs 224, 32 updates apart) showed normal ~16%
relative L2 movement - not a stalled optimizer, a real but unproductive drift.
Confirmed via `watch.py` that `update160` (archived to
`checkpoints_prewinstreak_collapse_20260827/`) genuinely plays better (1/5 win,
`mean_attack_troop_ratio` up to 0.245, peak_tile_fraction 0.807 on the winning seed)
than `update192`/`224` (0/5, near-zero attack-time troop ratio) - a real regression,
not noise. This is the same failure class already hit twice before in this project
(the `update416` lineage that "falls off" whenever PPO fine-tuning resumes on it; the
`VALUE_COEF` 10x jump's ~300-update entropy collapse) - a winning policy destabilizes
under continued PPO updates, and once it does, the loss of outcome diversity makes it
self-sustaining (no more winning trajectories to reinforce recovery).

**Root-cause mechanism (best current understanding):** advantage normalization
(`(advantages-mean)/std`, global across the rollout) rescales the *overall* magnitude
but doesn't reshape a heavy-tailed distribution - once real wins mix into mostly-loss
rollouts, win-episode advantages remain large outliers even after normalization.
Worse: GAE with `GAE_LAMBDA=0.95` propagates a winning episode's terminal advantage
backward across nearly its *entire* trajectory, so a single win elevates thousands of
correlated timesteps at once, and `torch.randperm` spreads those into every minibatch
that epoch - a shared, correlated push across the whole update, not a rare isolated
outlier.

**Fix attempt 1: `ADV_CLIP=8.0`** (clamp normalized advantages post-normalization,
`train.py`). Resumed from `update160`. Same `kl_stopped`/`approx_kl` 0.02-0.03
pattern recurred within 15-18 updates, win rate reverted to ~0 for the next ~18+
updates - looked like the same collapse, just faster, though caveat: the original
collapse only followed a much longer (~100-update) sustained winning stretch, so a
single good update reverting isn't necessarily proof of the same failure mode acting
identically. Per-sample clipping alone evidently doesn't stop the *correlated,
many-samples-at-once* version of the problem.

**Fix attempt 2: `GAE_LAMBDA` 0.95 -> 0.90**, on top of `ADV_CLIP` (both active),
resumed again from `update160` (the second attempt's collapsed checkpoint archived to
`checkpoints_postcollapse_20260827/...attempt2.pt`). Delayed the collapse but didn't
prevent it: win rate sustained ~32 updates (roughly double attempt 1's ~15-18) with
`mean_reward` frequently positive, then `kl_stopped` at update33 (`approx_kl=0.0215`)
was followed by 0 wins for the next 7+ updates straight, the same reversion signature
as both prior attempts. Genuine partial evidence the lambda reduction helps (longer
sustain), but not a fix on its own.

**Both fixes reverted (`ADV_CLIP` and `GAE_LAMBDA` back to 0.95)** - neither solved
the collapse outright, so removed rather than left in as unproven clutter. Each
delayed the failure point (never wins -> ~15-18 updates -> ~32 updates) without
removing the cause, which is itself a useful data point: the mitigations are treating
a symptom (single/correlated large-advantage samples) whose underlying trigger is
still unidentified. Next step is root-causing before trying another mitigation, not
another parameter-tweak-and-rerun cycle.

**Root cause found and fixed: `SHAPING_COEF=300`'s per-attack penalty, not GAE
propagation or advantage magnitude.** Instrumented a real rollout
(`training/models/watch.py`'s underlying loop, hooked to log raw shaping reward vs.
per-tick troop/maxTroops deltas) on both a winning and a losing seed under `update160`.
Ruled out the earlier `maxTroops`-denominator-artifact theory outright -
`corr(|shaping|, |Δmaxtroops|) ≈ -0.01`. What's actually driving every large spike
(other than the universal tick-0 troop-initialization one): `curr_troops ≈ 0.8 ×
prev_troops` with `maxTroops` unchanged - exactly `Config.attackAmount()` (20%-of-
troops per attack), not a bug. Since attacks mostly happen while `troop_ratio` is
below `TROOP_RATIO_TARGET=0.422` (true for most of the game per every prior
diagnostic), each one costs `-5.5` to `-5.8` in shaping - over a third of the *entire*
full-climb-from-spawn shaping total (`~15`), in one tick. A winning episode's
multi-attack finishing sequence can rack up `-25` to `-50+` this way, fighting
directly against the `+150` win reward within the same trajectory. Best explanation
for why this specifically destabilizes once wins appear (frequency of these spikes
was actually slightly *higher* in a losing-seed control, 2.3% of ticks vs. 1.6%): the
value function had recently become genuinely troop_ratio-aware (see the
`corr(V,troop_ratio)` finding above), so in a trajectory trending toward a win, `V(s)`
already expects a good outcome - the same `-5.5` reward dip is then a *bigger*
surprise/TD-error against a confident high baseline than the identical dip is in an
already-low-value losing trajectory the critic doesn't expect much from anyway.

**Fix: soften (don't zero out) the penalty within one attack's grace distance of the
floor** (`reward.py`'s `_troop_potential`) - flat-zero at/above `TROOP_RATIO_TARGET`
unchanged, but the deficit produced by exactly one attack
(`TROOP_RATIO_GRACE = TARGET × ATTACK_COST ≈ 0.0844`) is now scaled to 25% of the full
quadratic, with the full quadratic resuming (continuous at the boundary) only for
deficits beyond that - so a single unavoidable attack is much cheaper, but
compounding/chronic below-floor attacking still escalates. **Tried and reverted a
second idea first:** a symmetric bonus for banking above `TARGET`, to directly "bias
toward more than 42%" as requested - potential-based shaping telescopes to zero over
any round trip (Ng-Harada-Russell), so a bonus for building a buffer is
mathematically forced to be clawed back in full the instant it's spent. Verified
numerically before it ever ran: a clean attack from a well-prepared ratio down to
exactly `TARGET` went from *0 penalty* (already fine under the original formula) to
**-31.65** with the bonus added - worse than any real spike found, for the best-case
scenario. There's no free lunch for "reward holding high, don't penalize spending it"
under a state potential; dropped that half of the idea. Verified the grace-softening
fix against real logged spikes before trusting it: `tick140` case -5.676 -> -3.362,
`tick1321` case -5.625 -> -2.938, the well-prepared-buffer case stays exactly 0 (no
regression), a single clean attack from exactly `TARGET` -2.14 -> -0.534. Aggregate
effect on a full winning-seed rollout: shaping `std` 0.88 -> 0.64, worst spike -5.68
-> -3.68. Resumed training from `update160` with this fix alone (both `ADV_CLIP` and
`GAE_LAMBDA` already reverted) so any effect on the collapse can be attributed to the
reward fix specifically, not confounded with the earlier PPO-side attempts.

**Reward-shaping fix (grace-softened penalty) tested through update51: did not solve
the collapse either.** Wins were strong through update11 (mostly 4-6/update,
`mean_reward` frequently positive), then reverted starting update12 - earlier than
the `GAE_LAMBDA` attempt's update32, comparable to `ADV_CLIP`-only. Four different,
individually-verified fixes (cold continuation, `ADV_CLIP`, `GAE_LAMBDA`, and the
reward redesign) have now all eventually reverted to the same 0-win pattern after a
`kl_stopped` cluster, at roughly comparable timescales (12-32 updates) - strong signal
none of these mechanisms is the actual root cause, just things that perturb when it
hits. Root-causing needs a different angle than another targeted parameter change.

**`MAX_EPISODE_TICKS=10000` episode cap added** (`train.py`, `EpisodeState.ticks` new
field in `game_state.py`) - a stopgap, not a rigorously-correct time-limit-bootstrap
implementation. Motivated by a transient memory spike mid-run (35GB used vs. the
normal ~20GB baseline, self-resolved once long-running episodes finished) - games can
otherwise run to the engine's own ~170-min hard cap. At the cap, the episode is force-
ended with `won=False` (so it reads as a loss for reward purposes) rather than
properly bootstrapping `V(s)` for a true timeout - a known simplification, flagged via
a `[episode-cap]` print every time it fires so frequency is visible: if these prints
show up a lot, 10000 is cutting off real games and should be raised (or the timeout
handling should be done properly instead of approximated as a loss).

**Historical-log analysis (fork, ~62k lines / 1660 update records across 12 runs)
found 9 independent collapse events predating this session's four fix attempts** -
spanning 2026-08-24 onward, across at least three different reward/architecture
configs. This means none of the four mechanisms tried this session (cold
continuation, `ADV_CLIP`, `GAE_LAMBDA`, reward-shaping softening) was ever likely to
be sufficient alone - the pattern outlived all of them. Specifically checked and
**rejected**: entropy collapse right before the break (ranges 0.25-0.43 across all 9
cases, never near a floor - the policy isn't going deterministic/overconfident).
**Also found**: `kl_stopped` is a routine, harmless byproduct of any elevated-win-rate
batch, not a rare acute trigger - one 119-update streak hit it 8 times with no issue,
and the last occurrence in that run was 27 clean updates before the eventual
collapse. No pattern found in `policy_loss` sign, `episodes_finished`,`ticks/sec`, or
`sil_loss` timed to collapse onset. Conclusion: this looks like cumulative,
path-dependent instability, not a single-update magnitude problem.

**`TARGET_KL`/`kl_stopped` removed entirely** (`train.py`) given the above - it was
never real protection for this failure mode (only measures divergence on the
*current* rollout batch, which narrows to the winning-trajectory distribution as win
rate rises). Per ["RL Forgets! Towards Continual Policy Optimization"](https://arxiv.org/abs/2607.04364):
"the KL regularization used in common policy optimization methods is evaluated on
current-task data, whereas forgetting is caused by behavioral drift on prior-task
distributions" - exactly this project's blind spot. `approx_kl` still computed and
logged every update, just no longer acted on. Also relevant from the same search:
PPO's critic can develop "value churn"/misgeneralization as the visited-state
distribution narrows (general PPO literature), and "as the policy improves, crashes
happen less frequently, so the agent begins to forget about catastrophic states"
(self-play collapse literature - mechanism differs, since our opponent is a fixed
scripted bot not a co-adapting learner, but the narrowing-coverage dynamic
transfers). Working hypothesis going forward: nothing in the training loop currently
keeps the network anchored to the broader state distribution (losing/danger states)
once win rate rises and the on-policy rollout narrows around winning trajectories -
`SIL_BUFFER` only stores wins, doesn't serve this anchoring purpose. A general
(non-win-filtered) replay mixed into every PPO update is the leading candidate real
fix, not yet implemented.

**`SHAPING_COEF` 300 -> 100 tested** (`reward.py`), as a distinct experiment from the
grace-softening reshape above - that reshaped the curve near the floor and still
collapsed at update12 (if anything faster than `GAE_LAMBDA`, which didn't touch
shaping at all), evidence against magnitude being the dominant driver on its own.
Testing a straight cut to see whether collapse persists at the same rate even with
shaping much weaker (pointing further toward the structural explanation above) or
is meaningfully delayed (reopening reward magnitude as a real contributing factor
after all). Resumed from `update160`, `kl_stopped` already removed. Not yet
validated.

**Added `value_err_low_ratio`/`value_err_normal`/`frac_low_ratio` diagnostics**
(`train.py`) - direct test of whether value-function calibration on the rare/low-
troop-ratio tail degrades relative to the common bucket, rather than inferring it
from silence in the aggregate `value_loss`. First real result from this run
(`SHAPING_COEF=100` + `kl_stopped` removed, resumed from `update160`): a genuine
collapse occurred (wins=0 for 32 updates, 39-70), and the two error curves visibly
converged right at the transition (ratio ~0.78-0.80 at update 39-41, vs. ~0.5-0.6
before/after) - the clearest live evidence yet for the mechanism, though the
"low-ratio bucket specifically gets neglected" framing was too narrow: once
collapsed, `frac_low_ratio` stayed high (0.72-0.85, low-ratio became the *majority*
bucket) and `value_err_low_ratio` stayed *smaller* than `value_err_normal`
throughout, both before and after - no clean flip, just the transient convergence at
the actual moment of change. Correct framing: whichever bucket is *currently rare*
in the rollout is the one that goes uncalibrated, and that role can belong to either
bucket depending on what the policy happens to be doing.

**Self-recovery observed: update71 (14 updates after the deepest point of the
collapse), sustained through at least update84** - wins returned to a normal 1-6/
update range, `mean_reward` positive on several updates. Recovered in ~30 updates
total (39→71), much faster than the one earlier ~300-update precedent. The run was
killed at update84 before a checkpoint could capture the recovered state (checkpoints
save every 32; next would have been update96) - lost that specific improvement, but
confirms recovery-without-intervention is real and can be fast under these settings,
not just a one-off historical fluke.

**This directly undercuts a proposed fix and pointed at a better one.** A
"general (non-win-filtered) replay buffer" idea was proposed to keep the value
function calibrated on rare states - flagged by the user as illogical: the rollout
during a 30+-update collapse is already almost entirely loss data, so "add more loss
examples" can't address a scarcity that doesn't exist. Correct conclusion instead:
since abundant loss data alone doesn't self-correct quickly, the actual problem is
that continued gradient descent on current data isn't sufficient by itself - pointing
at either external intervention (rollback-on-detection) or something actively
fighting recovery, not a data-coverage gap.

**`SIL_MAX_AGE_UPDATES=4` expiration added** (`self_imitation.py`, `train.py`) -
targets a specific, checkable candidate for "something actively fighting recovery":
`SIL_BUFFER` only refills from wins, so during a 30+-update collapse it keeps
replaying increasingly stale pre-collapse transitions against a policy/value function
that's drifted well away from that distribution - two competing gradient signals
(fresh on-policy loss vs. stale SIL loss) fighting over the same shared backbone.
Honest caveat: the observed ~30-update self-recovery happened with SIL active and
presumably stale the entire time (`sil_buffer=3000`, full, no signs of early
refresh) - real evidence against SIL being a hard blocker, though it doesn't rule out
SIL slowing recovery down. `sample()` now returns `None` for a step with nothing
within `SIL_MAX_AGE_UPDATES` updates old, rather than falling back to stale data;
`sil_stale_skipped` logged per update for visibility. Smoke-tested the age-cutoff
logic (samples available through age 3, correctly `None` at age 4+) before trusting
it. Resumed from `update64` (this run's own latest checkpoint, itself mid-collapse -
not an ideal baseline for testing recovery speed, but is what "latest" meant at the
time). Not yet validated.

**Rollout buffer: `is_land`/`is_impassable` stored once per rollout, not per tick**
(`Rollout.static_planes`, `reconstruct_spatial_torch`). Per-tick bool planes 4→2
(`is_mine`, `is_enemy`), 6→4 bytes/px overall, anchor/low-ratio buffers shrink too.
Verified bit-exact against `encode_observation` on synthetic input, then run
through 100+ live PPO updates. `del rollout` after each member's update drops peak
RSS 31→12GB at 4 envs; `NUM_ENVS_PER_MEMBER` now 10 (~26GB, ~440 ticks/s). **Assumes land never changes mid-episode and the map is constant - will
break with water nukes** (the live actor sees the updated `land_mask`, PPO recompute
would see the stale one).

## Licensing note

Upstream OpenFrontIO is AGPLv3 (network copyleft — matters only if we run a modified
version as a public-facing network service; training locally is unaffected).
`/resources` assets are CC BY-SA 4.0. `/proprietary` assets (logos, fonts, music) are
all-rights-reserved and out of scope for this project (headless training needs none of
them). Not legal advice — revisit before any public/commercial release.
