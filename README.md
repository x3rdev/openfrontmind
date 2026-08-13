# OpenFrontMind

Training a reinforcement-learning agent to play [OpenFront](https://openfront.io) — a
real-time territorial strategy game — via self-play, in the spirit of DeepMind's
AlphaStar. Scaled down to fit a single local GPU, but built on the same bones: a
deterministic game engine driven headlessly, a vectorized self-play environment, and
a policy trained with PPO over a learned entity-attention encoder rather than a
hand-tuned feature set.

This is a from-scratch RL project, which means every architectural decision below was
earned by finding something broken first. The sections that follow are as much a log
of *how* those things were found as a description of the current system.

## Architecture

```
engine/   vendored, unmodified OpenFrontIO simulation core (deterministic TypeScript)
env/      headless Node harness — drives the engine, speaks a bridge protocol over stdio
training/ Python/PyTorch side — vectorized envs, PPO training loop, policy network
```

- **The game engine is untouched.** `engine/` is the real OpenFrontIO simulation core,
  not a simplified reimplementation — deterministic, seeded, no floats. What the agent
  learns to beat is the actual game, not an approximation of it.
- **Vectorized self-play.** `VecOpenFrontBridge` runs N independent game instances as
  separate OS processes (not threads — the sim is CPU-bound single-threaded JS, and
  Python's GIL makes threads a poor fit for the per-tick decode work), batching
  observations into single GPU forward passes for policy inference.
- **PPO with GAE**, clipped surrogate objective, clipped value loss, entropy
  regularization — the standard modern recipe, plus a custom network built for a game
  where "how many opponents exist right now" is not a fixed number.

## Progress log

### Phase 1 — Infrastructure

Built harness around open source game engine without editing original game. This 
helps with adapting to future updates. Since game runs in typescript and can
run for very long, parallelized training process early.

### Phase 2 — The reward-scale bug

Entropy pinned at `0.0000` from update zero, zero wins in ~290 episodes. Cause:
`WIN`/`LOSS` reward of `±10,000` swamped the shared-backbone gradient via
`VALUE_COEF`, next to per-tick rewards two orders of magnitude smaller. Rescaled to
`±100` — value loss dropped from 1e5-1e6 to normal, entropy stopped collapsing.

### Phase 3 — The horizon diagnostic

Reward fixed, but wins stayed at zero and reward stayed flat. Diagnostic: a fixed
greedy policy peaked at 62.5% map share by tick 95, then fully eliminated by tick
1951 — an ~1,850-tick arc. Rollout horizon was 500 ticks, so peak and collapse never
appeared in the same batch. Lengthened `STEPS` 500→2000, retuned `GAMMA` to match —
reward went positive immediately.

### Phase 4 — First wins

With reward and horizon fixed, a 720-update run against mixed Tribe/Nation opponents
got the first real wins: 26/13,154 episodes. Low, but proof the agent could close
out a game, not just accumulate reward.

### Phase 5 — The dumb-policy check

A sub-1% win rate could mean "hard task" or "no learning at all." Built a
zero-intelligence baseline (attack wilderness, else nearest target, else wait) and
ran it against a 300-update checkpoint on identical seeds: both hit the same 36.8%
peak tile share, same tick (487), same loss. Confirmed over 5 more runs. 300 updates
of training were statistically indistinguishable from no policy at all.

### Phase 6 — Root cause and the entity-encoder rewrite

Cause: attack targets were picked by position in a fixed 12-slot list built from
`Player.nearby()`, whose order depends on `Set` iteration order — reshuffles as
territory changes, so slot 3 could mean a different opponent tick to tick in the
same game. A feedforward net scoring fixed slots had no stable target to learn a
preference for. Fix: entity encoder + pointer-network action head, scoring
candidates by their own features instead of index.

### Phase 7 — Efficiency pass

Parameter audit found the backbone's flattened conv output was ~99% of the
network's params on a larger map, and tied its shape to one map resolution.
Replaced with adaptive average pooling to a fixed grid: 5.3M → 11.9M (bigger map +
entity encoder) → 414K params. Network no longer tied to a specific map size.

### Phase 8 — Throughput

Benchmarked rollout throughput vs. env count (16-core machine):

| envs | ticks/sec |
|---|---|
| 8 | 971 |
| 16 | 1,452 |
| 24 | 1,705 |
| 32 | 1,890 |
| 40 | 1,970 |
| 48 | 1,952 |

Plateaus at 32-40 envs. Bigger lever found: rollout and PPO ran strictly
sequentially, leaving the GPU idle during rollout and envs idle during PPO — an
async actor/learner split would fix that independent of env count.

### Phase 9 — Current state

Added value-loss clipping (mirrors the policy-ratio clip; motivated by a real run's
value loss sawtoothing on every episode end). Upgraded the training map — the
original was ~100x smaller than a real game, too dense for even a perfect heuristic
to win reliably.

### Phase 10 — Entity encoder re-tested: still stuck

Re-ran the dumb-policy comparison against the entity-encoder architecture (500
updates). Same result: matched the non-learning heuristic almost exactly. The
slot-identity bug was real but not the whole story.

### Phase 11 — The value function was learning tick number, not board state

Traced `V(s)` through a real game: declined smoothly the whole episode, including
while tile share was still climbing. Cause: observation only exposed raw troop
count, which grows with time regardless of strategy. Fixed: added
`troops/maxTroops` ratio (what the game's own AI gates attacks on) to the scalar
and entity features.

### Phase 12 — Patience beats greed, proven outside the network

Built a second heuristic — wait until troop ratio > 50%, then attack — against the
always-attack heuristic, 6 seeds, no learning involved. Patient: 6/6 wins. Greedy:
6/6 losses. Disproved the "needs buildings" theory and set a concrete ceiling
(~80% peak tile share, 100% win rate) to hold the trained policy against.

### Phase 13 — Rollout throughput was flat, not just slow

`NUM_ENVS` 24→32 gave zero throughput gain (~550 ticks/sec at both 8 and 32).
Cause: `encode_observation()` recomputed full spatial planes from scratch every
tick instead of updating changed tiles. Fixed incrementally, verified bit-exact
against the old method. Real scaling now holds to `NUM_ENVS=24` (925 ticks/sec),
falls off at 32 (16-core machine 2x oversubscribed). Set `NUM_ENVS=24`.

### Phase 14 — First full-scale run, and a new diagnostic toolkit

First run at real settings (`STEPS=3000`, `NUM_ENVS=24`) failed immediately: the
spatial rollout buffer was `float32` for binary planes, trying to allocate 48GB on
a 31GB machine. Fixed by storing as `bool` (4x smaller, lossless).

Win rate then stayed near zero for hundreds of updates with no way to tell why.
Added three diagnostic modes to `watch.py`:

- `--diagnostic` — troop_ratio/tile_fraction at decision time, attack vs. wait
  counts, first-attack timing.
- `--log-probs` — `p(wait)` vs. `p(attack)`, bucketed by troop_ratio.
- `--log-value` — `V(s)` traced per tick, correlated against tick/troop_ratio/tile
  share.

Building these found two bugs: `watch.py` didn't treat agent elimination as
episode-end (`train.py` did), and `watch.py` had drifted to a different opponent
mix (`bots=3, nations=1`) than what `train.py` actually trained on
(`bots=0, nations=1`) — every prior diagnostic run had measured the wrong thing.

With that fixed:

| check | result |
|---|---|
| action pattern | attacks within tick 1-4 of every game, troop_ratio ≈ 0.21 (starting value) |
| `p(wait)` vs. troop_ratio | real and graded — 0.6-0.77 near ratio=0, rising to 0.93-1.00 above ratio≈0.1-0.2 |
| but | `p(attack)` never fully dies out below ratio 0.1 (23-39%) |
| `corr(V, tick)` | -0.95 to -0.98, strong and consistent |
| `corr(V, troop_ratio)` | weak and sign-flipping between seeds |

Real, graded patience existed, but the value function still couldn't tell strong
from weak positions, and a residual near-zero-troop attack habit kept undoing it.
With ~97% of episodes ending in a loss regardless of troop_ratio, the value
function had no strong troop_ratio→outcome signal to learn from, so it fell back
to tracking time-to-end instead.

### Phase 15 — A curriculum experiment, tried and reverted

Tried a weaker opponent (`bots=1, nations=0`) to get more winning outcomes to
learn from — worked immediately (4/8 wins at update 0 vs. ~3% against a Nation)
but reverted: Tribes barely fight back, so the existing spam-attack habit already
beats them, reinforcing the wrong lesson.

Found along the way: `train.py` never explicitly set match difficulty, so training
had silently been running on `Difficulty.Easy` the whole time. Set explicitly.

### Phase 16 — The actual structural bug, and the fix

Real cause: `do_nothing` was scored as one row in a single softmax with however
many attack candidates existed that tick — same learned preference produced a
different actual probability depending on board crowding. Fix: dedicated
`gate_head` making a binary wait-vs-act decision first, independent of candidate
count; target selection only runs once the gate says act. Verified with a smoke
test before training.

Reset to fresh weights on `nations=1`/Easy to isolate the fix from any confound.
Old checkpoints archived, not discarded.

### Phase 17 — First real wins, then a policy collapse

Tried `torch.compile` + a bigger PPO minibatch for throughput. Both were
regressions (4.6x slower combined; isolating them showed the minibatch size was
the real cause). Reverted. Overlapping rollout collection with the PPO phase on a
background thread did work — a stable ~25% reduction in per-update time — and was
kept.

Resumed training on `nations=1`, `NUM_UPDATES` raised to 10,000. After ~400
updates of flat/declining win rate, it broke through around update 418 — several
updates hit 100% win rate, the first real evidence of a working strategy. Within
~15 updates entropy collapsed to near zero and win rate fell back to 0%: a large
win-heavy batch produced an oversized policy update that overfit, with no
exploration left to recover. The `update416` checkpoint was archived before
rotation could prune it.

Also noticed troop hoarding — `p(do_nothing)` staying ~0.99 even at high troop
ratios, gold banked past 1M unspent. Likely `SHAPING_COEF=1.0` overweighting the
per-tick patience reward against the sparse win/loss signal. Dropped to `0.3` and
restarted.

## What's next

- Confirm the `SHAPING_COEF` change fixes hoarding without reintroducing
  under-patience; still no fix for the entropy-collapse mechanism itself
  (candidates: KL-based early stopping, an entropy floor).
- Re-run the diagnostic toolkit once win rate stabilizes.
- Buildings and economy actions, once win rate is stable enough to build on.

## Credits / license

Built on [OpenFrontIO](https://github.com/openfrontio/OpenFrontIO) (AGPLv3). This
project is a research/learning exercise, not a redistribution of the game — see
`CLAUDE.md` for licensing notes on vendored assets.
