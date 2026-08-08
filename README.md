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

## The core technical problem: an unbounded action space

OpenFront doesn't have a fixed number of opponents — a game might have 2 players or
20, and that count shrinks as players get eliminated. Most RL action spaces assume a
fixed, enumerable set of actions. This project's action space is instead built around
an **entity encoder**: every legal attack target this tick (however many exist) is
turned into a feature vector — opponent type, tile share, gold, troop count — and run
through a shared per-entity network, pooled via learned attention into a single
context vector, in the style of Deep Sets / Set Transformer. Action *selection* then
works the same way AlphaStar's does for unit targeting: a pointer-network head scores
each candidate by its own embedding and picks from exactly as many options as exist
that tick — no fixed action count, no padding tricks that leak into what the model
can learn.

That design didn't arrive first, though. It was the fix for a very specific,
very stubborn bug — see below.

## Progress log

### Phase 1 — Infrastructure

Vendored the engine, built the stdio bridge protocol, and got a raw `reset`/`step`
loop working end-to-end in-process before ever touching a neural net. Parallelized
early: rollout collection across N subprocess-driven game instances, one GPU forward
pass per tick across the batch.

### Phase 2 — The reward-scale bug

First training attempts: entropy pinned at exactly `0.0000` from update zero, no
wins across ~290 episodes. Root cause wasn't a hyperparameter to tune — a
`WIN`/`LOSS` reward of `±10,000` next to a per-tick reward two orders of magnitude
smaller was swamping the shared-backbone gradient through `VALUE_COEF`. Rescaled to
`±100`; the value loss dropped from the 1e5–1e6 range to something sane, and entropy
stopped collapsing.

### Phase 3 — The horizon diagnostic

Even with the reward fixed, wins stayed at zero and reward stayed flat. Ran a
single-opponent diagnostic episode under a fixed (non-learning) greedy policy to see
what a *real* game trajectory looked like: **peak 62.5% map share at tick 95, full
elimination by tick 1951** — an ~1,850-tick rise-and-collapse arc. The rollout
horizon at the time was 500 ticks. The value function structurally could not connect
a mid-game peak to its eventual collapse because the two events never appeared in the
same training batch. Lengthened the horizon (`STEPS` 500 → 2000, `GAMMA` retuned to
match) — reward flipped from flat to consistently positive immediately.

### Phase 4 — First wins

With the reward and horizon fixes in place, a 720-update training run against a
mixed Tribe/Nation opponent pool recorded the first genuine wins: **26 out of 13,154
episodes.** Not a high win rate — but the first evidence the agent could close out a
game it had been winning, not just accumulate reward.

### Phase 5 — The dumb-policy check

A win rate under 1% is consistent with "the task is hard" — but it's also consistent
with "the model isn't learning anything at all." To tell them apart, a
zero-intelligence baseline was built: a fixed heuristic (*attack wilderness, else the
nearest target, else do nothing*) with no neural net involved. Run head-to-head
against a checkpoint from **300 updates and ~2,700 episodes of training**, on
identical seeds:

| | peak tile share | peak tick | outcome |
|---|---|---|---|
| dumb heuristic | 36.8% | 487 | LOSS |
| trained policy | 36.8% | 487 | LOSS — same opponent |

Not approximately the same — the same, down to the tick. Re-run five more times to
rule out the trained policy's own action sampling explaining it away: peak tile share
across all five held to a 0.5-point band, and all five lost to the same opponent.
**300 updates of training had produced a policy statistically indistinguishable from
having no policy at all.**

### Phase 6 — Root cause and the entity-encoder rewrite

Traced it into the engine's own iteration order. The old action space picked attack
targets by their position in a fixed 12-slot list, built fresh from
`Player.nearby()` every tick — and `nearby()`'s return order depends on the
iteration order of a `Set` that gets rebuilt as territory shape changes. Slot 3 could
mean a different opponent from one tick to the next, *in the same game*. A
feedforward network scoring fixed slot indices had no consistent target to attach a
learned preference to — it wasn't failing to learn "attack the weak target," it
structurally couldn't represent that idea. This is what the entity encoder and
pointer-network action head (described above) were built to fix: score candidates by
their own features, not by a meaningless index.

### Phase 7 — Efficiency pass

A parameter audit while building the fix above turned up a second issue: the
network's backbone flattened its full convolutional output straight into a dense
layer, which had grown to **~99% of the entire network's parameters** on a larger map
— and tied the network's architecture to one specific map resolution. Replaced with
adaptive average pooling to a fixed grid. Net effect:

| | params |
|---|---|
| original backbone | 5.3M |
| + entity encoder (bigger map) | 11.9M |
| + pooling fix | **414K** |

A ~29x reduction, and the network is no longer coupled to a specific map's
dimensions at all.

### Phase 8 — Throughput

Benchmarked rollout-collection throughput against parallel environment count on a
16-core machine:

| envs | ticks/sec | Δ |
|---|---|---|
| 8 | 971 | — |
| 16 | 1,452 | +50% |
| 24 | 1,705 | +17% |
| 32 | 1,890 | +11% |
| 40 | 1,970 | +4% |
| 48 | 1,952 | flat/regressed |

A clean diminishing-returns curve, plateauing at 32–40x parallelism on this hardware.
Also identified the next, bigger lever: rollout collection and the PPO update
currently run strictly sequentially, leaving the GPU idle during rollout and every
environment worker idle during training — an async actor/learner split (IMPALA-style)
would reclaim that for free, independent of environment count.

### Phase 9 — Current state

Value-loss clipping added (mirroring the existing policy-ratio clip — motivated by
directly observing the value loss sawtooth on every episode completion in a real
run). Training map upgraded after discovering the original map was roughly two
orders of magnitude smaller than a typical real game, dense enough that even a
perfect heuristic couldn't reliably win it.

### Phase 10 — Entity encoder re-tested: still stuck

Re-ran the dumb-policy comparison against the entity-encoder architecture (500
updates). Same result as before the rewrite: peak tile share and timing matched the
non-learning heuristic almost exactly, every game lost to the same opponent. The
slot-identity bug was real and worth fixing, but it wasn't the whole story.

### Phase 11 — The value function was learning tick number, not board state

Traced `V(s)` tick-by-tick through a real game: declined smoothly for the whole
episode, including the first 140 ticks while tile share was still climbing. A value
function tracking "expected return at this point in time" instead of "given this
board state" breaks PPO's advantage estimate no matter how good the policy is.
Cause: the observation only ever exposed raw troop count, which grows with elapsed
time regardless of strategy — an easy shortcut for a value function to latch onto.
Fixed: added `troops/maxTroops` ratio (what the game's own scripted AI actually
gates attacks on) to both the agent's scalar observation and each entity's features.

### Phase 12 — Patience beats greed, proven outside the network

Built a second hardcoded heuristic — wait until troop ratio crosses 50%, then attack
— against the original always-attack heuristic, 6 identical seeds, zero learning
either way. Patient: 6/6 wins. Greedy: 6/6 losses. Disproved the "the action space
needs buildings" theory, and handed the trained policy a concrete ceiling to hold
itself against (~80% peak tile share, 100% win rate) using a signal it can now see.

### Phase 13 — Rollout throughput was flat, not just slow

Bumping `NUM_ENVS` 24→32 for an overnight run didn't help: `ticks/sec` was identical
(~550) at 8 envs and at 32 — zero parallelism benefit. Cause: `encode_observation()`
recomputed the full spatial planes from scratch every tick, per env, in a serialized
loop, instead of updating only the tiles that changed. Fixed by maintaining those
planes incrementally; verified bit-exact against the old computation over 400 real
ticks. Re-benchmarked: real scaling now shows up to `NUM_ENVS=24` (925 ticks/sec),
with a hard falloff by 32 (629 — a 16-core machine oversubscribed 2x). `NUM_ENVS`
set to 24.

### Phase 14 — First full-scale run, and a new diagnostic toolkit

The first run at real settings (`STEPS=3000`, `NUM_ENVS=24`) failed before it even
started: the spatial rollout buffer was allocated `float32` for what are purely
binary planes, trying to claim 48GB on a 31GB machine. Fixed by storing it as `bool`
(lossless — the values are always exactly 0 or 1), a 4x reduction with no precision
cost.

With that fixed, the run itself exposed a harder problem: win rate stayed near zero
for hundreds of updates, and neither the training log nor a game replay explained
*why*. So `watch.py` grew three new diagnostic modes to answer that directly instead
of guessing:

- `--diagnostic` — troop_ratio/tile_fraction at decision time, attack vs. wait
  counts, first-attack timing.
- `--log-probs` — the policy's actual `p(wait)` vs. `p(attack)`, bucketed by
  troop_ratio.
- `--log-value` — `V(s)` traced tick-by-tick, correlated against tick number,
  troop_ratio, and tile share.

Two bugs turned up immediately while building these: `watch.py` didn't know an
eliminated agent should end the episode (the wider game keeps running after our
agent is wiped out — `train.py` already handled this, `watch.py` didn't, so it kept
"observing" a dead player's stale stats), and `watch.py` had silently drifted to a
different opponent mix (`bots=3, nations=1`) than what `train.py` actually trained
on (`bots=0, nations=1`) — every diagnostic run before the fix had been measuring
the wrong thing.

With clean tooling, the actual picture came into focus:

| check | result |
|---|---|
| action pattern | attacks within tick 1-4 of *every* game, troop_ratio ≈ 0.21 (starting value), regardless of seed |
| `p(wait)` vs. troop_ratio | real and graded — 0.6-0.77 near ratio=0, rising to 0.93-1.00 above ratio≈0.1-0.2 |
| but | `p(attack)` never fully dies out below ratio 0.1 (23-39%) — enough to keep re-crashing troops before they can accumulate |
| `corr(V, tick)` | -0.95 to -0.98, strong and consistent across seeds |
| `corr(V, troop_ratio)` | weak and **sign-flipping** between seeds (-0.46, -0.57, +0.64) |

Not "no learning" — a real, state-conditioned preference for patience exists. But
the value function still can't reliably tell a strong position from a weak one, and
a residual habit of attacking near-zero troops keeps undoing whatever patience the
policy has learned. Root-cause read: if ~97% of episodes end in a loss regardless of
how troop_ratio moves mid-game, there's no strong troop_ratio→outcome relationship
in the training data for the value function to learn — it falls back to tracking
"how close to the end" instead, which starves the policy's advantage estimate of
useful signal exactly where it matters.

### Phase 15 — A curriculum experiment, tried and deliberately reverted

If the problem is too few winning outcomes to learn from, the direct fix is a
weaker opponent. Swapping to a single weak Tribe (`bots=1, nations=0`) confirmed
this immediately — 4 wins out of 8 episodes at update 0, versus ~3% against a
Nation.

But it was reverted on reflection, not kept: Tribes barely fight back (no navy,
nukes, or diplomacy; they attack every 40-80 ticks), so the policy's existing
greedy/spam-attack habit *already* beats them. Training against it would reinforce
spamming as "correct" rather than teach the patience actually needed against real
competition — the opposite lesson, arrived at by the easy path.

A smaller, useful discovery along the way: `train.py` had never explicitly set
match difficulty, so training (and even the Phase 12 patience-heuristic diagnostic)
had been running against `Difficulty.Easy` Nations the whole time by silent
default. That's reassuring in one sense — patience-then-strike is proven achievable
at exactly this difficulty — and it ruled out "make the opponent easier" as a
further lever, since there's no easier standard tier to drop to. Difficulty is now
set explicitly rather than relying on an implicit default.

### Phase 16 — The actual structural bug, and the fix

The deeper cause behind both the attack-near-zero leak and the value function's
blindness turned out to be architectural, not a reward or curriculum problem:
`do_nothing` was scored as just one row in a single softmax alongside however many
attack candidates existed that tick. Its probability was `exp(score) / Σ exp(all
scores)` — so the *same* learned preference for waiting produced a different actual
probability depending on how crowded the board happened to be. "Should I act at
all" and "which target" were entangled into one decision instead of being
separable, and this would only get worse as more action types (buildings) are
added.

The fix: a dedicated gated action head. The policy now makes a genuine binary
wait-vs-act decision first, from a small new `gate_head`, independent of how many
attack candidates exist; target selection (unchanged pointer-network scoring) only
runs among real candidates once the gate says "act." Verified via a standalone
smoke test — probability mass sums to 1, the "no legal target" edge case forces the
correct wait, and log-probs stay self-consistent between the sampling and PPO
recompute paths — before ever touching a real training run.

Training was reset to fresh weights on the clean baseline (`bots=0, nations=1,
difficulty=Easy`) this diagnosis converged on, to isolate the effect of the new
action representation from any confound of a backbone trained under the old
entangled scheme. (A partial-load warm-start path — carrying over every layer that
still matches an old checkpoint, only randomly initializing the new `gate_head` —
was built and verified too, and stays available for later.) Old checkpoints
archived, not discarded. **Not yet validated**: whether the gated head actually
closes the patience gap — that's the next real test, using the same diagnostic
toolkit that caught the problem in the first place.

## What's next

- Re-run the `--diagnostic`/`--log-probs`/`--log-value` toolkit against the gated
  action head once it's had enough updates to say something, plus the
  trained-vs-dumb-baseline comparison that caught the original stuck-learning bug.
- Async actor/learner overlap for throughput (bigger lever than env count).
- Buildings and economy actions, once the agent can reliably survive long enough to
  spend the gold it hoards (currently ~135K per game, unspent, in an attack-only
  action space) — and once the gated action head is validated, since a third action
  family would otherwise just add a new dimension to the same entanglement problem.

## Credits / license

Built on [OpenFrontIO](https://github.com/openfrontio/OpenFrontIO) (AGPLv3). This
project is a research/learning exercise, not a redistribution of the game — see
`CLAUDE.md` for licensing notes on vendored assets.
