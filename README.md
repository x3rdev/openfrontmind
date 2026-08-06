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
perfect heuristic couldn't reliably win it. The entity-encoder rewrite is built,
verified end-to-end through the real training and playback CLIs, and not yet
re-evaluated at scale — that comparison (trained policy vs. the dumb baseline, again)
is the next milestone.

## What's next

- Re-run the dumb-policy comparison against the entity-encoder architecture — the
  test that caught the original bug is also the test that validates the fix.
- Observation/reward normalization via running statistics.
- Async actor/learner overlap for throughput.
- Buildings and economy actions, once the agent can reliably survive long enough to
  spend the gold it hoards (currently ~135K per game, unspent, in an attack-only
  action space).

## Credits / license

Built on [OpenFrontIO](https://github.com/openfrontio/OpenFrontIO) (AGPLv3). This
project is a research/learning exercise, not a redistribution of the game — see
`CLAUDE.md` for licensing notes on vendored assets.
