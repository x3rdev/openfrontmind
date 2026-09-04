# OpenFrontMind

Training a reinforcement-learning agent to play [OpenFront](https://openfront.io) via
self-play, in the spirit of DeepMind's AlphaStar, scaled down to a single local GPU.

## Architecture

```
engine/   vendored, unmodified OpenFrontIO simulation core (deterministic TypeScript)
env/      headless Node harness — drives the engine, speaks a bridge protocol over stdio
training/ Python/PyTorch side — vectorized envs, PPO training loop, policy network
```

Built harness around open source game engine without editing original game. This
helps with adapting to future updates. Since game runs in typescript and can run
for very long, parallelized training process early.

`VecOpenFrontBridge` runs N game instances as separate OS processes (sim is
CPU-bound single-threaded JS, threads fight the GIL). PPO with GAE, clipped value
loss, entropy bonus. Policy uses an entity-attention encoder + pointer-network
action head instead of fixed-size output, since the number of opponents/candidates
isn't fixed.

## Progress log

Reward scale bug: `WIN`/`LOSS` of `±10,000` swamped the shared backbone gradient,
entropy stuck at 0, zero wins in ~290 episodes. Rescaled to `±100`, fixed.

Horizon bug: rollout was 500 ticks, but a real game's peak-to-collapse arc was
~1,850 ticks — value function never saw both halves in the same batch. Bumped
`STEPS` and `GAMMA` to match, reward went positive.

Dumb-policy check became the standard validation method: run a zero-intelligence
heuristic on the same seeds as a checkpoint. Caught training being statistically
indistinguishable from no learning at all (300 updates, same peak tile share, same
tick, same loss as the heuristic).

Root cause: attack targets were picked by position in a list whose order
reshuffled every tick, so a fixed-slot policy had nothing stable to learn a
preference for. Rewrote as an entity encoder + pointer-network head, scoring
candidates by their own features instead of index.

Param audit: flattened conv output was ~99% of the network's params and tied to
one map size. Replaced with adaptive pooling — 5.3M params down to 414K,
map-size-independent.

Throughput: plateaus around 32-40 parallel envs on a 16-core machine. Bigger win
was overlapping rollout collection with the PPO phase on a background thread
(~25% faster), since they'd been running strictly sequentially.

Value function was tracking tick number, not board state — declined smoothly all
game even while winning. Cause: observation only had raw troop count, which grows
with time regardless of strategy. Added `troops/maxTroops` ratio instead (what the
game's own AI gates attacks on).

Proved patience beats greed outside the network entirely: a heuristic that waits
for troop ratio > 50% before attacking beat an always-attack heuristic 6/6 vs 0/6,
same seeds, no learning involved. Set a concrete ceiling (~80% peak tile share,
100% win rate) to hold the trained policy against.

Rollout throughput was flat 24→32 envs. Cause: observation encoding recomputed
full spatial planes from scratch every tick instead of updating changed tiles.
Fixed incrementally, verified bit-exact against the old method.

First full-scale run failed on launch: rollout buffer was `float32` for binary
planes, tried to allocate 48GB on a 31GB machine. Switched to `bool`, fixed.

Built diagnostic tooling (`watch.py --diagnostic/--log-probs/--log-value`) to see
*why* win rate was stuck, not just that it was. Found two real bugs this way:
`watch.py` wasn't treating elimination as episode-end like training did, and it
had drifted to a different opponent mix than training actually used — every prior
diagnostic run had been measuring the wrong thing.

Real structural bug: `do_nothing` competed in the same softmax as every attack
candidate, so the same learned preference produced a different actual probability
depending on how many candidates existed that tick. Fixed with a dedicated gate
head — binary wait-vs-act decision, independent of candidate count.

Hit first real wins, then a policy collapse: entropy fell to near zero within 15
updates of a win-heavy batch overfitting on it, wins dropped back to 0%. Also
found troop hoarding (gold banked past 1M unspent) from `SHAPING_COEF` overweighting
the per-tick patience reward.

Same collapse mechanism, different trigger: rare wins' value loss ran 10-20x normal
scale and dominated the shared-backbone gradient on every win. Fixed with return
normalization (running-std scale on the value target). Result: `nations=1` win rate
went from stuck at 2-5% to **97.5%** (39/40 seeds, verified via diagnostic batch).

Curriculum bump to `nations=2`, warm-started from the 97.5% checkpoint, got stuck
again — flat ~0.2-0.3% win rate for 350+ updates. Ruled out reward-magnitude and
enemy-identity-confusion as causes (measured directly: land-gain reward term is
negligible, shaping term nets positive, per-candidate features are provably
distinguishable). Real cause: outcome-diversity stalemate — a 40-seed sweep found
exactly one win (a genuine patience trajectory), meaning PPO almost never sees a
winning trajectory to reinforce.

Added self-imitation learning (Oh et al. 2018, `training/models/self_imitation.py`):
a replay buffer of past winning-episode transitions, weighted by `max(0, return -
V(s))`, so a rare win keeps teaching the policy across future updates instead of
just the one it was collected in. Win rate climbed immediately after enabling it —
0% → 1.3% → 1.8% → 3.5% → 5.9% over four 20-update buckets, ~6-8x the pre-SIL
baseline and still rising, not plateaued. Confirmed on a held-out 40-seed sweep
(not the training log's own numbers): 10% win rate, and losses spread from still-
greedy failures to genuine near-wins (60%+ peak tile share) — real generalization,
not memorized replay.

## What's next

- Get `nations=2` to a real win-rate bar (90%+, matching what `nations=1` hit).
- Larger fixed evaluation seed set for lower-noise progress checks.
- Buildings/economy actions (deprioritized until the stuck-learning problem was
  solved — that condition is now met).

## Credits / license

Built on [OpenFrontIO](https://github.com/openfrontio/OpenFrontIO) (AGPLv3). This
project is a research/learning exercise, not a redistribution of the game — see
`CLAUDE.md` for licensing notes on vendored assets.
