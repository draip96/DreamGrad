# Saved-gradient design

DreamGrad is upstream DreamerV3 plus replay-aligned recurrent-state adjoints.
This document fixes the implementation contract derived from
`/project/6101829/draip/GradientCache.md`.

## Boundary recurrence

For learner chunk `k`, let `S_k` be the complete incoming RSSM carry and let
`G_k` be the cotangent of later learner losses with respect to that carry:

```text
S_(k+1) = F_theta(S_k, X_k)
G_k = d ell_k / d S_k + (d S_(k+1) / d S_k)^T G_(k+1)
```

The optimized scalar adds the stopped boundary surrogate

```text
mean_b <stop_gradient(G_(k+1,b)), S_(k+1,b)>.
```

The cache stores per-example messages. Because DreamerV3's learner scalar is a
batch mean, incoming tap cotangents are multiplied by the local batch size
before being written. Consumption uses a batch mean. Parameter gradients retain
DreamerV3's existing cross-device averaging; per-row cached cotangents are not
averaged across devices.

`GradientCache.md` defines `ell_k` as the ordinary world-model loss. DreamGrad
uses exactly the unchanged dynamics, representation, reward, continuation, and
observation-reconstruction terms for cached state adjoints. DreamerV3 has one
unified optimizer, so native imagination and replay-value terms still
contribute their unchanged parameter gradients, while they do not enter the
saved `G` messages. A shared stochastic primal produces two VJP seeds: the
complete native-plus-boundary scalar for parameters and the literal
world-model-plus-boundary scalar for state taps.

## Physical replay alignment

With upstream `replay_context=1`, replay provides `T+1` physical rows
`q0..qT`. The learner initializes from cached posterior state `S_q0`, removes
the prefix, and optimizes rows `q1..qT`.

DreamGrad therefore performs these distinct updates:

| Payload | Read | Write step IDs |
| --- | --- | --- |
| initial state | `S_q0` | none |
| refreshed states | learner outputs `S_q1..S_qT` | `q1..qT` |
| future adjoint | `G_qT` | none |
| refreshed incoming adjoints | tap cotangents `G_q0..G_q(T-1)` | `q0..q(T-1)` |

The two write ranges must not be conflated. Rewriting `q0` as though it were a
fresh state would race with newer replay-context updates; writing incoming
adjoints against `q1..qT` would be an off-by-one scientific error.

Interior rows `q1..q(T-1)` are committed as joint `(S, G)` records from one
batch occurrence. Leading `G_q0` and trailing `S_qT` are necessarily one-sided.
If sampled windows overlap within a batch, a paired interior occurrence wins
over either one-sided endpoint occurrence; duplicate paired occurrences use the
same last-batch-item rule for both fields.

## Complete state and resets

The cached Markov state and adjoint both contain:

- `deter`: the deterministic block-GRU state;
- `stoch`: the straight-through categorical stochastic state.

Both are required because the next RSSM core step consumes both. Zero-valued
taps are inserted before the existing reset mask. Thus a true episode reset has
zero derivative with respect to the preceding state. Future adjoints are also
masked to zero when the sampled final row is terminal.

## Stochastic replay-row identity

An overlapping learner window can reconstruct the same physical posterior row
more than once. Independent categorical draws at that row do not, in general,
produce the full-BPTT straight-through estimator: the hard stochastic outcome
also changes the later deterministic state. DreamGrad therefore contains a
named `gradient_cache.posterior_rng_keys` mode that stores one immutable
`uint32[2]` sampling key per physical replay row and reuses it with the current
posterior logits whenever that row is reconstructed. The key fixes stochastic
path identity without freezing logits, states, parameters, or gradients.

This mode is default-off while the matched acquisition and full-BPTT controls
are evaluated. Enabling it requires saved gradients, `replay_context=1`, and a
fresh replay containing the key field. It is not a freshness policy: keys have
no age or version and are never rewritten by learner cache updates.

## Freshness policy

There is deliberately no freshness policy. A learner pass writes states from
its forward computation and adjoints from the corresponding reverse
computation, both under the same pre-update parameters. A future adjoint used by
that pass necessarily contains information from an earlier learner pass. No
age, version, rejection, EMA, burn-in, damping, clipping, consistency, priority,
or resampling mechanism is added.

## Evidence gates

- A frozen-parameter oracle compares cached-boundary and full-BPTT parameter
  gradients using identical scalar reductions.
- Reset tests ensure no cotangent crosses a true episode boundary.
- Replay-alignment tests use distinct sentinels for `q0..qT` to catch shifted
  writes.
- Short acquisition must pass before long-range transport is interpreted.
- Terminal reward-sign accuracy and error are logged as functional world-model
  diagnostics; cache hit rates and adjoint norms alone are never learning
  evidence.
- The no-curriculum toy gate requires a literal dependency distance greater
  than 256 before BSuite evaluation begins.

## Environment geometry

ToyMemory distance `d` has one cue at physical replay row `q0`, its query at
`q_d`, and terminal reward at `q_(d+1)`. Thus cue-to-query distance is exactly
`d`, reward dependency is `d+1`, and each episode contains `d+2` physical rows.

Official BSuite MemoryChain constructor value `m` repeats its reset cue at
`q1`, presents the decision observation at `q_m`, and emits terminal reward at
`q_(m+1)`. Reports must distinguish constructor `memory_length=m`, `m+1`
official actions, `m+2` Dreamer physical rows, first-cue distance `m`, and
last-cue distance `m-1`. Requested values are always literal official
constructor arguments; only 17 and 25 happen to be registered sweep settings.
