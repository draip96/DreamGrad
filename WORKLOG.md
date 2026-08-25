# DreamGrad worklog

This file is the source-of-record implementation and experiment log for
DreamGrad. Times are in UTC. Commands that produce scientific metrics are run
through Slurm; local commands are limited to source inspection, editing, and
small deterministic correctness checks.

## Frozen scope

- Upstream: `danijar/dreamerv3`.
- Upstream commit: `e3f02248693a79dc8b0ebd62c93683888ddaccfe`.
- Model profile: upstream `size12m`.
- Scientific delta: replay-aligned saved gradients for the complete RSSM carry
  `(deter, stoch)`, following the state/adjoint recurrence in
  `/project/6101829/draip/GradientCache.md`.
- Explicit exclusions: no cache ages, parameter versions, freshness rejection,
  EMA memory model, burn-in, damping, gradient clipping specific to cached
  adjoints, consistency loss, prioritized sweeping, curriculum, or replay
  resampling/filtering.
- Existing DreamerV3 replay sampling, world-model and actor-critic objectives,
  optimizer, and `size12m` widths remain unchanged unless a later entry records
  a user-approved exception.

## Success gates

1. A frozen-parameter numerical oracle must recover full-BPTT parameter
   gradients after boundary adjoints converge.
2. A short within-window positive control must learn before a failure is
   attributed to cross-window gradient transport.
3. With no curriculum, the saved-gradient implementation must learn a literal
   cue/query dependency whose distance is greater than 256.
4. Only after gate 3 passes, run BSuite MemoryChain with literal constructor
   `memory_length` values 11, 17, 25, 31, and 71.

## 2026-08-25

### 20:08 UTC - repository creation and provenance pin

- Verified that `/project/6101829/draip/DreamGrad` did not exist and that
  `draip96/DreamGrad` was not already present on GitHub.
- Verified live upstream `main` and `HEAD` at
  `e3f02248693a79dc8b0ebd62c93683888ddaccfe`.
- Created the GitHub fork `https://github.com/draip96/DreamGrad`, cloned it to
  `/project/6101829/draip/DreamGrad`, and added
  `https://github.com/danijar/dreamerv3.git` as remote `upstream`.
- Confirmed the clone starts clean at the exact upstream commit.

### 20:16 UTC - source and theory audit

- Read `GradientCache.md` completely. The implemented portion will be the
  bidirectional state-adjoint recurrence only: a sampled chunk starts from a
  detached cached state, adds the stopped future-adjoint inner product at its
  final state, and writes the resulting incoming adjoints back to replay.
- DreamerV3 already stores replay context for the RSSM state at every physical
  replay row. With `replay_context=1`, a sampled physical sequence `q0..qT`
  initializes from cached `S_q0` and optimizes observations `q1..qT`.
- Frozen alignment for the implementation:
  - read cached initial state `S_q0` and cached future adjoint `G_qT`;
  - write refreshed states `S_q1..S_qT`;
  - obtain dense incoming adjoints from zero-valued state taps and write
    `G_q0..G_q(T-1)`;
  - force terminal boundary adjoints to zero so credit cannot cross episodes.
- The full Markov carry is cached. DreamerV3's next recurrent transition
  consumes both `deter` and `stoch`; caching only the deterministic component
  would not implement the stated `S_k` contract.
- Prior R2R results are treated only as implementation/protocol provenance.
  Their chance-level ToyMemory results are not evidence about DreamGrad. They
  motivate the explicit short positive-control gate above.

### 20:30 UTC - first saved-gradient implementation slice

- Added `docs/GRADIENT_CACHE.md` to freeze formulas, complete-state coverage,
  replay indices, reset semantics, per-example scaling, and explicit freshness
  exclusions.
- Added an optional state-tap path to the RSSM. Taps cover both `deter` and
  `stoch` and are injected before the existing reset mask. The cache-disabled
  recurrence continues through the original source path.
- Extended the repository-owned optimizer wrapper with a mixed Ninjax
  module/input gradient transformation. Parameter gradients retain upstream
  cross-device averaging and optimizer transformations; tap cotangents remain
  local and unclipped. Both come from one stochastic learner evaluation.
- Added replay fields `grad/deter`, `grad/stoch`, and `grad/valid`. Collection
  initializes them as invalid zeros. Training reads `G_qT`, masks it at a
  terminal final row, and writes dense tap adjoints against `q0..q(T-1)` while
  retaining upstream state writes against `q1..qT`.
- Cache messages are normalized per example: multiply raw tap gradients by the
  local batch size on write and consume the boundary inner product with a batch
  mean.
- Used the zero-primal boundary expression
  `stop(G) * (S - stop(S))`. It has the required derivative but leaves the
  reported upstream loss value unchanged.
- Defined the cached local objective as the complete existing Dreamer learner
  scalar subject to its native gradient stops. This is the natural unified-
  optimizer generalization of `GradientCache.md`'s world-model scalar and
  includes the intentionally attached replay-value representation gradient.
- Extended `Replay.update()` to accept the paired state and adjoint physical
  writes without shifting either payload. No selector or sample path changed.
- Enabled saved gradients by default under `agent.gradient_cache.enabled`; the
  control remains available by setting it false.
- Added focused numerical-oracle, reset, batch-scaling, mixed-gradient,
  step-ID-alignment, duplicate-write, and uniform-sampler-parity tests.
- Added a literal-distance one-bit ToyMemory environment and repaired the
  upstream BSuite adapter to construct official `MemoryChain` instances at
  arbitrary literal `memory_length` values. The requested 11, 17, 25, 31, and
  71 will mean the official constructor parameter, not registered sweep IDs.
- Pinned Ninjax 3.6.3 because the local mixed-gradient helper is audited against
  that API; added BSuite and pytest as explicit dependencies.
- Submitted Slurm job `5019596` to build `.venv`, record the resolved package
  set, and enumerate JAX devices. No scientific metrics are produced by this
  setup job.

### 20:32 UTC - setup failure and packaging correction

- Setup job `5019596` failed before tests because upstream requirement
  `nvidia-cuda-nvcc-cu12<=12.2` now resolves to a placeholder package that
  refuses installation from PyPI. This is an environment failure, not a
  scientific result.
- Removed that obsolete duplicate requirement. `jax[cuda12]==0.4.33` resolves
  its CUDA Python wheels and experiment jobs also load the cluster CUDA module;
  the placeholder package is neither importable nor needed.
- Resubmitted the environment build as Slurm job `5019609`.

### 20:40 UTC - cluster JAX wheel resolution

- Setup job `5019609` failed before tests because the Compute Canada
  wheelhouse has no Python 3.11 build of the upstream-pinned
  `jaxlib==0.4.33`; its available cluster builds jump from 0.4.28 to 0.4.34.
  This is another environment failure, not a learner result.
- Kept upstream's exact JAX 0.4.33 pin and changed the setup job to install
  the official `jax[cuda12]==0.4.33` self-contained CUDA wheels from PyPI
  before resolving the remaining dependencies from the cluster wheelhouse.
- Removed the CUDA module load from GPU jobs because pip-installed JAX CUDA
  wheels provide their own user-space CUDA libraries; the host driver remains
  supplied by the allocated node.
- Added generated virtual environments and experiment output directories to
  `.gitignore`. All logs and artifacts remain on disk; only code, launchers,
  documentation, and reports are intended for version control.

### 20:46 UTC - exact local-CUDA installation fallback

- Setup job `5019634` showed that the cluster pip wrapper ignores an explicit
  alternate package index for platform wheels. It again failed before tests
  while resolving `jaxlib==0.4.33`.
- Retained JAX and jaxlib 0.4.33. The setup launcher now installs the exact
  official CPython 3.11 x86-64 wheels for jaxlib, the CUDA plugin, and CUDA
  PJRT by immutable URLs with SHA-256 fragments, and selects JAX's
  `cuda12-local` extra against the cluster CUDA 12.6 module. This changes only
  library delivery, not the pinned numerical implementation.
- The setup launcher clears only the generated `.venv` before rebuilding it,
  preventing failed resolver transactions from contaminating later attempts.

### 20:51 UTC - Compute Canada platform-tag adaptation

- Setup job `5019647` verified the wheel downloads but rejected the official
  `manylinux2014_x86_64` filename because Compute Canada's patched Python only
  advertises `linux_x86_64` compatibility tags. No test was started.
- The setup launcher now downloads the same three official wheels, verifies
  their published SHA-256 hashes, and mechanically retags them to the generic
  Linux tag accepted by this cluster before installation. Their binary
  payloads remain byte-for-byte the upstream 0.4.33 builds.

### 20:55 UTC - atomic paired replay visibility

- An adversarial concurrency review found that the initial tuple update held
  separate lock scopes for state and adjoint payloads, while sampling held no
  corresponding read lock. A prefetch sampler could therefore observe a new
  state with the previous adjoint.
- Replay sampling now holds the read side of the existing replay lock and a
  complete multi-payload update holds its write side. This does not change the
  sampler, probabilities, item set, or cache-validity policy; it only makes a
  learner-produced state/adjoint commit atomic.
- Added a deterministic thread-interleaving regression test that pauses after
  the state write and proves a sampler cannot enter until the adjoint write is
  committed.

### 20:58 UTC - persistence isolation from baseline replay

- A second review found that dirtying updated chunks unconditionally would
  alter cache-disabled Dreamer restart behavior and could race an asynchronous
  save. Added an opt-in `persist_updates` replay setting that is enabled only
  with saved gradients.
- Persisted mutable updates require synchronous replay saves. DreamGrad forces
  that combination when the gradient cache is enabled, so a save cannot race
  a state/adjoint mutation. Cache-disabled replay retains upstream's original
  save-once behavior.
- Added tests for both sides: saved-gradient updates survive restart, while a
  default replay update retains the exact upstream on-disk snapshot semantics;
  an unsafe asynchronous mutable configuration is rejected.

### 21:02 UTC - installed stack, missing setup GPU allocation

- Setup job `5019665` successfully verified and installed the exact JAX 0.4.33
  stack and all repository dependencies, then failed only at its final device
  probe because the launcher named a GPU partition without requesting a GPU.
- Added an explicit L40S GRES request so the setup job validates the same CUDA
  path that scientific jobs will use. No learner test or scientific run was
  counted from job `5019665`.

### 21:08 UTC - literal world-model adjoints and paired duplicate resolution

- Corrected the first implementation's broader unified-loss interpretation
  after adversarial review against the literal wording of `GradientCache.md`.
  Cached `G` now derives only from the unchanged Dreamer world-model terms:
  dynamics, representation, reward, continuation, and reconstruction.
- Native imagination and attached replay-value terms still contribute their
  unchanged parameter gradients. The parameter objective also receives the
  future world-model boundary surrogate. A shared stochastic primal is
  linearized once; separate cotangent seeds recover full parameter gradients
  and literal world-model state-tap gradients without rerunning stochastic
  sampling.
- Fixed a duplicate-ID counterexample in which one batch item's trailing state
  and another item's interior adjoint could be combined. Endpoint-only writes
  now commit first and joint interior `(S, G)` records commit last, so any row
  with a paired occurrence receives both fields from the same batch item.
- Added finite-message health accounting. Non-finite future messages are
  treated as unusable zeros and non-finite outgoing messages are stored as
  invalid zeros; this is numerical containment only and never changes replay
  selection.

### 21:12 UTC - 12M profile made explicit

- Both memory experiment profiles now merge the upstream `size12m` overlay.
  This prevents an easy-to-miss launch of the 200M default when only
  `--configs toy_memory` or `--configs bsuite` is supplied.
- Added resolved-config assertions for RSSM dimensions, stochastic classes,
  encoder/decoder depth and width, policy/value width, batch geometry, replay
  context, optimizer learning rate, and cache enablement.

### 21:15 UTC - bounded checkpoint cadence and final artifacts

- Source inspection showed that `LocalClock` treats a negative interval as
  "always", so upstream BSuite's `save_every: -1` would checkpoint after every
  ten environment rows rather than disable saving. Replaced it with a
  900-second cadence.
- The single-process `train` and `train_eval` loops now flush logger output and
  save one final checkpoint on normal step-budget completion. This is an
  artifact/reproducibility change only; it does not affect learner updates.

### 21:20 UTC - actual RSSM frozen-gradient oracle

- Added an FP32 oracle over the repository's real categorical RSSM, not a
  synthetic stand-in. It verifies that the first tap cotangents equal direct
  incoming-state derivatives for both `deter` and straight-through `stoch`.
- The same test freezes one shared parameter tree across three recurrent
  segments and verifies that the sum of boundary-seeded segment parameter
  gradients matches attached full-BPTT (relative L2 below `1e-5`, cosine above
  `0.99999`) under identical per-segment RNG seeds.
- Expanded the Slurm core suite to include upstream replay regression tests.

### 21:24 UTC - reproducible environment gate passed

- Slurm setup job `5019740` completed successfully on one L40S. It verified
  JAX 0.4.33, Ninjax 3.6.3 package installation, and a visible CUDA device.
- The complete resolved environment is recorded in
  `experiments/logs/pip-freeze-5019740.txt`; the success marker and setup log
  remain under the ignored experiment-log directory.

### 21:27 UTC - executable BSuite geometry contract

- Inspected the installed official BSuite 0.3.5 MemoryChain source and expanded
  adapter tests to execute full episodes at each requested constructor value.
  They verify the duplicated reset cue, time-to-decision row, terminal timing,
  correct-action reward, physical row count, and 10,000-episode official
  budget.
- Documentation now distinguishes BSuite constructor length, first-cue and
  last-cue distances, official actions, and Dreamer replay rows. This prevents
  the custom value 71 from being mislabeled as an unqualified 71-step
  dependency.

### 21:30 UTC - first core job rejected by runtime environment

- Slurm core job `5019790` did not produce a valid JAX test result. The CUDA
  plugin saw the GPU, but the launcher had not loaded cuDNN; the first JAX
  operation failed library initialization and all JAX-dependent tests were
  non-results.
- Added the cluster cuDNN 9.5.1 module to setup and test launchers, and added an
  actual convolution probe to environment validation rather than accepting
  device enumeration alone.
- Also removed the repository's stale broad `test_replay.py` from the focused
  gate: it targets a commented-out `Replay.dataset()` API and already fails at
  the pinned upstream commit for unrelated capacity-one selector assertions.
  DreamGrad's replay changes remain covered by dedicated alignment, sampling-
  parity, atomicity, persistence, and cache-disabled snapshot tests.

### 21:36 UTC - frozen scientific launch and analysis protocol

- Added fresh-logdir Slurm launchers for ToyMemory and official BSuite
  MemoryChain. They require a clean committed tree, select `size12m`
  explicitly, forbid checkpoint continuation, capture Git/config/package/
  module/GPU/environment provenance, and run on one H100 without a curriculum.
- Toy arms require explicit distance, seed, cache flag, and step budget so the
  distance-8 cache-off/cache-on controls can be exactly matched and distance
  257 always starts independently.
- BSuite accepts only literal constructor values 11, 17, 25, 31, and 71 and
  computes exact 10,000-episode step caps of 130,000, 190,000, 270,000,
  330,000, and 730,000 physical rows respectively.
- Added a JSONL analyzer that checks episode geometry, reports cumulative and
  tail accuracy/return plus Wilson bounds, keeps cache metrics labeled as
  mechanism health, and exits 3 when a scientific gate is missed. Toy success
  requires tail-1,000 accuracy at least 95%, mean return at least 0.90, and a
  95% Wilson lower bound above 0.90. BSuite reports both the official 62.5%
  cumulative criterion and a 90% tail criterion.

### 21:40 UTC - cache-disabled replay concurrency parity

- Final core review found that unconditional read/write locking would alter
  cache-disabled replay concurrency despite leaving sampled probabilities
  unchanged. Added an `atomic_updates` replay option enabled only with saved
  gradients; cache-disabled sampling and updates now retain upstream lock
  behavior exactly.
- Mutable cache persistence now requires both synchronous saving and atomic
  updates. Added an invariant test for this configuration.
- Scientific provenance records only an explicit allowlist of scheduler and
  device variables; it never dumps arbitrary environment variables or
  credentials.

### 21:43 UTC - dynamic CUDA library path correction

- Core job `5019806` still failed before code execution. Compute Canada's CUDA
  modules populate compile-time `LIBRARY_PATH` but intentionally leave
  `LD_LIBRARY_PATH` unchanged; JAX's local-CUDA plugin therefore could not
  dynamically load CUDA/cuDNN even though both modules were listed.
- Launchers now explicitly prepend the loaded CUDA, cuDNN, and NCCL library
  directories to `LD_LIBRARY_PATH`. Job `5019806` is an infrastructure
  non-result; its 19 non-JAX passes are not counted as the core gate.

### 2026-08-25 21:02 UTC - CUDA initialized; libdevice path rejected the run

- Slurm job `5019837` successfully initialized the CUDA and cuDNN backends, so
  the previous dynamic-library failure is resolved. It passed 25 focused tests
  before four tests that generate normal random values encountered
  `libdevice not found at ./libdevice.10.bc`.
- All four failures share that deployment error, including the synthetic reset
  oracle, the actual-RSSM oracle, and both mixed-gradient tests. They are
  infrastructure non-results, not failed numerical assertions.
- The cluster CUDA module contains the required file at
  `${CUDA_HOME}/nvvm/libdevice/libdevice.10.bc`. Every Slurm launcher now sets
  `--xla_gpu_cuda_data_dir=${CUDA_HOME}`, and the JAX setup preserves caller-
  supplied `XLA_FLAGS` while appending DreamerV3's existing optimization flags.
  This changes only deployment configuration, not model or learner semantics.

### 2026-08-25 21:07 UTC - focused GPU correctness gate passed

- Slurm job `5019912` completed on one L40S: all 29 focused tests passed in
  48.36 seconds. This is the first valid GPU result after resolving CUDA,
  cuDNN, and libdevice discovery.
- The passing set includes the pure segmented-gradient/reset/normalization
  oracles, the actual categorical-RSSM full-BPTT oracle, mixed Ninjax state and
  input gradients, replay alignment/atomicity/persistence/sampler parity, exact
  ToyMemory and BSuite geometry, and resolved 12M configuration invariants.
- Added a separate full-agent Slurm integration gate. It runs matched cache-off
  and cache-on 12M ToyMemory arms through real policy collection and optimizer
  updates, makes final checkpoints, reloads the saved replay, checks finite and
  nonzero persisted adjoints, requires later batches to consume saved future
  adjoints, and verifies that the two resolved configurations differ only by
  the cache-enabled flag and log directory.

### 2026-08-25 21:10 UTC - integration launcher boolean parse failure

- Slurm job `5019975` exited in five seconds, before environment construction
  or JAX initialization. Elements deliberately parses Boolean CLI values as
  `True` or `False`; the new shell launchers supplied lowercase values.
- Normalized all Boolean CLI arguments in the integration, ToyMemory, and
  BSuite launchers to canonical capitalization. Job `5019975` is an
  infrastructure non-result and contains no learner evidence.

### 2026-08-25 21:12 UTC - wall-clock logging caught by integration checker

- Slurm job `5019994` completed the 12M cache-disabled arm with 53 optimizer
  updates and a final checkpoint, but its post-run validator correctly failed
  because `run.log_every=10` means ten wall-clock seconds. The actual training
  portion completed before the first timed flush, so no optimizer metrics had
  been copied from the in-memory aggregate into `metrics.jsonl`.
- Set integration logging to negative-one (the existing `LocalClock` contract
  for every loop) and report/save clocks to zero; the explicit final save still
  runs. The checker now verifies the independent persisted checkpoint update
  counter as well as logged metrics. Job `5019994` is valid cache-disabled
  execution evidence but not a completed paired integration gate.

### 2026-08-25 21:13 UTC - complete optimizer routing test passed

- Closed the remaining low-risk unit-test gap by exercising the full
  `Optimizer.__call__` path with distinct parameter and state-adjoint
  objectives. The numerical test verifies that SGD updates the parameter from
  only the native objective while the returned tap derivative comes only from
  the world-model-style objective, with both using the same stochastic primal.
- Slurm job `5020002` passed the expanded set of 30 focused GPU tests in 42.75
  seconds on one L40S.

### 2026-08-25 21:16 UTC - full 12M integration gate passed

- Slurm job `5020023` completed in 1 minute 47 seconds on one L40S. Its matched
  cache-disabled and cache-enabled arms each executed 53 real optimizer updates
  from the same 100-row ToyMemory trajectory and each made a final checkpoint.
- The cache-disabled replay reloaded 84 items without any `grad/*` fields. The
  cache-enabled replay reloaded the same 84 items with 85 valid saved-gradient
  rows; at least one valid adjoint was nonzero, every adjoint was finite, and
  logged future-adjoint consumption reached 0.8.
- The validator compared both fully resolved YAML configurations after masking
  only the log directory and cache flag; they were identical. This is a
  mechanism/integration result, not a task-learning result.
- Tightened the reusable checker after review: it now derives replay geometry
  from the resolved config, checks every chunk rather than a key intersection,
  samples once after reload, asserts the 12M dimensions and CUDA platform,
  requires replay callbacks plus finite optimizer loss/gradient norm, and
  validates the completed final checkpoint at the exact step budget.

### 2026-08-25 21:19 UTC - numerical transport extended beyond 256

- Extended the actual categorical-RSSM frozen-parameter oracle to five
  65-transition segments, or 325 recurrent transitions end to end. It still
  compares both `deter` and straight-through `stoch` boundary messages and the
  sum of shared-parameter segment contributions against attached full BPTT.
- Slurm job `5020043` passed all 30 tests, including this longer oracle, in
  66.38 seconds on one L40S. This establishes numerical transport in the frozen
  FP32 oracle; it does not establish online task acquisition.
- Added an explicit zero-message identity assertion to the same actual-RSSM
  test for primal outputs and native parameter gradients under identical RNG.

### 2026-08-25 21:20 UTC - release documentation and provenance hardening

- Added a DreamGrad README preface that pins upstream provenance, names the
  narrow algorithmic delta, and states that no freshness mechanisms exist.
  Upstream module and packaging names remain unchanged for compatibility.
- Scientific launchers now record the exact shell-escaped training and analysis
  commands, live package freeze, effective derived environment seed, explicit
  source hashes, loaded modules, GPU, scheduler identifiers, resolved config,
  and clean Git revision without dumping arbitrary environment variables.
- Removed the provisional ToyMemory replay-capacity reduction. Both memory
  profiles now retain upstream's five-million-item capacity; `train_ratio=1024`
  is an explicit memory-task protocol setting shared by every saved-gradient
  and cache-disabled control arm, not a cache-dependent setting.

### 2026-08-25 21:21 UTC - exact zero-message RSSM identity passed

- Slurm job `5020064` passed all 30 focused tests in 68.41 seconds. The extended
  actual-RSSM oracle now additionally proves that enabling zero-valued taps and
  a zero future message leaves the recurrent primal and native parameter
  gradient exactly unchanged under the same RNG seed.
- Final-run artifact handling now moves any still-pending train/cache and
  episode aggregates into the logger before its final write, then saves the
  exact final checkpoint. This prevents short runs or the last wall-clock
  interval from silently losing mechanism-health metrics.

### 2026-08-25 21:24 UTC - stale upstream run-loop test excluded

- Slurm job `5020094` stopped during collection because upstream
  `embodied/tests/test_train.py` imports an undeclared `zerofun` development
  dependency. Source inspection also shows that test still calls the pre-stream
  `train()` signature, so installing the missing package would not make it a
  valid regression for this pinned commit.
- Removed that stale upstream test from the focused gate. Job `5020094` is a
  collection/infrastructure non-result; it ran no learner assertion. Final
  aggregate persistence remains exercised by the full-agent integration and
  the final-checkpoint validator.

### 2026-08-25 21:27 UTC - short-acquisition protocol frozen

- The first scientific gate will use fresh distance-8 cache-disabled and
  cache-enabled runs at seed 9407, each for exactly 20,000 physical rows: 2,000
  complete ten-row episodes. This leaves the final 1,000 episodes as an
  acquisition tail after the initial replay fill while keeping both arms
  configuration-matched and within one 64-step learner window.
- The artifact analyzer now requires the configured step budget to end exactly
  at the final logged episode, verifies the implied episode count, and rejects
  partial final episodes. Two deterministic analyzer contract tests pass
  locally; the complete suite remains subject to the final Slurm gate.

### 2026-08-25 21:29 UTC - exact-source pre-science gates passed

- Slurm job `5020141` passed the final 32-test GPU suite in 71.12 seconds. This
  includes the two analyzer artifact-contract tests in addition to every
  numerical, RSSM, optimizer, replay, and environment gate.
- Slurm job `5020144` repeated the complete matched 12M integration on the
  tightened source and passed in 1 minute 44 seconds, again with 53 optimizer
  updates per arm, persisted/reloaded nonzero finite adjoints, future-message
  consumption, callback evidence, exact final checkpoints, and matched configs.
- Added auxiliary terminal reward-sign accuracy and terminal reward MAE to the
  world-model metrics so a failed long run can distinguish reward-model
  acquisition from saved-adjoint transport. These metrics are stopped
  diagnostics and do not enter either objective or gradient.

### 2026-08-25 21:32 UTC - diagnostic-instrumented release gates passed

- Slurm job `5020182` passed all 32 focused GPU tests in 75.89 seconds after
  adding the stopped reward-model diagnostics.
- Slurm job `5020183` then passed the full matched 12M integration in 1 minute
  46 seconds. Both arms again executed 53 updates; cache-on reloaded 85 valid
  finite rows and reached 0.8 future-message use. The new terminal diagnostics
  were present and finite in both arms, while the configuration comparison
  still found no difference beyond the cache flag and log directory.
- These are the release gates for the implementation commit. The next action is
  the fresh, no-curriculum distance-8 learning control pair; no BSuite run is
  authorized by the protocol until a distance greater than 256 learns.

### 2026-08-25 21:34 UTC - implementation pushed and distance-8 runs started

- Committed the reviewed implementation as
  `ba01f56b81ab317dbe2ff9f9a4b751524047cca0` (`Implement replay-aligned saved
  gradients`) and pushed it to `draip96/DreamGrad` on GitHub. The scientific
  launchers verified an empty Git status and recorded that revision.
- Initial H100 submissions `5020215` and `5020216` remained pending with zero
  runtime because of the small long-duration partition queue. Cancelled them
  before allocation; they created no log directory and contain no result.
- Submitted the identical frozen distance-8 arms to the already GPU-validated
  L40S partition instead. Jobs `5020233` (cache disabled) and `5020234` (cache
  enabled) started simultaneously on separate L40S devices. Each is a fresh
  20,000-row, 2,000-episode run at seed 9407 with no checkpoint and no
  curriculum.

### 2026-08-25 21:49 UTC - distance-8 acquisition gate failed in both arms

- Both matched runs reached the exact 20,000-row budget and produced exactly
  2,000 complete 10-row episodes. Their Slurm state is `FAILED` only because
  the analyzer intentionally exits 3 when a scientific learning gate is
  missed; neither run crashed and both wrote a final checkpoint and complete
  analysis artifact.
- Cache-disabled job `5020233` achieved 0.5065 cumulative accuracy and 0.505
  tail-1,000 accuracy (tail mean return 0.010, Wilson lower bound 0.4741).
  Its final stopped world-model diagnostic was 0.5016 terminal reward-sign
  accuracy with terminal reward MAE 0.9999.
- Cache-enabled job `5020234` achieved 0.5130 cumulative accuracy and 0.517
  tail-1,000 accuracy (tail mean return 0.034, Wilson lower bound 0.4860).
  Its final stopped world-model diagnostic was 0.5101 terminal reward-sign
  accuracy with terminal reward MAE 0.9995.
- The saved-gradient mechanism itself remained numerically healthy in job
  `5020234`: adjoint finite fraction was 1.0, final future-hit and future-use
  rates were both 0.8935, future-adjoint RMS was 2.794e-4, and outgoing-adjoint
  RMS was 3.041e-4. These are health observations, not evidence of learning.
- Because the cache-disabled within-window control and the cache-enabled arm
  both failed while the reward model remained at chance, this pair is an
  acquisition failure upstream of any claim about long-range saved-gradient
  transport. The failed seed and all artifacts are preserved. A distance-257
  run and BSuite remain gated off while a cheaper no-curriculum acquisition
  diagnostic is designed.

### 2026-08-25 22:00 UTC - first acquisition falsifier frozen

- Audited all 2,000 cache-disabled replay episodes. Cue classes were 999/1001,
  terminal reward classes were 987/1013, query actions were 1,100/900, query
  and reward rows were exactly `q8` and `q9`, and every episode satisfied
  `reward(q9) = +1` exactly when `action(q8) == cue(q0)`. This rules out the
  environment/replay action-shift hypothesis before changing training.
- The final mean reward loss of 0.1380 is also informative. For the native
  255-bin symexp-TwoHot head, knowing terminal timing while assigning equal
  probability to reward signs has an analytic episode-mean loss near 0.1359;
  a sign-informed head can approach 0.0665. Together with MAE near one, this
  identifies loss convergence to a sign-blind solution rather than missing
  terminal examples. The cache-disabled run executed 18,913 optimizer updates.
- The native KL losses were pinned at the configured `free_nats=1.0` floor.
  A one-bit cue carries at most `ln(2) < 1` nat, so the cheapest retention
  falsifier is the existing upstream RSSM setting `free_nats=0.0`; it changes
  no model equation, replay rule, loss term, saved-gradient rule, or optimizer.
- Added a validated `RSSM_FREE_NATS` launcher input, defaulting to the untouched
  upstream value 1.0, and record it in the resolved command and environment
  provenance. Distinct run roots prevent any diagnostic from reusing a prior
  log directory.
- Froze the next independent, fresh, cache-disabled jobs at seed 9407:
  (1) native `free_nats=1.0`, distance 1, 6,000 rows/2,000 episodes, which tests
  basic cue/action binding with only one recurrent transition; and
  (2) `free_nats=0.0`, distance 8, 20,000 rows/2,000 episodes, which directly
  tests cue retention under the KL floor. Neither run loads a checkpoint, and
  neither can be used as curriculum for a later distance.
