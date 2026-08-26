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

### 2026-08-25 22:09 UTC - acquisition falsifier wave in progress

- Committed and pushed the first-wave launcher and log as
  `060ac2800437fb3dc43f1ebc9cbe475df8f40241`. Jobs `5020484` (native distance
  1) and `5020485` (distance 8 with only `free_nats=0`) started on separate
  L40S nodes from that clean revision and recorded the same model/environment
  seed pair as the failed distance-8 controls.
- Distance-1 job `5020484` reached its exact 6,000-row/2,000-episode budget and
  intentionally exited 3 after missing the strict policy gate. It was not a
  flat failure: final terminal reward-sign accuracy was 0.8487, terminal MAE
  was 0.4254, and tail-1,000 policy accuracy rose to 0.607. This is evidence
  that the action/reward alignment and native head can begin acquiring the
  task; the short budget supplied only 4,913 post-prefill optimizer updates.
- Launched independent job `5020522` from scratch at native distance 1 for
  15,000 rows/5,000 episodes. It uses no checkpoint from `5020484`, so it is a
  budget positive control and not curriculum.
- At the 11,890-row monitoring point, distance-8 job `5020485` remained
  sign-blind (terminal sign accuracy 0.5042 and MAE 0.9999) even though its KL
  losses had moved below the old floor to about 0.123. This is interim evidence
  only; the frozen 20,000-row run remains active and will be analyzed in full.
- Prepared the next single-variable launcher control, `REPVAL_GRAD=false`.
  This uses Dreamer's existing stop-gradient switch to prevent the replay-value
  objective from updating the shared RSSM feature while retaining the value
  loss, actor, every native world-model loss, replay, precision, optimizer, and
  12M architecture. The default remains the untouched upstream `true` value,
  and the resolved setting is added to provenance.

### 2026-08-25 22:16 UTC - distance-1 positive control passed

- Distance-8 `free_nats=0` job `5020485` completed the full frozen budget and
  intentionally exited 3. It achieved 0.509 cumulative and 0.518 tail-1,000
  accuracy; final terminal reward-sign accuracy was 0.5080 and MAE was 0.9996.
  Its KL losses were no longer clamped at one, so removing the native KL floor
  was a clean negative result and is not promoted.
- The independent longer native distance-1 job `5020522` completed normally and
  passed every predeclared gate. Across 5,000 episodes, final terminal
  reward-sign accuracy was 0.9960 and MAE was 0.0136. Tail-1,000 policy accuracy
  was 0.997, tail mean return was 0.994, and the 95% Wilson lower bound was
  0.9912. This is the first functional learning positive control and confirms
  the unchanged action alignment, reward head, optimizer, and full agent can
  solve the task when only one recurrent transition separates cue and query.
- Committed and pushed the replay-value gradient switch as
  `94a6523798136aa5dbb35dc34763e63f42794324`, then launched fresh native
  distance-8 job `5020575` with only `agent.repval_grad=False`. It is a
  one-variable acquisition diagnostic and does not reuse the successful
  distance-1 checkpoint.

### 2026-08-25 22:17 UTC - native auxiliary-loss ablation prepared

- At the 11,840-row monitoring point, job `5020575` remained sign-blind after
  stopping replay-value gradients into the RSSM: terminal reward-sign accuracy
  was 0.4994 and MAE was 0.9961. The frozen run remains active and this interim
  result is not substituted for its final artifact.
- Dreamer's native imagination actor/value inputs are already stopped with
  `ac_grads=False`; with `repval_grad=False`, every remaining RSSM parameter
  gradient comes from the world-model objective. Prepared the next exact
  acquisition ablation by adding `MODEL_AUX_ENABLED=false` to the launcher.
  It sets only the existing reconstruction, continuation, dynamics-KL, and
  representation-KL scales to zero. Reward remains the native all-row
  symexp-TwoHot loss; actor, value, and replay-value training remain enabled.
- This adds no terminal balancing, new loss, resampling, reward shaping,
  checkpoint transfer, or saved-gradient special case. The launcher default is
  the untouched native auxiliary objective and the resolved switch is captured
  in provenance. The ablation will be a diagnostic until a later matched
  cache-off/cache-on confirmation establishes whether it is suitable for the
  no-curriculum long-dependency gate.

### 2026-08-25 22:21 UTC - replay-value stop-gradient acquired distance 8

- Job `5020575` reached all 20,000 rows and exactly 2,000 episodes, then
  intentionally exited 3 because the fixed tail-1,000 gate included its
  pre-acquisition phase. Cumulative accuracy was 0.6285 and tail-1,000 accuracy
  was 0.772, so this failed artifact is retained rather than relabeled a pass.
- The functional diagnostics show a clear late acquisition transition rather
  than chance: final terminal reward-sign accuracy was 0.9745, terminal MAE was
  0.0775, and the final stopped imagined-return diagnostic was 0.6859. The last
  200 online episodes contained 192 successes (0.96 accuracy, mean return
  0.92). This differs decisively from every native and `free_nats=0` distance-8
  run, whose reward-sign accuracy remained near 0.50 through the same budget.
- The sole scientific change was the existing upstream configuration switch
  `agent.repval_grad=False`; replay-value prediction still trained, but its
  gradient could not overwrite the RSSM representation. The prepared
  world-model auxiliary ablation is therefore not launched.
- Froze a fresh matched confirmation at 30,000 rows/3,000 episodes, seed 9407,
  native `free_nats=1.0`, and `repval_grad=False`, with cache-disabled and
  cache-enabled arms differing only by saved gradients. No checkpoint is
  transferred. The budget is not selected from a successful tail statistic:
  it places the predeclared final 1,000 episodes at rows 20,010--30,000, after
  the independently observed roughly 15,000-row acquisition transition.

### 2026-08-25 22:45 UTC - matched distance-8 confirmation passed

- Jobs `5020662` (cache disabled) and `5020663` (cache enabled) both completed
  normally from clean revision
  `8777c055ec4979588c5492d5670827ec0161c936`. Their resolved configurations,
  model/environment seeds, 12M dimensions, 30,000-row budgets, and all
  acquisition settings match exactly except for saved-gradient enablement.
- Cache-disabled job `5020662` passed with 0.969 tail-1,000 accuracy, 0.938 tail
  mean return, and 0.9563 Wilson lower bound. Its final terminal reward-sign
  accuracy was 0.9824 and terminal MAE was 0.0540.
- Cache-enabled job `5020663` passed with 0.965 tail-1,000 accuracy, 0.930 tail
  mean return, and 0.9517 Wilson lower bound. Its final terminal reward-sign
  accuracy was 0.9826 and terminal MAE was 0.0533. Saved adjoints were 100%
  finite; final future-hit/use rates were both 0.8971, future-adjoint RMS was
  5.460e-5, and outgoing-adjoint RMS was 9.484e-5.
- This qualifies `agent.repval_grad=False` as the common ToyMemory acquisition
  profile. It is an existing upstream Dreamer switch applied identically to
  both arms; it does not alter the saved-gradient algorithm, replay sampling,
  or any world-model loss. The unused auxiliary-loss ablation remains unrun.
- Froze the first requested greater-than-256 test at literal cue-to-query
  distance 257 and cue-to-reward dependency 258. The cache-enabled run is fresh
  seed 9407 with no checkpoint or curriculum and uses 777,000 physical rows:
  exactly 3,000 episodes of 259 rows. The final 1,000-episode gate therefore
  has episode endpoints 518,259--777,000. This budget compensates for the
  25.9-times lower
  terminal-row density relative to distance 8 while leaving a full independent
  1,000-episode tail after the acquisition region suggested by the short gate.

### 2026-08-25 22:46 UTC - no-curriculum distance-257 run started

- Committed and pushed the matched short-gate record as
  `ffcc2b8de80971dd0f45ee43af7bc5357e637dc7`, then launched cache-enabled
  distance-257 job `5021097` on the 12-hour L40S partition. It started from that
  clean revision on node `kn028`; provenance records 777,000 rows, seed 9407,
  `repval_grad=False`, native world-model auxiliaries, and an empty checkpoint
  source. This is a literal no-curriculum dependency longer than 256.
- Hardened the still-gated BSuite launcher to require an explicit
  `REPVAL_GRAD=true|false`, validate it, pass the canonical boolean to Dreamer,
  and capture it in provenance. No BSuite job is submitted before `5021097`
  passes the functional learning gate.
- If unlocked, the BSuite order remains official constructor lengths 11 first,
  then 17/25/31, then 71, each for exactly 10,000 fresh episodes and with
  `repval_grad=False` carried over as the common acquisition profile. Their
  physical-row budgets remain 130,000; 190,000; 270,000; 330,000; and 730,000.

### 2026-08-25 22:57 UTC - long-run checkpoint I/O falsifier

- Stopped job `5021097` after 10 minutes 54 seconds and before its first
  15-minute periodic checkpoint. It had logged only 13,220 rows/51 complete
  episodes, used finite saved gradients at 0.9932 future-message rate, and is
  retained as an intentionally cancelled infrastructure pilot with no learning
  interpretation or analyzer result.
- The saved-gradient replay persists mutable state and adjoint updates. Uniform
  training eventually marks nearly every replay chunk dirty, so each periodic
  checkpoint would rewrite an increasing fraction of the projected roughly
  14 GB replay. Repeating that every 15 minutes could dominate the 12-hour
  allocation and stress shared storage without adding scientific evidence.
- Added a validated, provenance-recorded `SAVE_EVERY` launcher input. Toy runs
  retain the upstream 900-second default unless explicitly overridden. Long
  no-resume runs use `-1`, which disables periodic saves in `elements.Clock`;
  the training loop still performs its unconditional final checkpoint after
  the exact step budget. BSuite defaults to the same final-only behavior because
  its official environment state is not resumable and every run is fresh.
- Values other than `-1` or a positive integer are rejected, specifically
  preventing the dangerous `0` setting, which means save on every loop rather
  than disable saving. Static shell syntax, default expansion, and diff checks
  pass. The distance-257 science configuration and budget are otherwise
  unchanged and will restart in a new log directory from a clean commit.

### 2026-08-25 23:03 UTC - checkpoint clock convention corrected

- The preceding clock interpretation was wrong for this training path and is
  superseded by this entry. Dreamer's `train.py` uses `embodied.LocalClock`, not
  `elements.when.Clock`: `LocalClock(0)` disables a periodic action and every
  negative interval triggers it on every call.
- Job `5021163` therefore saved repeatedly after its initial checkpoint. It was
  stopped after 4 minutes 50 seconds at only 1,410 logged rows/five episodes.
  The partial directory is 273 MB and retains two checkpoint generations after
  checkpoint cleanup. It has no analyzer artifact and is an infrastructure
  non-result.
- Corrected both launchers so `SAVE_EVERY=0` means final-only and all negative
  values are rejected. Toy's default remains 900 seconds for short jobs;
  long ToyMemory passes `0` explicitly, and BSuite defaults to `0`. The training
  loop's unconditional post-budget `cp.save()` remains unchanged.
- Added a direct regression test asserting that `LocalClock(0)` stays false and
  `LocalClock(-1)` stays true across repeated calls. This prevents the two clock
  APIs from being confused again before another long allocation.

### 2026-08-25 23:06 UTC - corrected distance-257 run verified

- Committed and pushed the clock correction and regression test as
  `9aaccd5f5a8d1e87fecace22ab214656cb298425`; the focused clock test passed 2/2.
- Launched fresh cache-enabled distance-257 job `5021202` on node `kn046` from
  that clean revision. Its provenance records the unchanged 777,000-row,
  3,000-episode science configuration and `SAVE_EVERY=0`; no cancelled-run
  checkpoint or replay is referenced.
- After compilation and 4,040 rows, the log contained exactly one checkpoint
  save: the mandatory initial snapshot. A later driver iteration still showed
  one save line, directly verifying that periodic saving is disabled. Saved
  adjoints were finite and future-message use was 0.9867. Learning metrics at
  only 15 complete episodes remain uninterpretable.

### 2026-08-25 23:20 UTC - zero-runtime H100 scheduling probe

- Submitted job `5022620` as a possible duplicate distance-257 allocation on
  the H100 partition, but Slurm reported no assigned node and no predicted
  start while the relevant nodes were reserved or down. Cancelled it after
  24 seconds pending, before allocation or execution (`Elapsed=00:00:00`).
- The probe created no run directory, checkpoint, replay, metric, or scientific
  result. Authoritative distance-257 evidence continues to come only from
  L40S job `5021202`; no duplicate seed was allowed to begin training.

### 2026-08-26 01:36 UTC - BSuite length 11 gated dependency queued

- Queued L40S job `5024478` for the cheapest requested BSuite check, official
  memory length 11, with Slurm dependency `afterok:5021202`. It remains pending
  with reason `Dependency` and cannot execute unless the distance-257 job runs
  its exact budget, completes its analyzer, and exits successfully.
- The queued configuration is fresh seed 9407, cache enabled, the frozen common
  acquisition setting `repval_grad=False`, `SAVE_EVERY=0`, 130,000 physical
  rows, and no checkpoint source. Lengths 17, 25, 31, and 71 remain unqueued;
  length 11 must pass before spending on them.

### 2026-08-26 15:53 UTC - first distance-257 gate failed

- Job `5021202` completed all 777,000 requested rows (exactly 3,000 episodes)
  in 8:32:05 and wrote its unconditional final checkpoint. The log contains
  exactly two saves, the initial and final snapshots, so final-only checkpoint
  behavior remained correct throughout the run.
- The formal analyzer intentionally returned exit 3. Cumulative accuracy was
  0.481 and mean return -0.038; the final 1,000 episodes had accuracy 0.475,
  mean return -0.050, and a 0.4442 Wilson lower bound. All three prespecified
  learning gates failed. The final 200 episodes were also chance-level at
  0.460 accuracy, so this is not a tail-boundary artifact.
- The mechanism remained healthy but did not yield functional policy learning:
  final adjoint-finite fraction was 1.0, future-message hit/use rates were both
  0.9954, outgoing/future adjoint RMS values were 7.234e-4 and 7.477e-4, and
  no runtime, device, or non-finite error occurred. Terminal reward-sign
  accuracy rose late to 0.8562 and terminal MAE fell to 0.3734, while behavior
  remained at chance. This is therefore a preserved learner failure, not proof
  of a cache transport defect or a passing greater-than-256 dependency.
- Dependent BSuite length-11 job `5024478` never allocated or executed and was
  cancelled automatically after the failed `afterok:5021202` dependency. It
  produced no log directory or scientific result. BSuite remains locked.
- The next cheap falsifier is a fresh cache-enabled distance-65 run: it places
  the cue and query on opposite sides of the native 64-step learner span while
  requiring only 201,000 rows for the same 3,000 independent episodes. If that
  passes, distance 129 localizes the transition across two spans before any
  larger distance-257 budget. Every localization run starts from random
  initialization with the same task throughout; none supplies a checkpoint or
  curriculum to a later run.

### 2026-08-26 15:57 UTC - fresh distance-65 localization started

- H100 request `5034451` remained pending for resources while an existing job
  array recycled the next available GPU. Cancelled it before allocation after
  `Elapsed=00:00:00`; it created no run directory or result.
- Submitted the identical configuration as L40S job `5034470`, which started
  immediately on node `kn104` from clean revision
  `8ebf1317565482ce0f63c1a8f0c3c30defa26c6b`. Provenance confirms an empty Git
  status, seed 9407, literal distance 65, cache enabled, `repval_grad=False`,
  native model auxiliaries, free nats 1.0, `SAVE_EVERY=0`, and an empty
  checkpoint source.
- Its exact 201,000-row budget is 3,000 episodes of 67 rows. This fresh task
  crosses one 64-step learner boundary and is solely a localization test; it
  neither consumes the failed distance-257 checkpoint nor supplies a
  checkpoint to a later run.
- Queued L40S job `5034475` for fresh distance 129 with dependency
  `afterok:5034470`. It requests 393,000 rows, exactly 3,000 episodes of 131
  rows, and the identical seed and acquisition profile. It cannot execute if
  distance 65 misses its analyzer gate and it has no checkpoint input, so this
  is an independent two-boundary localization point rather than curriculum.

### 2026-08-26 16:11 UTC - replay-backed 258-transition transport oracle

- An independent source audit found no physical-row or reset off-by-one in the
  saved-gradient path, but identified an evidence gap: the prior greater-than-
  256 RSSM oracle supplied exact suffix adjoints from attached BPTT rather than
  iterating messages through the mutable replay implementation.
- Extracted the existing cache payload slicing into the pure
  `_gradient_cache_payloads()` helper without changing its write ranges or
  ordering. Production still writes leading `G_q0`, trailing `S_qT`, then joint
  interior `(S, G)` rows so paired occurrences win duplicate IDs.
- Added a deterministic frozen-parameter oracle using the actual categorical
  RSSM tap path and actual `Replay.update()` across storage chunks. Five reverse
  backups over physical windows `q194..q258`, `q130..q194`, `q66..q130`,
  `q2..q66`, and `q0..q64` recover both leaves of `G_q0` from a terminal-only
  loss and match attached 258-transition BPTT at relative L2 below `1e-5` and
  cosine above `0.99999`. The deliberately offset `q2..q66` window makes the
  last backup consume dense interior message `G_q64`, so this is an iterative
  replay transport check rather than another fixed-boundary identity.
- The same test verifies exact state/adjoint physical write ranges, paired
  interior values, validity bits, and a reset at `q130` that makes `G_q129` and
  `G_q0` exactly zero. Parameters are never optimized in this correctness test.
- Local deterministic validation passed: the new test passed 1/1 in 20.80
  seconds, and the combined gradient-cache/RSSM/optimizer/replay set passed
  21/21 in 66.56 seconds. Added the oracle to the focused Slurm test manifest;
  an authoritative GPU rerun remains required before the next release claim.

### 2026-08-26 16:21 UTC - read-only failed-checkpoint functional audit

- Source/artifact comparison showed that the failed distance-257 terminal
  reward metric is a posterior-state readout, whereas actor training consumes
  stochastic prior imagination of length 15. Final posterior sign accuracy
  0.8562 therefore does not prove counterfactual prior reward fidelity or an
  exploitable policy signal. Relative to the passing distance-8 run, the long
  run also had about 7.4-times smaller imagination advantage magnitude and
  roughly 42-times lower absolute normalized-return rate; policy randomness
  remained 0.640 rather than 0.006. These are diagnostic observations, not a
  new gate or a cache-causality claim.
- Added `experiments/audit_toy_checkpoint.py`, a read-only artifact audit that
  traverses persisted replay through UUID successor links, asserts exact
  cue/query/reward geometry, and loads only the final agent checkpoint. It
  never calls `Agent.train()` or `Replay.update()`.
- The audit reports cached-state and full-current posterior results separately:
  exact policy probabilities and margins conditional on each hard state,
  terminal posterior reward accuracy, deterministic MAP next-state rewards,
  Monte Carlo stochastic-prior counterfactual rewards, terminal probability,
  action-by-cue coverage, and a fixed SHA256-fold nearest-centroid cue probe.
  MAP is explicitly not labeled expected reward, the probe is explicitly only
  linear separability, and historical state reconstruction is declared
  unavailable because parameter versions and stochastic RNG keys were not
  stored.
- Added a guarded Slurm launcher that requires a clean tree, refuses to place
  output or provenance inside the source artifact, refuses overwrites, records
  the exact command/modules/GPU/revision, and hashes the audit sources plus the
  resolved training config and exact agent/step checkpoint inputs.
- Added five deterministic audit-helper tests and included both new test sets
  in the focused Slurm manifest. Helper tests passed 5/5. My first local oracle
  invocation selected CUDA on the login node and failed before the test with a
  missing-DNN-library backend error; the explicit CPU rerun passed 1/1 in 20.95
  seconds. The cache/environment-focused local set separately passed 33/33 in
  67.53 seconds, and the final expanded local manifest passed 38/38 in 66.53
  seconds. These local checks are non-scientific; the expanded CUDA manifest
  and the checkpoint audit itself remain to run through Slurm.

### 2026-08-26 16:24 UTC - transport proof and audit dispatched

- Committed and pushed the semantics-preserving payload-helper extraction,
  iterative replay oracle, read-only checkpoint audit, launch guards, tests,
  and worklog as `caa598d02dfd779b98da2be7b1c780a5b97c1513`.
- Launched authoritative expanded CUDA test job `5034819` on L40S node `kn161`
  from that clean revision. The manifest now contains 38 focused tests,
  including the actual-Replay 258-transition reverse sweep and five audit
  helper contracts.
- Launched read-only distance-257 checkpoint audit job `5034831` on L40S node
  `kn035`, also from the clean revision. It uses batch size 16, 64 stochastic
  prior samples per action/state, four fixed-seed full-current posterior
  reconstructions, and audit seed 0. Its new output target is
  `runs/audits/distance-257-failed-seed9407/audit.json`, outside the immutable
  failed-run artifact; exact source and checkpoint hashes are captured beside
  it. This audit cannot turn the failed learning gate into a pass.

### 2026-08-26 16:25 UTC - CUDA test node ECC non-result

- CUDA test job `5034819` failed after 16 seconds on node `kn161` before any
  GPU computation: JAX reported `CUDA_ERROR_ECC_UNCORRECTABLE` and could not
  initialize a supported CUDA device. All 11 GPU-touching tests failed at the
  same backend initialization point while 27 CPU-only tests passed. This is an
  infrastructure non-result with no implementation interpretation.
- Submitted replacement `5034878` excluding `kn161`, then cancelled it at zero
  runtime when the short partition had no immediate resource. Replacement job
  `5034894` requests the identical clean-tree suite on the 12-hour L40S
  partition with a one-hour limit and still excludes `kn161`; it is pending for
  resources. No source or test configuration changed between attempts.

### 2026-08-26 16:27 UTC - checkpoint audit dtype seam corrected

- Audit job `5034831` loaded and verified the exact final checkpoint, then
  stopped after 1:49 at its first direct prior call. Persisted replay states are
  float32, while `RSSM.imagine()` expects the configured BF16 compute dtype;
  the audit hook bypassed the `RSSM.observe()` entry point that normally casts
  them. The assertion occurred before an audit JSON was produced. Its source
  artifact stayed untouched and its failed-attempt provenance is preserved.
- Added the native `nets.cast()` conversion inside both read-only prior hooks.
  This changes neither training, replay, cache values, nor model parameters.
  Audit helper tests still pass 5/5, Python compilation and diff checks pass,
  and the corrected runtime will use a new output path rather than overwrite
  the failed attempt.

### 2026-08-26 16:29 UTC - authoritative CUDA suite passed

- Replacement CUDA job `5034894` ran on healthy L40S node `kn046` and passed
  the complete expanded focused suite: 38/38 in 96.38 seconds. This is the
  authoritative GPU validation for the iterative 258-transition replay oracle,
  existing cache objectives/indexing/concurrency, environment geometry,
  analyzer gates, mixed VJPs, and the new audit helper contracts.
- Corrected audit job `5035049` was assigned to known-ECC node `kn161` before
  the exclusion could take effect. Cancelled it after 27 seconds and before
  model execution; its partial `audit-v2` provenance is retained. Request
  `5035079` excluding that node remained pending on the short partition and was
  cancelled at zero runtime. Identical replacement `5035096` is pending on the
  12-hour L40S partition, still excluding `kn161`, with new `audit-v3` output.
- Follow-up manifest audit found that the two already-added `LocalClock`
  final-only checkpoint regressions were not listed in `test_core.sh`. Added
  them to both its source checksum and pytest command; they pass 2/2 locally.
  The next/final authoritative CUDA run will therefore contain 40 tests. This
  does not invalidate the 38-test pass, but it prevents that pass from being
  presented as the eventual final manifest.
- Queued 40-test CUDA job `5035220` with `afterany:5035096`, still excluding
  `kn161`, so it will run as soon as the read-only audit releases the second
  GPU regardless of the audit's scientific outcome.

### 2026-08-26 16:34 UTC - failed distance-257 checkpoint audit completed

- Corrected read-only audit job `5035096` completed on healthy L40S node
  `kn033` in 2:26 and wrote the new immutable target
  `runs/audits/distance-257-failed-seed9407/audit-v3.json`. It verified the
  exact 777,000-row, 3,000-episode checkpoint/replay pair and reports that it
  called neither `Agent.train()` nor `Replay.update()` and did not modify the
  source artifact.
- Exact replay behavior was chance: 1,443 positive versus 1,557 negative
  terminal rewards (0.481 accuracy), with both cues well covered but an action
  imbalance of 1,912 versus 1,088. Conditional on cached query states, the
  current policy reached only 0.591 argmax accuracy and 0.592 expected sampled
  accuracy. Its very low mean entropy (0.054 nats) therefore reflects a mostly
  confident biased policy, not acquisition of the dependency.
- The stochastic-prior counterfactual audit was above chance but insufficient:
  the model-greedy action was correct for 0.649 of cached states and 0.643 of
  full-current posterior reconstructions. Cached-state and full-current
  posterior policy accuracies were similarly weak (0.591 and 0.576). This
  similarity does not establish equivalence to historical training states,
  whose parameter versions and RNG keys were never stored, but it gives no
  evidence that merely reconstructing states under the final parameters
  restores the missing behavior.
- The held-out nearest-centroid cue probe was at chance for `deter`, `stoch`,
  and their concatenation in both cached and full-current states (balanced
  accuracies 0.504--0.506 and 0.489--0.492). This is evidence only against
  simple linear cue separability, not proof that all cue information is absent.
  Together with the weak policy and counterfactual results, it localizes the
  failed gate to an insufficiently usable query representation/model-policy
  signal rather than a runtime error. It does not identify cache transport as
  the cause and cannot convert the failed distance-257 run into a pass.

### 2026-08-26 16:34 UTC - final focused CUDA manifest passed

- Authoritative job `5035220` ran on healthy L40S node `kn054` and passed the
  corrected complete manifest: 40/40 tests in 98.36 seconds. This adds the two
  final-only checkpoint clock regressions to all coverage already established
  by the 38-test pass, including the frozen actual-RSSM/actual-Replay
  258-transition reverse-sweep oracle. Source revision for the queued job was
  `b834432c7cd127ad60e88c417ec1faa4cd344591`; the later worklog-only queue
  record does not change tested code.

### 2026-08-26 16:40 UTC - first-boundary matched control prepared

- The live fresh distance-65 cache-on run reached 972 episodes without the
  abrupt acquisition seen in the matched distance-8 run: cumulative accuracy
  was 0.504, last-200 accuracy 0.520, and last-100 accuracy 0.480. This is an
  interim localization observation, not its predeclared 3,000-episode gate.
- Prepared the nearest shorter matched control at literal distance 63. Its
  65-row episodes and 64 optimized learner rows place the query at the end of
  the native learner span; distance 65 is the first selected condition beyond
  that span. The control remains fresh/no-checkpoint/no-curriculum, cache-on,
  seed 9407, `repval_grad=False`, default world-model losses, uniform replay,
  and final-only checkpointing. The exact budget is 195,000 rows = 3,000
  episodes. No optimizer, learning-rate, architecture, loss-scale, replay, or
  cache setting differs from the distance-65 arm.
- Launched that control as Slurm job `5035321` on L40S node `kn033`, excluding
  the known-ECC node. Its recorded source revision is
  `30218981a7959441f3d734815b35af66d7a7984d` and its recorded launch-time tree
  is clean.

### 2026-08-26 16:47 UTC - online-only cache seam audit

- Two independent read-only source/artifact reviews found no concrete defect
  in batch scaling, complete RSSM-state handling, delayed-output IDs, reset
  masks, physical row alignment, multi-payload ordering/atomicity, uniform
  replay coverage, BF16 representability, or the `repval_grad=False`
  interaction. The 65-distance snapshot had roughly 16 sampled windows per
  inserted row, 3.17 million cache-row updates per report interval, finite
  adjoints, and future use 0.9839 versus eligible nonterminal fraction
  66/67 = 0.9851. These facts establish availability and plumbing, not useful
  reward-credit depth.
- The single material untested assumption is the stochastic/moving-target
  boundary approximation. An interior cached `(S_q, G_q)` pair comes from one
  older forward pass, while a preceding sampled window contracts `G_q` with a
  newly reconstructed endpoint produced by an independent hard categorical
  posterior draw and newer parameters. The replay-backed 258-transition oracle
  deliberately freezes parameters, uses FP32, and forces deterministic
  posterior categories so it does not qualify this online seam. This is a
  hypothesis and an explicit limitation of the requested no-freshness design,
  not evidence for versions, rejection, EMA, resampling, or other forbidden
  freshness machinery.
- The running distance-63 control is therefore the cheapest decisive
  falsifier already in flight: if 63 also fails, the 65 trajectory cannot be
  attributed specifically to a bootstrapped boundary; if 63 passes and 65
  fails, the first online cached-adjoint boundary becomes localized without
  yet proving which stochastic/optimization mechanism caused it.
- Provenance note for dependent distance-129 job `5034475`: if it becomes
  eligible, it will read the then-current checkout. Relative to distance 65,
  the current `agent.py` differs only by the semantics-preserving extraction of
  unchanged payload slices into `_gradient_cache_payloads()`, validated by the
  40/40 CUDA suite. Retain the job if 65 passes, but report it as the same
  learner semantics/config under a tested mechanical refactor rather than as a
  byte-identical source artifact. Any further training-source change before
  allocation would require cancelling and requeuing it.

### 2026-08-26 16:59 UTC - full-BPTT diagnostic interface prepared

- Source geometry review clarified that increasing a context-one learner span
  still begins from detached cached `S_q0`; it is not literal full BPTT through
  the cue observation. A genuine oracle must set `replay_context=0` so the
  reset cue itself is inside the differentiable learner sequence, disable the
  gradient cache, and include a complete `q0..q_(d+1)` episode.
- Added guarded, recorded `BATCH_LENGTH` and `REPLAY_CONTEXT` environment knobs
  to the toy launcher. Their defaults remain exactly 64 and 1, respectively,
  and cache-on launches explicitly reject any context other than 1. Existing
  production/localization commands are therefore unchanged. These knobs are
  solely to make a possible cache-off full-BPTT diagnostic auditable; no such
  experiment has been launched, and the running 63/65 jobs are unaffected.
