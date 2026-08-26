#!/bin/bash
#SBATCH --account=aip-valenzan
#SBATCH --partition=gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=dreamgrad-toy
#SBATCH --output=experiments/logs/toy-%j.out

set -euo pipefail

: "${DISTANCE:?Set literal cue-to-query DISTANCE.}"
: "${SEED:?Set model/environment SEED.}"
: "${STEPS:?Set fresh-run environment STEPS.}"
: "${CACHE_ENABLED:?Set CACHE_ENABLED=true or false.}"
RSSM_FREE_NATS=${RSSM_FREE_NATS:-1.0}
REPVAL_GRAD=${REPVAL_GRAD:-true}
MODEL_AUX_ENABLED=${MODEL_AUX_ENABLED:-true}
SAVE_EVERY=${SAVE_EVERY:-900}
BATCH_LENGTH=${BATCH_LENGTH:-64}
REPLAY_CONTEXT=${REPLAY_CONTEXT:-1}
POSTERIOR_RNG_KEYS=${POSTERIOR_RNG_KEYS:-false}

ROOT=/project/6101829/draip/DreamGrad
PYTHON=${ROOT}/.venv/bin/python
RUNROOT=${DREAMGRAD_RUNROOT:-${ROOT}/runs}
cd "${ROOT}"

if test -n "$(git status --porcelain --untracked-files=all)"; then
  echo 'Refusing scientific run from a dirty worktree.' >&2
  exit 2
fi
case "${CACHE_ENABLED}" in
  true) CACHE_FLAG=True ;;
  false) CACHE_FLAG=False ;;
  *) echo 'CACHE_ENABLED must be true or false.' >&2; exit 2 ;;
esac
case "${POSTERIOR_RNG_KEYS}" in
  true) POSTERIOR_RNG_KEYS_FLAG=True ;;
  false) POSTERIOR_RNG_KEYS_FLAG=False ;;
  *) echo 'POSTERIOR_RNG_KEYS must be true or false.' >&2; exit 2 ;;
esac
case "${SEED}" in
  ''|*[!0-9]*) echo 'SEED must be a nonnegative integer.' >&2; exit 2 ;;
esac
case "${DISTANCE}" in
  ''|*[!0-9]*) echo 'DISTANCE must be a positive integer.' >&2; exit 2 ;;
esac
case "${STEPS}" in
  ''|*[!0-9]*) echo 'STEPS must be a positive integer.' >&2; exit 2 ;;
esac
if [[ ! "${RSSM_FREE_NATS}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  echo 'RSSM_FREE_NATS must be a nonnegative decimal.' >&2
  exit 2
fi
case "${REPVAL_GRAD}" in
  true) REPVAL_GRAD_FLAG=True ;;
  false) REPVAL_GRAD_FLAG=False ;;
  *) echo 'REPVAL_GRAD must be true or false.' >&2; exit 2 ;;
esac
case "${MODEL_AUX_ENABLED}" in
  true) MODEL_AUX_ARGS=() ;;
  false) MODEL_AUX_ARGS=(
    --agent.loss_scales.rec 0.0
    --agent.loss_scales.con 0.0
    --agent.loss_scales.dyn 0.0
    --agent.loss_scales.rep 0.0
  ) ;;
  *) echo 'MODEL_AUX_ENABLED must be true or false.' >&2; exit 2 ;;
esac
if [[ ! "${SAVE_EVERY}" =~ ^[0-9]+$ ]]; then
  echo 'SAVE_EVERY must be 0 (final only) or a positive number of seconds.' >&2
  exit 2
fi
case "${BATCH_LENGTH}" in
  ''|*[!0-9]*) echo 'BATCH_LENGTH must be a positive integer.' >&2; exit 2 ;;
esac
case "${REPLAY_CONTEXT}" in
  ''|*[!0-9]*) echo 'REPLAY_CONTEXT must be a nonnegative integer.' >&2; exit 2 ;;
esac
test "${DISTANCE}" -ge 1
test "${STEPS}" -ge $((1000 * (DISTANCE + 2)))
test "${BATCH_LENGTH}" -ge 1
if test "${CACHE_ENABLED}" = true && test "${REPLAY_CONTEXT}" -ne 1; then
  echo 'CACHE_ENABLED=true requires REPLAY_CONTEXT=1.' >&2
  exit 2
fi
if test "${POSTERIOR_RNG_KEYS}" = true && test "${CACHE_ENABLED}" != true; then
  echo 'POSTERIOR_RNG_KEYS=true requires CACHE_ENABLED=true.' >&2
  exit 2
fi
if test $((STEPS % (DISTANCE + 2))) -ne 0 || test $((STEPS % 10)) -ne 0; then
  echo 'STEPS must end on both an episode boundary and driver block.' >&2
  exit 2
fi

ARM=cache-${CACHE_ENABLED}
if test "${POSTERIOR_RNG_KEYS}" = true; then
  ARM=${ARM}-posterior-keys
fi
LOGDIR=${RUNROOT}/toy/distance-${DISTANCE}/seed-${SEED}/${ARM}
if test -e "${LOGDIR}"; then
  echo "Refusing to reuse existing logdir: ${LOGDIR}" >&2
  exit 2
fi

mkdir -p "${LOGDIR}/provenance"
module load cuda/12.6
module load cudnn/9.5.1.17
export LD_LIBRARY_PATH="${CUDNN_HOME}/lib:${CUDA_HOME}/lib:${EBROOTNCCL}/lib:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME} ${XLA_FLAGS:-}"
export JAX_PLATFORMS=cuda
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
ENV_SEED=$("${PYTHON}" -c \
  'import sys; print(hash((int(sys.argv[1]), 0)) % (2 ** 32 - 1))' \
  "${SEED}")
module -t list > "${LOGDIR}/provenance/modules.txt" 2>&1 || true
nvidia-smi -q > "${LOGDIR}/provenance/nvidia-smi.txt"
git rev-parse HEAD > "${LOGDIR}/provenance/git-revision.txt"
git status --porcelain --untracked-files=all > \
  "${LOGDIR}/provenance/git-status.txt"
"${PYTHON}" -m pip freeze > "${LOGDIR}/provenance/pip-freeze.txt"
sha256sum \
  dreamerv3/agent.py \
  dreamerv3/configs.yaml \
  dreamerv3/rssm.py \
  embodied/jax/opt.py \
  embodied/core/replay.py \
  embodied/envs/toy_memory.py \
  docs/GRADIENT_CACHE.md \
  experiments/analyze_memory.py \
  experiments/slurm/run_toy.sh \
  requirements.txt \
  > "${LOGDIR}/provenance/source-sha256.txt"
{
  printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID}"
  printf 'SLURM_NODELIST=%s\n' "${SLURM_NODELIST}"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
  printf 'MODEL_SEED=%s\n' "${SEED}"
  printf 'EFFECTIVE_ENV_SEED=%s\n' "${ENV_SEED}"
  printf 'RSSM_FREE_NATS=%s\n' "${RSSM_FREE_NATS}"
  printf 'REPVAL_GRAD=%s\n' "${REPVAL_GRAD}"
  printf 'MODEL_AUX_ENABLED=%s\n' "${MODEL_AUX_ENABLED}"
  printf 'SAVE_EVERY=%s\n' "${SAVE_EVERY}"
  printf 'BATCH_LENGTH=%s\n' "${BATCH_LENGTH}"
  printf 'REPLAY_CONTEXT=%s\n' "${REPLAY_CONTEXT}"
  printf 'POSTERIOR_RNG_KEYS=%s\n' "${POSTERIOR_RNG_KEYS}"
  printf 'PYTHONHASHSEED=0\n'
} > "${LOGDIR}/provenance/environment.txt"

CMD=("${PYTHON}" dreamerv3/main.py \
  --logdir "${LOGDIR}" \
  --configs toy_memory size12m \
  --seed "${SEED}" \
  --env.toymemory.distance "${DISTANCE}" \
  --agent.dyn.rssm.free_nats "${RSSM_FREE_NATS}" \
  --agent.repval_grad "${REPVAL_GRAD_FLAG}" \
  --agent.gradient_cache.enabled "${CACHE_FLAG}" \
  --agent.gradient_cache.posterior_rng_keys "${POSTERIOR_RNG_KEYS_FLAG}" \
  --batch_length "${BATCH_LENGTH}" \
  --replay_context "${REPLAY_CONTEXT}" \
  --run.steps "${STEPS}" \
  --run.save_every "${SAVE_EVERY}" \
  --run.from_checkpoint '' \
  --run.envs 1 \
  --jax.expect_devices 1 \
  "${MODEL_AUX_ARGS[@]}")
printf '%q ' "${CMD[@]}" > "${LOGDIR}/provenance/command.txt"
printf '\n' >> "${LOGDIR}/provenance/command.txt"
"${CMD[@]}"

ANALYZE=("${PYTHON}" experiments/analyze_memory.py \
  "${LOGDIR}" --kind toy --length "${DISTANCE}")
printf '%q ' "${ANALYZE[@]}" > \
  "${LOGDIR}/provenance/analysis-command.txt"
printf '\n' >> "${LOGDIR}/provenance/analysis-command.txt"
"${ANALYZE[@]}" | tee "${LOGDIR}/analysis.json"
