#!/bin/bash
#SBATCH --account=aip-valenzan
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=dreamgrad-integ
#SBATCH --output=experiments/logs/integration-%j.out

set -euo pipefail

ROOT=/project/6101829/draip/DreamGrad
PYTHON=${ROOT}/.venv/bin/python
RUN=${ROOT}/experiments/test_runs/integration-${SLURM_JOB_ID}
: "${EXPECTED_REVISION:?Submit with the exact EXPECTED_REVISION.}"
mkdir -p "${RUN}/provenance"
cd "${ROOT}"

ACTUAL_REVISION=$(git rev-parse HEAD)
printf '%s\n' "${ACTUAL_REVISION}" > \
  "${RUN}/provenance/git-revision.txt"
if test "${ACTUAL_REVISION}" != "${EXPECTED_REVISION}"; then
  echo "Revision drift: expected ${EXPECTED_REVISION}, got ${ACTUAL_REVISION}." >&2
  exit 2
fi
git status --porcelain --untracked-files=all > \
  "${RUN}/provenance/git-status.txt"
if test -s "${RUN}/provenance/git-status.txt"; then
  echo 'Refusing authoritative integration from a dirty worktree.' >&2
  exit 2
fi

module load cuda/12.6
module load cudnn/9.5.1.17
export LD_LIBRARY_PATH="${CUDNN_HOME}/lib:${CUDA_HOME}/lib:${EBROOTNCCL}/lib:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME} ${XLA_FLAGS:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1

module -t list > "${RUN}/provenance/modules.txt" 2>&1 || true
nvidia-smi -q > "${RUN}/provenance/nvidia-smi.txt"
git diff --binary HEAD -- > "${RUN}/provenance/working-tree.patch"
sha256sum \
  dreamerv3/agent.py \
  dreamerv3/configs.yaml \
  dreamerv3/main.py \
  dreamerv3/rssm.py \
  embodied/envs/toy_memory.py \
  embodied/jax/internal.py \
  embodied/jax/opt.py \
  embodied/core/replay.py \
  embodied/run/train.py \
  experiments/check_integration.py \
  > "${RUN}/provenance/source-sha256.txt"

for ARM in cache-false cache-true cache-true-posterior-keys; do
  case "${ARM}" in
    cache-false)
      CACHE_FLAG=False
      POSTERIOR_RNG_KEYS_FLAG=False
      ;;
    cache-true)
      CACHE_FLAG=True
      POSTERIOR_RNG_KEYS_FLAG=False
      ;;
    cache-true-posterior-keys)
      CACHE_FLAG=True
      POSTERIOR_RNG_KEYS_FLAG=True
      ;;
  esac
  LOGDIR=${RUN}/${ARM}
  "${PYTHON}" dreamerv3/main.py \
    --logdir "${LOGDIR}" \
    --configs toy_memory size12m \
    --seed 9407 \
    --env.toymemory.distance 8 \
    --agent.gradient_cache.enabled "${CACHE_FLAG}" \
    --agent.gradient_cache.posterior_rng_keys \
      "${POSTERIOR_RNG_KEYS_FLAG}" \
    --agent.report False \
    --batch_size 2 \
    --batch_length 16 \
    --report_length 8 \
    --replay.size 10000 \
    --replay.chunksize 128 \
    --run.steps 100 \
    --run.train_ratio 32 \
    --run.envs 1 \
    --run.report_every 0 \
    --run.log_every -1 \
    --run.save_every 0 \
    --run.from_checkpoint '' \
    --run.debug True \
    --jax.prealloc False \
    --jax.expect_devices 1 \
    2>&1 | tee "${RUN}/${ARM}.log"

  if test "${CACHE_FLAG}" = True; then
    "${PYTHON}" experiments/check_integration.py \
      "${LOGDIR}" --cache-enabled \
      > "${RUN}/${ARM}-validation.json"
  else
    "${PYTHON}" experiments/check_integration.py \
      "${LOGDIR}" \
      > "${RUN}/${ARM}-validation.json"
  fi
done

"${PYTHON}" experiments/check_integration.py \
  "${RUN}/cache-false" \
  --compare-config "${RUN}/cache-true" \
  > "${RUN}/matched-control-validation.json"

date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN}/INTEGRATION_PASSED"
