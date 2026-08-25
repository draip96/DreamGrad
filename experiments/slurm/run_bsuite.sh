#!/bin/bash
#SBATCH --account=aip-valenzan
#SBATCH --partition=gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=dreamgrad-bsuite
#SBATCH --output=experiments/logs/bsuite-%j.out

set -euo pipefail

: "${MEMORY_LENGTH:?Set official BSuite MEMORY_LENGTH.}"
: "${SEED:?Set model/environment SEED.}"
: "${REPVAL_GRAD:?Set REPVAL_GRAD=true or false explicitly.}"

case "${MEMORY_LENGTH}" in
  11|17|25|31|71) ;;
  *) echo 'MEMORY_LENGTH must be one of 11, 17, 25, 31, or 71.' >&2; exit 2 ;;
esac
case "${SEED}" in
  ''|*[!0-9]*) echo 'SEED must be a nonnegative integer.' >&2; exit 2 ;;
esac
case "${REPVAL_GRAD}" in
  true) REPVAL_GRAD_FLAG=True ;;
  false) REPVAL_GRAD_FLAG=False ;;
  *) echo 'REPVAL_GRAD must be true or false.' >&2; exit 2 ;;
esac

ROOT=/project/6101829/draip/DreamGrad
PYTHON=${ROOT}/.venv/bin/python
RUNROOT=${DREAMGRAD_RUNROOT:-${ROOT}/runs}
LOGDIR=${RUNROOT}/bsuite/memory-${MEMORY_LENGTH}/seed-${SEED}
STEPS=$((10000 * (MEMORY_LENGTH + 2)))
cd "${ROOT}"

if test -n "$(git status --porcelain --untracked-files=all)"; then
  echo 'Refusing scientific run from a dirty worktree.' >&2
  exit 2
fi
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
  dreamerv3/rssm.py \
  embodied/jax/opt.py \
  embodied/core/replay.py \
  embodied/envs/bsuite.py \
  docs/GRADIENT_CACHE.md \
  experiments/analyze_memory.py \
  experiments/slurm/run_bsuite.sh \
  requirements.txt \
  > "${LOGDIR}/provenance/source-sha256.txt"
{
  printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID}"
  printf 'SLURM_NODELIST=%s\n' "${SLURM_NODELIST}"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
  printf 'MODEL_SEED=%s\n' "${SEED}"
  printf 'EFFECTIVE_ENV_SEED=%s\n' "${ENV_SEED}"
  printf 'REPVAL_GRAD=%s\n' "${REPVAL_GRAD}"
  printf 'PYTHONHASHSEED=0\n'
} > "${LOGDIR}/provenance/environment.txt"

CMD=("${PYTHON}" dreamerv3/main.py \
  --logdir "${LOGDIR}" \
  --configs bsuite size12m \
  --seed "${SEED}" \
  --env.bsuite.memory_length "${MEMORY_LENGTH}" \
  --agent.repval_grad "${REPVAL_GRAD_FLAG}" \
  --agent.gradient_cache.enabled True \
  --run.steps "${STEPS}" \
  --run.from_checkpoint '' \
  --run.envs 1 \
  --jax.expect_devices 1)
printf '%q ' "${CMD[@]}" > "${LOGDIR}/provenance/command.txt"
printf '\n' >> "${LOGDIR}/provenance/command.txt"
"${CMD[@]}"

ANALYZE=("${PYTHON}" experiments/analyze_memory.py \
  "${LOGDIR}" --kind bsuite --length "${MEMORY_LENGTH}")
printf '%q ' "${ANALYZE[@]}" > \
  "${LOGDIR}/provenance/analysis-command.txt"
printf '\n' >> "${LOGDIR}/provenance/analysis-command.txt"
"${ANALYZE[@]}" | tee "${LOGDIR}/analysis.json"
