#!/bin/bash
#SBATCH --account=aip-valenzan
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=dreamgrad-audit
#SBATCH --output=experiments/logs/audit-%j.out

set -euo pipefail

: "${TOY_LOGDIR:?Set TOY_LOGDIR to a completed ToyMemory artifact directory.}"
: "${AUDIT_OUTPUT:?Set AUDIT_OUTPUT to a new JSON path outside TOY_LOGDIR.}"
AUDIT_BATCH_SIZE=${AUDIT_BATCH_SIZE:-16}
AUDIT_MC_SAMPLES=${AUDIT_MC_SAMPLES:-64}
AUDIT_POSTERIOR_SAMPLES=${AUDIT_POSTERIOR_SAMPLES:-4}
AUDIT_SEED=${AUDIT_SEED:-0}

ROOT=/project/6101829/draip/DreamGrad
PYTHON=${ROOT}/.venv/bin/python
cd "${ROOT}"

if test -n "$(git status --porcelain --untracked-files=all)"; then
  echo 'Refusing scientific audit from a dirty worktree.' >&2
  exit 2
fi
for value in \
    "${AUDIT_BATCH_SIZE}" "${AUDIT_MC_SAMPLES}" \
    "${AUDIT_POSTERIOR_SAMPLES}"; do
  case "${value}" in
    ''|*[!0-9]*|0) echo 'Audit sizes must be positive integers.' >&2; exit 2 ;;
  esac
done
case "${AUDIT_SEED}" in
  ''|*[!0-9]*) echo 'AUDIT_SEED must be a nonnegative integer.' >&2; exit 2 ;;
esac
test -d "${TOY_LOGDIR}"
TOY_LOGDIR_REAL=$(realpath -e "${TOY_LOGDIR}")
AUDIT_OUTPUT_REAL=$(realpath -m "${AUDIT_OUTPUT}")
case "${AUDIT_OUTPUT_REAL}" in
  "${TOY_LOGDIR_REAL}"|"${TOY_LOGDIR_REAL}"/*)
    echo 'AUDIT_OUTPUT must be outside the source artifact.' >&2
    exit 2
    ;;
esac
if test -e "${AUDIT_OUTPUT}"; then
  echo "Refusing to overwrite audit output: ${AUDIT_OUTPUT}" >&2
  exit 2
fi

PROVENANCE=${AUDIT_OUTPUT}.provenance
if test -e "${PROVENANCE}"; then
  echo "Refusing to overwrite audit provenance: ${PROVENANCE}" >&2
  exit 2
fi
mkdir -p "$(dirname "${AUDIT_OUTPUT}")" "${PROVENANCE}"

module load cuda/12.6
module load cudnn/9.5.1.17
export LD_LIBRARY_PATH="${CUDNN_HOME}/lib:${CUDA_HOME}/lib:${EBROOTNCCL}/lib:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME} ${XLA_FLAGS:-}"
export JAX_PLATFORMS=cuda
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1

module -t list > "${PROVENANCE}/modules.txt" 2>&1 || true
nvidia-smi -q > "${PROVENANCE}/nvidia-smi.txt"
git rev-parse HEAD > "${PROVENANCE}/git-revision.txt"
git status --porcelain --untracked-files=all > \
  "${PROVENANCE}/git-status.txt"
sha256sum \
  experiments/audit_toy_checkpoint.py \
  dreamerv3/agent.py \
  dreamerv3/rssm.py \
  embodied/jax/agent.py \
  embodied/jax/transform.py \
  > "${PROVENANCE}/source-sha256.txt"
LATEST=$(<"${TOY_LOGDIR}/ckpt/latest")
CHECKPOINT=${TOY_LOGDIR}/ckpt/${LATEST}
test -f "${CHECKPOINT}/agent.pkl"
test -f "${CHECKPOINT}/done"
sha256sum \
  "${TOY_LOGDIR}/config.yaml" \
  "${CHECKPOINT}/agent.pkl" \
  "${CHECKPOINT}/step.pkl" \
  > "${PROVENANCE}/input-sha256.txt"

CMD=("${PYTHON}" experiments/audit_toy_checkpoint.py \
  "${TOY_LOGDIR}" \
  --batch-size "${AUDIT_BATCH_SIZE}" \
  --mc-samples "${AUDIT_MC_SAMPLES}" \
  --posterior-samples "${AUDIT_POSTERIOR_SAMPLES}" \
  --seed "${AUDIT_SEED}" \
  --output "${AUDIT_OUTPUT}")
printf '%q ' "${CMD[@]}" > "${PROVENANCE}/command.txt"
printf '\n' >> "${PROVENANCE}/command.txt"
"${CMD[@]}"
