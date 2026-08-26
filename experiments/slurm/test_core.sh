#!/bin/bash
#SBATCH --account=aip-valenzan
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=dreamgrad-test
#SBATCH --output=experiments/logs/core-test-%j.out

set -euo pipefail

ROOT=/project/6101829/draip/DreamGrad
PYTHON=${ROOT}/.venv/bin/python
RUN=${ROOT}/experiments/test_runs/${SLURM_JOB_ID}
mkdir -p "${RUN}"
cd "${ROOT}"

module load cuda/12.6
module load cudnn/9.5.1.17
export LD_LIBRARY_PATH="${CUDNN_HOME}/lib:${CUDA_HOME}/lib:${EBROOTNCCL}/lib:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME} ${XLA_FLAGS:-}"
module -t list > "${RUN}/modules.txt" 2>&1 || true
nvidia-smi -q > "${RUN}/nvidia-smi.txt"
git status --porcelain --untracked-files=all > "${RUN}/git-status.txt"
git diff --binary HEAD -- > "${RUN}/working-tree.patch"
sha256sum \
  dreamerv3/agent.py \
  dreamerv3/configs.yaml \
  dreamerv3/main.py \
  dreamerv3/rssm.py \
  embodied/core/wrappers.py \
  embodied/envs/bsuite.py \
  embodied/envs/toy_memory.py \
  embodied/jax/internal.py \
  embodied/jax/opt.py \
  embodied/core/replay.py \
  experiments/analyze_memory.py \
  experiments/audit_toy_checkpoint.py \
  experiments/slurm/audit_toy_checkpoint.sh \
  tests/test_analyze_memory.py \
  tests/test_audit_toy_checkpoint.py \
  tests/test_gradient_cache_oracle.py \
  tests/test_iterative_gradient_cache.py \
  tests/test_rssm_gradient_cache_oracle.py \
  embodied/tests/test_gradient_cache_replay.py \
  embodied/tests/test_mixed_grad.py \
  embodied/tests/test_memory_envs.py \
  > "${RUN}/source-sha256.txt"

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1

"${PYTHON}" -m pytest -q \
  tests/test_analyze_memory.py \
  tests/test_audit_toy_checkpoint.py \
  tests/test_gradient_cache_oracle.py \
  tests/test_iterative_gradient_cache.py \
  tests/test_rssm_gradient_cache_oracle.py \
  embodied/tests/test_gradient_cache_replay.py \
  embodied/tests/test_mixed_grad.py \
  embodied/tests/test_memory_envs.py \
  | tee "${RUN}/pytest.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN}/TESTS_PASSED"
