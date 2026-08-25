#!/bin/bash
#SBATCH --account=aip-valenzan
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=dreamgrad-env
#SBATCH --output=experiments/logs/setup-env-%j.out

set -euo pipefail

ROOT=/project/6101829/draip/DreamGrad
VENV=${ROOT}/.venv
mkdir -p "${ROOT}/experiments/logs"
cd "${ROOT}"

module load python/3.11
module load cuda/12.6
module load cudnn/9.5.1.17
export LD_LIBRARY_PATH="${CUDNN_HOME}/lib:${CUDA_HOME}/lib:${EBROOTNCCL}/lib:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME} ${XLA_FLAGS:-}"
python -m venv --clear "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
# The cluster wheelhouse does not carry upstream's pinned jaxlib 0.4.33 for
# Python 3.11. Compute Canada's Python advertises the equivalent generic
# ``linux_x86_64`` tag rather than manylinux tags, so retag the exact official
# wheels after verifying their hashes. JAX uses the cluster CUDA module via its
# cuda12-local extra.
WHEELROOT=${SLURM_TMPDIR:-/tmp}/dreamgrad-wheels-${SLURM_JOB_ID}
mkdir -p "${WHEELROOT}"
curl -fsSL --retry 3 -o "${WHEELROOT}/jaxlib-0.4.33-cp311-cp311-manylinux2014_x86_64.whl" \
  'https://files.pythonhosted.org/packages/59/92/26f421354886d530ebf4e012addb7733c8ee10b5b5e2a3e01284944cc6bd/jaxlib-0.4.33-cp311-cp311-manylinux2014_x86_64.whl'
curl -fsSL --retry 3 -o "${WHEELROOT}/jax_cuda12_plugin-0.4.33-cp311-cp311-manylinux2014_x86_64.whl" \
  'https://files.pythonhosted.org/packages/36/6a/14b199d8a3e4de1fe5ba7338e9f0864ca06838b7b442fb3cd13f1becc450/jax_cuda12_plugin-0.4.33-cp311-cp311-manylinux2014_x86_64.whl'
curl -fsSL --retry 3 -o "${WHEELROOT}/jax_cuda12_pjrt-0.4.33-py3-none-manylinux2014_x86_64.whl" \
  'https://files.pythonhosted.org/packages/d3/1d/585fd2a2785f86e0a7a7562240ff3da9f9dc319782283d9548d6d417582f/jax_cuda12_pjrt-0.4.33-py3-none-manylinux2014_x86_64.whl'
(
  cd "${WHEELROOT}"
  sha256sum -c <<'EOF'
400f401498675fd42dcaf0b855f325691951b250d619a8cbc5955f947e2494aa  jaxlib-0.4.33-cp311-cp311-manylinux2014_x86_64.whl
ad8f8863ee8d5e11a867bd71b37e979939cad64d0f74efd52cdb37292517613e  jax_cuda12_plugin-0.4.33-cp311-cp311-manylinux2014_x86_64.whl
b43f199ec27fd9b3bb79b34ed297894ad50bb7a6eab62012baaa9ea6607b22de  jax_cuda12_pjrt-0.4.33-py3-none-manylinux2014_x86_64.whl
EOF
  "${VENV}/bin/python" -m wheel tags --remove \
    --platform-tag linux_x86_64 ./*manylinux2014_x86_64.whl
)
"${VENV}/bin/python" -m pip install --no-deps \
  "${WHEELROOT}/jaxlib-0.4.33-cp311-cp311-linux_x86_64.whl" \
  "${WHEELROOT}/jax_cuda12_plugin-0.4.33-cp311-cp311-linux_x86_64.whl" \
  "${WHEELROOT}/jax_cuda12_pjrt-0.4.33-py3-none-linux_x86_64.whl"
"${VENV}/bin/python" -m pip install -r requirements.txt
"${VENV}/bin/python" -m pip install -e . --no-deps
"${VENV}/bin/python" -m pip freeze > "${ROOT}/experiments/logs/pip-freeze-${SLURM_JOB_ID}.txt"
"${VENV}/bin/python" - <<'PY'
import jax
import jax.numpy as jnp
import ninjax
print('jax', jax.__version__)
print('ninjax', getattr(ninjax, '__version__', 'unknown'))
print('devices', jax.devices())
x = jnp.ones((1, 4, 4, 1), jnp.float32)
w = jnp.ones((2, 2, 1, 1), jnp.float32)
y = jax.lax.conv_general_dilated(
    x, w, (1, 1), 'VALID', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
print('cudnn_probe', y.block_until_ready().shape)
PY
date -u +%Y-%m-%dT%H:%M:%SZ > "${ROOT}/experiments/logs/ENV_READY"
