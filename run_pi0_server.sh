#!/bin/bash
# =============================================================================
# Start the pi0 policy server using OpenPI
#
# This runs in the openpi Python 3.11 venv and serves the pi0 model
# over a WebSocket for the eval client to connect to.
#
# Usage:
#   # Default: serve pi0.5 LIBERO model on port 8000
#   bash run_pi0_server.sh
#
#   # Specify GPU, port, and environment
#   CUDA_VISIBLE_DEVICES=1 bash run_pi0_server.sh --port 8001 --env LIBERO
#
#   # Use a custom checkpoint
#   bash run_pi0_server.sh policy:checkpoint \
#     --policy.config pi05_libero \
#     --policy.dir /path/to/your/checkpoint
# =============================================================================

set -e

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Default to GPU 0 if not set
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

OPENPI_DIR="/app/openpi"
PYTHON="${OPENPI_DIR}/.venv/bin/python"

echo "============================================"
echo "  pi0 Policy Server"
echo "  Python: ${PYTHON}"
echo "  GPU: ${CUDA_VISIBLE_DEVICES}"
echo "  Args: $@"
echo "============================================"

cd "${OPENPI_DIR}"

# Default: serve LIBERO environment on port 8000
if [ $# -eq 0 ]; then
    exec ${PYTHON} scripts/serve_policy.py --env LIBERO --port 8000
else
    exec ${PYTHON} scripts/serve_policy.py "$@"
fi
