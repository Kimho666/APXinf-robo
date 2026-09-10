#! /bin/bash
#
# Run an APXinf-robo evaluation config through RLinf's own eval entrypoint.
#
#     RLINF=/path/to/RLinf scripts/rlinf_eval.sh libero_10_apxinf_robo_pi05_eval \
#         rollout.model.model_path=/ckpt/pi05_libero
#
# This is a thin stand-in for RLinf's `evaluations/run_eval.sh`, which resolves
# its `--config-path` from the config name and therefore only finds files inside
# RLinf's own tree. Everything else it does for the LIBERO benchmark -- the env
# exports, the log directory, the hydra invocation -- is reproduced here so that
# our config can stay in this repository, versioned with the adapter it
# configures, instead of being copied into someone else's checkout where it
# silently goes stale.
#
# It really is only the plumbing: `SRC_FILE` is RLinf's own
# `evaluations/eval_embodied_agent.py`, and every extra argument is a hydra
# override passed through untouched. To use RLinf's launcher instead, copy the
# config into `$RLINF/evaluations/libero/` and call `run_eval.sh` -- both routes
# reach the same entrypoint with the same config.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${APXINF_ROBO_CONFIG_PATH:-$(dirname "${HERE}")/configs/rlinf}"

if [ $# -lt 1 ]; then
    echo "Usage: RLINF=/path/to/RLinf $0 <config_name> [hydra_overrides...]" >&2
    echo >&2
    echo "Available configs in ${CONFIG_PATH}:" >&2
    for f in "${CONFIG_PATH}"/*.yaml; do
        [ -e "$f" ] && echo "    $(basename "${f%.yaml}")" >&2
    done
    exit 1
fi

CONFIG_NAME="$1"
shift
EXTRA_ARGS=("$@")

if [ ! -f "${CONFIG_PATH}/${CONFIG_NAME}.yaml" ]; then
    echo "No such config: ${CONFIG_PATH}/${CONFIG_NAME}.yaml" >&2
    exit 1
fi

# --- RLinf's tree ------------------------------------------------------------
#
# RLinf is not pip-installable, so it is located by path rather than imported.

: "${RLINF:?set RLINF to your RLinf checkout, e.g. RLINF=/path/to/RLinf $0 ...}"
REPO_PATH="$(cd "${RLINF}" && pwd)"
SRC_FILE="${REPO_PATH}/evaluations/eval_embodied_agent.py"

if [ ! -f "${SRC_FILE}" ]; then
    echo "RLINF=${REPO_PATH} does not look like an RLinf checkout: ${SRC_FILE} is missing." >&2
    exit 1
fi

# `EMBODIED_PATH` is not decoration: the config's `hydra.searchpath` resolves
# `env/libero_10` out of it, which is how we reuse RLinf's env definitions
# instead of vendoring a copy that could drift from the one openpi is scored on.
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

# --- simulator env, mirroring run_eval.sh's setup_sim_env + libero branch -----

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export LIBERO_TYPE="${LIBERO_TYPE:-standard}"

LOG_DIR="${LOG_DIR:-${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval_embodiment.log"

cmd=(
    python "${SRC_FILE}"
    --config-path "${CONFIG_PATH}/"
    --config-name "${CONFIG_NAME}"
    "runner.logger.log_path=${LOG_DIR}"
)
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    cmd+=("${EXTRA_ARGS[@]}")
fi

echo "RLinf:  ${REPO_PATH}"
echo "config: ${CONFIG_PATH}/${CONFIG_NAME}.yaml"
echo "logs:   ${LOG_DIR}"
echo "${cmd[*]}" | tee "${LOG_FILE}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
