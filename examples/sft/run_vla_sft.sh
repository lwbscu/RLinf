#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_vla_sft.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH

export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH=${DREAMZERO_PATH}:$PYTHONPATH

if [ $# -eq 0 ]; then
    CONFIG_NAME="robotwin_sft_lingbotvla"
else
    CONFIG_NAME=$1
    shift
fi

HYDRA_FLAGS=()
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    arg="$1"
    case "${arg}" in
        --cfg|--config-path|--config-name|--config-dir|--package|--info|--experimental-rerun)
            HYDRA_FLAGS+=("${arg}")
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value for Hydra flag ${arg}" >&2
                exit 1
            fi
            HYDRA_FLAGS+=("$1")
            ;;
        --help|--hydra-help|--version|--resolve|--run|--multirun|--shell-completion|--*=*)
            HYDRA_FLAGS+=("${arg}")
            ;;
        *)
            if [[ "${arg}" == runner.post_save_eval.command=* ]]; then
                value="${arg#runner.post_save_eval.command=}"
                if [[ "${value}" != \"* && "${value}" != \'* ]]; then
                    value="${value//\\/\\\\}"
                    value="${value//\"/\\\"}"
                    arg="runner.post_save_eval.command=\"${value}\""
                fi
            fi
            EXTRA_ARGS+=("${arg}")
            ;;
    esac
    shift
done

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD=(
    python "${SRC_FILE}"
    --config-path "${EMBODIED_PATH}/config/"
    --config-name "${CONFIG_NAME}"
)
if [ ${#HYDRA_FLAGS[@]} -gt 0 ]; then
    CMD+=("${HYDRA_FLAGS[@]}")
fi
CMD+=("runner.logger.log_path=${LOG_DIR}")
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi
printf "%q " "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf "\n" >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
