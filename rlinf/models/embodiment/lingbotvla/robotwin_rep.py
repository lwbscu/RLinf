# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

ROBOTWIN_ENV_STATE_TO_REP_INDICES = (
    *range(6),
    *range(7, 13),
    6,
    13,
)
ROBOTWIN_REP_STATE_TO_MODEL_INDICES = (
    *range(12),
    *range(73, 75),
    *range(12, 14),
    *range(14, 73),
)
ROBOTWIN_MODEL_ACTION_TO_ENV_INDICES = (
    *range(6),
    14,
    *range(6, 12),
    15,
)


def _select_last_dim(
    tensor: torch.Tensor, indices: tuple[int, ...], name: str
) -> torch.Tensor:
    required_dim = max(indices) + 1
    if tensor.shape[-1] < required_dim:
        raise ValueError(
            f"LingbotVLA RoboTwin expects {name} to have at least "
            f"{required_dim} dims, got {tensor.shape[-1]}."
        )
    index = torch.as_tensor(indices, device=tensor.device, dtype=torch.long)
    return tensor.index_select(-1, index)


def robotwin_env_state_to_rep(state: torch.Tensor) -> torch.Tensor:
    return _select_last_dim(
        state,
        ROBOTWIN_ENV_STATE_TO_REP_INDICES,
        "raw env state before robotwin_rep reorder",
    )


def robotwin_rep_state_to_model(state: torch.Tensor) -> torch.Tensor:
    return _select_last_dim(
        state,
        ROBOTWIN_REP_STATE_TO_MODEL_INDICES,
        "padded robotwin_rep state before model reorder",
    )


def robotwin_env_action_to_model(
    action: torch.Tensor, max_action_dim: int
) -> torch.Tensor:
    env_action_dim = len(ROBOTWIN_MODEL_ACTION_TO_ENV_INDICES)
    model_action_dim = max(ROBOTWIN_MODEL_ACTION_TO_ENV_INDICES) + 1
    if action.shape[-1] < env_action_dim:
        raise ValueError(
            "LingbotVLA RoboTwin expects normalized env action with at least "
            f"{env_action_dim} dims, got {action.shape[-1]}."
        )
    if max_action_dim < model_action_dim:
        raise ValueError(
            "LingbotVLA RoboTwin requires max_action_dim >= "
            f"{model_action_dim} for robotwin_rep gripper targets, got "
            f"{max_action_dim}."
        )

    model_action = action.new_zeros((*action.shape[:-1], max_action_dim))
    model_indices = torch.as_tensor(
        ROBOTWIN_MODEL_ACTION_TO_ENV_INDICES,
        device=action.device,
        dtype=torch.long,
    )
    model_action.index_copy_(-1, model_indices, action[..., :env_action_dim])
    return model_action


def robotwin_model_action_to_env(
    action: torch.Tensor, action_env_dim: int
) -> torch.Tensor:
    max_env_dim = len(ROBOTWIN_MODEL_ACTION_TO_ENV_INDICES)
    if not 0 < action_env_dim <= max_env_dim:
        raise ValueError(
            "LingbotVLA action_env_dim must be in the range "
            f"[1, {max_env_dim}] for RoboTwin action reorder, got "
            f"{action_env_dim}."
        )
    return _select_last_dim(
        action,
        ROBOTWIN_MODEL_ACTION_TO_ENV_INDICES,
        "model action before env action reorder",
    )[..., :action_env_dim]


def robotwin_normalizer_types(norm_type: str) -> dict[str, str]:
    return {
        "observation.images.cam_high": "identity",
        "observation.images.cam_left_wrist": "identity",
        "observation.images.cam_right_wrist": "identity",
        "observation.state": norm_type,
        "action": norm_type,
    }
