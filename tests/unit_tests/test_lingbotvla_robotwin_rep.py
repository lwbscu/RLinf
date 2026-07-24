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

import pytest
import torch

from rlinf.models.embodiment.lingbotvla.robotwin_rep import (
    robotwin_env_action_to_model,
    robotwin_env_state_to_rep,
    robotwin_model_action_to_env,
    robotwin_rep_state_to_model,
)


def test_robotwin_env_state_to_rep_order():
    state = torch.arange(14)

    reordered = robotwin_env_state_to_rep(state)

    assert reordered.tolist() == [*range(6), *range(7, 13), 6, 13]


def test_robotwin_rep_state_to_model_order():
    state = torch.arange(75)

    reordered = robotwin_rep_state_to_model(state)

    assert reordered.tolist() == [
        *range(12),
        73,
        74,
        12,
        13,
        *range(14, 73),
    ]


def test_robotwin_action_round_trip_uses_16_model_dims():
    action = torch.arange(14, dtype=torch.float32).reshape(1, 1, 14)

    model_action = robotwin_env_action_to_model(action, max_action_dim=75)
    restored = robotwin_model_action_to_env(model_action, action_env_dim=14)

    assert model_action.shape == (1, 1, 75)
    assert model_action[..., :6].tolist() == action[..., :6].tolist()
    assert model_action[..., 6:12].tolist() == action[..., 7:13].tolist()
    assert model_action[..., 12:14].count_nonzero().item() == 0
    assert model_action[..., 14].item() == action[..., 6].item()
    assert model_action[..., 15].item() == action[..., 13].item()
    assert model_action[..., 16:].count_nonzero().item() == 0
    assert torch.equal(restored, action)


@pytest.mark.parametrize(
    ("action", "max_action_dim", "message"),
    [
        (torch.zeros(13), 75, "at least 14 dims"),
        (torch.zeros(14), 15, "max_action_dim >= 16"),
    ],
)
def test_robotwin_env_action_to_model_validates_dimensions(
    action: torch.Tensor, max_action_dim: int, message: str
):
    with pytest.raises(ValueError, match=message):
        robotwin_env_action_to_model(action, max_action_dim)


def test_robotwin_model_action_to_env_validates_dimensions():
    with pytest.raises(ValueError, match="at least 16 dims"):
        robotwin_model_action_to_env(torch.zeros(15), action_env_dim=14)

    with pytest.raises(ValueError, match="range"):
        robotwin_model_action_to_env(torch.zeros(75), action_env_dim=15)
