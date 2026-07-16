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

import os
from dataclasses import dataclass
from typing import Literal

from lerobot.configs.policies import PreTrainedConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def _import_lingbotvla_deps():
    try:
        from lingbotvla.data.vla_data.base_dataset import RobotwinDataset
        from lingbotvla.data.vla_data.transform import Normalizer
        from lingbotvla.models import build_processor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LingbotVLA SFT requires the external `lingbotvla` package. "
            "Install it with `bash requirements/install.sh embodied "
            "--model lingbotvla --env robotwin` before running "
            "`examples/sft/run_vla_sft.sh robotwin_sft_lingbotvla`."
        ) from exc

    return RobotwinDataset, Normalizer, build_processor


@dataclass
class LingbotDataConfig:
    norm_type: Literal[
        "meanstd", "bounds_99", "bounds_98", "bounds_98_woclip", "bounds_99_woclip"
    ] = "bounds_99_woclip"
    img_size: int = 224
    norm_stats_file: str = ""
    data_type: Literal["robotwin", "robotwin_rep"] = "robotwin_rep"


def _set_config_default(config, name, value):
    if not hasattr(config, name) or getattr(config, name) is None:
        setattr(config, name, value)


def _get_robotwin_data_type(lingbotvla_cfg):
    data_type = getattr(lingbotvla_cfg, "data_type", "robotwin_rep")
    if data_type not in {"robotwin", "robotwin_rep"}:
        raise ValueError(
            "LingbotVLA SFT data_type must be either 'robotwin' or "
            f"'robotwin_rep', got {data_type!r}."
        )
    return data_type


def _build_robotwin_dataset_cls(base_cls, normalizer_cls):
    class RobotwinDatasetWithRepNormalizer(base_cls):
        def __init__(self, *args, data_config=None, **kwargs):
            super().__init__(*args, data_config=data_config, **kwargs)
            data_type = getattr(data_config, "data_type", "robotwin_rep")
            self.normalizer = normalizer_cls(
                norm_stats=self.norm_stats["norm_stats"],
                from_file=True,
                data_type=data_type,
                norm_type={
                    "observation.images.cam_high": "identity",
                    "observation.images.cam_left_wrist": "identity",
                    "observation.images.cam_right_wrist": "identity",
                    "observation.state": data_config.norm_type,
                    "action": data_config.norm_type,
                },
            )

    return RobotwinDatasetWithRepNormalizer


def build_lingbot_sft_dataloader(cfg, world_size, global_rank, data_paths):
    base_dataset_cls, normalizer_cls, build_processor = _import_lingbotvla_deps()

    lingbotvla_cfg = getattr(
        cfg.actor.model,
        "lingbotvla",
        getattr(cfg.actor.model, "lingbot", cfg.actor.model),
    )
    config_path = getattr(
        lingbotvla_cfg,
        "config_path",
        os.path.join(os.environ.get("LINGBOT_VLA_PATH", ""), "lingbot-vla-4b"),
    )

    dataset_qwen_config = PreTrainedConfig.from_pretrained(config_path)
    dataset_qwen_config.n_action_steps = int(cfg.actor.model.num_action_chunks)
    dataset_qwen_config.action_dim = int(getattr(cfg.actor.model, "action_dim", 14))
    dataset_qwen_config.max_action_dim = int(
        getattr(cfg.actor.model, "max_action_dim", 75)
    )
    dataset_qwen_config.max_state_dim = int(
        getattr(cfg.actor.model, "max_state_dim", 75)
    )
    _set_config_default(
        dataset_qwen_config,
        "tokenizer_max_length",
        int(getattr(lingbotvla_cfg, "max_prompt_length", 512)),
    )
    _set_config_default(
        dataset_qwen_config,
        "resize_imgs_with_padding",
        [int(getattr(lingbotvla_cfg, "img_size", 224))] * 2,
    )

    processor = build_processor(cfg.actor.model.tokenizer_path)

    stats_path = getattr(
        lingbotvla_cfg,
        "stats_path",
        os.path.join(
            os.environ.get("LINGBOT_VLA_PATH", ""),
            "assets/norm_stats/robotwin_all_new.json",
        ),
    )
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"LingbotVLA RoboTwin stats file not found: {stats_path}"
        )

    data_config = LingbotDataConfig(
        norm_type=getattr(lingbotvla_cfg, "norm_type", "bounds_99_woclip"),
        norm_stats_file=stats_path,
        data_type=_get_robotwin_data_type(lingbotvla_cfg),
        img_size=int(getattr(lingbotvla_cfg, "img_size", 224)),
    )

    repo_id = data_paths if isinstance(data_paths, str) else data_paths[0]
    dataset_cls = _build_robotwin_dataset_cls(base_dataset_cls, normalizer_cls)
    dataset = dataset_cls(
        repo_id=repo_id,
        config=dataset_qwen_config,
        tokenizer=processor.tokenizer,
        data_config=data_config,
        image_processor=processor.image_processor,
        use_depth_align=False,
    )

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=global_rank, shuffle=True
    )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,
        sampler=sampler,
        num_workers=int(getattr(lingbotvla_cfg, "num_workers", 4)),
        pin_memory=bool(getattr(lingbotvla_cfg, "pin_memory", True)),
        drop_last=bool(getattr(lingbotvla_cfg, "drop_last", True)),
    )

    return data_loader, {}
