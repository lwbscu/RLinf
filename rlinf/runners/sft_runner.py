# Copyright 2025 The RLinf Authors.
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

import logging
import os
import subprocess
from typing import TYPE_CHECKING, Optional, Union

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.runner_utils import EarlyStopController, check_progress

if TYPE_CHECKING:
    from rlinf.workers.reward.reward_worker import FSDPRewardWorker
    from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

logger = logging.getLogger(__name__)


class _PostSaveEvalFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


class SFTRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union["FSDPSftWorker", "FSDPRewardWorker"],
        run_timer: Optional[ScopedTimer] = None,
    ) -> None:
        self.cfg = cfg
        self.actor = actor

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0
        early_stop_cfg = cfg.runner.get("early_stop", None)
        self.early_stop = (
            EarlyStopController(early_stop_cfg) if early_stop_cfg is not None else None
        )

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.metric_logger = MetricLogger(cfg)

    def init_workers(self) -> None:
        # create worker in order to decrease the maximum memory usage
        self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def run(self) -> None:
        start_step = self.global_step
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Global Step",
            ncols=800,
        )
        for _step in range(start_step, self.max_steps):
            if hasattr(self.actor, "set_global_step"):
                # set global step
                self.actor.set_global_step(self.global_step)

            with self.timer("step"):
                actor_handle: Handle = self.actor.run_training()
                actor_metrics = actor_handle.wait()

                self.global_step += 1

                eval_model, save_model, _ = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                if save_model:
                    checkpoint_dir = self._save_checkpoint()
                    self._run_post_save_eval(checkpoint_dir)

                should_stop = False
                if eval_model:
                    eval_handle: Handle = self.actor.run_eval()
                    eval_metrics = eval_handle.wait()

                    if self.early_stop is not None:
                        should_stop, best_val_acc_improved = self.early_stop.update(
                            eval_metrics[0]
                        )
                        if best_val_acc_improved:
                            self._save_checkpoint(is_best=True)

            time_metrics = self.timer.consume_durations()
            time_metrics["training"] = actor_handle.consume_duration()
            if eval_model:
                time_metrics["evaluate"] = eval_handle.consume_duration()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            merged_actor_metrics = {}
            # get the merged actor metrics from all ranks
            for metrics in actor_metrics:
                for k, v in metrics.items():
                    if k not in merged_actor_metrics:
                        merged_actor_metrics[k] = v
            training_metrics = {
                f"train/{k}": v for k, v in merged_actor_metrics.items()
            }
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)

            logging_metrics = time_metrics
            logging_metrics.update(training_metrics)

            if eval_model:
                evaluate_metrics = {f"eval/{k}": v for k, v in eval_metrics[0].items()}
                logging_metrics.update(evaluate_metrics)
                self.metric_logger.log(evaluate_metrics, _step)

            global_pbar.set_postfix(logging_metrics, refresh=False)
            global_pbar.update(1)
            if should_stop:
                break

        if self.early_stop is not None and self.early_stop.best_val_acc > 0:
            logger.info(
                f"Early stopping triggered! Best val_acc: {self.early_stop.best_val_acc:.4f}"
            )
        self.metric_logger.finish()

    def run_eval(self) -> None:
        with self.timer("evaluate"):
            eval_handle: Handle = self.actor.run_eval()
            eval_metrics = eval_handle.wait()

        time_metrics = self.timer.consume_durations()
        time_metrics["evaluate"] = eval_handle.consume_duration()
        time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}

        raw_eval = (
            eval_metrics[0]
            if isinstance(eval_metrics, (list, tuple)) and len(eval_metrics) > 0
            else {}
        )
        evaluate_metrics = {f"eval/{k}": v for k, v in raw_eval.items()}

        logging_metrics = {}
        logging_metrics.update(time_metrics)
        logging_metrics.update(evaluate_metrics)

        logger.info(f"Eval metrics: {evaluate_metrics}")
        self.metric_logger.finish()

    def _save_checkpoint(self, is_best: bool = False) -> str:
        checkpoint_root = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
        )
        if is_best:
            base_output_dir = os.path.join(checkpoint_root, "checkpoints/best_model")
        else:
            base_output_dir = os.path.join(
                checkpoint_root,
                f"checkpoints/global_step_{self.global_step}",
            )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()
        if is_best and self.early_stop is not None:
            logger.info(
                f"Saved best model (val_acc={self.early_stop.best_val_acc:.4f}) to {base_output_dir}"
            )
        return base_output_dir

    def _run_post_save_eval(self, checkpoint_dir: str) -> None:
        eval_cfg = self.cfg.runner.get("post_save_eval", None)
        if eval_cfg is None or not bool(eval_cfg.get("enabled", False)):
            return

        command_template = eval_cfg.get("command", None)
        if not command_template:
            raise ValueError(
                "runner.post_save_eval.enabled=True requires "
                "runner.post_save_eval.command to be set."
            )

        actor_dir = os.path.join(checkpoint_dir, "actor")
        full_weights_path = os.path.join(
            actor_dir, "model_state_dict", "full_weights.pt"
        )
        if "{full_weights_path}" in command_template and not os.path.exists(
            full_weights_path
        ):
            raise FileNotFoundError(
                "runner.post_save_eval.command references {full_weights_path}, "
                f"but the file does not exist: {full_weights_path}"
            )

        eval_log_dir = eval_cfg.get("eval_log_dir", None)
        if not eval_log_dir:
            eval_log_dir = os.path.join(
                self.cfg.runner.logger.log_path,
                self.cfg.runner.logger.experiment_name,
                "eval",
                f"global_step_{self.global_step}",
            )
        os.makedirs(eval_log_dir, exist_ok=True)

        format_values = _PostSaveEvalFormatDict(
            checkpoint_dir=checkpoint_dir,
            actor_dir=actor_dir,
            full_weights_path=full_weights_path,
            global_step=self.global_step,
            eval_log_dir=eval_log_dir,
        )
        command = str(command_template).format_map(format_values)
        logger.info(
            "Running post-save eval for global_step_%s: %s",
            self.global_step,
            command,
        )

        offloaded = False
        if bool(eval_cfg.get("offload_actor", False)):
            self._call_actor_helper("offload_for_external_eval")
            offloaded = True

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=eval_cfg.get("cwd", None),
                check=False,
            )
            self.metric_logger.log(
                {"post_save_eval/returncode": float(result.returncode)},
                self.global_step,
            )
            if result.returncode != 0:
                message = (
                    "post-save eval command failed with return code "
                    f"{result.returncode}: {command}"
                )
                if bool(eval_cfg.get("fail_on_error", True)):
                    raise RuntimeError(message)
                logger.warning(message)
        finally:
            if offloaded:
                self._call_actor_helper("load_after_external_eval")

    def _call_actor_helper(self, helper_name: str) -> None:
        try:
            helper = getattr(self.actor, helper_name)
        except AttributeError as exc:
            raise AttributeError(
                f"Actor worker does not support {helper_name}()."
            ) from exc

        handle = helper()
        if hasattr(handle, "wait"):
            handle.wait()

    def set_max_steps(self) -> None:
        self.num_steps_per_epoch = self.actor.get_max_steps_per_epoch().wait()[0]
        max_epochs = self.cfg.runner.get("max_epochs", -1)
        max_steps = self.cfg.runner.get("max_steps", -1)

        step_limits = []
        if max_epochs > 0:
            step_limits.append(self.num_steps_per_epoch * max_epochs)
        if max_steps >= 0:
            step_limits.append(max_steps)

        # If both limits are configured, stop at whichever one is reached first.
        # If neither is configured, keep the historical default of one epoch.
        self.max_steps = min(step_limits) if step_limits else self.num_steps_per_epoch

    @property
    def epoch(self) -> int:
        return self.global_step // self.num_steps_per_epoch
