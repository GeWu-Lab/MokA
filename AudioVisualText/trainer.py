# Adopted from: https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py
import os
import json
from typing import List, Optional, Union, Any, Mapping

import torch
import torch.nn as nn
from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
    TRAINER_STATE_NAME,
)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]

def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class UnifiedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._grad_sens_latest: Optional[Mapping[str, float]] = None
        self._grad_sens_jsonl_path: Optional[str] = None
        self._grad_sens_last_logged_step: int = -1
        self._grad_sens_hook_handles = []
        self._grad_sens_hooked_model_id: Optional[int] = None
        self._grad_sens_step_g2 = {}

    def _grad_sensitivity_enabled(self) -> bool:
        return bool(getattr(self.args, "grad_sensitivity_enable", False))

    def _grad_group_name(self, param_name: str) -> Optional[str]:
        """
        Map parameter names to analysis groups.
        LoRA branch mapping is based on this project's custom LoRA implementation:
        - lora_A0: text branch
        - lora_A1: visual branch
        - lora_A2: audio branch
        - lora_B0: shared projection
        """
        if "lora_A0" in param_name:
            return "lora_A_text"
        if "lora_A1" in param_name:
            return "lora_A_visual"
        if "lora_A2" in param_name:
            return "lora_A_audio"
        if "lora_B0" in param_name:
            return "lora_B_shared"

        if getattr(self.args, "grad_sensitivity_include_projectors", True):
            if "vl_projector" in param_name:
                return "vl_projector"
            if "al_projector" in param_name:
                return "al_projector"
        return None

    def _ensure_grad_sensitivity_hooks(self, model: nn.Module) -> None:
        """
        Register backward hooks once on the active training model.
        This is more reliable than reading param.grad directly under DeepSpeed/ZeRO.
        """
        model_id = id(model)
        if self._grad_sens_hooked_model_id == model_id and self._grad_sens_hook_handles:
            return

        for handle in self._grad_sens_hook_handles:
            handle.remove()
        self._grad_sens_hook_handles = []
        self._grad_sens_hooked_model_id = model_id
        self._grad_sens_step_g2 = {}

        for name, param in model.named_parameters():
            group = self._grad_group_name(name)
            if group is None or not param.requires_grad:
                continue
            self._grad_sens_step_g2.setdefault(group, 0.0)

            def _make_hook(group_name):
                def _hook(grad):
                    if grad is None:
                        return
                    g = grad.detach().float()
                    self._grad_sens_step_g2[group_name] += float((g * g).sum().item())
                return _hook

            self._grad_sens_hook_handles.append(param.register_hook(_make_hook(group)))

    def _reset_grad_sensitivity_step_accumulator(self) -> None:
        for group in list(self._grad_sens_step_g2.keys()):
            self._grad_sens_step_g2[group] = 0.0

    def _collect_grad_sensitivity(self, model: nn.Module) -> Mapping[str, float]:
        eps = 1e-12
        # Floor used to stabilize relative-grad metric when a group starts near zero
        # (e.g., LoRA B matrices initialized to zeros).
        param_norm_floor = float(getattr(self.args, "grad_sensitivity_param_norm_floor", 1e-2))
        accum = {}
        for name, param in model.named_parameters():
            group = self._grad_group_name(name)
            if group is None:
                continue
            if group not in accum:
                accum[group] = {"p2": 0.0, "n": 0}

            # Parameter norm is always tracked when present for scale reference.
            p = param.detach().float()
            accum[group]["p2"] += float((p * p).sum().item())
            accum[group]["n"] += int(param.numel())

        metrics = {}
        for group, vals in accum.items():
            # Gradient norm comes from backward hooks (more robust in ZeRO setups).
            # NOTE: hook accumulation is already per-group, so read it once here.
            g2 = float(self._grad_sens_step_g2.get(group, 0.0))
            grad_norm = g2 ** 0.5
            param_norm = vals["p2"] ** 0.5
            # Size-normalized gradient scale (independent of parameter magnitude).
            grad_rms = grad_norm / ((vals["n"] ** 0.5) + eps)
            # Original relative gradient norm.
            rel_grad_norm = grad_norm / (param_norm + eps)
            # Stabilized relative metric to reduce denominator-near-zero inflation.
            rel_grad_norm_floored = grad_norm / (max(param_norm, param_norm_floor) + eps)
            metrics[f"grad_sens/{group}_grad_norm"] = grad_norm
            metrics[f"grad_sens/{group}_grad_rms"] = grad_rms
            metrics[f"grad_sens/{group}_param_norm"] = param_norm
            metrics[f"grad_sens/{group}_relative_grad_norm"] = rel_grad_norm
            metrics[f"grad_sens/{group}_relative_grad_norm_floored"] = rel_grad_norm_floored
            metrics[f"grad_sens/{group}_num_params"] = float(vals["n"])
        return metrics

    def _append_grad_sensitivity_jsonl(self, metrics: Mapping[str, float]) -> None:
        if not self.is_world_process_zero():
            return
        if self._grad_sens_jsonl_path is None:
            self._grad_sens_jsonl_path = os.path.join(self.args.output_dir, "grad_sensitivity.jsonl")
        payload = {"step": int(self.state.global_step), "epoch": float(self.state.epoch or 0.0)}
        payload.update(metrics)
        with open(self._grad_sens_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def training_step(self, model: nn.Module, inputs: Mapping[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if self._grad_sensitivity_enabled():
            self._ensure_grad_sensitivity_hooks(model)
            self._reset_grad_sensitivity_step_accumulator()
        loss = super().training_step(model, inputs)
        if self._grad_sensitivity_enabled():
            # Capture per-step grad statistics accumulated by hooks during backward.
            self._grad_sens_latest = self._collect_grad_sensitivity(model)
        return loss

    def log(self, logs: Mapping[str, float]) -> None:
        logs = dict(logs)
        if (
            self._grad_sensitivity_enabled()
            and self._grad_sens_latest is not None
            and self.state.global_step != self._grad_sens_last_logged_step
        ):
            logs.update(self._grad_sens_latest)
            self._append_grad_sensitivity_jsonl(self._grad_sens_latest)
            self._grad_sens_last_logged_step = int(self.state.global_step)
        super().log(logs)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            # dtype = self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()
            # print('get train sampler: ',dtype)
            return super()._get_train_sampler()


    def _save_checkpoint(self, model, trial, metrics=None):
        # print('save checkpints...')
        # if getattr(self.args, 'tune_mm_mlp_adapter', False):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir,exist_ok=True)

        # if self.args.lora_enable:
        #     state_dict = get_peft_state_maybe_zero_3(self.model.named_parameters(), self.args.lora_bias)
        #     non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
        #     if self.args.local_rank == 0:
        #         self.model.config.save_pretrained(output_dir)
        #         self.model.save_pretrained(output_dir, state_dict=state_dict)
        #         torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))
        # else:
        #     # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        #     non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
        #     if self.args.local_rank == 0:
        #         self.model.config.save_pretrained(output_dir)
        #         torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))

        # Only save Adapter
        # keys_to_match = ['vl_projector', 'al_projector']
        # if self.args.lora_enable:
        #     keys_to_match.append('lora')
        ### for avs
        # keys_to_match = ['seg_module']
        keys_to_match = self.args.save_modules.split(',')
        
        weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            self.model.config.save_pretrained(output_dir)
            torch.save(weight_to_save, os.path.join(output_dir, f'finetune_weights.bin'))

        # Save optimizer and scheduler
        # self._save_optimizer_and_scheduler(output_dir)
        # self._save_rng_state(output_dir)
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        # self.args.distributed_state.wait_for_everyone()
        # else:
        #     super(VideoLLaMA2Trainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(UnifiedTrainer, self)._save(output_dir, state_dict)

   
