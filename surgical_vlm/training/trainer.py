"""
SurgicalTrainer - Complete multi-task training with temporal aggregator support.

Supports:
- Multi-task loss (phase, tool, action, triplet, language, grounding)
- Temporal aggregator integration
- LoRA fine-tuning
- Gradient accumulation, mixed precision
- Curriculum learning
- WandB logging
"""

import os
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from torch.amp import GradScaler as _AmpGradScaler, autocast as _amp_autocast
    _GradScaler = lambda: _AmpGradScaler("cuda")
    _autocast = lambda dtype: _amp_autocast("cuda", dtype=dtype)
except (ImportError, AttributeError):
    from torch.cuda.amp import GradScaler, autocast
    _GradScaler = GradScaler
    _autocast = autocast
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel

from surgical_vlm.models.surgical_vlm import SurgicalVLM, create_surgical_vlm
from surgical_vlm.models.base_vlm import BaseVLM
from surgical_vlm.training.lora_setup import apply_lora_to_model
from surgical_vlm.training.loss_functions import VLMNextTokenLoss, MultiTaskLoss
from surgical_vlm.training.output_adapter import OutputAdapter, TrainingOutputs, create_output_adapter
from surgical_vlm.data.vlm_dataset import VLMJSONLDataset
from surgical_vlm.data.collators import VLMCollator, MultiTaskCollator

logger = logging.getLogger(__name__)


def _normalize_label_list(value, batch_size):
    """Return value as a batch_size-aligned list/tensor, or None when misaligned."""
    if isinstance(value, torch.Tensor):
        return value if value.shape[0] == batch_size else None
    if not isinstance(value, (list, tuple)):
        value = [value]
    value = list(value)
    if len(value) != batch_size:
        return None
    return value


class MultiTaskSurgicalTrainer:
    """
    Complete multi-task surgical VLM trainer.
    
    Handles:
    - Multi-task heads (phase, tool, action, triplet, language, grounding)
    - Temporal aggregator integration
    - Multi-dataset training with proportional sampling
    - Curriculum learning stages
    - Checkpointing and resume
    """
    
    def __init__(
        self,
        model: SurgicalVLM,
        criterion: MultiTaskLoss,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: str,
        output_adapter: OutputAdapter,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        mixed_precision: str = "bf16",
        log_interval: int = 50,
        eval_interval: int = 500,
        save_interval: int = 1000,
        max_eval_samples: int = 500,
        output_dir: str = "checkpoints",
        use_wandb: bool = False,
        wandb_project: str = "surgical-vlm-multitask",
        label_mapper=None,
        temporal_aggregator=None,
        max_steps: Optional[int] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_adapter = output_adapter
        first_param = next((p for p in output_adapter.parameters()), None) if output_adapter is not None else None
        self._adapter_dtype = first_param.dtype if first_param is not None else torch.float32
        self.temporal_aggregator = temporal_aggregator
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.max_eval_samples = max_eval_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.label_mapper = label_mapper
        self.max_steps = max_steps
        
        # Mixed precision scaler. bf16 needs no gradient scaling; a GradScaler
        # would raise "Attempting to unscale FP16 gradients." in _GradScaler.unscale_.
        self.scaler = _GradScaler() if mixed_precision == "fp16" else None
        
        if use_wandb:
            import wandb
            wandb.init(project=wandb_project, config={
                "gradient_accumulation": gradient_accumulation_steps,
                "mixed_precision": mixed_precision,
            })
    
    def fit(self, train_loaders: Dict[str, DataLoader], val_loaders: Dict[str, DataLoader], epochs: int = 10):
        """Main training loop (NO curriculum learning - joint training from day 1)."""
        self.model.train_mode()
        global_step = 0
        max_steps = self.max_steps
        
        for epoch in range(epochs):
            # Early exit if global step budget already exhausted
            if max_steps is not None and global_step >= max_steps:
                logger.info(f"Reached max_steps={max_steps} at global_step={global_step}. Stopping training.")
                break
                
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Train on combined loader or individual loaders
            if hasattr(self, 'combined_loader') and self.combined_loader is not None:
                train_iterator = self.combined_loader
            else:
                # Cycle through loaders proportionally
                train_iterator = self._cycle_loaders(train_loaders)
            
            for batch_idx, batch in enumerate(train_iterator):
                # Hard stop: enforce max_steps budget on every iteration
                if max_steps is not None and global_step >= max_steps:
                    logger.info(f"Reached max_steps={max_steps}. Exiting epoch {epoch + 1} early.")
                    break
                # Training step
                loss_dict = self._training_step(batch)
                scalar_loss = loss_dict.get('total_loss', loss_dict.get('total'))
                if scalar_loss is None:
                    tensor_values = [v for v in loss_dict.values() if isinstance(v, torch.Tensor)]
                    if not tensor_values:
                        dataset_name = batch.get('dataset_name', 'unknown') if isinstance(batch, dict) else 'unknown'
                        raise RuntimeError(
                            f"No tensor loss values in loss dict for dataset '{dataset_name}' - "
                            "the batch was fully gated. Previously masked by a silent "
                            "torch.tensor(0.0) fallback; now a hard error."
                        )
                    scalar_loss = sum(tensor_values)
                
                global_step += 1
                
                if global_step % self.log_interval == 0:
                    loss_val = scalar_loss.item() if isinstance(scalar_loss, torch.Tensor) else float(scalar_loss)
                    logger.info(f"Step {global_step}, Loss: {loss_val:.4f}")
                    if self.use_wandb:
                        import wandb
                        log_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
                        wandb.log({"train_loss": loss_val, "step": global_step, "epoch": epoch, **log_dict})
                
                if global_step % self.eval_interval == 0:
                    self._validate(val_loaders, global_step)
                
                if global_step % self.save_interval == 0:
                    self.save_checkpoint(f"step_{global_step}")
            
            # If max_steps was hit during this epoch, stop all remaining epochs
            if max_steps is not None and global_step >= max_steps:
                logger.info(f"Global step budget exhausted at {global_step}. Finalizing.")
                break
        
        # Final validation so metrics are always produced for the run
        if val_loaders:
            logger.info("Training loop complete. Running final validation...")
            self._validate(val_loaders, global_step=global_step)
            logger.info("Final validation complete.")

        # Save final checkpoint
        self.save_checkpoint("final")
        logger.info("Training completed!")
    
    # REMOVED: _apply_curriculum method - curriculum learning is a trap
    
    def _cycle_loaders(self, loaders: Dict[str, DataLoader]):
        """Cycle through multiple loaders proportionally."""
        iterators = {name: iter(loader) for name, loader in loaders.items()}
        lengths = {name: len(loader) for name, loader in loaders.items()}
        total_batches = sum(lengths.values())
        
        for _ in range(total_batches):
            for name, iterator in iterators.items():
                try:
                    batch = next(iterator)
                    micro_batches = batch if isinstance(batch, list) else [batch]
                    for micro in micro_batches:
                        micro['dataset_name'] = name
                        yield micro
                except StopIteration:
                    # Reset iterator
                    iterators[name] = iter(loaders[name])
                    batch = next(iterators[name])
                    micro_batches = batch if isinstance(batch, list) else [batch]
                    for micro in micro_batches:
                        micro['dataset_name'] = name
                        yield micro
    
    def set_combined_loader(self, loader: DataLoader):
        """Set a pre-combined dataloader (e.g., with WeightedRandomSampler)."""
        self.combined_loader = loader
    
    def _training_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Single training step with multi-task loss.

        The collator may return either a single task-homogeneous batch (Dict)
        or a list of task-homogeneous micro-batches (List[Dict]) when the
        DataLoader batch mixed multiple tasks. Each micro-batch is forwarded
        independently through its task's loss path, then the per-task losses
        are mean-averaged (task-balanced) for the single optimizer step.

        Returns:
            Finalized mean loss dictionary for logging/metrics. The scalar
            loss used for backward() is extracted from loss_dict['total_loss']
            (or 'total').
        """
        micro_batches = batch if isinstance(batch, list) else [batch]
        scalar_losses: List[torch.Tensor] = []
        accumulated: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}

        for micro in micro_batches:
            micro = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in micro.items()}

            if self.scaler is not None:
                with _autocast(dtype=torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16):
                    loss_dict = self._forward_and_loss(micro)
            else:
                loss_dict = self._forward_and_loss(micro)

            scalar_loss = self._extract_scalar_loss(loss_dict)
            if scalar_loss is not None and scalar_loss.requires_grad:
                scalar_losses.append(scalar_loss)
                self._aggregate_loss_dicts(accumulated, counts, loss_dict)

        if not scalar_losses:
            return accumulated

        scalar_loss = sum(scalar_losses) / len(scalar_losses)

        self.optimizer.zero_grad()

        if self.scaler is not None:
            self.scaler.scale(scalar_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            scalar_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return self._finalize_aggregate(accumulated, counts)

    def _extract_scalar_loss(self, loss_dict: Dict) -> Optional[torch.Tensor]:
        """Extract a scalar (0-dim) loss tensor from a loss dict.

        Returns ``None`` when the dict contains no tensor values (fully-gated
        batch) so the caller can skip the optimizer step.
        """
        if isinstance(loss_dict, torch.Tensor):
            scalar_loss = loss_dict
        elif isinstance(loss_dict, dict):
            scalar_loss = loss_dict.get('total_loss', loss_dict.get('total'))
            if scalar_loss is None:
                tensor_values = [v for v in loss_dict.values() if isinstance(v, torch.Tensor)]
                if not tensor_values:
                    return None
                scalar_loss = sum(tensor_values)
        else:
            raise TypeError(f"_forward_and_loss returned unexpected type: {type(loss_dict)}")

        if not isinstance(scalar_loss, torch.Tensor):
            scalar_loss = torch.tensor(scalar_loss, device=self.device, dtype=torch.float32)
        if scalar_loss.dim() > 0:
            scalar_loss = scalar_loss.mean()
        return scalar_loss

    def _aggregate_loss_dicts(self, accumulated, counts, loss_dict) -> None:
        """Accumulate sum of each loss term across micro-batches.

        Values with dim > 0 (e.g. [B] per-sample losses) are summed and counted
        by element count so the finalized value is a true mean. Scalars count as
        one sample. ``total`` is normalized to ``total_loss``.
        """
        if isinstance(loss_dict, torch.Tensor):
            loss_dict = {'total_loss': loss_dict}
        for k, v in loss_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            k = 'total_loss' if k == 'total' else k
            v = v.detach().to(dtype=torch.float32)
            if k in accumulated:
                accumulated[k] = accumulated[k] + (v.sum() if v.dim() > 0 else v)
                counts[k] = counts[k] + (v.numel() if v.dim() > 0 else 1)
            else:
                accumulated[k] = v.sum() if v.dim() > 0 else v
                counts[k] = v.numel() if v.dim() > 0 else 1

    def _finalize_aggregate(
        self,
        accumulated: Dict[str, torch.Tensor],
        counts: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        """Convert accumulated sums into mean values for logging."""
        finalized = {}
        for k, total in accumulated.items():
            finalized[k] = total / counts.get(k, 1)
        finalized.setdefault('total_loss', torch.tensor(0.0, device=self.device, dtype=torch.float32))
        return finalized
    
    def _forward_and_loss(self, batch: Dict) -> Dict:
        """Forward pass and loss computation with temporal aggregation."""
        # Get model and ensure it's in train mode
        if hasattr(self.model, 'vlm') and self.model.vlm is not None:
            vlm = self.model.vlm
        else:
            vlm = self.model
        
        # Extract inputs
        pixel_values = batch.get('pixel_values')  # [B, T, C, H, W] | [B, C, H, W] | 2D patched [N, 1176] (B*T mode)
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels')
        image_grid_thw = batch.get('image_grid_thw')  # Qwen2.5-VL grid tensor
        video_grid_thw = batch.get('video_grid_thw')

        # Defense-in-depth: Qwen2.5-VL requires image_grid_thw whenever
        # pixel_values are present (see modeling_qwen2_5_vl.get_image_features).
        # A missing grid means the collator dropped it — fail loudly instead of
        # crashing deep in the model with a NoneType error.
        if pixel_values is not None and image_grid_thw is None:
            raise ValueError(
                "batch has pixel_values but image_grid_thw is None. The collator "
                "must emit image_grid_thw for every batch that carries "
                "pixel_values. Check HeadCollator/VQACollator output."
            )
        
        # ---- VQA task: standard HuggingFace causal language modeling loss ----
        # VQA rows (task == "vqa") carry an instruction prompt + expected answer.
        # Unlike the task heads (phase/tool/triplet loss via output_adapter) we run
        # a plain teacher-forced next-token loss over the base model's lm_head,
        # so gradients flow through the full Vision-LM backbone for captioning/
        # QA-style supervision.
        task = batch.get('task', '')
        if task == 'vqa':
            # Forward through the base Qwen3-VL model to get per-token hidden states.
            outputs = vlm.model(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            # Decoder output of the last layer -> [B, seq_len, hidden_dim]
            hidden_states = outputs.hidden_states[-1]

            # Project to vocabulary logits using the base model's lm_head.
            lm_head = getattr(vlm.model, 'lm_head', None)
            if lm_head is None and hasattr(vlm, 'lm_head'):
                lm_head = vlm.lm_head
            if lm_head is None:
                raise ValueError("VQA loss requires vlm.model.lm_head but none was found.")
            lm_logits = lm_head(hidden_states)                      # [B, seq_len, vocab]

            # Build per-token labels if the batch does not already carry them.
            # Convention: copy input_ids and mask padding with -100 (ignore_index),
            # identical to HF causal LM and our VLMCollator.
            lm_labels = labels
            if lm_labels is None:
                lm_labels = input_ids.clone()
                if attention_mask is not None:
                    lm_labels = lm_labels.masked_fill(attention_mask == 0, -100)

            # Standard causal next-token prediction: shift logits/labels by 1.
            lm_logits_shifted = lm_logits[:, :-1, :].contiguous()   # [B, seq_len-1, vocab]
            labels_shifted = lm_labels[:, 1:].contiguous()
            vqa_loss = F.cross_entropy(
                lm_logits_shifted.view(-1, lm_logits_shifted.size(-1)),
                labels_shifted.view(-1),
                ignore_index=-100,
            )

            return {
                'total_loss': vqa_loss,
                'vqa_loss': vqa_loss,
                'phase_loss': vqa_loss * 0.0,
                'tool_loss': vqa_loss * 0.0,
            }

        # Handle multi-frame input.
        # Two layouts are supported:
        #   1) Legacy 5D [B, T, C, H, W] (pre-patch mode) -> reshape to [B*T, C, H, W]
        #      so each frame runs through the vision encoder independently.
        #   2) 2D patched [B*T*N', 1176] produced by TemporalStackingCollator under B*T
        #      full-model expansion: pixel_values is already a flat 2D patch tensor and
        #      each sub-row i (0 <= i < B*T) carries frame t = i % T. The layout is
        #      recovered from the batch-wide frame_mask [B, T], which is never None here.
        is_multi_frame = pixel_values is not None and (
            pixel_values.dim() == 5 or batch.get('frame_mask') is not None
        )

        if is_multi_frame and pixel_values.dim() == 5:
            # [B, T, C, H, W] -> process each frame through vision encoder
            B, T, C, H, W = pixel_values.shape
            pixel_values = pixel_values.view(B * T, C, H, W)
            if image_grid_thw is not None and image_grid_thw.dim() == 3:
                image_grid_thw = image_grid_thw.view(B * T, -1)
            if input_ids is not None:
                input_ids = input_ids.view(B * T, -1)
            if attention_mask is not None:
                attention_mask = attention_mask.view(B * T, -1)
        elif is_multi_frame:
            # 2D patched B*T mode: pixel_values is already [B*T*N', 1176], so no reshape
            # is needed and input_ids/attention_mask/image_grid_thw are left untouched
            # (they are already laid out per sub-row by the collator). Recover B and T
            # from the batch-wide frame_mask so the temporal aggregator can reshape.
            frame_mask = batch.get('frame_mask')
            if frame_mask is None:
                raise ValueError(
                    "is_multi_frame was set from a 2D patched pixel_values but "
                    "batch['frame_mask'] is missing. TemporalStackingCollator must "
                    "emit frame_mask for B*T expansion."
                )
            B, T = frame_mask.shape
        
        # Forward through VLM
        outputs = vlm.model(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Get hidden states
        hidden_states = outputs.hidden_states[-1]  # [B*T, seq_len, hidden_dim] or [B, seq_len, hidden_dim]
        
        # Apply temporal aggregator if available and multi-frame
        if self.temporal_aggregator is not None and is_multi_frame:
            # Get vision features from the encoder output (before LM)
            # For Qwen2.5-VL, vision features are in outputs.hidden_states but we need the vision encoder output
            # Let's use the pooled hidden states as frame representations
            pooled = hidden_states.mean(dim=1)  # [B*T, hidden_dim]
            frame_features = pooled.view(B, T, -1)  # [B, T, hidden_dim]
            
            # Apply temporal aggregator
            frame_mask = batch.get('frame_mask', torch.ones(B, T, dtype=torch.bool, device=self.device))
            temporal_output = self.temporal_aggregator(frame_features, mask=frame_mask, pool=True)  # [B, hidden_dim]
            
            # Use temporal output for task heads
            training_outputs = self.output_adapter.forward_training(temporal_output.to(self._adapter_dtype))
        else:
            # Single frame or no temporal aggregator - pool over sequence
            pooled = hidden_states.mean(dim=1)  # [B, hidden_dim]
            training_outputs = self.output_adapter.forward_training(pooled.to(self._adapter_dtype))
        
        # ---- Full multi-task regime ----
        # All task heads (phase / instrument / action / triplet / grounding)
        # stay active and contribute to the loss via MultiTaskLoss.forward.
        # No logits are nulled here: the language/VQA branch is handled by the
        # early-return above, and language_logits is set by forward_training.

        # Prepare targets
        targets = self._prepare_targets(batch, pooled.shape[0] if not is_multi_frame else B)
        
        # Compute loss (pass dataset_name for grounding gating)
        dataset_name = batch.get('dataset_name', 'unknown')
        if isinstance(dataset_name, (list, tuple)):
            dataset_name = dataset_name[0] if len(dataset_name) else 'unknown'
        loss_dict = self.criterion(training_outputs, targets, dataset_name=dataset_name)

        # NaN/Inf diagnostic guard: dump per-head losses plus input/hidden
        # stats so the offending task and batch is identifiable from the log.
        scalar_tensors = [v for v in loss_dict.values() if isinstance(v, torch.Tensor) and v.numel() == 1]
        bad_tensors = [v for v in scalar_tensors if bool(torch.isnan(v).any() or torch.isinf(v).any())]
        total_loss = loss_dict.get('total_loss', loss_dict.get('total'))
        bad_float = isinstance(total_loss, float) and (total_loss != total_loss or total_loss in (float('inf'), float('-inf')))
        if bad_tensors or bad_float:
            print("=" * 80)
            print("[NAN-DIAG] NaN/Inf loss detected")
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    print(f"[NAN-DIAG]   loss {k}: numel={v.numel()} "
                          f"nan={bool(torch.isnan(v).any())} inf={bool(torch.isinf(v).any())} "
                          f"item={v.detach().float().item() if v.numel() == 1 else float(v.detach().float().nanmean()):.6f}")
                else:
                    print(f"[NAN-DIAG]   loss {k}: {v}")
            if pixel_values is not None:
                pv = pixel_values.float()
                print(f"[NAN-DIAG]   pixel_values: shape={tuple(pixel_values.shape)} "
                      f"finite={(torch.isfinite(pv).float().mean().item() * 100):.2f}% "
                      f"min={pv.min().item():.6f} max={pv.max().item():.6f} mean={pv.mean().item():.6f}")
            if input_ids is not None:
                print(f"[NAN-DIAG]   input_ids: shape={tuple(input_ids.shape)} "
                      f"min={input_ids.min().item()} max={input_ids.max().item()}")
            if labels is not None:
                lbl = labels.float()
                print(f"[NAN-DIAG]   labels: shape={tuple(labels.shape)} min={lbl.min().item()} "
                      f"max={lbl.max().item()} -100={(labels == -100).float().mean().item() * 100:.2f}%")
            hs = hidden_states.float()
            print(f"[NAN-DIAG]   hidden_states: shape={tuple(hidden_states.shape)} "
                  f"finite={(torch.isfinite(hs).float().mean().item() * 100):.2f}% "
                  f"mean={hs.mean().item():.6f} std={hs.std().item():.6f}")
            if image_grid_thw is not None:
                print(f"[NAN-DIAG]   image_grid_thw: {image_grid_thw.detach().cpu().tolist()}")
            print("=" * 80)

        return loss_dict
    
    def _prepare_targets(self, batch: Dict, batch_size: int) -> Dict:
        """Prepare targets for multi-task loss from batch."""
        targets = {}
        dataset_name = batch.get('dataset_name', 'unknown')
        if isinstance(dataset_name, (list, tuple)):
            dataset_name = dataset_name[0] if len(dataset_name) else 'unknown'
        
        # Convert text labels to indices using label mapper
        if self.label_mapper:
            # Phase labels
            phase_label = _normalize_label_list(batch.get('phase_label'), batch_size)
            if phase_label is not None and isinstance(phase_label, torch.Tensor):
                targets['phase_label'] = phase_label
            else:
                phase = _normalize_label_list(batch.get('phase'), batch_size)
                if phase is not None and all(p is not None for p in phase):
                    targets['phase_label'] = torch.tensor(
                        [self.label_mapper.text_to_phase_id(p) for p in phase],
                        device=self.device
                    )
            
            # Instrument labels
            instrument_label = _normalize_label_list(batch.get('instrument_label'), batch_size)
            if instrument_label is not None and isinstance(instrument_label, torch.Tensor):
                targets['instrument_label'] = instrument_label
            else:
                tools = _normalize_label_list(batch.get('tools'), batch_size)
                if (
                    tools is not None
                    and not isinstance(tools, torch.Tensor)
                    and all(t is not None for t in tools)
                ):
                    B = len(tools)
                    inst_targets = torch.zeros(B, 7, device=self.device)
                    for i, tools_row in enumerate(tools):
                        if isinstance(tools_row, list):
                            tools_row = ', '.join(str(x) for x in tools_row)
                        ids = self.label_mapper.text_to_instrument_ids(tools_row)
                        for idx in ids:
                            if idx < 7:
                                inst_targets[i, idx] = 1.0
                    targets['instrument_label'] = inst_targets
            
            # Action labels
            action_label = _normalize_label_list(batch.get('action_label'), batch_size)
            if action_label is not None and isinstance(action_label, torch.Tensor):
                targets['action_label'] = action_label
            else:
                action = _normalize_label_list(batch.get('action'), batch_size)
                if action is not None and all(a is not None for a in action):
                    targets['action_label'] = torch.tensor(
                        [self.label_mapper.text_to_action_id(a) for a in action],
                        device=self.device
                    )
            
            # Triplet labels
            for comp in ['instrument', 'verb', 'target']:
                key = f'triplet_{comp}_label'
                if key in batch:
                    value = _normalize_label_list(batch[key], batch_size)
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            targets[key] = value
                        else:
                            targets[key] = torch.tensor(value, dtype=torch.long, device=self.device)

            # Triplet labels from raw text fields (triplet_instrument/verb/target).
            # These are emitted by the JSONL dataset from the gpt answer.
            if (
                'triplet_instrument' in batch
                and 'triplet_verb' in batch
                and 'triplet_target' in batch
                and self.label_mapper
            ):
                inst_texts = _normalize_label_list(batch['triplet_instrument'], batch_size)
                verb_texts = _normalize_label_list(batch['triplet_verb'], batch_size)
                tgt_texts = _normalize_label_list(batch['triplet_target'], batch_size)
                if (
                    inst_texts is not None
                    and verb_texts is not None
                    and tgt_texts is not None
                    and not isinstance(inst_texts, torch.Tensor)
                    and not isinstance(verb_texts, torch.Tensor)
                    and not isinstance(tgt_texts, torch.Tensor)
                    and all(x is not None for x in inst_texts)
                    and all(x is not None for x in verb_texts)
                    and all(x is not None for x in tgt_texts)
                ):
                    B = len(inst_texts)
                    inst_ids = torch.zeros(B, dtype=torch.long, device=self.device)
                    verb_ids = torch.zeros(B, dtype=torch.long, device=self.device)
                    tgt_ids = torch.zeros(B, dtype=torch.long, device=self.device)
                    for i in range(B):
                        inst_row = inst_texts[i]
                        verb_row = verb_texts[i]
                        tgt_row = tgt_texts[i]
                        if isinstance(inst_row, (list, tuple)):
                            inst_row = ', '.join(str(x) for x in inst_row)
                        if isinstance(verb_row, (list, tuple)):
                            verb_row = ', '.join(str(x) for x in verb_row)
                        if isinstance(tgt_row, (list, tuple)):
                            tgt_row = ', '.join(str(x) for x in tgt_row)
                        ids = self.label_mapper.text_to_triplet_ids(
                            f"{inst_row}, {verb_row}, {tgt_row}"
                        )
                        inst_ids[i] = ids[0]
                        verb_ids[i] = ids[1]
                        tgt_ids[i] = ids[2]
                    targets['triplet_instrument_label'] = inst_ids
                    targets['triplet_verb_label'] = verb_ids
                    targets['triplet_target_label'] = tgt_ids
            
            # Language labels (next token)
            if 'labels' in batch:
                labels = batch['labels']
                if isinstance(labels, torch.Tensor) and labels.shape[0] == batch_size:
                    targets['language_labels'] = labels
        
        return targets
    
    def _validate(self, val_loaders: Dict[str, DataLoader], global_step: int = None):
        """Run validation on all validation loaders."""
        self.model.eval_mode()
        val_losses = []
        
        with torch.no_grad():
            for name, loader in val_loaders.items():
                loader_losses = []
                dataset_evaluated = 0
                for batch in loader:
                    if dataset_evaluated >= self.max_eval_samples:
                        break
                    micro_batches = batch if isinstance(batch, list) else [batch]
                    for micro in micro_batches:
                        if dataset_evaluated >= self.max_eval_samples:
                            break
                        batch_size = micro['input_ids'].shape[0] if 'input_ids' in micro else len(next(iter(micro.values())))
                        micro = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in micro.items()}
                        loss_dict = self._forward_and_loss(micro)
                        scalar_loss = self._extract_scalar_loss(loss_dict)
                        if scalar_loss is None:
                            scalar_loss = torch.tensor(0.0)
                        loader_losses.append(scalar_loss.item() if isinstance(scalar_loss, torch.Tensor) else float(scalar_loss))
                        dataset_evaluated += batch_size
                
                avg_loader_loss = sum(loader_losses) / len(loader_losses) if loader_losses else 0
                val_losses.append(avg_loader_loss)
                logger.info(f"  Val {name} loss at step {global_step}: {avg_loader_loss:.4f} "
                            f"(evaluated {dataset_evaluated} samples)")
                
                if self.use_wandb:
                    import wandb
                    wandb.log({f"val_{name}_loss": avg_loader_loss, "step": global_step})
        
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        logger.info(f"Validation loss at step {global_step}: {avg_val_loss:.4f}")
        
        if self.use_wandb:
            import wandb
            wandb.log({"val_loss": avg_val_loss, "step": global_step})
        
        self.model.train_mode()
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint with adapter heads."""
        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapter weights only (not full 4-bit model)
        if hasattr(self.model, 'vlm') and self.model.vlm is not None:
            vlm_model = self.model.vlm.model
            # Check if it's a PEFT model with LoRA
            if hasattr(vlm_model, 'peft_config'):
                # Save only the LoRA adapter
                vlm_model.save_pretrained(str(path))
                self.model.vlm.processor.save_pretrained(str(path))
                logger.info(f"Saved LoRA adapter to {path}")
            else:
                # Full model save (fallback)
                vlm_model.save_pretrained(str(path))
                self.model.vlm.processor.save_pretrained(str(path))
        else:
            self.model.save_pretrained(str(path))
        
        # Save adapter heads
        adapter_path = path / "output_adapter.pt"
        torch.save(self.output_adapter.state_dict(), adapter_path)
        
        # Save label mapper if exists
        if self.label_mapper:
            import pickle
            with open(path / "label_mapper.pkl", "wb") as f:
                pickle.dump(self.label_mapper, f)
        
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        logger.info(f"Loading checkpoint from {path}")
        path = Path(path)
        
        # Load base model
        if hasattr(self.model, 'vlm') and self.model.vlm is not None:
            from peft import PeftModel
            self.model.vlm.model = PeftModel.from_pretrained(self.model.vlm.model, str(path))
        
        # Load adapter heads
        adapter_path = path / "output_adapter.pt"
        if adapter_path.exists():
            self.output_adapter.load_state_dict(torch.load(adapter_path, map_location=self.device))
        
        # Load label mapper
        mapper_path = path / "label_mapper.pkl"
        if mapper_path.exists():
            import pickle
            with open(mapper_path, "rb") as f:
                self.label_mapper = pickle.load(f)
        
        logger.info("Checkpoint loaded successfully")


def create_surgical_trainer(
    model: SurgicalVLM,
    config: Dict,
    train_loaders: Dict[str, DataLoader],
    val_loaders: Dict[str, DataLoader],
) -> MultiTaskSurgicalTrainer:
    """Factory function to create fully configured trainer."""
    
    # Create output adapter
    output_adapter = create_output_adapter(config.get("output_adapter", {}))
    output_adapter.to(model.device)
    
    # Create loss
    loss_config = config.get("multitask", {}).get("loss_weights", {})
    criterion = MultiTaskLoss(
        phase_weight=loss_config.get("phase_classification", 1.0),
        instrument_weight=loss_config.get("tool_detection", 1.0),
        action_weight=loss_config.get("action_triplet", 1.5),
        language_weight=loss_config.get("language_modeling", 1.0),
        triplet_weight=loss_config.get("triplet", 1.5),
        grounding_weight=loss_config.get("grounding", 0.5),
    )
    
    # Get trainable parameters (LoRA + adapter heads)
    trainable_params = []
    if hasattr(model, 'vlm') and model.vlm is not None:
        for p in model.vlm.model.parameters():
            if p.requires_grad:
                trainable_params.append(p)
    
    # Add adapter head parameters
    for p in output_adapter.parameters():
        if p.requires_grad:
            trainable_params.append(p)
    
    # Optimizer
    train_config = config.get("training", {})
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_config.get("learning_rate", 2e-5),
        weight_decay=train_config.get("weight_decay", 0.01),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Scheduler
    scheduler = None
    if train_config.get("scheduler", "cosine") == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        steps_per_epoch = sum(len(loader) for loader in train_loaders.values())
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=train_config.get("epochs", 10) * max(1, steps_per_epoch),
            eta_min=train_config.get("min_lr", 1e-6),
        )
    
    # Label mapper
    from surgical_vlm.training.label_mapping import get_label_mapper
    label_mapper = get_label_mapper()
    
    trainer = MultiTaskSurgicalTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=model.device,
        output_adapter=output_adapter,
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        mixed_precision=train_config.get("mixed_precision", "bf16"),
        log_interval=train_config.get("logging_steps", 50),
        eval_interval=train_config.get("eval_steps", 500),
        save_interval=train_config.get("save_steps", 1000),
        max_eval_samples=train_config.get("max_eval_samples", 500),
        output_dir=train_config.get("output_dir", "checkpoints/multitask"),
        use_wandb=train_config.get("use_wandb", False),
        wandb_project=train_config.get("wandb_project", "surgical-vlm-multitask"),
        label_mapper=label_mapper,
        max_steps=train_config.get("max_steps", None),
    )
    
    # Set curriculum config if enabled
    multitask_config = config.get("multitask", {})
    if multitask_config.get("curriculum", {}).get("enabled", False):
        trainer.curriculum_config = multitask_config["curriculum"]
    
    return trainer


# Legacy compatibility
def train_surgical_vlm(
    config: Optional[Dict] = None,
    jsonl_path: Optional[str] = None,
    output_dir: str = "checkpoints",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_samples: Optional[int] = None,
) -> str:
    """Legacy single-task training function."""
    if config is None:
        config = {}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training config: {config}")
    
    # 1. Create model
    logger.info("Creating SurgicalVLM...")
    vlm = create_surgical_vlm(
        model_type="qwen3_vl",
        load_in_4bit=True,
    )
    vlm.load()
    
    # 2. Apply LoRA
    logger.info("Applying LoRA...")
    lora_config_path = "configs/training/lora_config.yaml"
    vlm.apply_lora(lora_config_path)
    
    # Get model and processor for Trainer
    model, processor = vlm.get_model_and_processor()
    
    # 3. Create dataset
    if jsonl_path:
        config["jsonl_path"] = jsonl_path
    if "jsonl_path" not in config:
        config["jsonl_path"] = "data/processed/vlm_jsonl/cholec80_phases_vlm.jsonl"
    
    logger.info(f"Loading dataset from {config['jsonl_path']}...")
    dataset = VLMJSONLDataset(
        jsonl_path=config["jsonl_path"],
        processor=processor,
        max_length=config.get("max_length", 2048),
        max_samples=max_samples,
    )
    
    # Split train/val
    val_size = min(100, len(dataset) // 10)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - val_size, val_size]
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # 4. Collator
    image_root = config.get("data", {}).get("image_root", "data")
    collator = VLMCollator(
        processor=processor,
        max_length=config.get("max_length", 2048),
        image_root=image_root,
    )
    
    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=learning_rate,
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        bf16=config.get("mixed_precision", "bf16") == "bf16",
        fp16=config.get("mixed_precision", "bf16") == "fp16",
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 500),
        eval_steps=config.get("eval_steps", 500),
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
    )
    
    # 6. Loss function
    loss_fn = VLMNextTokenLoss(grounding_weight=config.get("grounding_weight", 1.5))
    
    # Custom trainer that uses our loss
    class SurgicalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_dict = loss_fn(logits, labels, tokenizer=processor.tokenizer)
            loss = loss_dict["loss"]
            return (loss, outputs) if return_outputs else loss
    
    # 7. Train
    trainer = SurgicalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # 8. Save final checkpoint
    final_path = output_dir / "final"
    final_path.mkdir(exist_ok=True)
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))
    
    logger.info(f"Training complete. Model saved to {final_path}")
    return str(final_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SurgicalVLM")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--jsonl", type=str, help="Training JSONL path")
    parser.add_argument("--output", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_samples", type=int, help="Limit samples for debugging")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    config = None
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    
    train_surgical_vlm(
        config=config,
        jsonl_path=args.jsonl,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
    )