"""
Loss functions for VLM training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any


class VLMNextTokenLoss(nn.Module):
    """
    Standard next-token prediction loss for causal LMs.
    Optionally applies higher weight to grounding tokens (digits, brackets).
    """

    def __init__(self, grounding_weight: float = 1.5, ignore_index: int = -100):
        super().__init__()
        self.grounding_weight = grounding_weight
        self.ignore_index = ignore_index

    def _is_grounding_token(self, token_id: int, tokenizer) -> bool:
        """Check if a token is part of a bounding box coordinate."""
        token_str = tokenizer.decode([token_id]).strip()
        return token_str.isdigit() or token_str in ["[", "]", ",", "."]

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tokenizer=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [B, seq_len, vocab_size]
            labels: [B, seq_len] with -100 for ignored positions

        Returns:
            Dict with 'loss', 'ce_loss', 'grounding_loss' (if tokenizer provided)
        """
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Guard: raise if ALL labels are -100 (would produce 0.0 loss, masking the bug)
        active_labels = shift_labels[shift_labels != self.ignore_index]
        if len(active_labels) == 0:
            raise ValueError(
                "FATAL: Attempted to calculate loss on a batch where ALL labels are -100. "
                "This causes 0.0 loss (zero gradients) — the model learns nothing. "
                "Check the data collator for guard rails or ensure some labels are valid token ids."
            )

        # Standard cross-entropy
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction="mean",
        )

        result = {"loss": ce_loss, "ce_loss": ce_loss}

        # Optional: weighted loss for grounding tokens
        if tokenizer is not None and self.grounding_weight > 1.0:
            # Build per-token weight mask
            weight_mask = torch.ones_like(shift_labels, dtype=torch.float)
            for i in range(shift_labels.shape[0]):
                for j in range(shift_labels.shape[1]):
                    tid = shift_labels[i, j].item()
                    if tid != self.ignore_index and self._is_grounding_token(tid, tokenizer):
                        weight_mask[i, j] = self.grounding_weight

            # Weighted cross-entropy (manual)
            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.ignore_index,
                reduction="none",
            )
            per_token_loss = per_token_loss.view(shift_labels.shape)
            weighted_loss = (per_token_loss * weight_mask).sum() / weight_mask.sum().clamp(min=1)
            result["loss"] = weighted_loss
            result["grounding_loss"] = weighted_loss

        return result


def _assert_not_fully_gated(target: torch.Tensor, task: str, dataset_name: str, ignore_index: int = -100) -> None:
    """Raise if every target element equals the ignore index (fully-gated batch)."""
    if torch.all(target == ignore_index):
        raise ValueError(
            f"{task} targets for dataset '{dataset_name}' are ALL {ignore_index} "
            "(fully-gated / no labels). This is a data or collation bug - refusing "
            "to train on a batch with no signal."
        )


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task surgical video understanding.
    
    Tasks:
    - Phase classification: CrossEntropyLoss
    - Instrument detection: BCEWithLogitsLoss (multi-label)
    - Action classification: CrossEntropyLoss
    - Triplet recognition: CrossEntropyLoss (3 components)
    - Language modeling: CrossEntropyLoss (next token)
    - Grounding: L1Loss (bbox regression)
    """
    
    def __init__(
        self,
        phase_weight: float = 1.0,
        instrument_weight: float = 1.0,
        action_weight: float = 1.0,
        triplet_weight: float = 1.5,
        language_weight: float = 1.0,
        grounding_weight: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.phase_weight = phase_weight
        self.instrument_weight = instrument_weight
        self.action_weight = action_weight
        self.triplet_weight = triplet_weight
        self.language_weight = language_weight
        self.grounding_weight = grounding_weight
        
        # Loss functions
        self.phase_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-100)
        self.instrument_loss = nn.BCEWithLogitsLoss()
        self.action_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-100)
        self.triplet_inst_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.triplet_verb_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.triplet_target_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.language_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.grounding_loss = nn.L1Loss()
        
    def forward(
        self,
        outputs: "TrainingOutputs",
        targets: Dict[str, torch.Tensor],
        dataset_name: str = "unknown",
        tokenizer=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            outputs: TrainingOutputs from OutputAdapter
            targets: Dict with keys:
                - phase_label: [B] or [B, T] class indices
                - instrument_label: [B, 7] or [B, T, 7] binary
                - action_label: [B] or [B, T] class indices
                - triplet_inst_label: [B] or [B, T] class indices
                - triplet_verb_label: [B] or [B, T] class indices
                - triplet_target_label: [B] or [B, T] class indices
                - language_labels: [B, Seq] token ids
                - grounding_label: [B, 4] or [B, T, 4] normalized coords
            dataset_name: Name of dataset (e.g., "cholec80", "heichole")
            tokenizer: Optional tokenizer for VLMNextTokenLoss
        
        Returns:
            Dict with individual losses and total loss
        """
        from surgical_vlm.training.output_adapter import TrainingOutputs
        
        losses = {}
        total_loss = 0.0
        
        # Phase loss
        if outputs.phase_logits is not None and 'phase_label' in targets:
            phase_logits = outputs.phase_logits
            phase_target = targets['phase_label']
            
            # Handle sequence vs single
            if phase_logits.dim() == 3 and phase_target.dim() == 2:
                # [B, T, C] vs [B, T] -> flatten
                B, T, C = phase_logits.shape
                phase_logits = phase_logits.view(B * T, C)
                phase_target = phase_target.view(B * T)
            
            _assert_not_fully_gated(phase_target, "Phase", dataset_name)
            loss = self.phase_loss(phase_logits, phase_target)
            losses['phase_loss'] = loss
            total_loss += self.phase_weight * loss
        
        # Instrument loss (multi-label)
        if outputs.instrument_logits is not None and 'instrument_label' in targets:
            inst_logits = outputs.instrument_logits
            inst_target = targets['instrument_label']
            
            if inst_logits.dim() == 3 and inst_target.dim() == 3:
                B, T, C = inst_logits.shape
                inst_logits = inst_logits.view(B * T, C)
                inst_target = inst_target.view(B * T, C)
            
            loss = self.instrument_loss(inst_logits, inst_target.float())
            losses['instrument_loss'] = loss
            total_loss += self.instrument_weight * loss
        
        # Action loss
        if outputs.action_logits is not None and 'action_label' in targets:
            action_logits = outputs.action_logits
            action_target = targets['action_label']
            
            if action_logits.dim() == 3 and action_target.dim() == 2:
                B, T, C = action_logits.shape
                action_logits = action_logits.view(B * T, C)
                action_target = action_target.view(B * T)
            
            _assert_not_fully_gated(action_target, "Action", dataset_name)
            loss = self.action_loss(action_logits, action_target)
            losses['action_loss'] = loss
            total_loss += self.action_weight * loss
        
        # Triplet loss
        if outputs.triplet_logits is not None:
            triplet_losses = 0.0
            triplet_count = 0
            
            for comp_name, comp_logits in outputs.triplet_logits.items():
                target_key = f'triplet_{comp_name}_label'
                if target_key in targets:
                    comp_target = targets[target_key]
                    
                    if comp_logits.dim() == 3 and comp_target.dim() == 2:
                        B, T, C = comp_logits.shape
                        comp_logits = comp_logits.view(B * T, C)
                        comp_target = comp_target.view(B * T)
                    
                    _assert_not_fully_gated(comp_target, f"Triplet {comp_name}", dataset_name)
                    if comp_name == 'instrument':
                        loss = self.triplet_inst_loss(comp_logits, comp_target)
                    elif comp_name == 'verb':
                        loss = self.triplet_verb_loss(comp_logits, comp_target)
                    else:  # target
                        loss = self.triplet_target_loss(comp_logits, comp_target)
                    
                    losses[f'triplet_{comp_name}_loss'] = loss
                    triplet_losses += loss
                    triplet_count += 1
            
            if triplet_count > 0:
                losses['triplet_loss'] = triplet_losses / triplet_count
                total_loss += self.triplet_weight * losses['triplet_loss']
        
        # Language loss
        if outputs.language_logits is not None and 'language_labels' in targets:
            lang_logits = outputs.language_logits  # [B, Seq, V]
            lang_target = targets['language_labels']  # [B, Seq]
            
            # Shift for next-token prediction
            shift_logits = lang_logits[:, :-1, :].contiguous()
            shift_target = lang_target[:, 1:].contiguous()
            
            _assert_not_fully_gated(shift_target, "Language", dataset_name)
            loss = self.language_loss(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_target.view(-1)
            )
            losses['language_loss'] = loss
            total_loss += self.language_weight * loss
        
        # Grounding loss - no dataset provides bbox annotations
        if outputs.grounding_logits is not None:
            # Grounding is not computed for any dataset
            losses['grounding_loss'] = torch.tensor(0.0, device=outputs.grounding_logits.device)
        
        if not isinstance(total_loss, torch.Tensor):
            raise ValueError(
                f"No active loss terms for dataset '{dataset_name}' - every task was "
                "fully gated (all targets -100) or had no logits. MultiTaskLoss "
                "refuses to return a silent zero loss."
            )
        
        losses['total_loss'] = total_loss
        return losses


def create_multitask_loss(config: Dict) -> MultiTaskLoss:
    """Factory function for MultiTaskLoss."""
    return MultiTaskLoss(
        phase_weight=config.get("phase_weight", 1.0),
        instrument_weight=config.get("instrument_weight", 1.0),
        action_weight=config.get("action_weight", 1.0),
        triplet_weight=config.get("triplet_weight", 1.5),
        language_weight=config.get("language_weight", 1.0),
        grounding_weight=config.get("grounding_weight", 0.5),
        label_smoothing=config.get("label_smoothing", 0.1),
    )