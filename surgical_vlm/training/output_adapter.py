"""
Output Adapter Module
Handles the duality between TRAINING MODE (logits) and INFERENCE MODE (JSON text).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import json
import re


@dataclass
class TrainingOutputs:
    """Structured outputs for training loss computation."""
    phase_logits: Optional[torch.Tensor] = None          # [B, 8]
    instrument_logits: Optional[torch.Tensor] = None      # [B, 7]
    action_logits: Optional[torch.Tensor] = None          # [B, 10]
    triplet_logits: Optional[Dict[str, torch.Tensor]] = None  # {'inst': [B,6], 'verb': [B,10], 'target': [B,15]}
    language_logits: Optional[torch.Tensor] = None        # [B, Seq, Vocab]
    grounding_logits: Optional[torch.Tensor] = None       # [B, 4] for bbox
    hidden_states: Optional[torch.Tensor] = None          # [B, Seq, Hidden] for temporal


@dataclass
class InferenceOutputs:
    """Structured outputs for inference/demo."""
    phase: str = "Unknown"
    tools: List[str] = None
    description: str = ""
    confidence: float = 0.0
    temporal_consistency: bool = False


class OutputAdapter(nn.Module):
    """
    Adapter that converts between VLM internal representations and task-specific outputs.
    
    TRAINING MODE:
    - Takes hidden states from VLM
    - Applies task-specific heads
    - Returns logits for loss computation
    
    INFERENCE MODE:
    - Takes VLM generated text
    - Parses into structured JSON
    - Returns InferenceOutputs
    """
    
    def __init__(
        self,
        hidden_dim: int = 3584,
        num_phases: int = 8,
        num_instruments: int = 7,
        num_actions: int = 10,
        triplet_instruments: int = 6,
        triplet_verbs: int = 10,
        triplet_targets: int = 15,
        vocab_size: int = 152064,  # Qwen vocab
        use_temporal: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_temporal = use_temporal
        
        # Task-specific classification heads (for training)
        self.phase_head = nn.Linear(hidden_dim, num_phases)
        self.instrument_head = nn.Linear(hidden_dim, num_instruments)
        self.action_head = nn.Linear(hidden_dim, num_actions)
        
        # Triplet heads
        self.triplet_inst_head = nn.Linear(hidden_dim, triplet_instruments)
        self.triplet_verb_head = nn.Linear(hidden_dim, triplet_verbs)
        self.triplet_target_head = nn.Linear(hidden_dim, triplet_targets)
        
        # Grounding head (bbox regression)
        self.grounding_head = nn.Linear(hidden_dim, 4)  # x1, y1, x2, y2 normalized
        
        # Temporal pooling for sequence-level predictions
        if use_temporal:
            self.temporal_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),  # Attention weight per frame
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.phase_head, self.instrument_head, self.action_head,
                       self.triplet_inst_head, self.triplet_verb_head, self.triplet_target_head,
                       self.grounding_head]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward_training(
        self,
        hidden_states: torch.Tensor,      # [B, T, D] or [B, D] 
        temporal_mask: Optional[torch.Tensor] = None,  # [B, T] bool
        return_per_frame: bool = False,
    ) -> TrainingOutputs:
        """
        Training forward pass.
        
        Args:
            hidden_states: From VLM encoder + temporal aggregator
                          [B, T, D] for sequences or [B, D] for single frames
            temporal_mask: Valid frame mask [B, T]
            return_per_frame: If True, return logits per frame [B, T, ...]
        
        Returns:
            TrainingOutputs with logits for each task
        """
        B = hidden_states.shape[0]
        
        if hidden_states.dim() == 3:
            # Sequence input [B, T, D]
            T = hidden_states.shape[1]
            
            if self.use_temporal and temporal_mask is not None:
                # Attention pooling over time
                attn_weights = self.temporal_pool(hidden_states).squeeze(-1)  # [B, T]
                attn_weights = attn_weights.masked_fill(~temporal_mask, -1e9)
                attn_weights = F.softmax(attn_weights, dim=1)  # [B, T]
                pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
            else:
                # Simple mean pooling
                if temporal_mask is not None:
                    valid = temporal_mask.float().unsqueeze(-1)
                    pooled = (hidden_states * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
                else:
                    pooled = hidden_states.mean(dim=1)
        else:
            # Single frame [B, D]
            pooled = hidden_states
            return_per_frame = False
        
        # Apply heads to pooled representation
        outputs = TrainingOutputs(
            phase_logits=self.phase_head(pooled),
            instrument_logits=self.instrument_head(pooled),
            action_logits=self.action_head(pooled),
            triplet_logits={
                'instrument': self.triplet_inst_head(pooled),
                'verb': self.triplet_verb_head(pooled),
                'target': self.triplet_target_head(pooled),
            },
            grounding_logits=self.grounding_head(pooled),
            hidden_states=pooled,
        )
        
        if return_per_frame and hidden_states.dim() == 3:
            # Also return per-frame logits for dense supervision
            outputs.phase_logits = self.phase_head(hidden_states)
            outputs.instrument_logits = self.instrument_head(hidden_states)
            outputs.action_logits = self.action_head(hidden_states)
            outputs.triplet_logits = {
                'instrument': self.triplet_inst_head(hidden_states),
                'verb': self.triplet_verb_head(hidden_states),
                'target': self.triplet_target_head(hidden_states),
            }
            outputs.grounding_logits = self.grounding_head(hidden_states)
        
        return outputs
    
    def forward_inference(self, generated_text: str) -> InferenceOutputs:
        """
        Parse VLM generated text into structured inference output.
        
        Args:
            generated_text: Raw text from VLM.generate()
        
        Returns:
            InferenceOutputs with parsed phase, tools, description
        """
        # Try to parse as JSON first
        parsed = self._extract_json(generated_text)
        
        if parsed and isinstance(parsed, dict):
            return InferenceOutputs(
                phase=parsed.get("phase", "Unknown"),
                tools=parsed.get("tools", []) if isinstance(parsed.get("tools"), list) else [],
                description=parsed.get("description", "") if isinstance(parsed.get("description"), str) else "",
                confidence=parsed.get("confidence", 0.0),
            )
        
        # Fallback: keyword-based parsing
        return self._keyword_parse(generated_text)
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from text, handling common formatting issues."""
        text = text.strip()
        
        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Find JSON in code blocks
        import re
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass
        
        # Find first {...}
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _keyword_parse(self, text: str) -> InferenceOutputs:
        """Fallback keyword-based parsing for unstructured text."""
        text_lower = text.lower()
        
        # Phase detection
        phase_keywords = {
            "Preparation": ["preparation", "preparing", "setup", "port placement"],
            "CalotTriangleDissection": ["calot", "triangle", "dissection"],
            "ClippingCutting": ["clipping", "cutting", "clip", "cut"],
            "GallbladderDissection": ["gallbladder dissection", "gb dissection"],
            "GallbladderPackaging": ["packaging", "packing", "specimen bag"],
            "CleaningCoagulation": ["cleaning", "coagulation", "coagulating"],
            "GallbladderRetraction": ["retraction", "retracting"],
        }
        
        phase = "Unknown"
        for p, keywords in phase_keywords.items():
            if any(kw in text_lower for kw in keywords):
                phase = p
                break
        
        # Tool detection
        tool_keywords = {
            "Grasper": ["grasper", "grasping", "forceps"],
            "Bipolar": ["bipolar"],
            "Hook": ["hook", "cautery hook"],
            "Scissors": ["scissors", "shears"],
            "Clipper": ["clipper", "clip applier", "hemoclip"],
            "Irrigator": ["irrigator", "irrigation", "suction"],
            "SpecimenBag": ["specimen bag", "endobag", "bag"],
        }
        
        tools = []
        for tool, keywords in tool_keywords.items():
            if any(kw in text_lower for kw in keywords):
                tools.append(tool)
        
        return InferenceOutputs(
            phase=phase,
            tools=tools,
            description=text.strip(),
            confidence=0.5,
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
    - Grounding: L1Loss (bbox regression) + GIoU
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
        self.phase_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.instrument_loss = nn.BCEWithLogitsLoss()
        self.action_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.triplet_inst_loss = nn.CrossEntropyLoss()
        self.triplet_verb_loss = nn.CrossEntropyLoss()
        self.triplet_target_loss = nn.CrossEntropyLoss()
        self.language_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.grounding_loss = nn.L1Loss()
        
    def forward(
        self,
        outputs: TrainingOutputs,
        targets: Dict[str, torch.Tensor],
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
        
        Returns:
            Dict with individual losses and total loss
        """
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
            
            loss = self.language_loss(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_target.view(-1)
            )
            losses['language_loss'] = loss
            total_loss += self.language_weight * loss
        
        # Grounding loss
        if outputs.grounding_logits is not None and 'grounding_label' in targets:
            bbox_pred = outputs.grounding_logits
            bbox_target = targets['grounding_label']
            
            if bbox_pred.dim() == 3 and bbox_target.dim() == 3:
                B, T, _ = bbox_pred.shape
                bbox_pred = bbox_pred.view(B * T, 4)
                bbox_target = bbox_target.view(B * T, 4)
            
            loss = self.grounding_loss(bbox_pred, bbox_target)
            losses['grounding_loss'] = loss
            total_loss += self.grounding_weight * loss
        
        losses['total_loss'] = total_loss
        return losses


def create_output_adapter(config: Dict) -> OutputAdapter:
    """Factory function for OutputAdapter."""
    return OutputAdapter(
        hidden_dim=config.get("hidden_dim", 3584),
        num_phases=config.get("num_phases", 8),
        num_instruments=config.get("num_instruments", 7),
        num_actions=config.get("num_actions", 10),
        triplet_instruments=config.get("triplet_instruments", 6),
        triplet_verbs=config.get("triplet_verbs", 10),
        triplet_targets=config.get("triplet_targets", 15),
        vocab_size=config.get("vocab_size", 152064),
        use_temporal=config.get("use_temporal", True),
    )


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


if __name__ == "__main__":
    # Test
    adapter = OutputAdapter()
    loss_fn = MultiTaskLoss()
    
    # Simulate training forward
    B, T, D = 2, 8, 3584
    hidden = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)
    
    outputs = adapter.forward_training(hidden, mask)
    print("Training outputs:")
    for k, v in outputs.__dict__.items():
        if v is not None:
            print(f"  {k}: {v.shape}")
    
    # Simulate loss
    targets = {
        'phase_label': torch.randint(0, 8, (B, T)),
        'instrument_label': torch.randint(0, 2, (B, T, 7)).float(),
        'action_label': torch.randint(0, 10, (B, T)),
        'triplet_instrument_label': torch.randint(0, 6, (B, T)),
        'triplet_verb_label': torch.randint(0, 10, (B, T)),
        'triplet_target_label': torch.randint(0, 15, (B, T)),
        'grounding_label': torch.rand(B, T, 4),
    }
    
    losses = loss_fn(outputs, targets)
    print("\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")