"""
Applies LoRA to any base_vlm using PEFT.
Model-agnostic: works for Qwen3, InternVL, Molmo2.
Reads configs/training/lora_config.yaml.
"""

from typing import Tuple

import yaml
from peft import LoraConfig, get_peft_model, PeftModel


def apply_lora_to_model(model, processor, config_path: str = "configs/training/lora_config.yaml") -> Tuple:
    """
    Apply LoRA adapters to a VLM model.

    Args:
        model: The HuggingFace model
        processor: The associated processor (returned unchanged)
        config_path: Path to lora_config.yaml

    Returns:
        Tuple of (peft_model, processor)
    """
    with open(config_path, "r") as f:
        lora_cfg = yaml.safe_load(f)

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        modules_to_save=lora_cfg.get("modules_to_save"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    return peft_model, processor


def load_lora_checkpoint(model, checkpoint_path: str):
    """
    Load a saved LoRA checkpoint into a model.
    
    Args:
        model: The base model (with LoRA already applied via get_peft_model)
        checkpoint_path: Path to the saved LoRA adapter directory
        
    Returns:
        Model with loaded LoRA weights
    """
    model = PeftModel.from_pretrained(model, checkpoint_path)
    return model