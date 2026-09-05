"""
SurgicalVLM - Main wrapper for surgical video analysis.

Unifies Qwen2.5-VL-7B with surgical-specific prompting for 3-key JSON output:
phase, tools, description.
"""

import logging
import json
from collections import Counter
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .qwen3_vl import Qwen3VL
from .base_vlm import BaseVLM
from .temporal_aggregator import TemporalAttentionAggregator, TemporalAggregator

logger = logging.getLogger(__name__)


SURGICAL_JSON_PROMPT = """Analyze this surgical frame and output a JSON object with exactly these 3 keys:
{
  "phase": "Surgical phase name (e.g., Preparation, CalotTriangleDissection, ClippingCutting, GallbladderDissection, GallbladderPackaging, CleaningCoagulation, GallbladderRetraction, Unknown)",
  "tools": ["List of visible surgical tools from: Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator, SpecimenBag"],
  "description": "Concise natural language description of the surgical scene (max 2 sentences)"
}

Rules:
- Output ONLY valid JSON, no extra text
- If uncertain, use "Unknown" for phase, empty list for tools
- Tool names must match the provided list exactly
- Description must be factual and clinical"""


SURGICAL_MULTI_FRAME_PROMPT = """Analyze this sequence of surgical frames and output a JSON object with exactly these 3 keys:
{
  "phase": "Current surgical phase",
  "tools": ["Visible tools in current frame"],
  "description": "What is happening across these frames (2-3 sentences)"
}

Rules:
- Output ONLY valid JSON
- Phase must be one of: Preparation, CalotTriangleDissection, ClippingCutting, GallbladderDissection, GallbladderPackaging, CleaningCoagulation, GallbladderRetraction, Unknown
- Tools from: Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator, SpecimenBag"""


VALID_PHASES = [
    "Preparation",
    "CalotTriangleDissection", 
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
    "Unknown"
]

VALID_TOOLS = [
    "Grasper",
    "Bipolar",
    "Hook", 
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag"
]


class SurgicalVLM(nn.Module):
    """
    High-level surgical video analysis interface.
    
    Wraps Qwen2.5-VL-7B with surgical-specific prompting to produce
    structured JSON output for each frame: phase, tools, description.
    """
    
    def __init__(
        self,
        model_type: str = "qwen3_vl",
        load_in_4bit: bool = True,
        device: str = "auto",
        use_temporal: bool = False,
        temporal_config: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        model_class: Optional[str] = None,
        processor_class: Optional[str] = None,
        max_pixels: int = 12845056,
        **kwargs
    ):
        super().__init__()
        self.model_type = model_type
        self.load_in_4bit = load_in_4bit
        self.max_pixels = max_pixels
        
        # Store model-specific config for generic loaders
        self.model_name = model_name
        self.model_class = model_class
        self.processor_class = processor_class
        
        # Auto-detect device
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.use_temporal = use_temporal
        self.vlm: Optional[BaseVLM] = None
        self.lora_applied = False
        
        # Temporal aggregator for multi-frame reasoning
        if use_temporal:
            default_temporal_config = {
                "hidden_dim": 3584,  # Qwen2.5-VL-7B hidden size
                "num_heads": 8,
                "num_layers": 4,
                "dropout": 0.1,
                "max_frames": 256,
                "use_pos_encoding": True,
            }
            if temporal_config:
                default_temporal_config.update(temporal_config)
            default_temporal_config.pop("num_frames", None)
            
            self.temporal_aggregator = TemporalAttentionAggregator(**default_temporal_config)
        else:
            self.temporal_aggregator = None
        
        logger.info(f"SurgicalVLM initialized (temporal: {use_temporal})")
    
    def load(self) -> None:
        """Load the underlying VLM model."""
        if self.model_type == "qwen3_vl":
            self.vlm = Qwen3VL(load_in_4bit=self.load_in_4bit, max_pixels=self.max_pixels)
            self.vlm.load()
            self.vlm.to(self.device)
        elif self.model_type == "qwen2.5_vl_3b":
            # Qwen2.5-VL-3B-Instruct (smaller, CPU-friendly)
            self.vlm = Qwen3VL(load_in_4bit=self.load_in_4bit, model_name="Qwen/Qwen2.5-VL-3B-Instruct", max_pixels=self.max_pixels)
            self.vlm.load()
            self.vlm.to(self.device)
        elif self.model_type == "qwen2_vl":
            # Qwen2-VL (non-3 series)
            self.vlm = Qwen3VL(load_in_4bit=self.load_in_4bit, model_name="Qwen/Qwen2-VL-7B-Instruct", max_pixels=self.max_pixels)
            self.vlm.load()
            self.vlm.to(self.device)
        elif self.model_type == "llava":
            # LLaVA family - load generic
            self.vlm = self._load_generic_hf_vlm(
                model_name="llava-hf/llava-1.5-7b-hf",
                model_class="LlavaForConditionalGeneration",
                processor_class="LlavaProcessor",
            )
        elif self.model_type == "llava_next":
            # LLaVA-Next (LLaVA-1.6)
            self.vlm = self._load_generic_hf_vlm(
                model_name="llava-hf/llava-v1.6-mistral-7b-hf",
                model_class="LlavaNextForConditionalGeneration",
                processor_class="LlavaNextProcessor",
            )
        elif self.model_type == "blip2":
            # BLIP-2
            self.vlm = self._load_generic_hf_vlm(
                model_name="Salesforce/blip2-opt-2.7b",
                model_class="Blip2ForConditionalGeneration",
                processor_class="Blip2Processor",
            )
        elif self.model_type == "instructblip":
            # InstructBLIP
            self.vlm = self._load_generic_hf_vlm(
                model_name="Salesforce/instructblip-vicuna-7b",
                model_class="InstructBlipForConditionalGeneration",
                processor_class="InstructBlipProcessor",
            )
        elif self.model_type == "generic":
            # Generic HF VLM - user provides model name via model_name kwarg
            model_name = getattr(self, 'model_name', None) or "Qwen/Qwen2.5-VL-7B-Instruct"
            self.vlm = self._load_generic_hf_vlm(
                model_name=model_name,
                model_class=getattr(self, 'model_class', None),
                processor_class=getattr(self, 'processor_class', None),
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # If temporal aggregator is used, ensure it's on the right device
        if self.use_temporal and self.temporal_aggregator is not None:
            self.temporal_aggregator.to(self.device)

        logger.info(f"SurgicalVLM loaded: {self.model_type} (temporal: {self.use_temporal})")

    def parameters(self, recurse: bool = True):
        """Delegate parameters to the underlying VLM model for optimizer."""
        if self.vlm is not None and hasattr(self.vlm, 'model') and self.vlm.model is not None:
            return self.vlm.model.parameters(recurse=recurse)
        return iter([])

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Delegate named_parameters to the underlying VLM model."""
        if self.vlm is not None and hasattr(self.vlm, 'model') and self.vlm.model is not None:
            return self.vlm.model.named_parameters(prefix=prefix, recurse=recurse)
        return iter([])

    def _load_generic_hf_vlm(self, model_name: str, model_class: str = None, processor_class: str = None) -> BaseVLM:
        """Load a generic Hugging Face VLM model."""
        from transformers import AutoModelForVision2Seq, AutoProcessor, AutoModel
        
        logger.info(f"Loading generic HF VLM: {model_name}")
        
        # Try to load with Auto classes first
        try:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.load_in_4bit else torch.float16,
                device_map="auto",
                load_in_4bit=self.load_in_4bit,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"AutoModelForVision2Seq failed, trying AutoModel: {e}")
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.load_in_4bit else torch.float16,
                device_map="auto",
                load_in_4bit=self.load_in_4bit,
                trust_remote_code=True,
            )
        
        model.eval()
        
        # Create a simple wrapper
        class GenericVLMWrapper(BaseVLM):
            def __init__(self, model, processor, device):
                super().__init__({})
                self.model = model
                self.processor = processor
                self.device = device
            
            def generate(self, image, prompt, **kwargs):
                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, **kwargs)
                return self.processor.decode(output_ids[0], skip_special_tokens=True)
            
            def to(self, device):
                self.device = device
                return self
            
            def eval_mode(self):
                self.model.eval()
            
            def get_model_and_processor(self):
                return self.model, self.processor
        
        return GenericVLMWrapper(model, processor, self.device)
    
    def apply_lora(self, config_path: str = "configs/training/lora_config.yaml") -> None:
        """Apply LoRA adapter for fine-tuning."""
        if self.vlm is None:
            raise RuntimeError("Must call load() before apply_lora()")
        
        import yaml
        from peft import LoraConfig, get_peft_model
        
        with open(config_path, "r") as f:
            lora_config = yaml.safe_load(f)
        
        peft_config = LoraConfig(**lora_config)
        self.vlm.model = get_peft_model(self.vlm.model, peft_config)
        self.lora_applied = True
        
        # Print trainable parameters
        trainable = sum(p.numel() for p in self.vlm.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.vlm.model.parameters())
        logger.info(f"LoRA applied. Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def load_lora_checkpoint(self, checkpoint_path: str) -> None:
        """Load a trained LoRA checkpoint."""
        if self.vlm is None:
            raise RuntimeError("Must call load() before load_lora_checkpoint()")
        
        from peft import PeftModel
        
        self.vlm.model = PeftModel.from_pretrained(self.vlm.model, checkpoint_path, is_trainable=False)
        self.vlm.model = self.vlm.model.merge_and_unload()
        self.vlm.model.eval()
        logger.info(f"Loaded LoRA checkpoint from {checkpoint_path}")
    
    def get_model_and_processor(self) -> Tuple:
        """Get underlying model and processor for Trainer."""
        if self.vlm is None:
            raise RuntimeError("Must call load() first")
        return self.vlm.get_model_and_processor()
    
    def to(self, device: str) -> "SurgicalVLM":
        self.device = device
        if self.vlm is not None:
            self.vlm.to(device)
        return self
    
    def eval_mode(self) -> None:
        if self.vlm is not None:
            self.vlm.eval_mode()
    
    def train_mode(self) -> None:
        if self.vlm is not None:
            self.vlm.train_mode()
    
    def _validate_output(self, output: Dict) -> Dict:
        """Validate and sanitize model output."""
        # Ensure all required keys exist
        validated = {
            "phase": output.get("phase", "Unknown"),
            "tools": output.get("tools", []),
            "description": output.get("narrative", output.get("description", "")),
        }
        
        # Validate phase
        if validated["phase"] not in VALID_PHASES:
            validated["phase"] = "Unknown"

        # Validate tools — fuzzy-match each name against the trained vocabulary
        # (e.g. "bipolar forceps" -> Bipolar, "hemostat" -> Bipolar) instead of
        # requiring an exact string match, which dropped almost every real
        # tool the model named since it rarely uses the exact canonical name.
        from surgical_vlm.training.label_mapping import get_label_mapper
        label_mapper = get_label_mapper()
        matched_names = []
        for t in validated["tools"]:
            for tool_id in label_mapper.text_to_instrument_ids(str(t)):
                name = label_mapper.id_to_instrument.get(tool_id)
                if name and name not in matched_names:
                    matched_names.append(name)
        validated["tools"] = matched_names
        
        # Ensure description is string
        if not isinstance(validated["description"], str):
            validated["description"] = str(validated["description"])
        
        return validated
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from model output, handling common formatting issues."""
        text = text.strip()
        
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in code blocks
        import re
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find first {...} 
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    def analyze_frame(self, image: Image.Image, multi_frame: bool = False, prompt: str = None) -> Dict:
        """
        Analyze a single surgical frame.
        
        Args:
            image: PIL Image of surgical frame
            multi_frame: Whether to use multi-frame prompt (for sequences)
            prompt: Custom prompt to use (overrides default prompts)
            
        Returns:
            Dict with keys: phase, tools, description
        """
        if self.vlm is None:
            raise RuntimeError("Must call load() before analyze_frame()")
        
        if prompt is None:
            prompt = SURGICAL_MULTI_FRAME_PROMPT if multi_frame else SURGICAL_JSON_PROMPT
        
        # Generate
        response = self.vlm.generate(
            image=image,
            prompt=prompt,
            max_new_tokens=256,
            temperature=0.1,
        )
        
        # Extract and validate JSON
        parsed = self._extract_json(response)
        if parsed is None:
            logger.warning(f"Failed to parse JSON from response: {response[:200]}")
            return {
                "phase": "Unknown",
                "tools": [],
                "description": "Failed to parse model output"
            }
        
        return self._validate_output(parsed)

    def generate_single(self, image: Image.Image, prompt: str = None, max_new_tokens: int = 256, temperature: float = 0.1) -> Dict:
        """
        Generate analysis for a single frame (alias for analyze_frame for UI compatibility).
        
        Args:
            image: PIL Image of surgical frame
            prompt: Custom prompt to use
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dict with keys: phase, tools, description
        """
        if self.vlm is None:
            raise RuntimeError("Must call load() before generate_single()")
        
        if prompt is None:
            prompt = SURGICAL_JSON_PROMPT
        
        # Generate
        response = self.vlm.generate(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        # Extract and validate JSON
        parsed = self._extract_json(response)
        if parsed is None:
            logger.warning(f"Failed to parse JSON from response: {response[:200]}")
            return {
                "phase": "Unknown",
                "tools": [],
                "description": "Failed to parse model output"
            }
        
        return self._validate_output(parsed)

    def analyze_frames(self, images: List[Image.Image]) -> Dict:
        """
        Analyze multiple frames together for temporal context.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Dict with keys: phase, tools, description
        """
        if self.vlm is None:
            raise RuntimeError("Must call load() before analyze_frames()")
        
        # Build multi-image prompt
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images] + 
                       [{"type": "text", "text": SURGICAL_MULTI_FRAME_PROMPT}]
        }]
        
        # Apply chat template
        from qwen_vl_utils import process_vision_info
        
        text = self.vlm.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        
        inputs = self.vlm.processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.vlm.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.vlm.processor.tokenizer.eos_token_id,
            )
        
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        response = self.vlm.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        parsed = self._extract_json(response)
        if parsed is None:
            return {
                "phase": "Unknown",
                "tools": [],
                "description": "Failed to parse model output"
            }
        
        return self._validate_output(parsed)
    
    def analyze_temporal(self, frames: List[Image.Image], prompt: str) -> Dict:
        """
        Analyze multiple frames with temporal reasoning.
        
        Args:
            frames: List of PIL Images (sequence of frames)
            prompt: Text prompt for the analysis
            
        Returns:
            Dict with keys:
                - predictions: Aggregated text dict with phase, tools, description
                - temporal_attention_weights: Attention weights from temporal aggregator [B, T, T] or None
        """
        if self.vlm is None:
            raise RuntimeError("Must call load() before analyze_temporal()")
        
        if not self.use_temporal or self.temporal_aggregator is None:
            logger.warning("Temporal aggregator not enabled, falling back to single-frame analysis")
            # Fallback: analyze each frame independently and aggregate text results
            return self._analyze_temporal_fallback(frames, prompt)
        
        # Step 1: Extract vision features for each frame
        vision_features = []
        per_frame_outputs = []
        
        from qwen_vl_utils import process_vision_info
        
        for frame in frames:
            # Format frame for Qwen-VL processor
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": frame},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            text = self.vlm.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            
            inputs = self.vlm.processor(
                text=[text], images=image_inputs, padding=True, return_tensors="pt"
            ).to(self.device)
            
            # Extract vision encoder features (before language model)
            with torch.no_grad():
                # Get vision encoder outputs
                vision_outputs = self.vlm.model.model.visual(
                    inputs["pixel_values"],
                    grid_thw=inputs.get("image_grid_thw")
                )
                # vision_outputs shape: [num_patches, hidden_dim]
                # We'll use the mean-pooled features
                if vision_outputs.dim() == 3:
                    frame_feat = vision_outputs.mean(dim=1)  # [1, hidden]
                else:
                    frame_feat = vision_outputs.mean(dim=0, keepdim=True)  # [1, hidden]
                vision_features.append(frame_feat)
                
                # Also generate text output for this frame (for fallback aggregation)
                with torch.no_grad():
                    output_ids = self.vlm.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.vlm.processor.tokenizer.eos_token_id,
                    )
                    generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
                    response = self.vlm.processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0]
                    parsed = self._extract_json(response)
                    if parsed:
                        per_frame_outputs.append(self._validate_output(parsed))
                    else:
                        per_frame_outputs.append({
                            "phase": "Unknown",
                            "tools": [],
                            "description": "Failed to parse"
                        })
        
        # Stack features: [B, T, D]
        if vision_features:
            stacked_features = torch.stack(vision_features, dim=1)  # [1, T, D]
        else:
            # No features extracted
            return self._analyze_temporal_fallback(frames, prompt)
        
        # Step 2: Pass through temporal aggregator
        # temporal_aggregator returns enhanced features or pooled representation
        temporal_output = self.temporal_aggregator(
            stacked_features,
            mask=None,  # No padding mask for now
            pool=True  # Get pooled representation
        )  # [1, D]
        
        # Get attention weights for visualization
        attention_weights = None
        if hasattr(self.temporal_aggregator, 'get_attention_weights'):
            attention_weights = self.temporal_aggregator.get_attention_weights(
                stacked_features
            )
        
        # Step 3: Use the pooled temporal features to generate final prediction
        # For now, we'll use the per-frame text outputs and aggregate them
        # (Since we can't easily inject the temporal features back into the LM)
        aggregated = self._aggregate_text_outputs(per_frame_outputs)
        
        return {
            "predictions": aggregated,
            "temporal_attention_weights": attention_weights,
            "per_frame_outputs": per_frame_outputs,
        }
    
    # ==================== TRAINING/EVAL PREDICTION METHODS ====================
    # These methods return LOGITS for loss computation, not text.
    # They are used during training and evaluation only.
    
    def _get_vision_features(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extract vision encoder features from images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Tensor of shape [B, num_patches, hidden_dim] or [B, hidden_dim] if pooled
        """
        if self.vlm is None:
            raise RuntimeError("Must call load() before prediction")
        
        # Use the underlying Qwen3VL model to get vision features
        if isinstance(self.vlm, Qwen3VL):
            return self.vlm.get_vision_features(images)
        else:
            # Fallback for generic HF models
            from qwen_vl_utils import process_vision_info
            messages = []
            for img in images:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe this image."}
                    ]
                })
            
            texts = self.vlm.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            
            inputs = self.vlm.processor(
                text=texts, images=image_inputs, padding=True, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                vision_outputs = self.vlm.model.model.visual(inputs["pixel_values"])
            
            return vision_outputs.last_hidden_state
    
    def _get_pooled_features(self, images: List[Image.Image]) -> torch.Tensor:
        """Get pooled vision features for classification."""
        features = self._get_vision_features(images)
        # features: [B, num_patches, hidden_dim]
        # Pool over patches
        pooled = features.mean(dim=1)  # [B, hidden_dim]
        return pooled
    
    def _get_training_matching_features(self, images: List[Image.Image], prompt: Optional[str] = None) -> torch.Tensor:
        """Get mean-pooled LLM decoder last-layer hidden states, matching training-time construction.

        Mirrors HeadCollator: per-frame processor call with text
        "<|vision_start|>" + "<|image_pad|>"*n_vis + "<|vision_end|>" (+ prompt),
        padded to the batch max length, then forward through self.vlm.model
        (raw Qwen2.5-VL) and mean-pool hidden_states[-1] over the sequence.
        """
        if isinstance(images, Image.Image):
            images = [images]

        processor = self.vlm.processor
        tokenizer = processor.tokenizer
        image_processor = processor.image_processor
        merge_size = getattr(image_processor, "merge_size", 2)
        max_length = 2048
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        pixel_values_list = []
        grid_thw_list = []
        input_ids_list = []
        attention_mask_list = []
        seq_lengths = []

        for img in images:
            # text="" here is a placeholder only: this call is used solely for
            # pixel_values/image_grid_thw, and newer transformers versions crash
            # on text=None (Qwen2.5-VL processor indexes into `text`). The real
            # input_ids are built manually below via the tokenizer.
            proc = processor(images=img, text="", return_tensors="pt")
            pixel_values_list.append(proc["pixel_values"])
            grid_thw = proc.get("image_grid_thw")
            grid_thw_list.append(grid_thw)

            if grid_thw is not None:
                t, h, w = grid_thw[0].tolist()
                n_vis = (t * h * w) // (merge_size ** 2)
            else:
                n_vis = 0

            text = "<|vision_start|>" + "<|image_pad|>" * n_vis + "<|vision_end|>"
            if prompt:
                text += "\n" + prompt

            # Never truncate the vision placeholder: truncating <|image_pad|>
            # tokens breaks the image-token/feature-count correspondence that
            # the Qwen2.5-VL forward asserts on. Resolution is bounded upstream
            # by the image processor's max_pixels, so n_vis stays small.
            encoded = tokenizer(text, truncation=False, add_special_tokens=False)
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            seq_lengths.append(input_ids.shape[0])

        max_seq_len = max(seq_lengths)

        pixel_values = torch.cat(pixel_values_list, dim=0)
        if grid_thw_list[0] is not None:
            image_grid_thw = torch.cat(grid_thw_list, dim=0)
        else:
            image_grid_thw = None

        padded_input_ids = torch.full(
            (len(images), max_seq_len),
            pad_token_id,
            dtype=torch.long,
        )
        padded_attention_mask = torch.zeros((len(images), max_seq_len), dtype=torch.long)
        for i in range(len(images)):
            n = seq_lengths[i]
            padded_input_ids[i, :n] = input_ids_list[i]
            padded_attention_mask[i, :n] = attention_mask_list[i]

        model = self.vlm.model
        device = next(model.parameters()).device
        inputs = {
            "pixel_values": pixel_values.to(device),
            "video_grid_thw": None,
            "input_ids": padded_input_ids.to(device),
            "attention_mask": padded_attention_mask.to(device),
            "output_hidden_states": True,
            "return_dict": True,
        }
        if image_grid_thw is not None:
            inputs["image_grid_thw"] = image_grid_thw.to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        features = outputs.hidden_states[-1]  # [B, seq_len, hidden_dim]
        return features.mean(dim=1)  # [B, hidden_dim]
    
    def predict_phase(self, frames: List[Image.Image], prompt: Optional[str] = None) -> torch.Tensor:
        """
        Predict phase classification logits from frames.
        
        Args:
            frames: List of PIL Images (single frame or sequence)
            
        Returns:
            Tensor of shape [B, 8] with phase logits
        """
        if self.vlm is None:
            raise RuntimeError("Must call load() before predict_phase()")
        
        # Use training-matching features when a prompt is provided
        if prompt:
            pooled_features = self._get_training_matching_features(frames, prompt)
        else:
            pooled_features = self._get_pooled_features(frames)  # [B, hidden_dim]
        
        # Apply phase classification head
        if not hasattr(self, '_phase_head'):
            hidden_dim = pooled_features.shape[-1]
            self._phase_head = nn.Linear(hidden_dim, 8).to(self.device)
            nn.init.xavier_uniform_(self._phase_head.weight)
            nn.init.zeros_(self._phase_head.bias)
        
        logits = self._phase_head(pooled_features)
        return logits
    
    def predict_instruments(self, frames: List[Image.Image], prompt: Optional[str] = None) -> torch.Tensor:
        """
        Predict instrument detection logits from frames (multi-label).
        
        Args:
            frames: List of PIL Images
            
        Returns:
            Tensor of shape [B, 7] with instrument logits (sigmoid for probabilities)
        """
        if self.vlm is None:
            raise RuntimeError("Must call load() before predict_instruments()")
        
        if prompt:
            pooled_features = self._get_training_matching_features(frames, prompt)
        else:
            pooled_features = self._get_pooled_features(frames)  # [B, hidden_dim]
        
        if not hasattr(self, '_instrument_head'):
            hidden_dim = pooled_features.shape[-1]
            self._instrument_head = nn.Linear(hidden_dim, 7).to(self.device)
            nn.init.xavier_uniform_(self._instrument_head.weight)
            nn.init.zeros_(self._instrument_head.bias)
        
        logits = self._instrument_head(pooled_features)
        return logits
    
    def predict_action(self, frames: List[Image.Image], prompt: Optional[str] = None) -> torch.Tensor:
        """
        Predict action classification logits from frames.
        
        Args:
            frames: List of PIL Images
            
        Returns:
            Tensor of shape [B, 10] with action logits
        """
        if self.vlm is None:
            raise RuntimeError("Must call load() before predict_action()")
        
        if prompt:
            pooled_features = self._get_training_matching_features(frames, prompt)
        else:
            pooled_features = self._get_pooled_features(frames)  # [B, hidden_dim]
        
        if not hasattr(self, '_action_head'):
            hidden_dim = pooled_features.shape[-1]
            self._action_head = nn.Linear(hidden_dim, 10).to(self.device)
            nn.init.xavier_uniform_(self._action_head.weight)
            nn.init.zeros_(self._action_head.bias)
        
        logits = self._action_head(pooled_features)
        return logits
    
    def predict_triplet(self, frames: List[Image.Image], prompt: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Predict action triplet logits (instrument, verb, target).
        
        Args:
            frames: List of PIL Images
            
        Returns:
            Dict with 'instrument', 'verb', 'target' logits
        """
        if self.vlm is None:
            raise RuntimeError("Must call load() before predict_triplet()")
        
        if prompt:
            pooled_features = self._get_training_matching_features(frames, prompt)
        else:
            pooled_features = self._get_pooled_features(frames)  # [B, hidden_dim]
        
        if not hasattr(self, '_triplet_inst_head'):
            hidden_dim = pooled_features.shape[-1]
            self._triplet_inst_head = nn.Linear(hidden_dim, 6).to(self.device)
            self._triplet_verb_head = nn.Linear(hidden_dim, 10).to(self.device)
            self._triplet_target_head = nn.Linear(hidden_dim, 15).to(self.device)
            for head in [self._triplet_inst_head, self._triplet_verb_head, self._triplet_target_head]:
                nn.init.xavier_uniform_(head.weight)
                nn.init.zeros_(head.bias)
        
        return {
            'instrument': self._triplet_inst_head(pooled_features),
            'verb': self._triplet_verb_head(pooled_features),
            'target': self._triplet_target_head(pooled_features),
        }
    
    def _analyze_temporal_fallback(self, frames: List[Image.Image], prompt: str) -> Dict:
        """Fallback when temporal aggregator is not available."""
        per_frame_outputs = []
        for frame in frames:
            result = self.analyze_frame(frame)
            per_frame_outputs.append(result)
        
        aggregated = self._aggregate_text_outputs(per_frame_outputs)
        return {
            "predictions": aggregated,
            "temporal_attention_weights": None,
            "per_frame_outputs": per_frame_outputs,
        }
    
    def _aggregate_text_outputs(self, per_frame_outputs: List[Dict]) -> Dict:
        """Aggregate per-frame text outputs by majority voting for phase, union for tools."""
        if not per_frame_outputs:
            return {
                "phase": "Unknown",
                "tools": [],
                "description": "No frames processed"
            }
        
        # Majority vote for phase
        phase_votes = Counter([o.get("phase", "Unknown") for o in per_frame_outputs])
        majority_phase = phase_votes.most_common(1)[0][0] if phase_votes else "Unknown"
        
        # Union of tools
        all_tools = set()
        descriptions = []
        for o in per_frame_outputs:
            all_tools.update(o.get("tools", []))
            if o.get("description"):
                descriptions.append(o["description"])
        
        # Combine descriptions
        combined_desc = " | ".join(descriptions[:3]) if descriptions else ""
        
        return {
            "phase": majority_phase,
            "tools": sorted(list(all_tools)),
            "description": combined_desc
        }
    
    def process_video(
        self,
        video_path: str,
        config: Dict[str, Any]
    ) -> "VideoAnalysisResult":
        """
        Process a full surgical video.
        
        Args:
            video_path: Path to video file
            config: Dict with keys:
                - max_frames: int (default 500)
                - sampling_strategy: "uniform" | "keyframe" | "sliding" (default "uniform")
                - confidence_thresholds: dict with phase/tool thresholds
                
        Returns:
            VideoAnalysisResult with phase_timeline, frame_predictions, etc.
        """
        import cv2
        import numpy as np
        from dataclasses import dataclass
        from typing import List
        
        @dataclass
        class PhasePrediction:
            phase_name: str
            confidence: float
        
        @dataclass
        class Detection:
            label: str
            bbox: List[float]  # [x1, y1, x2, y2] normalized
            confidence: float
        
        @dataclass
        class VideoAnalysisResult:
            phase_timeline: List[Dict]
            frame_indices: List[int]
            timestamps: List[float]
            phase_predictions: List[PhasePrediction]
            tool_detections: List[List[Detection]]
        
        # Extract frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        max_frames = config.get("max_frames", 500)
        sampling = config.get("sampling_strategy", "uniform")
        
        if sampling == "uniform":
            frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        elif sampling == "sliding":
            interval = config.get("frame_interval", 5)
            frame_indices = np.arange(0, total_frames, interval)[:max_frames]
        else:  # keyframe - simplified to uniform
            frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        
        frames = []
        timestamps = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(frame)
                timestamps.append(idx / fps)
        
        cap.release()
        
        # Analyze frames
        phase_predictions = []
        tool_detections = []
        
        phase_thresh = config.get("confidence_thresholds", {}).get("phase", 0.5)
        tool_thresh = config.get("confidence_thresholds", {}).get("tool", 0.3)
        
        for frame in frames:
            result = self.analyze_frame(frame)
            
            phase_predictions.append(PhasePrediction(
                phase_name=result["phase"],
                confidence=phase_thresh  # Placeholder - model doesn't output confidence
            ))
            
            # Convert tools to Detection objects (no bboxes from VLM)
            tool_detections.append([
                Detection(label=t, bbox=[0, 0, 1, 1], confidence=tool_thresh)
                for t in result["tools"]
            ])
        
        # Build phase timeline
        phase_timeline = self._build_timeline(phase_predictions, timestamps, frame_indices.tolist())
        
        return VideoAnalysisResult(
            phase_timeline=phase_timeline,
            frame_indices=frame_indices.tolist(),
            timestamps=timestamps,
            phase_predictions=phase_predictions,
            tool_detections=tool_detections,
        )
    
    def _build_timeline(
        self,
        phase_predictions: List,
        timestamps: List[float],
        frame_indices: List[int]
    ) -> List[Dict]:
        """Build phase timeline from frame predictions."""
        if not phase_predictions:
            return []
        
        timeline = []
        current_phase = phase_predictions[0].phase_name
        segment_start_idx = 0
        segment_start_time = timestamps[0]
        
        for i, pred in enumerate(phase_predictions):
            if pred.phase_name != current_phase:
                # End previous segment
                segment_preds = phase_predictions[segment_start_idx:i]
                avg_conf = np.mean([p.confidence for p in segment_preds])
                
                timeline.append({
                    "phase": current_phase,
                    "start_frame": frame_indices[segment_start_idx],
                    "end_frame": frame_indices[i - 1],
                    "start_time": segment_start_time,
                    "end_time": timestamps[i - 1],
                    "duration": timestamps[i - 1] - segment_start_time,
                    "avg_confidence": float(avg_conf),
                })
                
                # Start new segment
                current_phase = pred.phase_name
                segment_start_idx = i
                segment_start_time = timestamps[i]
        
        # Final segment
        segment_preds = phase_predictions[segment_start_idx:]
        avg_conf = np.mean([p.confidence for p in segment_preds])
        
        timeline.append({
            "phase": current_phase,
            "start_frame": frame_indices[segment_start_idx],
            "end_frame": frame_indices[-1],
            "start_time": segment_start_time,
            "end_time": timestamps[-1],
            "duration": timestamps[-1] - segment_start_time,
            "avg_confidence": float(avg_conf),
        })
        
        return timeline
    
    def print_model_summary(self) -> None:
        """Print model parameter summary."""
        if self.vlm is None:
            logger.warning("Model not loaded")
            return
        
        model = self.vlm.model
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info("=" * 60)
        logger.info("SurgicalVLM Model Summary")
        logger.info("=" * 60)
        logger.info(f"Total parameters: {total:,}")
        logger.info(f"Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")
        logger.info(f"Frozen parameters: {total - trainable:,}")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"4-bit quantized: {self.load_in_4bit}")
        logger.info(f"LoRA applied: {self.lora_applied}")
        logger.info("=" * 60)


class MockSurgicalVLM:
    """Deterministic mock VLM for local UI testing without GPU (--mock flag).

    Returns a canned JSON response so every Gradio panel populates without
    loading a real checkpoint.
    """

    _model_version = "mock-v1"

    def generate_single(self, pil_image: Any, prompt: str = "") -> str:
        return (
            '{"phase": "CalotTriangle", '
            '"tools": ["Grasper", "Hook"], '
            '"narrative": "Mocked surgical frame — CalotTriangle dissection phase."}'
        )

    def load(self) -> None:
        pass

    def eval_mode(self) -> None:
        pass

    def train_mode(self) -> None:
        pass

    def to(self, device: Any) -> "MockSurgicalVLM":
        return self


def create_surgical_vlm(
    model_type: str = "qwen3_vl",
    load_in_4bit: bool = True,
    device: str = "auto",
    use_temporal: bool = False,
    temporal_config: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    max_pixels: int = 12845056,
    trust_remote_code: bool = True,
    **kwargs
) -> SurgicalVLM:
    """Factory function to create SurgicalVLM."""
    return SurgicalVLM(
        model_type=model_type, 
        load_in_4bit=load_in_4bit,
        device=device,
        use_temporal=use_temporal,
        temporal_config=temporal_config,
        model_name=model_name,
        max_pixels=max_pixels,
        trust_remote_code=trust_remote_code,
        **kwargs
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Quick test
    vlm = create_surgical_vlm()
    print(f"Created SurgicalVLM with {vlm.model_type}")