"""
Qwen3-VL wrapper — PRIMARY model.
Handles qwen-vl-utils image preprocessing.
Supports 4-bit quantization via bitsandbytes for VRAM efficiency.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from .base_vlm import BaseVLM

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

logger = logging.getLogger(__name__)


class Qwen3VL(BaseVLM):
    """Qwen3-VL-7B/8B wrapper for local inference with 4-bit support."""

    def __init__(
        self,
        config: Optional[Dict] = None,
        **kwargs
    ):
        # Handle both config dict (from YAML) and explicit kwargs
        if config is not None:
            super().__init__(config)
            self.model_name = config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
            self.max_pixels = config.get("max_pixels", 12845056)
            self.trust_remote_code = config.get("trust_remote_code", True)
            self.load_in_4bit = config.get("load_in_4bit", True)
            self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        else:
            # Direct instantiation with kwargs
            super().__init__(kwargs)
            self.model_name = kwargs.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
            self.max_pixels = kwargs.get("max_pixels", 12845056)
            self.trust_remote_code = kwargs.get("trust_remote_code", True)
            self.load_in_4bit = kwargs.get("load_in_4bit", True)
            self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Qwen3VL initialized: {self.model_name} (4-bit: {self.load_in_4bit}, device: {self.device})")

    def load(self) -> None:
        """Load local model with optional 4-bit quantization.
        
        CRITICAL FIX for single-GPU T4: 
        - Load with device_map={"": 0} to trigger BnB dispatch
        - Immediately strip ALL accelerate hooks to prevent forward-pass buffer bugs
        - Force ALL buffers (including rotary inv_freq) and parameters to cuda:0
        """
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            max_pixels=self.max_pixels,
        )
        
        # Handle CPU-only mode
        if self.device == "cpu":
            if self.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="cpu",
                    trust_remote_code=self.trust_remote_code,
                )
            else:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=self.trust_remote_code,
                )
        else:
            # GPU mode (CUDA) - Single T4 fix
            if self.load_in_4bit:
                # 4-bit quantization config for VRAM efficiency
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map={"": 0},           # Force dispatch to cuda:0 only
                    torch_dtype=torch.bfloat16,    # Match BnB compute dtype (fp16 vision tower overflows -> NaN at max_pixels 1003520)
                    trust_remote_code=self.trust_remote_code,
                )
            else:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map={"": 0},
                    trust_remote_code=self.trust_remote_code,
                )
            
            # ============ CRITICAL FIX: Strip accelerate hooks & force cuda:0 ============
            # After dispatch, accelerate registers AlignDevicesHook on modules.
            # These hooks mishandle non-parameter buffers (rotary inv_freq) during 
            # PEFT forward pass, creating phantom cuda:1 tensors. 
            # We strip ALL hooks and explicitly move every buffer/param to cuda:0.
            
            # 1. Properly remove all accelerate hooks from model and submodules
            # Using accelerate's remove_hook_from_module to fully clean up
            try:
                from accelerate.hooks import remove_hook_from_module
                for module in self.model.modules():
                    if hasattr(module, '_hf_hook') and module._hf_hook is not None:
                        remove_hook_from_module(module)
            except ImportError:
                # Fallback: manually delete the hook attribute
                for module in self.model.modules():
                    if hasattr(module, '_hf_hook'):
                        delattr(module, '_hf_hook')
            
            # 2. Force ALL parameters and buffers to cuda:0 explicitly
            # This catches rotary_emb.inv_freq and other non-parameter buffers
            # that device_map misses because they're not nn.Parameters
            target_device = torch.device("cuda:0")
            for param in self.model.parameters():
                if param.device != target_device:
                    param.data = param.data.to(target_device)
            for buffer in self.model.buffers():
                if buffer.device != target_device:
                    buffer.data = buffer.data.to(target_device)
            
            # 3. Extra safety: explicitly move rotary embedding buffers
            # Qwen2.5-VL stores rotary_emb in model.model.rotary_emb
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'rotary_emb'):
                rotary = self.model.model.rotary_emb
                for name, buf in rotary.named_buffers():
                    if buf.device != target_device:
                        buf.data = buf.data.to(target_device)
                for name, param in rotary.named_parameters():
                    if param.device != target_device:
                        param.data = param.data.to(target_device)
            
            # 4. Disable gradient checkpointing on vision tower if present (saves memory)
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'visual'):
                visual = self.model.model.visual
                if hasattr(visual, 'gradient_checkpointing'):
                    visual.gradient_checkpointing = False
            
            # 5. Ensure model knows it's on cuda:0
            self.model.to(target_device)
        
        self.model.eval()

    def _build_messages(self, image: Image.Image, prompt: str) -> List[Dict]:
        """Build Qwen-style message format."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return messages

    def _get_visual_encoder(self):
        """Get the vision encoder, handling both base model and LoRA-wrapped model."""
        # Try multiple paths to find the visual encoder
        if hasattr(self.model, "base_model"):
            # LoRA-wrapped model: PeftModel -> LoraModel -> PeftModel -> outer HF -> inner HF -> visual
            bm = self.model.base_model  # LoraModel
            # Path 1: bm.model is PeftModel -> base_model is LoraModel -> model is outer HF -> model is inner HF -> visual
            if (hasattr(bm, "model") and hasattr(bm.model, "base_model") and
                hasattr(bm.model.base_model, "model") and hasattr(bm.model.base_model.model, "model") and
                hasattr(bm.model.base_model.model.model, "visual")):
                return bm.model.base_model.model.model.visual
            # Path 2: bm.model.model.visual (PeftModel -> inner HF) - but bm.model is PeftModel
            if (hasattr(bm, "model") and hasattr(bm.model, "model") and
                hasattr(bm.model.model, "visual")):
                return bm.model.model.visual
            # Path 3: bm.model.base_model.model.visual (PeftModel -> LoraModel -> outer HF -> inner HF)
            if (hasattr(bm, "model") and hasattr(bm.model, "base_model") and
                hasattr(bm.model.base_model, "model") and hasattr(bm.model.base_model.model, "visual")):
                return bm.model.base_model.model.visual
            # Path 4: bm.model.visual
            if (hasattr(bm, "model") and hasattr(bm.model, "visual")):
                return bm.model.visual
            # Path 5: bm.base_model.model.visual (nested LoraModel)
            if (hasattr(bm, "base_model") and hasattr(bm.base_model, "model") and
                hasattr(bm.base_model.model, "visual")):
                return bm.base_model.model.visual
        # Base model (no LoRA): self.model -> outer HF model -> inner HF model -> visual
        if hasattr(self.model, "model") and hasattr(self.model.model, "visual"):
            return self.model.model.visual
        if hasattr(self.model, "visual"):
            return self.model.visual
        raise AttributeError("Could not find visual encoder in model")

    def _build_messages_multi(self, images: List[Image.Image], prompt: str) -> List[Dict]:
        """Build messages with multiple images."""
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Extract vision encoder hidden states."""
        messages = self._build_messages(image, "Describe this image briefly.")
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if process_vision_info is not None:
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.processor(
                text=[text], images=[image], return_tensors="pt"
            ).to(self.device)
        
        with torch.no_grad():
            outputs = self._get_visual_encoder()(
                inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]
            )
        
        return outputs.last_hidden_state

    def get_hidden_states(self, image: Image.Image, layer: int = -1) -> torch.Tensor:
        """
        Get hidden states from the vision encoder.
        Used by temporal aggregator for Stage 2 and the frame-level
        classification feature path.
        Returns tensor of shape [1, num_merged_tokens, out_hidden_size].
        """
        # For Qwen2.5-VL, we extract from the vision encoder
        # which is at model.model.visual
        messages = self._build_messages(image, "Describe this image briefly.")
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if process_vision_info is not None:
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.processor(
                text=[text], images=[image], return_tensors="pt"
            ).to(self.device)
        
        with torch.no_grad():
            outputs = self._get_visual_encoder()(
                inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]
            )
        
        # Return the merged (spatial_merge_size=2) and out-projected tokens.
        # pooler_output is [num_merged_tokens, out_hidden_size] (2D) on the
        # live model, so we add a batch dim to yield
        # [1, num_merged_tokens, out_hidden_size]. This matches the
        # output_adapter hidden_dim (3584) and the temporal aggregator's
        # expected feature dim. last_hidden_state is the raw pre-merge patch
        # sequence and is NOT compatible with those heads.
        return outputs.pooler_output.unsqueeze(0)

    def get_vision_features(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extract vision encoder features for a batch of images.

        Args:
            images: List of PIL Images (typically a frame sequence).

        Returns:
            Tensor of shape [B, num_patches, hidden_dim] where B is the number
            of images and num_patches is the number of vision tokens per image.
        """
        if not images:
            raise ValueError("get_vision_features requires at least one image")

        # Encode each image independently (vision tower is deterministic per frame)
        # and stack along the batch dimension. Each frame yields [1, num_patches, hidden].
        frame_features = [self.get_hidden_states(img) for img in images]
        return torch.cat(frame_features, dim=0)  # [B, num_patches, hidden_dim]

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        **kwargs
    ) -> str:
        """Generate text from image and prompt."""
        messages = self._build_messages(image, prompt)
        
        if process_vision_info is not None:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text], images=image_inputs, padding=True, return_tensors="pt"
            ).to(self.device)
        else:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        return response

    def get_model_and_processor(self) -> Tuple:
        """Return underlying model and processor for Trainer."""
        return self.model, self.processor

    def to(self, device: str) -> "Qwen3VL":
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self

    def eval_mode(self) -> None:
        if self.model is not None:
            self.model.eval()

    def train_mode(self) -> None:
        if self.model is not None:
            self.model.train()