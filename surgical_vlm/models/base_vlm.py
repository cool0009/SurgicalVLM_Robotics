"""
Abstract base class for all VLM wrappers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image


class BaseVLM(ABC):
    """Abstract interface that every model wrapper must implement."""

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    @abstractmethod
    def load(self) -> None:
        """Load model weights and processor onto device."""
        ...

    @abstractmethod
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Extract visual features from a single image.
        Returns tensor of shape [1, num_patches, hidden_dim].
        """
        ...

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> str:
        """Generate text given an image and prompt."""
        ...

    @abstractmethod
    def get_hidden_states(
        self, image: Image.Image, layer: int = -1
    ) -> torch.Tensor:
        """
        Get hidden states from a specific layer.
        Used by temporal aggregator for Stage 2.
        Returns tensor of shape [1, seq_len, hidden_dim].
        """
        ...

    @abstractmethod
    def get_model_and_processor(self) -> Tuple:
        """Return raw model and processor for LoRA wrapping."""
        ...

    def to(self, device: str) -> "BaseVLM":
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self

    def eval_mode(self) -> None:
        if self.model is not None:
            self.model.eval()

    def train_mode(self) -> None:
        if self.model is not None:
            self.model.train()