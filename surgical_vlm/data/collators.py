import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os
from pathlib import Path
from PIL import Image


class TemporalCollator:
    def __init__(
        self,
        max_frames: int = 32,
        pad_value: float = 0.0,
        return_mask: bool = True
    ):
        self.max_frames = max_frames
        self.pad_value = pad_value
        self.return_mask = return_mask
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        frames_list = [item['frames'] for item in batch]
        
        lengths = [f.shape[0] for f in frames_list]
        max_len = min(max(lengths), self.max_frames)
        
        padded_frames = []
        masks = []
        
        for frames in frames_list:
            seq_len = frames.shape[0]
            if seq_len > max_len:
                frames = frames[:max_len]
                seq_len = max_len
            
            if seq_len < max_len:
                pad_shape = (max_len - seq_len,) + frames.shape[1:]
                padding = torch.full(pad_shape, self.pad_value, dtype=frames.dtype)
                frames = torch.cat([frames, padding], dim=0)
            
            padded_frames.append(frames)
            
            if self.return_mask:
                mask = torch.zeros(max_len, dtype=torch.bool)
                mask[:seq_len] = True
                masks.append(mask)
        
        batched_frames = torch.stack(padded_frames, dim=0)
        
        result = {'frames': batched_frames}
        
        if self.return_mask:
            result['frame_mask'] = torch.stack(masks, dim=0)
        
        for key in batch[0].keys():
            if key not in ['frames', 'frame_mask']:
                values = [item[key] for item in batch]
                if isinstance(values[0], torch.Tensor):
                    result[key] = torch.stack(values, dim=0)
                else:
                    result[key] = values
        
        return result


class VLMPromptCollator:
    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        pad_to_multiple_of: int = 8,
        return_tensors: str = 'pt'
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        frames_list = [item['frames'] for item in batch]
        input_ids_list = [item['input_ids'] for item in batch]
        attention_mask_list = [item['attention_mask'] for item in batch]
        
        max_frames = max(f.shape[0] for f in frames_list)
        padded_frames = []
        frame_masks = []
        
        for frames in frames_list:
            seq_len = frames.shape[0]
            if seq_len < max_frames:
                pad_shape = (max_frames - seq_len,) + frames.shape[1:]
                padding = torch.zeros(pad_shape, dtype=frames.dtype)
                frames = torch.cat([frames, padding], dim=0)
            padded_frames.append(frames)
            
            mask = torch.zeros(max_frames, dtype=torch.bool)
            mask[:seq_len] = True
            frame_masks.append(mask)
        
        batched_frames = torch.stack(padded_frames, dim=0)
        batched_frame_masks = torch.stack(frame_masks, dim=0)
        
        max_text_len = max(ids.shape[0] for ids in input_ids_list)
        if self.pad_to_multiple_of:
            max_text_len = ((max_text_len + self.pad_to_multiple_of - 1) // 
                           self.pad_to_multiple_of) * self.pad_to_multiple_of
        max_text_len = min(max_text_len, self.max_length)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            seq_len = input_ids.shape[0]
            if seq_len > max_text_len:
                input_ids = input_ids[:max_text_len]
                attention_mask = attention_mask[:max_text_len]
                seq_len = max_text_len
            
            if seq_len < max_text_len:
                pad_len = max_text_len - seq_len
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_len, dtype=attention_mask.dtype)
                ])
            
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
        
        batched_input_ids = torch.stack(padded_input_ids, dim=0)
        batched_attention_mask = torch.stack(padded_attention_mask, dim=0)
        
        result = {
            'frames': batched_frames,
            'frame_mask': batched_frame_masks,
            'input_ids': batched_input_ids,
            'attention_mask': batched_attention_mask,
        }
        
        for key in batch[0].keys():
            if key not in ['frames', 'input_ids', 'attention_mask', 'frame_mask']:
                values = [item[key] for item in batch]
                if isinstance(values[0], torch.Tensor):
                    result[key] = torch.stack(values, dim=0)
                else:
                    result[key] = values
        
        return result


class SegmentationCollator:
    def __init__(self, ignore_index: int = -100):
        self.ignore_index = ignore_index
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = torch.stack([item['image'] for item in batch], dim=0)
        masks = torch.stack([item['mask'] for item in batch], dim=0)
        
        result = {
            'images': images,
            'masks': masks,
        }
        
        for key in batch[0].keys():
            if key not in ['image', 'mask']:
                values = [item[key] for item in batch]
                if isinstance(values[0], torch.Tensor):
                    result[key] = torch.stack(values, dim=0)
                else:
                    result[key] = values
        
        return result


class PhaseClassificationCollator:
    def __init__(self, num_classes: int = 7, include_tools: bool = True):
        self.num_classes = num_classes
        self.include_tools = include_tools
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        frames = torch.stack([item['frames'] for item in batch], dim=0)
        labels = torch.stack([item['phase_label'] for item in batch], dim=0)
        
        result = {
            'frames': frames,
            'labels': labels,
        }
        
        # Include tool labels if available
        if self.include_tools and 'tool_label' in batch[0]:
            tool_labels = torch.stack([item['tool_label'] for item in batch], dim=0)
            result['tool_labels'] = tool_labels
        
        for key in batch[0].keys():
            if key not in ['frames', 'phase_label', 'tool_label']:
                values = [item[key] for item in batch]
                if isinstance(values[0], torch.Tensor):
                    result[key] = torch.stack(values, dim=0)
                else:
                    result[key] = values
        
        return result


class MultiTaskCollator:
    def __init__(
        self,
        task_collators: Dict[str, Any],
        task_key: str = 'task',
        default_collator: Any = None,
        pad_token_id: int = 0
    ):
        self.task_collators = task_collators
        self.task_key = task_key
        self.default_collator = default_collator
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        """Collate a mixed-task batch into task-homogeneous micro-batches.

        The global DataLoader batch may mix samples from different tasks (e.g.
        3 temporal rows + 1 VQA row). Merging them into a single tensor dict is
        unsound: tasks use different collators and shapes (VQA rows carry no
        ``frame_mask``). Instead, group by task and emit one collated dict per
        task, each fully homogeneous. The trainer loops over the returned list,
        routing each micro-batch to the correct forward path.
        """
        task_indices: Dict[str, List[int]] = {}
        for i, item in enumerate(batch):
            task = item.get(self.task_key, 'default')
            task_indices.setdefault(task, []).append(i)

        micro_batches: List[Dict[str, torch.Tensor]] = []
        for task, indices in task_indices.items():
            task_batch = [batch[i] for i in indices]
            if task in self.task_collators:
                collated = self.task_collators[task](task_batch)
            elif self.default_collator is not None:
                collated = self.default_collator(task_batch)
            else:
                collated = self._default_collate(task_batch)
            collated[self.task_key] = task
            micro_batches.append(collated)

        return micro_batches

    def _pad_and_cat(self, key: str, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate tensors along dim 0, right-padding trailing dims to the
        batch-wide max so sub-batches with different sequence lengths merge.
        Padding values: ``attention_mask`` -> 0, ``labels`` -> -100 (loss
        ignore), ``input_ids`` -> pad_token_id, anything else -> 0."""
        shapes_tail = [t.shape[1:] for t in tensors]
        if all(s == shapes_tail[0] for s in shapes_tail):
            return torch.cat(tensors, dim=0)
        if len({len(s) for s in shapes_tail}) != 1:
            raise ValueError(f"Cannot merge tensors with different ndim for key {key}")
        pad_values = {'attention_mask': 0, 'labels': -100}
        pad_val = pad_values.get(
            key, self.pad_token_id if key == 'input_ids' else 0
        )
        max_tail = [max(s[d] for s in shapes_tail) for d in range(len(shapes_tail[0]))]
        padded = []
        for t in tensors:
            widths = []
            for d in range(len(max_tail) - 1, -1, -1):
                widths.extend([0, max_tail[d] - t.shape[d + 1]])
            if any(widths):
                t = F.pad(t, widths, value=pad_val)
            padded.append(t)
        return torch.cat(padded, dim=0)
    
    def _default_collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        result = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values, dim=0)
            else:
                result[key] = values
        return result


class VLMCollator:
    """Collator for VLM instruction tuning datasets (JSONL format)."""
    
    def __init__(self, processor=None, max_length=2048, image_root=None):
        self.processor = processor
        self.max_length = max_length
        self.image_root = Path(image_root) if image_root else None
    
    def __call__(self, batch):
        # Guard: raise if any sample has no image AND no pixel_values (silent data-poisoning)
        for i, sample in enumerate(batch):
            img_path = sample.get('image', sample.get('image_path'))
            has_image = img_path and os.path.exists(str(self.image_root / img_path)) if img_path and not os.path.isabs(img_path) else bool(img_path)
            # If image can't be loaded and pixel_values not provided, this batch will produce all-100 labels
            if not has_image:
                raise ValueError(
                    f"FATAL DATA ERROR at batch index {i}: image not found at {img_path}. "
                    "All labels will be -100, causing zero loss (silent data-poisoning). "
                    "Verify the image path is correct and the file exists."
                )
        
        images = []
        texts = []
        
        for sample in batch:
            # Load image
            img_path = sample.get('image', sample.get('image_path'))
            if img_path and self.image_root is not None and not os.path.isabs(img_path):
                img_path = str(self.image_root / img_path)
            if img_path and os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                images.append(image)
            else:
                images.append(None)
            
            # Build text from conversations
            conversations = sample.get('conversations', [])
            text_parts = []
            for conv in conversations:
                role = conv.get('from', conv.get('role', ''))
                value = conv.get('value', conv.get('content', ''))
                if role == 'human' or role == 'user':
                    text_parts.append(f"User: {value}")
                elif role == 'gpt' or role == 'assistant':
                    text_parts.append(f"Assistant: {value}")
            
            text = "\n".join(text_parts)
            if not text:
                text = "User: Describe this image.\nAssistant:"
            texts.append(text)
        
        # Process through VLM processor
        if self.processor is not None and any(img is not None for img in images):
            # Filter to only samples with images
            valid_indices = [i for i, img in enumerate(images) if img is not None]
            valid_images = [images[i] for i in valid_indices]
            valid_texts = [texts[i] for i in valid_indices]
            
            inputs = self.processor(
                text=valid_texts,
                images=valid_images,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # Add labels for training (copy input_ids, mask padding with -100)
            inputs['labels'] = inputs['input_ids'].clone()
            inputs['labels'][inputs['attention_mask'] == 0] = -100
            
            # Map pixel_values to frames for compatibility with trainer
            if 'pixel_values' in inputs:
                inputs['frames'] = inputs['pixel_values']
            
            # Store metadata for evaluation
            inputs['metadata'] = [batch[i] for i in valid_indices]
            
            return inputs
        else:
            # Fallback: return minimal batch
            return {
                'input_ids': torch.zeros(len(texts), self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(len(texts), self.max_length, dtype=torch.long),
                'labels': torch.full((len(texts), self.max_length), -100, dtype=torch.long),
                'metadata': batch
            }


class VQACollator:
    """Collator for VQA (visual question answering / instruction) samples.

    Builds a Qwen-style chat batch from ``conversations`` using the canonical
    processor pattern (messages -> apply_chat_template -> process_vision_info
    -> processor), producing pixel_values, input_ids, attention_mask plus the
    Qwen vision grid tensors. ``labels`` masks padding with ``-100`` so the
    HF causal-LM loss ignores them (mirrors the trainer's VQA branch default).
    """

    def __init__(self, processor=None, max_length: int = 2048, image_root=None):
        self.processor = processor
        self.max_length = max_length
        self.image_root = Path(image_root) if image_root else None

    def _resolve(self, p: str) -> str:
        if self.image_root is not None and not os.path.isabs(p):
            return str(self.image_root / p)
        return p

    @staticmethod
    def _role(conv: Dict[str, Any]) -> str:
        role = conv.get("from", conv.get("role", ""))
        if role in ("human", "user"):
            return "user"
        if role in ("gpt", "assistant"):
            return "assistant"
        return "user"

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = []
        image_lists = []
        for sample in batch:
            paths = sample.get("image_path") or sample.get("image") or []
            if isinstance(paths, str):
                paths = [paths]

            pil_images = []
            for p in paths:
                resolved = self._resolve(p) if p else p
                if resolved and os.path.exists(resolved):
                    try:
                        pil_images.append(Image.open(resolved).convert("RGB"))
                    except Exception:
                        continue

            messages = []
            first_user_seen = False
            for conv in sample.get("conversations", []):
                value = conv.get("value", conv.get("content", ""))
                if not str(value).strip():
                    continue
                role = self._role(conv)
                prompt = str(value)
                if role == "user" and pil_images and not first_user_seen:
                    content = [{"type": "image", "image": img} for img in pil_images]
                    content.append({"type": "text", "text": prompt})
                    messages.append({"role": "user", "content": content})
                    first_user_seen = True
                else:
                    messages.append({"role": role, "content": prompt})

            if not messages:
                messages.append({"role": "user", "content": "Describe this image."})
                if pil_images:
                    messages[0]["content"] = [
                        {"type": "image", "image": pil_images[0]},
                        {"type": "text", "text": "Describe this image."},
                    ]
            elif not first_user_seen and pil_images:
                messages.insert(0, {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_images[0]},
                        {"type": "text", "text": ""},
                    ],
                })

            texts.append(self._apply_chat_template(messages))
            if pil_images:
                image_lists.append(pil_images)

        if self.processor is not None and image_lists:
            try:
                inputs = self.processor(
                    text=texts,
                    images=image_lists,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
            except Exception:
                inputs = None
            if inputs is not None:
                inputs["labels"] = inputs["input_ids"].clone()
                inputs["labels"][inputs["attention_mask"] == 0] = -100
                inputs["frames"] = inputs.get("pixel_values")
                inputs["task"] = "vqa"
                inputs["dataset_name"] = (batch[0].get("dataset")
                                          or batch[0].get("dataset_name", "unknown"))
                inputs["metadata"] = batch
                return inputs

        return self._fallback(batch)

    def _apply_chat_template(self, messages) -> str:
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        parts = []
        for m in messages:
            prefix = "User: " if m["role"] == "user" else "Assistant: "
            content = m["content"]
            if isinstance(content, list):
                texts = [c["text"] for c in content if isinstance(c, dict) and c.get("type") == "text"]
                content = " ".join(texts) or ""
            parts.append(f"{prefix}{content}")
        return "\n".join(parts)

    def _fallback(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [self._apply_chat_template(
            [{"role": "user", "content": "Describe this image."}]
        ) for _ in batch]
        if self.processor is not None:
            try:
                inputs = self.processor(
                    text=texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=self.max_length,
                )
                inputs["labels"] = inputs["input_ids"].clone()
                inputs["labels"][inputs["attention_mask"] == 0] = -100
                inputs["task"] = "vqa"
                inputs["dataset_name"] = (batch[0].get("dataset")
                                          or batch[0].get("dataset_name", "unknown"))
                inputs["metadata"] = batch
                return inputs
            except Exception:
                pass
        return {
            "input_ids": torch.zeros(len(batch), 1, dtype=torch.long),
            "attention_mask": torch.zeros(len(batch), 1, dtype=torch.long),
            "labels": torch.full((len(batch), 1), -100, dtype=torch.long),
            "task": "vqa",
            "dataset_name": batch[0].get("dataset", "unknown"),
            "metadata": batch,
        }


class HeadCollator:
    """Collate VLM auto-regressive head-task samples.

    Encodes the (single) frame to Qwen2.5-VL ``pixel_values`` (flattened
    patches) + ``image_grid_thw``, rebuilds ``input_ids``/``attention_mask``
    so the prompt is prefixed with the exact number of Qwen image placeholder
    tokens (``<|vision_start|>`` + ``<|image_pad|>`` * N + ``<|vision_end|>``)
    that the vision tower produces for that frame — required because
    Qwen2.5-VL's forward asserts image-token count == feature count — and
    aggregates the multi-label arrays (``phase``/``tools``/``action``/
    ``triplet_*``) as lists of string labels. The ``dataset`` key is remapped
    to ``dataset_name`` (scalar first-row label, matching the SingleTask loss
    gating).
    """

    def __init__(self, processor=None, image_root=None, max_length: int = 2048):
        self.processor = processor
        self.image_root = Path(image_root) if image_root else None
        self.max_length = max_length
        self.tokenizer = getattr(processor, "tokenizer", None)
        image_processor = getattr(processor, "image_processor", None)
        self.merge_size = (
            getattr(image_processor, "merge_size", 2)
            if image_processor is not None else 2
        )
        # Force smaller resolution so vision tokens fit in max_length=2048
        self.pad_token_id = 0
        if self.tokenizer is not None:
            pad_id = self.tokenizer.pad_token_id
            self.pad_token_id = (
                pad_id if pad_id is not None else self.tokenizer.eos_token_id
            )

    @staticmethod
    def _prompt_of(row: Dict[str, Any]) -> str:
        for conv in row.get("conversations", []):
            if not isinstance(conv, dict):
                continue
            role = conv.get("from", conv.get("role", ""))
            if role in ("human", "user"):
                return str(conv.get("value", conv.get("content", "")))
        return str(row.get("prompt", ""))

    def _vision_token_count(self, grid: torch.Tensor) -> int:
        t, h, w = (int(x) for x in grid.reshape(-1)[:3])
        return (t * h * w) // (self.merge_size ** 2)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        drop = {"dataset", "input_ids", "attention_mask"}

        for key in batch[0].keys():
            if key in drop:
                continue
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                try:
                    result[key] = torch.stack(values, dim=0)
                except Exception:
                    result[key] = values
            else:
                result[key] = values

        # Multi-label classification fields -> list of per-row string labels
        for field in ("phase", "tools", "action", "triplet"):
            if field in result:
                result[field] = [row.get(field) for row in batch]
        result["triplet_verb"] = [row.get("triplet_verb") for row in batch]
        result["triplet_instrument"] = [row.get("triplet_instrument") for row in batch]
        result["triplet_target"] = [row.get("triplet_target") for row in batch]

        # Images -> pixel_values [sum_patches, patch_dim] + image_grid_thw [B, 3]
        pixel_list = []
        grid_list = []
        for row in batch:
            path = row.get("image_path") or row.get("image")
            if path and self.image_root is not None and not os.path.isabs(path):
                path = os.path.join(self.image_root, path)
            pixel = None
            grid = None
            if path and os.path.exists(path) and self.processor is not None:
                try:
                    img = Image.open(path).convert("RGB")
                    proc = self.processor(images=img, text="", return_tensors="pt")
                    pixel = proc["pixel_values"]
                    grid = proc.get("image_grid_thw")
                except Exception:
                    pixel = None
                    grid = None
            pixel_list.append(pixel)
            grid_list.append(grid)

        if not all(p is not None and g is not None for p, g in zip(pixel_list, grid_list)):
            missing = [
                (i, row.get("image_path") or row.get("image"))
                for i, (row, p, g) in enumerate(zip(batch, pixel_list, grid_list))
                if p is None or g is None
            ]
            raise RuntimeError(
                f"HeadCollator: {len(missing)}/{len(batch)} frames failed to load "
                "or are missing vision grid tensors. "
                f"Missing or unreadable images: {missing}. Either the path is "
                "wrong relative to image_root, the file is corrupt, or the "
                "image processor is None."
            )

        result["pixel_values"] = torch.cat(pixel_list, dim=0)
        result["frames"] = result["pixel_values"]
        result["image_grid_thw"] = torch.cat(grid_list, dim=0)

        # Rebuild text inputs: vision placeholder block + prompt, tokenized
        # with the processor's tokenizer so the model forward sees exactly as
        # many <|image_pad|> tokens as the vision tower emits features.
        ids_list = []
        for row, grid in zip(batch, grid_list):
            prompt = self._prompt_of(row)
            if self.tokenizer is not None:
                n_vis = self._vision_token_count(grid)
                text = (
                    "<|vision_start|>"
                    + "<|image_pad|>" * n_vis
                    + "<|vision_end|>"
                    + ("\n" + prompt if prompt else "")
                )
                enc = self.tokenizer(
                    text,
                    truncation=False,
                    add_special_tokens=False,
                )
                ids = torch.tensor(enc["input_ids"], dtype=torch.long)
            else:
                ids = torch.zeros(1, dtype=torch.long)
            ids_list.append(ids)

        max_len = max(ids.shape[0] for ids in ids_list)
        input_ids = torch.full(
            (len(ids_list), max_len), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros(len(ids_list), max_len, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            ids = ids[:max_len]
            input_ids[i, : ids.shape[0]] = ids
            attention_mask[i, : ids.shape[0]] = 1
        result["input_ids"] = input_ids
        result["attention_mask"] = attention_mask

        result["task"] = ""
        result["dataset_name"] = batch[0].get("dataset", "unknown")
        result["metadata"] = batch
        return result


class TemporalStackingCollator(HeadCollator):
    """Temporal Option-A collator.

    Expands each sample into ``T`` per-frame rows (the anchor frame plus the
    ``T-1`` nearest neighbors from ``frame_paths``) and delegates to
    :class:`HeadCollator` so every frame is encoded independently and the
    model sees a ``B*T`` stack of single-frame prompts (full-model B*T
    expansion). The collator then repairs the batch-ordered fields back to B
    (``[::t]`` selects each sample's anchor row) and attaches ``frame_mask`` of
    shape ``[B, T]`` so the trainer can fire the temporal branch and
    mean-pool the per-frame features.

    Keeps the anchor row's ``frame_idx``/``frame_paths``/``image_path`` on the
    repaired fields; sub-rows differ only in ``image_path`` (the frame to
    encode). Labels are intentionally not emitted here (same contract as
    :class:`HeadCollator`).
    """

    def __init__(
        self,
        processor=None,
        image_root=None,
        max_length: int = 2048,
        num_frames: int = 3,
    ):
        super().__init__(
            processor=processor, image_root=image_root, max_length=max_length
        )
        self.num_frames = max(1, num_frames)

    def _valid_frames(self, row: Dict[str, Any]) -> List[str]:
        anchor = row.get("image_path") or row.get("image")
        paths = row.get("frame_paths") or []
        valid = [str(p) for p in paths if p and str(p).strip()]
        if not valid and anchor:
            valid = [anchor]
        if not valid:
            raise RuntimeError(
                "TemporalStackingCollator: row has neither frame_paths nor "
                f"image_path. keys={sorted(row.keys())}"
            )
        return valid

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_lists = [self._valid_frames(row) for row in batch]
        t = max(1, min(self.num_frames, min(len(v) for v in valid_lists)))

        expanded: List[Dict[str, Any]] = []
        for row, valid in zip(batch, valid_lists):
            for k in range(t):
                sub = dict(row)
                sub["image_path"] = valid[k]
                expanded.append(sub)

        result = super().__call__(expanded)

        # Repair B*T-ordered fields back to B (keep each sample's anchor row).
        for key in (
            "phase",
            "tools",
            "action",
            "triplet_instrument",
            "triplet_verb",
            "triplet_target",
            "frame_idx",
            "frame_paths",
            "image_path",
            "video_id",
            "label",
        ):
            if isinstance(result.get(key), list):
                result[key] = result[key][::t]

        b = len(batch)
        result["frame_mask"] = torch.ones(b, t, dtype=torch.bool)
        result["num_frames"] = t
        result["metadata"] = batch
        return result