"""Small inference wrapper for the official Qwen3-VL embedding checkpoint.

Adapted from QwenLM/Qwen3-VL-Embedding's reference implementation.  Keeping
the wrapper local avoids installing that repository as an unpinned package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from qwen_vl_utils.vision_process import process_vision_info
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor


@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    # With ``from __future__ import annotations`` the annotation below is a
    # string. Transformers inspects config_class at runtime, so set the actual
    # class explicitly instead of relying on annotation resolution.
    config_class = Qwen3VLConfig
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values, image_grid_thw=None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


class Qwen3VLEmbedder:
    """Generate normalized text or sparse-video embeddings one item at a time."""

    def __init__(
        self,
        model_name_or_path='Qwen/Qwen3-VL-Embedding-2B',
        *,
        max_length=8192,
        max_pixels=336 * 336,
        total_pixels=4 * 336 * 336,
        attn_implementation='sdpa',
    ):
        self.max_length = max_length
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            dtype = 'auto'
        self.model = Qwen3VLForEmbedding.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            dtype=dtype,
            device_map='auto',
            attn_implementation=attn_implementation,
        )
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path, padding_side='right'
        )
        self.model.eval()

    @property
    def device(self):
        return self.model.device

    def _conversation(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        instruction = item.get('instruction') or "Represent the user's input."
        content: list[dict[str, Any]] = []
        if item.get('video'):
            frames = [
                value if str(value).startswith(('file://', 'http://', 'https://')) else f'file://{value}'
                for value in item['video']
            ]
            content.append({
                'type': 'video',
                'video': frames,
                'total_pixels': self.total_pixels,
                'sample_fps': item.get('sample_fps', 1.0),
            })
        if item.get('text') is not None:
            content.append({'type': 'text', 'text': str(item['text'])})
        if not content:
            content.append({'type': 'text', 'text': 'NULL'})
        return [
            {'role': 'system', 'content': [{'type': 'text', 'text': instruction}]},
            {'role': 'user', 'content': content},
        ]

    @staticmethod
    def _pool_last(hidden_state, attention_mask):
        last = attention_mask.shape[1] - attention_mask.flip(dims=[1]).argmax(dim=1) - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, last]

    @torch.inference_mode()
    def process(self, items: list[dict[str, Any]]) -> torch.Tensor:
        conversations = [self._conversation(item) for item in items]
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        images, video_inputs, video_kwargs = process_vision_info(
            conversations,
            image_patch_size=16,
            return_video_metadata=True,
            return_video_kwargs=True,
        )
        if video_inputs is not None:
            videos, metadata = zip(*video_inputs)
            videos, metadata = list(videos), list(metadata)
        else:
            videos, metadata = None, None
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=metadata,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=False,
            return_tensors='pt',
            **(video_kwargs or {}),
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        pooled = self._pool_last(outputs.last_hidden_state, inputs['attention_mask'])
        return F.normalize(pooled, p=2, dim=-1)
