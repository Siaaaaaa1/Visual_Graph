"""
Multi-turn VLM SFT dataset for training vision-language models on multimodal
conversation data produced by distill_data.py.

Expected parquet schema
-----------------------
  messages : list[dict]  — [{"role": "system"|"user"|"assistant", "content": str}, ...]
                            The FIRST user message may start with "<image>\n".
  images   : list[str]   — One base64-encoded JPEG string per sample (the graph view).

Returns per __getitem__
-----------------------
  input_ids      (seq_len,)
  attention_mask (seq_len,)
  position_ids   (3, seq_len)   — 3-D RoPE for Qwen3-VL / Qwen2-VL
  loss_mask      (seq_len,)     — 1 on assistant tokens, 0 elsewhere
  pixel_values   (n_patches, patch_dim)
  image_grid_thw (1, 3)
"""

import base64
from io import BytesIO
from typing import List, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from verl.utils.fs import copy_local_path_from_hdfs


class MultiTurnVLMSFTDataset(Dataset):
    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer,          # accepts Qwen3VLProcessor (or Qwen2VLProcessor)
        config=None,
    ):
        config = config or {}
        self.truncation  = config.get("truncation", "error")
        self.max_length  = config.get("max_length", 4096)
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.images_key   = multiturn_config.get("images_key",   "images")

        assert self.truncation in ("error", "left", "right")

        try:
            from omegaconf import OmegaConf
            if not isinstance(parquet_files, (str, list)):
                parquet_files = OmegaConf.to_container(parquet_files, resolve=True)
        except Exception:
            pass

        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        elif not isinstance(parquet_files, list):
            parquet_files = list(parquet_files)

        self.parquet_files = parquet_files

        # `tokenizer` is expected to be the full VLM processor
        if isinstance(tokenizer, str):
            from transformers import AutoProcessor
            tokenizer = AutoProcessor.from_pretrained(tokenizer, trust_remote_code=True)
        self.processor = tokenizer
        # text-only tokenizer portion (for apply_chat_template / encode)
        self._tok = getattr(tokenizer, "tokenizer", tokenizer)

        self._download()
        self._read_files()

    # ------------------------------------------------------------------
    def _download(self):
        for i, f in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(f, verbose=True)

    def _read_files(self):
        def _unwrap(val):
            import numpy, pandas
            while isinstance(val, (pandas.Series, numpy.ndarray)) and len(val) == 1:
                val = val[0]
            return val

        dfs = [pd.read_parquet(f) for f in self.parquet_files]
        df  = pd.concat(dfs, ignore_index=True)
        self.messages = df[self.messages_key].apply(_unwrap).tolist()
        if self.images_key in df.columns:
            self.images = df[self.images_key].apply(_unwrap).tolist()
        else:
            self.images = [None] * len(self.messages)

    def __len__(self):
        return len(self.messages)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decode_pil_images(self, images_b64) -> List[Image.Image]:
        """Base64 list → PIL Image list."""
        if not images_b64:
            return []
        result = []
        for b64 in images_b64:
            if b64 is None:
                continue
            result.append(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))
        return result

    def _expand_image_placeholders(self, text: str, image_grid_thw: torch.Tensor) -> str:
        """Replace every '<image>' in *text* with the correct number of vision tokens."""
        merge_length = self.processor.image_processor.merge_size ** 2
        for thw in image_grid_thw:
            n_tokens = int(thw.prod().item()) // merge_length
            text = text.replace(
                "<image>",
                "<|vision_start|>" + "<|image_pad|>" * n_tokens + "<|vision_end|>",
                1,
            )
        return text

    def _has_image(self, messages) -> bool:
        return any(
            isinstance(msg.get("content"), str) and "<image>" in msg["content"]
            for msg in messages
        )

    def _tokenize_subset(
        self,
        messages,
        pil_images: List[Image.Image],
        image_grid_thw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply chat template + expand image tokens for a (possibly partial)
        message list.  Returns 1-D input_ids tensor.
        """
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if self._has_image(messages) and pil_images and image_grid_thw is not None:
            text = self._expand_image_placeholders(text, image_grid_thw)

        ids = self._tok.encode(text, add_special_tokens=False)
        return torch.tensor(ids, dtype=torch.long)

    # ------------------------------------------------------------------
    def __getitem__(self, item):
        messages   = list(self.messages[item])      # ensure plain list
        images_b64 = self.images[item]

        pil_images = self._decode_pil_images(images_b64 if images_b64 is not None else [])

        # ── Process image with the VLM image processor ─────────────────
        if pil_images:
            img_inputs     = self.processor.image_processor(pil_images, return_tensors="pt")
            pixel_values   = img_inputs["pixel_values"]   # (n_patches, patch_dim)
            image_grid_thw = img_inputs["image_grid_thw"] # (n_images, 3)
        else:
            pixel_values   = None
            image_grid_thw = None

        # ── Full sequence tokenization ──────────────────────────────────
        input_ids      = self._tokenize_subset(messages, pil_images, image_grid_thw)
        attention_mask = torch.ones_like(input_ids)
        seq_len        = input_ids.shape[0]

        # ── Loss mask (1 on assistant tokens, 0 elsewhere) ─────────────
        # Find the index of the first user message that contains <image>.
        image_msg_idx = next(
            (i for i, m in enumerate(messages)
             if isinstance(m.get("content"), str) and "<image>" in m["content"]),
            None,
        )

        loss_mask = torch.zeros(seq_len, dtype=torch.long)

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            # Decide which images to pass for the prefix/prev tokenization.
            # Include pil_images whenever the first-image message is inside
            # the prefix (i.e. image_msg_idx < prefix_length).
            prefix_imgs = pil_images if (image_msg_idx is not None and image_msg_idx < i + 1) else []
            prev_imgs   = pil_images if (image_msg_idx is not None and image_msg_idx < i)     else []

            # image_grid_thw is needed only when images are included
            pfx_thw  = image_grid_thw if prefix_imgs else None
            prev_thw = image_grid_thw if prev_imgs   else None

            end_pos   = self._tokenize_subset(messages[: i + 1], prefix_imgs, pfx_thw).shape[0]
            start_pos = self._tokenize_subset(messages[:i],       prev_imgs,   prev_thw).shape[0] if i > 0 else 0

            loss_mask[start_pos:end_pos] = 1

        # ── 3-D RoPE position IDs ───────────────────────────────────────
        if image_grid_thw is not None:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )  # (3, seq_len)
        else:
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(3, 1)

        # ── Padding / Truncation ────────────────────────────────────────
        pad_id = getattr(self._tok, "pad_token_id", None) or 0

        if seq_len < self.max_length:
            pad = self.max_length - seq_len
            input_ids      = torch.cat([input_ids,    torch.full((pad,), pad_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad, dtype=torch.long)])
            loss_mask      = torch.cat([loss_mask,    torch.zeros(pad, dtype=torch.long)])
            position_ids   = torch.cat([position_ids, torch.zeros(3, pad, dtype=torch.long)], dim=1)

        elif seq_len > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"Sequence length {seq_len} > max_length {self.max_length}")
            elif self.truncation == "right":
                input_ids      = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask      = loss_mask[: self.max_length]
                position_ids   = position_ids[:, : self.max_length]
            else:  # left
                input_ids      = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask      = loss_mask[-self.max_length :]
                position_ids   = position_ids[:, -self.max_length :]

        # ── Build output dict ───────────────────────────────────────────
        out = {
            "input_ids":      input_ids,       # (max_length,)
            "attention_mask": attention_mask,   # (max_length,)
            "position_ids":   position_ids,     # (3, max_length)
            "loss_mask":      loss_mask,         # (max_length,)
        }
        if pixel_values is not None:
            out["pixel_values"]   = pixel_values    # (n_patches, patch_dim)
            out["image_grid_thw"] = image_grid_thw  # (1, 3)

        return out
