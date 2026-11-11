"""Load the pinned Qwen model with transformers and answer one chat at a time.

Loads the model the same way the main repository's LoRA eval does (pinned
revision, bf16 or fp32, greedy decoding, thinking off), so a baseline differs
from the LoRA eval only in prompt and weights. Weights are read only from the
folder ``download_model.py`` filled and checked; nothing is fetched here.
torch and transformers are imported here only, so the rest of the job runs
without them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ptune.download import check_model_dir
from ptune.prompts import Message
from ptune.runner import Complete, TuningError, hf_chat_messages
from ptune.schema import JobConfig


def load_complete(config: JobConfig, root: Path, device: str) -> Complete:
    """Return ``complete(messages) -> reply text`` backed by the pinned model."""

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    folder = check_model_dir(config, root)
    data_root = root / "data"
    processor: Any = AutoProcessor.from_pretrained(folder, local_files_only=True)
    template = processor.chat_template
    if not isinstance(template, str):
        raise TuningError("处理器没有字符串形式的 chat template")
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if digest != config.chat_template_sha256:
        raise TuningError(
            f"chat template 哈希 {digest} 与配置 {config.chat_template_sha256} 不符"
        )
    model = AutoModelForImageTextToText.from_pretrained(
        folder,
        local_files_only=True,
        dtype=torch.bfloat16 if config.precision == "bf16" else torch.float32,
    ).to(device)
    model.eval()
    size = {
        "shortest_edge": config.preprocess.min_pixels,
        "longest_edge": config.preprocess.max_pixels,
    }

    def complete(messages: tuple[Message, ...]) -> str:
        chat, image_paths = hf_chat_messages(messages)
        text = processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        images = []
        for path in image_paths:
            with Image.open(data_root / path) as opened:
                images.append(opened.convert("RGB"))
        inputs = processor(
            text=[text], images=images, size=size, return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    return complete
