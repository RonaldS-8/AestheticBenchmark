"""LoRA SFT of a Qwen VLM on EVA samples with transformers, peft and torch.

This is the only training module that imports torch; install it with
``requirements-train.txt`` on the GPU machine. Everything that can be checked
without a model lives in ``lora_sft.train.eva_data``.

Train/inference parity is enforced, not assumed: the processor's chat template
must hash to ``chat_template_sha256``, the pixel budget comes from the config,
and the training sequence is ``generation prompt + reply + end-of-turn`` with
loss on the reply tokens only, exactly what the model sees when it generates.

Only the reply is supervised (about 15 tokens out of up to ~760), so the batch
carries ``logits_to_keep``: the positions whose next token is a reply token in
any row. The model projects just those hidden states onto the 248k vocabulary
and ``reply_loss`` scores them, instead of materialising logits (and their
fp32 copies and gradients) for every prompt and image token.

Two more things are checked rather than hoped for: the Gated DeltaNet kernels
the config requires must import (``check_kernels``), because transformers
otherwise falls back to pure torch without failing; and with
``selection.metric = "srcc_mean"`` every evaluation also decodes the val split
and the best checkpoint is the one with the highest mean SRCC, not the
lowest loss (``EvaSftTrainer``).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import time
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedModel,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.utils import ModelOutput

from lora_sft.data.eva_sft import EvaSftSample
from lora_sft.eval.eva_eval import SelectionScore, selection_score
from lora_sft.train.config import (
    ImagePreprocess,
    KernelSettings,
    LoraSettings,
    LoraSftConfig,
    SelectionSettings,
)
from lora_sft.train.eva_data import (
    IGNORE_INDEX,
    TrainDataError,
    chat_messages,
    completion_text,
    mask_labels,
)

TOKEN_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids")
IMAGE_KEYS = ("pixel_values", "image_grid_thw")
PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "peft",
    "accelerate",
    "flash-linear-attention",
    "causal-conv1d",
    "flash-attn",
)
# (package, module, function) exactly as transformers resolves the Qwen3.5
# Gated DeltaNet kernels; a failed import there means the pure-torch path.
FLA_KERNELS = (
    ("fla", "fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
    ("fla", "fla.ops.gated_delta_rule", "fused_recurrent_gated_delta_rule"),
)
CONV1D_KERNELS = (
    ("causal_conv1d", "causal_conv1d", "causal_conv1d_fn"),
    ("causal_conv1d", "causal_conv1d", "causal_conv1d_update"),
)

logger = logging.getLogger(__name__)


class TrainSetupError(RuntimeError):
    """Raised when the model, processor or runtime disagree with the config."""


def kernel_status(
    kernels: Sequence[tuple[str, str, str]] = (*FLA_KERNELS, *CONV1D_KERNELS),
) -> dict[str, str | None]:
    """``module.function -> None`` if importable, else why it is not."""

    status: dict[str, str | None] = {}
    for _package, module_name, function in kernels:
        key = f"{module_name}.{function}"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # any import failure means the slow path
            status[key] = f"{type(exc).__name__}: {exc}"
            continue
        if callable(getattr(module, function, None)):
            status[key] = None
        else:
            status[key] = f"{module_name} 里没有 {function}"
    return status


def check_kernels(settings: KernelSettings) -> dict[str, str | None]:
    """Fail unless every kernel the config requires imports; return all states."""

    required = [
        *(FLA_KERNELS if settings.require_fla else ()),
        *(CONV1D_KERNELS if settings.require_causal_conv1d else ()),
    ]
    status = kernel_status()
    missing = {
        f"{module}.{function}": status[f"{module}.{function}"]
        for _package, module, function in required
        if status[f"{module}.{function}"] is not None
    }
    if missing:
        raise TrainSetupError(
            f"配置要求的加速算子不可用: {missing}；transformers 会静默退回纯 torch "
            "慢路径，请先按 docs/cloud-training.md 安装"
        )
    return status


def preflight_kernels(settings: KernelSettings) -> dict[str, Any]:
    """Kernel check for preflight, strict wherever training could run.

    A machine without a CUDA device (the dev machine) cannot train, so
    preflight there only reports the kernel states. ``check_kernels`` before
    training is always strict.
    """

    if torch.cuda.is_available():
        return {"enforced": True, "status": check_kernels(settings)}
    return {
        "enforced": False,
        "reason": "无 CUDA 设备，这台机器不能训练，只报告内核状态",
        "status": kernel_status(),
    }


def image_size(preprocess: ImagePreprocess) -> dict[str, int]:
    """Processor ``size`` for the Qwen image processor (pixel counts)."""

    return {
        "shortest_edge": preprocess.min_pixels,
        "longest_edge": preprocess.max_pixels,
    }


def load_processor(config: LoraSftConfig) -> ProcessorMixin:
    """Load the pinned processor and prove its chat template matches the config."""

    processor = AutoProcessor.from_pretrained(
        config.base_model.model_id, revision=config.base_model.revision
    )
    template = processor.chat_template
    if not isinstance(template, str):
        raise TrainSetupError("处理器没有字符串形式的 chat template")
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if digest != config.chat_template_sha256:
        raise TrainSetupError(
            f"chat template 哈希 {digest} 与配置 {config.chat_template_sha256} 不符"
        )
    if processor.tokenizer.pad_token_id is None:
        raise TrainSetupError("分词器没有 pad_token_id，无法补齐批次")
    return processor


def render_prompt(processor: ProcessorMixin, sample: EvaSftSample) -> str:
    """The generation prompt exactly as used at inference (thinking off)."""

    return processor.apply_chat_template(
        chat_messages(sample, include_reply=False),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def open_image(data_root: Path, sample: EvaSftSample) -> Image.Image:
    """Decode the sample's image as RGB."""

    with Image.open(data_root / sample.image_path) as opened:
        return opened.convert("RGB")


class EvaSftCollator:
    """Encode EVA samples into a padded batch with reply-only labels."""

    def __init__(
        self,
        processor: ProcessorMixin,
        data_root: Path,
        preprocess: ImagePreprocess,
        max_seq_len: int,
    ) -> None:
        self.processor = processor
        self.data_root = data_root
        self.size = image_size(preprocess)
        self.max_seq_len = max_seq_len

    def encode(self, sample: EvaSftSample) -> dict[str, torch.Tensor]:
        """One sample as unbatched tensors plus ``labels``."""

        prompt = render_prompt(self.processor, sample)
        full = self.processor.apply_chat_template(
            chat_messages(sample, include_reply=True),
            tokenize=False,
            enable_thinking=False,
        )
        tail = completion_text(prompt, full, sample.messages[-1].text)
        encoded = self.processor(
            text=[full],
            images=[open_image(self.data_root, sample)],
            size=self.size,
            return_tensors="pt",
        )
        unknown = set(encoded.keys()) - {*TOKEN_KEYS, *IMAGE_KEYS}
        if unknown:
            raise TrainSetupError(f"处理器输出了未处理的字段: {sorted(unknown)}")
        input_ids = encoded["input_ids"][0]
        if input_ids.numel() > self.max_seq_len:
            raise TrainDataError(
                f"{sample.sample_id} 长度 {input_ids.numel()} 超过 "
                f"max_seq_len={self.max_seq_len}；不截断，请调整像素预算或长度"
            )
        completion_ids = self.processor.tokenizer(tail, add_special_tokens=False)[
            "input_ids"
        ]
        labels = mask_labels(input_ids.tolist(), completion_ids)
        item = {key: encoded[key][0] for key in TOKEN_KEYS if key in encoded}
        item.update({key: encoded[key] for key in IMAGE_KEYS})
        item["labels"] = torch.tensor(labels, dtype=torch.long)
        return item

    def __call__(self, samples: Sequence[EvaSftSample]) -> dict[str, torch.Tensor]:
        items = [self.encode(sample) for sample in samples]
        longest = max(item["input_ids"].numel() for item in items)
        pad_values = {
            "input_ids": self.processor.tokenizer.pad_token_id,
            "attention_mask": 0,
            "mm_token_type_ids": 0,
            "labels": IGNORE_INDEX,
        }
        batch: dict[str, torch.Tensor] = {}
        for key, pad in pad_values.items():
            if key not in items[0]:
                continue
            rows = []
            for item in items:
                row = item[key]
                filler = row.new_full((longest - row.numel(),), pad)
                rows.append(torch.cat([row, filler]))
            batch[key] = torch.stack(rows)
        for key in IMAGE_KEYS:
            batch[key] = torch.cat([item[key] for item in items])
        batch["logits_to_keep"] = reply_logit_positions(batch["labels"])
        return batch


def reply_logit_positions(labels: torch.Tensor) -> torch.Tensor:
    """Positions ``p`` where ``labels[:, p + 1]`` is supervised in any row.

    Logits at ``p`` predict token ``p + 1``; these are the only logits the
    reply loss reads. Shared by the whole batch, as ``logits_to_keep`` wants.
    """

    positions = (labels[:, 1:] != IGNORE_INDEX).any(dim=0).nonzero().squeeze(1)
    if positions.numel() == 0:
        raise TrainDataError("批次里没有任何需要监督的回复 token")
    return positions


def reply_loss(
    outputs: ModelOutput,
    labels: torch.Tensor,
    num_items_in_batch: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Causal LM loss on logits computed only at ``reply_logit_positions``.

    Equal to the transformers loss on full logits: fp32 cross entropy summed
    over supervised tokens, divided by ``num_items_in_batch`` (the token count
    across gradient-accumulation steps) or by this batch's count.
    """

    positions = reply_logit_positions(labels).to(labels.device)
    targets = labels[:, positions + 1]
    logits = outputs.logits
    if logits.shape[:2] != targets.shape:
        raise TrainSetupError(
            f"logits 形状 {tuple(logits.shape)} 与回复位置 {tuple(targets.shape)} "
            "不一致；模型没有按 logits_to_keep 只算回复位置"
        )
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1).to(logits.device),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    if num_items_in_batch is None:
        return loss / (targets != IGNORE_INDEX).sum()
    if isinstance(num_items_in_batch, torch.Tensor):
        num_items_in_batch = num_items_in_batch.to(loss.device)
    return loss / num_items_in_batch


def lora_target_pattern(settings: LoraSettings) -> str:
    """PEFT regex matching the target modules, language model only if frozen."""

    names = "|".join(settings.target_modules)
    scope = r"model\.language_model\." if settings.freeze_vision else r"model\."
    return rf"^{scope}.*\.({names})$"


def load_lora_model(config: LoraSftConfig) -> PeftModel:
    """Load the pinned base model in the configured precision and add LoRA."""

    dtype = torch.bfloat16 if config.runtime.precision == "bf16" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        config.base_model.model_id,
        revision=config.base_model.revision,
        dtype=dtype,
        attn_implementation=config.runtime.attn_implementation,
    )
    return attach_lora(model, config)


def attach_lora(model: PreTrainedModel, config: LoraSftConfig) -> PeftModel:
    """Wrap ``model`` with the configured LoRA adapter; fail if nothing matched."""

    model.config.use_cache = False
    if config.runtime.gradient_checkpointing:
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=lora_target_pattern(config.lora),
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora)
    wrapped = {
        name.rsplit(".", 1)[-1]
        for name, module in peft_model.named_modules()
        if hasattr(module, "lora_A")
    }
    missing = set(config.lora.target_modules) - wrapped
    if missing:
        raise TrainSetupError(f"LoRA 目标模块在模型里不存在: {sorted(missing)}")
    return peft_model


def training_arguments(
    config: LoraSftConfig, *, output_dir: Path, num_workers: int
) -> TrainingArguments:
    """Map the run config one-to-one onto ``TrainingArguments``."""

    if config.runtime.strategy == "fsdp":
        raise TrainSetupError("当前脚手架只支持 single / ddp，fsdp 尚未实现")
    by_srcc = config.selection.metric == "srcc_mean"
    optim = config.optim
    return TrainingArguments(
        output_dir=str(output_dir),
        seed=config.seed,
        data_seed=config.seed,
        learning_rate=optim.learning_rate,
        num_train_epochs=optim.epochs,
        per_device_train_batch_size=optim.per_device_batch_size,
        per_device_eval_batch_size=optim.per_device_batch_size,
        gradient_accumulation_steps=optim.grad_accum_steps,
        warmup_steps=optim.warmup_ratio,
        weight_decay=optim.weight_decay,
        lr_scheduler_type=optim.lr_scheduler,
        max_grad_norm=optim.max_grad_norm,
        bf16=config.runtime.precision == "bf16",
        gradient_checkpointing=config.runtime.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=config.runtime.eval_steps,
        save_strategy="steps",
        save_steps=config.runtime.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_srcc_mean" if by_srcc else "eval_loss",
        greater_is_better=by_srcc,
        logging_steps=10,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=num_workers,
        ddp_find_unused_parameters=False,
        report_to="none",
    )


class JsonlLogCallback(TrainerCallback):
    """Append every Trainer log entry to ``metrics.jsonl`` on the main process."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: object,
    ) -> None:
        if not state.is_world_process_zero or logs is None:
            return
        entry = {"step": state.global_step, "epoch": state.epoch, **logs}
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def environment_info() -> dict[str, Any]:
    """Package versions and devices, recorded with every run."""

    info: dict[str, Any] = {
        name: metadata.version(name) for name in PACKAGES if _installed(name)
    }
    info["cuda"] = torch.version.cuda
    info["devices"] = [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ]
    info["linear_attention_kernels"] = kernel_status()
    return info


def _installed(name: str) -> bool:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def build_trainer(
    config: LoraSftConfig,
    model: PeftModel,
    processor: ProcessorMixin,
    train: Sequence[EvaSftSample],
    val: Sequence[EvaSftSample],
    *,
    data_root: Path,
    output_dir: Path,
    num_workers: int,
) -> Trainer:
    """Assemble the Trainer; samples are encoded lazily by the collator."""

    collator = EvaSftCollator(
        processor, data_root, config.preprocess, config.max_seq_len
    )
    return EvaSftTrainer(
        model=model,
        args=training_arguments(config, output_dir=output_dir, num_workers=num_workers),
        data_collator=collator,
        train_dataset=list(train),
        eval_dataset=list(val),
        processing_class=processor,
        compute_loss_func=reply_loss,
        callbacks=[JsonlLogCallback(output_dir / "metrics.jsonl")],
        selection=config.selection,
        score_samples=list(val),
        data_root=data_root,
        preprocess=config.preprocess,
    )


class EvaSftTrainer(Trainer):
    """Trainer whose evaluation can also score generated val replies.

    With ``selection.metric = "srcc_mean"``, ``evaluate`` adds
    ``eval_srcc_mean`` (the selection metric), the per-dimension SRCC and the
    parse rate to the loss metrics, so ``load_best_model_at_end`` keeps the
    checkpoint that ranks val images most like the voters. Under DDP each
    rank decodes a strided shard and the replies are gathered.
    """

    def __init__(
        self,
        *args: object,
        selection: SelectionSettings,
        score_samples: Sequence[EvaSftSample],
        data_root: Path,
        preprocess: ImagePreprocess,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.selection = selection
        self.score_samples = list(score_samples)
        self.data_root = data_root
        self.preprocess = preprocess

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if self.selection.metric != "srcc_mean":
            return metrics
        result = self.score_replies()
        extra = {
            f"{metric_key_prefix}_srcc_mean": result.score,
            f"{metric_key_prefix}_parse_rate": result.parse_success_rate,
            **{f"{metric_key_prefix}_srcc_{dim}": v for dim, v in result.srcc.items()},
        }
        self.log(extra)
        if result.ineligible and self.is_world_process_zero():
            logger.warning(
                "第 %s 步不能入选: %s", self.state.global_step, result.ineligible
            )
        metrics.update(extra)
        return metrics

    def score_replies(self) -> SelectionScore:
        """Greedy-decode ``score_samples`` with the current weights and score them."""

        model = self.model
        was_training = model.training
        model.eval()
        rank, world = self.args.process_index, self.args.world_size
        replies = [
            (
                sample.image_id,
                generate_reply(
                    model,
                    self.processing_class,
                    sample,
                    data_root=self.data_root,
                    preprocess=self.preprocess,
                    max_new_tokens=self.selection.max_new_tokens,
                )[0],
            )
            for sample in self.score_samples[rank::world]
        ]
        if world > 1:
            from accelerate.utils import gather_object

            replies = gather_object(replies)
        if was_training:
            model.train()
        return selection_score(
            dict(replies),
            self.score_samples,
            min_parse_rate=self.selection.min_parse_rate,
        )


def generate_reply(
    model: PreTrainedModel,
    processor: ProcessorMixin,
    sample: EvaSftSample,
    *,
    data_root: Path,
    preprocess: ImagePreprocess,
    max_new_tokens: int,
) -> tuple[str, float]:
    """Greedy-decode one reply; return text and latency in milliseconds.

    ``use_cache`` is forced on: training sets ``config.use_cache = False``
    for gradient checkpointing, and decoding without a cache would recompute
    the whole prompt for every new token.
    """

    inputs = processor(
        text=[render_prompt(processor, sample)],
        images=[open_image(data_root, sample)],
        size=image_size(preprocess),
        return_tensors="pt",
    ).to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
    latency_ms = (time.perf_counter() - started) * 1000
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, latency_ms
