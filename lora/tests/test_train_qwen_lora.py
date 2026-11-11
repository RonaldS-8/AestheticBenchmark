"""GPU-stack tests; skipped unless ``requirements-train.txt`` is installed.

A tiny randomly initialised Qwen3.5 model stands in for the real one, so no
weights are downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from lora_sft.data.eva_sft import EvaSftSample  # noqa: E402
from lora_sft.train.config import LoraSftConfig, load_train_config  # noqa: E402
from lora_sft.train.eva_data import IGNORE_INDEX  # noqa: E402
from lora_sft.train.qwen_lora import (  # noqa: E402
    EvaSftCollator,
    EvaSftTrainer,
    JsonlLogCallback,
    TrainSetupError,
    attach_lora,
    check_kernels,
    image_size,
    kernel_status,
    lora_target_pattern,
    reply_logit_positions,
    reply_loss,
    training_arguments,
)
from tests.test_train_eva_data import SAMPLES, repo  # noqa: E402, F401

ROOT = Path(__file__).resolve().parents[1]
EVA_CONFIG = ROOT / "configs" / "eva-qwen35-4b-lora-r16.toml"


def eva_config(**runtime: object) -> LoraSftConfig:
    config = load_train_config(EVA_CONFIG)
    assert isinstance(config, LoraSftConfig)
    if runtime:
        updated = config.runtime.model_copy(update=runtime)
        config = config.model_copy(update={"runtime": updated})
    return config


def tiny_qwen35() -> torch.nn.Module:
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    config = Qwen3_5Config(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "vocab_size": 512,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 64,
        },
    )
    return Qwen3_5ForConditionalGeneration(config)


def test_target_pattern_scopes_to_language_model() -> None:
    config = eva_config()
    pattern = lora_target_pattern(config.lora)

    assert pattern.startswith(r"^model\.language_model\.")
    unfrozen = config.lora.model_copy(update={"freeze_vision": False})
    assert lora_target_pattern(unfrozen).startswith(r"^model\.")


def test_attach_lora_wraps_both_attention_kinds_only_in_text_model() -> None:
    model = attach_lora(tiny_qwen35(), eva_config())

    wrapped = [n for n, m in model.named_modules() if hasattr(m, "lora_A")]
    suffixes = {name.rsplit(".", 1)[-1] for name in wrapped}

    assert {"in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "down_proj"} <= suffixes
    assert all(".language_model." in name for name in wrapped)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)


def test_attach_lora_rejects_unknown_target() -> None:
    config = eva_config()
    lora = config.lora.model_copy(
        update={"target_modules": (*config.lora.target_modules, "nonexistent_proj")}
    )

    with pytest.raises(TrainSetupError, match="nonexistent_proj"):
        attach_lora(tiny_qwen35(), config.model_copy(update={"lora": lora}))


def test_training_arguments_follow_config(tmp_path: Path) -> None:
    # bf16 needs a GPU; the CPU test env checks the mapping in fp32.
    config = eva_config(precision="fp32")

    args = training_arguments(config, output_dir=tmp_path, num_workers=0)

    assert args.per_device_train_batch_size == config.optim.per_device_batch_size
    assert args.gradient_accumulation_steps == config.optim.grad_accum_steps
    assert args.warmup_steps == config.optim.warmup_ratio
    assert args.metric_for_best_model == "eval_srcc_mean"
    assert args.greater_is_better is True
    assert args.eval_steps == args.save_steps == config.runtime.eval_steps
    assert args.remove_unused_columns is False


def test_training_arguments_select_by_loss(tmp_path: Path) -> None:
    config = eva_config(precision="fp32")
    selection = config.selection.model_copy(update={"metric": "eval_loss"})

    args = training_arguments(
        config.model_copy(update={"selection": selection}),
        output_dir=tmp_path,
        num_workers=0,
    )

    assert args.metric_for_best_model == "eval_loss"
    assert args.greater_is_better is False


def test_training_arguments_reject_fsdp(tmp_path: Path) -> None:
    with pytest.raises(TrainSetupError, match="fsdp"):
        training_arguments(
            eva_config(strategy="fsdp", num_gpus=2), output_dir=tmp_path, num_workers=0
        )


# --- Gated DeltaNet kernels ---


def fake_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> None:
    import sys
    import types

    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def test_kernel_status_reports_import_and_missing_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module(monkeypatch, "fake_kernels", good=lambda: None)

    status = kernel_status(
        (
            ("fake_kernels", "fake_kernels", "good"),
            ("fake_kernels", "fake_kernels", "absent"),
            ("nope", "definitely_not_installed_pkg", "fn"),
        )
    )

    assert status["fake_kernels.good"] is None
    assert "没有 absent" in str(status["fake_kernels.absent"])
    assert "ModuleNotFoundError" in str(status["definitely_not_installed_pkg.fn"])


def test_check_kernels_requires_fla_only_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lora_sft.train import qwen_lora

    status = {f"{m}.{f}": "ModuleNotFoundError: x" for _, m, f in qwen_lora.FLA_KERNELS}
    status |= {f"{m}.{f}": None for _, m, f in qwen_lora.CONV1D_KERNELS}
    monkeypatch.setattr(qwen_lora, "kernel_status", lambda: status)
    required = eva_config().kernels

    with pytest.raises(TrainSetupError, match="chunk_gated_delta_rule"):
        check_kernels(required)
    relaxed = required.model_copy(update={"require_fla": False})
    assert check_kernels(relaxed) == status


def test_preflight_kernels_strict_only_with_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lora_sft.train import qwen_lora

    status = {f"{m}.{f}": "ModuleNotFoundError: x" for _, m, f in qwen_lora.FLA_KERNELS}
    status |= {f"{m}.{f}": None for _, m, f in qwen_lora.CONV1D_KERNELS}
    monkeypatch.setattr(qwen_lora, "kernel_status", lambda: status)
    settings = eva_config().kernels

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = qwen_lora.preflight_kernels(settings)
    assert report["enforced"] is False and report["status"] == status

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(TrainSetupError, match="chunk_gated_delta_rule"):
        qwen_lora.preflight_kernels(settings)


def test_check_kernels_passes_with_all_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module(monkeypatch, "fla")
    fake_module(monkeypatch, "fla.ops")
    fake_module(
        monkeypatch,
        "fla.ops.gated_delta_rule",
        chunk_gated_delta_rule=lambda: None,
        fused_recurrent_gated_delta_rule=lambda: None,
    )
    fake_module(
        monkeypatch,
        "causal_conv1d",
        causal_conv1d_fn=lambda: None,
        causal_conv1d_update=lambda: None,
    )
    settings = eva_config().kernels
    assert settings.require_fla and settings.require_causal_conv1d

    status = check_kernels(settings)

    assert set(status.values()) == {None}


def test_check_kernels_requires_causal_conv1d_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lora_sft.train import qwen_lora

    status = {f"{m}.{f}": None for _, m, f in qwen_lora.FLA_KERNELS}
    status |= {
        f"{m}.{f}": "ModuleNotFoundError: x" for _, m, f in qwen_lora.CONV1D_KERNELS
    }
    monkeypatch.setattr(qwen_lora, "kernel_status", lambda: status)

    with pytest.raises(TrainSetupError, match="causal_conv1d_fn"):
        check_kernels(eva_config().kernels)


def test_image_size_uses_pixel_budget() -> None:
    assert image_size(eva_config().preprocess) == {
        "shortest_edge": 65536,
        "longest_edge": 1048576,
    }


class StubTokenizer:
    pad_token_id = 0


class StubProcessor:
    tokenizer = StubTokenizer()


def test_collator_pads_tokens_and_concatenates_images() -> None:
    collator = EvaSftCollator(StubProcessor(), Path("."), eva_config().preprocess, 64)
    items = {
        "a": {
            "input_ids": torch.tensor([5, 6, 7]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "mm_token_type_ids": torch.tensor([0, 1, 0]),
            "labels": torch.tensor([IGNORE_INDEX, IGNORE_INDEX, 7]),
            "pixel_values": torch.ones(4, 3),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
        "b": {
            "input_ids": torch.tensor([8, 9]),
            "attention_mask": torch.tensor([1, 1]),
            "mm_token_type_ids": torch.tensor([1, 0]),
            "labels": torch.tensor([IGNORE_INDEX, 9]),
            "pixel_values": torch.zeros(8, 3),
            "image_grid_thw": torch.tensor([[1, 2, 4]]),
        },
    }
    collator.encode = items.__getitem__  # type: ignore[method-assign]

    batch = collator(["a", "b"])  # type: ignore[list-item]

    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 9, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"][1].tolist() == [IGNORE_INDEX, 9, IGNORE_INDEX]
    assert batch["pixel_values"].shape == (12, 3)
    assert batch["image_grid_thw"].tolist() == [[1, 2, 2], [1, 2, 4]]
    # next-token targets: row a supervises position 2, row b position 1
    assert batch["logits_to_keep"].tolist() == [0, 1]


# --- reply-only logits ---

IG = IGNORE_INDEX


def test_reply_logit_positions_union_of_rows() -> None:
    labels = torch.tensor([[IG, IG, IG, 4, 5], [IG, IG, 6, IG, IG]])

    assert reply_logit_positions(labels).tolist() == [1, 2, 3]


def test_reply_logit_positions_rejects_unsupervised_batch() -> None:
    with pytest.raises(Exception, match="没有任何需要监督"):
        reply_logit_positions(torch.full((2, 4), IG))


def causal_out(logits: torch.Tensor) -> object:
    from transformers.modeling_outputs import CausalLMOutputWithPast

    return CausalLMOutputWithPast(logits=logits)


def full_loss(
    logits: torch.Tensor, labels: torch.Tensor, n: int | None
) -> torch.Tensor:
    from transformers.loss.loss_utils import ForCausalLMLoss

    return ForCausalLMLoss(logits, labels, logits.shape[-1], num_items_in_batch=n)


@pytest.mark.parametrize("num_items", [None, 7])
def test_reply_loss_matches_full_logit_loss(num_items: int | None) -> None:
    torch.manual_seed(0)
    labels = torch.tensor([[IG, IG, IG, 4, 5, 1], [IG, IG, 6, 2, IG, IG]])
    logits = torch.randn(2, 6, 11, dtype=torch.bfloat16)
    kept = reply_logit_positions(labels)

    got = reply_loss(causal_out(logits[:, kept]), labels, num_items_in_batch=num_items)

    want = full_loss(logits, labels, num_items)
    assert got.item() == pytest.approx(want.item(), rel=1e-6)


def test_reply_loss_rejects_full_logits() -> None:
    labels = torch.tensor([[IG, IG, 3, 4]])

    with pytest.raises(TrainSetupError, match="logits_to_keep"):
        reply_loss(causal_out(torch.randn(1, 4, 8)), labels)


def test_model_honours_tensor_logits_to_keep() -> None:
    model = tiny_qwen35()
    ids = torch.randint(0, 512, (2, 6))
    kept = torch.tensor([1, 4])

    out = model(input_ids=ids, logits_to_keep=kept)
    full = model(input_ids=ids)

    assert out.logits.shape == (2, 2, 512)
    torch.testing.assert_close(out.logits, full.logits[:, kept])


def test_jsonl_callback_writes_main_process_logs(tmp_path: Path) -> None:
    from transformers import TrainerControl, TrainerState

    path = tmp_path / "metrics.jsonl"
    callback = JsonlLogCallback(path)
    state = TrainerState()
    state.global_step = 3
    state.epoch = 0.5

    callback.on_log(None, state, TrainerControl(), logs={"loss": 1.25})  # type: ignore[arg-type]
    callback.on_log(None, state, TrainerControl(), logs=None)  # type: ignore[arg-type]

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {"step": 3, "epoch": 0.5, "loss": 1.25}
    ]


# --- checkpoint selection by generated val SRCC ---


def selection_trainer(
    tmp_path: Path, samples: tuple[EvaSftSample, ...], metric: str = "srcc_mean"
) -> EvaSftTrainer:
    from transformers import TrainingArguments

    config = eva_config(precision="fp32")
    return EvaSftTrainer(
        model=tiny_qwen35(),
        args=TrainingArguments(
            output_dir=str(tmp_path), report_to="none", use_cpu=True
        ),
        selection=config.selection.model_copy(update={"metric": metric}),
        score_samples=samples,
        data_root=tmp_path,
        preprocess=config.preprocess,
    )


@pytest.fixture
def val_samples(repo: Path) -> tuple[EvaSftSample, ...]:  # noqa: F811
    from lora_sft.train.eva_data import load_eva_samples
    from tests.test_eval_eva_eval import with_varied_technical

    return with_varied_technical(
        load_eva_samples(repo / SAMPLES / "train.jsonl", "train")
    )


def fake_replies(
    monkeypatch: pytest.MonkeyPatch, replies: dict[str, str]
) -> list[tuple[bool, object]]:
    from lora_sft.train import qwen_lora

    calls: list[tuple[bool, object]] = []

    def generate(
        model: torch.nn.Module,
        processor: object,
        sample: EvaSftSample,
        **kwargs: object,
    ) -> tuple[str, float]:
        calls.append((model.training, kwargs["max_new_tokens"]))
        return replies[sample.image_id], 1.0

    monkeypatch.setattr(qwen_lora, "generate_reply", generate)
    monkeypatch.setattr(
        transformers.Trainer, "evaluate", lambda self, **kwargs: {"eval_loss": 0.5}
    )
    return calls


def test_evaluate_adds_generated_srcc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    val_samples: tuple[EvaSftSample, ...],
) -> None:
    from tests.test_eval_eva_eval import tracking_replies

    calls = fake_replies(monkeypatch, tracking_replies(val_samples))
    trainer = selection_trainer(tmp_path, val_samples)
    trainer.model.train()

    metrics = trainer.evaluate()

    assert metrics["eval_loss"] == 0.5
    assert metrics["eval_srcc_technical"] == pytest.approx(1.0)
    assert metrics["eval_srcc_mean"] > 0.9
    assert metrics["eval_parse_rate"] == 1.0
    # decoded every val image in eval mode, then went back to training
    budget = eva_config().selection.max_new_tokens
    assert calls == [(False, budget)] * len(val_samples)
    assert trainer.model.training


def test_evaluate_marks_unparsable_checkpoint_ineligible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    val_samples: tuple[EvaSftSample, ...],
) -> None:
    from lora_sft.eval.eva_eval import INELIGIBLE_SELECTION_SCORE

    fake_replies(monkeypatch, {s.image_id: "我觉得还行" for s in val_samples})

    metrics = selection_trainer(tmp_path, val_samples).evaluate()

    assert metrics["eval_srcc_mean"] == INELIGIBLE_SELECTION_SCORE
    assert metrics["eval_parse_rate"] == 0.0


def test_evaluate_by_loss_skips_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    val_samples: tuple[EvaSftSample, ...],
) -> None:
    calls = fake_replies(monkeypatch, {})

    metrics = selection_trainer(tmp_path, val_samples, "eval_loss").evaluate()

    assert metrics == {"eval_loss": 0.5}
    assert calls == []
