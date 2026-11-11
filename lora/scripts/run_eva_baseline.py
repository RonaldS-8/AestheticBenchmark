"""Run a prompt baseline (B0 / B1 / B2) on the EVA prompt-tuning set.

Every baseline scores the same images: all of EVA except the few-shot
examples, fixed in the tuning-set file (``scripts/prepare_prompt_tuning.py``).
Model id, revision and pixel budget come from the LoRA training config, so a
baseline differs from the LoRA eval (``scripts/eval_eva_sft.py``) only in
prompt and weights. Writes ``data/runs/<run_id>/``: ``config.json``,
``predictions.jsonl`` (appended per image, so the same command resumes),
``report.json`` over the tuning set and ``report-eval.json`` restricted to
the SFT eval split, for the comparison with LoRA.

Two runtimes:

- ``transformers`` (GPU, bf16, same loader as the LoRA eval; the chat
  template hash is checked against the config).
- ``llama_cpp`` (local pilot on a GGUF; starts llama-server, records the
  server's own chat-template hash and the GGUF hashes).

Example (cloud, B1)::

    python scripts/run_eva_baseline.py \\
        --config configs/eva-qwen35-4b-lora-r16.toml \\
        --tuning-set data/manifests/eva/prompt-tuning/v1.json \\
        --method b1_rubric --prompt-id rubric-score@v2 \\
        --run-id b1-qwen35-4b-tune --runtime transformers --device cuda

Example (local smoke test, B0 on the first 60 images)::

    python scripts/run_eva_baseline.py \\
        --config configs/eva-qwen35-4b-lora-r16.toml \\
        --tuning-set data/manifests/eva/prompt-tuning/v1.json \\
        --method b0_zero_shot --prompt-id zero-shot-score@v2 --limit 60 \\
        --run-id b0-qwen35-4b-q4-smoke60 --runtime llama_cpp \\
        --server ../OCR_test/llama.cpp/build/bin/llama-server.exe \\
        --model ../OCR_test/models/gguf/Qwen3.5-4B-Q4_K_M.gguf \\
        --mmproj ../OCR_test/models/gguf/mmproj-Qwen3.5-4B-f16.gguf \\
        --quantization q4_k_m --threads 8 --device cpu-core-ultra5-225u

B2 (``few-shot-score@v2``) shows the tuning set's examples automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lora_sft.crawl.eva_runner import SHANGHAI_TIMEZONE  # noqa: E402
from lora_sft.crawl.http import sha256_file  # noqa: E402
from lora_sft.data.samples import Message  # noqa: E402
from lora_sft.data.schema import eval_fingerprint  # noqa: E402
from lora_sft.eval.eva_baseline import (  # noqa: E402
    EvaBaselineRunConfig,
    PromptTuningSet,
    hf_chat_messages,
    openai_chat_messages,
    run_baseline,
    tuning_samples,
    write_baseline_report,
    write_eval_slice_report,
)
from lora_sft.eval.eva_eval import EvaPrediction  # noqa: E402
from lora_sft.eval.prompts import find_prompt  # noqa: E402
from lora_sft.train.config import LoraSftConfig, load_train_config  # noqa: E402
from lora_sft.train.eva_data import load_eva_samples, verify_images  # noqa: E402

Complete = Callable[[tuple[Message, ...]], str]
HEALTH_TIMEOUT_S = 600
REQUEST_TIMEOUT_S = 1800


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the baseline command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("b0_zero_shot", "b1_rubric", "b2_few_shot"), required=True
    )
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--tuning-set", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--runtime", choices=("transformers", "llama_cpp"), required=True
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--runs-root", type=Path, default=Path("data/runs"))
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="默认取训练配置的 selection.max_new_tokens，与 LoRA 评测一致",
    )
    parser.add_argument(
        "--limit", type=int, help="只评前 N 张，用于冒烟测试；不写 report-eval.json"
    )
    llama = parser.add_argument_group("llama_cpp")
    llama.add_argument("--server", type=Path)
    llama.add_argument("--model", type=Path)
    llama.add_argument("--mmproj", type=Path)
    llama.add_argument("--quantization")
    llama.add_argument("--threads", type=int, default=8)
    llama.add_argument("--port", type=int, default=8091)
    args = parser.parse_args(argv)
    llama_args = (args.server, args.model, args.mmproj, args.quantization)
    if args.runtime == "llama_cpp" and None in llama_args:
        parser.error("llama_cpp 需要 --server --model --mmproj --quantization")
    return args


def _post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        return json.loads(response.read())


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def _wait_healthy(base: str, server: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"llama-server 已退出，返回码 {server.returncode}")
        try:
            if _get_json(f"{base}/health").get("status") == "ok":
                return
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(2)
    raise TimeoutError("llama-server 启动超时")


@contextmanager
def llama_backend(
    args: argparse.Namespace, config: LoraSftConfig, log_path: Path
) -> Iterator[tuple[Complete, str]]:
    """Start llama-server; yield ``complete`` and the server's template hash."""

    base = f"http://127.0.0.1:{args.port}"
    command = [
        str(args.server),
        *("-m", str(args.model), "--mmproj", str(args.mmproj)),
        *("--host", "127.0.0.1", "--port", str(args.port)),
        *("-t", str(args.threads), "-tb", str(args.threads)),
        *("-c", "16384", "-np", "1", "--cache-ram", "0", "--ctx-checkpoints", "0"),
        *("--image-min-tokens", "16", "--image-max-tokens", "1024"),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait_healthy(base, server)
            template = _get_json(f"{base}/props")["chat_template"]

            def complete(messages: tuple[Message, ...]) -> str:
                reply = _post_json(
                    f"{base}/v1/chat/completions",
                    {
                        "messages": openai_chat_messages(
                            messages, args.data_root, config.preprocess
                        ),
                        "temperature": 0,
                        "seed": 0,
                        "max_tokens": args.max_new_tokens,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                return reply["choices"][0]["message"]["content"]

            yield complete, hashlib.sha256(template.encode()).hexdigest()
        finally:
            server.terminate()
            server.wait(timeout=60)


@contextmanager
def transformers_backend(
    args: argparse.Namespace, config: LoraSftConfig
) -> Iterator[tuple[Complete, str]]:
    """Load the pinned model as the LoRA eval does; yield ``complete``."""

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText

    from lora_sft.train.qwen_lora import image_size, load_processor

    processor = load_processor(config)
    model = AutoModelForImageTextToText.from_pretrained(
        config.base_model.model_id,
        revision=config.base_model.revision,
        dtype=torch.bfloat16 if config.runtime.precision == "bf16" else torch.float32,
    ).to(args.device)
    model.eval()

    def complete(messages: tuple[Message, ...]) -> str:
        chat, image_paths = hf_chat_messages(messages)
        text = processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        images = []
        for path in image_paths:
            with Image.open(args.data_root / path) as opened:
                images.append(opened.convert("RGB"))
        inputs = processor(
            text=[text],
            images=images,
            size=image_size(config.preprocess),
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    yield complete, config.chat_template_sha256


def main(argv: list[str] | None = None) -> int:
    """Score every remaining image, then write and print the report."""

    args = parse_args(argv)
    config = load_train_config(args.config)
    if not isinstance(config, LoraSftConfig):
        raise TypeError(f"{args.config} 不是 lora_sft 配置")
    if args.max_new_tokens is None:
        args.max_new_tokens = config.selection.max_new_tokens
    prompt = find_prompt(args.prompt_id)
    data_root = args.data_root.resolve()
    tuning_set = PromptTuningSet.model_validate_json(
        args.tuning_set.read_text(encoding="utf-8")
    )
    sft_manifest = data_root / tuning_set.sft_manifest
    if sha256_file(sft_manifest) != tuning_set.sft_manifest_sha256:
        raise ValueError(f"{sft_manifest} 与提示词调优集记录的哈希不符")
    splits = {
        name: load_eva_samples(sft_manifest.parent / f"{name}.jsonl", name)
        for name in ("train", "val", "eval")
    }
    items, all_examples = tuning_samples(
        tuning_set, splits["train"], splits["val"], splits["eval"]
    )
    samples = items if args.limit is None else items[: args.limit]
    examples = all_examples if prompt.is_few_shot else ()
    verify_images([*samples, *examples], data_root)
    run_dir = args.runs_root / args.run_id

    if args.runtime == "llama_cpp":
        backend = llama_backend(
            args, config, run_dir.with_name(f"{args.run_id}.server.log")
        )
    else:
        backend = transformers_backend(args, config)
    with backend as (complete, template_sha256):
        run_config = EvaBaselineRunConfig.model_validate(
            {
                "schema_version": "1.0",
                "scoring": {
                    "schema_version": "1.0",
                    "run_id": args.run_id,
                    "split": "all",
                    "ids_fingerprint": eval_fingerprint(s.image_id for s in samples),
                    "model": config.base_model,
                    "adapter": None,
                    "prompt_id": args.prompt_id,
                    "chat_template_sha256": template_sha256,
                    "preprocess": config.preprocess,
                    "max_new_tokens": args.max_new_tokens,
                    "device": args.device,
                },
                "tuning_set": {
                    "path": args.tuning_set.as_posix(),
                    "sha256": sha256_file(args.tuning_set),
                },
                "method": args.method,
                "runtime": args.runtime,
                "gguf": (
                    {
                        "model_file": args.model.name,
                        "model_sha256": sha256_file(args.model),
                        "mmproj_file": args.mmproj.name,
                        "mmproj_sha256": sha256_file(args.mmproj),
                        "quantization": args.quantization,
                    }
                    if args.runtime == "llama_cpp"
                    else None
                ),
            }
        )

        def progress(done: int, total: int, prediction: EvaPrediction) -> None:
            status = "ok" if prediction.scores is not None else "解析失败"
            print(
                f"[{done}/{total}] {prediction.image_id} {status} "
                f"{prediction.latency_ms / 1000:.1f}s",
                file=sys.stderr,
                flush=True,
            )

        count = run_baseline(
            run_config,
            prompt,
            tuning_set,
            samples,
            examples,
            run_dir=run_dir,
            complete=complete,
            on_progress=progress,
        )
    now = datetime.now(SHANGHAI_TIMEZONE)
    reports = [write_baseline_report(run_config, samples, run_dir=run_dir, now=now)]
    if args.limit is None:
        reports.append(
            write_eval_slice_report(
                run_config, tuning_set, splits["eval"], run_dir=run_dir, now=now
            )
        )
    print(f"本次评了 {count} 张")
    for report in reports:
        print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
