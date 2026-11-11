# 提示词调优（独立上云包）

在 EVA 图片上跑 B0 / B1 / B2 三种提示词，选出最佳提示词。
这个文件夹可以单独打包上云：代码、提示词、依赖、启动脚本都在这里。
数据随包上传，权重在云端下载。决定和流程见主仓库的 ADR-0011、
ADR-0012 和 `prompts/baseline-v2/README.md`。

## 和主仓库的关系

- 与 SFT / LoRA 共用的部分（输出格式、解析、指标、报告、提示词）
  **复制了一份**到这里，主仓库的源文件一个都没改。
- 复制的部分必须和主仓库一致：`tests/test_sync_with_main.py`
  逐项比对记录格式、提示词文件、消息组装和报告结果；
  打包时还会再检查一遍提示词文件是否逐字节相同。
  主仓库的共享部分改了，这里要跟着改。
- 这些核对**只在本机打包前做**。上传包里没有主仓库，也没有
  `tests/` 和 `pack.py`；云端只核对包内自洽（图片、清单、调优集
  指纹）和权重哈希，不再和主仓库比对。
- 输出的 `config.json`、`report.json`、`report-eval.json`、
  `selection.json` 与主仓库格式相同，下载回来可以直接用
  `scripts/report_lora_gain.py` 算 LoRA 的提升。

## 目录

```text
ptune/                  代码：schema、提示词、运行器、指标、权重下载、推理
prompts/baseline-v2/    三份提示词（与主仓库逐字节相同）
download_model.py       下载并核对权重
run_baseline.py         跑一种提示词
select_best_prompt.py   选最佳提示词，打印对比表
setup.sh / run.sh       云端建环境 / 一键跑完
requirements.txt        只有推理需要的依赖
pack.py                 本机打包（不上云）
tests/                  测试（不上云）
```

## 1. 本机打包

在主仓库根目录运行，先提交改动再打包：

```bash
python prompt_tuning/pack.py \
    --config configs/eva-qwen35-4b-lora-r16.toml \
    --tuning-set data/manifests/eva/prompt-tuning/v1.json \
    --weights-from ../OCR_test/models/Qwen3.5-4B \
    --output dist/prompt-tuning-v1.tar
```

- 包里有：代码、`data/job.json`、`data/tuning-set.json`、
  `data/items.jsonl`、4070 张图片（4067 张待评 + 3 张示例），
  约 570 MB。权重**不打包**。
- `job.json` 里的模型、revision、对话模板哈希、像素预算、精度、
  `max_new_tokens`、解析率门槛都取自 LoRA 训练配置，保证和 LoRA 评测
  只差提示词和权重。
- `--weights-from` 只用来算每个权重文件的 sha256，写进 `job.json`。
  这份本地副本的 11 个文件已逐个核对，与 Hugging Face
  `Qwen/Qwen3.5-4B` 锁定的 revision `851bf6e` 完全一致。
- 加 `--dry-run` 只检查和统计，不写文件。

## 2. 上传

把 tar 传到云主机（scp，或先传到同区域的阿里云 OSS 再内网拉取），
然后解压：

```bash
tar -xf prompt-tuning-v1.tar
cd prompt-tuning
```

## 3. 云端运行

```bash
bash setup.sh
SOURCE=hf HF_ENDPOINT=https://hf-mirror.com bash run.sh
```

- `setup.sh`：建 `.venv`，装 `requirements.txt`，Linux 上检查
  CUDA 12 和两个线性注意力加速内核（与训练用的相同）。
- `run.sh`：依次下载权重、跑 B0、B1、B2，最后选最佳提示词。
  中断后直接重跑：已评的图会跳过，已下载完的权重只复查。
- 环境变量：

| 变量 | 说明 |
| --- | --- |
| `SOURCE` | 必填：`hf` 或 `modelscope`，不会自动切换 |
| `HF_ENDPOINT` | `SOURCE=hf` 时可设镜像，如 `https://hf-mirror.com` |
| `DEVICE` | 默认 `cuda` |
| `TAG` | 运行名中缀，默认 `qwen35-4b` |

### 权重下载

- 按 `job.json` 里的文件清单逐个下载到 `models/Qwen--Qwen3.5-4B/`，
  支持断点续传；每个文件都要和清单里的 sha256 一致，
  不一致就删掉并报错。
- `hf`：按锁定的 revision 下载。阿里云上一般要设 `HF_ENDPOINT` 走镜像。
- `modelscope`：从 modelscope.cn 下载。它的版本号和 HF 不同，
  靠 sha256 证明文件就是锁定的那一版。
- 下载完写 `DOWNLOAD.json`；推理只从这个目录加载，不联网。

### 耗时参考

A10 上 B0 / B1 各约半小时，B2 每张图带 3 张示例，约 1–2 小时；
权重约 9.3 GB，下载时间取决于镜像速度。

## 4. 结果

```text
runs/
  b0-qwen35-4b-tune/  config.json  predictions.jsonl
                      report.json（4067 张）  report-eval.json（407 张）
  b1-…  b2-…
  selection.json      最佳提示词与 LoRA 要用的提示词
logs/                 每一步的日志
```

把 `runs/` 整个下载回本机：各运行目录放进 `data/runs/`，
`selection.json` 放到 `data/manifests/eva/prompt-tuning/v1-selection.json`。
之后按 `prompts/baseline-v2/README.md` 的第 4 步算 LoRA 提升。

## 冒烟测试

只想先确认能跑通：

```bash
python download_model.py --source hf
python run_baseline.py --method b0_zero_shot --prompt-id zero-shot-score@v2 \
    --run-id b0-smoke --device cuda --limit 20
```

`--limit` 的运行只写 `report.json`，不能用来选提示词。

## 开发与测试

在主仓库里：

```bash
cd prompt_tuning
pytest
```

lint 和格式化跟主仓库一样，在主仓库根目录跑 `ruff check .` 和 `black .`。
