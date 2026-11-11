# Lora-SFT：照片美学模型训练

为 AI-NAS 本地智能体训练照片美学评分模型，最终在 NAS 上本地推理。
本文介绍现有技术选型和整体架构；开发约定见 `CLAUDE.md`，
关键决定的原因见 `docs/adr/`。

## 1. 目标与路线

| 路线 | 做法 | 输出 |
| --- | --- | --- |
| LoRA SFT | Qwen 视觉语言模型 + LoRA 微调 | JSON 分数（后续加评语） |
| 美学头 | 参考 DOVER，冻结视觉骨干，只训小回归头 | 只有分数 |

原则是**评测先行**：训练后的模型必须明显好过
“原版模型 + 同一提示词”（B0 基线）才能部署（ADR-0001、0002）。

当前阶段（ADR-0009）：用 EVA 数据对 `Qwen/Qwen3.5-4B` 做
**双分数弱监督 LoRA SFT**，只学 `aesthetic` 和 `technical` 两项
1–5 分，不补造 EVA 没有的总体分和评语。

## 2. 技术选型

| 环节 | 选型 | 理由 |
| --- | --- | --- |
| 基座模型 | Qwen3.5-4B，锁定 commit `851bf6e` | 图文模型，端侧可跑，可升 9B |
| 微调方式 | LoRA（peft），r=16，只训语言模型 | 显存小，适配器可单独导出 |
| 训练框架 | transformers `Trainer` + 自写 collator | 无需 TRL，损失遮罩完全可控 |
| 并行 | 单卡或 DDP（torchrun） | 4B 单卡即可；不用 DeepSpeed |
| 精度 | bf16 + 梯度检查点 | 24 GB 显卡也能跑 |
| 数据与配置 | pydantic 严格 schema、JSONL、TOML | 字段缺失直接报错，不兜底 |
| 评测指标 | SRCC / PLCC / MAE、解析成功率 | 与人工投票均值比较 |
| 端侧推理 | GGUF + llama.cpp 或 MLX；美学头 ONNX | NAS 本地运行 |

训练栈固定在 `requirements-train.txt`：torch 2.8.0、
torchvision 0.23.0、transformers 5.20.2、peft 0.21.2、
accelerate 1.15.0、flash-linear-attention 0.5.2，Linux 上另装
causal-conv1d 1.7.0 和 flash-attn 2.8.3.post2 的官方预编译包。
本机和云端共用同一个 `.venv`（Python 3.11–3.13），都用
`scripts/cloud/setup.sh` 建；本机是 CPU 版 torch，不能训练。

### Qwen3.5 的特殊点

Qwen3.5 是混合架构：每 4 层里 3 层是 Gated DeltaNet 线性注意力，
1 层是全注意力。因此 LoRA 目标模块同时包含：

- 全注意力：`q_proj`、`k_proj`、`v_proj`、`o_proj`
- 线性注意力：`in_proj_qkv`、`in_proj_z`、`out_proj`
- 所有层的 MLP：`gate_proj`、`up_proj`、`down_proj`

用正则把 LoRA 限定在 `model.language_model.` 下，视觉塔和投影层冻结。
云端默认安装 `flash-linear-attention` 加速线性注意力层；
缺了它 transformers 会静默走纯 torch 实现，所以 EVA 配置
（`kernels.require_fla = true`）在训练前检查，缺失直接报错。

## 3. 数据：EVA

来源：EVA（Explainable Visual Aesthetics，Kang 等 2020），
从 AVA 中选出的 4,070 张图、136,943 条过滤后投票，
每张图 30–46 票。仅限研究用途（无标准开源许可）。

| EVA 字段 | 量表 | 项目字段 |
| --- | --- | --- |
| `score` | 0–10 总体审美分 | `aesthetic` |
| `quality` | 1–4（很差/差/好/很好） | `technical` |
| 无 | — | `overall = null` |

每张图的投票均值线性映射到 1–5 并保留一位小数：

```text
target = 1 + 4 × (mean − scale_min) / (scale_max − scale_min)
```

EVA 还有 `visual`、`composition`、`semantic` 属性分、判断难度和
“影响因素”勾选，目前未使用。按 seed 42 划分
train 3,256 / val 407 / eval 407，eval 集指纹固定，永不进入训练。

本地实测的标注一致性（随机半分投票的 SRCC）：美学约 0.82、
技术约 0.72；这大致是模型 SRCC 的参考上限。

## 4. 架构

```text
crawl ──> data ──> train ──> eval ──> export
抓取原始数据  生成样本     LoRA 训练   打分与门禁   GGUF/MLX/ONNX
```

```text
lora_sft/
  crawl/      抓取与 schema：EVA 适配器、HTTP、数据集记录
  data/       样本构造：eva_sft.py 生成 Qwen 消息格式 JSONL 与划分
  train/      config.py      TOML 训练配置（LoRA / 美学头）
              eva_data.py    样本加载、划分与泄漏检查、标签遮罩（无 torch）
              qwen_lora.py   collator、LoRA 挂载、Trainer、生成（需 torch）
  eval/       eva_eval.py    EVA 运行配置、严格解析、指标报告
              metrics.py     SRCC / PLCC 等
  export/     导出清单 schema
configs/      训练配置，如 eva-qwen35-4b-lora-r16.toml
prompts/      版本化提示词；eva-sft-v1/ 为本阶段提示词
scripts/      命令行入口；cloud/ 为云端一键脚本
data/         不入库：images/、manifests/、runs/ 等（ADR-0007）
outputs/      不入库：训练输出（适配器、检查点、日志）
```

### 训练与推理一致性

这些约束在代码里强制检查，不一致直接报错：

- chat template 的 SHA-256 必须等于配置值（`a4aee8af…`）。
- 非思考模式 `enable_thinking=false`；训练序列 =
  推理时的生成提示 + 回复 + `<|im_end|>`，损失只算回复部分。
- 像素预算 `min_pixels` / `max_pixels` 训练与推理相同。
- 序列超过 `max_seq_len` 时报错，不截断。
- 训练前检查划分版本、eval 指纹，以及 train/val 与 eval
  在 id 和图片哈希上都不重叠。

### 评测

`scripts/eval_eva_sft.py` 对同一切分分别跑 B0（`--base`）和
LoRA（`--adapter`），输出到 `data/runs/<run_id>/`：
`config.json`、逐图追加的 `predictions.jsonl`（可断点续跑）、
`report.json`。回复按 JSON 严格解析，失败只记录不修补，
计入解析成功率；相关系数与未取整的 EVA 投票均值比较。

## 5. 使用流程

本机准备（无 GPU）：

```bash
python scripts/train_lora_sft.py --config configs/eva-qwen35-4b-lora-r16.toml \
    --eval-samples data/manifests/eva/sft/v1/eval.jsonl --dry-run
python scripts/pack_for_cloud.py --config configs/eva-qwen35-4b-lora-r16.toml \
    --output dist/eva-qwen35-4b-lora-r16.tar
```

打包只含代码和训练用到的数据（约 574 MB），不含 18 GB 的
`data/raw/`。上传解包后在云端：

```bash
bash scripts/cloud/setup.sh
bash scripts/cloud/train.sh configs/eva-qwen35-4b-lora-r16.toml
```

`train.sh` 依次执行：preflight（加载处理器、编码全部样本）→
B0 基线（val）→ 训练 → LoRA 评测（val）。
`EVAL_SPLIT=1` 时再跑 eval 集；`RESUME=1` 从检查点续训；
国内机器可设 `HF_ENDPOINT` 使用镜像。

默认超参：lr 1e-4、3 个 epoch、全局 batch 16
（每卡 4 × 累积 4）、cosine、warmup 5%，共 612 步；
每半个 epoch（102 步）在 val 上生成打分算 SRCC，最终保留
美学、技术平均 SRCC 最高的适配器。

## 6. 现状与待定

- 已完成：EVA 抓取与样本生成、训练/评测/打包/云端脚本，
  本机已通过 dry-run。
- 尚未在 GPU 上实跑；首次上云请先看 preflight 输出。
- 待定：是否扩到 Qwen3.5-9B、美学头骨干选型、门禁阈值、
  感知哈希去重；第二阶段接入项目人工标注，学习总体分和评语。
