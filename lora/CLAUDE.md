# Lora-SFT：照片美学模型训练

> 本文件与 `AGENTS.md` 内容必须完全一致，改一个就同步另一个。
> `tests/test_agent_docs.py` 会检查。

## 项目目标

为 AI-NAS 本地智能体训练照片美学模型，最终在 NAS 上本地推理。
对比两条路线：

- **LoRA**：在 Qwen 视觉语言模型（4B 起步，后扩到 7B–9B）上做
  LoRA SFT，输出美学、技术、总体三项分数和一句评语（JSON）。
- **美学头**：参考 DOVER，冻结视觉骨干，只训练很小的美学、
  技术回归头，只输出分数，不生成评语。

任务形式：打分加一句评语，以打分为主；门禁只看分数指标。
当前只做 LoRA SFT 和美学头，DPO、PPO、全参 SFT 暂不做。

EVA 第一阶段已定为 `Qwen/Qwen3.5-4B` 双分数弱监督 LoRA SFT；
暂不接项目自建人工标注，不补造 EVA 缺失的总体分和评语，见 ADR-0009。

原则是评测先行：先跑不训练的基线（B0–B4），训练后的模型
必须明显好过 “原版模型 + 好的提示词” 才能部署。

**第一轮是 EVA 图片上的提示词调优**（ADR-0011、ADR-0012）：
不训练的 Qwen 加 `prompts/baseline-v2/` 的提示词（B0–B2），输出美学、
技术两个分数，在全量 EVA（去掉 B2 示例图）上与人工投票均分比
SRCC / PLCC，不分训练和测试。输出格式、解析、指标和报告与 LoRA 评测
完全相同。LoRA 用其中最佳的单图提示词训练，提升 = eval 集上 LoRA
减去同一提示词不加 LoRA。流程和例子见 `prompts/baseline-v2/README.md`。
视频与 DOVER 的旧对比已归档到 `archive/video-dover/`。
背景文档见 `docs/background/`；标注、样本、训练配置、评测、
导出的全部结构见 `docs/training-data-design.md`；
已定的关键决定及原因见 `docs/adr/`（ADR），新的重大决定也要写 ADR。

## 目录结构

```text
lora_sft/            Python 包
  crawl/             抓包与爬虫：记录 schema、站点适配器基类
    adapters/        每个站点一个 <Site>Adapter
  common.py          共用基类与字段类型
  data/              标注记录、数据划分、SFT 样本、美学头样本
  models/            LoRA 配置与 DOVER 式美学头
  train/             训练配置（LoRA SFT / 美学头）与训练循环
  eval/              评测运行、预测、报告、门禁
  export/            LoRA 合并转 GGUF/MLX、美学头转 ONNX
configs/             训练与评测配置
prompts/             提示词，按版本分文件夹；baseline-v2/ 为评测基线，
                     eva-sft-v1/ 为 SFT；格式与改版规则见各自 README.md
scripts/             命令行入口
docs/                背景文档、设计文档、ADR（adr/）、来源核对记录
data/                不入库，布局见 ADR-0007：raw/ 原样下载（只读），
                     images/ 照片，videos/ 统一命名的视频，
                     manifests/ 清单与划分，runs/ 评测运行，
                     review/ 客户盲评包；只增不改，不覆盖
tests/               测试；fixtures/ 放样例记录和抓包样本
archive/             用不上但留档的代码与提示词，不参与测试和 lint
```

## 常用命令

```bash
bash scripts/cloud/setup.sh         # 建共用 .venv，本机与云端同一套
PYTHON=<python3.11 路径> bash scripts/cloud/setup.sh   # 指定解释器
.venv\Scripts\Activate.ps1          # Windows PowerShell
source .venv/Scripts/activate       # Windows Git Bash
source .venv/bin/activate           # Linux / macOS
ruff check .
black .
pytest                              # 含分支覆盖率，低于 80% 失败
pytest -k <test_name>
```

全队共用一个 `.venv`（Python 3.11–3.13），本机和云端都装
`requirements-train.txt` 加 `requirements-dev.txt`，版本完全一致。
本机是 CPU 版 torch，能跑全部测试、dry-run 和 preflight，不能训练；
两个 CUDA 内核只在 Linux x86_64 上从官方预编译包安装，不现场编译。
新增依赖写进 `requirements.txt`（运行时）、`requirements-train.txt`
（训练栈）或 `requirements-dev.txt`（开发工具），固定版本，
并在 PR 里说明用途。

## 云端训练

EVA 第一阶段的训练配置是 `configs/eva-qwen35-4b-lora-r16.toml`，
在组里的 AWS A10 上训练。LoRA 必须用提示词调优选出的提示词，
切换、打包、上传、启动、算提升的完整步骤见 `docs/cloud-training.md`：

```bash
python scripts/apply_prompt_selection.py --selection <selection.json>
python scripts/train_lora_sft.py --config <toml> \
    --eval-samples <eval.jsonl> --dry-run
python scripts/pack_for_cloud.py --config <toml> \
    --selection <selection.json> --output dist/<run>.tar
bash scripts/cloud/setup.sh                 # 云端
bash scripts/cloud/train.sh <toml>          # 云端：预检、训练、评测
python scripts/report_lora_gain.py ...      # 本机：相对同一提示词的提升
```

EVA 打分标准与标签统计见 `docs/eva-scoring-standard.md`。

## 爬虫数据流

记录 schema 在 `lora_sft/crawl/schema.py`，每种记录存成 JSONL：

1. `CapturedResponse`：抓包或爬取得到的一条原始响应，
   body 存在 `data/raw/`，记录里有路径和 sha256。
2. `PhotoCandidate`：适配器从响应里解析出的一张照片，
   有来源、授权、站点互动数据，还没下载图片。
3. `PhotoRecord`：图片已下载并算好哈希；`image_id`、
   `image.path`、`source` 直接进入标注记录。

适配器继承 `lora_sft/crawl/base.py` 的 `SiteAdapter`，只做解析；
请求、限速、下载由爬虫运行器负责。示例记录见
`tests/fixtures/photo_record.json`。

同一 sha256 只分配一个 `image_id`；下游按 id 检查评测集泄漏。

改 schema 时同步改 `SCHEMA_VERSION`、fixture 和测试。

## 数据与合规

- 评测集固定，永不进入训练数据；按 sha256 去重。
- 禁止兜底：解析失败、字段缺失、格式不符直接抛异常，
  不填默认值，不给 LLM 硬编码返回值。
- 写数据库、删文件等高风险操作先给 dry run 预览。

## 代码规范

- PEP 8，4 空格缩进，完整类型标注。
- Python 文件开头写 `from __future__ import annotations`。
- 函数和模块用 `snake_case`，类用 `PascalCase`，
  适配器以 `Adapter` 结尾。
- 公共函数和类写英文 docstring；异常信息用中文。
- schema 用 pydantic，`extra="forbid"`、`frozen=True`；
  “站点不提供” 显式写 `None`，不和 0 混用。
- 提示词文件用 kebab-case，如 `intent-summary.prompt.md`。
- Markdown 每行不超过 80 字符。
- 改完 Python 代码依次跑 `ruff check .` 和 `black .`；
  工具配置统一写在 `pyproject.toml`。

## 测试要求

- 每个模块都要覆盖正常路径、边界值、合规拒绝（反例）。
- 适配器测试用 `tests/fixtures/` 里保存的抓包样本，
  测试中不访问网络。
- 分支覆盖率 ≥ 80%；达不到时在 PR 里说明风险和计划。
- 测试失败时把日志贴到 PR 讨论里，方便复现。

## 待定事项

- 后续标准三分数 + 评语阶段是否扩到 Qwen3.5-9B。
- 美学头的视觉骨干选型。
- 门禁阈值（等标注一致性测出后定）。
- 近重复图片去重（感知哈希），防止评测集泄漏。
