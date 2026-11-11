# ADR-0011：第一轮对比改为 EVA 图片上的双分数提示词基线

- 状态：已采纳
- 日期：2026-11-05
- 取代：ADR-0005、ADR-0006、ADR-0008
- 补充：数据范围和 LoRA 用的提示词见 ADR-0012

## 背景

模型输出改成了两个分数（美学、技术），当前任务是图片打分。
EVA 第一阶段的 LoRA SFT 已经定为双分数（ADR-0009），
而原来的第一轮对比是视频上提示词基线对比 DOVER，
输出三项分数加评语，和现在的任务对不上。

提示词基线是第一个要跑的东西：它决定 LoRA 至少要超过多少。
所以基线和 LoRA 必须在同一份数据、同一个输出格式、同一套指标上比。

## 决定

- **数据**：EVA 图片，用 `data/manifests/eva/sft/v1/` 的同一份
  划分；先跑 val，最终结论用 eval。
- **输出**：`{"aesthetic": x.x, "technical": x.x}`，与 EVA SFT
  完全相同，用同一个严格解析器，失败计数、不修补。
- **对照对象**：EVA 人工投票均分，不再对比 DOVER。
  指标是 SRCC / PLCC，对原始投票均分计算，同时报告解析成功率。
- **提示词**：`prompts/baseline-v2/` 的 B0、B1、B2。
  B2 的示例按美学分分位数从训练集固定选取，回复用 SFT 数据原文；
  B3（检索增强）暂不实现；B4 是换模型尺寸重跑。
- **运行**：`scripts/run_eva_baseline.py`。模型、revision、
  像素预算从 LoRA 训练配置读取；产出的 `predictions.jsonl` 和
  `report.json` 与 LoRA 评测格式相同，用
  `scripts/summarize_eva_runs.py` 汇总成一张表。
- **归档**：视频与 DOVER 的代码、脚本、提示词和测试移到
  `archive/video-dover/`，不删除，可从提交 `ece31c2` 恢复。

## 放弃的方案

- 继续在视频上对比 DOVER：任务已经改成图片打分，
  DOVER 也不支持图片。
- 基线继续输出三项分数加评语：EVA 没有总体分和评语，
  无法对照，也和 LoRA 的输出不一致。
- 客户盲评排序（ADR-0006）作为第一轮依据：EVA 有人工投票均分，
  直接算相关系数更客观；盲评以后需要时再从归档恢复。

## 后果

- LoRA 的门槛是 B0–B2 里最好的那个，不再是 DOVER。
- EVA 上人群自身的一致性上限约为：美学 SRCC 0.90，技术 0.84
  （`docs/eva-scoring-standard.md`），解读结果时以此为参照。
- `scripts/cloud/train.sh` 里的 “B0” 是基座模型加 SFT 提示词
  `eva-dual-score@v1`，是训练前后的同提示词对照，
  不等同于本 ADR 的 B0（`zero-shot-score@v2`）。
- 本机用 llama.cpp 跑的量化模型和云端 bf16 不是同一份权重，
  正式对比要在同一运行时上做。
- DIVIDE-MaxWell 和 DOVER 相关的待确认事项（ADR-0005、0008）
  随归档暂停，不再阻塞当前工作。
