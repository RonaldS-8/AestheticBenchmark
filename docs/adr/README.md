# 架构决策记录（ADR）

记录项目里已经定下来的关键决定：为什么这么做、放弃了什么、
带来什么后果。代码和设计文档说明“是什么”，ADR 说明“为什么”。

## 规则

- 一个决定一份文件，命名 `NNNN-短标题.md`，编号递增，不复用。
- 状态：`提议` → `已采纳`；被推翻时不删除原文件，
  改为 `已被 ADR-NNNN 取代`，再新建一份 ADR。
- 只记录难以回退、影响多个模块或团队约定的决定；
  参数调整、实现细节不写 ADR。
- 每份 ADR 包含：背景、决定、放弃的方案、后果。

## 索引

| 编号 | 标题 | 状态 |
| --- | --- | --- |
| [0001](0001-training-routes.md) | 训练路线：LoRA SFT 与 DOVER 式美学头 | 已采纳 |
| [0002](0002-task-output.md) | 任务形式：打分加一句评语，以打分为主 | 已采纳 |
| [0003](0003-strict-schemas.md) | 数据与配置：严格 schema、JSONL、TOML | 已采纳 |
| [0004](0004-versioned-prompts.md) | 提示词按版本文件夹管理 | 已采纳 |
| [0005](0005-first-comparison-video.md) | 第一版对比：视频上提示词基线对比 DOVER | 已被 0011 取代 |
| [0006](0006-blind-ranking-review.md) | 客户盲评：只看排序，不看分数 | 已被 0011 取代 |
| [0007](0007-data-layout.md) | 数据目录与文件管理 | 已采纳 |
| [0008](0008-local-pilot-run.md) | 本机小样本试跑（B0 + B1 对比 DOVER） | 已被 0011 取代 |
| [0009](0009-eva-dual-score-sft.md) | EVA 先做双分数弱监督 SFT | 已采纳 |
| [0010](0010-lora-training-stack.md) | LoRA 训练栈与云端运行方式 | 已采纳 |
| [0011](0011-first-comparison-eva-two-scores.md) | 第一轮对比：EVA 图片双分数基线 | 已采纳 |
| [0012](0012-prompt-tuning-scope.md) | 调优用全量 EVA，LoRA 用最佳提示词 | 已采纳 |
| [0013](0013-reply-format-wrappers.md) | 解析前去掉回复外层的格式外壳 | 已采纳 |
