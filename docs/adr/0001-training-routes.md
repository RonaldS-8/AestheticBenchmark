# ADR-0001：训练路线：LoRA SFT 与 DOVER 式美学头

- 状态：已采纳
- 日期：2026-11-02

## 背景

最初的方案（`docs/background/`）列了全参 SFT、LoRA SFT、DPO、PPO
四种后训练方式。美学模型要在 NAS 上本地推理，算力和内存都有限；
PPO 需要两台 8 卡节点，成本和不确定性都最高。

## 决定

- 只做两条路线，互相对比：
  - **LoRA SFT**：在 Qwen 视觉语言模型（4B 起步）上训练 LoRA，
    输出分数和评语。
  - **美学头**：参考 DOVER，冻结视觉骨干，只训练小的回归头，
    只输出分数。
- 训练侧复用 Transformers、PEFT、Accelerate（DDP / FSDP），
  不使用 DeepSpeed ZeRO-3（保存 LoRA 有已知问题）。

## 放弃的方案

- 全参 SFT：显存需求高，有灾难性遗忘风险，NAS 部署收益不明显。
- DPO / PPO：先看两条基础路线的效果，确有需要再单独立项。

## 后果

- 训练配置按 `route` 分为 `lora_sft` 和 `head` 两种。
- 导出分两条链路：LoRA 转 GGUF / MLX，美学头转 ONNX。
- 以后要加 DPO，需要新写一份 ADR 取代本文件。
