# ADR-0010：LoRA 训练栈与云端运行方式

- 状态：已采纳
- 日期：2026-11-03

## 背景

ADR-0009 定了 EVA 第一阶段训练 `Qwen/Qwen3.5-4B` 双分数 SFT。
本机没有 NVIDIA 显卡，训练要在云端 GPU 上跑，并且要求打包上传后
一条命令开始。Qwen3.5 是混合结构：每 4 层里 3 层是 Gated DeltaNet
线性注意力（`in_proj_qkv`、`in_proj_z`、`out_proj`），1 层是普通注意力
（`q/k/v/o_proj`），LoRA 目标不能照搬 Qwen3-VL。

## 决定

- 训练栈用 `transformers` 的 `Trainer` + `peft`，版本固定在
  `requirements-train.txt`（transformers 5.20.2，支持 `qwen3_5`）。
  本机开发环境不装 torch；只有 `lora_sft/train/qwen_lora.py` 导入 torch，
  其余检查逻辑在 `eva_data.py`，本机可测。
- 损失只算回复：训练序列 = 生成提示（`enable_thinking=false`）+
  回复 + 模板结束符。先校验模板完整渲染以生成提示开头，再校验序列
  末尾与单独分词的回复一致，不一致直接报错。超长样本报错，不截断。
- LoRA 只加在语言模型上，目标包括两种注意力的投影和 MLP；
  配置里的目标模块在模型中找不到时报错。视觉塔和投影层冻结。
- 启动时校验处理器的 chat template SHA-256 等于配置值；像素预算
  从配置传入处理器，推理时用同一套值。
- 按 val 损失保留最佳检查点；选超参数只看 val，eval 集只做最终报告。
- 每次训练前先跑 B0（原版模型 + 同一提示词）作对照，评测按原始投票
  均值算 SRCC / PLCC，解析失败计入失败率，不修补输出。
- 上传包只含代码、样本目录和样本引用的图片，不含 `data/raw/`。
- 暂只支持单卡和 DDP；FSDP、DeepSpeed 不做。

## 放弃的方案

- TRL `SFTTrainer`、LLaMA-Factory、ms-swift：多一层封装和版本耦合，
  对新出的 Qwen3.5 支持滞后时难以排查；我们只需要回复掩码和
  多模态拼批，自己写的部分很小。
- 用模板渲染的 token 前缀长度做掩码：BPE 在边界合并时会错位，
  改为按回复 token 后缀校验。
- 对所有 token 计算损失：模型会去拟合固定的系统提示词，浪费容量。

## 后果

- 换模型或升级 transformers 时，preflight 会先暴露模板或处理器
  输出字段的变化（未知字段直接报错），需要人工确认后改代码。
- 不装 `flash-linear-attention` 时线性注意力层较慢，但结果一致。
- val 损失低不等于 SRCC 高，最终仍以评测报告为准。

## 修订（2026-11-05）

- 选模：由按 val 损失改为按 val 上生成打分的平均 SRCC
  （`selection.metric = "srcc_mean"`）；`save_steps` 必须等于
  `eval_steps`，否则获胜的那一步可能没保存。原因：门禁看 SRCC，
  loss 低不等于排序好。
- 版本统一：全队共用一个 `.venv`（Python 3.11–3.13），本机与云端装同一份
  `requirements-train.txt` + `requirements-dev.txt`。torch 固定为 2.8.0，
  因为它是 flash-linear-attention、causal-conv1d、flash-attn 三者
  都有官方预编译包的唯一版本；CUDA 内核只装预编译包，不现场编译。
- 线性注意力内核改为必需：transformers 缺内核时静默走纯 torch，
  EVA 配置要求 FLA 和 causal-conv1d 都能导入，训练前强制检查。
- 上面“不装 `flash-linear-attention` 时较慢但结果一致”的后果作废。
