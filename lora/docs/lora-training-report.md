# 技术报告：EVA 阶段 LoRA SFT 的训练设计

- 日期：2026-11-03
- 配置：`configs/eva-qwen35-4b-lora-r16.toml`
- 代码：`lora_sft/train/qwen_lora.py`、`lora_sft/train/eva_data.py`
- 状态：代码已写完，本机 dry-run 通过；**尚未在 GPU 上实跑**，
  下文超参都是起步值，不是调优结果。

## 1. 任务回顾

- 输入：一张照片 + 固定提示词（`eva-dual-score@v1`）。
- 输出：`{"aesthetic":2.9,"technical":3.3}`，两项 1–5 分，
  保留一位小数。
- 数据：EVA，train 3,256 / val 407 / eval 407。
- 训练方式：标准下一词预测（交叉熵），**只对 assistant 回复和
  结束符计算损失**，系统提示、图片、用户问题都不计损失。

打分是以文本形式生成的，没有回归头。好处是和最终要加评语的输出
格式一致；代价是“2.9 和 3.0 差多少”模型只能从 token 学到，
损失本身不知道数值距离（见第 7 节）。

## 2. LoRA 挂在哪里

### 2.1 Qwen3.5-4B 的结构

```text
Qwen3_5ForConditionalGeneration
├─ model.visual            视觉塔 24 层 + merger 投影   → 冻结，不挂 LoRA
├─ model.language_model    32 层解码器                 → 挂 LoRA
│   ├─ 24 层 linear_attn   Gated DeltaNet（线性注意力）
│   ├─  8 层 self_attn     全注意力（每 4 层 1 层）
│   └─ 32 层 mlp           SwiGLU
└─ lm_head                 与词嵌入共享权重             → 冻结
```

Qwen3.5 是混合架构，这是和 Qwen2.5-VL / Qwen3-VL 最大的不同：
只写 `q_proj/k_proj/v_proj/o_proj` 的话，只会挂到 8 层全注意力上，
另外 24 层的注意力完全没被训练。

### 2.2 目标模块

| 位置 | 模块 | 形状（输入 → 输出） | 层数 |
| --- | --- | --- | ---: |
| 全注意力 | `q_proj` | 2560 → 8192 | 8 |
| 全注意力 | `k_proj` / `v_proj` | 2560 → 1024 | 8 |
| 全注意力 | `o_proj` | 4096 → 2560 | 8 |
| 线性注意力 | `in_proj_qkv` | 2560 → 8192 | 24 |
| 线性注意力 | `in_proj_z` | 2560 → 4096 | 24 |
| 线性注意力 | `out_proj` | 4096 → 2560 | 24 |
| MLP | `gate_proj` / `up_proj` | 2560 → 9216 | 32 |
| MLP | `down_proj` | 9216 → 2560 | 32 |

形状按官方 `config.json` 推算（`q_proj` 含输出门，所以是 2 倍）。

**不挂**的部分及原因：

- 视觉塔和 merger：EVA 只有约 3 千张图，训练视觉特征容易过拟合；
  而且端侧导出时视觉投影（mmproj）可以直接复用原版文件。
- `in_proj_a` / `in_proj_b`：输出只有 32 维（每个头一个门控标量），
  参数极少，低秩分解意义不大。
- 词嵌入、`lm_head`、各类 norm：输出只涉及少量数字和 JSON 符号，
  不需要改词表。

### 2.3 可训练参数量

LoRA 每个模块新增 `r × (输入维度 + 输出维度)` 个参数，r = 16 时：

| 部分 | 每层参数 | 层数 | 合计 |
| --- | ---: | ---: | ---: |
| 全注意力 | 393,216 | 8 | 3.15 M |
| 线性注意力 | 385,024 | 24 | 9.24 M |
| MLP | 565,248 | 32 | 18.09 M |
| **总计** | | | **约 30.5 M** |

约占基座参数的 0.7%，适配器文件 bf16 约 60 MB。
这是按配置推算的数值，实跑时 `run_info.json` 会记录
实际的 `trainable_parameters`。

## 3. r 和 alpha

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `r` | 16 | 低秩维度 |
| `alpha` | 32 | 缩放系数，实际缩放 = alpha / r = 2 |
| `dropout` | 0.05 | 只作用在 LoRA 分支的输入上 |

**为什么 r = 16**：

- 任务输出空间很窄（两个一位小数），需要学的主要是
  “视觉特征 → 分数”的映射，不需要大容量。
- 数据只有 3,256 条，r 太大（64、128）更容易记住训练集。
- r = 8 也合理；r = 16 是常见起点，留了一点余量。
  建议首轮跑完后对比 r = 8 / 16 / 32。

**为什么 alpha = 2r**：

- LoRA 的输出是 `(alpha / r) × B·A·x`。固定 alpha / r = 2，
  改 r 时更新幅度大致不变，学习率不用跟着重调。
- alpha = 2r 是 LoRA 社区最常用的设置；alpha = r（缩放 1）
  也常见，通常配稍高的学习率。

## 4. 优化器

使用 transformers `Trainer` 的默认优化器 **AdamW**
（`adamw_torch`），代码里没有另外指定：

| 参数 | 取值 | 来源 |
| --- | --- | --- |
| β1 / β2 | 0.9 / 0.999 | Trainer 默认 |
| ε | 1e-8 | Trainer 默认 |
| 学习率 | 1e-4 | 配置 `optim.learning_rate` |
| weight decay | 0.0 | 配置 `optim.weight_decay` |
| 梯度裁剪 | 1.0（全局范数） | 配置 `optim.max_grad_norm` |

**为什么用 AdamW**：LoRA 只更新约 30 M 参数，优化器状态
（两份动量，fp32）约 250 MB，显存不是瓶颈，没必要用 8-bit Adam
或 Adafactor 这类省显存的优化器。AdamW 是 LoRA 微调的标准选择，
行为最可预期。

**为什么 weight decay = 0**：LoRA 的 B 矩阵初始化为 0，
训练本身已经是从“不改动基座”出发的小幅更新；再加权重衰减
主要是把 LoRA 往 0 拉，在小数据上意义不大。过拟合靠
dropout 和按 val SRCC 选检查点控制。

## 5. 训练超参

| 参数 | 取值 | 理由 |
| --- | --- | --- |
| 学习率 | 1e-4 | LoRA 常用区间 5e-5 ~ 2e-4 的中间值 |
| 调度 | cosine | 后期平滑降到 0，常规选择 |
| warmup | 5%（约 31 步） | 前期避免大更新破坏基座 |
| epoch | 3 | 共 612 步，小数据集需要多看几遍 |
| 每卡 batch | 4 | 24 GB 显卡也能放下 |
| 梯度累积 | 4 | 全局 batch = 16 |
| 精度 | bf16 | 基座原生 bf16，比 fp16 稳定 |
| 梯度检查点 | 开 | 用约 20% 速度换显存 |
| max_seq_len | 1024 | EVA 图最大约 64 万像素，约 620 个视觉 token |
| 像素预算 | 65,536 ~ 1,048,576 | EVA 图不缩放，保留原尺寸 |
| 评估 / 保存 | 每 102 步 | 每半个 epoch，共 6 次；必须相等 |
| 选模型 | val 平均 SRCC 最高的检查点 | 生成打分、严格解析，训练结束自动加载 |
| seed | 42 | 数据顺序与初始化可复现 |

步数计算：3,256 ÷ 16 = 204 步 / epoch，3 个 epoch 共 612 步。

**关于 epoch 的风险**：3 千张图跑 3 遍，后期可能过拟合。
因此每半个 epoch 在 val 上生成打分、算 SRCC，最后保留
美学和技术平均 SRCC 最高的检查点，而不是最后一步。
不按 val loss 选：loss 低不等于排序好，而门禁看的是 SRCC。
解析率低于 0.95 或 SRCC 无定义的检查点不能入选。

**关于像素预算**：EVA 图是作者缩放过的版本（长边约 640–800 px），
所以训练时不再缩小。**端侧推理必须用同样的像素预算**，
否则视觉 token 数不同，分数会漂移。

## 6. 代码实现

### 6.1 样本编码与损失遮罩

`EvaSftCollator.encode`（`lora_sft/train/qwen_lora.py`）：

```python
prompt = processor.apply_chat_template(
    messages_without_reply, tokenize=False,
    add_generation_prompt=True, enable_thinking=False)
full = processor.apply_chat_template(
    messages_with_reply, tokenize=False, enable_thinking=False)
tail = completion_text(prompt, full, reply)   # 回复 + "<|im_end|>\n"
encoded = processor(text=[full], images=[image],
                    size={"shortest_edge": min_px,
                          "longest_edge": max_px})
completion_ids = tokenizer(tail, add_special_tokens=False)["input_ids"]
labels = mask_labels(encoded["input_ids"][0], completion_ids)
```

- `completion_text` 检查完整对话一定以生成提示开头，保证训练时的
  前缀和推理时完全一样（包括 `<think>\n\n</think>\n\n`）。
- `mask_labels` 检查序列末尾确实是回复的 token，然后把前面的位置
  全部设为 -100，损失只算回复。对不上直接报错，不猜。
- 超过 `max_seq_len` 直接报错，不截断（截断会切掉回复）。

`encode` 一次只处理一条样本，遮罩也按条打好。Trainer 每次取
`per_device_batch_size` 条交给 `EvaSftCollator.__call__`，在这里
逐条编码后补齐成一个批次：

```python
def __call__(self, samples):
    items = [self.encode(s) for s in samples]   # 每条各自算 n、打遮罩
    longest = max(it["input_ids"].numel() for it in items)
    pad_values = {
        "input_ids": tokenizer.pad_token_id,
        "attention_mask": 0,          # 补齐位置不参与注意力
        "mm_token_type_ids": 0,
        "labels": -100,               # 补齐位置不算损失
    }
    batch = {}
    for key, pad in pad_values.items():
        rows = []
        for it in items:
            row = it[key]                                  # [L_i]
            filler = row.new_full((longest - row.numel(),), pad)
            rows.append(torch.cat([row, filler]))          # 右侧补齐
        batch[key] = torch.stack(rows)                     # [B, longest]
    for key in ("pixel_values", "image_grid_thw"):
        batch[key] = torch.cat([it[key] for it in items])  # 按图片拼接
    return batch
```

- token 类字段按每个字段各自的值右侧补齐，再叠成 `[B, longest]`。
- 图片字段长度与序列无关，直接首尾拼接：`pixel_values` 为
  `[所有图的小块数之和, 1536]`，`image_grid_thw` 为 `[B, 3]`，
  模型按后者把小块切回每张图。
- 处理器输出未知字段时，`encode` 直接报错，不猜如何补齐。

### 6.2 挂 LoRA

```python
def lora_target_pattern(settings):
    names = "|".join(settings.target_modules)
    scope = r"model\.language_model\." if settings.freeze_vision \
        else r"model\."
    return rf"^{scope}.*\.({names})$"

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules=lora_target_pattern(cfg.lora),
                  bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base_model, lora)
```

- 用正则而不是模块名列表，确保只匹配语言模型里的层，
  不会误挂到视觉塔。
- 挂完后检查配置里的每个模块名都确实被包装了，
  有遗漏（比如模型版本变了、层改名了）直接报错。
- 开梯度检查点时先调用 `enable_input_require_grads()`，
  并用 `use_reentrant=False`，这是 PEFT + 梯度检查点的标准写法。

### 6.3 训练循环

`training_arguments` 把 TOML 配置一一映射到 `TrainingArguments`，
不在代码里藏默认超参；`build_trainer` 组装 `Trainer`：

- `remove_unused_columns=False`：样本是自定义对象，
  由 collator 现场编码。
- `load_best_model_at_end=True`；`selection.metric = "srcc_mean"` 时
  `metric_for_best_model="eval_srcc_mean"`、越大越好。
- `EvaSftTrainer.evaluate` 先算 val loss，再用当前权重对 val 逐张
  贪心解码（强制 `use_cache=True`），用 `selection_score` 严格解析并算
  两项 SRCC；多卡时每卡解码一份分片后汇总。不合格的检查点记为 -2，
  低于 SRCC 下限，永远不会入选；全部不合格时脚本不写 adapter、报错。
- `save_steps` 必须等于 `eval_steps`（配置校验），否则获胜的那一步
  可能没保存，Trainer 会静默保留更早的检查点。
- 只支持 single 和 DDP；配置写 fsdp 会直接报错。
- 每条日志追加写入 `outputs/<run>/metrics.jsonl`。

### 6.4 启动前的检查

训练开始前（`scripts/train_lora_sft.py`）：

1. 配置 schema 校验，所有字段必填、无默认值。
2. 划分版本和 eval 指纹与配置一致。
3. train / val 与 eval 在 image_id 和图片哈希上都不重叠。
4. 所有图片存在且哈希正确。
5. chat template 的 SHA-256 等于配置值。
6. 配置要求的 Gated DeltaNet 内核（`kernels.require_fla`）能导入，
   否则报错，不让 transformers 静默退回纯 torch。
7. preflight 模式下把全部 train / val 样本编码一遍，
   提前发现超长或遮罩错误。

## 7. 已知局限与后续实验

| 问题 | 影响 | 建议 |
| --- | --- | --- |
| 交叉熵不知道数值距离 | 3.0 预测成 3.1 和 4.5 罚得一样多 | 对比美学头路线；或加回归损失 |
| 技术分很集中 | 目标标准差约 0.36（1–5 分制） | 单独看技术分 SRCC，别只看平均 |
| 超参未调 | 当前只是起步值 | 首轮后扫 lr（5e-5 / 1e-4 / 2e-4）和 r（8 / 16 / 32） |
| 未在 GPU 实跑 | 可能有接口问题 | 先跑 preflight，再正式训练 |

参考上限：EVA 的标注者半分一致性（SRCC）美学约 0.82、
技术约 0.72，模型做到接近这个数就已经很好。
