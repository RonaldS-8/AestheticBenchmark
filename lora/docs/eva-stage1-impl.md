# EVA 第一阶段：实现接口与训练问题对策

核对日期：2026-11-05。对应客户计划 `pipeline/eva-stage1-plan.html`
第 6、7 节；现有训练代码见 `docs/lora-training-report.md` 第 6 节。

实现状态：第 1 节已按 `logits_to_keep` 方式实现（`reply_loss`），
第 2 节的内核检查与安装、第 3 节均已实现，CPU 小模型测试通过，
尚未在 GPU 上实跑。第 1.2、1.3 节的代码是备选写法，未合入。

第 1–3 节是加速方案，第 4 节是训练中常见问题的对策。

## 1. 只对答案位置算输出层和损失

### 1.1 现状

`Qwen3_5ForConditionalGeneration.forward` 末尾是：

```python
hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
logits = self.lm_head(hidden_states)        # [B, L, 248320]
```

`logits_to_keep=0` 时对整条序列算全词表 logits，再由损失函数丢掉
`labels == -100` 的位置。

### 1.2 方案 A：取答案位置的隐藏状态（不加依赖）

不传 `labels`，拿最后一层隐藏状态，自己算损失：

```python
def answer_only_loss(model, batch):
    labels = batch.pop("labels")                      # [B, L]
    out = model.model(**batch)                        # 不经过 lm_head
    hidden = out.last_hidden_state                    # [B, L, d]
    # 位置 t 的隐藏状态预测 t+1 的 token：先错位再取有效位置
    shift_h = hidden[:, :-1]                          # [B, L-1, d]
    shift_y = labels[:, 1:]                           # [B, L-1]
    keep = shift_y != -100
    h = shift_h[keep]                                 # [N_ans, d]
    y = shift_y[keep]                                 # [N_ans]
    logits = model.lm_head(h).float()                 # [N_ans, V]
    loss = torch.nn.functional.cross_entropy(logits, y, reduction="sum")
    return loss, keep.sum()
```

- 在自定义 `Trainer.compute_loss` 里调用；返回值按全局答案 token 数
  归一化（梯度累积、多卡时用 `num_items_in_batch`），不要按 batch 均值。
- `model.model` 是 PEFT 包装后的 `base_model.model.model`，
  取法以实际包装层级为准，取不到直接报错。
- 校验：同一 batch 上与原生 `labels` 路径的 loss 差 < 1e-3。

### 1.3 方案 B：Liger 融合线性 + 交叉熵

```python
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

flce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="sum")
loss = flce(model.lm_head.weight, shift_h.reshape(-1, d), shift_y.reshape(-1))
```

- 分块计算 `hidden @ W^T` 与交叉熵，不落整块 logits。
- 与方案 A 可叠加：先按 `keep` 取位置，再送入 FLCE。
- 新依赖写进 `requirements-train.txt` 并固定版本。

## 2. Gated DeltaNet 专用算子

transformers 的 `modeling_qwen3_5.py` 在 `causal_conv1d_fn` /
`causal_conv1d_update` 上用 `use_kernel_func_from_hub_with_fallback`，
找不到算子时退回 `torch_chunk_gated_delta_rule` 等纯 torch 实现。

transformers 5.20.2 在导入时解析内核：先导入原包
（`fla.ops.gated_delta_rule`、`causal_conv1d`），任何异常都被吞掉，
改用纯 torch 实现，只打一条警告。

- 版本：torch 固定为 2.8.0，这是 flash-linear-attention 0.5.2
  （torch ≥ 2.7）、causal-conv1d 1.7.0（预编译包 torch 2.6–2.10）、
  flash-attn 2.8.3.post2（预编译包 torch 2.4–2.8）共同支持的唯一版本。
- 安装：全部写在 `requirements-train.txt`；causal-conv1d 和 flash-attn
  用官方 wheel 直链，按 Python 3.11–3.13 和 Linux x86_64 选包，
  不在训练机上源码编译。`setup.sh` 在 Linux 上最后检查内核能导入。
- 启动前检查：`qwen_lora.check_kernels` 按 transformers 同样的模块和
  函数名导入，配置要求的（EVA 两个都要求）导入失败直接报错，
  错误信息里带原始异常；`kernel_status` 的结果写进 `run_info.json`。
  preflight 只在有 CUDA 的机器上强制（`preflight_kernels`），
  训练前的检查始终强制。

## 3. 梯度检查点

已在现有代码中开启：`gradient_checkpointing=True`，
`gradient_checkpointing_kwargs={"use_reentrant": False}`，
并在挂 LoRA 前调用 `enable_input_require_grads()`。

## 4. 训练中常见问题与对策

### 4.1 灾难性遗忘

**现象**：只训打分后，模型看图描述、遵循指令变差；第二阶段
写评语时用词贫乏、句式单一，或别的提问也只回打分 JSON。

**原因**：小数据、单一格式反复训练，梯度只朝“输出两个数字”
一个方向推，覆盖了原有能力用到的方向。

**对策（按优先级）**：

1. **数据回放（replay）**：训练集按 10–20% 掺入通用数据。
   - 图文指令：开源图片描述、视觉问答样本，或用原模型
     对训练图生成描述（自蒸馏），抽检后使用。
   - 纯文本指令：少量通用问答，保住语言能力。
   - 回放样本同样只在 assistant 上算损失；在 collator 里
     按比例混采，并在 `run_info.json` 记录混合比例和来源版本。
   - 自蒸馏的好处：答案来自原模型本身，学的是“保持原样”，
     不引入外部风格；缺点是不纠错，只适合做回放。
2. **限制改动幅度**：
   - LoRA 只学低秩增量，原权重不动；`r` 不盲目调大。
   - epoch 不超过 3，按 val SRCC 提前停止。
   - 学习率不高于 2e-4；warmup 保留。
3. **能力回归集**：固定 100–200 道通用题（看图描述、格式遵循、
   简单视觉问答），原模型输出作参考。每个候选检查点用
   LLM 评审或规则打分，相对原模型退化超过阈值即淘汰。
4. **备选**：
   - 只挂 MLP 或只挂注意力做对比，看哪种遗忘更少。
   - 权重插值（WiSE-FT 思路）：合并时用 `scale < 1` 缩小 LoRA
     增量，在打分精度和通用能力之间取折中，取值按验证集选。
   - 对原模型输出加 KL 约束：需要同时前向参考模型，显存翻倍，
     只在前几种不够时才用。

**回放数据源**（授权核对于 2026-11-05，上线前再复核）：

| 用途 | 来源 | 取什么 | 授权 |
| --- | --- | --- | --- |
| 看图描述 | `google/docci` train | 9,647 张图片 | CC BY 4.0 |
| 视觉问答 | VQAv2 train2014 | 图片 + 问题 | 标注 CC BY 4.0 |
| 纯文本 | 原模型 Magpie 自生成 | 指令 + 回答 | Apache 2.0 |
| 备选图文 | `allenai/pixmo-cap` | 图片 URL | ODC-BY 1.0 |
| 备选文本 | `HuggingFaceTB/smoltalk` | 问题 | Apache 2.0 |

- VQAv2 来自 visualqa.org，图片是 COCO train2014，版权按 Flickr
  条款；PixMo-Cap 定位研究与教育用途，图片要自己下载，链接会失效。
- SmolTalk 只有新增子集（smol-magpie-ultra 等）是 Apache 2.0，
  收录的公开子集按各自原授权。
- 不用 LLaVA-Instruct-150K（GPT-4 生成，受 OpenAI 条款约束）和
  ShareGPT4V（CC BY-NC 4.0）。BELLE 是 GPL-3.0，也不用。
- 能力回归题取 DOCCI test 和 VQAv2 val2014，与回放样本不重叠。

**生成步骤**：

1. DOCCI 只取图片，指令从固定的中英文模板里选，如“详细描述这张
   照片”“照片的主体是什么，在做什么”；不用 DOCCI 原描述作答案。
2. VQAv2 每张图取 1 道问题，保留英文原文。
3. Magpie：只给原模型 `<|im_start|>user\n`，采样让它写出一条指令；
   中英文各一半，按语义去重，去掉过短和半截的指令。
4. 全部答案由原模型生成：同一对话模板、`enable_thinking=false`、
   低温采样；限制 `max_new_tokens`，保证整条样本不超过
   `max_seq_len`。抽检 5%，答非所问的整批重做。
5. 图片先缩到与 EVA 相近的尺寸（≤ 约 64 万像素）。当前
   `max_pixels` 下 1024×1024 以上的图会占满约 1,024 个视觉 token，
   加上文字就超过 `max_seq_len = 1024`，collator 直接报错。
6. 配比：第一阶段约 400–800 条（占 10–20%），看图描述、视觉
   问答、纯文本按 2 : 1 : 1 起步；配比、来源版本和生成配置写进
   `run_info.json`。

### 4.2 过拟合

- **现象**：train loss 持续下降，val SRCC 先升后降。
- **对策**：按 val SRCC 选检查点（不只看 loss）；`lora_dropout`
  0.05–0.1；减小 `r`；数据增强只做不改变美感的变换（不裁剪
  构图、不调色，最多轻微缩放到同一像素预算）。

### 4.3 分数塌缩到中间值

- **现象**：预测几乎都在 3.3–3.7，MAE 好看但 SRCC 低。
- **原因**：标签集中在 3.5 附近，交叉熵对数值距离不敏感。
- **对策**：
  - 推理时读数字 token 概率取期望分，比贪心解码更连续。
  - 按分数分桶重采样，提高两端样本的出现频率（只在训练集做）。
  - 监控预测分的标准差，与标签标准差（约 0.4）对比。

### 4.4 两项分数只学会一项

- **现象**：美学、技术预测相关远高于人群的 0.76。
- **对策**：单独看技术分 SRCC；技术分样本里挑“美学高、技术低”
  和相反的样本加权，让模型看到两项分离的情况。

### 4.5 训练不稳定

- **loss 出现 NaN 或尖峰**：确认 bf16（不用 fp16）；梯度裁剪 1.0；
  检查有没有全为 -100 的样本（分母为 0），collator 里直接报错。
- **loss 不降**：先在 16 条样本上过拟合，确认遮罩和标签对齐；
  把 labels 不为 -100 的部分解码打印，必须只剩答案。

### 4.6 训练与推理不一致

- **现象**：云端评测好，端侧掉点。
- **对策**：提示词、chat template、像素预算的哈希写进导出清单，
  端侧加载时比对，不一致直接报错；量化后在同一评测集复测。

### 4.7 显存不足

按收益依次：只对答案算 logits（第 1 节）→ 梯度检查点 →
减小每卡 batch、增大梯度累积 → 降低最大像素预算（会改变视觉
token 数，训练和推理必须同步改）。
