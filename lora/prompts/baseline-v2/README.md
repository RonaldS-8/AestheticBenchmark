# 评测基线提示词 v2（图片，双分数）

第一轮要跑的东西：**不训练的 Qwen 加提示词，在 EVA 图片上打美学分和
技术分**，与 EVA 人工投票均分算 SRCC / PLCC（ADR-0011、ADR-0012）。

| 方法 | 提示词 | 作用 |
| --- | --- | --- |
| B0 零样本 | `zero-shot-score@v2` | 只给任务和输出格式 |
| B1 评分细则 | `rubric-score@v2` | 写明两项 1–5 分的标准 |
| B2 少样本 | `few-shot-score@v2` | B1 + 3 张示例图；作参考基线 |
| B3 检索增强 | `few-shot-score@v2` | 按相似度选示例；**还没实现** |
| B4 模型尺寸 | 同上 | 换 `model` 重跑，不是新提示词 |

LoRA 用的提示词不是固定的：用 B0 / B1 里得分最高的那个
（**最佳单图提示词**），训练和评测都用它。B2 每张图要带 3 张示例，
训练和 NAS 推理都会慢几倍，所以只作参考，不给 LoRA 用。

## 流程

提示词不训练，所以调优**不分训练集和测试集**：所有基线都评同一批图，
即全部 EVA 图片（4070 张）去掉 B2 的 3 张示例图，共 4067 张。

```bash
# 1. 固定示例图和要评的图（已生成 v1，不用重跑）
python scripts/prepare_prompt_tuning.py \
    --sft-manifest manifests/eva/sft/v1/manifest.json --k 3 --version v1
# 2. 每个提示词跑一遍（B0、B1、B2 换 --method 和 --prompt-id）
python scripts/run_eva_baseline.py \
    --config configs/eva-qwen35-4b-lora-r16.toml \
    --tuning-set data/manifests/eva/prompt-tuning/v1.json \
    --method b1_rubric --prompt-id rubric-score@v2 \
    --run-id b1-qwen35-4b-tune --runtime transformers --device cuda
# 3. 选最佳提示词，写下 LoRA 必须用的那个
python scripts/select_best_prompt.py --min-parse-rate 0.95 \
    --output data/manifests/eva/prompt-tuning/v1-selection.json \
    data/runs/b0-qwen35-4b-tune data/runs/b1-qwen35-4b-tune \
    data/runs/b2-qwen35-4b-tune
# 4. LoRA 训练评测完后，看提升了多少
python scripts/report_lora_gain.py \
    --selection data/manifests/eva/prompt-tuning/v1-selection.json \
    --base-run data/runs/b1-qwen35-4b-tune \
    --lora-run data/runs/eva-qwen35-4b-lora-r16-eval
```

- 第 2 步每次写两份报告：`report.json` 是 4067 张的结果，用来选提示词；
  `report-eval.json` 只取其中属于 SFT eval 集的 407 张，
  就是“同一个提示词、不加 LoRA”的成绩，用来算 LoRA 的提升。
- 选择规则：两项 SRCC 的平均分最高；解析率低于门槛、或某项相关系数
  无定义的不参与；并列时报错，交给人决定。
- LoRA 用选定提示词之前，运行 `scripts/apply_prompt_selection.py
  --selection <选择文件>`：用它重新生成 SFT 数据 v2（划分不变），
  再把训练配置的 `prompt_id` 和样本路径改过去，步骤见
  `docs/cloud-training.md`。第 4 步会检查基线和 LoRA 用的是不是同一个选定提示词、同一个模型、
  同一份 eval 集，不一致就报错。
- 本机没显卡时可以用 `--runtime llama_cpp --limit 60` 冒烟测试。
  量化模型和云端 bf16 不是同一份权重，不能拿来选提示词或算提升。

## 和 LoRA 评测统一的地方

| 项目 | 做法 |
| --- | --- |
| 输出 | `{"aesthetic": x.x, "technical": x.x}`，与 SFT 回复相同 |
| 解析 | 同一个解析器：去掉 think 块后取回复里唯一的完整 JSON 对象，严格解析（ADR-0013）；多余的键、超范围都算失败 |
| 指标 | 对 EVA 原始投票均分算 SRCC / PLCC，同时报解析率 |
| 报告 | `predictions.jsonl` + `report.json`，格式与 LoRA 评测相同 |
| 模型 | 模型、revision、像素预算都从 LoRA 训练配置读 |
| 示例回复 | B2 的示例回复就是 SFT 训练数据里的回复原文 |

## 文件格式

```text
---
id: few-shot-score@v2             # 文件名@版本，必须和文件名一致
methods: b2_few_shot, b3_retrieval
media: image
---

（第一个 ## 标题之前是给人看的说明，不发给模型）

## system                         # 可选，单独成为 system 消息
## example                        # 只有少样本提示词有，且必须有，带 {i}
## user                           # 必需，待评那一轮的文字
```

- 段内可以用 `###` 小标题和列表，原样发给模型。
- `{i}`（示例序号）、`{k}`（示例总数）只能用在少样本提示词里。

## 图片是怎么加进去的

提示词文件里没有图片。每评一张图时，程序按下面三步组装：

1. **组装消息**（`lora_sft/eval/prompts.py` 的 `build_messages`）：
   在 user 消息的**第一个位置**插入一个图片项
   `{"type": "image", "image": "<相对 data/ 的路径>"}`，
   后面接提示词文件 `## user` 段的文字。有 `## system` 段时，
   它单独成为第一条 system 消息。
   B2 的示例轮用 `## example` 段的文字（不是 `## user`），
   `{i}` 换成示例序号、`{k}` 换成示例总数，
   所以模型看到的是“【示例 1/3】…【示例 3/3】…【待评照片】…”，
   不会把示例和待评照片混在一起。
2. **交给模型**，按运行时不同：
   - transformers：图片项只留 `{"type": "image"}` 作占位，
     对话模板把它渲染成 `<|vision_start|><|image_pad|><|vision_end|>`；
     图片本身按出现顺序作为列表交给 processor，
     processor 再把 `<|image_pad|>` 展开成与像素预算对应数量的视觉 token。
     这和训练时完全一样。
   - llama.cpp：图片按像素预算缩放、边长对齐 32，转成 JPEG，
     以 `image_url` 的 base64 data URI 发给 llama-server。
3. **思考模式关闭**：模板参数固定 `enable_thinking=false`，
   所以生成提示末尾是一段空的 `<think></think>`，模型直接输出 JSON。

## 例子

以下内容由代码真实生成：待评图是 eval 集的 `photo_000132`
（`images/eva/132.jpg`），渲染用的是 Qwen3.5-4B 自带的对话模板。
长文本用 `…` 省略。

### B0：只有一条 user 消息

组装后的消息：

```json
[
  {"role": "user", "content": [
    {"type": "image", "image": "images/eva/132.jpg"},
    {"type": "text", "text": "请为这张照片分别打一个美学分和一个技术分，…"}
  ]}
]
```

模板渲染后的文本（图片列表：`images/eva/132.jpg`）：

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>请为这张照片分别打一个美学分和
一个技术分，都是 1–5 分。

只输出一个 JSON 对象，不要代码块，不要其他文字。只能有下面两个键，
键的顺序固定，分数是 1.0–5.0 之间、保留一位小数的数字：

{"aesthetic": 分数, "technical": 分数}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

（为了排版，第二行在“和”后面折了一下，实际是同一行。）

### B1：system 放细则，user 放图片和一句话

```json
[
  {"role": "system", "content": [
    {"type": "text", "text": "你是照片美学评审。按下面的细则，…"}
  ]},
  {"role": "user", "content": [
    {"type": "image", "image": "images/eva/132.jpg"},
    {"type": "text", "text": "请按细则为这张照片评分。"}
  ]}
]
```

```text
<|im_start|>system
你是照片美学评审。按下面的细则，为照片分别给出美学分和技术分，
两项都是 1–5 分。
…（细则全文）…
{"aesthetic": 分数, "technical": 分数}<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>请按细则为这张照片评分。<|im_end|>
<|im_start|>assistant
<think>

</think>

```

### B2：3 轮示例在前，待评图在最后

3 张示例图来自 `data/manifests/eva/prompt-tuning/v1.json`，
按美学分从低到高各取一张。每轮示例的 user 文字带【示例 i/3】编号，
最后一轮带【待评照片】：

| 示例 | 图片 | 回复（即 SFT 目标） | EVA 美学均分（0–10） |
| --- | --- | --- | --- |
| 1 | `images/eva/841258.jpg` | `{"aesthetic":3.1,"technical":3.5}` | 5.13 |
| 2 | `images/eva/661415.jpg` | `{"aesthetic":3.5,"technical":3.4}` | 6.15 |
| 3 | `images/eva/183933.jpg` | `{"aesthetic":3.9,"technical":3.7}` | 7.13 |

```json
[
  {"role": "system", "content": [{"type": "text", "text": "…细则…"}]},
  {"role": "user", "content": [
    {"type": "image", "image": "images/eva/841258.jpg"},
    {"type": "text", "text": "【示例 1/3】这张照片已由人工评好，…"}]},
  {"role": "assistant", "content": [
    {"type": "text", "text": "{\"aesthetic\":3.1,\"technical\":3.5}"}]},
  {"role": "user", "content": [
    {"type": "image", "image": "images/eva/661415.jpg"},
    {"type": "text", "text": "【示例 2/3】这张照片已由人工评好，…"}]},
  {"role": "assistant", "content": [
    {"type": "text", "text": "{\"aesthetic\":3.5,\"technical\":3.4}"}]},
  {"role": "user", "content": [
    {"type": "image", "image": "images/eva/183933.jpg"},
    {"type": "text", "text": "【示例 3/3】这张照片已由人工评好，…"}]},
  {"role": "assistant", "content": [
    {"type": "text", "text": "{\"aesthetic\":3.9,\"technical\":3.7}"}]},
  {"role": "user", "content": [
    {"type": "image", "image": "images/eva/132.jpg"},
    {"type": "text", "text": "【待评照片】以上 3 张示例到此结束。…"}]}
]
```

图片列表按出现顺序是 841258、661415、183933、132，
与下面 4 个 `<|image_pad|>` 一一对应：

```text
<|im_start|>system
你是照片美学评审。…
对话开头标着【示例】的几轮是人工评好的照片，紧跟的 assistant 回复
就是它的人工分数，只用来校准尺度，不需要你再评；标着【待评照片】的
最后一张才是你要评分的。
…（与 B1 相同的细则）…<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>【示例 1/3】这张照片已由…<|im_end|>
<|im_start|>assistant
{"aesthetic":3.1,"technical":3.5}<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>【示例 2/3】这张照片已由…<|im_end|>
<|im_start|>assistant
{"aesthetic":3.5,"technical":3.4}<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>【示例 3/3】这张照片已由…<|im_end|>
<|im_start|>assistant
{"aesthetic":3.9,"technical":3.7}<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>【待评照片】以上 3 张示例到此
结束。请按细则为这张照片评分，
只输出这张照片的 JSON。<|im_end|>
<|im_start|>assistant
<think>

</think>

```

示例轮的完整文字是“【示例 i/3】这张照片已由人工评好，下一条回复是它的
人工分数。”；最后一轮“结束。”后面的换行只是排版，实际在同一行。

注意示例回复里的 JSON 没有空格（SFT 数据就是这样写的），
提示词里展示的格式有空格；两者都是合法 JSON，解析结果相同。

## 和 v1 的区别

v1 是视频提示词（三项分数加评语，对比 DOVER），已归档到
`archive/video-dover/prompts/baseline-v1/`。v2 换成图片、两项分数，
细则按 EVA 的打分定义重写：美学分对应 EVA 总体审美分，技术分对应
EVA 的“质量”属性（`docs/eva-scoring-standard.md`）。

## 改提示词的规则

- 已经跑过评测的版本不改内容：改动就新建 `prompts/baseline-v3/`，
  id 升版本，旧版本留着以便复现。
- B1 与 B2/B3 的评分细则必须一字不差，B2/B3 的 system 只多一段
  关于示例的说明；B2/B3 的示例轮和待评轮必须用不同的文字标明
  （`## example` 段带 `{i}` 编号），测试会检查。
- 每份提示词都必须写明输出格式：两个键、1.0–5.0、保留一位小数、
  只输出一个 JSON 对象；测试会检查。
