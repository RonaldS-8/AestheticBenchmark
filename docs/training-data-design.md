# 训练数据与结构设计：LoRA SFT 与美学头

2026-11-02 · 适用于 schema 版本 1.0

## 0. 范围与结论

- **做**：LoRA SFT（Qwen 视觉语言模型）和 DOVER 式美学头。
- **暂不做**：DPO、PPO、全参 SFT。
- **任务形式**：打分加一句评语，以打分为主。
  门禁只看分数指标，评语只做人工抽检。
- **第一版对比在视频上**：提示词调优的 VLM 对比 DOVER 原版，
  都不训练，见第 12 节。
- **代码即规范**：下文所有结构都在 `lora_sft/` 里用 pydantic
  定义，并有测试。本文负责解释，两者冲突时以代码为准。

## 1. 数据流

```text
PhotoRecord            爬虫产出（crawl/schema.py）
  │  image_id / image.path / sha256 / source
  ▼
AnnotationRecord       每张图多人标注 + 均分（data/schema.py）
  ├─ SplitManifest     train / val / eval，eval 冻结
  ├─ SftSample         LoRA SFT 样本（data/samples.py）
  └─ HeadSample        美学头样本（data/head_samples.py）
        ▲
        └─ 站点评分（SiteRating）也可直接生成弱标签样本
  ▼
TrainConfig            LoraSftConfig | HeadConfig（train/config.py）
  ▼
EvalRunConfig → Prediction → EvalReport → GateDecision
                       （eval/schema.py）
  ▼
ExportManifest         LoraExportManifest | HeadExportManifest
                       （export/schema.py）
```

两条路线共用标注、划分、评测集和门禁，所以结果可以直接对比：

- LoRA 输出分数和评语。
- 美学头只输出分数，体积小、速度快。

## 2. 通用约定

- **存储**：每种记录存成 JSONL，一行一条，带 `schema_version`。
  训练配置用 TOML（标准库 `tomllib` 读取，不增加依赖）。
- **严格**：所有记录不可变、拒绝未知字段、没有默认值。
  “没有这个值” 必须显式写 `null`，不能省略。
- **路径**：
  - 数据记录里的路径相对 `data/`，如 `images/photo_000123.jpg`。
  - 配置里的路径相对仓库根，如 `data/manifests/split-v1.json`。
  - 导出清单里的路径相对导出包根目录。
- **时间**：一律带时区。
- **分数**：统一 1–5 分。
  - 标注者打整数。
  - 均分保留 2 位小数。
  - SFT 目标保留 1 位小数。
- **ID 格式**：

| 名称 | 格式 | 例子 |
| --- | --- | --- |
| 图片 | `photo_` + 至少 6 位数字 | `photo_000123` |
| 运行 / 导出 | kebab-case | `qwen-4b-lora-r16` |
| 提示词 | 文件名@版本 | `rubric-score@v1` |
| 划分 | `split-v` + 数字 | `split-v1` |
| 标注规范 | `v` + 数字 | `v1` |
| 标注者 | `A` + 至少 2 位数字 | `A01` |

## 3. 标注记录 `AnnotationRecord`

每张图一条，结构与评测基线计划里的例子一致，另外补了几个字段。
示例见 `tests/fixtures/annotation_record.json`。

| 字段 | 说明 |
| --- | --- |
| `image_id` / `image_path` | 来自 PhotoRecord，文件名必须等于 id |
| `image_sha256` | 用来检查同一文件是否同时进了训练集和评测集 |
| `category` | 9 个类别，见下 |
| `source` | 站点、站内照片 id、授权类型 |
| `guideline_version` | 标注规范版本，规范一改就升版本 |
| `annotations` | 每位标注者一条，见下 |
| `mean` | 三项均分，写入时校验必须等于实际均值 |

每位标注者的一条 `Annotation`：

| 字段 | 说明 |
| --- | --- |
| `aesthetic` / `technical` / `overall` | 1–5 整数 |
| `reason` | 一句理由，1–100 字 |
| `highlights` / `issues` | 多选维度：亮点、问题，可以为空 |
| `annotated_at` | 标注时间 |

**维度**（8 个）：

- 美学：构图 `composition`、光线 `lighting`、色彩 `color`、
  主体 `subject`。
- 技术：对焦 `focus`、曝光 `exposure`、噪点 `noise`、
  模糊 `blur`。

**类别**：`portrait` 人像、`landscape` 风景、
`family_daily` 家庭日常、`night` 夜景、`pet`、`food`、
`architecture`、`street`、`other`。

**为什么用多选维度，而不是给 8 个维度各打一个子分**：
标注成本几乎不增加，又能回答基线计划的第 3 个问题
（剩余差距集中在哪些维度）。
一个维度不能同时既是亮点又是问题。

### 3.1 外部数据集双评分记录

EVA 一类公开数据集不是项目人工标注，因此不转换成
`AnnotationRecord`，也不补造评语或 `overall`。爬虫分别输出：

- `DatasetScoreRecord`：原始量表上的美学、技术评分统计。
- `DatasetImageRecord`：图片相对路径、SHA-256、大小和 JPEG 完整性
  标记。
- `DatasetCrawlManifest`：来源 URL、清单路径、样本数和排除数。

两类记录以 `dataset + source_image_id` 关联。进入训练样本生成阶段后，
才把不同原始量表线性映射到 1–5；缺失的 `overall` 保持缺失，不用
美学分代替。

### 3.2 EVA 第一阶段 Qwen3.5 SFT

按 ADR-0009，第一阶段不等待项目自建人工标注，只用 EVA 已有评分训练
`Qwen/Qwen3.5-4B`。这批数据使用独立的 `EvaSftSample`：

- `messages` 是 `system/user/assistant` 三轮；user 内容按官方模板放
  `{"type":"image","image":"images/eva/<id>.jpg"}` 和文本。
- assistant 只输出 `aesthetic`、`technical` 两个 1–5 分。
- 不生成 EVA 没有的 `overall` 与 `comment`。
- 每条样本同时保留图片 SHA-256、原始量表、均值、标准差、投票人数
  和 capture id。
- Qwen chat template 固定 `enable_thinking=false`，训练只计算
  assistant token 的损失。

生成命令：

```text
python scripts/prepare_eva_sft.py --dry-run
python scripts/prepare_eva_sft.py
```

输出放在 `data/manifests/eva/sft/v1/`，包含 train、val、eval JSONL、
固定划分和完整生成清单。目标目录存在时拒绝覆盖。

## 4. 数据划分 `SplitManifest`

- **三份**：
  - `train` 用于训练。
  - `val` 用于训练过程中挑 checkpoint、调参。
  - `eval` 只用于门禁，**不能拿来调参**。
- **`eval_fingerprint`**：eval 图片 id 排序后算 sha256。
  eval 列表一旦改动，指纹就对不上，训练配置、评测报告、
  门禁决定都会记录这个指纹。
- **`check_split`**：训练前检查三件事：
  - 每张图都有标注。
  - eval 图片至少 3 人标注。
  - eval 与 train/val 之间没有相同文件（按 sha256）。
- **`ensure_not_eval`**：检查任意一批训练样本有没有用到 eval 图片，
  包括没有人工标注的站点评分样本。
- **建议**：eval 取 500–1,000 张，按类别分层抽样。

## 5. LoRA SFT 样本 `SftSample`

示例见 `tests/fixtures/sft_sample.json`。
`messages` 直接用 Qwen VL processor 的格式，
训练时不需要再转换：

```text
[system（可选，纯文本）]
user:      1 张图片 + 指令文本
assistant: 一段 JSON 文本（AestheticOutput）
```

**输出契约 `AestheticOutput`**：

```json
{"aesthetic": 4.3, "technical": 2.7, "overall": 3.7,
 "comment": "逆光氛围温暖，构图干净；主体对焦偏软。"}
```

- 只能有这 4 个键，分数在 1–5 之间，评语 1–60 字。
- 分数在前、评语在后：模型先生成分数，评语不影响打分。
  评语上限 60 字（约一句话），也是为了减少 NAS 上的生成延迟。
- 评测时用同一个模型严格解析，见第 8 节。

**目标值**：标注均分四舍五入到 1 位小数，
由 `check_sample_matches_annotation` 校验。

**评语来源 `comment_provenance`**：

| `method` | 含义 | 是否必须复核 |
| --- | --- | --- |
| `human_written` | 人直接写 | 否 |
| `llm_merged` | LLM 把标注者理由合成一句 | 是，`reviewed_by` 必填 |

**训练**：损失只计算 assistant 部分的 token。

## 6. 美学头样本 `HeadSample`

美学头直接从像素回归分数，样本只有图片、目标分数和权重。

| 字段 | 说明 |
| --- | --- |
| `label_source` | `human_mean` 人工均分，或 `site_rating` 站点评分 |
| `targets` | 三项目标；站点评分样本只有 `overall` |
| `weight` | 损失权重，取值 (0, 1] |

**站点评分换算**：线性映射到 1–5 分，

```text
score = 1 + 4 × (mean − scale_min) / (scale_max − scale_min)
```

**两阶段训练**：

1. 阶段 1：用大量站点评分样本预训练，只学 `overall`。
2. 阶段 2：用人工均分微调，配置里 `init_from_run` 指向阶段 1。

**`weight` 建议**（由样本生成脚本决定，schema 只限定范围）：

- 人工样本取 1.0，或按标注者分歧降权。
- 站点样本按评分人数降权，如 `min(1, count / 100)`。

## 7. 训练配置

两条路线共用一个加载函数 `load_train_config`，按 `route` 区分。
示例见 `tests/fixtures/lora_sft.example.toml` 和
`tests/fixtures/head.example.toml`；EVA 第一阶段的实际配置是
`configs/eva-qwen35-4b-lora-r16.toml`，训练栈见 ADR-0010。

**共有部分**：

- `data`：划分文件、划分版本、eval 指纹、train/val 样本文件。
- `optim`：学习率、轮数、batch、梯度累积、warmup、
  weight decay、调度器、梯度裁剪。
- `runtime`：
  - `single` / `ddp` / `fsdp`，**不支持 DeepSpeed**。
  - GPU 数、精度、梯度检查点、保存与评估间隔、输出目录。

**`LoraSftConfig`**（`route = "lora_sft"`）：

| 字段 | 说明 |
| --- | --- |
| `base_model` | HF 模型 id + 40 位 commit，必须锁定版本 |
| `prompt_id` | 训练用的提示词，评测时必须一致 |
| `chat_template_sha256` | 对话模板哈希，推理端必须一致 |
| `preprocess` | `min_pixels` / `max_pixels`，推理端必须一致 |
| `max_seq_len` | 最大序列长度 |
| `lora` | `r`、`alpha`、`dropout`、`target_modules`、`freeze_vision` |

**`HeadConfig`**（`route = "head"`），参照 DOVER 的双分支结构：

```text
美学分支：原图整图缩放到 size        → 美学骨干 → 美学头
技术分支：切 grid×grid 个原分辨率小块 → 技术骨干 → 技术头
总体头：  两个分支的池化特征拼接     → 总体头
```

| 字段 | 说明 |
| --- | --- |
| `views.aesthetic` | 骨干 + `size`：整图缩放，看构图、色彩、主体 |
| `views.technical` | 骨干 + `grid` + `patch`：原分辨率小块拼接 |
| `freeze_backbones` | 是否冻结骨干 |
| `head` | MLP 隐层维度、dropout |
| `loss` | `mse`，或 `plcc_rank`（PLCC + 排序损失，需填 `rank_weight`） |
| `label_sources` | 用哪些来源的样本 |
| `init_from_run` | 从哪次运行接着训练，阶段 1 填 `null` |

为什么技术分支要切原分辨率小块：缩放会把噪点、模糊、
失焦一起抹掉，原像素小块能保留这些信息。

## 8. 评测与门禁

**评测运行**按 `route` 分三种：

- `VlmEvalRunConfig`（`vlm`）：
  - 方法：`b0_zero_shot`、`b1_rubric`、`b2_few_shot`、
    `b3_retrieval`、`lora_sft`。
  - B4（模型尺寸对照）不是单独的方法：
    换一个 `model` 再跑同一种方法即可。
  - 只有 B2 / B3 填 `few_shot_k`（1–8）。
  - 只有 `lora_sft` 填 `train_run_name`。
  - 端侧运行（llama.cpp / MLX）必须填 `export_id`。
  - `media` 填 `image` 或 `video`；视频必须填 `video_sampling`
    （均匀抽帧，`num_frames` 1–64），每帧像素预算用 `preprocess`。
- `HeadEvalRunConfig`（`head`）：
  云端用 `torch` 运行；NAS 上用 `onnxruntime` 运行，并填 `export_id`。
- `DoverEvalRunConfig`（`dover`）：DOVER 原版预训练权重，只推理。
  锁定权重哈希、DOVER 代码 commit 和配置文件哈希；
  抽帧方式由 DOVER 自己的配置决定。

**单条预测 `Prediction`**（照片或视频，用 `media_id` 标识）：

- VLM 的原始输出（`raw_output`）全部保留。
- `parse_prediction` 严格解析：不去掉 markdown 代码块，
  不修补残缺 JSON。解析失败就记下 `parse_error` 并计入失败率。
- 美学头和 DOVER 的预测只有分数。
- 分数保留模型自己的尺度（DOVER 不是 1–5 分），
  只要求是有限数；相关性指标与尺度无关。

**汇总 `EvalReport`**：

| 字段 | 说明 |
| --- | --- |
| `metrics` | 三项分数各自的 SRCC / PLCC（只算解析成功的） |
| `by_category` | 按类别拆分的同样指标 |
| `n_parsed` / `n_items` | 解析成功率，必须和相关性一起看 |
| `pairwise_agreement` | 两两比较一致率，见下 |
| `latency_p50_ms` / `latency_p95_ms` | 延迟 |
| `annotator_ceiling` | 标注者一致性上限；数据集只有均分时为 `null` |

- **两两一致率**：从单图的总体分推出来，不单独让模型比较两张图。
  只取同类别、人工均分差 ≥ `pairwise_min_gap` 的图对，
  看模型给的分数是否和人工排出相同的先后。
- **标注者上限**：建议用留一法，即每位标注者的分数
  和其余标注者均分算相关性，再取平均。

**门禁 `GateDecision`**：

- 阈值有 4 项，**都没有默认值**，等标注一致性测出来后再定：
  - `min_overall_srcc_gain`：相对最佳基线，总体分 SRCC 至少提升多少。
  - `min_parse_success_rate`：最低解析成功率。
  - `max_nas_srcc_drop`：NAS 导出版相对云端最多掉多少。
  - `max_latency_p95_ms`：p95 延迟上限。
- 判定通过时，必须已经在 NAS 上评测过导出模型（`nas_run_id`）。
- `passed` 必须与 `failures` 是否为空一致。
- 评语质量不进门禁，单独人工抽检。

## 9. 导出清单

两种导出共有的字段：`export_id`、`train_run_name`、
`checkpoint_sha256`、量化标签、运行时版本、文件列表（每个文件带
sha256 和用途）。文件列表里至少要有一个 `weights` 文件。

| | LoRA（`lora_sft`） | 美学头（`head`） |
| --- | --- | --- |
| 格式 / 运行时 | GGUF + llama.cpp，或 MLX + mlx | ONNX + onnxruntime |
| 必带文件 | GGUF 需恰好一个 mmproj 视觉投影文件 | 不能有投影或分词器文件 |
| 一致性字段 | 基座、对话模板哈希、像素预算 | 两个视图的全部设置 |
| `checkpoint_sha256` | LoRA 适配器文件 | 美学头 checkpoint |

## 10. 对爬虫侧的要求

- **sha256 唯一**：同一个 sha256 只分配一个 `image_id`。
  `ensure_not_eval` 按 id 判断泄漏，依赖这一点。
- **字段映射**：PhotoRecord 到 AnnotationRecord 是直接映射：
  - `image_id` 不变。
  - `image.path` → `image_path`。
  - `image.sha256` → `image_sha256`。
  - `source.site`、`source.site_photo_id` 不变。
  - `source.license.license_type` → `source.license_type`。
- **站点评分**：美学头阶段 1 会用到，尽量抓全
  `rating` 的均值、人数和评分区间。
- **外部数据集**：双评分数据写入 `DatasetScoreRecord`，图片哈希写入
  `DatasetImageRecord`；它们不是 `PhotoRecord` 或人工标注。

## 11. 待定事项

- **基座**：Qwen3.5（4B / 9B）还是 Qwen3-VL（4B / 8B）。
- **美学头骨干**：示例配置里的 ConvNeXt-T + Swin-T 只是占位。
  另一个选项是两路都用 Qwen 的视觉编码器，
  这样 NAS 上的两个模型可以共用一个编码器。
- **门禁阈值**：4 项都等标注一致性测出来后再定。
- **评语合成方式与复核比例**：人写，还是 LLM 合成后人工复核。
- **严格解析**：代码块包裹的 JSON 是否算失败（EVA 双分数已按
  ADR-0013 去掉外壳再解析；三分数加评语的解析器仍算失败）。
- **评语长度**：上限 60 字是否合适。
- **站点样本 `weight` 公式**：如何按评分人数降权。
- **视频评测集来源**：DIVIDE-MaxWell（DIVIDE-3k 超集），授权待确认，
  见第 12 节和 ADR-0005。

## 12. 第一版对比：视频上的提示词基线 vs DOVER

**目的**：在不训练的前提下，先看 “VLM + 好提示词” 和专门的
视频质量模型 DOVER 差多少，决定后面走 LoRA 还是美学头。

| | VLM 提示词基线 | DOVER |
| --- | --- | --- |
| 输入 | 均匀抽 `num_frames` 帧 | DOVER 自带采样（片段 + 小块） |
| 方法 | B0–B3，B4 换模型尺寸 | 原版预训练权重 |
| 输出 | 1–5 分 + 评语 | 美学、技术分和融合后的总体分 |
| 配置 | `VlmEvalRunConfig` | `DoverEvalRunConfig` |
| 提示词 | `prompts/baseline-v1/` | 不需要 |

**视频标签 `VideoLabelRecord`**（`lora_sft/data/video.py`）：

| 字段 | 说明 |
| --- | --- |
| `video_id` / `video_path` | `video_000123` / `videos/video_000123.mp4` |
| `video_sha256` | 防止同一文件同时出现在评测集和示例池 |
| `dataset` / `dataset_item` | 数据集名和原始文件名 |
| `duration_s` | 时长（秒），未测量时为 `null` |
| `mos` / `scale` | 数据集公布的三项均分及其评分区间 |
| `n_ratings` | 评分人数，数据集不公布时为 `null` |

- 公开数据集只有均分、没有逐人标注，所以不套用 `AnnotationRecord`，
  也没有标注者上限。
- `SplitManifest` 同时支持照片和视频 id，但一份划分只能有一种；
  `check_video_split` 检查标签齐全、来自同一数据集、
  评测集和 train/val 没有相同文件。
- train/val 只用来给 B2/B3 提供示例，不训练。

**少样本示例 `FewShotExample`**（`lora_sft/eval/prompts.py`）：
示例视频 + 目标回复（分数和评语）+ 评语来源。
公开数据集没有评语，示例的评语要人工补写。

**需要注意**：

- 抽帧看不到卡顿、掉帧、闪烁，技术分上 VLM 可能吃亏。
- 少样本很占上下文，B2/B3 建议 k ≤ 3。
- DOVER 有多个权重版本，其中有的用 DIVIDE 训练过（据记忆是
  DOVER++，原版用 LSVQ），在 DIVIDE 上评测会偏高。
  正式跑之前要核对所选权重的训练数据，用没见过评测集的版本，
  并在 `DoverEvalRunConfig.checkpoint` 里写清楚。
- DIVIDE-MaxWell 的下载和使用授权仍待确认，只用于内部评测。
- 导入：`scripts/import_divide.py`（先 `--dry-run`），
  标签列顺序为 文件名、美学、技术、总体，取自 DOVER 读取标签的代码；
  官方验证集作 eval，官方训练集按种子切出 val。
  目录约定见 ADR-0007。
- 官方文件里有 13 对字节相同的视频（7 对跨官方 train/val）。
  导入按 sha256 每个文件只留一行：val 优先，其次文件名小的；
  丢弃的行写在 `import.json` 的 `dropped_duplicates`。
  导入后 4530 段：train 3258、val 363、eval 909。
