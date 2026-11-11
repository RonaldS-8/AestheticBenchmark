# AestheticBenchmark：照片美学打分三方案对比

给照片打美学、技术两项分数（1–5），比较三种做法在同一份 EVA
评测集上的准确度和速度，目标是在 NAS 上本地推理。

## 结果（第一版，EVA eval 407 张）

所有方案用同一批图、同一段指标代码，只在 val 上选模，eval 各评
一次。SRCC / PLCC 对 EVA 投票均值，越接近 1 越好；MAE 对 1–5 分
目标，越小越好。

| 方案 | 美学 SRCC | 美学 PLCC | 技术 SRCC | 技术 PLCC |
| --- | --- | --- | --- | --- |
| 基线：原版 + 提示词 | 0.490 | 0.507 | 0.561 | 0.602 |
| 方案一：LoRA | **0.810** | **0.834** | **0.764** | **0.789** |
| 方案二：CNN 美学头 | 0.768 | 0.775 | 0.722 | 0.740 |
| 方案三：视觉塔挂头 | 0.804 | 0.830 | 0.756 | 0.780 |

| 方案 | 美学 MAE | 技术 MAE | A10 每张中位 | 参数 |
| --- | --- | --- | --- | --- |
| 基线：原版 + 提示词 | 0.552 | 0.734 | 1,056 ms | 约 45 亿 |
| 方案一：LoRA | 0.173 | 0.166 | 1,044 ms | 约 45 亿 + LoRA |
| 方案二：CNN 美学头 | 0.197 | 0.188 | 5.4 ms | 2,860 万 |
| 方案三：视觉塔挂头 | 0.179 | 0.172 | 30 ms | 3.37 亿 |

- CNN 美学头的 5.4 ms 去掉了进程里首次遇到某种图片尺寸的 199 张
  （每张慢约 1 秒）；`report.json` 里原始 p50 为 6.6 ms。
- 配对 bootstrap：视觉塔挂头与 LoRA 分不出高下（均值差 −0.007，
  区间含 0）；比 CNN 美学头高 0.035（区间 +0.0004 到 +0.068）。
- 结论简述：要评语选 LoRA；要在设备上又快又准选视觉塔挂头；
  资源最紧选 CNN 美学头。细节见各方案的报告。

数字汇总在 `results/summary.csv`，由各方案的 `report.json` 生成。
三个方案的完整技术报告（带图）：
`docs/eva-aesthetic-technical-report.html`，下载后用浏览器打开。
报告内嵌 12 张 EVA 示例照片，**仅限研究用途**。

## 三个方案

| 文件夹 | 方案 | 训练了什么 | 报告 |
| --- | --- | --- | --- |
| `lora/` | Qwen3.5-4B + LoRA r16 | LoRA（只改语言模型） | `lora/docs/` |
| `cnn_head/` | ConvNeXt-T + 两个回归头 | 骨干和头 | `cnn_head/docs/` |
| `tower_head/` | Qwen3.5-4B 视觉塔（冻结）+ 两个头 | 只训头 | `tower_head/docs/` |

- `lora/` 也是数据和基线的来源：`lora/lora_sft/` 里有 EVA 下载、
  标注映射、train / val / eval 划分；`lora/prompt_tuning/` 是基线
  “原版 + 提示词” 的提示词调优。两个挂头方案用 `prepare_data.py`
  从 `lora/data/` 复制同一份划分，文档里的 “主仓库” 指 `lora/`。
- 每个方案自带依赖、测试和运行说明，见各自的 `README.md`。
- 每个方案的 `results/` 放第一版 eval 结果：`report.json`、
  `config.json`、逐张预测 `predictions.jsonl`、选模记录。

## 目录

```text
README.md            本文件：结果总表与目录说明
docs/                公用文档
  eva-aesthetic-technical-report.html  三方案技术报告（仅研究用）
  overview.md        项目总览、技术选型、数据说明
  adr/               关键决定及原因（ADR）
  eva-scoring-standard.md, training-data-design.md, source-review-eva.md
results/             公用结果
  summary.csv        四个方案的 eval 指标汇总
  prompt-baseline/   基线（原版 + 提示词）的 eval 结果
lora/                方案一：LoRA SFT（含数据准备与基线提示词调优）
cnn_head/            方案二：CNN 美学头
tower_head/          方案三：Qwen 视觉塔挂头
```

## 数据

- 数据集：EVA（Kang 等 2020），从 AVA 选出的 4,070 张图，每张
  30–46 票。仅限研究用途，**本仓库不含图片、标注和清单**，按
  `lora/` 的说明自行下载生成。例外是技术报告里的 12 张示例照片，
  同样仅限研究用途。
- 划分：种子 42，train 3,256 / val 407 / eval 407；eval 集指纹
  `fe56c426…11a3`，永不进入训练。
- 模型权重和检查点也不入库。
