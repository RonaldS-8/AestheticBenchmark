# Qwen 视觉塔挂头（独立子文件夹）

冻结 Qwen3.5-4B 自带的视觉塔（不训练），只在它的特征上训练美学、
技术两个回归头，只输出分数，不生成评语。用来和 ConvNeXt-T 美学头
（`cnn_head/`）、LoRA SFT 在同一份 EVA 划分上对比。

这个文件夹和主仓库解耦：代码、数据、权重、依赖、测试都在这里，
运行时不导入 `lora_sft`，也不读主仓库的文件。主仓库只在两处用到：
`prepare_data.py` 一次性复制数据，`tests/test_sync_with_main.py`
核对复制的部分没有走样。

## 现状（2026-11-10，A10 上已跑完）

- 特征：3,663 张约 2 分钟，视觉塔每张 p50 30 ms、p95 44 ms。
- val 岭回归探针（美学 / 技术 SRCC）：`all` 0.783 / 0.764，
  `merger` 0.774 / 0.736，`last` 0.752 / 0.715。
- 头 sweep：lr 3e-4 的种子平均最好（0.771），选中 `lr3e-4-s3`
  （第 7 轮）；端到端 val 复核与缓存特征一致。
- **eval（只评一次，407 张）**：美学 SRCC 0.804、PLCC 0.830、
  MAE 0.179；技术 SRCC 0.756、PLCC 0.780、MAE 0.172；
  每张 p50 30 ms（A10，视觉塔 + 头）。
- 报告三（给客户看，比 Qwen3.5-4B 的速度、比 CNN 美学头的准确度）：
  `docs/tower-head-report-v1.md`；带图的 HTML 版在
  `dist/report-tower-head.html`（不入库）。
- 本机 `runs/` 只拉回了记录和报告；特征缓存和各组检查点（`.pt`）
  留在 A10 的 `~/lora-sft/qwen_vision_head/qwen-vision-head/runs/`
  （下载约 15 KB/s，太慢）。

## 已定

| 项 | 做法 | 依据 |
| --- | --- | --- |
| 数据 | EVA，与 LoRA 同一份划分（`sft/v2`），复制到 `data/` | ADR-0009 |
| 目标 | 美学、技术两项，各按原量表线性映射到 1–5 | ADR-0009 |
| 视觉塔 | Qwen3.5-4B 同一 revision，`model.visual.*` 原样切出，冻结 | 对照“未训练” |
| 输入 | 基础模型自带的图像处理器，像素预算同 LoRA 配置 | 与 LoRA 同输入 |
| 指标 | SRCC / PLCC 对 EVA 投票均值，MAE 对 1–5 目标 | 同 LoRA |
| 选择 | 只看 val；eval 集只在最后跑一次 | ADR-0001 |

- eval 集永不进训练，也不缓存特征；只在最后端到端评一次。
- 禁止兜底：读图失败、哈希不符、张量缺失或多余都直接抛异常。
- 数据、权重、特征缓存、报告只增不改，已存在就报错，不覆盖。

## 复制了什么

| 这里 | 来源 | 核对方式 |
| --- | --- | --- |
| `qv_head/metrics.py` | 主仓库 `lora_sft/eval/metrics.py` | 同一组数结果相同 |
| `ScoreSummary` 与映射 | 主仓库 `lora_sft/crawl/dataset_schema.py` | 目标逐条相同 |
| `EvaEvalReport` | 主仓库 `lora_sft/eval/eva_eval.py` | 同一组分数报告相同 |
| `data/`（清单 + 4,070 张图） | 主仓库 `data/manifests/eva/sft/v2/` | 逐图哈希、逐条目标 |
| `models/qwen35-4b-visual/` | `Qwen/Qwen3.5-4B` 的视觉塔 | `WEIGHTS.json` 记哈希 |
| 数据读写、探针、损失、选择 | `cnn_head/aes_head/` | 同一套做法 |

## 方法

- **视觉塔**：24 个块、宽 1024，约 3.3 亿参数；bf16、sdpa，一次一张。
  `job.py weights` 从基础模型的分片里只读 `model.visual.*`，严格
  载入 `Qwen3_5VisionModel`（缺或多一个张量都报错），按原精度存成
  `models/qwen35-4b-visual/`，连同图像处理器配置，`WEIGHTS.json`
  记下每个文件和来源分片的 sha256。
- **输入**：基础模型自带的图像处理器，`shortest_edge` 65,536、
  `longest_edge` 1,048,576 像素，与 LoRA 配置相同；EVA 图片
  （最多约 64 万像素）保持原尺寸，只按 32 像素对齐。
- **特征**：第 6、12、18、24 块的输出各自对所有 token 求平均
  （4 × 1024），加上 merger 输出（送进语言模型的那份，2560）的
  平均，拼成 6,656 维。train、val 只抽一次，缓存在
  `runs/<prefix>-features/`。
- **探针**：与 ConvNeXt 探针同一方法（岭回归，train 上 5 折交叉验证
  选正则，val 上报 SRCC 和 bootstrap 区间），分 `all`、`last`
  （第 24 块）、`merger` 三组，和 CNN 冻结探针直接可比。
- **头**：特征按 train 的均值方差标准化，美学、技术各一个 MLP
  （隐层 256，GELU，dropout 0.1），预测 z 分数再换算回 1–5。
  损失与 ConvNeXt-T 头相同：每个分数 MSE + (1 − PLCC)/2 +
  0.3 × 排序损失。AdamW，weight decay 0.05，warmup 5% 后余弦，
  batch 16，30 轮，每轮在 val 上打分，留平均 SRCC 最高的一轮。
  不做增广（特征是缓存的）。
- **sweep**：lr 取 1e-4、3e-4、1e-3，种子 1、2、3，共 9 组；按 lr 看
  种子平均，选最好的 lr，再取其中最好的种子。
- **评测**：选中的头接回视觉塔，图片 → 视觉塔 → 头端到端打分。
  先在 val 上重跑一遍，SRCC 与缓存特征上的差超过 0.01 就报错
  （说明两条路径不一致），通过后才评 eval，评一次。报告格式与
  LoRA 评测的 `report.json` 相同；延迟只算视觉塔加头在 GPU 上的
  时间，不含解码和切块。

## 用法

本机（主仓库的 `.venv`）：

```bash
python tower_head/prepare_data.py --dry-run   # 先看看
cd tower_head && pytest                      # 测试，覆盖率 ≥ 80%
python job.py pack                                  # 只打包代码
```

云端（A10 上已有主仓库的 EVA 数据和 Qwen3.5-4B 缓存，不用上传）：

```bash
tar -xf qwen-vision-head.tar && cd qwen-vision-head
bash setup.sh
python prepare_data.py --source-root ~/lora-sft/eva-qwen35-4b-lora-r16/data
HF_HOME=~/lora-sft/hf-cache HF_HUB_OFFLINE=1 .venv/bin/python job.py weights
bash task.sh start           # 特征、探针、9 组头
EVAL=1 bash task.sh start    # 看过 val 后：选模、val 复核、评 eval
```

`task.sh` 把 `run.sh` 放进 tmux 会话 `qv-head`，SSH 断开也不停，
全部输出追加到 `job.log`；`status`、`log`、`attach`、`stop` 用法与
`cnn_head/` 相同。已完成的步骤重跑时跳过；没跑完的运行目录
会让 `start` 拒绝启动，检查后改名为 `aborted-<id>` 或删除。

## 结果在哪

| 文件 | 内容 |
| --- | --- |
| `runs/qwen35-4b-tower-head-features/features.json` | 特征维度、抽取延迟 |
| `runs/qwen35-4b-tower-head-probe/report.json` | 岭回归探针（val） |
| `runs/qwen35-4b-tower-head-lr*-s*/train-summary.json` | 每组每轮 val |
| `runs/qwen35-4b-tower-head-selection.json` | 选中的 lr 与种子 |
| `runs/qwen35-4b-tower-head-val/report.json` | 端到端 val 复核 |
| `runs/qwen35-4b-tower-head-eval/report.json` | eval，格式同 LoRA |

## 目录

```text
qv_head/              代码
  schema.py           记录格式（数据集、视觉塔权重、探针报告）
  data.py             读数据集、校验哈希、解码图片
  prepare.py          从主仓库复制数据的逻辑
  weights.py          从基础模型切出视觉塔并锁定哈希
  tower.py            载入视觉塔、多层池化特征
  features.py         抽 train/val 特征并缓存
  probe.py            岭回归、交叉验证、bootstrap
  probe_run.py        按层组跑探针、写报告
  head.py             两个分数头与检查点
  losses.py           MSE、PLCC、排序损失
  training.py         在缓存特征上训头、sweep
  selection.py        sweep 里选学习率和种子
  evaluation.py       端到端评测与 LoRA 格式的报告
  records.py          特征、训练、选择、评测的记录格式
  metrics.py          SRCC / PLCC（主仓库的副本）
  packing.py          打包上云（只含代码）
configs/              配置
job.py                命令行：weights / features / probe / sweep /
                      select / evaluate / pack
prepare_data.py       命令行：复制数据
setup.sh / run.sh     云端：建环境 / 跑全流程
task.sh               云端：tmux 任务管理
tests/                测试（含与主仓库的一致性检查，不上云）
data/ models/ runs/   不入库：数据副本、视觉塔、运行结果
```

测试在本文件夹里跑；lint 和格式化在主仓库根目录跑 `ruff check .`
和 `black .`。
