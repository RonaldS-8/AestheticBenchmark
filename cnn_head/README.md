# 美学头（独立子文件夹）

参考 DOVER，用视觉骨干加很小的美学、技术回归头，只输出分数，
不生成评语（ADR-0001）。用来和 LoRA 路线对比。

这个文件夹和主仓库解耦：代码、数据、权重、依赖、测试都在这里，
运行时不导入 `lora_sft`，也不读主仓库的文件。主仓库只在两处用到：
`prepare_data.py` 一次性复制数据，`tests/test_sync_with_main.py`
核对复制的部分没有走样。

## 现状

- 第一轮**冻结探针**已跑完（`docs/probe-native-v1.md`）：
  三个 ImageNet CNN 骨干，原图输入，多层池化特征加岭回归。
  val 上 ConvNeXt-T 最好：美学 SRCC 0.698，技术 0.679；
  ResNet-50 与它分不出高下，EfficientNet-B0 技术分明显差。
- 文献依据见 `docs/literature-review.md`。
- **ConvNeXt-T 微调**已在 A10 上跑完 9 组 sweep 并评了 eval
  （选中 `convnext-t-ft-lr1e-4-s1`，eval 美学 SRCC 0.768、
  技术 0.722）。步骤见下面“微调（上云）”。
- 两份对比报告：`docs/finetune-vs-frozen-v1.md`（微调对冻结，
  val）、`docs/head-vs-lora-v1.md`（美学头对 LoRA 和原版加
  提示词，eval）。带图的 HTML 版在 `dist/report-*.html`（不入库）。
- 定下部署路线后补一份 ADR。

## 已定

| 项 | 做法 | 依据 |
| --- | --- | --- |
| 数据 | EVA，与 LoRA 同一份划分，复制到 `data/` | ADR-0009 |
| 划分 | 种子 42，train 3,256 / val 407 / eval 407 | ADR-0009 |
| 目标 | 美学、技术两项，各按原量表线性映射到 1–5 | ADR-0009 |
| 总体分 | EVA 没有，不设总体头，也不补造 | ADR-0009 |
| 输入 | 原图尺寸、保宽高比，不缩放、不裁剪 | 实测前已定 |
| 骨干 | CNN；Qwen3.5-4B 视觉塔太慢，已排除 | 本机实测 |
| 指标 | SRCC / PLCC 对 EVA 投票均值，MAE 对 1–5 目标 | 同 LoRA |
| 选择 | 只看 val；eval 集只在最后跑一次 | ADR-0001 |

- eval 集永不进训练，探针也不抽 eval 的特征（代码里会报错）。
- 禁止兜底：读图失败、哈希不符、字段缺失都直接抛异常。
- 数据、特征缓存、报告只增不改，已存在就报错，不覆盖。

## 复制了主仓库的什么

| 这里 | 主仓库 | 核对方式 |
| --- | --- | --- |
| `aes_head/metrics.py` | `lora_sft/eval/metrics.py` | 同一组数结果相同 |
| `ScoreSummary` 与映射 | `lora_sft/crawl/dataset_schema.py` | 目标逐条相同 |
| eval 指纹算法 | `lora_sft/data/schema.py` | 指纹相同 |
| `data/`（清单 + 4,070 张图） | `data/manifests/eva/sft/v1/` 与图片 | 逐图哈希、逐条目标 |

主仓库的 `HeadSample` / `HeadConfig` 是给后面三分数阶段设计的
（要求有总体分），这里没有用，也没有改。

## 用法

在主仓库的 `.venv` 里运行（依赖版本与主仓库相同，见
`requirements.txt`）。

```bash
# 1. 从主仓库复制数据（只需一次；先 --dry-run 看看）
python cnn_head/prepare_data.py --dry-run
python cnn_head/prepare_data.py

# 2. 下载三个骨干的 ImageNet 权重到 models/，记录 sha256
cd cnn_head
python download_weights.py

# 3. 冻结探针：先冒烟，再全量
python run_probe.py --run-id probe-smoke --limit 40
python run_probe.py --run-id probe-native-v1
```

- 每个骨干的特征缓存在 `runs/<run_id>/features-<骨干>.pt`，
  中断后用同一个 `run_id` 重跑会复用；`report.json` 已存在则报错。
- `--limit` 只取 train、val 的前 N 张，报告里标为冒烟，不作结论。

## 探针怎么做

- 输入：图片按存储尺寸解码成 RGB，ImageNet 均值方差归一化，
  不缩放、不裁剪，一次一张。
- 特征：从细到粗四个阶段的输出各自做全局平均池化。
  `all` 是四层拼接，`last` 只用最后一层。

  | 骨干 | 取的层 | 维度 |
  | --- | --- | --- |
  | ResNet-50 | layer1–4 | 256 + 512 + 1024 + 2048 |
  | EfficientNet-B0 | features.2 / 3 / 5 / 8 | 24 + 40 + 112 + 1280 |
  | ConvNeXt-T | 四个 stage | 96 + 192 + 384 + 768 |

- 回归：特征按 train 的均值方差标准化，岭回归；正则强度在 train
  上 5 折交叉验证选（看平均 SRCC），再用全部 train 拟合，
  在 val 上算 SRCC、PLCC、MAE，SRCC 附 bootstrap 95% 区间。

## 微调（上云）

配方在 `configs/convnext-t-ft.toml`：

- 模型：ConvNeXt-T 四个 stage 的池化特征拼接（1,440 维），
  LayerNorm 后接美学、技术两个 MLP 头（隐层 256）。
  头预测 z 分数，模型里存着 train 的均值方差，换算回 1–5 分。
- 输入：原图尺寸，batch 内逐张前向，损失按整个 batch 算。
  ConvNeXt 只有 LayerNorm，所以这和真正的批处理完全等价，
  不用补边也不用缩放。
- 两段训练：
  1. 冻结骨干只训头 3 轮（lr 1e-3）。
  2. 全部放开训 10 轮：骨干 lr 由命令行给，头是骨干的 10 倍。
  两段都是 AdamW、warmup 5% 后余弦衰减，batch 16。
- 损失：每个分数 MSE + (1 − PLCC)/2 + 0.3 × 排序损失，
  都在 z 分数上算。增广只有水平翻转。
- 选检查点：每轮结束在 val 全量打分，留美学、技术平均 SRCC 最高
  的一轮。sweep 里按学习率分组，看各种子的平均，选最好的学习率，
  再取其中最好的种子。eval 只评这一个，评一次。
- 显存估计：本机实测每张 640×480 原图训练时约 0.65 GB 激活（fp32），
  A10 上 bf16 约减半，batch 16 约 6–8 GB。

```bash
# 本机：先 dry-run（检查数据、权重、配置，跑一次前向，不写文件）
python train.py --config configs/convnext-t-ft.toml --run-id x --seed 1 \
    --backbone-lr 3e-5 --device cpu --precision fp32 --dry-run
# 本机：打包（代码、data/、ConvNeXt-T 权重，约 650 MB）
python pack.py --config configs/convnext-t-ft.toml
```

把 `dist/aesthetic-head-convnext-t-ft.tar` 传到 AWS A10，然后：

```bash
tar -xf aesthetic-head-convnext-t-ft.tar && cd aesthetic-head
bash setup.sh
LRS="3e-5" SEEDS="1" bash task.sh start   # 先只跑一组看看
bash task.sh start                        # 整个 sweep（已完成的跳过）
EVAL=1 bash task.sh start                 # 全部跑完后：在 val 上选，再评 eval
```

`task.sh` 把 `run.sh` 放进 tmux 会话 `aes-head`，SSH 断开也不停，
全部输出追加到 `sweep.log`：

| 命令 | 作用 |
| --- | --- |
| `bash task.sh status` | 是否在跑、当前第几组、最新进度、各运行每轮 val、GPU |
| `bash task.sh log` | 跟踪 `sweep.log`（Ctrl-C 退出，不影响训练） |
| `bash task.sh attach` | 进入 tmux 会话（Ctrl-b d 离开） |
| `bash task.sh stop` | 中断训练；没跑完的运行目录要先改名或删除再启动 |

- 选模和评 eval 只在 `EVAL=1` 时做，试跑不会碰 eval 集。
- 启动前如果 `runs/` 里有没跑完的运行，`start` 会列出来并拒绝启动。

**看进度**：`bash task.sh log`，或者看单个运行的 `logs/<run_id>.log`。

- 每组开始时打印 `[第几组/共几组]` 和时间，开头一行列出数据量、
  每轮步数、两个阶段各几轮。
- 每 20 步一行：`[full 3/10] 步 140/203  loss …  lr …
  本轮已用 … 分，约剩 … 分`。
- 每轮结束：val 上美学、技术 SRCC 和均值，是否刷新最佳并存了
  检查点，本阶段还剩几轮、约多少分钟。
- `runs/<run_id>/history.jsonl` 每轮结束追加一行（loss 和 val
  指标），中途挂掉也留着。
- `run.sh` 里已设 `PYTHONUNBUFFERED=1`，经过 `tee` 或重定向
  也能实时看到。

- `run.sh` 默认骨干 lr 取 1e-5、3e-5、1e-4，种子 1、2、3，共 9 组，
  运行编号是 `convnext-t-ft-lr<lr>-s<种子>`。
- **9 组在同一个 Python 进程里依次跑**（`sweep.py`）。原因：
  cuDNN 每遇到一种新的图片尺寸要先准备一次卷积方案，进程内会缓存；
  EVA 有一千多种尺寸，A10 上实测第 1 轮 11.4 分钟、缓存后第 2 轮
  0.4 分钟。分进程跑每组都要重付一遍，同一进程只有第 1 组付。
  每组照样重新设种子、重建模型和数据加载，结果与分开跑相同。
- 已完成的运行会跳过；有没跑完的运行目录时，开跑前就报错停下，
  检查后改名（`aborted-<run_id>`）或删除再重跑。
- 每个运行留下 `runs/<run_id>/best.pt` 和 `train-summary.json`
  （每轮的 loss 和 val 指标）；`runs/convnext-t-ft-selection.json`
  是选择结果；`runs/convnext-t-ft-eval/` 是 eval 报告，
  格式与 LoRA 评测的 `report.json` 相同。日志在 `logs/`。
- 把 `runs/` 和 `logs/` 下载回本机的 `cnn_head/` 下。

## 待定

1. **NAS 推理尺度**：用 EVA 训练时，推理要缩到 EVA 尺度
   （长边约 640–800）。要做 1200 万像素级的技术分，需另找按
   高分辨率标注的数据。
2. **导出与 NAS 推理**：ONNX + onnxruntime；NAS 只有 AMD 核显，
   要确认用 CPU 还是其他执行后端。

## 目录

```text
aes_head/             代码
  schema.py           记录格式（数据集、权重、探针报告）
  data.py             读数据集、校验哈希、解码图片
  prepare.py          从主仓库复制数据的逻辑
  backbones.py        三个骨干与多层池化特征
  features.py         抽特征与缓存
  probe.py            岭回归、交叉验证、bootstrap
  run.py              探针整体流程与报告
  weights.py          下载并锁定权重
  metrics.py          SRCC / PLCC（主仓库的副本）
  config.py           微调配方（TOML）
  model.py            骨干 + 两个分数头
  losses.py           MSE、PLCC、排序损失
  finetune.py         两段训练、按 val 选检查点
  evaluation.py       评测与 LoRA 格式的报告
  selection.py        sweep 里选学习率和种子
  sweep.py            同一进程依次跑完整个 sweep
  records.py          训练、选择、评测的记录格式
  packing.py          打包上云
configs/              微调配方
prepare_data.py       命令行：复制数据
download_weights.py   命令行：下载权重
run_probe.py          命令行：跑探针
train.py              命令行：训练一个运行（含 --dry-run）
sweep.py              命令行：同一进程依次训练整个 sweep
select_run.py         命令行：选 sweep 结果
evaluate.py           命令行：评测
pack.py               命令行：打包（不上云）
setup.sh / run.sh     云端：建环境 / 跑 sweep、选择、评测
docs/                 文献调研与实验记录
tests/                测试（含与主仓库的一致性检查，不上云）
data/ models/ runs/   不入库：数据副本、权重、运行结果
logs/ dist/           不入库：日志、打包结果
```

测试在本文件夹里跑：`cd cnn_head && pytest`（分支覆盖率
低于 80% 失败）。lint 和格式化在主仓库根目录跑 `ruff check .`
和 `black .`。
