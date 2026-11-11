# 云端训练操作手册（EVA 第一阶段 LoRA）

本机没有 NVIDIA 显卡，训练放到组里的 AWS A10 服务器。LoRA 必须用
提示词调优选出的提示词训练（ADR-0012），所以顺序是：

1. 提示词调优（`prompt_tuning/` 上云包，见其 README）→ 下载结果。
2. 本机：切换到选定提示词 → 提交 → 打包。
3. A10：建环境 → 一条命令完成预检、训练、评测。
4. 本机：下载结果 → 算 LoRA 相对同一提示词的提升。

## 1. 机器

| 项目 | 配置 | 对训练的影响 |
| --- | --- | --- |
| 显卡 | A10 24 GB（Ampere，sm_86），驱动 570 | bf16、flash-attn、FLA 都支持 |
| CPU / 内存 | 36 核 / 440 GB | 数据加载不是瓶颈 |
| 系统盘 | 128 GB，剩约 85 GB，**关机不丢** | 仓库、`.venv`、模型、输出都放这里 |
| 临时盘 | `/mnt` 1.4 TB，**关机清空** | 只放 pip 缓存这类可重下的 |
| Python | `~/torch-env`（3.12 + torch 2.11），登录自动激活 | 不用它的 torch，另建 `.venv` |
| 网络 | 直连 Hugging Face 约 800 Mbps | 不设 `HF_ENDPOINT` |
| 账号 | 普通账号，无 sudo | 只用 pip，不装系统包 |

登录方式、私钥和 VS Code 远程见 `docs/A10接入说明.md`。

系统盘大概要用 35–45 GB：`.venv` 约 8 GB、模型缓存 9.3 GB、
上传包 0.6 GB、检查点每个约 0.4 GB（适配器约 0.12 GB + AdamW 状态，
全程 6 个约 2.2 GB）；提示词调优包也在这台机器上的话，它的环境和
权重再加约 17 GB。pip 缓存放 `/mnt` 可以省约 4 GB。

显存：Qwen3.5-4B bf16 权重约 9 GB；每卡 batch 4、开梯度检查点，
最长 1,119 token 的样本也在 24 GB 以内（估算，未实测）。

## 2. 本机：切换提示词、打包

提示词调优结束后，把它的 `runs/` 下载回来：各运行目录放进
`data/runs/`，`selection.json` 放到
`data/manifests/eva/prompt-tuning/v1-selection.json`。

```bash
python scripts/apply_prompt_selection.py \
    --selection data/manifests/eva/prompt-tuning/v1-selection.json --dry-run
python scripts/apply_prompt_selection.py \
    --selection data/manifests/eva/prompt-tuning/v1-selection.json
```

它用选定的 `lora_prompt_id` 生成 `data/manifests/eva/sft/v2/`（种子、
比例与 v1 相同，并核对 train/val/eval 三份 id 和 eval 指纹完全不变），
再把训练配置里的 `prompt_id` 和 `[data]` 四个路径改过去，配置其他部分
一个字都不动。预览会打印配置的改动；目标版本已存在时拒绝覆盖。

`max_seq_len = 1280` 已经按最长的候选（`rubric-score@v2`，1,119 token）
留好，选哪个提示词都不用再改配置。

然后提交，再检查、打包：

```bash
python scripts/train_lora_sft.py --config configs/eva-qwen35-4b-lora-r16.toml \
    --eval-samples data/manifests/eva/sft/v2/eval.jsonl --dry-run
python scripts/pack_for_cloud.py --config configs/eva-qwen35-4b-lora-r16.toml \
    --selection data/manifests/eva/prompt-tuning/v1-selection.json \
    --output dist/eva-qwen35-4b-lora-r16.tar
```

dry-run 检查配置、划分、评测集泄漏和全部图片哈希。打包前先加
`--dry-run` 可以只看文件数和大小（约 574 MB）；tar 已存在时拒绝覆盖。
配置的提示词不是选定的那个时，打包直接报错。

包里是 git 跟踪的代码（含未提交的新文件）、当前样本目录、样本引用到的
4,070 张图片和选择文件，不含 `data/raw/`（18 GB）。包里没有 `.git`，
所以另写一个 `BUILD_INFO.json`（commit、未提交改动清单、打包时间），
训练把它记进 `run_info.json` 的 `source`。有未提交改动时打包会提示；
**先提交再打包**，结果才能对上代码版本。

## 3. A10：建环境、训练

```bash
scp -i <ssh-key> dist/eva-qwen35-4b-lora-r16.tar <user>@<a10-host>:~/
ssh -i <ssh-key> <user>@<a10-host>
```

登录后，在家目录（系统盘）里：

```bash
tar -xf eva-qwen35-4b-lora-r16.tar && cd Lora-SFT
export PIP_CACHE_DIR=/mnt/pip-cache      # 可选：pip 缓存放临时盘
bash scripts/cloud/setup.sh
tmux new -s train                         # SSH 断开不会杀掉训练
bash scripts/cloud/train.sh configs/eva-qwen35-4b-lora-r16.toml
```

不要解包到 `/mnt`：关机就清空，检查点和结果全丢。

`setup.sh` 用当前的 `python3`（登录自动激活的 `~/torch-env`，3.12）
建一个独立的 `.venv`，不动 `torch-env`。`train.sh` 自己激活 `.venv`，
`torch-env` 激活着也没关系。版本按“每个 CUDA 内核都有官方预编译包”定，
不现场编译，没有 sudo 也能装：

| 包 | 版本 | 约束 |
| --- | --- | --- |
| torch / torchvision | 2.8.0 / 0.23.0 | PyPI 版带 CUDA 12.8，驱动 570 支持 |
| transformers | 5.20.2 | 要 torch ≥ 2.6 |
| flash-linear-attention | 0.5.2 | 要 torch ≥ 2.7、triton ≥ 3.3 |
| causal-conv1d | 1.7.0 | 预编译包覆盖 torch 2.6–2.10 |
| flash-attn | 2.8.3.post2 | 预编译包只覆盖 torch 2.4–2.8、CUDA 12 |

torch 2.8 是这些包共同支持的唯一版本。后两个以官方 wheel 直链写在
`requirements-train.txt` 里，按 Python 版本选包；cp312 的两个包已确认
可以下载。

`setup.sh` 最后检查 CUDA 12、两组线性注意力内核和 flash-attn 都能导入，
缺一个就失败。这一步很重要：32 层里 24 层是 Gated DeltaNet，缺内核时
transformers 不报错、静默走纯 torch 实现。EVA 配置写了
`kernels.require_fla = true` 和 `require_causal_conv1d = true`，训练开始前
还会再强制检查一次；内核状态记在 `run_info.json` 的
`environment.linear_attention_kernels`。

`train.sh` 依次执行：

| 步骤 | 内容 |
| --- | --- |
| 1 预检 | 校验 chat template 哈希和内核，编码全部 train/val 样本，超长就停 |
| 2 训练 | 3 个 epoch、612 步；每 102 步在 val 上解码打分选检查点 |
| 3 LoRA 评测 | 最佳检查点分别在 val（407 张）和 eval（407 张）上打分 |

样本目录从配置里读，训练和评测一定用同一个提示词；评测的
`max_new_tokens` 也从配置读，与提示词调优一致。“同一提示词不加 LoRA”
不在这里跑，提示词调优已经算过（ADR-0012）。

耗时估计（未实测）：权重下载约 2 分钟；训练 1–2 小时，选
`rubric-score@v2` 时序列约长一倍，会更慢；每次 val 选模解码约几分钟，
共 6 次；最后两次评测共约 10–20 分钟。

选模型按 val 上生成打分的平均 SRCC（`selection.metric = "srcc_mean"`）：
美学和技术两项 SRCC 取平均，最高的检查点入选。解析率低于
`selection.min_parse_rate` 或 SRCC 无定义（如预测全相同）的检查点
不能入选；全部不合格时不写 adapter，训练以报错结束。

损失只算回复 token：批次带 `logits_to_keep`，模型只在回复位置
算词表 logits，数值与全量 logits 的损失相同，省下数 GB 显存。

## 4. 中断与续训

不管在哪一步中断，处理方法都是在 tmux 里重跑同一条命令：

```bash
RESUME=1 bash scripts/cloud/train.sh configs/eva-qwen35-4b-lora-r16.toml
```

| 中断时在 | 重跑时发生什么 |
| --- | --- |
| 预检 | 重新预检；训练从头开始 |
| 训练中，已有检查点 | 从最新的**完整**检查点继续，最多重训约 100 步 |
| 训练中，第 102 步之前 | 报错“没有完整的检查点”，见表下说明 |
| 写检查点写到一半 | 跳过这个不完整的，从上一个完整的继续 |
| 训练已完成（有 `train_result.json`） | 跳过预检和训练，直接评测 |
| LoRA 评测 | 同上，已打分的图片跳过 |

第 102 步之前中断没有可恢复的进度：先看一眼 `outputs/<run_name>` 里
确实没有 `checkpoint-*`，删掉这个目录，再不带 `RESUME=1` 重跑。
跳过不完整检查点的原因记在 `run_info.json`。

续训会检查：

- 配置与 `run_info.json` 里的完全相同，否则拒绝。
- torch、transformers、peft、accelerate 版本与首次训练相同，否则拒绝；
  换了机器就按 `requirements-train.txt` 重装。
- “完整检查点”指适配器、优化器、调度器和 `trainer_state.json` 都在，
  且 `trainer_state.json` 能解析、记录的步数与目录名一致。Trainer 自己的
  续训只认编号最大的目录，遇到写了一半的检查点会带着新权重从第 0 步重训，
  所以这里改为显式指定检查点。
- 输出目录用相对路径，仓库换了位置也能续训，最佳检查点记录不会丢。

每次续训在 `run_info.json` 的 `resumes` 里追加一条（时间、从哪个检查点、
代码版本、环境），并在 `metrics.jsonl` 写一行 `{"event": "resume"}`。
检查点之后、中断之前的那几步会重训，`metrics.jsonl` 里同一步可能出现两次，
以最后一个 resume 标记之后的为准。

## 5. 本机：下载结果、算提升

```bash
scp -r -i <ssh-key> <user>@<a10-host>:~/Lora-SFT/outputs ./
scp -r -i <ssh-key> \
    "<user>@<a10-host>:~/Lora-SFT/data/runs/eva-qwen35-4b-lora-r16-*" \
    data/runs/
```

`outputs/` 里的 `checkpoint-*` 不带回来也行。然后在本机：

```bash
python scripts/report_lora_gain.py \
    --selection data/manifests/eva/prompt-tuning/v1-selection.json \
    --base-run data/runs/<选定提示词的调优运行> \
    --lora-run data/runs/eva-qwen35-4b-lora-r16-eval
```

`--base-run` 是选择文件里 `lora_prompt_id` 对应候选的 `run_id`，
例如 `data/runs/b1-qwen35-4b-tune`。它会核对两边是同一提示词、同一模型、
同一像素预算、同一 `max_new_tokens`、同一份 eval 集，结果写到
`<lora-run>/gain.json`。可达上限见 `eva-scoring-standard.md` 第 4 节。

| 路径 | 内容 |
| --- | --- |
| `outputs/<run_name>/run_info.json` | 配置、代码版本、包版本、GPU、计划步数、续训记录 |
| `outputs/<run_name>/metrics.jsonl` | 训练损失；每次评估的 val 损失、SRCC、解析率 |
| `outputs/<run_name>/checkpoint-*/` | Trainer 检查点（可续训） |
| `outputs/<run_name>/adapter/` | val 平均 SRCC 最高的 LoRA 适配器 |
| `outputs/<run_name>/train_result.json` | 总步数、选模指标、最佳值与检查点 |
| `outputs/<run_name>.logs/` | 各步骤日志 |
| `data/runs/<run_name>-val`、`-eval` | 评测配置、逐张预测、`report.json` |

## 6. 改配置

复制 `configs/eva-qwen35-4b-lora-r16.toml` 并改 `run_name` 和
`runtime.output_dir`，不要改已经跑过的配置。

- 显存不够：减小 `per_device_batch_size`，加大 `grad_accum_steps`。
- 多卡：`runtime.strategy = "ddp"`，`num_gpus` 填卡数；全局 batch
  会随卡数变大，必要时减小 `grad_accum_steps`。
- `save_steps` 必须等于 `eval_steps`：Trainer 只认在获胜那一步
  保存过的检查点，只评估不保存的步数即使最好也会被静默跳过。
- `fsdp` 暂未实现，配置里写了会直接报错。
- 像素预算（`preprocess`）改了，导出到 NAS 推理时必须同步。
