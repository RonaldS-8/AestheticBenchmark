# ADR-0008：本机小样本试跑（B0 + B1 对比 DOVER）

- 状态：已被 ADR-0011 取代（实现已归档到 `archive/video-dover/`）
- 日期：2026-11-02

## 背景

开发机是 Intel Core Ultra 5 225U 笔记本，16 GB 内存，没有 NVIDIA 显卡。
按 OCR 项目在这台机器上的实测，Qwen3.5-4B q4 处理提示词约
10–15 token/s，一段 16 帧的视频要 6 分钟左右。
全量 909 段 × B0–B3 需要几百小时，本机跑不了。

## 决定

- **评测子集**：从 DIVIDE-MaxWell 冻结的评测集中按总体分分层抽 60 段：
  按分数排序后等分成 60 份，每份用固定种子抽 1 段。
  子集有自己的 `eval_fingerprint`，所有运行都记录这个指纹。
- **VLM**：只跑 B0、B1。
  - 模型：Qwen3.5-4B，复用 OCR 项目里的 Q4_K_M GGUF 和
    mmproj，通过 llama-server 推理，温度 0。
  - 本地 safetensors 与 Hugging Face `Qwen/Qwen3.5-4B`
    commit `851bf6e` 的文件哈希一致，以此锁定 revision；
    GGUF 文件锁定 sha256。
  - 视频按时间均匀抽 8 帧，每帧不超过 147,456 像素，
    边长对齐到 32 像素（与 Qwen 的视觉 token 对齐）。
    llama.cpp 不支持直接输入视频，帧作为多张图片发送。
- **DOVER**：原版 `DOVER.pth`（Hugging Face `teowu/DOVER`
  commit `841d2f9`），代码锁定在 DOVER 仓库 `f1ddc96`，
  在单独的 conda 环境 `dover`（Python 3.11、torch CPU、decord）里
  按官方 `evaluate_one_video.py` 的流程运行。
  - DOVER 推理时也有随机采样，每段视频前固定随机种子。
  - 总体分用 DOVER 自带的分数融合公式。
- **产出**：`data/runs/<run_id>/`（配置与预测）；
  B1 与 DOVER 的盲评包放在 `data/review/`。

## 放弃的方案

- 全量运行：本机算力不够，等有 GPU 时再跑。
- B2 / B3：少样本示例要人工补写评语，而且每个示例视频都会占用
  上下文，本机太慢。

## 后果

- 60 段算出的 SRCC / PLCC 波动大，只能作参考，不能作门禁依据。
- 用的是 Qwen3.5-4B，不是 Qwen3-VL；基座选型仍是待定事项。
- **许可**：DOVER 代码在 `f1ddc96` 改为非商用许可，
  所以只能用于内部评测和对照，不能进入产品。
  DIVIDE-MaxWell 数据集卡片标注 apache-2.0，视频内容本身的
  使用范围仍需确认后才能发给外部客户。
- DOVER 权重的训练数据未能从官方页面核实；按论文，原版在 LSVQ
  上训练、DOVER++ 用了 DIVIDE。报告里要注明这一点。
