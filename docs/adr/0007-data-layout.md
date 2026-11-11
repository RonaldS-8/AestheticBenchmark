# ADR-0007：数据目录与文件管理

- 状态：已采纳
- 日期：2026-11-02

## 背景

数据来源越来越多（EVA 图片、DIVIDE 视频，以后还有自爬照片），
评测运行和客户评审包也要放在固定的位置。没有约定的话，
文件会被覆盖，结果也找不到对应的输入。

## 决定

`data/` 不进 git，只保留目录结构。约定如下：

```text
data/
  raw/<数据集>/            原样下载的文件，永不修改
  images/<数据集>/         照片（爬虫产出）
  videos/                  视频，统一命名 video_NNNNNN.mp4，
                           由导入脚本从 raw/ 硬链接过来
  manifests/<数据集>/      JSONL 清单与划分
    labels.jsonl           标签（视频为 VideoLabelRecord）
    split-v1.json          SplitManifest
    import.json            导入记录：来源文件哈希、数量、种子、
                           按 sha256 去重丢弃的行
  runs/<run_id>/           一次评测运行
    config.json            EvalRunConfig
    predictions.jsonl      Prediction，每行一条
    report.json            EvalReport（有标签时）
  review/<名称>/           发给客户的盲评包
  review/<名称>.key.json   盲评对照表，不外发
```

- **只增不改**：导入脚本和评审包生成器发现目标已存在时直接报错，
  不覆盖；要重做就换一个新名称或版本号。
- **raw 只读**：所有加工结果都写到 raw 以外，raw 里的文件可以随时
  和官方源核对。
- **硬链接**：视频从 raw 硬链接到 `videos/`，不额外占空间；
  `data/` 必须和 raw 在同一个磁盘分区，否则导入直接报错，
  不会自动改成复制。
- **路径**：记录里的路径都相对 `data/`。
- **先演练**：导入脚本要先用 `--dry-run` 跑一遍，
  确认数量和 eval 指纹后再正式写入。
- **测试数据**：测试只用 `tests/fixtures/` 和临时目录，
  不读写正式数据。

## 放弃的方案

- 视频直接引用 raw 里的原始文件名：不同数据集会重名，
  也无法用 `video_id` 统一管理。
- 复制视频：DIVIDE 有几千个视频，复制会让占用的空间翻倍。

## 后果

- 换一批视频或换一种划分，都会产生新的清单和新的 eval 指纹，
  旧结果仍然可以复现。
- 删除数据前要先确认没有运行和评审包还在引用它。
