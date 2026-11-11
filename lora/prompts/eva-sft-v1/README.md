# EVA SFT 提示词 v1

这是一阶段弱监督提示词，只用于 EVA 数据。

- 输出 `aesthetic` 和 `technical` 两项 1–5 分。
- 不补造 EVA 没有提供的 `overall` 和 `comment`。
- user 消息的第一部分由数据准备器插入本地图片，第二部分是提示文本。
- Qwen3.5 应用 chat template 时使用 `enable_thinking=false`。
- 训练损失只计算 assistant token。

以后加入带总体分和人工评语的数据时，使用新的提示词版本和标准
`AestheticOutput`，不能直接修改本版本。
