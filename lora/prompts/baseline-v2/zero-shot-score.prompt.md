---
id: zero-shot-score@v2
methods: b0_zero_shot
media: image
---

# B0 零样本提示词（图片，双分数）

## user

请为这张照片分别打一个美学分和一个技术分，都是 1–5 分。

只输出一个 JSON 对象，不要代码块，不要其他文字。只能有下面两个键，
键的顺序固定，分数是 1.0–5.0 之间、保留一位小数的数字：

{"aesthetic": 分数, "technical": 分数}
