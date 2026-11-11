# 美学头文献调研：CNN、分辨率与双分数

- 日期：2026-11-09
- 目的：给第一轮实测定方向（骨干、输入分辨率、冻结还是微调、
  两个分数怎么接）。
- 标 ✓ 的数字本次逐字核对过原文；其余是调研时读原文得到的，
  没有二次核对。来源见文末。

## 结论

1. **CNN 够用。** 在 EVA 上，ImageNet 预训练的 CNN 直接微调
   （224 级输入），美学 SRCC 0.72–0.74；Swin 0.76；专门设计的
   方法 0.79–0.80。我们测的标注者一致性上限是 0.82。
2. **分辨率要保住，但上限由标注决定。**
   - 不缩到 224：冻结特征换成原图，AVA 美学 +0.05；
     技术质量的冻结特征提高分辨率，KonIQ +0.12。
   - EVA 图最长边不超过 800，标注者看的就是这个尺度。
     训练和推理尺度不一致会掉很多（KonX：0.927 → 0.790）。
     用 EVA 训练，NAS 上就要先缩到 EVA 尺度再打分。
   - 要看 1200 万像素级的噪点和模糊，得另找按高分辨率标注的
     技术质量数据。
3. **技术分是难点。** EVA 技术分唯一的公开结果是 0.548
   （VGG16），我们测的上限 0.72，标签又很集中。
4. **要微调。** EVA 上 VGG16 冻结卷积层 0.600，放开一部分
   0.695。做法是先训头、再用小学习率放开骨干。
5. **两个分数共用骨干、各接一个头。** 两项不能互相替代，
   联合训练基本中性。
6. **许可和泄漏卡得很严。** DOVER、FAST-VQA、CLIP-IQA、Q-Align、
   pyiqa 都是非商用；在 AVA 上训过或见过 AVA 图片的权重都不能用
   （EVA 是 AVA 的子集）。

## 1. EVA 上的公开结果

论文大多用 3,500 训练 / 570 测试，我们是 3,256 / 407 / 407，
数字只能看量级，不能逐位比较。论文里的 “EVA 总分” 就是我们的
`aesthetic`，“质量属性” 就是我们的 `technical`。

**ImageNet 预训练骨干直接训练**（TAVAR 论文 Table VI）✓

| 骨干 | 美学 SRCC | 美学 PLCC |
| --- | --- | --- |
| EfficientNet-B0 | 0.719 | 0.725 |
| MobileNet-v3 | 0.742 | 0.750 |
| ResNet-50 | 0.743 | 0.756 |
| DenseNet-121 | 0.722 | 0.746 |
| DeiT | 0.739 | 0.751 |
| Swin | 0.763 | 0.772 |

**各方法在 EVA 上重训**（同一论文 Table II）✓：
NIMA（ResNet-50）0.725，MLSP 0.677，PA-IAA 0.742，MUSIQ 0.715，
TANet（MobileNet-v2）0.794，TAVAR 0.799（PLCC 0.810）。
TANet 说明小 CNN 加合适的结构也能到 0.79 左右。

**冻结与微调**（Soydaner 与 Wagemans，VGG16，MSE 损失）✓

| | 冻结卷积层 | 放开部分卷积层 |
| --- | --- | --- |
| 美学总分 SRCC | 0.600 | 0.695 |
| 质量属性 SRCC | 0.463 | 0.548 |

## 2. 分辨率：两项都受益，但训练和推理要同一尺度

**美学**

- MLSP（CVPR 2019）✓：冻结 Inception-v3，头不变。
  - 从 256 缩放图切 224：AVA SRCC 0.676；原分辨率：0.729。
  - 224 上端到端微调最好也只有 0.680，不如原图冻结特征。
  - 冻结时多层特征很关键：只用最后 5 层 0.570，全部 11 层 0.644。
  - 最终（InceptionResNet-v2，多层空间池化）0.756。
- Charm（CVPR 2025）✓：部分区域保留原分辨率，不裁剪、
  不改宽高比，摘要称最多提升 8.1%。
- Mai 等（CVPR 2016）：裁剪对构图伤害最大。分类准确率：
  中心裁剪 71.2%，拉伸 73.8%，补边 72.9%，保尺寸池化 76.0%。
- MUSIQ（ICCV 2021）：AVA 上保宽高比 +0.006，再加原图到 0.726。

**技术质量**

- KonIQ-10k（TIP 2020）✓：
  - 冻结 ImageNet 特征加头：512×384 为 0.668，1024×768 为 0.788。
  - 微调时 224 最差，512×384 最好，
    作者用它做了 KonCept512（0.921）。
- UIQA（AIM 2024 UHD-IQA 冠军）✓：
  - 输入 224 / 384 / 480：0.765 / 0.809 / 0.817。
  - 只用缩小的整图 0.721，只用原分辨率小块拼图 0.791，
    两者合用 0.811（验证集）。
- KonX（2022）✓：同一批图在 512 / 1024 / 2048 下各标一次。
  - Effnet-2C-MLSP 在 512 训练：512 测 0.927，2048 测 0.790；
    在 1024 训练、2048 测 0.911。
  - 原文：多分辨率训练没有改善跨分辨率表现，反而常常变差。
  - 人在不同分辨率下给的分本身也会变。

**对 EVA 的含义**

- EVA 图（本机统计 4,070 张）：最长边中位 640、最大 800，
  中位尺寸 640×480。标注者看到的就是这个尺度。
- 对 EVA 来说，“高分辨率” 就是：按原图尺寸输入，保宽高比，
  不缩到 224，不裁剪，不放大。ConvNeXt-T 跑 640×480，
  本机 CPU 实测每张约 0.45 秒、峰值多占约 0.33 GB。
- EVA 图小，整张原图进 CNN 就同时保住了构图和像素细节，
  不必像 DOVER 那样拆两个视图；拆视图是为远大于网络输入的图
  准备的。
- NAS 上的照片约 1200 万像素。用 EVA 训练的模型打分前，
  要缩到 EVA 尺度（长边约 640–800），否则训练和推理尺度不一致。
- 想要 1200 万像素级的技术分，要另加按高分辨率标注的技术质量
  数据（KonIQ 1024×768、SPAQ、UHD-IQA 3840 宽、KonX），
  这些数据的许可都还没查。

## 3. 冻结还是微调，小数据怎么训

- EVA 上放开部分卷积层：美学 +0.095，质量 +0.085 ✓。
- MLSP：原图冻结多层特征（0.729）好过 224 微调（0.680）✓。
  先做冻结探针，结论就有参考价值。
- 训练集大小（KonIQ，微调，512×384）✓：1k 0.831，2k 0.871，
  3k 0.889，5k 0.908，7k 0.915。3k 以后涨得慢，EVA 的 3,256 张
  够用；但 KonIQ 的分数分布比 EVA 技术分宽得多。
- 微调要分两步：先冻结骨干训头，再用比头小 10–100 倍的学习率
  放开骨干（LP-FT，Kumar 等 ICLR 2022）。NIMA 用的是
  骨干 3e-7、头 3e-6。
- 结果对学习率很敏感：CFA 论文里 CLIP ViT-L 在 PARA 上，
  学习率 5e-6 / 1e-5 / 5e-5 对应 SRCC 0.884 / 0.919 / 0.851。
- eval 集只有 407 张：SRCC 的标准误约 ±0.02（ρ≈0.75）到
  ±0.03（ρ≈0.6）。差距在 0.02 以内分不出高下，要多跑几个种子。

## 4. 两个分数怎么接

- **不能互相替代**：Q-Align 只训质量的模型，在 AVA（美学）上
  SRCC 只有 0.208；只训美学的，在 KonIQ（质量）上 0.574。
- **联合训练基本中性**：Q-Align 质量加美学一起训，
  AVA 0.822 → 0.822，KonIQ 0.938 → 0.940，LIVE-C 0.883 → 0.862。
- **常见结构**：共用骨干，各接自己的头；DOVER++ 每个分支用
  自己的标签监督。我的推测：技术头适合多用浅层（细节）特征，
  美学头多用深层（语义）特征，MLSP 式多层池化两头都能用上。
- **损失**：
  - FAST-VQA / DOVER：PLCC 损失 + 0.3 × 排序损失。
  - KonIQ：MSE 略好于 MAE 和 Huber。
  - NIMA：用 EMD 拟合投票分布。

## 5. 高分辨率下的高效模型

AIM 2024 UHD-IQA 挑战赛：4K 照片的技术质量，算力上限 50 GMACs。

| 方法 | 骨干 | 测试 SRCC | GMACs | 参数 |
| --- | --- | --- | --- | --- |
| SJTU（第一） | 3 个 Swin-T | 0.846 | 43.5 | 83M |
| CIPLAB | MobileNetV3 + ViT | 0.835 | 44 | 113M |
| EQCNet | GhostNetV2 | 0.795 | 13.0 | 30M |
| NF-RegNets | RegNet | 0.772 | 44.5 | 28.5M |
| 官方基线 | MobileNetV3 | 0.746 | 4.2 | 3.2M |

- **对 CNN 的警示** ✓：UIQA 用 32 像素小块拼图时，ImageNet
  预训练的 ResNet-50 只有 0.600、MobileNetV2 0.625，
  Swin-T 0.789。小块拼图对 CNN 不友好；给 CNN 应该喂整图
  或完整的原分辨率大块。
- SJTU 和 UIQA 的 Swin-T 用了 AVA 预训练权重，我们不能用。

## 6. 许可与泄漏

- **非商用，只借思路**：DOVER、FAST-VQA、CLIP-IQA、Q-Align
  （S-Lab License 1.0）；pyiqa 代码是 PolyForm Noncommercial，
  权重是 CC BY-NC-SA 4.0。评测也不依赖 pyiqa，
  SRCC / PLCC 用主仓库自己的实现。
- **可商用的代码**：MLSP、KonIQ（MIT），MUSIQ（Apache-2.0）。
- **骨干权重**：
  - Hugging Face 上 `facebook/convnext-tiny-224`、
    `microsoft/resnet-50`、`google/efficientnet-b0` 标的是
    Apache-2.0。
  - 但 ImageNet 数据本身的条款是否影响商用，timm 建议咨询法务；
    上产品前要法务确认。
  - ConvNeXt-V2 的 ImageNet 权重是 CC BY-NC，不能用。
- **泄漏**：EVA 是 AVA 的子集。在 AVA 上训练过、或预训练时见过
  AVA 图片的权重都不能用，例如 NIMA、MLSP、TANet、
  LAION 美学打分器、DOVER 美学分支、CONTRIQUE，以及
  UHD-IQA 冠军用的 AVA 版 Swin-T。
- **EVA 自身** ✓：GitHub 仓库标的是 CC0-1.0，但图片是 AVA 原图的
  缩小版，图片版权不在其内。主仓库 `docs/source-review-eva.md`
  写的 “仓库没有提供标准开源许可证” 需要复核。

## 7. 对我们的建议

- **骨干**：
  - ConvNeXt-T 为主：现代 CNN，KonIQ 上 0.926，接近 Swin-T 0.932。
  - ResNet-50 作对照：EVA 上有现成的公开数可比。
  - EfficientNet-B0 看最小能做到多少，对 NAS 最友好。
- **输入**：EVA 原图尺寸，保宽高比；按宽高比分组凑批，
  或者批大小取 1。NAS 推理时缩到同一尺度。
- **结构**：一个骨干、一个视图（整张原图），多层特征池化，
  接美学、技术两个头。
- **训练**：先冻结骨干训头（这一步同时就是探针），
  再用小学习率放开骨干。
- **损失**：每个头 MSE + PLCC 损失 + 0.3 × 排序损失。
- **增广**：只用水平翻转和轻微缩放。不裁剪、不旋转；
  不加模糊、噪声、JPEG 压缩、颜色抖动，这些会改变技术分的含义。
- **预期**（推测）：美学 0.74–0.80（上限 0.82），
  技术 0.55–0.65（上限 0.72）。

## 8. 第一轮实测方案

2026-11-09 定：不做 224 对照，直接用原图；只比三个骨干；
许可问题先不管。

**阶段 A：冻结探针，本机 CPU**

- 骨干：ResNet-50、EfficientNet-B0、ConvNeXt-T，ImageNet 权重。
- 输入：原图尺寸，保宽高比。
- 每个骨干抽一次多层池化特征，用岭回归拟合两个分数，
  在 val 上比 SRCC / PLCC，并给 bootstrap 置信区间。
- 回答：冻结时哪个骨干最好；多层特征比只用最后一层好多少。

**阶段 B：微调，AWS A10**

- 阶段 A 最好的 1–2 个骨干，原图输入，先训头再放开骨干。
- 骨干学习率扫 3 档 × 3 个种子，按 val 上美学、技术平均 SRCC 选。
- 本机 CPU 估计一轮要一小时以上，所以上 A10。

**收尾**：最好的模型在 eval 上只跑一次，报告格式与 LoRA 相同，
和 LoRA、最佳提示词放在一张表里。

**需要先做的事**：下载三个骨干的 ImageNet 权重（每个约 20–110 MB）。

## 来源

- TAVAR（Li 等，TCSVT 2023）：
  <https://yipoh.github.io/files/TCSVT2023_TAVAR.pdf>
- Soydaner 与 Wagemans（EVA 多任务）：
  <https://arxiv.org/abs/2305.09373>
- MLSP：<https://arxiv.org/abs/1904.01382>
- Charm：<https://arxiv.org/abs/2504.02522>
- Mai 等：Composition-Preserving Deep Photo Aesthetics Assessment，
  CVPR 2016（CVF Open Access）
- MUSIQ：<https://arxiv.org/abs/2108.05997>
- KonIQ-10k：<https://arxiv.org/abs/1910.06180>
- KonX：<https://arxiv.org/abs/2212.05813>
- UIQA：<https://arxiv.org/abs/2409.00749>
- AIM 2024 UHD-IQA 挑战赛报告：<https://arxiv.org/abs/2409.16271>
- UHD-IQA 数据集：<https://arxiv.org/abs/2406.17472>
- Q-Align：<https://arxiv.org/abs/2312.17090>
- FAST-VQA：<https://arxiv.org/abs/2207.02595>
- DOVER：<https://arxiv.org/abs/2211.04894>
- LP-FT：<https://arxiv.org/abs/2202.10054>
- NIMA：<https://arxiv.org/abs/1709.05424>
- CFA / CSKD：<https://arxiv.org/abs/2307.15640>
- ConvNeXt、Swin-T 在 KonIQ 上的对比：<https://arxiv.org/abs/2309.14868>
- IQA-PyTorch（pyiqa）：<https://github.com/chaofengc/IQA-PyTorch>
- timm（权重许可说明）：<https://github.com/huggingface/pytorch-image-models>
- ConvNeXt-V2：<https://github.com/facebookresearch/ConvNeXt-V2>
- EVA 数据集：<https://github.com/kang-gnak/eva-dataset>
