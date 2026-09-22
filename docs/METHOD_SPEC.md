# SCOPE：基于图像状态—残差条件统计建模的 Real-Only AI 生成图像检测方法

版本：2.0（方法整理版）  
日期：2026-09-21  
协议基础：SCOPE_CR68_MDN3_v1.0  
状态：主方法定义已固定；本文档用于统一说明方法思想、实现流程、训练与评估协议。

---

## 0. 方法定位

SCOPE 的目标是进行 **AI 生成图像检测（AI-Generated Image Detection）**，但与常见的“真实图 + AI 图”二分类训练不同，SCOPE 的核心约束是：

> **训练阶段只使用真实照片，不使用任何 AI 生成图像。**

SCOPE 不直接让模型学习“AI 图长什么样”，而是反过来学习：

> **真实照片在某种图像状态下，应该具有怎样的残差统计规律。**

如果测试图像的残差统计明显偏离真实照片中学到的正常规律，则把它视为异常，更可能属于 AI 生成图像。

本文档采用的核心表示为：

- **C（Condition / image state）**：描述当前图块处于什么图像状态；
- **R（Residual statistics）**：描述当前图块的高频残差统计表现；
- **条件模型**：学习真实照片中的

\[
P(R\mid C)
\]

即：

> 在给定图像状态 C 的条件下，真实照片通常会出现怎样的 R。

因此，SCOPE 本质上不是一个传统真假分类器，而是一个 **真实照片条件分布建模 + 异常检测方法**。

---

## 1. 核心思想：SCOPE 到底在学什么

### 1.1 为什么不能直接只看 R

图像残差并不是纯粹的“相机指纹”。

例如，一个图块的残差强弱会受到很多因素影响：

- 图像本身是平滑天空还是复杂草地；
- 局部亮度高还是低；
- 边缘和纹理多还是少；
- 是否接近黑端或白端截断；
- JPEG、缩放、锐化等图像处理。

因此，即使两张图都是真实照片，它们的残差 R 也可能差别很大。

如果只学习无条件分布：

\[
P(R)
\]

模型会把所有真实图块的残差混在一起，容易把“正常的内容差异”误认为异常。

SCOPE 因此引入 C，并学习：

\[
P(R\mid C)
\]

也就是先告诉模型：

> “这个图块大概是什么亮度、颜色、对比度和纹理状态？”

再判断：

> “在这种状态下，它现在出现的残差统计是不是像真实照片？”

---

### 1.2 C 和 R 的直观含义

可以把一张图想象成摄影过程最后得到的结果。

**C 更接近‘成像后的可见状态’**，例如：

- RGB 大致有多亮；
- 局部亮度变化有多大；
- 边缘/纹理有多强；
- 是否存在大量接近纯黑或纯白的像素。

**R 更接近‘从图像中剥离低频后剩下的细小变化统计’**，例如：

- RGB 三个通道的残差强度；
- 残差在左右、上下邻域之间是否相关；
- RGB 通道之间的残差是否同步变化。

需要特别强调：

> R 只是“残差统计量”，不能严格等同于纯摄影痕迹、传感器噪声或相机指纹。

SCOPE 学习的是 **真实照片中 C 与 R 之间稳定的统计对应关系**，而不是宣称 C 与 R 存在严格因果关系。

---

## 2. 总体框架

对一张输入图像 \(x\)，SCOPE 的完整流程如下：

```text
输入图像 x
    ↓
统一解码 / EXIF方向校正 / 小图处理
    ↓
确定性截取 4 个 64×64 图块
    ↓
每个图块执行固定 5×5 低通
    ↓
得到低频部分 L 与残差 E = P - L
    ↓
提取 6维图像状态 C
提取 8维残差统计 R
    ↓
使用训练集统计量标准化 C、R
    ↓
条件混合密度网络 MDN
输入 C → 输出 P(R | C)
    ↓
计算真实 R 的负对数似然 NLL
    ↓
4 个图块 NLL 取平均
    ↓
得到图级异常分数 S(x)
    ↓
与仅由真实校准集确定的阈值 τ 比较
    ↓
输出 Real / AI
```

其中网络只负责一件事情：

\[
\tilde C \longrightarrow P(\tilde R\mid\tilde C)
\]

它并不直接读取原图，也不存在 ResNet、ViT 或可训练特征提取器。

---

## 3. 方法的三个阶段

SCOPE 可以清楚地分成三个阶段。

### 3.1 Stage A：固定特征提取

对每张图确定性提取：

- 4 个 64×64 图块；
- 每块 6 维 C；
- 每块 8 维 R。

这一阶段没有训练，所有操作都是固定公式。

---

### 3.2 Stage B：Real-Only 条件分布学习

只使用真实训练图：

\[
(C,R)_{real}
\]

训练一个很小的条件混合密度网络，使其学会：

\[
P(R\mid C,\text{real})
\]

训练目标是让真实图块在模型下具有尽可能高的似然，即最小化负对数似然 NLL。

---

### 3.3 Stage C：异常检测

测试时无需重新训练。

对测试图像提取 C、R 后，检查：

\[
R_{test}
\]

在模型根据 \(C_{test}\) 给出的真实分布中是否常见。

- 常见 → 分数低 → 更像真实图；
- 少见 → 分数高 → 更像异常 / AI 图。

---

## 4. 输入图像预处理

### 4.1 支持的图像类型

当前协议面向可正常解码为 **8 位 RGB** 的静态 JPEG / PNG 图像。

固定流程：

```text
读取完整图像
→ 应用 EXIF orientation
→ 转为 RGB
→ 灰度图复制为 RGB
→ 透明图在白色背景上合成
→ 不做 ICC 色彩管理
→ 不做 gamma 线性化
→ 不做自动曝光/自动对比度
```

RAW、HDR、16 位图、浮点图、动画或多页图像不在当前正式协议内。

---

### 4.2 小于 64 像素的图像

若图像短边已经满足：

\[
\min(H,W)\ge 64
\]

则保持原尺寸。

若短边小于 64，则使用双线性插值等比例放大，使短边恰好达到 64。

例如：

```text
40×80 → 64×128
100×50 → 128×64
```

这样做的目的是让所有图像都能够进入统一的 64×64 图块处理流程。

需要注意：

> 放大只用于统一输入协议，并不能恢复已经丢失的真实高频信息。

因此后续实验需要单独记录哪些图像发生过放大。

---

## 5. 每张图如何取 4 个图块

SCOPE 每张图固定取 4 块，而不是随机裁剪。

对于尺寸为 \(H\times W\) 的图像，分别在纵向和横向选择靠近 1/4 与 3/4 位置的两个 64 像素窗口。

定义：

\[
a(L)=\operatorname{clip}(\lfloor L/4\rfloor-32,0,L-64)
\]

\[
b(L)=\operatorname{clip}(\lfloor3L/4\rfloor-32,0,L-64)
\]

四个图块左上角依次为：

\[
(a(H),a(W)),
(a(H),b(W)),
(b(H),a(W)),
(b(H),b(W))
\]

例如：

### 128×128 图像

```text
(0, 0)      (0, 64)
(64, 0)     (64, 64)
```

### 256×256 图像

```text
(32, 32)      (32, 160)
(160, 32)     (160, 160)
```

允许图块重叠。

如果整张图刚好只有 64×64，那么四个坐标都相同；协议仍保留四个图块，不临时改变规则。

---

## 6. 低通与残差构造

设一个 RGB 图块为：

\[
P\in\mathbb R^{64\times64\times3}
\]

像素值首先转换到：

\[
[0,1]
\]

### 6.1 固定低通核

采用一维核：

\[
b=\frac1{16}[1,4,6,4,1]
\]

构成二维可分离核：

\[
B=b^Tb
\]

对每个 RGB 通道执行 5×5 valid 卷积，不做 padding。

得到：

\[
L^{60}\in\mathbb R^{60\times60\times3}
\]

---

### 6.2 为什么不做 padding

如果在图块边缘进行 zero padding、reflect padding 等处理，会人为制造边缘统计。

SCOPE 想让 C 和 R 尽量来自真实存在的像素，因此统一使用 valid 卷积，并进一步丢掉外圈区域。

最终使用：

\[
P^\circ=P[3:61,3:61,:]
\]

\[
L^\circ=L^{60}[1:59,1:59,:]
\]

两者都是：

\[
58\times58\times3
\]

残差定义为：

\[
E=P^\circ-L^\circ
\]

因此每个图块最终得到一个 58×58×3 的残差图。

---

## 7. 六维图像状态 C

每个图块提取：

\[
C\in\mathbb R^6
\]

六个维度依次为：

| 维度 | 名称 | 直观含义 |
|---|---|---|
| C1 | lp_mean_r | 低频 R 通道平均亮度 |
| C2 | lp_mean_g | 低频 G 通道平均亮度 |
| C3 | lp_mean_b | 低频 B 通道平均亮度 |
| C4 | lp_luma_std | 低频亮度的局部变化强度 |
| C5 | lp_grad_rms | 低频图中的边缘/纹理强度 |
| C6 | clipped_pixel_fraction | 接近纯黑或纯白的像素比例 |

---

### 7.1 C1–C3：RGB 低频均值

直接计算低通图内部区域三个通道的平均值：

\[
C_1=\operatorname{mean}(L_R^\circ)
\]

\[
C_2=\operatorname{mean}(L_G^\circ)
\]

\[
C_3=\operatorname{mean}(L_B^\circ)
\]

它们主要告诉模型当前图块的整体颜色与亮度状态。

---

### 7.2 C4：低频亮度标准差

先定义亮度代理：

\[
Y=0.299L_R+0.587L_G+0.114L_B
\]

然后计算：

\[
C_4=\sqrt{\operatorname{mean}[(Y-\bar Y)^2]}
\]

它表示图块的低频亮度变化程度。

- 平滑天空：通常较低；
- 建筑、树林、人物轮廓：通常更高。

---

### 7.3 C5：低频梯度 RMS

对低频亮度图计算中心差分：

\[
G_x=\frac{Y(x+1)-Y(x-1)}2
\]

\[
G_y=\frac{Y(y+1)-Y(y-1)}2
\]

然后：

\[
C_5=\sqrt{\operatorname{mean}(G_x^2+G_y^2)}
\]

它比单纯亮度标准差更直接反映边缘和纹理强度。

---

### 7.4 C6：黑端 / 白端截断比例

对 58×58 的原始内部区域 \(P^\circ\)，如果一个像素的任一 RGB 通道满足：

\[
\min_c P_c\le \frac1{255}
\]

或：

\[
\max_c P_c\ge \frac{254}{255}
\]

则认为该像素接近黑端或白端。

最后计算这些像素占总像素的比例：

\[
C_6=q
\]

这里统计的是“像素比例”，不是 RGB 通道元素比例。

---

## 8. 八维残差统计 R

每个图块提取：

\[
R\in\mathbb R^8
\]

组成如下：

| 维度 | 名称 | 统计内容 |
|---|---|---|
| R1 | log_res_std_r | R 通道残差强度 |
| R2 | log_res_std_g | G 通道残差强度 |
| R3 | log_res_std_b | B 通道残差强度 |
| R4 | fisher_spatial_h | 水平相邻残差相关性 |
| R5 | fisher_spatial_v | 垂直相邻残差相关性 |
| R6 | fisher_channel_rg | R/G 残差通道相关性 |
| R7 | fisher_channel_rb | R/B 残差通道相关性 |
| R8 | fisher_channel_gb | G/B 残差通道相关性 |

可以把它理解为三类信息：

```text
残差有多强？          → R1~R3
空间上怎么关联？      → R4~R5
颜色通道之间怎么关联？→ R6~R8
```

---

## 9. R1–R3：RGB 残差强度

对每个残差通道：

\[
v_c=\operatorname{mean}[(E_c-\bar E_c)^2]
\]

然后定义：

\[
R_c=\log\sqrt{v_c+\delta^2}
\]

其中：

\[
\delta=10^{-4}
\]

因此：

\[
R_1,R_2,R_3
\]

分别描述 RGB 三个通道的残差标准差大小，只是额外进行了自然对数变换。

使用 log 的目的，是把跨度较大的残差强度压缩到更容易建模的数值范围。

---

## 10. R4–R8：残差相关统计

### 10.1 为什么不直接使用普通 Pearson

如果一个区域非常平滑，残差可能接近常数，此时普通 Pearson 相关系数的分母可能非常小甚至为 0。

因此 SCOPE 定义了一个带稳定项的 Pearson 型统计量。

对向量 \(u,v\)：

\[
\rho_\delta(u,v)=
\frac{C_{uv}}
{\sqrt{(V_u+\delta^2)(V_v+\delta^2)}}
\]

其中：

\[
\delta=10^{-4}
\]

随后把相关值限制在：

\[
[-0.999,0.999]
\]

再做 Fisher 变换：

\[
\phi(u,v)=\operatorname{atanh}(\rho)
\]

最终 R4–R8 保存的是 Fisher 变换后的值。

---

### 10.2 R4–R5：空间相关

先将 RGB 残差转为亮度残差：

\[
E_Y=0.299E_R+0.587E_G+0.114E_B
\]

#### 水平相关 R4

比较：

```text
每个像素 ↔ 它右边的像素
```

即：

\[
R_4=\phi(E_Y[:,0:57],E_Y[:,1:58])
\]

#### 垂直相关 R5

比较：

```text
每个像素 ↔ 它下面的像素
```

即：

\[
R_5=\phi(E_Y[0:57,:],E_Y[1:58,:])
\]

---

### 10.3 R6–R8：RGB 通道相关

在完全相同的像素位置上比较不同颜色通道残差：

\[
R_6=\phi(E_R,E_G)
\]

\[
R_7=\phi(E_R,E_B)
\]

\[
R_8=\phi(E_G,E_B)
\]

这些统计用于描述真实照片中不同颜色通道高频变化之间的耦合关系。

---

## 11. C 与 R 的标准化

C 和 R 的量纲差别很大。

例如：

- RGB 均值大约在 0–1；
- 残差强度经过 log 后可能是负数；
- Fisher 相关值可能具有完全不同的尺度。

因此训练 MDN 之前，需要对每个维度做训练集标准化。

只使用 **real_train** 中所有图块统计：

\[
m_d=\operatorname{mean}(f_d)
\]

\[
s_d=\operatorname{std}(f_d)
\]

然后：

\[
\tilde f_d=\frac{f_d-m_d}{a_d}
\]

其中：

\[
a_d=
\begin{cases}
s_d,&s_d\ge10^{-6}\\
1,&s_d<10^{-6}
\end{cases}
\]

C 和 R 分别计算自己的均值与标准差。

### 非常重要的限制

以下数据都 **不能** 用来计算标准化参数：

- real_val；
- real_calibration；
- COCO external real；
- GenImage；
- 任何 AI 图像。

训练、验证、校准和测试阶段都必须读取同一份冻结的训练集标准化参数。

---

## 12. 条件混合密度网络（MDN）

### 12.1 为什么用 MDN

即使 C 完全相同，真实照片的 R 也不一定只有一个唯一答案。

例如两块亮度、纹理强度相似的真实区域，仍可能因为：

- 不同相机；
- 不同 JPEG 处理；
- 不同图像来源；
- 不同局部结构；

而出现不同的残差状态。

因此，SCOPE 不让网络输出一个单一预测：

\[
\hat R=f(C)
\]

而是让它输出一个 **概率分布**：

\[
P(R\mid C)
\]

当前使用 3 个高斯成分的混合密度网络。

---

### 12.2 网络结构

网络非常小：

```text
Input: 6维 C
↓
Linear(6 → 64)
ReLU
↓
Linear(64 → 64)
ReLU
↓
Linear(64 → 51)
```

总参数量约：

```text
7,923
```

没有：

- CNN；
- Transformer；
- BatchNorm；
- LayerNorm；
- Dropout；
- 额外分类头。

---

### 12.3 为什么输出 51 维

模型使用：

\[
K=3
\]

个高斯成分，每个成分需要：

- 1 个混合权重；
- 8 个均值；
- 8 个标准差。

因此总输出维数为：

\[
3+3\times8+3\times8=51
\]

输出划分为：

```text
[0:3]   → 3个 mixture logits
[3:27]  → 3×8 个均值 μ
[27:51] → 3×8 个尺度参数 a
```

---

## 13. 三成分条件高斯模型

MDN 实际表示：

\[
p_\theta(\tilde R\mid\tilde C)
=
\sum_{k=1}^{3}
\pi_k(\tilde C)
\mathcal N
\left(
\tilde R;
\mu_k(\tilde C),
\operatorname{diag}(\sigma_k^2(\tilde C))
\right)
\]

含义是：

> 对某个 C，真实 R 可以落在三个不同的典型统计区域之一，而模型会自动决定三个区域各自的概率、中心和波动大小。

三个高斯成分没有预先定义的语义。

不能解释成：

```text
成分1 = 手机照片
成分2 = 单反照片
成分3 = JPEG照片
```

它们只是为了让条件分布具有比单高斯更强的表达能力。

---

## 14. 标准差参数化

网络不能直接输出任意标准差，因为标准差必须：

\[
\sigma>0
\]

因此当前定义：

\[
\sigma_{kd}=0.05+\operatorname{softplus}(a_{kd})
\]

其中 0.05 是 **标准化后的 R 空间** 中的最小标准差。

其目的主要是数值稳定，防止某个高斯把标准差无限缩小从而产生不合理的超高似然。

注意：

> 这里的 0.05 和残差计算中的 \(\delta=10^{-4}\) 是两套完全不同的稳定参数，不能混用。

---

## 15. 训练目标：负对数似然 NLL

对于一个真实图块，给定其：

\[
(C,R)
\]

模型根据 C 输出三个高斯分布。

如果真实 R 位于高概率区域，那么：

\[
p(R\mid C)
\]

较大。

如果真实 R 很不符合模型的条件分布，则概率密度较低。

因此定义图块异常分数：

\[
s(C,R)=-\log p(R\mid C)
\]

也就是 **Negative Log-Likelihood，NLL**。

---

### 15.1 单个高斯成分的 log-density

对于第 \(k\) 个高斯：

\[
g_k=
-\frac12\sum_{d=1}^{8}
\left[
\left(\frac{r_d-\mu_{kd}}{\sigma_{kd}}\right)^2
+2\log\sigma_{kd}
+\log(2\pi)
\right]
\]

随后进行三成分混合：

\[
s(C,R)=
-\operatorname{logsumexp}_k
(\log\pi_k+g_k)
\]

实际实现必须在 log 空间中完成，以避免浮点下溢。

---

## 16. 图级异常分数

一张图固定有四个图块。

第 \(i\) 个图块得到：

\[
s_i=s(C_i,R_i)
\]

最终图级分数：

\[
S(x)=\frac14\sum_{i=1}^{4}s_i
\]

因此：

```text
S 小 → R 在真实条件分布中比较常见 → 更像真实图
S 大 → R 在真实条件分布中比较少见 → 更像异常 / AI 图
```

需要注意：

> S 是异常分数，不是 AI 概率。

它不需要 sigmoid，也不要求落在 0–1。

NLL 甚至可以为负数，这是连续概率密度正常允许的现象。

---

## 17. 训练数据设计

当前首轮协议中，只使用真实图训练。

| 数据集合 | ImageNet | LSUN | COCO | 总数 | 用途 |
|---|---:|---:|---:|---:|---|
| real_train | 5000 | 5000 | 0 | 10000 | 标准化 + 模型训练 |
| real_val | 1000 | 1000 | 0 | 2000 | 选 checkpoint / LR 调度 / Early Stop |
| real_calibration | 1000 | 1000 | 0 | 2000 | 确定真假阈值 |
| real_external_eval | 0 | 0 | 2000 | 2000 | 测试跨真实来源误报率 |

AI 图只在最终冻结模型以后用于检测评估。

---

## 18. 为什么训练、验证、校准要分开

四类真实数据作用不同。

### real_train

用于：

- 计算 C、R 标准化参数；
- 更新 MDN 参数。

### real_val

用于：

- 判断训练是否继续改善；
- 学习率调度；
- 选择 best checkpoint。

### real_calibration

只在模型训练完成后使用，用于确定：

\[
\tau
\]

也就是最终真假判定阈值。

### real_external_eval

完全不参与训练、调参和校准。

用于检查：

> 换一个新的真实图像来源后，会不会大量把真实图片误报成 AI。

---

## 19. 数据隔离与去重

SCOPE 必须以 **图片级别** 划分数据，不能把同一图的不同图块分到不同集合。

正式协议使用规范化后的 RGB 内容计算 SHA256 内容 ID，并根据内容 ID 完成去重和划分。

主要要求：

1. GenImage 测试清单先冻结；
2. COCO external real 清单先冻结；
3. real_train / val / calibration 不得与固定评估清单内容重复；
4. ImageNet、LSUN 与 COCO 原始数据先作为候选池。对稳定排序后的候选 identity 使用固定 `candidate_sampling_seed=20260917` 产生确定性随机顺序，沿该顺序增量解码、计算 canonical content ID、去重并排除与已冻结正式集合重叠的内容，直到各来源达到固定数量；候选不足时沿同一顺序继续补样；
5. 三个随机种子使用完全相同的数据划分。

正式内容隔离适用于最终入选的 ImageNet、LSUN、COCO 与完整 GenImage benchmark。未入选的 ImageNet 候选不需要全部解码，也不用于排除 LSUN。COCO、ImageNet、LSUN 依次冻结；各来源按随机顺序中的入选顺序划分。GenImage benchmark 仍全量解码与冻结。

标准化统计量只使用最终 `real_train` 的 10000 张真实图对应的 40000 个 patch，不使用完整候选数据集。

目的就是避免数据泄漏。

---

## 20. 完整训练流程

### 20.1 特征准备阶段

```text
Step 1
冻结 real_train / real_val / real_calibration / test 清单

Step 2
所有图片统一解码

Step 3
每图固定取 4 个 64×64 patch

Step 4
为每个 patch 计算：
    C_raw ∈ R^6
    R_raw ∈ R^8

Step 5
只使用 real_train 的所有 patch：
    计算 C_mean / C_std
    计算 R_mean / R_std

Step 6
保存标准化参数
```

---

### 20.2 单个 batch 的训练步骤

Batch size 按图像计算：

```text
256 张图 / batch
每张图 4 个 patch
通常共 1024 对 (C,R)
```

训练步骤：

```text
1. 读取 B 张真实训练图对应的 C_raw / R_raw
2. 使用冻结训练统计得到 C_z / R_z
3. C_z 输入 MDN
4. 输出 3 个高斯成分的 π、μ、σ
5. 计算每个 patch 的完整混合 NLL
6. 每张图先平均自己的 4 个 patch NLL
7. 再对 B 张图求平均
8. 反向传播
9. 梯度裁剪
10. AdamW 更新参数
```

训练损失：

\[
\mathcal L_{batch}
=
\frac1B\sum_{j=1}^{B}
\left[
\frac14\sum_{i=1}^{4}s(C_{ji},R_{ji})
\right]
\]

---

## 21. 优化器与训练超参数

默认配置：

| 项目 | 设置 |
|---|---|
| Network | 6→64→64→51 MLP |
| Gaussian components | 3 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Betas | (0.9, 0.999) |
| Adam eps | 1e-8 |
| Weight decay | Linear 权重 1e-4；bias 0 |
| Batch size | 256 images |
| Max epochs | 50 |
| Gradient clipping | global L2 norm ≤ 5 |
| AMP | Off |
| TF32 | Off |
| Network precision | float32 |
| Feature extraction | CPU float64 |

由于模型非常小，当前版本不使用混合精度训练，以优先保证数值可复现性。

---

## 22. 学习率调度

使用 ReduceLROnPlateau：

```yaml
mode: min
factor: 0.5
patience: 2
threshold: 0.0001
threshold_mode: abs
cooldown: 0
min_lr: 0.00001
eps: 0.00000001
```

监控指标是完整 real_val 上的平均图级 NLL：

\[
V_e=\frac1{N_{val}}\sum_x S_e(x)
\]

含义：

> 如果验证 NLL 长时间没有明显下降，就把学习率减半。

---

## 23. Early Stopping 与 best checkpoint

每个 epoch 结束后对 real_val 做一次完整验证。

维护：

- `best_value`：历史最低验证 NLL；
- `stop_reference`：上一次足够明显改善的 NLL；
- `bad_epochs`：连续无有效改善轮数。

有效改善要求：

\[
V_e < stop\_reference-10^{-4}
\]

如果连续 5 个 epoch 没有达到这一改善幅度，则停止训练。

最终：

> **校准和测试只加载验证 NLL 最低的 best checkpoint。**

不是使用最后一个 epoch。

---

## 24. 三随机种子训练

固定训练三次：

```text
seed = 17
seed = 42
seed = 2026
```

三次训练：

- 数据划分相同；
- C/R 特征完全相同；
- 标准化参数相同；
- 模型结构相同；
- 超参数相同；
- 只改变初始化和训练样本顺序。

因此最终得到：

```text
Model_17
Model_42
Model_2026
```

每个模型都有自己的：

- best checkpoint；
- calibration threshold；
- GenImage 测试结果。

不做三模型 ensemble。

如果后续只需要展示一个默认模型，则预先指定：

```text
seed = 42
```

---

## 25. 真实图校准：阈值怎么来

训练完成后，冻结 best checkpoint。

然后对 2000 张 real_calibration 图像计算：

\[
S(x)
\]

得到 2000 个真实图异常分数。

取它们的 95% 分位数：

```python
tau = numpy.quantile(real_scores, 0.95, method="higher")
```

作为最终阈值：

\[
\tau
\]

判定规则：

\[
\hat y(x)=
\begin{cases}
AI,&S(x)>\tau\\
Real,&S(x)\le\tau
\end{cases}
\]

直观上相当于：

> 用一批从未参与训练的真实图，确定“正常真实图异常分数大约能高到哪里”。

95% 分位数只是一条经验阈值，不意味着新数据上一定恰好有 5% 的真实图被误报。

---

## 26. 单张测试图的完整推理流程

```text
Input: image x

1. 解码并修正 EXIF orientation
2. 必要时把短边放大到 64
3. 固定提取 4 个 64×64 patch
4. 每个 patch：
       做 5×5 valid 低通
       构造 58×58×3 残差 E
       提取 6维 C
       提取 8维 R
5. 使用训练集冻结参数标准化 C、R
6. 将 C 输入 MDN
7. 计算实际 R 的 mixture NLL
8. 四个 patch NLL 平均得到 S(x)
9. 与该 seed 自己的 τ 比较
10. 输出 Real / AI
```

整个过程中：

- 不再训练；
- 不重新计算标准化；
- 不重新拟合高斯；
- 不使用测试集确定阈值。

---

## 27. 主测试集与评估方式

当前主检测 benchmark 使用冻结的 **GenImage test** 清单。

每个 seed 分别报告：

- 每个生成器 AUROC；
- 每个生成器 AP；
- AI TPR；
- Real FPR；
- Balanced Accuracy；
- Macro AUROC；
- Overall AUROC；
- COCO external real FPR。

其中：

### Macro AUROC

先分别计算每个生成器 AUROC，再对生成器等权平均。

### Overall AUROC

把全部测试样本放在一起计算，因此不同生成器会按照样本数量产生权重。

二者都需要保留，不能互相替代。

---

## 28. 三次训练结果如何汇总

假设某项指标三次结果为：

\[
m_1,m_2,m_3
\]

平均值：

\[
\bar m=\frac{m_1+m_2+m_3}{3}
\]

样本标准差：

\[
s_m=
\sqrt{
\frac{\sum_{j=1}^{3}(m_j-\bar m)^2}{2}
}
\]

最终报告：

```text
mean ± std
```

例如：

```text
AUROC = 78.4 ± 1.2
```

如果显示为百分数，标准差同样表示百分点。

---

## 29. 完整方法伪代码

```text
Algorithm: SCOPE Training

Input:
    Real training images D_train
    Real validation images D_val
    Number of patches M = 4
    Condition dimension = 6
    Residual dimension = 8
    Gaussian components K = 3

Stage 1: Fixed feature extraction

    For each image x in D_train ∪ D_val:
        Decode image using the fixed protocol.
        Upscale if min(H,W) < 64.
        Extract 4 deterministic 64×64 patches.

        For each patch P:
            Apply fixed 5×5 low-pass filter.
            Construct aligned low-frequency image L.
            Compute residual E = P - L.
            Extract 6-D condition feature C.
            Extract 8-D residual statistic R.

        Cache all C and R.

Stage 2: Training-set normalization

    Use D_train only:
        Estimate C_mean, C_std.
        Estimate R_mean, R_std.
        Freeze normalization statistics.

Stage 3: Conditional density training

    Initialize MDN:
        6 → 64 → 64 → 51

    For each epoch:
        Shuffle training images.

        For each image batch:
            Load four (C,R) pairs per image.
            Standardize C and R.
            Feed C into MDN.
            Obtain mixture weights π,
                   means μ,
                   standard deviations σ.
            Compute full mixture NLL for each patch.
            Average four patch NLLs per image.
            Average image losses over the batch.
            Backpropagate.
            Clip gradients.
            Update parameters with AdamW.

        Evaluate full D_val.
        Save best checkpoint by validation NLL.
        Update ReduceLROnPlateau.
        Apply early stopping.

Output:
    Best MDN checkpoint
    Training-set normalization statistics
```

---

## 30. 校准与检测伪代码

```text
Algorithm: SCOPE Calibration and Detection

Calibration:
    Load best checkpoint.

    For each image x in real_calibration:
        Compute image anomaly score S(x).

    τ = 95th percentile of real calibration scores
        using method="higher".

Detection:
    For a test image x:
        Extract four fixed patches.
        Extract C and R from each patch.
        Standardize using training statistics.
        Compute four patch NLLs.
        S(x) = mean(patch NLLs).

        If S(x) > τ:
            predict AI
        Else:
            predict Real
```

---

## 31. 推荐默认配置汇总

| 模块 | 默认设置 |
|---|---|
| Training paradigm | Real-only anomaly detection |
| Patch number | 4 |
| Patch size | 64×64 |
| Valid internal region | 58×58 |
| Low-pass kernel | [1,4,6,4,1]/16 outer product |
| C dimension | 6 |
| R dimension | 8 |
| Correlation regularizer δ | 1e-4 |
| Fisher correlation clip | [-0.999, 0.999] |
| Feature precision | CPU float64 |
| MDN input | standardized C |
| MDN output | P(R | C) |
| MLP | 6→64→64→51 |
| Gaussian components | 3 |
| Covariance | diagonal |
| Minimum σ | 0.05 in standardized R space |
| Training objective | NLL |
| Image score | mean of 4 patch NLLs |
| Optimizer | AdamW |
| LR | 1e-3 |
| Batch size | 256 images |
| Max epochs | 50 |
| LR scheduler | ReduceLROnPlateau |
| Early stopping | 5 epochs |
| Model seeds | 17, 42, 2026 |
| Calibration | real-only 95th percentile |
| Main AI benchmark | GenImage test |
| External real test | COCO |

---

## 32. 实现时最重要的注意事项

### 32.1 C、R 必须来自同一个固定实现

训练、验证、校准、测试不能使用不同版本的特征代码。

一旦修改以下任一内容：

- low-pass kernel；
- crop boundary；
- C 的公式；
- R 的公式；
- Fisher clip；
- δ；

就应视为新的协议版本，旧缓存不得继续复用。

---

### 32.2 不能泄露测试集统计量

以下所有量只能由 real_train 得到：

- C mean/std；
- R mean/std。

以下量只能由 real_calibration 得到：

- threshold τ。

GenImage 不能参与：

- 特征标准化；
- 训练；
- checkpoint 选择；
- 阈值选择。

---

### 32.3 不要把 NLL 当成概率

SCOPE 最终输出的是异常分数。

不能写成：

```text
S = 0.83 → 83%概率是AI
```

正确解释是：

```text
S 越大，表示该图像的 R 在真实条件分布下越不常见。
```

---

### 32.4 不要把 R 描述成纯相机噪声

当前 R 同时可能受到：

- 图像边缘；
- 纹理；
- 压缩；
- resize；
- 后处理；
- 传感器与 ISP；

等多种因素影响。

论文表述应使用：

> residual statistics / residual-domain statistics

而不是直接宣称：

> camera fingerprint / pure sensor noise

除非后续实验能够充分证明。

---

## 33. 当前方法的主要研究假设

SCOPE 是否有效，核心取决于下面这个假设：

> **真实摄影图像中的图像状态 C 与残差统计 R 之间存在具有一定稳定性、可学习且能跨真实数据来源泛化的条件统计规律，而 AI 生成图像会更频繁地违反这种规律。**

如果这个假设成立，那么：

\[
P_{real}(R\mid C)
\]

能够作为“真实成像统计规律”的参考分布。

AI 图像即使在视觉上十分逼真，其残差结构如果与真实成像过程不一致，就可能获得更高的 NLL。

反过来，如果 AI 图像与真实图在当前 8 维 R 上已经高度重叠，或者 C 无法充分解释内容因素，那么方法性能就会受到限制。

因此，本方法真正需要实验验证的并不是“MLP 能否训练成功”，而是：

> **当前 6维 C + 8维 R 是否包含足够的真实摄影统计信息。**

---

## 34. 与传统二分类检测器的区别

传统监督检测器通常学习：

\[
p(y\mid x),\qquad y\in\{Real,Fake\}
\]

训练需要：

```text
真实图 + AI图
```

SCOPE 学习的是：

\[
p(R\mid C, Real)
\]

训练只需要：

```text
真实图
```

因此它的重点不是记住已有生成器，而是建立一个真实分布参考。

理论上的主要优势是：

> 不依赖训练阶段见过哪些 AI 生成器。

但是否能够真正获得更强的 unseen-generator 泛化能力，需要由实验结果验证，不能仅由方法设计直接推断。

---

## 35. 当前版本不包含的内容

为了保持主方法清晰，当前版本明确 **不加入**：

- AI 图训练；
- GAN / Diffusion 生成反事实样本；
- CNN / ViT 特征提取器；
- 无条件 \(P(R)\) 分支；
- C-R 打乱辅助任务；
- 对比学习损失；
- 分类损失；
- 图级共享 latent component；
- test-time adaptation；
- 多模型 ensemble；
- 测试集阈值搜索。

这些内容如果未来需要研究，应作为单独扩展或消融，不混入当前主方法定义。

---

## 36. 可用于论文的方法贡献表述

如果当前方案后续实验验证有效，可以将方法贡献概括为以下三个方向。

### 1. Real-only conditional residual modeling

提出一种只依赖真实照片训练的 AI 生成图像检测框架，不直接学习特定生成器伪迹，而是建模真实照片中图像状态与残差统计之间的条件分布。

### 2. Compact state–residual statistical representation

使用固定低通和确定性局部区域，从每个图块构造紧凑的 6 维图像状态 C 与 8 维残差统计 R，在避免大型可训练视觉 backbone 的同时显式控制内容状态对残差统计的影响。

### 3. Conditional likelihood based anomaly detection

通过三成分条件混合密度网络学习 \(P(R\mid C)\)，使用真实残差的条件负对数似然作为图块异常度，并通过真实校准集建立图级检测阈值，从而实现不使用 AI 训练样本的异常检测。

这些表述最终应根据实际实验结果和论文创新性检索结果进一步调整。

---

## 37. 一句话总结 SCOPE

SCOPE 可以用一句最直观的话概括：

> **先用真实照片学会“在这种图像状态下，真实照片通常应该出现什么样的残差统计”，测试时再看一张图是否违反这套真实规律。**

其完整固定主方法为：

```text
真实图训练
→ 每图4个固定64×64图块
→ 6维状态 C + 8维残差 R
→ 训练集标准化
→ 3成分条件MDN学习 P(R|C)
→ NLL作为图块异常分数
→ 四块平均形成图级分数
→ 独立真实校准集95%分位数确定阈值
→ GenImage检测 + COCO真实误报评估
→ 3个随机种子报告 mean ± std
```

---

## 38. 协议级固定参数附录

为了保证实现与现有 SCOPE_CR68_MDN3_v1.0 一致，下面列出不能在实现时随意替换的关键细节。

### 38.1 图像与特征

```text
patch size                  = 64
patches per image           = 4
low-pass kernel             = [1,4,6,4,1]/16 outer product
convolution                 = valid, no padding
statistics region           = 58×58
C dimension                 = 6
R dimension                 = 8
residual stability δ        = 1e-4
correlation clip            = [-0.999, 0.999]
C/R extraction dtype        = float64
```

### 38.2 MDN

```text
architecture                = Linear(6,64)-ReLU-Linear(64,64)-ReLU-Linear(64,51)
number of mixtures          = 3
R dimensions per mixture    = 8
covariance                  = diagonal
sigma                       = 0.05 + softplus(a)
network dtype               = float32
AMP                          = off
TF32                         = off
```

### 38.3 Optimization

```text
optimizer                    = AdamW
learning rate                = 1e-3
betas                        = (0.9, 0.999)
eps                          = 1e-8
weight decay (weights)       = 1e-4
weight decay (biases)        = 0
batch size                   = 256 images
max epochs                   = 50
gradient clipping            = 5.0
```

### 38.4 Model selection

```text
validation metric            = mean image NLL
scheduler                    = ReduceLROnPlateau
scheduler factor             = 0.5
scheduler patience           = 2
scheduler threshold          = 1e-4 absolute
early-stop patience          = 5 epochs
best checkpoint criterion    = minimum validation NLL
```

### 38.5 Seeds and calibration

```text
model seeds                  = 17, 42, 2026
calibration set              = 2000 real images
threshold                    = 95th percentile
quantile method              = higher
prediction                   = AI if S > τ, else Real
```

---

**最终方法定义：SCOPE 使用固定图像统计特征构造 6维 C 和 8维 R，仅以真实照片训练三成分条件混合密度网络学习 \(P(R\mid C)\)，以四图块平均负对数似然作为图级异常分数，并使用独立真实校准集确定判别阈值。**
