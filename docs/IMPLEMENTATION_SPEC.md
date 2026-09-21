# SCOPE Codex 实施文档：CR68-MDN3 Real-Only 条件统计检测器

版本：1.0  
日期：2026-09-21  
目标仓库：`https://github.com/zenghao0718/SCOPE`  
方法协议：`SCOPE_CR68_MDN3_v1.0`  
适用对象：Codex 代码实施代理  
状态：可直接据此从空仓库开始实现；本地阶段只完成代码、测试和 smoke，不启动正式数据训练。

---

## 0. 执行目标与最高优先级规则

本任务不是修改旧版 SCOPE，而是在一个全新的空仓库中，从零实现当前 SCOPE 主方法：

> **只使用真实图像训练，以 6 维图像状态 C 为条件，对 8 维残差统计 R 的真实条件分布 `P(R|C)` 进行三成分对角高斯 MDN 建模，并使用 NLL 作为测试时异常分数。**

必须遵守以下优先级：

1. `docs/METHOD_SPEC.md` 是唯一的方法权威；
2. 本文件 `docs/IMPLEMENTATION_SPEC.md` 负责规定工程实现方式；
3. 如果本文件与 `METHOD_SPEC.md` 在公式、数据隔离、超参数、评分或训练逻辑上冲突，必须以 `METHOD_SPEC.md` 为准，并停止相关实现、报告冲突；
4. 不允许为了“代码更方便”“训练更快”“结果更好”而私自改变方法；
5. 不允许加入 METHOD_SPEC 未定义的新训练目标、新特征、新异常样本过滤、新数据增强、新 AI 图训练或测试集调参；
6. 本阶段只要求代码、测试和 synthetic smoke 通过并推送 GitHub；**不得在本机启动正式训练，也不得擅自下载正式数据集。**

本方法与旧版 SCOPE V1 无兼容要求。仓库应按当前方法从零组织，不保留旧版 ResNet、Transformer、VICReg、反事实视图、prototype bank 等设计。

`docs/METHOD_SPEC.md` 必须由用户提供的当前 SCOPE 方法文档原样落盘；Codex 不得自行根据记忆重写、补全或“优化”方法文档。如果执行环境中没有该方法文档，应停止并向用户索取，而不是自行生成一个替代版本。

---

## 1. 开工前 Git 检查

### 1.1 必须使用全新 clone

当前远端仓库是重新创建的空仓库。由于仓库 URL 与已经删除的旧仓库相同，**不得直接复用旧本地 SCOPE 目录及其 `.git` 历史**。

执行前：

```bash
git clone https://github.com/zenghao0718/SCOPE.git
cd SCOPE
```

若 `git clone` 提示空仓库，这是预期行为。

检查：

```bash
git remote -v
git status
git branch --show-current
```

要求：

- `origin` 必须指向 `https://github.com/zenghao0718/SCOPE.git`；
- 工作区必须干净；
- 当前分支应为 `main`；若空仓库尚未建立本地分支，则创建 `main`；
- 如果发现当前目录存在旧仓库提交历史、旧 SCOPE V1 文件或非本任务文件，立即停止，不要覆盖或迁移。

### 1.2 不新建 V2 子目录或 V2 分支

当前仓库本身就是新版 SCOPE。

禁止创建：

```text
v2/
scope_v2/
legacy_v1/
```

正式代码直接放在仓库根目录的标准模块中。

---

## 2. 建议仓库结构

从零建立以下结构：

```text
SCOPE/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── scope_cr68_mdn3.yaml
│   └── local_paths.example.yaml
│
├── docs/
│   ├── METHOD_SPEC.md
│   └── IMPLEMENTATION_SPEC.md
│
├── scope_data/
│   ├── __init__.py
│   ├── decoding.py
│   ├── sources.py
│   ├── lsun_lmdb.py
│   ├── manifests.py
│   ├── patches.py
│   └── feature_cache.py
│
├── features/
│   ├── __init__.py
│   ├── lowpass.py
│   ├── cr_features.py
│   └── standardization.py
│
├── models/
│   ├── __init__.py
│   └── mdn.py
│
├── losses/
│   ├── __init__.py
│   └── mdn_nll.py
│
├── engine/
│   ├── __init__.py
│   ├── seed.py
│   ├── checkpoint.py
│   ├── logging.py
│   └── trainer.py
│
├── evaluation/
│   ├── __init__.py
│   ├── inference.py
│   ├── calibration.py
│   ├── metrics.py
│   └── summarize.py
│
├── scripts/
│   ├── build_manifests.py
│   ├── extract_features.py
│   ├── fit_standardizer.py
│   ├── train.py
│   ├── calibrate.py
│   ├── eval_genimage.py
│   ├── eval_coco.py
│   └── summarize_seeds.py
│
└── tests/
    ├── test_decoding.py
    ├── test_patches.py
    ├── test_cr_features.py
    ├── test_standardization.py
    ├── test_mdn.py
    ├── test_nll.py
    ├── test_manifests.py
    ├── test_calibration.py
    ├── test_metrics.py
    └── test_smoke_pipeline.py
```

说明：

- `scope_data/` 专门负责输入、数据源、manifest、patch 坐标和 feature cache；不要命名为顶层 `data/`，避免与用户正式数据目录混淆；
- `features/` 只包含固定公式特征，不允许可训练参数；
- `models/` 只包含当前 MDN；
- `losses/` 只包含完整混合高斯 NLL；
- `scripts/` 是正式命令行入口；
- `tests/` 必须使用 synthetic data 即可独立运行；
- 正式图片、特征缓存和实验结果均不得提交 Git。

---

## 3. 外部工作区与 Git 忽略规则

工程上固定采用“代码、数据、运行结果分离”的布局：

```text
<workspace>/
├── SCOPE/          # Git 仓库
├── SCOPE_DATA/     # 正式数据、manifest、特征缓存
└── SCOPE_RUNS/     # 训练、校准、评测输出
```

`.gitignore` 至少排除：

```text
__pycache__/
*.pyc
.pytest_cache/
.venv/
venv/
.DS_Store
.idea/
.vscode/

configs/local_paths.yaml

SCOPE_DATA/
SCOPE_RUNS/
runs/
outputs/
cache/
*.pt
*.pth
*.ckpt
*.npy
*.npz
*.csv.gz
```

不要排除 `tests/` 中由代码动态创建的临时数据，因为这些应写入 pytest 的临时目录而不是仓库。

---

## 4. `AGENTS.md` 必须明确的项目规则

创建 `AGENTS.md`，至少写明：

- `docs/METHOD_SPEC.md` 是最高方法权威；
- METHOD_SPEC 与 IMPLEMENTATION_SPEC 冲突时停止并报告；
- 训练、标准化、checkpoint 选择不得使用任何 AI 图；
- `real_calibration` 只用于阈值；
- GenImage 只能用于冻结后的最终评测；
- 不得使用测试结果选择 seed、checkpoint、阈值、K、特征或超参数；
- 正式配置禁止静默改写；
- 正式协议中所有随机 seed、数据清单、特征缓存必须可追溯；
- feature protocol 发生任何变化时旧 cache 必须失效；
- 本地阶段禁止正式训练；
- 所有最终改动必须通过 compile、pytest、synthetic smoke 后再提交。

不要在 AGENTS 中加入旧版 SCOPE V1 的任何规则。

---

## 5. 配置系统

### 5.1 正式配置文件

创建：

```text
configs/scope_cr68_mdn3.yaml
```

正式默认值必须至少包含：

```yaml
protocol:
  id: SCOPE_CR68_MDN3_v1.0

image:
  patch_size: 64
  patches_per_image: 4
  min_short_side: 64
  resize_mode: bilinear
  supported_extensions: [".jpg", ".jpeg", ".png"]
  alpha_background: white

feature:
  lowpass_1d: [1, 4, 6, 4, 1]
  lowpass_divisor: 16.0
  delta: 0.0001
  correlation_clip: 0.999
  near_constant_std: 0.000001
  c_dim: 6
  r_dim: 8
  c_names:
    - lp_mean_r
    - lp_mean_g
    - lp_mean_b
    - lp_luma_std
    - lp_grad_rms
    - clipped_pixel_fraction
  r_names:
    - log_res_std_r
    - log_res_std_g
    - log_res_std_b
    - fisher_spatial_h
    - fisher_spatial_v
    - fisher_channel_rg
    - fisher_channel_rb
    - fisher_channel_gb

model:
  hidden_dims: [64, 64]
  num_components: 3
  sigma_floor: 0.05

train:
  batch_size_images: 256
  max_epochs: 50
  learning_rate: 0.001
  betas: [0.9, 0.999]
  adam_eps: 0.00000001
  weight_decay_weight: 0.0001
  weight_decay_bias: 0.0
  grad_clip_norm: 5.0
  amp: false
  tf32: false

scheduler:
  mode: min
  factor: 0.5
  patience: 2
  threshold: 0.0001
  threshold_mode: abs
  cooldown: 0
  min_lr: 0.00001
  eps: 0.00000001

early_stopping:
  min_delta: 0.0001
  patience: 5

seeds: [17, 42, 2026]

data:
  split_salt: "20260917"
  real_train:
    imagenet: 5000
    lsun: 5000
  real_val:
    imagenet: 1000
    lsun: 1000
  real_calibration:
    imagenet: 1000
    lsun: 1000
  real_external_eval:
    coco: 2000

calibration:
  quantile: 0.95
  method: higher
```


### 5.2 不能把 51 写死

代码必须按：

```text
output_dim = K * (1 + 2 * r_dim)
```

计算输出维度。

正式默认：

```text
K = 3
r_dim = 8
output_dim = 51
```

这样未来做 K 消融时只改配置，不改模型源码。

### 5.3 路径配置

创建：

```text
configs/local_paths.example.yaml
```

示例：

```yaml
workspace:
  data_root: "../SCOPE_DATA"
  runs_root: "../SCOPE_RUNS"

sources:
  imagenet_root: "/path/to/imagenet"
  lsun_roots:
    - "/path/to/lsun/source_or_lmdb"
  coco_root: "/path/to/coco"
  genimage_root: "/path/to/genimage"
```

真正使用的 `configs/local_paths.yaml` 必须被 `.gitignore` 忽略。

### 5.4 配置校验

启动任何正式脚本前，至少验证：

- patch size = 64；
- patches per image = 4；
- C dim = 6；
- R dim = 8；
- num_components >= 1；
- 正式默认 num_components = 3；
- sigma_floor > 0；
- batch size = 256；
- max epochs = 50；
- seeds 精确为 `[17,42,2026]`；
- calibration method = `higher`；
- AMP = false；
- TF32 = false。

正式配置若被修改，不要静默纠正；应报错或在日志中明确显示非正式配置。

---

## 6. 输入图像统一解码

实现 `scope_data/decoding.py`。

### 6.1 支持范围

当前正式协议只支持可正常解码为 8-bit RGB 的静态 JPEG / PNG，包括灰度形式。

支持后缀：

```text
.jpg
.jpeg
.png
```

RAW、HDR、16 位/浮点图、动画、多页图像不进入正式协议。

### 6.2 固定解码顺序

必须严格按以下顺序：

1. Pillow 完整解码，不使用 thumbnail、draft 或解码阶段降采样；
2. 检查多帧/动画，若 `n_frames > 1` 则返回 `unsupported_format`；
3. `ImageOps.exif_transpose` 应用 EXIF orientation，只用于像素摆正；
4. 其他 EXIF 字段完全不进入特征、训练或 manifest 排序；
5. 灰度图复制到 RGB；
6. 调色板/CMYK 等 8-bit 输入使用固定 Pillow RGB 转换；
7. 含透明度的图像先转 RGBA，再在纯白背景 `(255,255,255,255)` 上 alpha composite，最后转 RGB；
8. 不做 ICC 色彩管理；
9. 不做 gamma 线性化；
10. 不做自动曝光、自动对比度、自动锐化；
11. 最终输出连续 C-order 的 `np.uint8 [H,W,3]`。

所有输入来源使用同一个函数，不允许训练/测试走不同解码路径。

### 6.3 16 位和浮点图判定

不要把 16-bit PNG 或浮点图静默 `convert("RGB")` 后当正式输入。

必须在转换前检查 Pillow mode / dtype，并明确返回：

```text
unsupported_format
```

发生损坏、空尺寸或解码失败时返回：

```text
decode_error
```

正式评测时必须记录错误，不得静默删图。

### 6.4 解码元数据

每张图至少记录：

```text
original_width
original_height
oriented_width
oriented_height
input_mode
alpha_composited
exif_transposed
status
```

这些字段只用于审计，不加入 C 或 R。

---

## 7. Canonical content ID 与数据去重

实现 `canonical_content_id(rgb_uint8)`。

### 7.1 哈希输入

对完成 EXIF orientation、颜色模式转换、透明合成后的 RGB uint8 图，**但在任何小图放大之前**，依次写入 SHA256：

1. 高度 H：8 字节 unsigned little-endian；
2. 宽度 W：8 字节 unsigned little-endian；
3. C-order 的 RGB uint8 原始字节。

得到：

```text
content_id = SHA256(H || W || RGB_bytes)
```

不要对原 JPEG/PNG 文件字节直接 hash 作为内容 ID。

可额外保存 `file_sha256` / `record_sha256` 作为审计字段，但数据隔离必须使用 canonical `content_id`。

### 7.2 原因

相同像素内容即使文件 metadata 不同，也应被视为相同内容；内容 ID 必须基于规范化像素而不是容器字节。

---

## 8. 数据源与 manifest

实现 `scope_data/sources.py`、`scope_data/lsun_lmdb.py`、`scope_data/manifests.py` 和 `scripts/build_manifests.py`。

### 8.1 manifest 通用字段

至少包含：

```text
image_id
content_id
source
source_category
split
storage_type
path
container_path
sample_key
label
generator
original_width
original_height
decode_status
group_id
```

说明：

- `storage_type`：`file` 或 `lmdb`；
- real 数据 `label=0`；
- GenImage AI 数据 `label=1`；
- 非 GenImage 数据 `generator` 留空；
- `group_id` 仅在数据源本身能够可靠给出“同一原图不同版本”组信息时填写；不要用感知哈希猜测 group_id。

### 8.2 LSUN

允许两种读取方式：

- 普通图片目录；
- LMDB 原始记录。

LMDB 必须在内存中解码原始图像 bytes，不得为了 manifest 或 feature extraction 先批量重编码成 JPEG/PNG。

### 8.3 GenImage layout

METHOD_SPEC 没有规定用户本地 GenImage 的唯一目录结构，因此工程实现必须：

- 提供可配置的 GenImage discovery；
- 允许指定 evaluation split / generator 路径映射；
- 如果目录结构存在歧义，明确报错，不要靠文件夹名字猜一个“最可能”的结果；
- 一旦成功构建正式 GenImage manifest，之后所有 seed 共用这一份冻结 manifest。

不要在代码中假设某个生成器集合永远固定；manifest 中按实际冻结清单保存 generator 名称。

---

## 9. 数据划分与隔离顺序

这是正式协议的一部分，不得改变。

### 9.1 第一步：冻结 GenImage evaluation manifest

先构建完整 GenImage 评估条目并计算 canonical `content_id`。

评估清单本身保持 benchmark 原有条目；即使评估集中出现重复，也不要为了“清洁”而自动重写 benchmark。

### 9.2 第二步：冻结 COCO external real

COCO 候选先在来源内部按 canonical content ID 去重。

固定 salt：

```text
20260917
```

排序 key：

```text
SHA256("20260917|coco|" + content_id)
```

按十六进制字符串升序；同 key 再按 `content_id`。

取前：

```text
2000
```

作为 `real_external_eval`。

### 9.3 第三步：真实训练候选排除固定评估内容

建立：

```text
eval_content_ids = GenImage content IDs ∪ COCO external content IDs
```

ImageNet / LSUN 候选中凡 `content_id` 落入该集合全部排除。

### 9.4 第四步：ImageNet / LSUN 全局去重

- 同来源重复：按稳定原始记录 ID 字典序保留一个；
- 跨 ImageNet / LSUN 完全相同内容：固定优先 ImageNet；
- 若数据源提供可靠 `group_id`，同组不能跨 train/val/calibration；
- 不做额外 perceptual hash 近重复推断。

### 9.5 第五步：确定性排序

每个来源独立计算：

```text
SHA256("20260917|" + source + "|" + content_id)
```

按十六进制升序；相同 key 再按 `content_id`。

### 9.6 第六步：固定数量切分

ImageNet：

```text
前 5000  → real_train
后 1000  → real_val
再后 1000 → real_calibration
```

LSUN：

```text
前 5000  → real_train
后 1000  → real_val
再后 1000 → real_calibration
```

COCO：

```text
2000 → real_external_eval
```

最终：

```text
real_train       = 10000
real_val         = 2000
real_calibration = 2000
real_external    = 2000
```

任何来源候选不足时必须在正式训练前失败，不允许缩小规模、不允许从其他 split 挪样本。

### 9.7 manifest 输出

建议：

```text
SCOPE_DATA/manifests/SCOPE_CR68_MDN3_v1.0/
├── real_train.csv
├── real_val.csv
├── real_calibration.csv
├── real_external_coco.csv
├── genimage_eval.csv
└── manifest_meta.json
```

`manifest_meta.json` 至少记录：

- protocol id；
- salt；
- 每个 manifest SHA256；
- 每来源候选数、拒绝数、重复数、最终数量；
- Pillow 版本；
- 构建时间；
- 数据源根路径仅作本地记录，不用于内容身份。

---

## 10. 小图处理

实现 `resize_if_needed(rgb_uint8)`。

设方向校正后尺寸为 `H0 × W0`。

- 若 `min(H0,W0) >= 64`：不 resize；
- 若 `H0 <= W0` 且 `H0 < 64`：
  - `H = 64`
  - `W = ceil(64 * W0 / H0)`
- 若 `W0 < H0` 且 `W0 < 64`：
  - `W = 64`
  - `H = ceil(64 * H0 / W0)`

使用：

```text
Pillow Image.Resampling.BILINEAR
```

输入和输出都保持 RGB uint8；不得保存中间 JPEG。

随后转：

```text
np.float64 / 255.0
```

记录：

```text
processed_width
processed_height
upscaled
scale_x
scale_y
```

---

## 11. 固定四图块坐标

实现 `scope_data/patches.py`。

对单轴长度 `L >= 64`：

```text
a(L) = clip(floor(L/4) - 32, 0, L - 64)
b(L) = clip(floor(3L/4) - 32, 0, L - 64)
```

四个左上角坐标按固定顺序：

```text
(a(H), a(W))
(a(H), b(W))
(b(H), a(W))
(b(H), b(W))
```

坐标语义固定为 `(y,x)`。

patch：

```python
x[y:y+64, x:x+64, :]
```

不得去重、随机补块、随机裁剪、翻转、旋转或增强。

即使 64×64 图得到四个完全相同 patch，也保留四条记录。

额外记录：

```text
patch_yx [4,2]
num_unique_patches
```

---

## 12. 固定低通实现

实现 `features/lowpass.py`，全部使用 CPU `np.float64`。

一维核：

```text
b = [1,4,6,4,1] / 16
```

二维核：

```text
B = outer(b,b)
```

即：

```text
1/256 *
[[ 1,  4,  6,  4, 1],
 [ 4, 16, 24, 16, 4],
 [ 6, 24, 36, 24, 6],
 [ 4, 16, 24, 16, 4],
 [ 1,  4,  6,  4, 1]]
```

对 64×64×3 patch 每通道做 5×5、stride=1、valid 卷积，不 padding。

不得依赖 SciPy；可使用 `numpy.lib.stride_tricks.sliding_window_view` + `einsum` 或等价确定性 NumPy 实现。

输出：

```text
L60: [60,60,3] float64
```

统一内部区域：

```text
P_inner = P[3:61, 3:61, :]       # [58,58,3]
L_inner = L60[1:59, 1:59, :]     # [58,58,3]
E       = P_inner - L_inner       # [58,58,3]
```

所有均值、方差、协方差使用总体统计，即 `ddof=0`。

---

## 13. 6 维 C 的精确实现

实现 `features/cr_features.py`。

固定 feature order：

```text
C = [
  lp_mean_r,
  lp_mean_g,
  lp_mean_b,
  lp_luma_std,
  lp_grad_rms,
  clipped_pixel_fraction,
]
```

### C1-C3

```text
mean(L_inner_R)
mean(L_inner_G)
mean(L_inner_B)
```

### C4

先在完整 `L60` 上构造：

```text
Y60 = 0.299*L60_R + 0.587*L60_G + 0.114*L60_B
Y   = Y60[1:59,1:59]
```

然后：

```text
C4 = sqrt(mean((Y - mean(Y))^2))
```

### C5

中心差分精确使用：

```text
Gx = (Y60[1:59, 2:60] - Y60[1:59, 0:58]) / 2
Gy = (Y60[2:60, 1:59] - Y60[0:58, 1:59]) / 2
C5 = sqrt(mean(Gx^2 + Gy^2))
```

### C6

在 `P_inner [58,58,3]` 上，对每个像素：

```text
min(channel) <= 1/255
OR
max(channel) >= 254/255
```

该像素计一次。

```text
C6 = count / (58*58)
```

不是通道元素比例。

---

## 14. 8 维 R 的精确实现

固定 order：

```text
R = [
  log_res_std_r,
  log_res_std_g,
  log_res_std_b,
  fisher_spatial_h,
  fisher_spatial_v,
  fisher_channel_rg,
  fisher_channel_rb,
  fisher_channel_gb,
]
```

### 14.1 R1-R3

对 `E_c [58,58]`：

```text
mean_c = mean(E_c)
v_c    = mean((E_c - mean_c)^2)
R_c    = log(sqrt(v_c + delta^2))
```

固定：

```text
delta = 1e-4
```

log 为自然对数。

### 14.2 正则化 Pearson 型统计

实现单一函数：

```text
regularized_fisher_corr(u, v, delta=1e-4, clip=0.999)
```

必须：

```text
u_mean = mean(u)
v_mean = mean(v)
Vu = mean((u-u_mean)^2)
Vv = mean((v-v_mean)^2)
Cuv = mean((u-u_mean)*(v-v_mean))

rho = Cuv / sqrt((Vu + delta^2)*(Vv + delta^2))
rho = clip(rho, -0.999, 0.999)
phi = atanh(rho)
```

向量必须等长；所有统计 float64。

### 14.3 R4-R5

亮度残差：

```text
EY = 0.299*E_R + 0.587*E_G + 0.114*E_B
```

水平：

```text
u = EY[:,0:57].reshape(-1, order="C")
v = EY[:,1:58].reshape(-1, order="C")
R4 = fisher_corr(u,v)
```

长度均为：

```text
58*57 = 3306
```

垂直：

```text
u = EY[0:57,:].reshape(-1, order="C")
v = EY[1:58,:].reshape(-1, order="C")
R5 = fisher_corr(u,v)
```

### 14.4 R6-R8

同一像素位置：

```text
R6 = fisher_corr(np.ravel(E_R, order="C"), np.ravel(E_G, order="C"))
R7 = fisher_corr(np.ravel(E_R, order="C"), np.ravel(E_B, order="C"))
R8 = fisher_corr(np.ravel(E_G, order="C"), np.ravel(E_B, order="C"))
```

长度均为：

```text
58*58 = 3364
```

### 14.5 输出检查

每个 patch 返回：

```text
C_raw: float64 [6]
R_raw: float64 [8]
```

所有值必须 finite；出现 NaN/Inf 视为实现错误并立即报错，不允许 `nan_to_num`。

---

## 15. 特征缓存

实现 `scope_data/feature_cache.py` 和 `scripts/extract_features.py`。

### 15.1 为什么必须缓存

正式三个 seed 必须使用完全相同的 C/R；训练阶段不要每个 epoch 重复解码图片和提取固定特征。

### 15.2 每个 split 的 cache 目录

建议：

```text
SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0/<split>/
├── c_raw.npy
├── r_raw.npy
├── patch_yx.npy
├── metadata.csv
└── cache_meta.json
```

数组：

```text
c_raw.npy    float64 [N,4,6]
r_raw.npy    float64 [N,4,8]
patch_yx.npy int32   [N,4,2]
```

`metadata.csv` 每行对应一张图片，至少包含：

```text
image_id
content_id
source
label
generator
upscaled
original_width
original_height
processed_width
processed_height
num_unique_patches
status
```

### 15.3 protocol hash

计算 `feature_protocol_hash`，输入至少包括：

- protocol id；
- decode protocol；
- resize rule；
- patch rule；
- low-pass kernel；
- C/R feature names 和顺序；
- delta；
- correlation clip；
- feature precision。

使用 canonical JSON `sort_keys=True` 后 SHA256。

cache meta 至少保存：

```text
protocol_id
feature_protocol_hash
manifest_sha256
sample_count
c_shape
r_shape
numpy_version
pillow_version
python_version
created_at
```

如果已有 cache 的 hash 或 manifest hash 不匹配，默认拒绝复用；只有显式 `--overwrite` 才能重新生成。

---

## 16. 训练集标准化

实现 `features/standardization.py` 和 `scripts/fit_standardizer.py`。

### 16.1 统计范围

只读取 `real_train` cache。

设 N=10000，每图四块，因此每个维度使用：

```text
4N = 40000
```

个观测。

C、R 独立按维度计算总体 mean/std：

```text
mean = sum / (4N)
std  = sqrt(mean((x-mean)^2))
```

使用 float64；`ddof=0`。

### 16.2 near-constant

对每个维度：

```text
if std >= 1e-6:
    scale = std
else:
    scale = 1.0
    near_constant = true
```

标准化：

```text
z = (x - mean) / scale
```

不做 clip。

### 16.3 artifact

保存：

```text
SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0/standardizer.json
```

至少包含：

```text
protocol_id
feature_protocol_hash
train_manifest_sha256
train_cache_sha256
c_mean[6]
c_std[6]
c_scale[6]
c_near_constant[6]
r_mean[8]
r_std[8]
r_scale[8]
r_near_constant[8]
artifact_sha256
```

验证、校准、COCO、GenImage 都只能读取该 artifact，禁止重新拟合。

---

## 17. MDN 模型

实现 `models/mdn.py`。

### 17.1 结构

```text
Linear(6,64)
ReLU
Linear(64,64)
ReLU
Linear(64, output_dim)
```

无 BatchNorm、LayerNorm、Dropout、CNN、Transformer 或额外 head。

### 17.2 输出拆分

若：

```text
K = num_components
D = r_dim = 8
```

则：

```text
logits: [B,K]
mu:     [B,K,D]
raw_a:  [B,K,D]
```

最后层平铺顺序必须是：

```text
K logits
component 0 的 D 个 mu
component 1 的 D 个 mu
...
component K-1 的 D 个 mu
component 0 的 D 个 raw_a
...
```

正式 K=3 时等价：

```text
[0:3]   logits
[3:27]  mu reshape [3,8]
[27:51] raw_a reshape [3,8]
```

### 17.3 概率参数化

```python
log_pi = F.log_softmax(logits, dim=-1)
sigma = 0.05 + F.softplus(raw_a, beta=1, threshold=20)
```

不要先 softmax 再 log。

### 17.4 初始化

必须按当前 model seed 完成模块创建和参数重置。

隐藏两层 Linear：

```text
weight: Kaiming uniform
        a=0
        mode=fan_in
        nonlinearity=relu
bias: 0
```

输出 Linear：

```text
weight ~ Normal(0, 1e-3)
```

输出 bias：

- logits 的 K 个 bias = 0；
- mu 的 K*8 个 bias ~ `Normal(0,0.1)`；
- raw_a 的 K*8 个 bias = `log(expm1(0.95))`。

这样初始 sigma 约为：

```text
0.05 + 0.95 = 1.0
```

### 17.5 参数量测试

正式 K=3 必须精确得到：

```text
7923 trainable parameters
```

写单元测试锁定。

---

## 18. 完整混合 NLL

实现 `losses/mdn_nll.py`。

输入：

```text
r_z:    [N,8]
log_pi: [N,K]
mu:     [N,K,8]
sigma:  [N,K,8]
```

第 k 个成分：

```text
g_k = -0.5 * sum_d(
    ((r_d - mu_kd)/sigma_kd)^2
    + 2*log(sigma_kd)
    + log(2*pi)
)
```

混合：

```text
patch_nll = -logsumexp(log_pi + g, dim=K)
```

必须保留：

- 二次项；
- `2*log(sigma)`；
- `log(2*pi)`；
- mixture logsumexp。

不得替换成最近中心距离。

NLL 在连续密度下允许为负；不得 clamp 到 0，不得 sigmoid。

---

## 19. 训练数据对象

训练阶段直接读取 feature cache，不再读取 JPEG/PNG。

Dataset 每个 index 返回一张图：

```text
C_raw [4,6] float64
R_raw [4,8] float64
image_id
```

在 batch collate 或 Dataset 内使用冻结 standardizer 转成：

```text
C_z [B,4,6] float32
R_z [B,4,8] float32
```

训练网络时 reshape：

```text
C_z -> [4B,6]
R_z -> [4B,8]
```

NLL 后恢复：

```text
patch_nll -> [B,4]
image_nll = mean(patch_nll, dim=1)
loss = mean(image_nll)
```

一张图始终等权，不允许因为最后一个 batch 较小而在 epoch 汇总时把 batch 等权。

---

## 20. 随机性与确定性

实现 `engine/seed.py`。

正式 seeds：

```text
17
42
2026
```

每次运行设置：

```text
random.seed
numpy.random.seed
torch.manual_seed
torch.cuda.manual_seed_all
```

并：

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
cudnn.benchmark = False
cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
TF32 = False
AMP = False
```

`CUBLAS_WORKSPACE_CONFIG` 必须在创建 CUDA context 前设置。

### 20.1 每 epoch 排列

训练图像排列使用独立 CPU RNG：

```text
shuffle_seed = model_seed + 100000 * epoch
```

其中 epoch 从 1 开始。

建议使用：

```text
torch.Generator(device="cpu")
manual_seed(shuffle_seed)
torch.randperm(N, generator=generator)
```

不要依赖 DataLoader 默认 shuffle RNG。

---

## 21. Optimizer 与训练步

### 21.1 AdamW

精确参数：

```text
lr = 1e-3
betas = (0.9,0.999)
eps = 1e-8
amsgrad = False
foreach = False
fused = False
```

parameter groups：

```text
所有 Linear.weight → weight_decay = 1e-4
所有 bias          → weight_decay = 0
```

不要手写额外 L2 loss。

### 21.2 单步顺序

固定：

```text
optimizer.zero_grad(set_to_none=True)
forward
compute loss
assert finite(loss)
backward
assert all gradients finite
clip_grad_norm_(..., max_norm=5)
optimizer.step()
assert all parameters finite
```

任何非有限 loss / gradient / parameter：立即终止运行并写明错误；禁止跳过 batch、删样本、`nan_to_num` 后继续。

---

## 22. Epoch、验证、调度器与 Early Stop

### 22.1 一个 epoch

一个 epoch = 完整遍历 10000 张 `real_train` 一次。

Batch size：

```text
256 images
```

最后不足 256 的 batch 保留。

### 22.2 验证

每 epoch 完成后，对全部 2000 张 `real_val` 固定顺序评估。

每张图先得到：

```text
S(x) = mean(4 patch NLL)
```

再：

```text
V_e = sum(all image S) / N_val
```

跨 batch 累计使用 Python/NumPy float64；模型前向仍 float32。

### 22.3 ReduceLROnPlateau

精确：

```text
mode=min
factor=0.5
patience=2
threshold=1e-4
threshold_mode=abs
cooldown=0
min_lr=1e-5
eps=1e-8
```

每 epoch 完整验证后调用一次：

```text
scheduler.step(V_e)
```

新 LR 从下一 epoch 生效。

### 22.4 best 与 early stopping

维护：

```text
best_value = +inf
stop_reference = +inf
bad_epochs = 0
```

每 epoch：

```text
if V_e < best_value:
    best_value = V_e
    save best checkpoint

if V_e < stop_reference - 1e-4:
    stop_reference = V_e
    bad_epochs = 0
else:
    bad_epochs += 1

scheduler.step(V_e)
save last checkpoint

if bad_epochs >= 5 or epoch == 50:
    stop
```

相同 best 值保留较早 epoch。

LR 下降不得重置 early-stopping 计数。

最终校准和测试只加载 `best.pt`。

---

## 23. Checkpoint 与运行目录

每个 seed 独立目录：

```text
SCOPE_RUNS/SCOPE_CR68_MDN3_v1.0/
├── seed_17/
├── seed_42/
└── seed_2026/
```

单 seed：

```text
seed_17/
├── resolved_config.yaml
├── environment.json
├── logs/
│   ├── train.log
│   └── metrics.csv
├── tensorboard/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── calibration/
└── evaluation/
```

### 23.1 `last.pt` 至少保存

```text
model_state
optimizer_state
scheduler_state
epoch
best_value
stop_reference
bad_epochs
model_seed
python_rng_state
numpy_rng_state
torch_rng_state
cuda_rng_state
protocol_id
feature_protocol_hash
standardizer_hash
train_manifest_hash
val_manifest_hash
config
```

### 23.2 `best.pt`

至少保存：

```text
model_state
epoch
val_nll
seed
protocol_id
feature_protocol_hash
standardizer_hash
config
```

恢复训练必须从 `last.pt` 的下一 epoch 继续，不得用 `best.pt` 伪装完整 resume。

---

## 24. 训练日志与 TensorBoard

这是工程监控，不改变方法。

每 epoch 至少记录：

```text
epoch
train_nll
val_nll
learning_rate
best_value
stop_reference
bad_epochs
elapsed_seconds
```

建议诊断记录：

```text
mean mixture weights pi_k
min/median/max sigma
mean patch_nll
mean image_nll
grad_norm_before_clip
```

这些只用于诊断，不得自动触发样本删除、K 变更或损失调整。

训练结束后可另外输出：

```text
最高 train NLL 的若干 image_id
最高 val NLL 的若干 image_id
```

仅用于人工排查，禁止第一版自动移除这些图后重训。

---

## 25. 三个 seed 的独立训练

三个 seed：

```text
17, 42, 2026
```

必须共享：

- 同一 manifests；
- 同一 raw feature caches；
- 同一 standardizer；
- 同一正式配置；
- 同一 GenImage manifest；
- 同一 COCO external manifest。

只允许改变：

- 模型初始化；
- 每 epoch train image 顺序。

每个 seed 自己产生：

- best checkpoint；
- calibration threshold；
- GenImage 结果；
- COCO FPR。

不得 ensemble；不得根据 GenImage 结果选择表现最好的 seed。

默认展示单模型如需一个固定 seed，则用 `42`。

---

## 26. 校准

实现 `evaluation/calibration.py` 与 `scripts/calibrate.py`。

对某个 seed：

1. 加载该 seed `best.pt`；
2. 加载唯一冻结 standardizer；
3. 对 2000 张 `real_calibration` 计算图级 `S(x)`；
4. 使用：

```python
tau = np.quantile(real_scores, 0.95, method="higher")
```

5. 判定：

```text
S > tau  → AI / anomaly / label 1
S <= tau → Real / label 0
```

保存：

```text
calibration/scores.csv
calibration/calibration.json
```

`calibration.json` 至少包含：

```text
threshold
quantile
method
num_images
checkpoint_sha256
standardizer_sha256
calibration_manifest_sha256
protocol_id
seed
```

阈值必须每个 seed 独立计算。

---

## 27. 统一推理函数

实现 `evaluation/inference.py`，不要在多个评测脚本重复写评分逻辑。

提供至少两个入口：

```text
score_cached_features(...)
score_image(...)
```

其中正式批量评测优先使用缓存特征；单图演示可直接从图片走完整解码和 C/R 提取流程。

必须输出：

```text
patch_nll [4]
image_score S
prediction
threshold
upscaled
```

训练、校准、COCO、GenImage 必须共享相同 MDN NLL 函数。

---

## 28. GenImage 评测

实现 `scripts/eval_genimage.py` 和 `evaluation/metrics.py`。

### 28.1 标签

```text
Real = 0
AI   = 1
```

连续 score 使用原始 `S(x)`，不得 sigmoid。

### 28.2 每个 generator 报告

至少：

```text
n_real
n_ai
AUROC
AP
FPR
TPR
balanced_accuracy
coverage
num_errors
```

阈值指标使用该 seed 自己的 `tau`。

### 28.3 总体指标

至少：

```text
Macro AUROC
Overall AUROC
Overall AP
Overall FPR
Overall TPR
Overall balanced accuracy
```

Macro AUROC：先逐 generator 算 AUROC，再等权平均。

Overall AUROC：把冻结清单的所有条目合并后按样本计算。

### 28.4 upscaled 分组

分别报告：

```text
upscaled == true
upscaled == false
```

的样本数和可计算指标。

如果某组只有单类别，则 AUROC/AP 写为 `null` / `NaN` 并报告该类错误率，不要伪造 AUC。

### 28.5 错误覆盖率

协议内、可解码图像必须都有分数。

评测错误必须保留记录：

```text
image_id
status
error_type
error_message
```

不能为了提高指标静默删掉失败项。

---

## 29. COCO external-real 评测

实现 `scripts/eval_coco.py`。

COCO external 全部 label=0，只报告：

```text
N
FPR = mean(S > tau)
score mean/std/median
score quantiles (可作为诊断)
coverage
num_errors
upscaled subgroup FPR
```

COCO 结果不得反馈到模型、标准化、checkpoint、阈值或超参数。

---

## 30. 三 seed 汇总

实现 `evaluation/summarize.py` 和 `scripts/summarize_seeds.py`。

对每一项指标，先得到三个 seed 值：

```text
m1, m2, m3
```

报告：

```text
mean = (m1+m2+m3)/3
sample_std = std(ddof=1)
```

最终保留：

```text
per_seed_metrics.csv
summary_mean_std.csv
summary.json
```

Macro AUROC 必须先在每个 seed 内计算，再对三个 seed 的 Macro 值做 mean±std。

禁止把三个 seed 的预测行拼起来当成三个独立测试集后重算一个 AUROC。

---

## 31. 单元测试：必须锁定的方法行为

测试不能只验证“能跑”，必须验证公式。

### 31.1 `test_decoding.py`

至少：

- 普通 RGB PNG/JPEG 输出 `[H,W,3] uint8`；
- 灰度图复制三通道且三个通道相等；
- RGBA 白底 alpha composite 正确；
- EXIF orientation 转置后尺寸/像素方向正确；
- PNG metadata 不同但规范化像素相同 → content_id 相同；
- 16-bit / float / animated input 返回 unsupported；
- 损坏输入返回 decode_error。

### 31.2 `test_patches.py`

锁定：

```text
64  → a=b=0
80  → a=0, b=16
128 → a=0, b=64
256 → a=32, b=160
```

检查四块顺序和 `(y,x)` 语义。

### 31.3 `test_cr_features.py`

必须有常数图测试。

例如所有像素恒为 `128/255`：

- low-pass 后仍为常数；
- E 全零；
- C4=0；
- C5=0；
- C6=0；
- R1-R3 = `log(1e-4)`；
- R4-R8 = 0。

另写人工小矩阵测试：

- regularized correlation 与手算一致；
- 水平配对数 3306；
- 通道配对数 3364；
- Fisher clip 绝不输入 ±1；
- C/R shape 与顺序固定。

### 31.4 `test_standardization.py`

- 使用 ddof=0；
- 只使用 train array；
- std < 1e-6 时 scale=1；
- 不 clip z；
- save/load round trip 一致；
- protocol hash 不匹配时拒绝复用。

### 31.5 `test_mdn.py`

正式 K=3：

```text
input [B,6]
logits [B,3]
mu [B,3,8]
sigma [B,3,8]
parameter count = 7923
```

再测试 K=4 自动得到 output_dim=68，不需修改源码。

所有 sigma > 0.05 或数值上至少不低于 floor。

### 31.6 `test_nll.py`

- 与独立手算/参考实现一致；
- 保留高斯常数项；
- mixture 使用 logsumexp；
- 极小概率不下溢为 0；
- NLL 允许负值；
- image loss 是 4 patch 均值。

### 31.7 `test_manifests.py`

- content_id 去重；
- ImageNet 跨源重复优先保留；
- train/val/calibration 不重叠；
- eval IDs 被训练候选排除；
- salt 排序确定性；
- 候选不足时失败。

### 31.8 `test_calibration.py`

构造固定 score 数组，验证：

```text
np.quantile(..., method="higher")
```

以及：

```text
S == tau → Real
S > tau  → AI
```

### 31.9 `test_metrics.py`

使用人工 labels/scores 检查：

- AUROC；
- AP；
- FPR；
- TPR；
- balanced accuracy；
- Macro AUROC；
- Overall AUROC。

### 31.10 `test_smoke_pipeline.py`

在 pytest `tmp_path` 中动态生成少量 synthetic JPEG/PNG：

```text
构建 mini manifests
→ 提取 C/R cache
→ fit standardizer
→ 初始化 MDN
→ 跑 1~2 epoch 小训练
→ 保存/加载 checkpoint
→ calibration
→ synthetic evaluation
```

此 smoke 只验证端到端工程流程，不作为方法效果证据。

---

## 32. 本地验收命令

正式提交前必须全部通过：

```bash
python -m compileall scope_data features models losses engine evaluation scripts tests
python -m pytest -q
```

再运行一个显式 smoke CLI（如实现）：

```bash
python scripts/smoke_test.py
```

如果不额外提供 smoke script，则 `test_smoke_pipeline.py` 必须覆盖完整流程。

还要执行：

```bash
git diff --check
git status
```

### 32.1 禁止的本地动作

本任务阶段不要：

- 下载 ImageNet / LSUN / COCO / GenImage；
- 在本机生成正式 10000/2000/2000 manifest；
- 运行正式 50 epoch；
- 运行三个正式 seed；
- 使用任何真实 GenImage 结果改代码或参数。

---

## 33. README 要求

README 必须简洁说明：

1. SCOPE 是 real-only 条件残差统计异常检测；
2. 核心 `C∈R^6`、`R∈R^8`、`P(R|C)`、3-component MDN；
3. 仓库目录；
4. 环境安装；
5. 外部 `SCOPE_DATA` / `SCOPE_RUNS` 布局；
6. 正式执行顺序；
7. 各脚本命令样例；
8. 数据隔离规则；
9. 不要把 NLL 解释为 AI 概率；
10. `docs/METHOD_SPEC.md` 是方法定义。

正式 workflow 推荐写成：

```text
build_manifests
→ extract_features
→ fit_standardizer
→ train(seed 17/42/2026)
→ calibrate(each seed)
→ eval_coco(each seed)
→ eval_genimage(each seed)
→ summarize_seeds
```

---

## 34. `pyproject.toml` 与依赖

最低依赖：

```text
numpy
Pillow
PyYAML
pandas
lmdb
scikit-learn
tensorboard
pytest (test extra)
```

PyTorch 是运行 MDN 所需依赖，但 **不要在 AutoDL 上通过 requirements 强行覆盖已有 CUDA PyTorch**。

建议 README 明确：

- 本机/新环境先安装与硬件匹配的 PyTorch；
- 再 `pip install -e .`；
- AutoDL 若已有可用 PyTorch，不执行会覆盖 CUDA stack 的固定 wheel 安装。

不要在 `requirements.txt` 写死某个 CUDA index，除非用户之后明确要求。

---

## 35. 正式脚本接口建议

### 35.1 manifests

```bash
python scripts/build_manifests.py \
  --config configs/scope_cr68_mdn3.yaml \
  --paths-config configs/local_paths.yaml \
  --output-dir ../SCOPE_DATA/manifests/SCOPE_CR68_MDN3_v1.0
```

### 35.2 features

```bash
python scripts/extract_features.py \
  --config configs/scope_cr68_mdn3.yaml \
  --manifest ../SCOPE_DATA/manifests/SCOPE_CR68_MDN3_v1.0/real_train.csv \
  --split real_train \
  --output-root ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0
```

对 val / calibration / COCO / GenImage 同样调用。

### 35.3 standardizer

```bash
python scripts/fit_standardizer.py \
  --config configs/scope_cr68_mdn3.yaml \
  --train-cache ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0/real_train \
  --output ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0/standardizer.json
```

### 35.4 train

```bash
python scripts/train.py \
  --config configs/scope_cr68_mdn3.yaml \
  --train-cache ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0/real_train \
  --val-cache ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0/real_val \
  --standardizer ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0/standardizer.json \
  --seed 17 \
  --run-dir ../SCOPE_RUNS/SCOPE_CR68_MDN3_v1.0/seed_17
```

### 35.5 calibrate

```bash
python scripts/calibrate.py \
  --config configs/scope_cr68_mdn3.yaml \
  --cache ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0/real_calibration \
  --standardizer ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0/standardizer.json \
  --run-dir ../SCOPE_RUNS/SCOPE_CR68_MDN3_v1.0/seed_17
```

### 35.6 eval

```bash
python scripts/eval_coco.py ...
python scripts/eval_genimage.py ...
```

### 35.7 summarize

```bash
python scripts/summarize_seeds.py \
  --runs-root ../SCOPE_RUNS/SCOPE_CR68_MDN3_v1.0
```

具体 CLI 参数名可以在不改变语义的前提下略微调整，但 README 和代码必须一致。

---

## 36. 实施顺序

严格按下面顺序工作，不要一次性堆完所有代码再排错。

### Phase 0：Bootstrap

创建：

- 基础目录；
- `.gitignore`；
- `pyproject.toml`；
- `AGENTS.md`；
- `configs/`；
- `docs/`；
- README 框架。

验收：包可安装、空测试框架可运行。

### Phase 1：固定图像协议

实现：

- decoding；
- canonical content ID；
- small-image resize；
- four-patch coordinate extraction。

先写并通过相关测试。

### Phase 2：C/R 固定特征

实现：

- low-pass；
- C1-C6；
- R1-R8；
- finite checks。

必须先通过常数图和人工矩阵单元测试，再继续。

### Phase 3：manifest 与 feature cache

实现：

- sources；
- LSUN LMDB adapter；
- deterministic split；
- isolation；
- cache artifact；
- protocol hash。

### Phase 4：standardizer

实现 train-only mean/std/scale 与 artifact。

### Phase 5：MDN + NLL

实现：

- configurable K；
- exact initialization；
- sigma parameterization；
- exact mixture NLL。

锁定 7923 参数测试。

### Phase 6：trainer

实现：

- deterministic epoch permutation；
- AdamW parameter groups；
- finite checks；
- grad clipping；
- validation；
- ReduceLROnPlateau；
- early stop；
- best/last checkpoint；
- logging/TensorBoard；
- resume。

### Phase 7：calibration / evaluation / summary

实现：

- quantile higher；
- threshold rule；
- GenImage metrics；
- COCO FPR；
- three-seed mean±sample std。

### Phase 8：End-to-end smoke

用 synthetic data 跑通：

```text
manifest
→ feature extraction
→ standardization
→ train
→ checkpoint reload
→ calibration
→ evaluation
```

### Phase 9：文档和最终审查

检查 README、METHOD_SPEC、IMPLEMENTATION_SPEC、CLI 与代码一致。

---

## 37. 方法一致性检查清单

最终提交前逐条确认：

- [ ] 正式训练没有任何 AI 图；
- [ ] 每图固定 4 个 64×64 patch；
- [ ] 小图只在短边 <64 时等比例双线性放大；
- [ ] 5×5 valid 低通，无 padding；
- [ ] 最终内部区域 58×58；
- [ ] C 精确 6 维且顺序固定；
- [ ] R 精确 8 维且顺序固定；
- [ ] delta=1e-4；
- [ ] Fisher clip ±0.999；
- [ ] feature extraction CPU float64；
- [ ] standardizer 只 fit real_train；
- [ ] 标准化后不 clip；
- [ ] MDN 正式 K=3；
- [ ] covariance 为 diagonal；
- [ ] sigma=0.05+softplus；
- [ ] 完整 Gaussian log-density 常数项保留；
- [ ] patch mixture 使用 logsumexp；
- [ ] image score = 4 patch NLL 算术平均；
- [ ] train loss = image score batch mean；
- [ ] batch=256 images；
- [ ] AdamW bias weight decay=0；
- [ ] grad clip=5；
- [ ] AMP off；
- [ ] TF32 off；
- [ ] val 只用于 scheduler / early stop / best；
- [ ] calibration 只用于 tau；
- [ ] GenImage 不影响任何训练决策；
- [ ] tau=95% quantile, method=higher；
- [ ] S==tau 判 Real；
- [ ] seeds=17/42/2026；
- [ ] 三 seed 不 ensemble；
- [ ] mean±std 使用 ddof=1；
- [ ] 本地没有启动正式训练。

---

## 38. 明确禁止实现的内容

当前第一版不要加入：

- `P(R)` 无条件 baseline；
- shuffled C-R 配对实验；
- 自动异常真图剔除；
- 高 NLL 样本降权；
- robust loss；
- 可学习 C/R 特征提取器；
- CNN / ViT；
- 数据增强；
- AI 图参与训练；
- 用 GenImage 调 K、阈值、feature、checkpoint；
- mixture component balancing loss；
- full covariance Gaussian；
- 图级共享 mixture component；
- 多 patch NLL 乘积概率解释；
- 三 seed ensemble；
- sigmoid 后再做 AUC。

如果觉得这些能提高性能，只能在最终报告中列为后续建议，不得在本次实现中加入。

---

## 39. Git 提交与推送

全部验收通过后：

```bash
git status
git diff --check
python -m compileall scope_data features models losses engine evaluation scripts tests
python -m pytest -q
```

确认没有数据文件、run、cache、模型权重被跟踪。

然后：

```bash
git add .
git commit -m "feat: implement SCOPE CR68 conditional MDN pipeline"
git push -u origin main
```

不要 force push。

如果远端在执行期间出现他人新提交，停止并报告，不要自动 rebase/merge 覆盖。

---

## 40. Codex 最终报告格式

完成后必须在对话中给出一份简洁但完整的报告，至少包含：

```text
IMPLEMENTATION_COMPLETED = YES / NO

1. Git
- repository
- branch
- starting state
- final commit SHA
- pushed to origin/main: YES/NO

2. Implemented modules
- decoding
- manifests
- feature extraction
- standardization
- MDN
- NLL
- trainer
- calibration
- evaluation
- summary

3. Method conformance
- C/R dimensions
- K
- parameter count
- sigma floor
- train/val/calibration isolation
- seed handling

4. Tests
- compileall result
- pytest result (passed/failed count)
- smoke pipeline result

5. Artifacts intentionally NOT produced
- no formal manifests
- no formal feature cache
- no 50-epoch training
- no GenImage benchmark result

6. Remaining issues
- only genuine blockers or known limitations
```

若未完成任何一项，不得写 `IMPLEMENTATION_COMPLETED = YES`。

---

# 附录 A：正式方法最小数据流

```text
JPEG / PNG
    ↓ fixed decode
RGB uint8
    ↓ optional short-side upscale
RGB float64 [0,1]
    ↓ 4 deterministic 64×64 patches
P[4,64,64,3]
    ↓ fixed 5×5 valid low-pass
P_inner / L_inner / E [4,58,58,3]
    ↓ fixed statistics
C_raw [4,6]
R_raw [4,8]
    ↓ train-only standardizer
C_z [4,6]
R_z [4,8]
    ↓ MDN(C_z)
π [4,K]
μ [4,K,8]
σ [4,K,8]
    ↓ full mixture NLL against R_z
patch_nll [4]
    ↓ mean
S(x)
    ↓ compare with seed-specific real calibration tau
Real / AI
```

---

# 附录 B：本次实现完成后的下一步，但不要现在执行

代码推送完成并由用户检查 GitHub 后，下一阶段才是在 AutoDL：

```text
1. git clone / git pull 新仓库
2. 配置真实数据路径
3. 构建并冻结 manifests
4. 提取所有 split 的 C/R cache
5. 检查 feature 统计与异常值
6. fit train-only standardizer
7. 分别正式训练 seed 17 / 42 / 2026
8. 每个 seed 用 real_calibration 得到 tau
9. COCO external-real 评测
10. GenImage 冻结 benchmark 评测
11. 汇总 mean ± std
```

本实施任务到 GitHub 代码推送即停止。
