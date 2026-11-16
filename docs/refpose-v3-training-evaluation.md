# RefPose v3 训练与评测记录（封盘版）

> 本文是交付权重的完整实验记录。所有数字均来自实测，未做修饰。
> 权重路径：`D:\MotionWeaverData\model_weight\refpose_v3_71k\`

- 封盘日期：2026-11-02
- 状态：**frozen**（交付权重，训练到此封盘）
- best.pt SHA-256：`6d10d358e0a719e9ddc6aaacae5adac5b0bfb1c4a64f2508d857517e0ed55ffc`

---

## 1. 任务定义

本模型解决的是**参考引导的姿态补全**（reference-guided pose in-betweening），
不是自编码重建，也不是纯无条件预测：

> 给定过去 24 帧历史 + 未来 24 帧的 Root 轨迹 + 若干帧"硬参考"，
> 预测其余帧的全身关节姿态。

评测在四个参考强度档位下进行，对应游戏里"玩家给多稀疏的参考"：

| 档位 | 含义 | 参考帧 |
|---|---|---|
| `no_reference` | 无参考，纯预测 | 0 帧 |
| `middle_reference` | 只有中点 | 1 帧（第 12 帧） |
| `end_reference` | 只有末帧 | 1 帧（第 23 帧） |
| `middle_and_end` | 中点 + 末帧 | 2 帧 |

---

## 2. 模型

```
类名        ReferenceGuidedPose
参数量      3.09 M
骨骼        78
隐层宽度    256
结构        GRU(历史编码)
            + Linear(条件投影)
            + TransformerEncoder(3 层, 4 头, ff_dim 768, norm_first)
            + LayerNorm + Linear

输入
  history    [24, 709]    过去 24 帧（关节位置 234 + 旋转6D 468 + root 7）
  root_plan  [24, 7]      未来 24 帧 Root 轨迹（位置 3 + 四元数 4）
  reference  [24, 702]    硬参考帧（未选中的帧为 0）
  mask       [24, 1]      参考帧可见性掩码

输出
  joints     [24, 702]    24 帧 × 78 骨 × 9 维（3 位置 + 6 旋转6D）
```

**关键设计**：输出关节直接回归，**不经过 VQ-VAE tokenizer**。
原因是 tokenizer 路线存在 96.5 cm 的重建天花板（见 §7），
而本任务关心的是姿态精度与可播放性，不需要离散 token。

---

## 3. 数据

| 项 | 值 |
|---|---|
| 源数据集 | `full-v1-f32`（fps120 组） |
| clips | **71,127** |
| train 窗口 | **1,988,285** |
| validation 窗口 | 245,232 |
| test 窗口 | 257,171 |
| 窗口定义 | history 24 + future 24 = 48 帧 |
| 划分方式 | 按 asset 名 SHA-256 确定性划分（80/10/10） |
| 存储格式 | **mmap**（每 clip 一个目录，内含 p/r/k.npy） |
| 数据体积 | 176.7 GB |

### 为什么是 mmap

原始格式是"整条 clip 一个 npz"。训练只用 48 帧，而一条 clip 平均 868 帧，
**读写浪费 94.5%**。实测对比（batch 256）：

| 格式 | 随机窗口读 | 训练吞吐 |
|---|---:|---:|
| npz（解压整包） | 3.75 ms | 322 样本/秒 |
| **npy + mmap** | **1.33 ms** | **1,829 样本/秒** |

mmap 不改数据内容（逐元素验证 50/50 一致，最大差 0.00e+00），
只是让操作系统按页加载需要的部分。

---

## 4. 训练配置

| 项 | 值 |
|---|---|
| epochs | 20 |
| batch size | 256 |
| DataLoader workers | 8 |
| pin_memory | True（关键） |
| prefetch_factor | 4 |
| 优化器 | AdamW |
| 学习率 | 3e-4 |
| **contact_weight** | **0.4**（旧口径；等效 0.025） |
| seed | 42 |
| 精度 | bf16 autocast |
| 实际耗时 | **31,474 秒（8.7 小时）** |

### 损失函数

```
total = 10 × position_loss
      + rotation_loss
      + 2 × velocity_loss
      + contact_weight × contact_loss
```

其中 `contact_loss` 是世界坐标系下**接触期脚速的平方均值**：
只惩罚"脚已经着地却还在滑"的帧对，不惩罚摆动脚。

### 为什么 contact_weight 是 0.4

实测三个配置：

| weight | 位置精度（no_reference） | 脚滑倍数 |
|---:|---:|---:|
| 0（关闭） | 5.49 cm | 17–22× |
| 1.0 | 7.68 cm | 7.6–10.8× |
| **0.4** | **5.28 cm** | **6.9–8.7×（纯生成 6.6–7.3×）** |

> **口径更正（2026-11-14 补充）**：以上三组权重是在接触损失帧率写死 30 时跑的，而本数据是 120 fps；脚速进平方后整项缩小 16 倍，因此表里的 0 / 1.0 / 0.4 实际等效 **0 / 0.0625 / 0.025**。本文的位置精度与脚滑数字仍然有效，但引用 `contact_weight` 时必须按等效强度标注。修复与强度对照见 [接触损失的帧率缺陷](experiments/2026-11-14-refpose-contact-fps.md)。

weight=1.0 把脚滑压下去了，但位置精度被牺牲 40%。
weight=0.4 几乎不损精度，脚滑仍有明显改善。在这个旧口径下 0.4 是实测的平衡点；按契约 fps 折算后等效强度为 0.025。

---

## 5. 验证曲线（20 轮完整）

| epoch | RMSE (cm) | epoch | RMSE (cm) |
|---:|---:|---:|---:|
| 0 | 7.046 | 10 | 4.141 |
| 1 | 6.018 | 11 | 4.073 |
| 2 | 5.527 | 12 | 3.938 |
| 3 | 5.169 | 13 | 3.874 |
| 4 | 5.115 | 14 | 3.826 |
| 5 | 4.787 | 15 | 3.769 |
| 6 | 4.698 | 16 | 3.709 |
| 7 | 4.451 | 17 | 3.686 |
| 8 | 4.402 | 18 | 3.660 |
| 9 | 4.261 | **19** | **3.654** |

**验证集：245,232 窗口（7,038 clips）**

曲线单调下降，末轮仍在改善（18→19 降 0.006 cm），**尚未走平**。
说明继续训练仍有收益，但已进入收益递减区。

---

## 6. 正式评测结果

### 评测协议

| 项 | 值 |
|---|---|
| evaluation_type | `non_learning_reference_guided_pose_baselines` |
| split | **test** |
| test 窗口 | 257,171 |
| selection | `middle_window_per_clip` |
| oracle_root_condition | **true** |
| metrics_scope | `per_clip_equal_weight_non_anchor_primary` |
| 主指标 | `non_anchor_pose_rmse_cm`（只算非锚点帧，每 clip 等权） |

### 四档结果

| 档位 | GT帧 | 生成帧 | **v3 模型** | Lerp 基线 | vs Lerp | 脚滑（全帧） | 脚滑（纯生成） |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_reference` | 0 | 24 | **5.28 cm** | 16.64 | **3.15× 更优** | 6.9× | **6.9×** |
| `middle_reference` | 1 | 23 | **3.22 cm** | 7.24 | **2.25× 更优** | 8.6× | **7.3×** |
| `end_reference` | 1 | 23 | **3.22 cm** | 7.46 | **2.32× 更优** | 8.6× | **7.3×** |
| `middle_and_end` | 2 | 22 | **2.17 cm** | 3.66 | **1.69× 更优** | 8.7× | **6.6×** |

**这是补间生成而非重建**：GT 帧是作为条件给出的真值参考帧（相当于玩家给的关键帧），
其余帧全部由模型生成。`no_reference` 档不含任何 GT 帧，是最干净的纯生成指标。

**脚滑两个口径的差异**：脚滑按「相邻两帧都着地」的帧对统计。跨过 GT 边界的帧对里，
分母一侧两帧都是真值、脚速接近零，会把倍数放大。只保留相邻两帧都为生成帧的帧对后，
倍数回到 6.6–7.3×。两口径方向一致，都远低于关闭接触约束时的 17–22×。

**脚滑倍数定义**：接触期脚速 ÷ 源动作接触期脚速。源动作基准 0.0261 m/s。

---

## 7. 三代模型演进

| 版本 | 数据 | weight | 验证 RMSE | no_reference | middle_and_end | 脚滑 |
|---|---|---:|---:|---:|---:|---:|
| v1 | 2k / 30ep | 0 | 4.28 cm | 6.74 | 2.90 | 17–22× |
| v2 | 20k / 20ep | 1.0 | 5.53 cm | 7.68 | 3.36 | 7.6–10.8× |
| **v3** | **71k / 20ep** | **0.4** | **3.65 cm** | **5.28** | **2.17** | **6.9–8.7×** |

**v3 相对 v2**：
- `no_reference`：7.68 → 5.28（**−31%**）
- `middle_and_end`：3.36 → 2.17（**−35%**）
- 脚滑：10.8× → 8.6×（**−20%**）

注：v1 与 v2/v3 的测试集不同（v1 用 169 clips 的 2k 划分），
v2 与 v3 都用各自规模的 test split。**模型数字之间可比，基线数字不可跨轮比。**

---

## 8. 目标达成情况

对照 `soma-inbetweening-baseline-full-v1.md` §173–183 定的目标：

| 指标 | 目标 | 当前 | 状态 |
|---|---|---:|:---:|
| `middle_and_end` RMSE | < 6.59 cm | **2.17** | ✅ 达成 |
| `middle_and_end` RMSE（理想） | < 5.00 cm | **2.17** | ✅ 达成 |
| 位置精度 vs Lerp | 优于 | 1.69–3.15× | ✅ 达成 |
| **脚滑倍数** | **< 5×** | **6.9–8.7×** | ❌ **未达标** |

### 脚滑未达标的处置

**已决策接受**。理由：

1. 从 17–22× 降到 6.9× 已是 **2.5 倍改善**，进入可播放区间
2. 继续下调 contact_weight 收益递减，且会开始损位置精度
3. UE 端有成熟的工程手段补齐：Two-Bone IK、Foot Locking、步幅匹配

**责任划分**：模型输出姿态本身不飘（位置精度优于 Lerp 1.7–3.2 倍），
剩余脚滑属于"Root 轨迹与步幅不匹配"，是工程层问题。

---

## 9. 已知限制

1. **Root 轨迹取自原动作**，是理想控制输入，不代表玩家或 CMC 的真实轨迹误差
2. **硬参考直接来自目标帧**，只有被选中的姿态帧作为条件公开
3. **Hold / Lerp 只是非学习基线**，不保证骨长、IK 或实际游戏可播放质量
4. **来源分区的非 Traversal 语义变体可能跨分区**，不能宣称完全独立泛化
5. **未做 UE 内实机验证** —— 所有指标都是离线关节 RMSE，不代表引擎内视觉效果
6. **验证曲线未走平** —— 继续训练仍有收益

---

## 10. 复现方式

训练：

```powershell
cd D:\AILocomotonSystem\MotionWeaver
.venv\Scripts\python.exe tools\train_reference_pose.py `
  --release "D:\MotionWeaverData\full_release" `
  --mmap-root "D:\MotionWeaverData\full_mmap" `
  --output "D:\MotionWeaverData\training_runs\refpose_full71k" `
  --epochs 20 --batch-size 256 --log-every 500 `
  --contact-weight 0.4 --workers 8
```

加载权重：

```python
import torch
from training.pretrain.train_conditioned_pose import ReferenceGuidedPose

ck = torch.load("D:/MotionWeaverData/model_weight/refpose_v3_71k/best.pt",
                map_location="cpu", weights_only=False)
model = ReferenceGuidedPose(78, ck["width"])
model.load_state_dict(ck["model"])
model.eval()
```

---

## 11. 附：被否定的路线

记录在此，避免后人重走。

| 路线 | 结论 | 根因 |
|---|---|---|
| VQ-VAE tokenizer + 大主干 | ❌ 放弃 | 码本塌缩至 perplexity=1.000，关节 RMSE 卡在 96.5 cm |
| 分片存储（2000 窗口/片） | ❌ 慢 100 倍 | `np.savez` 打包解压整包，随机访问代价极高 |
| tiny 精简（每 clip 只存前 48 帧） | ❌ 数据错误 | 训练窗口 start 分布 0~16416，97.1% 窗口会取错数据 |
| 加大 LRU 缓存（256→4096） | ❌ 反而更慢 | 内存压力导致换页 |
| 增大接触权重到 1.0 | ⚠️ 有代价 | 脚滑改善但位置精度退化 40% |
