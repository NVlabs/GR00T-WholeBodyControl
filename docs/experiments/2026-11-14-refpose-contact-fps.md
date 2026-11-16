# refpose 接触损失的帧率缺陷：修好它，并测出接触权重真正的性价比

| 字段 | 内容 |
| --- | --- |
| 实验 ID / 类型 | `2026-11-14-refpose-contact-fps`；缺陷修复 + 训练 + 离线评估 |
| 开始、结束时间与时区 | 2026-11-14 20:29 起跑第一组，23:16 两组完整评测结束；w0.4 于 ep3 初中止，未评四档（Asia/Shanghai） |
| 状态 | 完成（对照组 w0.4 因发散在 ep3 初主动中止，见「异常与处理」） |
| 执行者与来源 | 本项目执行者；模型与训练入口沿用 [refpose v2 全量 20 epoch](2026-11-14-refpose-v2-20ep.md) |
| 关联任务或前序记录 | [refpose v2 20 epoch](2026-11-14-refpose-v2-20ep.md)；[refpose v2 30 epoch](2026-11-14-refpose-v2-30ep.md)；[reference-guided 数据契约](../reference-guided-data-contract-v1.md) |

## 问题与方案

- **研究问题（先是缺陷，后是对照）**：接触损失把脚速换算的帧率写死成 30，而 full_release_v2 契约是 120.0048 fps。脚速进了平方，整项按 (120/30)² = 16 倍缩放，因此此前所有轮次日志里记的 `contact_weight=0.4` 实际只生效 1/16（等效 0.025）。修掉这个缺陷之后，真正的问题是：**0.4 这个数还算不算合理？接触权重应该取多少？**
- **本次操作**：
  1. 把帧率从写死改为跟契约走（`fps` 参数 + 调用方传 `contract["fps"]`），并用数值等价验证确认修复本身不改变任何已调好的强度。
  2. 修好后跑三组强度对照：`contact_weight` = 0.4 / 0.1 / 0.025；其中 w0.4 在 ep3 初因发散中止，未跑满 20 epoch。
- **对照组**：上一轮 20 epoch（`refpose_v2_20ep`，fps 写死 30 + weight 0.4，即**实际强度恰为 0.025**）。同契约、同 test 集、同选窗口径、同源动作脚速分母（`source_contact_foot_speed_mps` 各轮均为 0.03105057869044021），因此 **w0.025 与它有可比性，是「修复本身是否中性」的检验**。
- **固定条件**：契约、骨架、划分、种子 42、batch 256、width 256、lr 3e-4 余弦退火 20 epoch（T_max = 155,620 步）、chunk 16、workers 8 全部相同；唯一变化因素是接触权重的实际强度。
- **预定评价指标**：验证集 RMSE 逐 epoch；test 四档硬参考 RMSE（cm）、中位数、最差 clip、相对 Lerp 倍数；脚滑倍数（pooled 与逐 clip 均值两个口径都记）。

## 代码、环境与数据

- 仓库提交 SHA：`32eba6c`（`refactor(repo): 收敛 refpose 边界并分组 data/tools`）。**工作区 clean**，修复与测试都在该提交里，训练与评测用的就是工作区代码。
- 改了什么：
  - `model/MotionWeaver/refpose/losses.py:53`：`speed = torch.linalg.vector_norm(...) * fps`（原为 `* 30`），`losses()` 签名新增 `fps=30.0`。默认值 30 只为兼容冻结的 legacy fixture（`tests/fixtures/refpose_legacy.py` 也写 30，是历史副本，故意不动）。
  - `training/pretrain/train_conditioned_pose.py:109`：`fps = float(contract.get("fps") or 30.0)`，`validate()` 新增 `fps` 形参并透传；`config` 增加 `contact_fps`。
  - `tools/train_reference_pose.py:165`：同样从契约取 `fps`，损失调用传 `fps=fps`（第 271 行）。
  - `tools/bench_training_throughput.py`：`bench_end_to_end` 增加 `fps` 形参并透传（第 105/115/214 行）。
  - `tests/test_conditioned_pose_training.py`：新增 `test_contact_loss_scales_with_contract_fps_squared`，锁定 (fps/30)² 缩放关系与「省略 fps 时等于旧的 30」。
- **修复的等价性（数值验证，不是估算）**：同一批随机张量上

```text
旧口径 fps=30  w=0.4    total = 2774.2341308594
新口径 fps=120 w=0.025  total = 2774.2341308594
相对差 = 0.000e+00
不改权重 fps=120 w=0.4  total = 43895.851562（比旧口径大 1482%）
```

  即：只要把 weight 按 16 倍折算，修复前后损失逐位相同；**修复本身中性，动的是「weight 这个数到底代表多强的约束」**。
- 实际命令（经 `tools/run_detached.sh` 跑成独立 systemd user unit `mw-train`，三组只换 `--contact-weight`）：

```text
tools/run_detached.sh --name mw-train \
  --log /home/acear/MotionWeaverData/train_v3d_w0025.log \
  --mem-high 10G --mem-max 16G --swap-max 8G -- \
  .venv/bin/python tools/train_reference_pose.py \
    --release /home/acear/MotionWeaverData/full_release_v2 \
    --mmap-root /home/acear/MotionWeaverData/full_mmap_v2 \
    --output /home/acear/MotionWeaverData/runs/refpose_v3d_fps120_w0025_20ep \
    --epochs 20 --batch-size 256 --workers 8 --log-every 100 \
    --contact-weight 0.025 --chunk-size 16 --seed 42 --width 256 --lr 3e-4
```

  评测：`RUN_DIR=... OUT_DIR=... TRAIN_UNIT=mw-train tools/run_full_evaluation.sh` 同样走独立 unit（环境变量必须用 `--env` 显式传，见 [run-detached-jobs](../../.agents/skills/run-detached-jobs/SKILL.md)）。
- 数据：`/home/acear/MotionWeaverData/full_release_v2`，契约 fps = 120.00480019200768，clips 71,128，窗口 2,490,713，split {train 56,972 / validation 7,110 / test 7,046}；train 窗口 1,992,148、val 247,979；每 epoch 7,781 步。骨架 sha256 前缀 `159050ea134c282c`。
- 软件与设备：Python 3.14 venv、PyTorch 2.15.0 (CUDA)、RTX 4070 Laptop 8 GB、30 GB RAM + 46 GB swap。

## 结果与证据

### 三组强度对照（test，7,046 clip 中心窗口，oracle root）

| 强度（fps 修正后的 contact_weight） | 等价旧口径 | 验证 RMSE ep19 | 四档 RMSE（无参考 / 中 / 末 / 中末） | 相对 Lerp 倍数 |
| --- | ---: | ---: | --- | --- |
| **0.025**（本轮与旧轮同强度） | 0.025（= 旧脚本里的 0.4） | **4.086** | **5.65 / 3.52 / 3.09 / 2.49** | 2.78 / 1.98 / 2.28 / 1.39 |
| 0.1（×4） | 0.1 | 5.159 | 6.89 / 4.61 / 4.04 / 3.39 | 2.28 / 1.51 / 1.74 / 1.02 |
| 0.4（×16，本轮发散，未评四档） | 0.4 | ep2 时 10.00 | — | — |
| 参照：旧 20ep（fps 写死 30，weight 0.4） | 0.025 | 4.047 | 5.59 / 3.50 / 3.04 / 2.46 | 2.81 / 1.99 / 2.31 / 1.40 |

脚滑（pooled，四档同序）：

| 组 | 无参考 | 中间 1 帧 | 末帧 | 中末 |
| --- | ---: | ---: | ---: | ---: |
| w0.025 | 6.603 | 8.296 | 8.601 | 9.063 |
| w0.1 | **5.885** | **7.515** | **8.179** | **8.588** |
| 旧 20ep（等效 0.025） | 6.616 | 8.243 | 8.538 | 8.998 |

> 脚滑两个口径都在 `model.json` 里：`foot_slip_ratio` 是 pooled（先合后除），`foot_slip_ratio_mean` 是逐 clip 求比再等权平均。上表用 pooled。w0.025 的逐 clip 均值是 9.996 / 12.933 / 11.815 / 13.295，旧 20ep 是 10.003 / 12.848 / 11.749 / 13.190。

### 验证曲线

```text
w0.025  8.32 7.07 6.50 6.06 5.69 5.38 5.28 5.04 4.84 4.88 4.77 4.54 4.46 4.42 4.29 4.22 4.15 4.11 4.09 4.09
旧 20ep 8.40 7.12 6.40 6.00 5.78 5.34 5.27 5.06 4.84 4.85 4.67 4.52 4.58 4.36 4.25 4.19 4.12 4.07 4.06 4.05
w0.1   11.12 8.32 7.69 7.29 7.02 6.59 6.56 6.31 6.10 6.01 5.80 5.64 5.56 5.46 5.41 5.27 5.22 5.18 5.17 5.16
w0.4  15.68 11.47 10.00（ep3 初中止）
```

- 检查点：
  - `/home/acear/MotionWeaverData/runs/refpose_v3d_fps120_w0025_20ep/best.pt`，sha256 `3630016cc825170bfb635e1f63df0a2b43da2ebabf83a03b1b5d27cbf1364f8f`，37,093,072 字节，epoch 19，val 4.0863。
  - `/home/acear/MotionWeaverData/runs/refpose_v3c_fps120_20ep/best.pt`，sha256 `cbebd0d81bbf47a327bdf0395af739ac2b5d8f0ad7bbb4bf017c26d854843b55`，同一尺寸，epoch 19，val 5.1593。
- 日志：`train_v3d_w0025.log`、`train_v3c_fps120.log`、`train_v2_contactfix_20ep.log`（w0.4 中止那组）、`probe_cfix_w010.log`（ep0–ep1 探针）；评测日志 `eval_v3d_w0025.log`、`eval_v3c_fps120.log`。
- 报告：`reports/refpose_v3d_w0025_test/`、`reports/refpose_v3c_fps120_test/`（各自的 `model.json` / `baselines/report.json` / `comparison.json` / `comparison.md`）。
- Windows 交付副本：`C:\MotionWeaverData\model_weight\refpose_v3c_fps120_20ep\` 与 `C:\MotionWeaverData\model_weight\refpose_v3d_fps120_w0025_20ep\`，各含权重、`history.json`、`evaluation/`、`logs/` 和 `SHA256SUMS.txt`；已逐文件核对。
- 用时：w0.1 4,954 s、w0.025 5,071 s（各 20 epoch，含验证）。均以 `Result=success ExecMainStatus=0` 结束。
- 异常与处理：
  1. **w0.4 组发散并主动中止**：fps 改成 120 后 weight 0.4 的实际强度变成旧轮的 16 倍，训练侧 loss 从旧轮同期的 1.42（step100 口径）升到 13.93，验证 RMSE ep0 15.68 / ep1 11.47 / ep2 10.00，明显不会在 20 epoch 内追上。在 ep3 初 `systemctl --user stop mw-train` 并 `gio trash` 该目录。**这是设计上的否证，不是故障**——结论是「0.4 + 正确 fps」这个组合在本数据上不可用。
  2. 中止后误启动了一个完全同参的 `refpose_v2_cfix_w04_20ep`，立刻 stop 并 trash，日志 `train_cfix_w04.log` 留在盘上作为痕迹。
  3. 为在开整轮前先验量级，用 `--max-steps` 做过 300 步冒烟（该参数本轮新增，跑满即 break、不验证不写检查点）。冒烟显示 w0.4 的 loss 比旧轮同期大 8–13 倍，是这次没有直接把 0.4 跑满 20 epoch 的原因。
  4. 评测 watcher 首次启动时参数顺序写错（`--` 放到了 `--env` 前面），`run_detached.sh` 报 `Failed to find executable --env` 并直接退出，没有产生任何副作用；改正顺序后重跑。
- 人工视觉复核：未做，本记录只覆盖离线数值。

## 结论与论文使用边界

- **直接测得：**
  1. 缺陷是真的：`losses.py` 的脚速帧率写死 30，与 120 fps 契约差 16 倍（平方关系），此前所有轮次的 `contact_weight=0.4` 实际只有 0.025 的强度。
  2. 修复是**中性**的：把 weight 折算成 0.025 后，损失与旧口径逐位相同（相对差 0.000e+00）。因此本轮 w0.025 与旧 20ep 的差异只反映「取契约 fps 后数值细节的变化」，不反映强度变化：验证 4.086 vs 4.047（+0.04），四档 5.65/3.52/3.09/2.49 vs 5.59/3.50/3.04/2.46（差 +0.02–+0.07 cm），脚滑 6.60–9.06 vs 6.62–9.00。差异很小；本轮每组只跑了一个 seed，不能严格归因于噪声，但在本实验口径下可认为等价。
  3. 加大接触强度是**用位置精度换脚滑，且非常不划算**：w0.1（×4）让四档 RMSE 涨 21.8% / 31.2% / 30.9% / 35.9%，只换到脚滑降 10.9% / 9.4% / 4.9% / 5.2%；w0.4（×16）直接把训练推到发不出来的状态。
- **基于结果的推断：** 当前损失里位置项（系数 10）与接触项的量级关系决定了这个交换比——接触项在 0.025 时已经足够改变脚部动力学，再往上只是让模型为了压脚速而牺牲整体姿态。想同时降误差和脚滑，这个旋钮不行。
- **尚未验证、不能据此宣称：** 未做 UE 内蒙皮/IK/碰撞与连续玩家控制验证；Root 是 oracle；未做人工视觉复核，「脚滑 9× 看起来能不能接受」没有主观证据；只测了 3 个强度点，没有做 0.025–0.1 之间的细扫（例如 0.05）——不排除中间存在更优点的可能，但那需要在「位置涨 20% 换脚滑降 10%」的斜率上找，先验上不乐观。
- **口径提醒：** w0.025 与旧 20ep 可比（同契约、同 test、同分母）；但两者**都不能与 v3 封盘直接比**，理由见 [20 epoch 记录](2026-11-14-refpose-v2-20ep.md) 的四条口径差异。
- **后续**：不要再靠调 `contact_weight` 压脚滑。脚滑仍远高于 1.0（6.6–9.1×），按既有决策交 UE 端 Two-Bone IK / Foot Locking / 步幅匹配补齐；若要继续在训练侧改善，应改条件输入方式或损失形式（例如对接触帧单独加权、按脚速分箱），而不是调这一个标量。

## 修订历史

| 日期 | 变更 | 依据 |
| --- | --- | --- |
| 2026-11-14 | 首次记录：帧率缺陷、等价性验证、三组强度对照与结论 | 数值等价脚本输出、两组 20 epoch 日志、`comparison.json`、检查点 sha256 |

## 提交对照

修复与测试已包含在 `32eba6c`（`refactor(repo): 收敛 refpose 边界并分组 data/tools`），训练与评测时工作区 clean，代码状态与该提交一致。`--max-steps` 冒烟参数也在该提交内。
