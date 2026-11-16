# MotionWeaver

MotionWeaver（原 AIAnimationSystem）是面向 Unreal Engine 的 AI 角色动画实验项目，探索从动画数据训练到实时姿态生成的完整流程。

本项目基于 NVIDIA [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) 仓库中的 [MotionBricks](https://nvlabs.github.io/motionbricks/) 开发，由本仓库独立维护。MotionBricks 的模型架构、动作表示、原有训练与推理代码、预训练权重和上游演示属于原项目成果；本项目在其基础上面向游戏动画做裁剪、中文交互界面与 Windows 启动器适配，并添加实验性的 UE 动画导出和自定义骨架训练入口。

> **当前主线：** 参考引导姿态补全（reference-guided pose in-betweening）已跑通全库训练与权威评测，
> 权重封盘于 `D:\MotionWeaverData\model_weight\refpose_v3_71k\`，详见 [训练与评测记录](docs/refpose-v3-training-evaluation.md)。
> 早期离散 Token 路线因码本塌缩已停止推进。
>
> **实验阶段：** 已有训练、评测与 FP32/ONNX 导出工具。UE 侧目前只保留实验宿主与空壳运行时插件；尚未完成 AnimGraph 接线、打包、多角色并发、网络或 CMC / Mover 接入。本项目不代表 NVIDIA 官方产品，也不表示获得 NVIDIA 的认可或背书。

当前先完成 Locomotion、跳跃和蹲伏的数据与动画流程；项目名称覆盖更广泛的角色动画方向，后续动作范围以开发方案与实际验证为准。

[Unreal 集成](unreal-script/README.md) · [UE 实验宿主](unreal-sample/README.md) · [MotionBricks 演示与技术说明](docs/motionbricks/README.md) · [来源、引用与许可](引用与许可说明.md)

## 当前进度

| 内容 | 当前状态 |
| --- | --- |
| MotionBricks G1 参考演示 | 保留上游模型与 MuJoCo 预览流程，增加中文界面和 Windows 启动器；不等同于 UE 运行时效果 |
| UE 侧现状 | 仅保留 `unreal-sample/AIMotionSample` 实验宿主（GASP 示例工程，含 Content）与空壳 `MotionWeaver` 插件；旧的 79 骨 UE 导出插件与实验推理插件已于 2026-11 移除，理由见 [Unreal 集成](unreal-script/README.md) |
| 已记录的验证 | 程序生成动画夹具上的 CPU 小网络训练与 ONNX 数值对齐测试；UE 内实机闭环未验证 |
| 跑酷参考姿态实验 | 可在本地 HTML 手选关键帧，对比软条件重建与融合后硬约束；已验证单条低障碍动画的参考帧精确到位，仍有邻域连续性和姿态偏差限制 |
| SOMA 全库训练视图 | 已建成 71,127 条 fps120 四场景（A 27,182 / B 14,235 / C 2,621 / D 27,089），训练端可直接读取四键 NPZ |
| 分词器（VQ-VAE） | 已在全库上训练至 100,000 步，关节 RMSE 稳定在 84–86 cm；但码本在 20k 步后塌缩至 perplexity 1.000。**该路线已证否，不再是主线** |
| **参考引导姿态补全（主线）** | 全库 71,127 条 / 1,988,285 窗口训满 20 epoch，验证 **3.654 cm**；四档位置精度优于 Lerp 基线 **1.69–3.15 倍**；权重已封盘（详见 [训练与评测记录](docs/refpose-v3-training-evaluation.md)） |
| 接触（脚滑）约束 | 已接入训练；脚滑倍数由 17–22 降至 **6.9–8.7**（只统计纯生成帧对时 **6.6–7.3**），未达 &lt;5× 目标，余量交由 UE 端 Foot Locking / Two-Bone IK 补齐 |
| 训练数据格式 | 已改为可 mmap 的逐字段 npy；随机窗口读 3.75→1.33 ms，吞吐 322→1,829 样本/秒，逐元素与原数据一致 |
| 统一评测 | `benchmark/` 收敛原先散在三处的评测代码，以 `evaluate_all()` 为唯一入口，官方已有的指标原语直接复用 |
| 损失函数与评测口径 | 已修正一处使训练与评测喂不同归一化分布的口径缺陷（−11.3%）；关节位置损失经 17 组同步数对照判定无效应，默认关闭 |
| 正式 UE 运行时动画 | 资源生命周期、游戏工具函数、统一人形重定向、CMC / Mover、多人并发与联机接入均为后续工作 |

## 主线结果：参考引导姿态补全

任务：给定过去 24 帧历史、未来 24 帧 Root 轨迹与 0–2 帧硬参考，生成其余帧的全身关节姿态。
这是**补间生成**（in-betweening），不是重建——锚点以外的 22–24 帧全部由模型生成。

| 档位 | GT 帧 | 生成帧 | 模型 | Lerp 基线 | 优于基线 | 脚滑（全帧） | 脚滑（纯生成） |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_reference` | 0 | 24 | **5.28 cm** | 16.64 | **3.15×** | 6.9× | **6.9×** |
| `middle_reference` | 1 | 23 | **3.22 cm** | 7.24 | **2.25×** | 8.6× | **7.3×** |
| `end_reference` | 1 | 23 | **3.22 cm** | 7.46 | **2.32×** | 8.6× | **7.3×** |
| `middle_and_end` | 2 | 22 | **2.17 cm** | 3.66 | **1.69×** | 8.7× | **6.6×** |

- 指标：test split 257,171 窗口，每 clip 等权，只算非锚点帧的关节位置 RMSE
- 脚滑倍数 = 接触期脚速 ÷ 源动作接触期脚速（源基准 0.0261 m/s）；「纯生成」只保留相邻两帧都为生成帧的接触帧对
- 模型 3.09 M 参数，78 骨，隐层 256；训练全库 71,127 条 / 1,988,285 窗口，20 epoch，8.7 小时（单卡 RTX 4070 Laptop）

**已知限制**：脚滑未达 &lt;5× 目标；Root 轨迹取自源动作，是理想控制输入；所有指标为离线关节误差，**未做 UE 内实机验证**。

完整记录见 [docs/refpose-v3-training-evaluation.md](docs/refpose-v3-training-evaluation.md)。

## 文档

| 类别 | 位置 |
|---|---|
| 训练与评测记录 | [docs/refpose-v3-training-evaluation.md](docs/refpose-v3-training-evaluation.md) |
| 技术报告（论文） | [paper/main.html](paper/main.html)（HTML 工作稿，可导出 PDF） |
| 软件著作权材料 | [paper/软著/](paper/软著/)（能力清单、说明书、源程序、申请表） |
| 评测协议基线 | [docs/soma-inbetweening-baseline-full-v1.md](docs/soma-inbetweening-baseline-full-v1.md) |
| 开发环境与测试 | [docs/development.md](docs/development.md) |
| 架构与目录 | [docs/motionweaver-architecture.md](docs/motionweaver-architecture.md)、[docs/repository-layout.md](docs/repository-layout.md) |
| 代码整理计划 | [docs/refactor-plan.md](docs/refactor-plan.md) |

现有 G1 权重与参考骨架绑定。自定义 UE 训练入口不支持直接加载 G1 权重微调，原 G1 演示也不能直接预览新的 UE 骨架检查点。

## 快速开始

### 获取仓库

安装 Git LFS 后执行：

```bash
git lfs install
git clone https://github.com/Aceared0829/MotionWeaver.git
cd MotionWeaver
```

当前默认分支为 `main`。预训练检查点默认不自动下载；按下面的使用场景准备环境和模型。

### 开发环境

只改代码、跑测试时推荐用仓库 `.venv` 和统一开发入口：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python tools/dev.py setup
.venv/bin/python tools/dev.py doctor
.venv/bin/python -m pytest -q
```

Windows 将 `.venv/bin/python` 换成 `.venv\Scripts\python.exe`。`setup` 默认安装
`.[training,dev]`；另加 `--demo` / `--export` 可补 MuJoCo 与 ONNX 依赖。
仓库已带共享 VSCode 配置，打开后安装推荐扩展即可使用 Python/Pytest、Ruff、
调试任务和 UE clangd 索引。完整搭建、测试分类、Lint 策略和数据路径见
[开发环境](docs/development.md)。

### Unreal 侧范围

训练数据链路已不经过 UE：原始动作来自 `soma_uniform/bvh`，跨骨架转换由离线 Python 完成（`data/tools/ingest/retarget_uefn_to_soma.py`），训练契约是 78 骨 SOMA / 120 fps / 24+24 窗口。

UE 侧目前只保留 `unreal-sample/AIMotionSample` 实验宿主与空壳 `MotionWeaver` 插件，用来承接后续 refpose v3 的 AnimNode 接线。该宿主是 GASP 示例工程的副本，尚无自有 `Source/` 与 `Plugins/`。旧的 79 骨 `AILocomotionDataset` 导出插件、`AIAnimation` 实验推理插件、Motion Matching 桥接与旧宿主 `UMWSamplePreview` 均已于 2026-11 移除，理由见 [Unreal 集成](unreal-script/README.md)。

### 手选跑酷参考姿态

从仓库根目录运行 `./inference/profiling/Start-PoseReferenceLab.ps1`，在本地页面逐帧选姿态，再运行真实模型对照。需要本机 V2 检查点和准备好的原始数据；这些数据与权重不随本次源码提交发布。

[实验室操作、软/硬对照与限制](docs/experiments/2026-09-23-pose-reference-lab.md)记录了用户所选六帧的结果：平均关节误差 11.322→6.903 cm，参考帧通过后处理等式约束精确满足；加速度误差略升，不能宣称解决抽搐。编码器仍读取完整已知窗口，代理模式还读取已知未来；尚非稀疏参考生成未知过渡，也未接入 UE 跑酷执行逻辑。

### MotionBricks 参考演示

该演示使用 G1 参考骨架、MuJoCo 和预训练权重。按现有演示环境说明准备 Python 3.10、支持 CUDA 的显卡及依赖，在仓库根目录执行：

```bash
git lfs pull --include="model-weight/base/motionbricks/**" --exclude=""
git lfs pull --include="inference/assets/skeletons/g1/meshes/**" --exclude=""
# 以下命令在仓库根目录执行
conda create -n motionbricks python=3.10 -y
conda activate motionbricks
pip install -e ".[training,demo]"
python inference/cli/interactive_demo_g1.py
```

Windows 下默认打开中文单窗口界面，其他平台在不能嵌入 MuJoCo 窗口时回退为两个窗口。详见 [演示说明](docs/motionbricks/README.md) 和 [中文交互界面](docs/motionbricks/chinese_interface.md)。

仓库根目录的 `MotionBricks.exe` 是该参考演示的 Windows 启动器，固定读取根目录 `.venv` 中的环境；上面的 Conda 环境不会自动供 EXE 使用。它不包含 Python、模型和依赖，也不是 UE 插件启动器。

## 最近的实测结论

2026-10 的一轮实验给出两条结论，都记在 [技术报告 §9](paper/main.html) 与
[实验记录归档](docs/archive.md)：

1. **评测口径曾经是错的。** 训练与评测喂给模型的是两种不同的归一化分布
   （`--dataset` 落盘前已标准化，`--soma_views` 产出原始特征，而
   `is_normalized` 被写死成一个默认值）。后果不是「训练变慢」，而是
   训练目标与评测目标自始至终不是同一个量：30,000 步之后关节误差从 108 cm
   断崖到 310 cm。把口径对齐后，同一评测口径下关节 RMSE 由 94.54 cm 降至
   **83.85 cm（−11.3%）**，这是本项目至今最大的一次单点收益，且不来自任何损失项。
2. **把验收指标写进损失的尝试失败了。** 在 17 组同步数对照下，关节位置损失的
   效应为 **+0.10 cm**，小于同配置换种子造成的 0.56 cm 抖动（符号检验 p=0.75）。
   原因是码本在 20,000 步后塌缩至 `perplexity = 1.000`，每批所有 token 量化到
   同一码字，分词器退化为自编码器——码字里没有可供优化的信息。

下一步的主线因此不是继续调损失，而是解决码本塌缩。验收判据写在
[技术报告 §11](paper/main.html)：`perplexity_pose` 稳定 > 8，且关节 RMSE 不退化。

## 开发方向

第一阶段由 CMC / Mover 负责移动模拟，模型根据实际运动状态与未来轨迹生成角色姿态，再适配目标骨架：

```text
训练：UE 动画资产 → 采样与数据转换 → 模型训练 → 模型包

运行时目标：玩家 / AI 输入 → CMC 或 Mover → 运动状态与轨迹条件
                                                   ↓
                                      动画模型 → 骨架适配 → UE 姿态
```

运行时部分为开发目标。多人联机、统一人形模型、重定向与接触修正的职责和验收要求见 [SOMA 数据与双路径推理](docs/soma-training-data-direction.md)。

## 仓库与文档

已按数据、训练、模型、推理职责重组。模型源码位于 `model/MotionWeaver/`，Python 包名为 `motionweaver`；数据、训练和推理分别使用 `data.runtime`、`training`、`inference`。安装入口为仓库根目录，旧检查点导入集中兼容；已有数据和权重保留。目录与测试说明见 [仓库结构](docs/repository-layout.md)。

```text
data/               runtime/ · tools/ · prepared/（本地）
training/           pretrain/ · posttrain/ · common/ · models/ · evaluation/ · configs/ · runs/（本地）
model/MotionWeaver/ 网络、几何、动作表示与共享实现
model-weight/       base/motionbricks/ · 自定义发布产物
inference/          runtime/ · export/ · profiling/ · cli/ · demo/ · assets/
unreal-script/      MotionWeaver 插件镜像与同步工具（其余 UE 插件 2026-11 移除）
unreal-sample/      UE 推理/动画工具链实验宿主源码与资产交付边界（Content 暂缓）
benchmark/          统一评测：recon 库、CLI 与四臂对比（唯一评测入口）
paper/              技术报告（main.html → PDF）与软著材料
tests/              单元、训练、导出与目录回归测试
tools/compat/       历史命令与检查点导入兼容源码
.build/             python/ 安装元数据 · experiments/ 本地实验输出
```

| 位置 | 用途 |
| --- | --- |
| [unreal-sample](unreal-sample/README.md) | UE 5.8 实验宿主 `AIMotionSample` 的边界；当前尚未接入模型推理 |
| [unreal-script](unreal-script/README.md) | UE 插件安装、动画导出、数据约定与训练说明 |
| [模型与参考演示说明](docs/motionbricks/README.md) | 上游模型技术说明与 G1 参考演示 |
| [动作表示](docs/motionbricks/motion_representation.md) | 骨架、Root Motion、姿态特征和坐标约定 |
| [接入自有数据集](docs/motionbricks/adding_your_own_dataset.md) | MotionBricks 通用数据集接入；UE 资产优先使用上面的插件说明 |
| [学习与项目历史](docs/project-background.md) | 源码阅读顺序、游戏动画裁剪范围与上游关系 |
| [实验记录](docs/experiments/README.md) | 可提交的逐次数据、训练、评估与 UE 验证记录及证据索引 |
| [技术报告](paper/main.html) | 完整技术报告源码；PDF 见 `paper/MotionWeaver-technical-report.pdf`（31 页） |
| [软著材料](paper/软著/README.md) | 软件著作权登记材料：说明书、源程序文档、能力清单与一致性检查 |
| [关节误差与方案 A 结论（归档）](docs/archive.md) | 归一化口径缺陷的定位与修复、关节位置损失的 17 组对照、码本塌缩证据 |
| [分词器评测（归档）](docs/archive.md) | 统一 benchmark 的四臂结果与口径边界 |
| [权重清单](docs/assets/plan-a-final/WEIGHTS_MANIFEST.md) | 长期保留的 25 个 run 的配置、步数、最优指标与位置 |
| [launcher](launcher) | Windows 参考演示启动器源码 |

## 来源与许可

- **上游来源：** NVIDIA [GR00T-WholeBodyControl / MotionBricks](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks)。原作者、版权及论文信息予以保留；上游展示不能作为本项目新增 UE 功能的验证结果。
- **源代码：** 采用 Apache License 2.0，保留适用的版权与变更说明。
- **NVIDIA 预训练权重：** 单独适用 NVIDIA Open Model License，不能按源代码的 Apache 2.0 授权理解。
- **第三方资源与动画数据：** 依照各自授权使用；本仓库的代码许可证不授予外部动画资产、网格或数据集的使用权。

Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

完整来源、许可文本入口、第三方声明和论文引用见 [引用与许可说明](引用与许可说明.md)、[NOTICE](NOTICE) 与 [LICENSE](LICENSE)。

---

本文件由 AIAnimationSystem 维护者基于上游 README 改写；2026-09-07 更新项目命名、
使用入口、开发状态及来源声明；2026-10-28 补充 SOMA 全库训练视图、统一评测、
分词器现状与最近的实测结论；2026-11-14 补充开发环境、统一入口与测试说明。
