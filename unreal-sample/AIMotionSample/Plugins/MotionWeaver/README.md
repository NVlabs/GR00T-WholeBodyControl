# MotionWeaver 插件

UE 5.8 动画工具链插件。当前提供两条链路：

1. **姿态代理**：从动画预览的暂停帧抽出全身姿态，存成独立 `.uasset`
2. **姿态代理组**：把多个姿态按帧号串成一段运动，支持求解轨迹与视口拖动编辑；
   视口里所有关键帧同屏显示，选中帧实体高亮，其余帧以线框幽灵呈现

不接推理、不做骨架转换、不碰 AnimGraph。

本文件说明源码怎么分目录、模块怎么依赖，以及新功能该放哪。

## 模块与分层

三个模块，按「能不能打包」切开：

| 模块 | Type | 说明 |
|---|---|---|
| `MotionWeaver` | Runtime | 打包后也要活着：数据结构、求解算法、关卡 Actor |
| `MotionWeaverEditor` | UncookedOnly | 编辑能力：资产编辑器、视口、Gizmo、抽取 |
| `MotionWeaverTests` | Editor | 自动化测试。单独成模块是为了不编进发行包 |

Runtime 模块内部按 **四层** 组织，依赖只能从上往下：

```text
① Pose/Data/      纯数据，零依赖
       ↓
② Pose/Solve/     求解算法，只依赖 Data
       ↓
③ Pose/Runtime/   关卡里的 Actor，依赖 Data + Solve
```

Editor 模块横向按职责分：

```text
MotionWeaverEditor/
├── Pose/Assets/Factory/       新建资产入口
├── Pose/Assets/TypeActions/   右键菜单与双击打开
├── Pose/Assets/Editors/       编辑器本体
├── Pose/Elements/             Typed Element 接入（Gizmo 靠它）
├── Pose/Extraction/           从动画抽帧 + 命名规则
├── Pose/Viewport/             预览视口、时间轴、视口客户端
└── Pose/PoseGroupEditMode     编辑模式（ITF 手柄需要）
```

## 完整目录

```text
Source/
├── MotionWeaver/                      # Runtime
│   ├── Public/
│   │   ├── Core/MotionWeaverLog.h
│   │   └── Pose/
│   │       ├── Data/                  # ① 数据层
│   │       │   ├── PoseTypes.h             快照结构（USTRUCT）
│   │       │   ├── PoseProxyAsset.h        单帧姿态资产
│   │       │   ├── PoseGroupProxyAsset.h   姿态组资产
│   │       │   └── PoseKeyElementData.h    元素数据（弱引用 + KeyIndex）
│   │       ├── Solve/                 # ② 求解层
│   │       │   ├── PoseChain.h             求解器
│   │       │   ├── PoseChainTypes.h        关键帧 · 设置 · 结果
│   │       │   └── PoseChainDebugComponent.h  骨架线框显示
│   │       ├── Runtime/               # ③ 表现层
│   │       │   ├── PoseActor.h             单帧姿态的关卡载体
│   │       │   ├── PoseComponent.h         显示组件
│   │       │   └── PoseGroupActor.h        姿态组的关卡载体（只显示）
│   │       └── Elements/PoseKeyElementRegistry.h   元素类型注册
│   └── Private/                       # 与 Public/ 同构
│
├── MotionWeaverEditor/                # UncookedOnly
│   ├── Public/ 与 Private/ 同构
│   └── Pose/
│       ├── Assets/
│       │   ├── Factory/               Factory + TypeActions + Editors 三种职责
│       │   ├── TypeActions/
│       │   └── Editors/
│       ├── Elements/                  Typed Element 的编辑器半边
│       ├── Extraction/                抽取 + PoseAssetNaming
│       ├── Viewport/                  视口 · 时间轴 · Gizmo 客户端
│       └── PoseGroupEditMode.h        ITF 编辑模式
│
└── MotionWeaverTests/                 # Editor 类型，不进发行包
    └── Private/PoseChainTests.cpp     16 项求解器测试
```

## 数据流

```text
Extraction 读预览组件
    → 写进 UPoseProxyAsset.Snapshot（PoseTypes 定义结构）
    → FPoseKey 引用它，组成 UPoseGroupProxyAsset.Keys
    → FPoseChainSolver 求解出轨迹与姿态
    → 关卡里由 APoseGroupActor 显示（只读）
    → 资产编辑器里由 SPoseGroupViewport 显示并编辑
      （每个关键帧一个预览组件，选中帧高亮，其余帧线框幽灵）
```

## 拖动链路（改这块之前务必先读）

实测确认的调用顺序，与「用 Typed Element 就该走 SetWorldTransform」的直觉不同：

```text
Gizmo
  → FPoseGroupEditMode::InputDelta
  → ApplyDeltaToComponent
  → OnKeyDragged
  → ApplyDraggedKey      ← 数据在这里落盘（在 PoseGroupProxyAssetEditor_Drag.cpp）
  → FinishDrag           ← 松手才求解
```

**不走** `ITypedElementWorldInterface::SetWorldTransform`。ITF 只提供手柄的外观与命中，
数据写入仍由 legacy EdMode 负责。所以事务、写回、刷新都挂在 `ApplyDraggedKey` 这条线上。

## 加新东西时放哪

- **运行时也要用的新能力**：`Source/MotionWeaver/Public/Pose/<层>/`，按数据/求解/表现归类。
  一条能力一个目录，不要平铺。
- **只在编辑器里用的能力**：`Source/MotionWeaverEditor/Pose/<能力>/`。
- **新测试**：加进 `MotionWeaverTests`，不要放回 Runtime 模块。

## 约定

- `Public/` 与 `Private/` 的子目录结构保持一致，找 `X.cpp` 时它的头文件永远在镜像路径上。
- `UCLASS()` 里的 `*.generated.h` 永远短名，不跟目录。
- Runtime 模块不能依赖 `UnrealEd`；需要 `GEditor` 的代码放 Editor 模块。
- 数据结构定义在 Runtime，Editor 只 `#include` 它，不能反向。
- 新增模块要改 `.uplugin`、各模块 `Build.cs`，代价大；只有「打包体积」或「依赖隔离」
  真的出问题才拆。这次拆 `MotionWeaverTests` 就是因为测试不该编进发行包。

## 已知限制

- 姿态组的命名规范尚未制定（它按运动类型命名，与单帧姿态不是同一套规则）。
- 导出给模型时竖直分量（Z）如何处理仍是未定项。
- Gizmo 只开放平移与旋转，不开放缩放（求解不吃缩放）。
- 视口每个关键帧创建一个蒙皮组件；帧数很多时绘制与内存开销线性增长，
  后续可按需改成只显示相邻帧或退回火柴人。
