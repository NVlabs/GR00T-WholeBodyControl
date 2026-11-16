# 开发环境

本页是环境搭建、日常命令和排障的统一入口。目录职责见
[仓库结构](repository-layout.md)，训练与数据契约见各自 README。

## 支持范围

- Python `>= 3.10`（`setup.py` 的硬性要求）。当前仓库在 Python 3.14 上跑过
  单元测试，但 `torch.jit` 在 3.14+ 有上游限制；训练和 ONNX 导出建议
  3.10–3.13。
- 核心数据/训练代码跨平台；UE 启动器、MuJoCo 单窗口嵌入和部分批处理脚本
  仍是 Windows 优先。
- GPU 训练需要与驱动匹配的 PyTorch。`pip install -e .` 不会替你选择 CUDA
  版本，先按 PyTorch 官方说明安装目标构建。

## 从零搭建

Linux / macOS：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python tools/dev.py setup --demo --export
.venv/bin/python tools/dev.py doctor
```

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe tools\dev.py setup --demo --export
.venv\Scripts\python.exe tools\dev.py doctor
```

`setup` 默认安装 `.[training,dev]`，即核心运行依赖、训练依赖、pytest、ruff
和 pre-commit。`--demo` 增加 MuJoCo / G1 演示依赖，`--export` 增加
ONNX / ONNX Runtime。只想改纯 Python、不跑训练时用 `--minimal`。

已有 `.venv` 且 CUDA PyTorch 已装好时，不要重建环境，直接补 editable 包：

```bash
.venv/bin/python -m pip install -e ".[training,dev]" --no-deps
```

`--no-deps` 只更新本仓库包，不碰现有 CUDA 构建；随后用 `doctor` 检查缺项。

如果当前机器已有全局 PyTorch，可让 bootstrap 复用系统包：

```bash
python tools/dev.py bootstrap --system-site-packages
```

## 统一入口

`tools/dev.py` 在 `.venv` 存在时会自动使用它，所以即使忘记激活，也不会
误用系统 Python：

| 命令 | 用途 |
|---|---|
| `python tools/dev.py doctor` | 检查解释器、依赖、CUDA、数据根、权重与 LFS |
| `python tools/dev.py setup --demo --export` | 在当前解释器安装 editable 包 |
| `python tools/dev.py bootstrap` | 创建 `.venv` 并安装基础开发依赖 |
| `python tools/dev.py test -q` | 完整 pytest |
| `python tools/dev.py test tests/test_train_resume.py -q` | 单文件测试 |
| `python tools/dev.py lint` | 项目固定的 ruff 规则 |
| `python tools/dev.py lint --fix` | 自动修复可安全修复的 lint 问题 |
| `python tools/dev.py check` | 先 lint，再跑完整测试 |

`doctor --strict` 会在必需依赖缺失时返回非零，适合脚本或 CI 使用；默认模式
只报告，不因为“没挂数据盘”而阻断纯代码开发。

## VSCode

仓库共享 `.vscode/` 下的项目配置，打开仓库后 VSCode 会提示安装推荐扩展：

- Python / Pylance / debugpy：解释器、补全与调试；
- Ruff：使用 `pyproject.toml` 固定的 lint 规则；
- EditorConfig：遵循 `.editorconfig` 的缩进与换行；
- clangd：解析 `unreal-script/` 与 `unreal-sample/` 的 UE C++；
- TOML / YAML / GitHub Actions：配置与 CI 文件的基础支持。

项目配置默认：

- 解释器默认指向 `${workspaceFolder}/.venv/bin/python`；Windows 上让 Python
  扩展通过 `python.venvPath` 自动发现 `.venv\Scripts\python.exe`，或在
  User Settings / 命令面板手动选择，不要改共享的 `settings.json`；
- Test Explorer 使用 pytest，测试目录为 `tests/`；
- 保存时不自动格式化、不自动整理 import，避免把历史格式差异混进功能改动；
- `Dev: doctor`、`Dev: test`、`Dev: lint`、`Dev: check` 等任务可直接从
  `Terminal → Run Task` 运行；
- `UE: generate clangd compile database` 调用
  `unreal-script/gen_compile_commands.py` 生成 clangd 索引；
- `UE: sync plugin mirror` / `UE: check plugin mirror` 调用插件同步脚本；
  这两个任务需要 PowerShell 7（`pwsh`），Linux 未安装 `pwsh` 时会失败；
- 在 Markdown 文件中输入 `mwexp` 可插入实验记录骨架。

共享文件只有 `settings.json`、`tasks.json`、`launch.json`、`extensions.json`
和 `experiments.code-snippets`；其他 `.vscode` 本地文件仍被忽略。

## 测试

pytest 是唯一总入口：

```bash
.venv/bin/python -m pytest -q
```

Windows 使用 `.venv\Scripts\python.exe -m pytest -q`。`pyproject.toml` 的
`testpaths` 为 `tests/`，并设置 `pythonpath = ["."]`。

不要再把 `unittest discover -s tests` 当作全量测试：仓库里已有 pytest
fixture / parametrize 用例，unittest discover 会静默漏掉一部分。

测试大致分四类：

- **纯代码/契约测试**：默认应全绿，不依赖本机数据。
- **临时夹具测试**：在 `tmp_path` 里造数据，可离线跑。
- **数据/权重测试**：缺少 `MW_DATA_ROOT` 下的封盘数据时 skip，例如
  `test_manifest_rebuild.py`、`test_onnx_export.py`。
- **GPU/网络测试**：CUDA 不可用或本机 socket 被限制时 skip/fail；在沙箱
  Agent 里跑全量测试可能被拒绝 AF_UNIX/socket，这不等于代码回归。

常用筛选：

```bash
.venv/bin/python -m pytest tests/test_train_resume.py -q
.venv/bin/python -m pytest -q -k "not onnx and not mujoco"
.venv/bin/python -m pytest -q --maxfail=1
```

## Lint 与格式

`pyproject.toml` 显式固定了最小正确性规则：

```text
E9  语法/IO 级别错误
F63 比较字面量时误用 is
F7  语法相关错误
F82 未定义名称
B018 裸表达式（死语句粗筛）
```

这不是“全仓已经符合 PEP 8”的声明。历史代码里有大量 `E701`/`E702`/`I001`，
因此不要在没有单独变更说明的情况下全仓 `ruff format` 或
`ruff check --fix --unsafe-fixes`，否则 diff 会淹没真实改动。

可选安装 pre-commit：

```bash
.venv/bin/python -m pre_commit install
```

它只检查 staged Python 文件，并跑三个快速仓库守卫测试。

## 数据与权重

仓库不含大数据、训练视图和大权重。路径覆盖约定：

| 变量 | 含义 | 默认 |
|---|---|---|
| `MW_DATA_ROOT` | 数据根 | `D:/MotionWeaverData`（历史 Windows 位置） |
| `MW_VIEWS` | 训练视图 | `<数据根>/soma_train_v1/full-v1-f32` |
| `MW_RELEASE` | 契约发布目录 | `<数据根>/full_release` |
| `MW_MMAP` | mmap 训练集 | `<数据根>/full_mmap` |
| `MW_RUNS` | 训练产物 | `<数据根>/training_runs` |
| `MW_REPORTS` | 评测报告 | `<数据根>/reports` |
| `AIANIMATION_ROOT` | 完整仓库检出 | 当前包所在仓库根 |

相对路径会按仓库根解析。缺数据时数据相关测试 skip 是正常行为，不要为了让
测试变绿而删除 skip 或放宽契约校验。

大文件由 Git LFS 管理：

```bash
git lfs install
git lfs pull
```

## CI 范围

`.github/workflows/checks.yml` 只跑：

- 固定版本的 ruff；
- 维护目录的 byte-compile；
- 三个不依赖 torch / 数据的仓库守卫测试（路径、死语句、仓库布局）。

完整 pytest、ONNX 数值对齐、MuJoCo 演示和 UE 侧验证依赖本机数据、GPU 或
编辑器，仍按各文档在本机执行。

## 常见问题

**`No module named pytest` / `ruff`**
说明用了系统 Python。改用 `.venv/bin/python`（Windows
`.venv\Scripts\python.exe`），或直接运行 `tools/dev.py`，它会优先选 `.venv`。

**测试因 socket / AF_UNIX 报 `PermissionError`**
当前进程被沙箱限制了本机网络或多进程。把测试放到普通终端运行；这不是
`pose_reference_lab` 或 `build_refpose_release` 的代码错误。

**数据相关测试全是 skip**
检查 `MW_DATA_ROOT` 是否指向真实数据根，并运行 `tools/dev.py doctor`。

**CUDA 不可用**
先确认 `torch.cuda.is_available()`，再检查驱动与 PyTorch 构建是否匹配。
CPU 环境仍可跑绝大多数单元测试和小网络冒烟。

**Git LFS 文件是几十字节的文本**
那是 LFS 指针。运行 `git lfs pull`；不要把它当损坏权重覆盖掉。

**pip 报缓存目录不可写**
把缓存指到可写目录再装，例如：

```bash
PIP_CACHE_DIR=/tmp/pip-cache .venv/bin/python -m pip install -e ".[dev]"
```

完全离线且环境里已有 `setuptools` 时，可追加 `--no-build-isolation`。
