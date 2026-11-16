# 实验记录

这里保存 AIAnimationSystem 数据、训练、评估、推理和 UE 验证实验的 **Git 可追溯摘要**。一次实验一份 Markdown，包含实际配置、数据身份、对照与评价口径、结果、失败和原始证据位置。新记录复制 [_template.md](_template.md)；执行细则见项目 skill [record-research-experiments](../../.agents/skills/record-research-experiments/SKILL.md)。

记录文件命名为 `YYYY-MM-DD-short-slug.md`，日期采用实验开始的本地日期。状态用“进行中 / 完成 / 失败 / 中止”，并在续跑或复核时更新修订历史。可分享的小型配置、指标报告、图表和审核导出随记录放在 `assets/<实验 ID>/` 并提交。训练权重、原始数据和大体积输出留在项目既有本地目录；记录应写出稳定标识、可访问的归档位置或重建方法及可用的哈希。路径只在作者机器上有效时须明确说明。

## 索引

已有历史结果仍保留在 [Root 驱动与硬参考姿态数据契约](../reference-guided-data-contract-v1.md) 和 [跑酷参考姿态实验](2026-09-23-pose-reference-lab.md) 中；旧 UE 插件说明已随对应目录移除，见 [仓库目录](../repository-layout.md)。它们不因本索引建立而变成统一口径的对照实验。

新记录在这里新增一行，注明日期、类型、状态和主要证据：

| 日期 | 实验 | 类型 | 状态 | 主要证据 |
| 2026-09-23 | [参考姿态实验审查与验证](2026-09-23-pose-reference-review.md) | 参考姿态 / 审查 | 完成；修复路径、绘图与浏览器存储退化 | 审查项、复测结论与限制 |
| 2026-09-23 | [手选跑酷参考与已知代理动画对照](2026-09-23-proxy-reference-results.md) | 参考姿态 / 手选参考 | 完成；案例研究，不作为未见测试集结论 | 四条动作的参考帧与逐窗口记录 |
| 2026-09-23 | [跑酷阶段参考与条件遵守审计](2026-09-23-phase-reference-results.md) | 参考姿态 / 阶段候选 | 完成；中间阶段候选未确认真值 | 13 条跑酷的候选阶段与条件对照 |
| 2026-09-23 | [手选参考姿态实验室](2026-09-23-pose-reference-lab.md) | 参考姿态 / 硬约束对照 | 完成；离线投影，非因果实时能力 | 六参考帧软/硬误差对照与限制 |
| 2026-11-14 | [refpose 接触损失的帧率缺陷与接触权重性价比](2026-11-14-refpose-contact-fps.md) | 缺陷修复 / 训练 / 离线评估 | 完成；帧率写死 30 致 contact_weight=0.4 实际只生效 1/16；修好后 0.025 与旧轮等价（+0.02–0.07 cm），0.1 让四档 RMSE 涨 22–36% 只换脚滑降 5–11%，0.4 发散 | 数值等价验证（相对差 0.000e+00）、三组强度对照（w0.4 提前中止）、验证曲线与脚滑双口径 |
| 2026-11-14 | [refpose v2 从 20 epoch 延到 30 epoch](2026-11-14-refpose-v2-30ep.md) | 续训 / 离线评估 | 完成；test 四档 5.40/3.39/2.94/2.38 cm（较 20ep 各降 0.08–0.19），10 个 epoch 只换 0.14 cm 验证收益 | 同口径单变量对照、逐 epoch LR 与 RMSE、续训 T_max 与 LR 衔接证据 |
| 2026-11-14 | [refpose v2 全量 20 epoch 训练](2026-11-14-refpose-v2-20ep.md) | 训练 / 离线评估 | 完成；test 四档 5.59/3.50/3.04/2.46 cm，录得两次中断与续训根因 | 检查点 sha256、`comparison.md` 四档对照、验证 RMSE 20 点曲线 |
| 2026-10-18 | [SOMA 全库处理速度优化](2026-10-18-soma-processing-speed.md) | I/O / FK / 等价续跑 | 8 条对照通过；优化版本续跑全库 | 两轮同样本 8 进程计时、24 次文件与数组等价核对 |
| 2026-10-18 | [SOMA 非镜像全库处理启动](2026-10-18-soma-full-processing.md) | 固定参考 / Root / 候选 / 持久批次 | 已启动 71,132 条，尚未全量完成 | 四轮技术试点、续跑回读、E 盘逐条日志与进度 |
| 2026-10-18 | [UE → 固定 SOMA 重定向](2026-10-18-uefn-soma-retarget.md) | 动画转换 / BVH / 同步审核 | 1,969 条转换完成；1 条重试后待复核 | 全批数值与文件清单汇总、独立待复核页面、试点和对照截图 |
| 2026-10-17 | [SOMA 原生数据与轻量处理](2026-10-17-soma-processing.md) | 导入 / Root / 接触候选 / 审核 | 30 条技术试点完成；未扩量 | 同日 JSON 回执、HTTP 全量回读、处理审核截图 |
| --- | --- | --- | --- | --- |
| 2026-10-04 | [10 条拒绝动作的完整片段重定向修复](2026-10-04-seed-review-repair.md) | UE 重定向 / 完整片段 / 二审 | 10 条重跑完成，2 条疑似穿地，待二审 | 完整 5541 帧预览、逐条回执、初筛与 25 项检查 |
| 2026-10-04 | [E 盘持续处理与异常二审](2026-10-04-continuous-secondary-review.md) | 数据处理 / 人工二审 | 二审服务已启动；持续处理启动被自动审批拒绝 | 19 项检查、[运行配置](assets/2026-10-04-continuous-secondary-review/startup.json)、本机服务日志 |
| 2026-10-03 | [UMWSamplePreview 项目迁移与 CR 回归](2026-10-03-umw-project-migration.md) | UE 迁移 / 默认场景与工具回归 | 本页范围完成；未接模型或打包 | [前后对照与最终验证摘要](assets/2026-10-03-umw-project-migration/verification.json)；原始日志和 Content 留本机 |
| 2026-09-25 | [P0 站蹲数据与审核基线审计](2026-09-25-p0-stance-data-audit.md) | 数据审计 / 离线视觉复核 | 完成；107 条按用户决定全部通过 | 4 条逐帧图、窗口跨度、13 组家族表、[107 条确认清单](assets/2026-09-25-p0-stance-data-audit/restart_2026-09-25/review_acceptance.csv) |
| 2026-09-28 | [站蹲早期过渡与状态条件试验](2026-09-28-stance-transition-v2.md) | 数据索引 / 云端训练 / 本地推理评估 | B0、B1 试验完成；UE 玩家闭环未实现 | [B0](assets/2026-09-28-stance-transition-v2/b0_test.json)、[B1 冻结测试](assets/2026-09-28-stance-transition-v2/b1_test.json)、[全部窗口](assets/2026-09-28-stance-transition-v2/b1_test_all.json) |
| 2026-09-28 | [UE 单机 CMC 站蹲与执行 Root 观测演示](2026-09-28-ue-cmc-preview.md) | UE 功能/碰撞试验 | 自动场景通过；未接入模型姿态 | [119 条观测 CSV](assets/2026-09-28-ue-cmc-preview/cmc_preview_20260928T110529Z.csv)、[摘要](assets/2026-09-28-ue-cmc-preview/summary.json)、本机截图 |
| 2026-09-28 | [UE CMC + B1 本地在线姿态预览](2026-09-28-ue-b1-online-preview.md) | ONNX 导出 / UE 本地推理 | 接线通过；生成历史回灌质量未通过 | [源历史 40 行](assets/2026-09-28-ue-b1-online-preview/source_history.csv)、[回灌 40 行](assets/2026-09-28-ue-b1-online-preview/generated_feedback.csv)、本机截图 |
| 2026-09-28 | [MotionWeaver Actor Root 条件训练与本地推理](2026-09-28-motionweaver-a800-actor-root.md) | A800 训练 / 本地推理 | 完成 20000 步；玩家 UE 质量门槛未通过 | [20k 验收汇总](assets/2026-09-28-motionweaver-a800-actor-root/motionweaver_20k_acceptance_summary_20260928.json)、[CMC 逐条报告](assets/2026-09-28-motionweaver-a800-actor-root/motionweaver_cmc_report.json)、四段离线骨架视频与蹲伏失败帧；原始输出留本地归档 |
| 2026-09-25 | [Closed-loop 优化](2026-09-25-closed-loop-optimization.md) | CMC Root / 接触消融 | 完成；保留失败与对照 | CMC Root、接触和两步预测指标资产 |
| 2026-09-25 | [接触速度审计](2026-09-25-contact-speed-audit.md) | 数据审计 / 接触评价 | 完成 | 速度窗口、验证和审计脚本 |
| 2026-09-25 | [已有动作诊断](2026-09-25-existing-motion-diagnostics.md) | 动作质量诊断 | 完成 | Root、接触和来源动作报告 |
| 2026-09-25 | [Root 姿态对齐](2026-09-25-root-pose-alignment.md) | UE / Root 对齐 | 完成 | 四路对齐和 UE 读回数据 |
| 2026-09-25 | [UE 蒙皮推理](2026-09-25-ue-skinned-inference.md) | UE 本地推理 | 完成；后续模型仍需玩家闭环验收 | UE 蒙皮推理指标和测试说明 |
| 2026-09-26 | [接触复核训练设计](2026-09-26-contact-review-training-design.md) | 训练设计 | 完成 | 接触审查与训练方案 |
| 2026-09-27 | [Contact V2 试验](2026-09-27-contact-v2-pilot.md) | 接触条件训练 | 完成 | UE 脚速与条件训练指标 |
| 2026-09-27 | [实时玩家探针](2026-09-27-live-player-probe.md) | UE 玩家输入 / Root | 完成；用于诊断 | 交互轨迹和意图对照 |
| 2026-09-27 | [Motion Matching 可运行预览](2026-09-27-motion-matching-plugin-playable.md) | UE 插件 / Motion Matching | 完成；最终联机未验收 | 可运行主机与插件记录 |
| 2026-09-27 | [预训练对齐计划](2026-09-27-pretraining-alignment-plan.md) | 训练计划 | 完成；作为历史方案保留 | 配置与对齐指标 |
