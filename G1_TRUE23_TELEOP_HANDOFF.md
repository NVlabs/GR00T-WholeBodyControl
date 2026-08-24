# G1 rev-1.0 (23-DoF) SONIC teleoperation — working handoff

## 2026-08-09 PPO update-delta 2x2 attribution (current controlling status)

- The diagnostic-only four-policy ablation completed without training. Its
  immutable report is
  `/root/g1_true23_runs/sonic_task_space_ppo_full_support_update_delta_ablation_v1.json`,
  SHA `a44fcf7ca73e8bd03e99cb43568659c4a507f2327bdacf1924960f12dc47c1af`.
  It binds provenance snapshot
  `4c16c4dd8716a1c19921a0d1648e8b96389159c558b1f85818ba1935c210661c`
  and exactly reconstructs the baseline, hidden-block-only, head-only, and
  full-update policies without tensor aliasing.
- Baseline completes `155` transitions through q9 `163`. Updating only the
  last hidden block completes `152` through q9 `160`; updating only the final
  head also completes `152` through q9 `160`; the full update completes `151`
  through q9 `159`. Every policy terminates on right-wrist `ee_body_pos`.
- On the shared nonterminal q9 `9..158` window, total weighted reward is
  `4.3973 / 2.2480 / 2.2115 / -2.9598` for baseline/block/head/full. The block
  effect is `-2.1493`, the head effect is `-2.1858`, and their interaction is
  another `-3.0220`. Right-wrist barrier alone changes from `-0.2727` to
  `-1.3279 / -1.3741 / -5.4887`; its interaction is `-3.0595`.
- Therefore neither trainable submodule contains a usable update by itself.
  Freeze-one-module continuation is rejected, model 1 remains rejected, and
  no candidate is selected. Whole-episode returns remain diagnostic-only.
- Next evidence may only test update direction and magnitude without Adam: an
  exact symmetric interpolation of the full actor delta at alpha
  `-0.25, 0, 0.25, 0.5, 1`, with fresh deterministic environments, endpoint
  reproduction, q9-aligned reward/wrist attribution, and no candidate claim.
  No more PPO is permitted until that test separates excessive step size from
  a harmful advantage/gradient direction.
- No hardware, DDS, ZMQ, gantry, deployment, or robot actuation is authorized.

## 2026-08-09 full-support PPO one-update result

- The temporal-support experiment executed exactly once and is a valid negative
  result. Its immutable run is
  `/root/g1_true23_runs/sonic_task_space_ppo_full_support_v1_seed20260805`;
  `full_support_result.json` SHA is
  `bb0ff6d4e6517174b0cd0a51eab60f958ca4bc6cec5cfbe74ce9e672a4bef30a`.
- Before Adam, it collected exactly `128 x 160 = 20,480` transitions. The
  first-episode-only evidence contains `18,152` samples and excludes `2,328`
  autoreset samples from the gate. Survivor counts are `128/96/94/76/74/73`
  at q9 `88/126/133/154/163/168`; `56` first episodes terminate on
  `ee_body_pos`. Raw clipping, nonfinite values, action-semantic mismatch,
  q9 discontinuity, and anchor termination counts are all zero. Rollout
  evidence SHA is
  `66b72261e160df34cd9defca27eb229068b0efce0b1b27f7183669f2f81e9bc6`.
- The coverage/state/RNG/reward/memory gate passed, then exactly one stock PPO
  update ran: `8` Adam steps and no more. Update-1 checkpoint SHA is
  `10ae7bb7b67133c9b1c1330ea4093715f63531636291902d4b7212573d43d959`.
  Frozen actor tensors and fixed std remain exact; only the declared last
  hidden block and output head change. Twelve Adam states all report step `8`.
- Deterministic update 0 reproduces `155` transitions through q9 `163`, ending
  on `ee_body_pos`. Update 1 reaches only `151` transitions through q9 `159`,
  also on `ee_body_pos`. Its return is less negative, but the task cliff moves
  four frames earlier; candidate selection therefore correctly returns null.
  Value loss is `362.304`, while the actor delta is only `0.0617%` relative.
- The paired model-0/model-1 trace is now immutable at
  `/root/g1_true23_runs/sonic_task_space_ppo_full_support_model0_vs_model1_trace_v1.json`,
  SHA `0fb71869eb1249958f7f4137df52b9b0bd92b4666d07c74a614ee11622a878d3`.
  It exactly reproduces q9 `163/159`, both on right-wrist `ee_body_pos`, and
  rebuilds every reward/action/EE/contact identity. Historical absolute return
  remains diagnostic-only because MuJoCo-Warp replay is context-sensitive.
- On the equal preterminal q9 `9..158` window, model 1 loses `7.2195` reward.
  The largest losses are right-wrist barrier `-4.9980`, worst-EE `-0.9945`, and
  evaluator-aligned recovery `-0.9020`. Thus the earlier termination, not
  better control, caused the apparently less-negative whole-episode return.
  Raw-action divergence crosses `0.25` at q9 `132` and `0.5` at q9 `136`;
  a new `107.27 N` landing event appears at q9 `138`; contact state diverges at
  q9 `156/158`; model 1 then reaches right-wrist error `0.258851 m` at q9 `159`
  versus model 0 at `0.199231 m` on the same frame.
- Temporal coverage is no longer the missing input. Do not continue model 1 or
  add another PPO update blindly. The next allowed evidence must first isolate
  whether the declared last hidden block or final head caused this gradient
  response; it cannot train, select, promote, or deploy a policy.
- No hardware, DDS, ZMQ, gantry, deployment, or robot actuation is authorized.

## 2026-08-09 task-space PPO pilot and causal trace

- The first bounded genuine-SONIC task-space PPO pilot is valid negative
  evidence, not a candidate. The immutable v2 run is
  `/root/g1_true23_runs/sonic_task_space_ppo_seed20260805_v2`; its
  `pilot_result.json` SHA is
  `f78ee6758fd7e947339422a51e1ebb85fe0de5200d9b75bd04ac962847d74d0e`.
  Update 0 reproduces the original overlay policy at `155` transitions through
  q9 `163`, ending on `ee_body_pos`. Five PPO updates execute exactly `10,240`
  training transitions and `40` Adam steps, but update 5 regresses to `125`
  transitions through q9 `133`, also on `ee_body_pos`. No candidate is selected.
- The update-5 checkpoint itself is internally valid, not corrupted: file SHA
  `99b9a19d26f9a1ae9d685bde73faaf2842983b64551d447c65879ba97071cdbd`,
  policy-state SHA
  `fd0ea210bec6c403eb18fa0466ae11030b075ba1a5aa028cef94a458905c6f18`.
  Frozen actor tensors remain exact; only the declared final hidden block and
  output head change. The failure is therefore policy behavior, not lineage,
  checkpoint, action ABI, raw clipping, or V2 double application.
- The decisive paired trace is
  `/root/g1_true23_runs/sonic_task_space_ckpt0_vs_ckpt5_trace_v3.json`, SHA
  `608e126d61d14225149706d90a876445489982c7d88a49a02088d76f453fb22d`.
  It replays both immutable policies under the same probe, preserves exact
  policy/q9/termination identities, and reconstructs every reward term into
  each step and episode. Historical absolute return is diagnostic-only because
  MuJoCo-Warp CUDA is not cross-replay deterministic.
- Cause is now localized to temporal support. The five old 16-step updates see
  only q9 `9..88`. Inside that seen interval, update 5 improves summed reward by
  `0.0320`. Immediately outside it, raw-action L-infinity divergence first
  exceeds `0.25` at q9 `90`, `0.5` at q9 `106`, and `1.0` at q9 `111`.
  End-effector divergence reaches `0.05 m` and the right foot loses contact at
  q9 `115`; the right-wrist pre-threshold penalty starts at q9 `126`; the wrist
  reaches `0.255199 m` and terminates at q9 `133`. Over q9 `89..125`, update 5
  loses `3.2626` reward versus update 0, then loses another `7.9055` over q9
  `126..132` before the common `-100` death cost.
- Do not resume model 5, lower the learning rate on the same 16-step schedule,
  add teacher labels, or widen the network. The next isolated experiment starts
  from the exact update-0 actor and critic, collects one first-generation
  128-environment, 160-step rollout before any Adam step, proves late-q9 and
  wrist-barrier/death reward coverage, then permits exactly one PPO update and
  one deterministic evaluation. Any failed coverage/state/memory gate produces
  no update and no candidate.
- No hardware, DDS, ZMQ, gantry, deployment, or robot actuation is authorized.

## 2026-08-09 genuine SONIC closed-loop result (current controlling status)

- Genuine SONIC integration is now proved end to end in MJLab: tokenizer
  `267 ->` model-250 encoder `64`, concatenate with causal H10 proprio `930`
  to decoder `994`, emit raw native23, then apply V2 exactly once. The accepted
  offline-BC decoder is
  `artifacts/g1_true23/sonic_native124_21204_bc_last_affine_ridge_seed20260805_v2.decoder.onnx`,
  SHA `011740f86483323fc0f1c39ab25b784cf9411b401e56fee8b7a716664e921ee1`;
  manifest SHA
  `e43fb5e531fdccda46d2e28ce7a987c8d1d064e64ae58325e6caea4d758240db`.
  Encoder and decoder layers 0-7 remain frozen; only the final affine changed.
- The offline fit is not a generalization result. Exported training-row RMSE is
  `0.0707504`, but blocked out-of-fold RMSE is `0.209868`, including a per-DoF
  regression. The manifest permits only a closed-loop simulator experiment.
- Two pre-action harness failures are preserved, not overwritten: the first
  report SHA `4f7b638635d9ccc95a783adf8ea772f9a770587fc5364cefe093b850b5033bb4`
  exposed `510 * 0.02 -> ceil(...) = 511`; retry-1 SHA
  `0fd218ee40370887e445e1afbc65105c9dd1b3a2652a63a929f4c8605d3dd0f2`
  exposed TensorDict value iteration. Both executed zero actions. The horizon
  now resolves exactly to 510 and runtime observations use exact ordered keys.
- The first genuine student-controlled result is retry-2 report
  `artifacts/g1_true23/sonic_native124_21204_bc_last_affine_ridge_seed20260805_v2.initial510.student_closed_loop_qualification_retry2.json`,
  SHA `0d0142a53e9a4f54012ddac6eab6954ef96895d7148532dbd9386c4f549d0a1c`.
  It executes 155 consecutive student actions, q9 `9..163`, then terminates on
  `ee_body_pos`. The right-wrist vertical error is `0.267852 m` versus the
  `0.25 m` threshold. Minimum height is `0.656433 m`; maximum tilt is
  `0.639467 rad`.
- This is a policy-distribution failure, not an action ABI or V2 failure.
  Action-link mismatch, raw clipping, nonfinite, target/actuator/measured soft
  limits, and joint-velocity violations are all zero. Maximum raw magnitude is
  `2.84353`, maximum tracking RMSE `0.228326`, and maximum force ratio
  `0.528905`. At the same phase on the teacher trajectory the candidate still
  matches the teacher at `0.02731` RMSE, but the terminal student `policy930`
  has 257 coordinates outside the entire bootstrap range and cosine `0.374`
  to the phase-matched row. This is accumulated covariate shift.
- The v2 student is rejected for support, DAgger, promotion, and deployment.
  No student-state teacher labels have been admitted. More seeds, wider ridge,
  or full-decoder Adam on the same 510 rows are not the next step.
- The first mixed-controller recovery diagnostic is immutable at
  `artifacts/g1_true23/g1_true23_sonic_student_teacher_recovery_cutoff100_seed20260805_v1.json`,
  SHA `14b358ad26e8c5b007696b4e8a89fb25e56c8d1a12d7400f22166ea2d2fb16d3`.
  Student control is exact for q9 `9..108`; the selected PT/ONNX teacher then
  controls q9 `109..168` before `ee_body_pos` terminates the episode. Thus a
  60-step teacher suffix does not recover this late learner state. Teacher
  PT/ONNX error stays below `1.44e-6`; action/composite/state mismatches,
  clipping, nonfinite values, limit/velocity violations, and hard-safety
  violations are zero. No labels or training arrays were published.
- Cutoff75 is also immutable negative evidence: report
  `artifacts/g1_true23/g1_true23_sonic_student_teacher_recovery_cutoff75_seed20260805_v2.json`,
  SHA `1163f92dc7d3d35614932abb1ba659303c035b536a21ddfb363d5967c5d1b190`.
  Student q9 `9..83` is followed by 100 exact teacher actions q9 `84..183`,
  then the right wrist breaches the vertical EE gate at `0.261582 m` versus
  `0.25 m`. Earlier takeover extends survival by 15 frames but still does not
  recover. Base tilt stays below `0.086 rad`; action/parity/state/safety
  mismatch counts remain zero; labels remain zero.
- Cutoff50 proves a narrow recovery basin: report
  `artifacts/g1_true23/g1_true23_sonic_student_teacher_recovery_cutoff50_seed20260805_v3.json`,
  SHA `8d2089903013721739652d005ae360db6ff94023c91e2c97cff8702f0cd83442`.
  Student q9 `9..58` is followed by 460 teacher actions q9 `59..518`; the run
  completes `510/510` with final timeout only. Required mismatch/clip/nonfinite/
  limit/velocity/termination counts are all zero; PT/ONNX max error is
  `1.66893e-6`; minimum height `0.775807 m`, maximum tilt `0.138833 rad`, and
  final right-wrist z error `0.173643 m` versus `0.25 m`.
- This pass proves recoverability at exactly the one-second cutoff, not student
  quality, robustness, support, DAgger admission, or deployment. It publishes
  zero labels. The separate recovery campaign is now immutable at
  `artifacts/g1_true23/g1_true23_sonic_recovery_qualification_campaign_v1.json`,
  SHA `be8151713528f2cb9bcbc9b22c1fc8e2b913cfd5a07b6e246a38a112f17829b3`.
  Nominal seeds 0 and 1 pass; disturbance seed 0 also passes and returns to the
  evaluator-aligned recovery envelope in `0.10 s`. Disturbance seed
  `1542545985` then terminates physically at transition `258`, q9 `267`, after
  the exact additive impulse at transition `100`, q9 `109`.
- The failed disturbance is not an ABI or harness failure: qvel write is exact
  with world-delta error `1.34e-7`; PT/ONNX error is `1.91e-6`; action-chain
  error is `7.15e-7`; hard-safety violations are zero. It accumulates `118`
  soft warnings and one termination while height stays above `0.624 m`, tilt
  below `0.176 rad`, and tracking RMSE below `0.410 rad`. Campaign fail-fast
  stops at 4/20 runs, admits zero labels/rows, and does not authorize the
  recovery collector.
- Stop teacher-intervention BC for this branch. Next implementation is bounded
  genuine-SONIC task-space PPO: verified model250 actor weights plus the v2
  affine head, fresh critic/Adam/counters, encoder and decoder modules 0-13
  frozen, modules 14 and 16 trainable, V2 once, with deterministic closed-loop
  checkpoints. No teacher labels are used.
- No hardware, DDS, ZMQ, gantry, deployment, or robot actuation is authorized.

## 2026-08-09 full-clip map and genuine-SONIC bootstrap

- Immutable selected source remains the only teacher candidate: checkpoint
  `9cb0a06db441b8ceb51404b45ba25a81bd4120114aa6b97d6f660cac3f742f81`,
  actor `17302f7076cb480fe4ffc253e7b8228fcbaa033ccb3bf7aac1ed34940b8648ec`,
  one-input ONNX
  `321504108e677fb4b70d1398ff9a20e168def2231eb574e6d8fc1f39385d7b9b`.
  Four-ankle PPO/model55 is rejected as dominated; do not resume it or unlock
  more native rows.
- First nominal slice passes from q9 `9`: w0 executes `500/500` actions through
  q9 `508`, timeout only, hard/soft `0/0`, with PT/ONNX parity below `1.5e-6`.
  This is a local fixed-start result, not global teacher support.
- Continuous full-clip qualification fails physically after `868` actions at
  q9 `876` on `anchor_pos` and `ee_body_pos`; both wrists exceed the `0.25 m`
  z-error gate while ankle z errors remain below `0.07 m`. Hard-safety count is
  zero, but the failed suffix is quarantined and admits no labels.
- Fixed-phase restart shards also fail: w1 at q9 `868`, w2 at q9 `1054`, w3
  during burn-in at q9 `1272`, and w4 during burn-in at q9 `1686`. Failures are
  dominated by root/wrist drift, not ankle-limit or ankle-z termination. Because
  mid-contact reset state is incomplete, shard failure proves non-restartability;
  it does not by itself prove continuous local infeasibility. W2 surviving q9
  `868` also proves frame number alone is not the failure cause.
- A false numerical rejection in the selected-target-to-V2 composite was fixed
  without widening safety bounds: strict raw `abs<10`, inverse-domain, and
  forward safe/target checks remain; only redundant raw-space equality after
  ill-conditioned `atanh` was replaced by forward-equivalence checks. The q9
  `862` witness now has safe error `2.38e-7` and target error `1.19e-7`.
- Published teacher-controlled bootstrap:
  `artifacts/g1_true23/native124_selected_21204_teacher_bootstrap_seed20260805_v1.npz`,
  SHA
  `136768fd1595265d9743d5a9e5f7ef38e431de9a57f9ff85246123a7d649f475`;
  manifest SHA
  `5ab761c3b82d62c3a4524f1f195f6f187b91e56b41e854d186339ddcca86f4a1`.
  It contains `510` finite rows q9 `9..518`: `10` reset-prefix rows with real
  history depth `1..10`, then `500` full-H10 rows. PT/ONNX teacher parity max is
  `1.9073e-6`; max plain SONIC raw magnitude is `3.31565`; issue count is zero.
- Dataset admission is deliberately narrow: BC-eligible `510`, support-admitted
  `0`, learner-on-policy `0`, DAgger `0`, promotion/deployment false. Behavior
  controller is selected teacher, not SONIC student. Failed full-clip regions
  remain zero-label quarantine.
- Current implementation step: keep causal model-250 encoder and decoder layers
  0–7 byte-frozen; fit only decoder final affine (`11,799` parameters) by
  deterministic CPU PCA-ridge residual on raw-native23 labels. Source decoder
  mismatch on these rows is large (RMSE `0.76175`), while full decoder has
  `37,390,871` parameters and no source PT/optimizer, so full-network Adam is not
  justified by one contiguous trajectory.
- Next hard gate is genuine SONIC closed loop: encoder `267 -> 64`, concatenate
  `64 + H10 proprio930 -> 994`, adapted decoder `994 -> raw native23`, then V2
  exactly once. Student must act first at q9 `9`, with no warmup/substitution,
  and control all `510` actions. Only subsequent student-state, teacher-supported
  rows can become DAgger data.
- No hardware, DDS, ZMQ, gantry, deployment, or robot actuation is authorized.

## 2026-08-09 reset-seam correction (current controlling status)

- The prior two-action warmup evaluator changed the policy distribution: it applied fixed default-target actions at q9 `9` and `10`, then first called the actor at q9 `11`. PPO and the exact causal wrapper call the actor immediately at q9 `9`, with virtual reset torso history and zero previous action. Old q9 `58`/`85` failures therefore diagnose warmup/action-substitution sensitivity, not a proved 23-DoF capacity boundary.
- Formal no-update reset-seam reports start the actor at q9 `9`, perform zero substituted actions, and keep actor/critic/optimizer state unchanged:
  - selected source/`initial_model.pt`: `artifacts/g1_true23/native124_selected_v2_ankle_stage1/seed20260805_train_peak_v1/reset_seam_initial_v1.json`, SHA `a7abcc5f48070fc22c66446f4f75e00a9394db58a0912a6760ddb073de108954`;
  - model55: `artifacts/g1_true23/native124_selected_v2_ankle_stage1/seed20260805_train_peak_v1/reset_seam_model_55_v1.json`, SHA `48e12ea0a5692ea737d40e2db3aa701684014a314974d9299003cbb5e8c7b060`.
- Both complete `500/500`, time out at q9 `508`, and record zero hard violations and zero soft warnings. This proves only the first nominal 500-step DadDance slice under the exact training reset seam.
- Original selected source is the winner; model55 is safe on this slice but dominated. Relative to source, model55 has `3.68%` lower mean reward, `2.50%` lower ankle-position reward, `9.30%` worse torque penalty, `16.3%`/`12.5%` worse mean/max V2 projection, `2.86%` worse target RMSE, and `8.84%` higher maximum force ratio. Model55 improves ankle-orientation reward only `0.58%`.
- Stop four-ankle PPO. Do not unlock left knee, add seeds, or continue this derivative. Advance immutable source actor SHA `17302f7076cb480fe4ffc253e7b8228fcbaa033ccb3bf7aac1ed34940b8648ec` as the next simulator teacher candidate, still bound to checkpoint SHA `9cb0a06db441b8ceb51404b45ba25a81bd4120114aa6b97d6f660cac3f742f81` and ONNX SHA `321504108e677fb4b70d1398ff9a20e168def2231eb574e6d8fc1f39385d7b9b`.
- Next evidence order: per-step source PT-to-ONNX parity on the canonical q9 `9` rollout; continuous full-clip and fixed-phase restart qualification; separately identified teacher-controlled BC bootstrap data; then genuine SONIC student-controlled support/DAgger windows. Teacher-controlled rows must never be called learner-on-policy data.
- Still unproved: remaining `1,590` DadDance frames, arbitrary phase restarts, other motions, disturbance/DR, heldout retention, genuine SONIC policy success, hardware, gantry, or deployment.
- No hardware, DDS, ZMQ, gantry, deployment, or robot actuation is authorized.
- Canonical source qualification is now complete: `artifacts/g1_true23/selected_source_daddance_q9_9_nominal500_seed20260805_v1.json`, SHA `10684c42485a0526783405624526c42cbe903af4805c56a06646f3afd65b149b`, verdict `selected_source_daddance_nominal_slice_qualified`.
- Exact result: actor first runs q9 `9`; `500/500`; timeout-only done at transition `499`, q9 `508`; hard/soft `0/0`; runtime action-link mismatch `0`; selected PT/CUDA versus selected ONNX/CPU `500` checks, worst absolute error `1.1920928955078125e-06`, violations `0`; minimum height `0.7772897481918335 m`; maximum tilt `0.13973006791918682 rad`; maximum joint-velocity ratio `0.17032791674137115`; maximum target RMSE `0.1819543708292011 rad`; maximum plain SONIC raw magnitude `3.2780826091766357`. Actor, critic, optimizer, iteration, and rollout storage remain unchanged.
- This qualifies only the fixed-start nominal slice. Continuous q9 `9..2088`, fixed-phase restart, disturbances, other motions, genuine SONIC, and hardware remain unqualified.

## 2026-08-09 post-sim status (superseded by reset-seam correction above)

- The selected baseline artifact is diagnostic only: `artifacts/g1_true23/model21204_sonic_v2_neutral_seed20260805_nominal500.json`, immutable SHA `311691d5e89a86a73a5435a20208b93d0ce9919362402267e1a4b657b17e065e`, with exact phrase "nominal diagnostic only"; `500/500` records, q9 `11` through final `511`, no termination, discontinuity, nonfinite, raw-clip, action-semantics, target soft-limit, actuator-target soft-limit, measured soft-limit, or velocity failure; minimum height `0.754001259803772 m`; maximum tilt `0.13636365936818398 rad`; maximum tracking RMSE `0.21920073093275602 rad`; maximum join-plus-CPU-inference latency `9.437081 ms`.
- DadDance-v3 diagnostic SHA is `aff51f0dc92446025c09196a000aec291b9fcf44576fdffab8e016da8adab4fa`: whole-window quarantine: `47` valid teacher-controlled records followed by transition `47` at `q9=58` terminating on `ee_body_pos`, not `anchor_pos`, `anchor_ori`, or `timeout`; the failed run is not a support pass and admits zero teacher labels.
- Across those records: right ankle pitch V2 projection max/mean `0.9860610961914062 / 0.6995989582005967`, left ankle pitch max/mean `0.5811665058135986 / 0.42308512551987426`, mean absolute target tracking errors `0.581019479186928` and `0.6142912316829601` respectively. Left knee max `0.1672501564025879` remains only an evidence-gated unlock condition.
- Technical integration wiring for offline diagnostics is valid, but the current control state is not support-qualified, not DAgger-ready, not deployment-ready, and not hardware-authorized.
- BONES-seed and generic-seed replay are explicitly rejected as a primary adaptation path because extra labels from the same unsafe, unrepresentable position-setpoint semantics are not a first fix; they do not repair in-loop SONIC V2 projection/dynamics/contact.
- Next and only implementation commitment: a selected one-input native actor final-head adaptation scaffold where only the four left/right ankle pitch/roll final head rows remain trainable; left knee is a fifth row only if ankles-only heldout torque/contact/root/task gates fail while ankle gates pass. Remaining `18` outputs and genuine SONIC topology remain frozen.
- Action chain is exactly: model21204 raw hardware output -> selected HOME/scale candidate target -> SONIC default/scale conversion and hardware-to-native permutation to plain SONIC raw native -> reject abs >= 10 without clipping -> exactly one operational V11/V2 safe-target application -> safe native action and final hardware target. The selected candidate target never actuates, and comparator-only recomputation is not chained.
- Objective for the scaffold: in-loop SONIC V2 ON-POLICY task tracking using effective actuator torque, foot contact/force/slip, root/anchor/EE/task metrics on `500`-step heldout windows with strict disjoint windows, `10/10` nominal + `10/10` disturbance/recovery rollout gating, and zero termination, anchor, EE, nonfinite, raw-clip, action-semantics, limit, velocity, or contact-gate failures; whole-window quarantine on any failure.
- Keep hardware boundaries explicit: no robot run-time, no DDS, no ZMQ, no command-writer, no hardware execution surface.
- Keep whole-window quarantine/no row masking, no clipping-to-label conversion, explicit no-overwrite report behavior, and identity-locked base artifact behavior.
- No training/rollout authorization is granted by this handoff update.
- selected one-input identity does not generalize to or alter the legacy two-input Native124Policy `obs[1,124] + time_step[1,1]` ABI; genuine SONIC encoder267->token64 plus token64+H10 proprio930->decoder994->action23 remains the active topology; no 124-to-994 padding, relabel, or native weight transplant; hash/config/shape/action-order checks fail closed.
- Not support-qualified: all remaining `2026-08-06` plan/alternative sections are historical and not current commitments.
- Any derivative scaffold gets a new identity/hashes and may not overwrite selected immutable one-input artifacts.

## Historical status as of 2026-08-06 (superseded)

Status as of 2026-08-06. Written to let a fresh session resume without
rediscovering what has already been measured.

Current decision: correct low-latency action-row projection is proven exactly
equivalent to the 23-output warm start, so more plain subset distillation cannot
repair the embodiment mismatch. A staged whole-horizon lower-body/root solver,
safe-action-reachable retiming, fixed repair windows, and serialization-safe
schema v6 are implemented. Broad offline qualification still fails: only idle
passes its category gate, 19 hard-gate entries remain, and broad authorization
is `none`. The separately declared idle-only subset passes offline Step 1A, but
its exact fixed-horizon Step 1B campaign rejects schema-v6 action targets plus
live joint PD at steps 43 and 38 while the 29-DoF neural teacher completes both
500-step horizons.

Step 1C now tests real native 23-DoF state-feedback actors. Their exact
hardware-order 124-to-23 contract and checkpoint-to-ONNX parity are verified.
State feedback helps, but no current actor qualifies: model 11,500 passes the
change-idle static hold yet fails hands-on-back static at step 136 and full
tracking at steps 187/341; the same-lineage 500-iteration checkpoint sweep
through model 14,000 has no hands-on-back static pass under either preserved
legacy or explicit zero-reference-velocity hold semantics. Model 3,500 remains
best at 250/500 legacy or 230/500 zero-velocity steps. This is motion-specific
policy failure, not a valid dynamic expert.

The exact two-clip/four-span corpus, span-safe stock-native command, and a
hash-bound static A/B plan are now implemented. The plan fixes 512 environments,
24 rollout steps per PPO update, 2,000 updates per seed, and 250-update sessions
for exact model-3,500/model-11,500 bytes. It is deliberately read-only: there is
no run subcommand and `execution.available=false`. A strict weights-only
checkpoint format and CPU controller scaffold now prove no-overwrite publication,
counter, cap, poison, and one-save-per-session semantics, but plain-mapping
initialization/resume is test-only and production cannot initialize, learn, or
save. Native fine-tuning therefore remains blocked on a verified diagnostic
raw-trace replay validator, path/SHA-bound loading, full MJLab environment and
CUDA RNG state, byte-binding the installed RSL runner, and explicit Step 1C
authorization. DAgger/student adaptation and deployment remain forbidden.
Unsupported classes or failed rollout neighborhoods remain rejected, never
hidden with per-frame masks.

## Objective

Full-body VR teleoperation of a Unitree G1 EDU rev-1.0 (`mode_machine == 4`,
23 actuated joints) driven by a PICO 4 Ultra with two ankle trackers, using the
GEAR-SONIC controller family.

Not per-clip motion playback. Not the released 29-DoF path.

## Hardware and environment

| Item | Value |
|---|---|
| GPU | RTX 3070 Laptop, 8 GB VRAM |
| Host RAM | 32 GB, WSL2 capped at 8 GB (`memory=8GB` in `.wslconfig`) |
| WSL | Ubuntu-22.04, repo at `/mnt/z/codex/GR00T-WholeBodyControl` |
| Robot | G1 EDU rev-1.0, on gantry, ethernet `192.168.123.200/24` (eth0) |
| PICO LAN | workstation `192.168.1.140` (eth2) |
| Python | `/root/.venvs/g1_true23_mjlab/bin/python` (mjlab, torch 2.9, mujoco 3.5) |

A `memory=8GB` cap was added to `.wslconfig` because bulk file work made the
host unresponsive. The CPU count was deliberately left at the host default: the
shadow runbook pins CPU sets `0-3 / 4,5 / 6-15` and checks
`expected_online_cpus=0-15`, so shrinking it would fail that isolation contract.

## Assets that exist and are verified

| Asset | Path | Notes |
|---|---|---|
| BONES-SEED archive | `/root/bones_seed/g1.tar.gz` | 23.5 GB, sha256 matches the repo's pinned `MOTION_DATASET_SOURCE_ARCHIVE` |
| Extracted CSVs | `/root/bones_seed/g1_extracted/g1/csv/<session>/` | ~74k files; extraction was stopped early and is resumable with `tar --skip-old-files` |
| Converted clips | `/root/bones_seed/mjlab_npz/` | 5,318 npz |
| Full corpus | `/root/bones_seed/corpus/g1_true23_corpus_v3.npz` | 5,318 clips, 1,927,735 frames, 10.71 h |
| Easy corpus | `/root/bones_seed/corpus/g1_true23_easy_v1.npz` | 1,329 easiest clips, 3.22 h, max root speed 0.19 m/s |
| Difficulty ranking | `/root/bones_seed/corpus/difficulty_v1.json` | all 5,318 clips scored |
| Warm start | `sonic_release/g1_23dof_rev_1_0_low_latency_init.pt` | genuine SONIC low-latency weights |
| Released teacher | HF cache `models--nvidia--GEAR-SONIC/snapshots/*/model_decoder.onnx` | 994 → 29 |
| Task-space smoke | `/root/g1_true23_eval/taskspace_smoke_20260806_v6/` | 633-frame real BONES clip; motion, expert arrays, hashes, and report |
| Held-out task-space audit | `/root/g1_true23_eval/taskspace_heldout_20260806_v1/` | 12 clips / 4,524 frames; failed qualification plus exact regression audit; diagnostic only |
| Frozen held-out v2 diagnostic | `/root/g1_true23_eval/taskspace_heldout_20260806_v2/` | Frozen manifest; 11/12 schema-v4 clips completed, 10,746 audited frames; failed qualification; generated before final serialization margin and never training-authorized |
| Current held-out v4 publish | `/root/g1_true23_eval/taskspace_heldout_20260806_v4/` | Complete 12/12 schema-v6 triplets; hashes/certificates verified; qualification failed with authorization `none`; regression and omitted-joint audits complete |
| Idle-only supported-subset Step 1A | `/root/g1_true23_eval/taskspace_supported_idle_20260806_v1/` | Exact two frozen idle clips; schema-v6 hashes/certificates/audits verified; Step 1A passed and its sole authorization was consumed by the now-failed Step 1B campaign |
| Idle-only Step 1B canonical evidence | `/mnt/z/codex/GR00T-WholeBodyControl/artifacts/g1_true23_step1b_idle_nominal_20260806_v1/` | Exact 2 clips x 1 nominal episode x 500 steps; evidence valid, physical qualification failed, authorization `none` |
| Step 1C native model-11,500 bundle | `/mnt/z/codex/GR00T-WholeBodyControl/artifacts/g1_true23_step1c_native_policy_model_11500/` | Immutable checkpoint-bound ONNX plus official-MJLab static and exact Step1B-model static/full diagnostics; one static clip passes, no full clip passes |
| Step 1C checkpoint sweep | `/mnt/z/codex/GR00T-WholeBodyControl/artifacts/g1_true23_step1c_checkpoint_screen/` | Same DadDance lineage every 500 iterations from 2,500 through 14,000; all 24 checkpoints fail hands-on-back static under legacy and zero-velocity holds; model 3,500 is best at 250/500 and 230/500 respectively |
| Step 1C short model-2,000 probe | `/mnt/z/codex/GR00T-WholeBodyControl/artifacts/g1_true23_step1c_native_policy_short_model2000/` | Separate short-run actor; change-idle static/full both pass 500, hands-on-back static/full both fail at step 7, proving strong motion specialization |
| Step 1C supported-idle native corpus | `/mnt/z/codex/GR00T-WholeBodyControl/artifacts/g1_true23_step1c_supported_idle_native_corpus_v1/` | Exact four-span/2,047-frame stock-loadable NPZ plus fail-closed sidecar; corpus SHA `97cd8d8a...56c87`, sidecar SHA `acca7815...ca06`; diagnostic only, training unauthorized |
| Schema-v6 publish smoke | `/root/g1_true23_eval/schema6_publish_smoke_20260806/` | Real 547-frame idle clip; fresh float32 certificate, hashes, kinematic pass; fixed-horizon expert gate still false |
| Retiming walk probe | `/root/g1_true23_eval/retime_probe_walk_20260806_v3/` | experimental 315-to-1,142-frame continuous/time-scaled probe; kinematic gate passes, not corpus-qualified |
| Lower/root focused probes | `/root/g1_true23_eval/lower_root_probe_{crouch,walk}_20260806_v*/` | production-path diagnostics for clipped-seed retiming, projection, repair windows, and root acceptance; not qualification artifacts |

Each corpus has a matching `.spans.json` (clip boundaries) and `.json`
(recovery-style metadata the preflight requires).

## Code written this session

| File | Purpose |
|---|---|
| `gear_sonic/data_process/convert_bones_seed_to_mjlab_npz.py` | BONES-SEED CSV → mjlab npz via MuJoCo FK |
| `gear_sonic/scripts/build_g1_23dof_motion_corpus.py` | concatenate clips + span sidecar |
| `gear_sonic/utils/g1_23dof_multi_motion.py` | corpus loader with clip-boundary containment |
| `gear_sonic/scripts/score_g1_23dof_motion_difficulty.py` | rank clips by root speed / EE excursion |
| `gear_sonic/trl/mjlab/causal_teleop_runner_v13.py` | trains full `g1_dyn` decoder, freezes encoder |
| `gear_sonic/scripts/train_g1_23dof_mjlab_teleop_v13.py` | V13 trainer: corpus, curriculum, reward rebalance |
| `gear_sonic/scripts/distill_g1_23dof_decoder.py` | **QUARANTINED**: ELU, wrong native output selector, wrong teacher discovery, unvalidated Gaussian sampling, and an unsafe extra checkpoint root key. Correct subset distillation is now proven redundant. |
| `gear_sonic/scripts/capture_g1_23dof_decoder_inputs.py` | **INCOMPLETE and superseded for the current decision**: captures 268-dim tokenizer observation, not the 994-dim decoder input; hook is never registered |
| `gear_sonic/utils/g1_23dof_task_space_retarget.py` | exact-model 29D-to-true23 constrained task-space expert, staged whole-horizon lower/root feasibility, safe-action inversion, diagnostics, and residual-supervision arrays |
| `gear_sonic/utils/g1_23dof_trajectory_projection.py` | sparse whole-horizon projection onto q/v/a and time-varying repair-window bounds with explicit initial velocity and independent audit |
| `gear_sonic/utils/g1_23dof_safe_target_transform.py` | v2 exact runner/export transform; raw native action clamps to +/-10 before asymmetric tanh |
| `gear_sonic/scripts/retarget_g1_29dof_to_23dof_task_space.py` | batch BONES CSV retargeter; writes standard MJLab motion, expert npz, and provenance report |
| `gear_sonic/scripts/analyze_g1_23dof_task_space_regressions.py` | exact invalid-frame windows, gate causes, contact/constraint correlations, and worst-frame task deltas |
| `gear_sonic/scripts/analyze_g1_23dof_omitted_joint_correlates.py` | non-causal omitted-joint angle/velocity/acceleration correlations with exact two-pass reconstruction and lineage checks |
| `gear_sonic/utils/g1_23dof_task_space_qualification.py` | frame-weighted per-clip/category/aggregate Step 1A gates; never authorizes deployment |
| `gear_sonic/config/sim_validation/g1_23dof_task_space_heldout_v1.json` | preselected two-clip-per-category held-out manifest, 4,524 base frames |
| `gear_sonic/tests/test_g1_23dof_low_latency_action_subset_parity.py` | permanent exact-release structural and forward parity proof |
| `gear_sonic/tests/test_g1_23dof_task_space_retarget.py` | semantic-point, priority, constraint, safe-transform, motion-schema, and residual-target tests |
| `gear_sonic/tests/test_g1_23dof_lower_root_feasibility.py` | exact true23 adversary proving staged root/lower recovery while preserving q/v/a and root limits |
| `gear_sonic/tests/test_g1_23dof_trajectory_projection.py` | nearest-path, initial-velocity, fixed-window, infeasibility, and audit tests for the whole-horizon projector |
| `gear_sonic/tests/test_g1_23dof_task_space_qualification.py` | frame-weighted gate and missing-evidence tests |
| `gear_sonic/tests/test_analyze_g1_23dof_task_space_regressions.py` | exact frame-window and diagnostic-association tests |
| `gear_sonic/utils/g1_29dof_low_latency_teacher.py` | hash-pinned exact encoder + FSQ-32 + SiLU decoder runtime for `low_latency/last.pt` |
| `gear_sonic/utils/g1_true23_step1b_mujoco.py` | paired source29 neural-teacher / true23 schema-v6-live-PD fixed-horizon MuJoCo evidence runner |
| `gear_sonic/utils/g1_true23_step1b_qualification.py` | strict contained-artifact/raw-trace Step 1B validator; always fail-closed for training/deployment |
| `gear_sonic/scripts/run_g1_true23_step1b_mujoco.py` | plan/smoke/full CLI for the frozen idle campaign |
| `gear_sonic/scripts/validate_g1_true23_idle_step1b.py` | offline strict report validator and qualification writer |
| `gear_sonic/config/sim_validation/g1_true23_idle_step1b_qualification_v1.json` | frozen nominal-only schedule, identities, shipped gates, and stance thresholds |
| `gear_sonic/utils/g1_23dof_native124_actor_export.py` | weights-only strict native RSL actor reconstruction, static opset-18 export, exact initializer checks, 82-probe parity, hash lineage, and no-overwrite publication |
| `gear_sonic/scripts/export_g1_23dof_native124_actor.py` | standalone checkpoint-to-ONNX CLI; imports no MJLab and constructs no environment |
| `gear_sonic/utils/g1_true23_step1c_onnx_diagnostic.py` | diagnostic-only hardware-order native-actor replay in the exact Step1B true23 MuJoCo model, with static/full modes and early-stop support |
| `gear_sonic/scripts/run_g1_true23_step1c_onnx_diagnostic.py` | hash-pinned Step1C diagnostic CLI with clip selection; never grants training or deployment authorization |
| `gear_sonic/tests/test_g1_23dof_native124_actor_export.py` | strict loader/schema/no-overwrite/export/parity tests, including real model-11,500 vs immutable official actor |
| `gear_sonic/tests/test_g1_true23_step1c_onnx_diagnostic.py` | saved deploy-parameter and exact 124-D observation-contract tests |
| `gear_sonic/utils/g1_true23_supported_idle_corpus.py` | pinned two-clip/four-span native corpus materializer and strict re-open/source/hash/shape/semantic validator |
| `gear_sonic/scripts/build_g1_true23_step1a_idle_native_corpus.py` | no-overwrite CLI for two static zero-velocity spans and two trajectory spans with exact terminal padding |
| `gear_sonic/tests/test_g1_true23_supported_idle_corpus.py` | real-fixture golden corpus plus adversarial sidecar/hash/array/velocity/quaternion tests |
| `gear_sonic/envs/mjlab/native124_supported_idle.py` | exact-corpus, external-SHA-pinned, span-contained native `MotionCommand` plus stock no-state-estimation environment-config builder |
| `gear_sonic/tests/test_g1_true23_native124_supported_idle.py` | pinned identity/layout/tamper, balanced sampling, inclusive window, reset-boundary, and stock-config-preservation tests |
| `gear_sonic/scripts/train_g1_true23_native124_supported_idle.py` | CPU-only static A/B planner and strict authorization/preflight validator; hashes all runtime inputs, preserves diagnostic-only corpus flags, exposes no run path, and never lets self-attested diagnostics unlock later phases |
| `gear_sonic/tests/test_train_g1_true23_native124_supported_idle.py` | exact plan/seed/runtime/package bindings, authorization, diagnostic-schema, ONNX/checkpoint parity, phase-blocking, and resume-counter adversaries |
| `gear_sonic/trl/mjlab/supported_idle_checkpoint.py` | exact native actor/critic/Adam/RNG/lineage schema with weights-only reload, exclusive POSIX publication, no-overwrite race defenses, and byte/SHA-bound loading |
| `gear_sonic/tests/test_supported_idle_checkpoint.py` | checkpoint schema, optimizer, RNG, lineage, symlink, publication-race, tamper, and exact-load tests |
| `gear_sonic/trl/mjlab/supported_idle_runner.py` | CPU **test scaffold only** for 24-step PPO updates, 250-update cap, one session save, poison-on-partial-failure, and no-replay counters; production initialization/resume is intentionally unavailable |
| `gear_sonic/tests/test_supported_idle_runner.py` | rollout-count, cap, single-save, poison, counter, lineage, initialization-state, and production-containment tests |

Also modified: `gear_sonic/utils/g1_23dof_contract.py` (added
`TELEOP_REFERENCE_PROFILES` / `ALL_REFERENCE_PROFILES` and a 2-frame teleop
profile), `gear_sonic/utils/g1_23dof_semantic_reference.py` (per-profile frame
counts), and the sim-validation manifest hash (regenerated; audit at
`/root/g1_true23_eval/teleop2f/runtime_source_manifest_audit.json`).

## The single most important number

Measured on the reshaped warm start at **shipped 0.25 m thresholds**, no
training:

```
mean episode length   4.44 steps  (0.09 s)
per-step error        0.02441
target                500 steps   (10 s)
```

Every future attempt should be compared against this. The gap is roughly 100x
in survival — this is not a tuning gap.

At a relaxed 0.75 m threshold the same policy gets 26.16 steps and per-step
error 0.00620.

## What worked

- **Data pipeline.** Converter validated against real CSVs: 36 columns
  (`Frame` + 6 root + 29 `_dof`), all 23 model joints present, cm→m and
  degree→radian conversions confirmed by pelvis height 0.768–0.788 m and
  quaternion norms exactly 1.0.
- **Multi-motion training.** Clip containment verified: 0 boundary crossings
  across 2,048 sampled episodes.
- **V13 runner.** 18 trainable tensors (full decoder) vs V12's 2, encoder
  frozen, verified via `v13_runtime_contract.json`.
- **Difficulty ranking.** Cleanly separates idle loops (0.003 m/s) from jogging
  (1.733 m/s).

## What failed, with evidence

### 1. RL from the reshaped warm start — the core failure

Every configuration produced the same degenerate optimum: reward improves while
tracking degrades and episodes shorten, because shorter episodes accumulate
less penalty. The policy learns to terminate, not to track.

### 2. Learning rate is NOT the cause

A 200x sweep at fixed threshold produced near-identical degradation:

```
lr 1e-4   error_body_pos 1.014 -> 2.088
lr 1e-5   error_body_pos 1.038 -> 2.139
lr 3e-6   error_body_pos 0.981 -> 2.096
lr 5e-7   error_body_pos 0.988 -> 2.111
```

A separate run at 1e-6 collapsed *earlier* (iteration ~108) than 1e-5
(iteration ~560), not later. **Do not spend more time on learning rates.**

### 3. Termination thresholds — necessary but not sufficient

The shipped 0.25 m is unreachable for a fresh decoder (its own error is
0.38–0.45 m), so episodes die faster than learning can improve them. Relaxing
`ee_body_pos` alone just moved the binding constraint: `anchor_pos`
terminations grew 5x and became dominant. All distance gates must move
together (`--ee-termination-threshold` now does this).

### 4. Reward composition — the real structural cause

```
positive total (ALL tracking terms):  +0.0568
negative total (penalties):           -0.7253
ratio                                  12.8x
```

Dominated by `action_target_soft_limit_barrier`, weight **−50.0**
(`sonic_true23_causal_history_disturbance_v9.py:20`) against tracking terms
totalling **+5.0**. That barrier is a V11/V12 calibration scaffold sized to keep
a two-tensor probe at 5e-7 from moving. PPO correctly optimized penalty
avoidance. `--barrier-weight-scale` now rescales it without touching the
contract-verified V9 file.

### 5. Distillation attempt — INVALID RUN, discard the result

```
epoch 29   train 0.334   val 0.754   <- val plateaus
epoch 59   train 0.107   val 0.756   <- train falls 4x, val flat
```

**This run proves nothing about correct distillation.** It contained three
deterministic contract errors, plus one checkpoint-format error:

1. **Wrong activation.** `distill_g1_23dof_decoder.py` rebuilt the student with
   `nn.ELU()`. The architecture is **SiLU**
   (`gear_sonic/trl/mjlab/true23_actor.py:47,57`), confirmed at the ONNX graph
   level: the teacher has 7 `MatMul`, 6 `Sigmoid`+`Mul` pairs (SiLU), and zero
   `Elu` nodes. A network with the wrong nonlinearity cannot fit the teacher
   regardless of data quality.
2. **Wrong action order.** The script selected
   `SOURCE_IL29_KEEP_INDICES`. That selector is for compact padded
   observations, not the native rev-1.0 PhysX action order. It differs from
   `NATIVE_IL23_TO_CANONICAL_IL29` at **16 of 23** output positions.
3. **Wrong teacher.** The HF cache holds only root-level
   `model_decoder.onnx` — the **default step5** release. There is no
   `low_latency/` directory cached. The student derives from
   `sonic_low_latency` (`source_revision 7c90a56…`). The releases also have
   different decoder topology, so this is not a same-checkpoint row-selection
   comparison.
4. **Unsafe output shape.** The script adds
   `g1_23dof_distillation_report` as an unexpected checkpoint root key. The
   exact safe loader rejects the produced file before evaluation.

An earlier version of this document blamed the failure on synthetic Gaussian
input sampling being off-manifold. **That was asserted, not established** — a
favoured explanation reasoned backwards from a result while a bug was already
sufficient. Off-manifold sampling may still be a genuine problem, but this run
is not evidence for it.

`g1_23dof_distilled_v1.pt` should be treated as a **quarantined artifact**:
never evaluate it, never promote it.

### 6. Correct subset parity — now proved, not inferred

The permanent test uses the exact pinned `low_latency/last.pt`:

```
source sha256                 0031ae7d…d4507c
source revision               7c90a56c…
hidden decoder tensors exact  16 / 16
final weight rows exact        yes, native-23 selector
final bias and std exact       yes
forward MSE                    ~4e-14
forward max error              ~1e-6 (kernel roundoff)
```

For this warm start, the 23-output decoder is literally the same hidden trunk
and selected final affine rows as the 29-output teacher. Therefore, for every
994-D input, `f23(x) = P f29(x)` up to floating-point roundoff. Correct
same-teacher/native-map/SiLU subset distillation starts at zero loss and cannot
create morphology compensation. Fixing the old script would only reproduce the
checkpoint already present.

This does **not** mean the two robots have task-space parity. An exact-model
455-frame walking audit of the direct subset measured:

```
semantic hand error mean / p95 / max   12.75 / 14.44 / 16.50 cm
whole-robot COM error mean              1.70 cm
torso orientation error mean            0.233 rad
ankle-origin error                      exact parity
```

Network projection is a structural no-op; physical embodiment is not. This is
the reason to stop subset distillation and adapt in task space.

## Known confounds — read before interpreting any metric

- **`Metrics/motion/error_body_pos` scales with episode length.** Longer
  episodes sample later, higher-error timesteps. Always divide by
  `Train/mean_episode_length`. This was misread once as policy degradation.
- **`Train/mean_reward` is not comparable across changing episode lengths.**
  It improves when episodes shorten.
- Judge runs by `Train/mean_episode_length`, per-step error, and
  `Episode_Termination/time_out` rising above zero.
- `find_best.py` smoothing is unreliable in the last few rows.

## Measurement tools (in `/root`)

```
progress.py <run_dir>              honest metrics + per-step error
find_best.py <run_dir> <interval>  trajectory over iterations, best checkpoint
reward_breakdown.py <run_dir>      which reward terms dominate
compare_lr.py / compare_thresholds.py
```

## Task-space retargeting status — diagnostic foundation, current expert rejected

`gear_sonic/utils/g1_23dof_task_space_retarget.py` now performs constrained
29D-to-true23 retargeting on the exact MuJoCo models. It uses the repository's
semantic task points, including the 18 cm hand offsets on source wrist-yaw and
target fused wrist-roll bodies. The solver:

- uses the identical lower-body chain and protects feet first in each frame;
- protects feet/contact and whole-robot COM before soft upper-body objectives;
- treats hand position before hand orientation and elbow shaping;
- enforces the deployed inner safe-target envelope, velocity, acceleration,
  braking viability, and temporal smoothness;
- exactly inverts the asymmetric safe-target tanh into native IL23 actions;
- emits the standard MJLab motion schema plus expert arrays, per-frame validity,
  before/after task errors, hashes, and an offline-only report;
- never changes a policy checkpoint or grants deployment status.

Real 633-frame BONES smoke (`thinking_R_003__A140_M.csv`, eight IK iterations):

```
weighted task cost mean             0.3511 -> 0.2544  (-27.6%)
mean per-frame relative improvement                   28.5%
left semantic hand position         9.13 -> 4.21 cm
right semantic hand position        8.91 -> 5.66 cm
whole-robot COM                      2.61 -> 2.45 cm
left/right foot position            0.00 / 0.00 cm
valid expert frames                 98.4%
joint-limit hits / relaxed bounds   0 / 0
measured max velocity               5.66 rad/s  (limit 8.0)
measured max acceleration           80.0 rad/s² (limit 80.0)
maximum raw native action           7.76        (runner clip 10.0)
```

The smoke's apparent 1.6% failure is now localized exactly: frames 13--22, one
0.20 s startup window. Frames 13--20 improve hands and total weighted cost but
trip the old `1e-7` protected-tier tolerance for only 0.01--0.59 mm of COM
regression. Frames 21--22 are real, small temporal regressions while desired
hand speed is 1.4--2.3 m/s. Both feet remain exact, contacts never change,
safe-envelope hits are zero, and missing-wrist excursion is not explanatory.
Evidence: `taskspace_smoke_20260806_v6/regression_audit.json`.

### Broad held-out gate -- FAILED, DIAGNOSIS COMPLETE

Selection was frozen before results: two clips each for idle, walking, turning,
crouching, reaching, and fast motion; every category has at least 500 frames.
The first exact run produced 4,524 frames. Only 3/12 per-clip kinematic reports
passed and aggregate qualification correctly refused Step 1B:

```
valid expert frames                    72.21%   (required >=95%)
weighted task cost improvement         -29.95%  (required >=10%)
bilateral hand improvement              21.02%
bilateral hand error after              11.10 cm (required <=8 cm)
invalid frames / windows              1,257 / 119
invalid frames in runs >=10              1,024  (81.5%)
invalid frames at acceleration ceiling   1,175  (93.5%)
invalid frames with bad feasible seed    1,257  (100%)
```

Failure is not one undifferentiated morphology problem:

- Locomotion is mainly a solver-created temporal failure. Source paths reach
  15.85 rad/s and 725.9 rad/s^2. `_trajectory_bounds` clips all 23 joints,
  then the solver freezes the lower 12, so leg drift cannot be repaired. Worst
  examples: jog foot error 0 -> 411 mm, jump 0 -> 209 mm, sideways walk
  0 -> 170 mm. Errors form long gait-phase windows, not isolated noise.
- Contact flips correlate but do not cause the defect. Only 31.5% of invalid
  frames are within +/-2 frames of a flip, and stance-foot errors still reach
  115--176 mm.
- Reach failures are sparse and contact-stable, but expose genuine fixed-root
  workspace limits: bounded solves leave about 225--230 mm hand residual on
  hard frames. Pelvis/torso adaptation is required.
- The old `position_limit_hit_count` counts contact with the intentional safe
  envelope, not hard-limit violations. It is diagnostic, not a hard violation.

Exact windows, every invalid frame, cause masks, lift ratios, and worst-frame
task deltas are in
`taskspace_heldout_20260806_v1/regression_audit.json`. Do not train on this
expert, even with invalid rows masked: long on-policy neighborhoods are wrong.

### Action-envelope correction -- IMPLEMENTED, INVALIDATES OLD EXPORTS

RSL clamps raw actions to +/-10 before the environment, while the prior safe
transform/export did not. Three held-out clips therefore emitted unreachable
labels (maximum 37.86). Safe-target transform v2 now clamps before tanh in both
Torch and NumPy, binds the clip into its contract/hash, and the retarget solver
uses the exact forward image of +/-10 as its joint envelope. Existing
safe-transform-bound ONNX/evidence artifacts are intentionally hash-incompatible
and must be regenerated; training behavior below the existing runner clamp is
unchanged.

Task-space schema v3 also separates safe-envelope contacts from true hard-limit
violations, adds accepted-step/fallback diagnostics, tolerates empty hierarchy
tiers, uses continuous position/quaternion resampling, and can apply auditable
whole-clip time dilation. A sideways-walk probe improved from 46.35% valid with
12.2/17.0 cm foot maxima to 97.90% valid with zero foot-position error, no hard
limit violation, no action clipping, and cost 0.2721 -> 0.2572. Its remaining
24 invalid frames form nine short windows, none >=10 frames; 19 touch the
acceleration ceiling. This proves the temporal diagnosis, but it does not solve
fixed-root infeasibility or qualify the corpus.

### Staged whole-horizon lower/root feasibility -- IMPLEMENTED, NOT QUALIFIED

The production retargeter now has a staged path solver rather than only causal
per-frame clipping:

- It jointly represents a smooth world-frame pelvis offset and the 12 true23
  leg joints over the whole clip. Root orientation and the 11 upper joints stay
  fixed during this stage; accepted lower/root motion is frozen before the
  upper-body solve.
- Retiming is computed from the retained 23 joints after clipping into the exact
  raw-action +/-10 forward image. PCHIP position interpolation and quaternion
  rotation splines preserve continuous source semantics.
- A sparse projector enforces safe q bounds, 8 rad/s joint velocity, 80
  rad/s^2 joint acceleration, free measured initial velocity, and root limits
  of 80 mm, 0.75 m/s, and 6 m/s^2 across the full horizon.
- Repair is lexicographic: foot position is the first tier; foot orientation,
  COM, and the remaining task objectives are secondary. Only invalid cores and
  a stopping-distance halo may move; the complement is equality-frozen.
- A candidate may not create a newly invalid frame. Selection first minimizes
  invalid-frame count and normalized gate excess, then weighted error, then
  deviation from the clipped direct path. If no candidate passes, root remains
  zero and the certified projected seed is retained rather than emitting a
  poisoned expert.
- Schema v6 reserves a `0.995` q/v/a limit fraction before float32 persistence,
  independently audits the serialized trajectory, and recomputes the persisted
  per-frame velocity/acceleration diagnostics from those same float32 position
  arrays. It emits source/root paths, root offsets and derivatives,
  solver/projection counts, and lower/root validity before/after.

`test_g1_23dof_lower_root_feasibility.py` proves this machinery can recover a
synthetic exact-model knee-envelope adversary: fixed-root clipping moves both
feet about 39 mm, while staged root/leg motion brings both below 5 mm without
violating q/v/a or root limits. That is a capability test, not evidence that a
useful root correction exists for every real clip.

### Current held-out v4/schema-v6 publish -- 12/12, COMPLETE BUT NOT QUALIFIED

`/root/g1_true23_eval/taskspace_heldout_20260806_v4/` contains all 12
schema-v6 triplets. After cache replay, `batch.json` records `all_complete=true`,
12 verified cache skips, and zero failures. Every source, motion, and expert
hash is verified; reports match the batch; every persisted report has:

- `serialization_constraint_audit_passed=true`;
- certificate basis `serialized_float32_position_arrays`;
- derivative convention `free_initial_velocity_equal_first_interval`;
- per-frame q/v/a diagnostics recomputed from the same stored float32 position
  arrays;
- zero accepted lower/root steps and exactly zero selected root offset.

The separate real publish smoke at
`/root/g1_true23_eval/schema6_publish_smoke_20260806/` validates the complete
write/read path on 547 frames. Stored-q reconstruction exactly matches the NPZ
per-frame derivative diagnostics; measured maxima are 1.721577 rad/s and
51.058829 rad/s^2. Old schema-v4/v5 caches cannot satisfy schema-v6 provenance,
and cache reuse now also requires the serialization certificate and basis.

The retimer convergence fix is applied. The formerly missing
`walk__220713__sideway_right_loop_a021m` converges on its sixth update at
6,278 frames and scale 9.3268945. It is 99.0602% expert-valid; final serialized
q maxima are 4.782397 rad/s and 79.600513 rad/s^2. Root offset remains exactly
zero and no lower/root candidate step is accepted.

Current v4 aggregate state:

| Metric | Result |
|---|---:|
| Frames | 17,024 |
| Expert-valid fraction | 97.2921% |
| Weighted task-cost improvement | 7.5227% |
| Hand improvement | 14.7058% |
| Bilateral hand error after | 10.5412 cm |
| Maximum stance-foot position error | 77.6002 mm |

`qualification.json` has `requested_state_passed=true`, proving the requested
12-clip state is complete and internally consistent, but
`qualification_gate_passed=false` and `authorization=none`. Nineteen hard-gate
entries remain; idle is the only category that passes. Broad crouch, walk,
turn, reach/lift, and fast support is rejected. Completion is provenance, not
authorization.

The schema-v6 regression audit is integrity-clean. It finds 461 invalid frames
(2.70794%), 49 contiguous windows, eight long windows, and 329 invalid frames
inside long windows. Cause counts overlap:

- safe-envelope hit 359; right-foot threshold 259; bad feasible seed 184;
- acceleration ceiling 178; weighted regression 150;
- protected-priority-1 regression 69; COM regression 33;
- near/exact contact transition 32/7; left-foot threshold 32;
- high desired hand speed 0.

Exact long windows:

| Motion | Frames | Length | Main diagnostic association |
|---|---:|---:|---|
| crouch / kneeling start | `91--144` | 54 | stance-foot/safe-envelope failure |
| reach / down A036 | `485--500` | 16 | bounded task regression |
| reach / down A036 | `505--514` | 10 | bounded task regression |
| turn / start-walk | `384--474` | 91 | right-foot safe-envelope mismatch |
| turn / Mohak 045 | `261--284` | 24 | right-foot/safe-envelope mismatch |
| turn / Mohak 045 | `547--571` | 25 | right-foot/safe-envelope mismatch |
| turn / Mohak 045 | `587--668` | 82 | right-foot/safe-envelope mismatch |
| walk / sideways loop | `2929--2955` | 27 | COM/protected-priority regression |

The omitted-joint audit is descriptive and explicitly **non-causal**. Pooled
Cliff delta / invalid lift, followed by macro per-clip Cliff delta / lift:

| Feature | Pooled | Macro per clip |
|---|---:|---:|
| waist angle | 0.379 / 4.21 | 0.390 / 2.05 |
| waist velocity | 0.476 / 3.29 | 0.391 / 3.10 |
| left-wrist angle | 0.083 / 2.88 | -0.172 / 0 |
| left-wrist velocity | 0.509 / 4.10 | 0.371 / 2.42 |
| right-wrist angle | 0.057 / 2.88 | 0.260 / 0 |
| right-wrist velocity | 0.523 / 3.18 | 0.275 / 0.76 |
| all-six maximum velocity | 0.631 / 4.54 | 0.414 / 1.73 |

Static wrist excursion is not a universal correlate. Rapid omitted waist and
wrist motion correlates with invalid frames, but may be a motion-phase proxy;
the audit does not establish causality.

Step 1A verification state: 65 focused tests pass and Ruff is clean across 13
affected files. The retimer fix, complete publish, qualification, regression
audit, and omitted-joint audit are all finished. DAgger remains unauthorized.

### Idle-only supported subset -- Step 1A PASSED; Step 1B FAILED

`/root/g1_true23_eval/taskspace_supported_idle_20260806_v1/` uses exactly the
two unchanged frozen-manifest idle clips: 362-frame
`idle__220713__change_idle_left_a021` and 547-frame
`idle__220721__hands_on_back_loop_a036m`. Its manifest declares only
`qualification_categories=["idle"]`; source paths, source/model hashes,
motion/expert hashes, schema-v6 provenance, and float32 serialization
certificates all match the batch and per-clip reports. Regression audit v2 is
integrity-clean; the omitted-joint audit covers the same two clips and 909
frames.

Qualification is internally coherent between `batch.json` and
`qualification.json`: requested state passed, zero hard violations, two
independent clips, 909 frames, and 100% valid frames. Frame-weighted cost
improvement is 59.1847%; bilateral hand improvement is 53.4133%; bilateral
hand error after is 6.9269 cm; stance-foot maximum is effectively zero.
`qualification_gate_passed=true` and
`authorization=step_1b_fixed_horizon_expert_collection_only`.

Scope is deliberately narrow: this passes idle-only offline Step 1A and allowed
only the fixed-horizon Step 1B simulator qualification for that exact declared
support. That qualification has now run and rejected the schema-v6/live-PD
candidate. It does not authorize DAgger, training, deployment, or any
broad-motion class. Held-out v4 remains rejected broad negative evidence with
`authorization=none`.

### Frozen held-out v2 -- FAILED, DIAGNOSTIC ONLY

**Important provenance label:**
`/root/g1_true23_eval/taskspace_heldout_20260806_v2/` is a frozen **schema-v4
diagnostic generated before the final float32 serialization margin**. It must
not be promoted, merged into training data, or described as a current schema-v6
qualification result. Eight of its 11 persisted expert arrays measure
80.000162--80.000460 rad/s^2 after float32 round-trip, although their pre-write
float64 reports were at 80 within numerical noise. Schema v6's 0.995 reserve is
the fix; v2 must be rerun fresh rather than cache-reused.

The frozen manifest still provides useful negative evidence:

```
requested clips                         12
completed clips                         11
missing clip                             1  (required scale 8.58623 > max 8)
audited completed frames            10,746
invalid frames / fraction             372 / 3.4618%
invalid windows                         31
frames in runs >=10 / long windows     295 / 6
lower/root accepted steps                0 across all 11 clips
maximum selected root offset             0 m across all 11 clips
projection-created new invalid frames     0 across all 11 clips
Step 1B / DAgger authorization          false / false
```

The 96.54% valid fraction over the 11 serialized clips is **not an aggregate
pass**: the twelfth clip is absent, six long invalid neighborhoods remain,
multiple categories fail, and the artifacts predate the serialization margin.
Qualification correctly reports aggregate metrics as unavailable.

Exact failure clusters from `regression_audit.json`:

| Motion | Invalid cluster | Main cause |
|---|---:|---|
| crouch/kneeling start | 58 frames; `65--68`, `91--144` | 32 left-foot and 30 right-foot threshold failures at the safe envelope; four weighted regressions |
| fast/jump right | 36 frames in six windows; longest 19 | 32 weighted regressions plus four COM regressions; 15 touch acceleration ceiling |
| reach/down pair | 28 frames total; all windows <=5 | sparse acceleration-bounded weighted/COM regressions, not contact transitions |
| turn/start-walk | one 91-frame window, `384--474` | right-foot safe-envelope mismatch on all 91 frames |
| turn/mohak | 141 frames; long runs 24, 25, 82 | 138 right-foot threshold failures; safe-envelope contact on all 141 |
| walk/sideways stop | 18 frames in six short windows | all weighted regressions and acceleration-ceiling frames |
| idle pair, kneeling loop, jog backward | zero invalid frames | still subject to category/time-scale gates below |

Cause counts overlap by design. Across all invalid frames: safe-envelope hit
305, right-foot threshold 259, left-foot threshold 32, acceleration ceiling
117, bad feasible seed 93, weighted regression 85, near/exact contact transition
32/7, protected-priority-1 regression 17, and COM regression 6. There were zero
action-clip exceedances, hard-limit violations, trajectory relaxations, or foot
orientation gate failures in the completed reports. Contact changes correlate
with some windows but do not explain the long stance-foot failures.

Only the **idle** category passed its category gate. Other decisive failures:

- Crouch: 92.71% valid, 3.35% cost improvement, 53.1 mm stance-foot maximum.
- Turn: 90.86% valid, 2.05% cost improvement, 77.6 mm stance-foot maximum;
  required time scales 3.98 and 4.79.
- Fast: 98.96% valid but only 5.49% cost improvement; required scales 5.57 and
  6.07 exceed the qualification cap of 1.25.
- Reach/lift: 97.40% valid, but 9.28% cost improvement, 19.72% hand improvement,
  and 184.7 mm bilateral hand error after; one clip requires scale 1.81.
- Walk: aggregate unavailable. The completed clip requires scale 6.22; the
  missing loop requires 8.586 and exceeded the converter's maximum of 8.

No real v2 clip accepted a lower/root optimization step; every selected root
offset is exactly zero. This is an intentional fail-closed result: proposed
pelvis/leg motion could not reduce the real invalid set without introducing a
new weighted/protected-task regression. The staged solver is implemented, but
v2 does not establish that pelvis adaptation solves these real motion classes.

### Supported-subset direction

Broad all-motion expert construction is not the next training step. Use a
versioned support contract:

1. Freeze the complete held-out-v4/schema-v6 evidence. It is integrity-clean
   but fails broad qualification; do not reinterpret completion as support.
2. The separate **idle-only support contract** and Step 1A qualification are
   complete, and Step 1B has now rejected the schema-v6/live-PD candidate. Idle
   remains the only category with two independent clips and a passing offline
   gate, but its current action-target artifact is not a dynamic expert and is
   not authorized training data.
3. Treat kneeling-loop and the easier reach-down clip only as candidates for
   collecting more independent examples. A single passing clip cannot override
   its failed class; the reach clip's large hand residual remains unacceptable.
4. Exclude deep kneeling-start, turns, fast motion, and walking from initial
   support. Their long foot-envelope windows or 4--9x time dilation change the
   commanded behavior rather than merely smoothing it.
5. Persist a motion/window support label and reject unsupported commands before
   rollout. Never convert a long unsupported neighborhood into training data by
   masking individual rows.
6. Freeze the failed Step 1B bundle as negative evidence. Build a genuinely
   state-feedback 23-DoF balance/contact expert, then rerun the same gate. Only
   after that new expert passes may DAgger collect student-state supervision.

Current focused verification covers action-subset parity, safe-target v2,
schema-v6 retargeting, whole-horizon projection, staged lower/root feasibility,
manifest/cache handling, qualification, frame auditing, and omitted-joint
correlation semantics: the earlier 65-test set passes. The new Step 1B teacher,
runner, raw validator, and adversarial suite add 49 passing tests plus one
guarded real-checkpoint test; Ruff is clean across the eight Step 1B files. The
real checkpoint and canonical 2x1x500 campaign also ran successfully. Current
v4 evidence is serialization-safe but broad qualification failed; frozen v2
remains pre-final and cannot satisfy current guarantees.

The current focused Step1B/Step1C WSL regression command covers the teacher,
MuJoCo runner/qualification, native actor export, exact-model diagnostic,
supported-idle corpus/command, static planner, checkpoint format, and controller:
**174 passed, 1 guarded skip, and 1 known local-asset failure**. The failure is
the already documented empty
`gear_sonic/data/robots/g1/meshes/waist_support_link.STL` while compiling the
pinned source29 XML; it is unrelated to these changes. The 68 directly affected
planner/checkpoint/controller tests pass. Ruff check and format are clean across
their six files. Two independent audits found the initial mapping-resume and
uninitialized-training P1s; both are now contained by making all mapping paths
test-only and leaving production with no initialization or execution route.
This is not a claim that the missing production adapter is complete. A real CPU
MJLab environment separately confirmed actor shape 124, action shape 23, frame 0
held after reset, frame 499 consumed, and timeout reset to the next span start
rather than crossing the boundary. The published 2,047-frame corpus and sidecar
were rehashed unchanged at
`97cd8d8acf06a396ba041a7c5742a3eadc442e4235f0932f4e0b75a2cef85687` and
`acca78155ea075c65f6af0a66369dafc2a2dca88a9982ba8b26f20a206ceca06`.

## Historical — next steps (recommended plan, superseded)

This supersedes the earlier "fix the distillation inputs" plan. The core
diagnosis below is not mine; it came from a review of this work and is sharper
than my framing, which treated the embodiment mismatch as a risk to monitor
rather than as the thing to solve.

**Core diagnosis: the six missing joints change both kinematics and dynamics.
Copying or distilling retained action rows cannot compensate, and offline
task-space IK plus joint PD is not a dynamically valid expert.** The kinematic
retargeter is useful for constructing a feasible reference. It does not supply
centroidal balance, contact decisions, or corrective state feedback. A dynamic
23-DoF tracking policy or contact-aware WBC/MPC must close that gap before any
on-policy residual adaptation is admissible.

### Step 0 — parity test and quarantine — COMPLETE

- Quarantine `g1_23dof_distilled_v1.pt`. Never evaluate it.
- Permanent exact-release parity test added and passing.
- Correct subset distillation is proved redundant for the current warm start.
- Do not spend more time fixing the old distiller unless preserving it as a
  negative regression test.

### Step 1A — offline kinematic expert — STAGED SOLVER DONE, BROAD GATE FAILED

1. Keep the original 12-clip selection frozen. v1 and v2 remain diagnostic
   evidence; do not replace failed motions with easier examples.
2. Keep the completed held-out v4 publish, qualification, regression audit, and
   omitted-joint audit frozen as broad negative evidence. All 12 schema-v6
   triplets and their serialized float32 q/v/a certificates are verified;
   never mix schema-v4 outputs into this evidence set.
3. Record minimum time dilation as support evidence. Reject any candidate above
   1.25; do not reinterpret 4--9x slow motion as support for the commanded class.
4. The versioned idle-only supported-subset manifest now passes Step 1A with
   two independent clips. Expand only after another class passes its own hand,
   COM, foot, time-scale, and long-window gates.
5. Persist support labels and explicit rejection reasons alongside every expert
   shard. Unsupported clips/windows are absent from training, not row-masked.
6. Do not start DAgger while any supported class has a long invalid window or
   until a replacement dynamic true23 expert passes the Step 1B contract. A
   zero selected root offset is evidence, not a failure to log: it means no safe
   improvement was found under the current objective.

### Step 1B — simulator expert qualification — COMPLETE, FAILED

Canonical evidence:
`/mnt/z/codex/GR00T-WholeBodyControl/artifacts/g1_true23_step1b_idle_nominal_20260806_v1/`.
The report hash is
`464ac285accb1290dfa7848ea04ad5834701ab9bc5538b60e599366ec5df7f65`;
the qualification hash is
`2354dffad80e8a52b0f38bfb8e982a6ee2e7bb89e848c3a9749c42b44ac3bf97`.
The validator reports `evidence_valid=true`, `qualification_gate_passed=false`,
and `authorization=none`.

The two arms were exact and intentionally different:

- Teacher: pinned neural 29-DoF low-latency SONIC checkpoint
  `0031ae7d...d4507c`, live state feedback, exact 267-D encoder, FSQ-32, and
  930-D H10 decoder proprioception.
- Candidate: immutable schema-v6 `action_target_native`, exact safe-target-v2
  transform, then live q/dq PD at every physics substep. This is precomputed
  task-space retargeting, **not** online task-space IK or WBC.

| Frozen idle clip | 29-DoF teacher | true23 candidate | true23 hard-limit steps |
|---|---:|---:|---:|
| hands-on-back loop | timeout at 499 | `anchor_pos` termination at 43 | 0 |
| change-idle-left | timeout at 499 | `ee_body_pos` termination at 38 | 10 |

Aggregate hard-gate result:

```
paired timeouts                         0 / 2  (required >=95% per clip)
hard-limit violation steps                10  (required 0)
stance-foot position max          0.362242 m  (required <=0.005 m)
stance-foot orientation max       2.093615 rad (required <=0.005 rad)
contact mismatch                   1760/2000  (0.88; required 0)
nonfinite samples                           0
```

This is not merely post-fall inflation. Through each first termination, stance
position/orientation already reached 57.4 mm / 0.321 rad on hands-on-back and
37.3 mm / 0.169 rad on change-idle, both far outside the 5 mm / 0.005 rad gate.
The exact teacher completed both horizons with zero limit or nonfinite faults.

Root-cause ablations:

- Safe-target reconstruction matches the serialized q reference within
  2e-6 rad; direct reference targets behave the same.
- No joint torque saturated during steps 25--50.
- Holding frame 0, full playback, +1-frame lead, and +3-frame lead terminate on
  exactly the same 43/38 steps. Timing lag and reference acceleration are not
  the primary cause.
- Ankle-pitch tracking error grows to about 0.57/0.53 rad. Pelvis tilt reaches
  46.8/36.8 degrees at termination while both feet are still in contact; an
  approximate COM-vs-contact-foot support check is already outside by about
  0.51/0.46 m. Contact loss happens later.

Conclusion: plain joint PD cannot even stabilize the first idle posture in the
true23 model. The schema-v6 trajectory remains a kinematic **reference**, but it
is rejected as a DAgger action expert. Removing the validator's independent-FK
authorization blocker would not change this result: 19 physical gate failures
remain.

### Step 1C — dynamic true23 expert — STATIC PLAN/CHECKPOINT SCAFFOLD READY; PRODUCTION EXECUTION BLOCKED

The shortest state-feedback path is now implemented and measured. The local
`unitree_rl_mjlab` task is
`Unitree-G1-23Dof-Tracking-No-State-Estimation` at repository commit
`1425b15f73bd4095f0df53709d7c389c3eb9e790`. Its deterministic actor runs at
50 Hz and is exactly:

```
observations  qref23, dqref23, anchor-orientation6, base-angular-velocity3,
              q-minus-home23, dq23, previous-action23  = 124 float32 values
actor         normalize by saved mean/(std + 0.01), ELU 512/256/128, 23 outputs
action        home23 + saved-scale23 * actor-output23
```

All asymmetric 23-vectors use the saved Unitree hardware-compact order from
hip/ankle pairs through waist and arms. They must **not** be permuted through
Gear's separate breadth-first `NATIVE_IL23_JOINT_NAMES` order. The first replay
made that mistake; its report is quarantined as
`INVALID_order_permutation_exact_model_diagnostic_v1.json` and is never valid
physics evidence.

Pinned model-11,500 identity:

- checkpoint SHA-256
  `1302ed2d7128c5f129611c29a34181d1ac7e27d2c15f551e49453e41ee81ec4a`;
- immutable actor SHA-256
  `b0476b3e5d281d2f3bb47efd5dfe8fdf2fd93a3daf2d6c752d977a9ba7e05b02`;
- deploy-parameter SHA-256
  `40fdcadb3096842414d6e5307d50ef69bec42f696d0866a40cc487f1ae7103b8`.

The standalone weights-only exporter validates the exact 13-tensor actor
schema, saved normalizer, ELU graph, static opset-18 topology, checkpoint
initializers, and exclusive no-overwrite publication without importing MJLab.
Across 82 deterministic probes, official actor versus checkpoint and fresh
export versus checkpoint both stay below `6.676e-6` maximum absolute error at
`atol=rtol=1e-5`; fresh export versus official actor is bit-identical.

Nominal diagnostic results (`step` is the zero-based first shipped
termination; `PASS` means all 500 control steps):

| Actor / environment / reference | hands-on-back | change-idle |
|---|---:|---:|
| model 11,500, official MJLab, repeated frame 0 | `anchor_pos` step 224 | PASS |
| model 11,500, exact Step1B model, repeated frame 0 | `anchor_pos+ee_body_pos` step 136 | PASS |
| model 11,500, exact Step1B model, full terminal-hold playback | `ee_body_pos` step 187 | `ee_body_pos` step 341 |
| short-run model 2,000, exact model, static / full | `ee_body_pos` step 7 / step 7 | PASS / PASS |
| main-lineage model 3,500, exact model, static / full | `ee_body_pos` step 249 / step 228 | PASS / `ee_body_pos` step 197 |

Official-MJLab static evidence hash is
`60ad932458fa20e2f6d1720c8d6a4813c6ab0c5c075402027190061c6041929c`.
Binding exact-model report hashes are
`5321c1b901db2296245efbd00ac16f57c122d47220554195a2043e394d14f835`
(model 11,500),
`7d817db493ed3580bbf88df8c53b5cebb7cb13af9e765abe4ca670133b33336d`
(short model 2,000), and
`8c5d3f41e26008119250b70bb292b7f45194a830f7a23c51484c58671615c0ed`
(model 3,500). Counts after first termination in the non-stopping reports are
post-fall diagnostics and must not be interpreted as pre-failure faults.

Held-reference velocity is now versioned explicitly without changing those
reports. `static_frame0_zero_velocity` repeats frame-0 pose with qdref=0;
`full_terminal_hold_zero_velocity` keeps original qdref through the final source
frame and switches to zero only in the padded suffix. Both legacy modes remain
unchanged. Zero velocity improves model 11,500 hands static from 137 to 192
executed steps but reduces model 3,500 from 250 to 230; neither passes. Full
results remain 187/341 and 228/197 because every failure occurs before a
terminal-hold suffix is reached. The zero-velocity report hashes are
`d1bd484de05961109c10e6ae38bf86a0529dcde9d5222fb5cfac3627d12ac158`
(model 11,500 early-stop) and
`639c0553c086d9c0a1d44f8e3271325af62c08c01f8893f80f363c423ca1c7d4`
(model 3,500 early-stop).

The exact-model prerequisite screen covers every main-lineage checkpoint at
500-iteration spacing from 2,500 through 14,000. Every actor fails the
hands-on-back repeated-frame-0 hold under **both** held-velocity contracts: 0/24
passes in each screen. Best legacy result is model 3,500 at step 249/250
executed, with zero hard-limit and torque-saturation steps through termination.
Best zero-velocity result is also model 3,500 at step 229/230 executed, with zero
hard-limit and one torque-saturation step. Model 13,500 reaches 95 legacy / 107
zero-velocity steps; the final stable model 14,000 bytes reach only 78 / 78.
Separately, the native DadDance validator has no strict pass through its first
model-14,000 screen; its best strict completion count remains 74/100 at model
8,000. Therefore no existing checkpoint is a Step1B action expert, even though
feedback produces a large improvement over raw PD and can solve one
pose/motion.

The accepted model-13,500 checkpoint/ONNX/two-mode-report hashes are
`8c87e74ec7c18c96bf180f70dd2484f48a198e00438d6da02765c26e83605b9c`,
`cb094784e35c98c4ae575850fdaa75c125f14f33c00870bbf87eff7d3559b525`,
and `d0ecc6dfb2aa4c090f993d342c53469804c3f7973e79a8814a5951907507427e`.
The final stable model-14,000 equivalents are
`a63c4822ee6cb05fb7743d5b078abe9012b62b9cffddfd7b0946531fc04b0ca2`,
`e50479cd07a8f9198e3b324f9649e526c0d7a29268477f804fb82701fbc1f330`,
and `a9050b1c983b206f591b56b6ada93cb38dca82211dd6865cae6209a572a34184`.
Both exports pass all 82 deterministic parity probes. The active DadDance
wrapper rewrote `model_14000.pt` once after its first screen; the earlier
`41958e68...4024` checkpoint lineage and its reports are marked stale and
excluded in the screen README, not silently overwritten or cited as current.

Next construction:

1. **COMPLETE:** materialize one stock-loadable, hash-bound corpus with four typed spans:
   two 500-frame zero-velocity frame-0 holds; the 362-frame change clip plus
   138 zero-velocity terminal-hold frames; and the complete 547-frame hands
   clip. Persist exact source hashes, original/stored lengths, transforms, and
   span boundaries in a fail-closed sidecar. The result contains 2,047 frames;
   corpus SHA-256 is
   `97cd8d8acf06a396ba041a7c5742a3eadc442e4235f0932f4e0b75a2cef85687`
   and sidecar SHA-256 is
   `acca78155ea075c65f6af0a66369dafc2a2dca88a9982ba8b26f20a206ceca06`.
2. **COMPLETE:** the dedicated span-aware native `MotionCommand` balances
   source clips, filters static versus trajectory phase, constrains each
   500-step start to `[span.start, span.stop-500]`, and never clamps, wraps, or
   crosses a boundary. It requires a caller-supplied corpus SHA independent of
   the sidecar and rehashes after stock `MotionLoader` returns. Offset zero is
   the binding start; the later bounded-window mode exposes only offsets 0--47
   for the 547-frame clip.
3. **COMPLETE, READ-ONLY:** the static A/B planner binds exact model-3,500 and
   model-11,500 checkpoint bytes, the corpus/sidecar, 26 runtime files, package
   versions, seed `20260806`, 512 environments, 24 rollout steps, 2,000 updates
   per seed, and 250-update sessions. Its current payload SHA-256 is
   `7888268168a7275b66284957eb972525cc43c8eafc1d6e9b77c6a9948c319c49`.
   It has no run subcommand, reports execution unavailable, and rejects every
   trajectory plan until independently verified diagnostic replay exists.
   Self-attested curriculum JSON can never promote a parent or unlock training.
4. **COMPLETE, CPU/TEST SCAFFOLD ONLY:** the checkpoint module validates exact
   actor, critic, 17-parameter Adam, RNG, lineage, trainer, and command schemas;
   POSIX publication reloads and hashes held bytes, fsyncs, publishes exclusively,
   and never overwrites. The controller tests exactly 24 environment steps per
   PPO update, at most 250 updates, one checkpoint after a successful session,
   no saved-iteration replay, and poison-on-rollout/update/capture/save failure.
   Plain `Mapping` initialization/resume is explicitly test-only; normal
   construction cannot initialize, learn, or save.
5. **BLOCKED BEFORE EXECUTION:** implement a production adapter that loads source
   and resume checkpoints internally from exact path+SHA bindings; captures and
   restores the complete native command, MuJoCo, environment, action/observation,
   optimizer-schedule, CUDA, and per-generator RNG state; byte-binds the installed
   RSL runner; and validates diagnostic reports by independent replay of raw
   traces. Until then the static plan is evidence, not an executable trainer.
6. **BLOCKED ON EXPLICIT STEP1C TRAINING AUTHORIZATION AFTER ITEM 5:** when the
   GPU is free, run short deterministic A/B trials from model 3,500 and model
   11,500. Select on the exact Step1B-model static/full gates, not reward,
   self-attested curriculum JSON, or DadDance validation.
7. Keep live q/dq/IMU feedback, COM/foot/contact objectives, fixed horizons,
   and shipped 0.25 m termination thresholds. Add perturbations only after a
   2/2 nominal full-motion timeout.
8. Reuse the exact Step1B runner/validator contract. Require valid provenance,
   2/2 nominal timeout, zero limits, and stance/contact gates before collecting
   any DAgger label or expanding support. A contact-aware WBC/MPC remains the
   higher-effort fallback.

Important authorization boundary: current Step1A authorization was consumed by
the failed Step1B campaign and does not authorize GPU training. Corpus/command
implementation, CPU tests, read-only planning, and simulator diagnostics may
proceed. Explicit Step1C authorization is necessary but not sufficient: the
production adapter and replay validator in item 5 must pass review first.

Remaining estimate, not a success guarantee: approximately **1--2 focused
engineering days** for the production adapter, raw-trace validator, and review.
After that work, a free GPU, and explicit authorization, expect roughly
**80 minutes** for the first static A/B evidence and **4--5 GPU hours** for the
full static → trajectory-start → bounded-window funnel if every gate passes.
Any failed gate stops the funnel earlier. A separate user-owned torque run owned
the GPU at the timestamped snapshot below and is not part of this estimate;
availability must be rechecked before execution.

### Step 2 — V14 residual objective on student states — BLOCKED

- Do not collect DAgger labels from the failed schema-v6/live-PD arm.
- Once a dynamic 23-DoF expert passes Step 1B, collect its actions on student
  states and train the existing full decoder. Do not add a deployed residual
  branch that breaks the SONIC state-key or ONNX topology contract.
- Train only inside clips/windows admitted by the versioned support contract;
  `expert_valid` row masking alone is insufficient. Bind every shard to robot,
  motion, student checkpoint, reference profile, expert, schema, and support
  manifest hashes.

### Step 3 — do not freeze the encoder permanently

The causal-history profile changes future-reference semantics and states
`retraining_required=True`
(`gear_sonic/envs/mjlab/sonic_true23_causal_history.py:46`). Train an adapter or
unfreeze later encoder layers. The V13 runner currently freezes all encoders —
that was an assumption about preserving the SONIC prior, not a verified choice.

### Step 4 — PPO fine-tune

- Corrected barrier weight (`--barrier-weight-scale`).
- **Fixed-horizon rollouts** so terminating early cannot improve return. This is
  a better fix than reweighting alone: it removes the degenerate optimum
  structurally rather than rebalancing against it.
- Per-step normalized tracking metrics (see confounds above).
- Easy-motion curriculum, then the full corpus.
- Tighten gates gradually back to 0.25 m.

### Step 5 — validation

Held-out fixed-horizon tracking, disturbance campaign, ONNX parity, live
shadow, then gantry.

## What is actually ruled out, and what is not

**Ruled out by measurement:**

- Learning-rate tuning (200x sweep, near-identical degradation; 1e-6 collapsed
  *earlier* than 1e-5)
- Relaxing `ee_body_pos` alone (moves the binding constraint to `anchor_pos`)
- Corpus difficulty alone (easy corpus still collapsed)
- Correct same-teacher/native-map/SiLU action-subset distillation as an
  improvement over the current warm start (exact structural and forward parity)
- Direct retained-joint projection as a morphology solution (large measured
  hand/COM/torso task-space error despite exact ankle parity)
- Current greedy fixed-root, causal per-frame task-space solver as a broad
  expert (held-out gate failed; 1,257/4,524 invalid frames in long clusters)
- Frozen schema-v4 staged output as a broad expert (only 11/12 clips completed,
  six long invalid windows remain, eight persisted clips exceed the acceleration
  limit after float32 round-trip, and no real root correction was accepted)
- Per-row masking as a cure for unsupported motion (295 frozen-v2 invalid frames
  occur inside six long neighborhoods)
- Current fast, turn, and walk references at near-original timing: required
  dilation is about 4--9x, above the 1.25 qualification support cap
- Contact-transition relabeling as the locomotion fix (large stance errors and
  most failures occur away from exact contact flips)
- Unreachable post-hoc action labels (safe-target v2 now clamps raw actions and
  retarget schema v3 solves only inside the exact +/-10 forward image)
- Schema-v6 task-space action targets plus live joint PD as a dynamic true23
  expert, even for the two qualified idle clips (terminations at 43/38, zero
  paired timeouts, 10 limit steps, 88% contact mismatch)
- Static-frame holding or +1/+3 reference phase lead as a repair for that
  failure (all variants terminate on the same 43/38 steps)
- Existing native DadDance policies as a drop-in dynamic expert for the two
  supported idle clips. The exact main-lineage static prerequisite sweep from
  model 2,500 through 14,000 has 0/24 hands-on-back passes under both held-frame
  velocity contracts; best is model 3,500 at 250/500 legacy and 230/500
  zero-velocity steps. Model 11,500 also fails both full clips at steps 187/341.
- Treating one successful pose or motion as general support. The separate short
  model 2,000 passes both change-idle static/full runs but fails hands-on-back at
  step 7; later checkpoints show different tradeoffs.

**NOT ruled out — untested or not yet qualified:**

- Off-manifold input sampling as a failure cause. Asserted, never established.
- A dynamically controlled supported subset. Idle is the only frozen-v2/current
  v4 category that passed offline Step 1A, but the current live-PD action expert
  failed Step 1B and is frozen as negative evidence.
- Real-clip pelvis/root adaptation under a better objective or reference. The
  synthetic feasibility test passes, but every frozen-v2 real candidate was
  rejected and retained zero root offset.
- A native 23-DoF policy fine-tuned on the exact clip-contained two-idle support
  set, or a contact-aware WBC/MPC, at shipped gates. Existing DadDance actors
  fail, but the intended multi-motion expert has not yet been trained or
  qualified.
- On-policy residual learning / DAgger. Dataset target plumbing exists; the V14
  runner and student-state collection do not.
- Unfreezing the encoder. Never attempted.

## Current confidence

- Exact action-subset parity conclusion: **>99.9%** (structural proof plus exact
  release forward test).
- Diagnostic/provenance pipeline and frame-audit correctness: **~95%**; frozen
  v2 also demonstrated why post-serialization auditing is mandatory.
- The current schema-v6/live-PD artifact is safe to use as a DAgger action
  expert: **<0.1%**. Its canonical Step 1B report is evidence-valid but fails
  19 physical gates; authorization is explicitly `none`.
- Diagnosis that offline task-space targets fail because they lack dynamic
  balance/contact feedback: **>95%**. Static/full/+1/+3 playback is invariant,
  raw-PD torques do not saturate before failure, the 29-DoF neural teacher
  passes both horizons, and native state feedback extends some exact-model
  holds from 38--43 steps to 137--500 steps.
- Any currently exported native 23-DoF actor is safe as the two-clip DAgger
  expert: **<1%**. No main-lineage checkpoint through 14,000 passes even both
  static prerequisites; no tested actor passes both full clips.
- Current staged solver selects useful pelvis/root motion on real supported
  clips: **<25% confidence** until one accepted real correction survives the
  full validity and serialization gates.
- DAgger student beats the 4.44-step baseline if a new dynamic expert first
  passes: **~60–75%**; no DAgger/student training is authorized before that
  condition.
- Clip-contained native multi-motion fine-tuning can produce a 2/2 static
  expert: **~70%**. Full-motion 2/2 qualification remains **~40–60%** until the
  first deterministic fine-tune A/B result; current checkpoints show strong
  motion specialization.
- Reliable 500-step true23 behaviour at shipped 0.25 m gates with the current
  schema-v6/live-PD arm: **<1%**, measured 0/2 timeouts.

## Historical — alternative if the recommended plan fails

The `unitree_rl_mjlab` tracking task (`Unitree-G1-23Dof-Tracking-No-State-Estimation`)
is **natively 23-DoF** with no reshaping and no missing-joint assumption, and
is already known to train on this robot. Its observation contract can accept
live reference fields plus measured state, so a **qualified multi-motion** actor
could consume the retargeted PICO stream without SONIC's decoder. Existing
DadDance actors are not that qualified policy.

Cost: the SONIC-shaped deployment chain (`g1_true23_live_shadow`,
`g1_true23_active_gantry`, promotion sidecars, shape contracts bound to
267→64 / 994→23) would need its model-binding layer reworked. The DDS layer,
freshness gates, CPU pinning and gantry logic still apply.

Loses: latent token interface (planner, MotionBricks, text-to-motion), VLA /
GR00T N1.7 integration, and SONIC's 288 h behavioural breadth.

## Safety state — must be restored before any validation

Currently relaxed **for training only**:

- `--ee-termination-threshold 0.75` (shipped: 0.25) — also relaxes `anchor_pos`
  and scales `anchor_ori`
- `--barrier-weight-scale` on `action_target_soft_limit_barrier`

Neither weakens a deployment check — joint limits are enforced by the
termination terms, the MuJoCo campaign, and the promotion gates. **Both must be
restored to shipped values before MuJoCo validation, ONNX export, promotion, or
any live shadow.**

Unrelated pre-existing test failures (not caused by this work): local G1
URDF/XML/mesh assets differ from the pinned Unitree source, and a sim-report
fixture has a `runs[0]` ordering mismatch.

## Human-gated steps that no amount of compute removes

1. **Live shadow** — PICO powered, worn, calibrated, both ankle trackers
   connected, operator standing in the neutral pose. Five earlier runs failed
   on `XR24 acquisition pose is not neutral standing`.
2. **Gantry authorization** — the phrase `I_CONFIRM_G1_TRUE23_STAGE1_GANTRY`
   typed at an interactive TTY within 300 s of a passing shadow.

Also verify before a session: the runbook pins PC Service at `192.168.1.182`,
but the workstation's PICO-side IP is `192.168.1.140`. If the headset targets
`.182` it will never reach the service.

## Ongoing native DadDance run

The original scheduled DadDance-to-30,000 wrapper is no longer active. A
read-only audit on `2026-08-06T10:45:42+10:00` found that its 16,000 leg started
at `2026-08-06T10:14:03+10:00` from `model_14000.pt` with
`remaining=2001`, logged learning iterations 14,000 through 14,009, and then
stopped. Its stdout ends at `2026-08-06T10:14:44+10:00`; there is no terminal
success/failure line for that leg and no checkpoint newer than
`model_14000.pt`. The reason it stopped is unproven. The preceding model-14,000
screen failed at 53/100 strict completions versus 95 required; the main-lineage
screen had model 8,000 best at 74/100. The shared screening summary has since
been replaced by the torque-run screen and is not durable main-lineage evidence.

At the latest read-only snapshot, a separate user-owned GPU experiment was
launched by
`external_dependencies/unitree_rl_mjlab/scripts/run_dad_dance_torque_finetune.sh`.
The wrapper has changed/restarted during this work, so the following is only a
read-only snapshot at `2026-08-06T12:08:12+10:00`: the active child was the
`Torque-Strong` immutable-v3 8,500→9,000 leg, loading the immutable-v2 candidate
`model_8500.pt` and requesting 501 iterations. The immutable-v2 candidate pool
contained model 8,500 SHA-256
`d0c9da72a310e941d103c3a813176a5dad46883b5725e7e3e7ab6d30776d82d6`
at 13/20 completed screens and model 9,000 SHA-256
`3d77373cb59d5360d1fe0922e067e3267be07c4492204e741ddd6256d91ecf29`
at 11/20. Neither entered strict validation and the shared summary remained
`passed=false`. Candidate/segment copying now preserves screened source bytes;
this is a useful immutability improvement, not a Step1B qualification. Do not
stop, restart, edit, or otherwise alter this process from Step1C work.

The underlying no-replay bug is proven and remains present in the active wrapper,
notwithstanding its safer cross-directory copying. It computes the inclusive
count `remaining_iterations=$((target_iteration + 1 - latest_iteration))` at
line 61 and passes that count to `train.py`. `train.py`
loads the checkpoint at lines 123--125 and calls `learn` at line 132. The MJLab
runner restores `current_learning_iteration=loaded_dict["iter"]` at
`external_dependencies/mjlab/src/mjlab/rl/runner.py:125`; RSL then starts its
loop at that same value, updates the policy, and saves `model_{it}.pt` at
`/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages/rsl_rl/runners/on_policy_runner.py:77-79,107-112,127-133`.
MJLab's unguarded `torch.save` is at
`external_dependencies/mjlab/src/mjlab/rl/runner.py:65-68`. Thus the first
resumed update repeats iteration `n`. Older same-directory wrappers overwrote
the loaded `model_n.pt`; the immutable wrappers instead write the repeated
`model_n.pt` into a new segment directory, preserving the source bytes but not
fixing the duplicated optimization step.
Model 13,500 has `common_step_counter=324240`; current model 14,000 has
`336264`. With `num_steps_per_env=24`, the expected 500-iteration delta is
12,000 but the observed delta is 12,024: exactly one repeated update. Its actor
hash also changed from `ed816797...17af` to `e50479cd...f330`. The Step1C
diagnostic retains the superseded evidence but accepts only the final
`a63c4822...b0ca2` checkpoint under `stable_v2` names.

Minimal future fix: after `runner.load(...)` in training-only `train.py`, set
`runner.current_learning_iteration += 1`; in every affected wrapper change the
count to
`remaining_iterations=$((target_iteration - latest_iteration))`. The resulting
loop covers `n+1` through `target`, leaving `model_n.pt` unchanged. Regression
proof should hash the loaded checkpoint before/after, require final internal
`iter==target`, and require `common_step_counter` to advance by exactly
`(target-n)*24`. A separate redundant overwrite remains: because every target
is divisible by `save_interval=500`, RSL saves `model_target.pt` in the interval
branch and immediately saves it again in the final-save branch. That second
write performs no PPO update, but strict immutability requires skipping the
final save when the current iteration was already interval-saved.

Any future pass qualifies only exact DadDance checkpoint bytes bound by that
validator, not the two frozen idle clips and not Step1B. Continue screening new
stable checkpoints against the hands-on-back static prerequisite, but do not
wait for iteration 30,000 before building the correct multi-motion expert.
