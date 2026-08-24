# SONIC TRUE23 policy decision memo

## 2026-08-09 PPO update-delta attribution decision (current)

- Reject both components of the full-support PPO actor update. Immutable 2x2
  report SHA is
  `a44fcf7ca73e8bd03e99cb43568659c4a507f2327bdacf1924960f12dc47c1af`.
  Baseline reaches `155` transitions/q9 `163`; hidden-block-only and head-only
  each reach `152`/q9 `160`; the full update reaches `151`/q9 `159`. All four
  terminate on right-wrist `ee_body_pos`.
- Shared q9 `9..158` reward effects versus baseline are `-2.1493` for the last
  hidden block and `-2.1858` for the final head. Combining them adds a further
  `-3.0220` interaction. The interaction is dominated by right-wrist barrier
  cost (`-3.0595`), not action clipping, ABI drift, or a different terminal.
- Decision: do not freeze one module and continue the other; neither isolated
  update is acceptable. Model 1 and both hybrids are diagnostic policies only,
  candidate selection remains null, and no second PPO update is allowed.
- Next test is a no-training symmetric line search along the exact full actor
  delta at alpha `-0.25, 0, 0.25, 0.5, 1`. It must reproduce both endpoints
  and compare all intermediate policies on one shared preterminal q9 support.
  Positive-small improvement indicates excessive step magnitude; negative
  improvement indicates harmful gradient direction; neither authorizes a
  candidate without a later frozen qualification.
- No hardware, gantry, deployment, networking, or robot actuation authorization.

## 2026-08-09 full-support PPO decision

- Reject the one-update full-support derivative. Immutable result SHA is
  `bb0ff6d4e6517174b0cd0a51eab60f958ca4bc6cec5cfbe74ce9e672a4bef30a`.
  The pre-Adam gate genuinely fixes temporal coverage: `20,480` transitions,
  `18,152` first-episode samples, `74` survivors at q9 `163`, and `73` at q9
  `168`. It records `56` `ee_body_pos` deaths and real wrist-barrier signal,
  with zero raw-clip, nonfinite, action-semantic, q9, or anchor faults.
- Exactly one PPO update executes (`8` Adam steps). Update 0 reaches q9 `163`;
  update 1 reaches q9 `159`; both terminate on `ee_body_pos`. The update-1
  policy is internally valid and changes only the declared last hidden block
  and head, but it is not a candidate. Less-negative return does not override
  the four-frame survival regression.
- The missing-late-phase hypothesis is now resolved. The remaining problem is
  the optimization signal: value loss is `362.304`, surrogate change is only
  `0.002381`, and a tiny `0.0617%` actor-state delta crosses the task cliff.
  Do not resume model 1 or permit a second update.
- Paired trace SHA
  `0fb71869eb1249958f7f4137df52b9b0bd92b4666d07c74a614ee11622a878d3`
  reproduces q9 `163/159` and proves the apparent return improvement is an
  early-termination artifact. On shared preterminal q9 `9..158`, model 1 loses
  `7.2195`, dominated by wrist barrier `-4.9980`, worst-EE `-0.9945`, and
  recovery `-0.9020`. Action drift becomes material at q9 `132`; landing force
  diverges at q9 `138`; foot-contact state diverges at q9 `156/158`; model 1
  crosses right-wrist `0.25 m` at q9 `159` while model 0 is still `0.199231 m`.
- Do not change reward, std, critic/advantage, or step size by guess. First run
  a diagnostic-only 2x2 attribution of the update-1 last-hidden-block and
  final-head deltas in memory. No hybrid may be called a candidate without a
  separate frozen qualification contract.
- No hardware, gantry, deployment, networking, or robot actuation authorization.

## 2026-08-09 task-space PPO pilot decision

- Keep the genuine SONIC architecture, but reject the first task-space PPO
  derivative. The immutable v2 pilot result SHA is
  `f78ee6758fd7e947339422a51e1ebb85fe0de5200d9b75bd04ac962847d74d0e`.
  Exact update 0 reaches q9 `163`; update 5 reaches only q9 `133`. Both terminate
  on `ee_body_pos`, all action/q9/nonfinite/raw-clip gates remain clean, and no
  checkpoint is selected.
- Paired trace SHA
  `608e126d61d14225149706d90a876445489982c7d88a49a02088d76f453fb22d`
  proves the cause. Training covered q9 `9..88`; its policy gains `0.0320`
  reward there, then extrapolates badly starting at q9 `90`. Right-foot contact
  is lost at q9 `115`, right-wrist barrier cost starts q9 `126`, and right-wrist
  z error reaches `0.255199 m` at the q9 `133` termination.
- Decision: change temporal coverage, not architecture or teacher data. Build
  one fresh exact-overlay experiment with a 160-step first rollout across 128
  environments. Before Adam, require first-episode late-q9 coverage, exact
  actor/critic/optimizer provenance, real wrist-barrier and terminal reward
  evidence, and sufficient GPU memory. If those gates pass, perform one PPO
  update and accept a simulator candidate only if deterministic evaluation
  reaches at least q9 `164` with no semantic/safety regression and a bounded
  reward floor.
- Do not resume update 5, run another 5x16 schedule, add generic seeds/BONES,
  admit unsupported teacher labels, or unlock more decoder layers.
- No hardware, gantry, deployment, networking, or robot actuation authorization.

## 2026-08-09 genuine SONIC v2 decision (current)

- End-to-end SONIC wiring is valid: `267 -> 64`, `64 + 930 -> 994`, decoder
  `994 -> raw native23`, V2 exactly once. Candidate decoder SHA is
  `011740f86483323fc0f1c39ab25b784cf9411b401e56fee8b7a716664e921ee1`;
  manifest SHA is
  `e43fb5e531fdccda46d2e28ce7a987c8d1d064e64ae58325e6caea4d758240db`.
- The final-affine ridge candidate is rejected as a closed-loop policy. Genuine
  retry-2 SHA
  `0d0142a53e9a4f54012ddac6eab6954ef96895d7148532dbd9386c4f549d0a1c`
  controls 155 actions through q9 `163`, then fails `ee_body_pos` on right-wrist
  z error `0.267852 m`. All action-chain, clipping, finite, joint-limit,
  velocity, and force-hard gates remain clean.
- Cause is covariate shift: teacher-state error at q9 `163` is only `0.02731`
  RMSE, while the diverged student state is far outside the single-trajectory
  training manifold (257/930 policy coordinates outside its range). Wider
  bootstrap-only ridge gives negligible heldout improvement and full-decoder
  fitting remains unjustified.
- Do not call this policy good enough, support-qualified, DAgger-ready, or
  deployable. Do not add generic BONES/seeds or train on unsupported shadow
  labels.
- Cutoff100 recovery is a clean negative result: report SHA
  `14b358ad26e8c5b007696b4e8a89fb25e56c8d1a12d7400f22166ea2d2fb16d3`.
  After 100 student actions, the exact teacher controls 60 actions q9 `109..168`
  and still terminates on `ee_body_pos`. The teacher/action/state seams pass;
  the result admits zero labels and proves that takeover at q9 `109` is too late.
- Cutoff75 also fails physically: report SHA
  `1163f92dc7d3d35614932abb1ba659303c035b536a21ddfb363d5967c5d1b190`.
  Teacher takeover at q9 `84` extends the episode to q9 `183`, but the right
  wrist z error reaches `0.261582 m` against the `0.25 m` gate. All wiring and
  safety semantics remain clean; zero labels are admitted.
- Cutoff50 passes exactly: report SHA
  `8d2089903013721739652d005ae360db6ff94023c91e2c97cff8702f0cd83442`.
  Student q9 `9..58` then teacher q9 `59..518` completes `510/510` to timeout,
  with all action/parity/state/safety gates passing. Final right-wrist z error
  is `0.173643 m`, below the `0.25 m` gate.
- Decision: stop cutoff probing and reject intervention BC. The fixed campaign
  report SHA is
  `be8151713528f2cb9bcbc9b22c1fc8e2b913cfd5a07b6e246a38a112f17829b3`.
  Nominal 0/1 and disturbance 0 pass, but disturbance seed `1542545985`
  terminates at transition `258`, q9 `267`. Exact qvel/action/PT-ONNX seams and
  hard-safety remain clean; soft warnings reach `118`. Campaign result is 3/4
  attempted, 16 unrun by fail-fast, zero labels/rows, collector unauthorized.
- Next path is genuine-SONIC task-space PPO, not more teacher cutoffs, recovery
  labels, generic BONES/seeds, or wider BC. Initialize actor weights from the
  verified model250 checkpoint plus current v2 head; create a fresh critic,
  optimizer, counters, and fixed low std; train only final hidden block and
  output head under exact H10/V2-once environment and deterministic checkpoint
  qualification. Student remains rejected and undeployable meanwhile.
- No hardware, gantry, deployment, networking, or robot actuation authorization.

## 2026-08-09 full-clip and BC decision

- Keep genuine SONIC as final architecture. Selected native124 is a bounded
  simulator teacher only; it is neither parameter warm-start nor final policy.
- Teacher support is local. W0 q9 `9..508` passes. Continuous control fails at
  q9 `876` after `868` actions on root/wrist tracking. W1–w4 phase restarts all
  fail, at q9 `868`, `1054`, `1272`, and `1686`. These failed windows admit zero
  teacher labels. Extra generic seeds/BONES data cannot repair this measured
  support boundary.
- Selected-to-V2 forward-equivalence now judges physically applied safe action
  and hardware target, while retaining strict raw `abs<10` and inverse-domain
  checks. This removes a float32/`atanh` false rejection; it does not project,
  clip, or weaken target safety.
- Exact q9 `9..518` bootstrap is published at NPZ SHA
  `136768fd1595265d9743d5a9e5f7ef38e431de9a57f9ff85246123a7d649f475`
  with manifest SHA
  `5ab761c3b82d62c3a4524f1f195f6f187b91e56b41e854d186339ddcca86f4a1`.
  All `510` rows are teacher-controlled BC candidates only: support `0`,
  on-policy `0`, DAgger `0`, deploy false.
- Train smallest truthful SONIC derivative: encoder unchanged; decoder layers
  0–7 unchanged; deterministic PCA-ridge residual on final affine only. Labels
  are plain raw native23; V2 stays external and executes once. No Adam, native
  weight transplant, invented model250 checkpoint, or resume claim.
- Next decision comes only from closed-loop student control: first action q9 `9`,
  no substituted warmup, `510` actions through q9 `518`, strict safety/history/
  action-semantics checks. Pass yields simulator candidate only. DAgger remains
  blocked until same student states also pass teacher support admission.
- No hardware, gantry, deployment, networking, or actuation authorization.

## Architecture-only scope statement

This memo is architecture and evidence only.
It does not authorize execution, hardware, gantry, or live robot actuation.
It is read-only by design.

## 2026-08-09 reset-seam decision reversal (current)

- Reject the four-ankle PPO derivative as an improvement. Keep it as bounded negative evidence; do not extend it or unlock more rows.
- Cause of the earlier DadDance failure comparison: evaluator inserted two fixed default-target actions at q9 `9` and `10`; training/canonical wrapper used the actor immediately at q9 `9`. This changed robot state, torso history, previous-action history, and every later policy input.
- With zero warmups/action substitution, original selected source and model55 both complete `500/500` to timeout q9 `508`, hard `0`, soft `0`. Reports are SHA `a7abcc5f48070fc22c66446f4f75e00a9394db58a0912a6760ddb073de108954` and `48e12ea0a5692ea737d40e2db3aa701684014a314974d9299003cbb5e8c7b060`.
- Source dominates model55 on reward, ankle position, recovery, torque, V2 projection, RMSE, and force; model55 wins only ankle orientation by `0.58%`. Selected source actor SHA `17302f7076cb480fe4ffc253e7b8228fcbaa033ccb3bf7aac1ed34940b8648ec` is the next simulator teacher candidate.
- Old q9 `58` and q9 `85` failures remain valid negative tests for substituted-warmup robustness. They no longer prove that the true-23 body cannot reproduce nominal DadDance.
- Current proof is one nominal 500-step slice only. Full clip, fixed phases, disturbance, other motions, learner-state support, genuine SONIC, hardware, and deployment remain unqualified.
- Ordered implementation: canonical source PT/ONNX qualifier -> continuous/full-phase source qualification -> teacher-controlled BC bootstrap with explicit behavior-controller identity -> genuine SONIC student evaluation -> support-gated learner-on-policy DAgger. Never represent teacher-controlled bootstrap rows as learner-on-policy rows.
- Canonical qualifier PASS: `artifacts/g1_true23/selected_source_daddance_q9_9_nominal500_seed20260805_v1.json`, SHA `10684c42485a0526783405624526c42cbe903af4805c56a06646f3afd65b149b`. It executes q9 `9..508`, `500/500`, timeout only, hard/soft `0/0`, action-link mismatch `0`; `500` PT/CUDA-to-ONNX/CPU checks have worst error `1.1920928955078125e-06` and zero violations. No training state changes.
- Claim stays narrow: fixed-start nominal slice candidate only, no support-label admission or promotion.

## 2026-08-09 post-sim control (superseded by reset-seam decision above)

- Neutral diagnostic artifact is `artifacts/g1_true23/model21204_sonic_v2_neutral_seed20260805_nominal500.json` with immutable SHA `311691d5e89a86a73a5435a20208b93d0ce9919362402267e1a4b657b17e065e` and is confirmed as **nominal diagnostic only**.
- The neutral diagnostic is a coherent integration with `500/500` records, q9 `11` through final `511`; no termination, discontinuity, nonfinite, raw-clip, action-semantics, target soft-limit, actuator-target soft-limit, measured soft-limit, or velocity failure; minimum height `0.754001259803772 m`; maximum tilt `0.13636365936818398 rad`; maximum tracking RMSE `0.21920073093275602 rad`; and maximum join-plus-CPU-inference latency `9.437081 ms`.
- DadDance-v3 diagnostic is fixed at SHA `aff51f0dc92446025c09196a000aec291b9fcf44576fdffab8e016da8adab4fa`: whole-window quarantine: `47` valid teacher-controlled records followed by transition `47` at `q9=58` terminating on `ee_body_pos`, not `anchor_pos`, `anchor_ori`, or `timeout`; the failed run is not a support pass and admits zero teacher labels.
- Across those `47` records, V2 projection max/mean for right ankle pitch is `0.9860610961914062 / 0.6995989582005967` and left ankle pitch `0.5811665058135986 / 0.42308512551987426`; mean absolute target tracking errors are `0.581019479186928` and `0.6142912316829601` respectively.
- Left knee max projection reached `0.1672501564025879`; that value is only evidence for a gated unlock path, not automatic unlock.
- **Important:** aggregate `ee_body_pos` failure does not identify which end-effector failed.
- Exact action chain is: model21204 raw hardware output -> selected HOME/scale candidate target -> SONIC default/scale conversion and hardware-to-native permutation to plain SONIC raw native -> reject abs >= 10 without clipping -> exactly one operational V11/V2 safe-target application -> safe native action and final hardware target. The selected candidate target never actuates, and comparator-only recomputation is not chained.
- Reference-side recomputation is comparator-only and remains non-actuating; it does not form a second chained transform.
- Wrapper clip is explicitly disabled for this diagnostic pass. Actual raw/safe/processed values are explicitly checked.
- Failure cannot become data through clipping or row masks. Whole-window quarantine/no row masking is mandatory for teacher admission.
- Current bridge state: selected one-input adapter/identity does not generalize to or alter the legacy two-input Native124Policy `obs[1,124] + time_step[1,1]` ABI; no 124-to-994 padding, relabel, or native weight transplant; hash/config/shape/action-order checks fail closed; report write is explicit no-overwrite; no robot run-time, no DDS, no ZMQ, no command-writer, no hardware surface.
- Diagnostic module remains software-only: no robot runtime, no DDS/ZMQ/command writer, no gantry execution surface.
- The technical integration/wiring for offline exact diagnostic is valid, but `model21204`/composite is **not** support-qualified, DAgger-ready, teacher-label-admission-qualified, deployment-ready, or hardware-authorized.
- Disposition rationale: this is not abandoned because neutral proves exact integration; generic seeds/BONES labels were explicitly rejected as the primary path because unsafe, non-representable setpoint semantics cannot repair in-loop V2 projection/contact dynamics, and every failed window remains zero-label quarantine. Targeted in-loop adaptation is the selected path.

## Decision header (historical, superseded)

This section is preserved as historical context; current control follows the 2026-08-09 post-sim section above.

## Decision header (fixed)

This subsection is historical context only and is superseded by the 2026-08-09 post-sim control section.

- The final runtime architecture remains genuine SONIC: encoder267->token64, concatenate token64+H10 proprio930->decoder994->action23.
- `model21204_alpha25` remains a non-actuating **simulator/offline causal native124 tracker** and is not an immediate in-loop teacher without evidence admission.
- It is not a parameter warm-start and not a final teleoperation actor.
- Final unrestricted policy path is genuine causal SONIC with official low-latency initialization and on-policy causal PICO state training.

## Single implementation commitment

Current and only implementation commitment:

- Keep selected base artifacts immutable and do not overwrite them.
- Add a **single hash-bound derivative adaptation scaffold** where the selected one-input NATIVE actor's shared `512/256/128` trunk and non-target final-head rows remain frozen.
- The only rows unfrozen initially are the four final output rows for left/right ankle pitch/roll; left knee is an evidence-gated fifth row only if predeclared ankle-only gates fail while ankle gates pass.
- The remaining 18 native output rows and all other genuine SONIC topology remain frozen.
- Objective is in-loop V2 ON-POLICY tracking optimization on effective actuator torque, foot contact/force/slip, root/anchor/EE/task metrics, not imitation of unrepresentable position targets; no run or training authorization is granted.
- Any derivative gets new artifact/manifest hashes, a new identity, and may not overwrite selected immutable one-input artifacts.
- No training or rollout is authorized in this phase; this commitment is constrained to the diagnostic/software boundary.
- Additional objective gates for any derivative attempt: every candidate window must be exactly `500` steps with strict disjoint heldout; `10/10` nominal + `10/10` declared disturbance/recovery per window; zero termination, anchor, EE, nonfinite, raw-clip, action-semantics, limit, velocity, or contact-gate failures; and whole-window quarantine with fail-closed reporting.

Current committed architecture remains constrained to: genuine final SONIC remains encoder267->token64, concatenate token64+H10 proprio930->decoder994->action23; no 124-to-994 padding, relabel, or native weight transplant.

Each required causal on-policy window record must log:
- synchronized causal PICO/reference inputs
- encoder `267`
- `token64`
- `H10` proprio930
- `decoder994`
- teacher action [23] only when evidence-admitted, and student action [23]
- next state
- termination/safety
- timestamps/freshness
- whole-session splits
- learner-induced states

Teacher admission is evidence-gated. A failed or unsupported window receives **no teacher labels** and is quarantined whole, with no row/frame masking, so DAgger cannot learn known expert failures.

Current GO in this phase is only the in-scope offline adaptation scaffold above.
No checkpoint/composite artifact is teacher-ready, support-qualified, DAgger-ready, or deployment-ready.
Generic Seeds/BONES label-first routes are not first-line and do not replace this path.
Any future branch remains blocked from deployment until this adaptation is separately authorized and evidence-proven.

This section commits order only. The integration itself and every later collection,
evaluation, training, hardware, gantry, deployment, or actuation stage are proposed
future work requiring separate authorization; this memo authorizes none of them.

## Immutable identifiers and evidence anchors

- checkpoint iteration: `21204`
- checkpoint SHA: `9cb0a06db441b8ceb51404b45ba25a81bd4120114aa6b97d6f660cac3f742f81`
- ONNX SHA: `321504108e677fb4b70d1398ff9a20e168def2231eb574e6d8fc1f39385d7b9b`
- model lineage: `model21204_alpha25.pt` is interpolated from pilot `model_21124.pt` and broad `model_21204.pt` with `broad_weight=0.25`

## Evidence lock and what it proves

### Core strict gates (model21204_alpha25, selected artifact)

| Item | Evidence |
|---|---|
| ONNX parity cases | `82` total cases |
| Parity split | `34` adversarial + `48` random |
| Parity max abs | `9.5367431640625e-06` |
| Mean completion | `71.31147540983606` |
| Mean survival | `81.22950819672134` |
| Perfect clips | `23` |
| Original-six retention | `60/60` (`original_six_retention_gate`) |
| Modified references | `30/61` |
| Non-perfect results | `38` entries |
| Zero completion clips | exactly `3` |

The protocol is exactly `61 clips x 10 x 500`.
Mean completion is the percent of 10 runs that complete per clip, then averaged over 61 clips.
Mean survival is the mean percent of the 500-step horizon survived.
Qualification scope is MuJoCo/MJLab clip tracking only.
Live teleoperation is not qualified.

This proves:
- CPU artifact contracts are internally coherent.
- ONNX forward numerical parity is finite and within tolerance.
- Curated MuJoCo clip tracking works at the measured level above.

This does **not** prove:
- causal PICO-to-SONIC generalization,
- on-policy SONIC generalization,
- recovery under disturbance distribution shift,
- final deployment reliability,
- hardware execution validity.

### Zero completion and failure inventory

Zero completion set (exact):
- `J_Dance17_Shuffle`
- `M_Move9`
- `M_ShortMove13`

The full `38` non-perfect clips are those with `ee_body_pos_failures >= 1` in `alpha25_scorecard.json`.
Anchor counters were not a driver:
- `anchor_ori_failures = 0` across all these rows,
- `anchor_pos_failures = 0` across all these rows.

The `38` ee-body-failure clips are exactly:
- `V_PullOver`
- `B_LongDance`
- `B_StretchDance`
- `J_Dance11_Gnarly`
- `J_Dance12_LushLife`
- `J_Dance17_Shuffle`
- `J_Dance18_TikTok`
- `J_Dance19_LetsGO`
- `J_Dance1_Modern`
- `J_Dance21_Blunt`
- `J_Dance2_Salsa`
- `J_Dance3_Woah`
- `J_Dance4_Broadway`
- `J_Dance5_Hype`
- `J_Dance6_Sassy`
- `J_Dance8_WestCoast`
- `J_Dance9_PeaceMaker`
- `J_ShortDance15_Nineties`
- `B_AttackKarate`
- `B_ChopsKarate`
- `B_CrazyChopsKarate`
- `B_ForwardKarate`
- `B_LongKarate`
- `B_SpinKarate`
- `M_Move1`
- `M_Move10`
- `M_Move11`
- `M_Move17`
- `M_Move18`
- `M_Move2`
- `M_Move3`
- `M_Move4`
- `M_Move5`
- `M_Move6`
- `M_Move7`
- `M_Move8`
- `M_Move9`
- `M_ShortMove13`

## Decision rationale (historical; archived, not a current GO line)

Route C is the only primary implementation route. Routes A and B record rejected or subordinate reasoning; they are not parallel next-step options.

| Route | What it is | Evidence status today | Immediate value | Final value | Observation distribution | Decision |
|---|---|---|---|---|---|---|
| A. Direct native124 final (`obs[1,124] -> actions[1,23]`) | Use selected ONNX as final runtime actor | Non-drop-in against current runtime; no final policy evidence | Useful as non-actuating simulator tracker and bounded teacher source when evidence-admitted | Not suitable for final teleop | Learned from curated offline `q_ref`/`qd_ref` plus simulated robot-state context; not causal `267`/`H10`/PICO | Not selected as final |
| B. More native124 training | Additional native offline training + distillation | Requires separate support gating and data coverage work; no claim that Bones-only or current seed closes distribution | Can improve local narrow support and aid immediate diagnostics | Secondary only if teacher support bottleneck is measured and bounded | Same observation/action distribution as direct native124 unless collection changes | Subordinate only |
| C. Genuine SONIC bridge (genuine final) | `267 -> 64 -> 994 -> 23` causal SONIC path with on-policy causal PICO training | Supported by contract files and architecture; requires evidence expansion and hard gates below | Immediate integration is teacher and adapter-only (offline data integration and student-state capture) | The only allowed unrestricted architecture | Matches final causal tensors and learner-state distribution | Selected primary route |

## Architecture and interface mismatch summary

### Selected candidate shape
- Native candidate ONNX signature is single-input.
- Contract is `obs [1,124] -> actions [1,23]`.
- Activation is `ELU`.
- Hidden layers are exactly `512/256/128`.

### Current public runtime signatures (stale)
- Python wrapper (`gear_sonic/utils/g1_23dof_native124_policy.py`) requires two inputs: `obs [1,124]` and `time_step [1,1]`.
- C++ live shadow (`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_true23_live_shadow.cpp`) validates two input names `obs` and `time_step` for native path.
- C++ live shadow pins legacy policy SHA:
  `cc644839807b6ef522e47b3bcb69845843aa345b4fb895847c76642830b5d2b9`.
- These are not aligned for direct selected ONNX swap-in.

### SONIC causal path shape constraints
- Teleop encoder dimension: `TELEOP_ENCODER_INPUT_DIM = 267`.
- Token contract: `64`-value token.
- Proprioception + history for SONIC decoder is `930` (`H10`).
- Combined decoder input is `994`.
- Causal action output dimension is `23`.
- Tokenizer observation is `268` with routing bit outside the encoder.
- Final causal stack is `token64 + H10 proprio930 -> 994 decoder -> 23`.
- Do not route `124` into `994`.
- Never relabel/re-order native weights for SONIC migration.
- Bridge boundary intentionally forbids post-hoc conversion.

### Bridge checks
- The bridge check in `gear_sonic/utils/g1_23dof_mjlab_bridge.py` declares:
  - `conversion_permitted = False`
  - `posthoc_relabeling_forbidden = True`
  - `bridge_decision = blocked_posthoc_conversion`
- That blocks direct migration claims without a full causal path rebuild.

## Handoff contradictions and resolution

- The Aug-6 handoff statement ("no actor through 14k qualifies / intended multi-motion not yet trained") is stale as existence status because iteration `21204` evidence now exists. It remains negative evidence for old motion-specialized DadDance actors and preserves the no-deployment boundary.
- Official SONIC warm-start config is `initialization_only: true` in low-latency warm-start config.
- Handoff operational semantics are inconsistent:
  - capture script hook (not "legacy causal hook") was never registered;
  - the capture script captured `268` tokenizer observation, not decoder `994`;
  - current offline environment reads real future clip frames while the handoff claims semantic change from causal history requires retraining.
- Distillation script is quarantined.
- Any relaxed training thresholds must return to shipped gates before validation/export/promotion/shadow.
- The Aug-6 statement remains valid negative evidence for old motion-specialized DadDance actors and the no-deployment boundary; it is context-bound and does not override the current bridge-to-genuine-SONIC plan.
- Native path currently lacks token/planner/MotionBricks/VLA/GR00T surfaces relative to final policy.

## Offline Bones/seed vs ON-policy, and simulator state responsibility

Existing causal PICO producer runs every `20 ms` with:
- exact `q9` robot-anchor join,
- `q10` control-proprio join,
- history warmup buffers,
and branches to:
- candidate-specific `native124` adapter for non-actuating offline/sim teacher integration, or
- final causal SONIC `267` encoder for student path.

The resulting tuple is:
- raw/stamped PICO terms,
- derived causal encoder `267`,
- `token64`,
- `H10 proprio930`,
- `decoder994`,
- `native124 observation [124]`,
- `frozen teacher action [23]`,
- current student action [23],
- next state,
- termination/safety + timestamps/freshness.
- Do not route `124` to `994`.
- Producer freshness is mandatory and fail-closed on stale/invalid chain input.

### Bones distinction
Offline Bones provides useful kinematic reference diversity only.
It does not supply:
- operator or retargeter noise,
- causal timing,
- learner-induced robot states,
- latency/jitter/dropout/recovery behavior,
- paired actions.

## Minimum causal tranche for progression

At minimum, causal windows for training and qualification must include:
- six families:
  `idle/neutral`, `reach`, `crouch`, `turn`, `walk`, `fast/transitions`
- at least `3` independent headset/reference sessions per family
- `>=1` whole session per family held out
- `>=1` `500-step` window per session
- `10` seeded `500-step` simulator rollouts per window
- this is minimum tranche coverage, not a sufficiency claim

## DAgger and teacher safety policy

This section is historical/superseded context only and does not represent current commitments.

- Offline teacher seed support is limited to the `23` perfect clips.
- No causal window is inherited or admitted automatically.
- Causal windows are collected only after per-window `10/10` nominal and `10/10` declared disturbance/recovery with zero anchor, zero EE, zero limit, zero NaN, zero clamp failure.
- All `38` non-perfect clips remain excluded until requalified under support gate.
- 23 perfect clips are strictly offline seed teacher support.
- On window failure:
  - stop rollout and simulator episode immediately,
  - add no teacher labels from the full failed window or preceding failure neighborhood,
  - do not row-mask.
- Teacher query policy:
  - use frozen teacher only on hash-bound, admitted support and current student states.
- Unsupported states:
  - use reward fallback or separately qualified privileged/WBC/MPC expert.
- Never apply support via partial masking only.
- On any window failure, diagnose identity, adapter correctness, or rollout drift before any data capture path changes.

## Training answer (historical)

This section is historical/superseded. Current objective remains the scoped on-policy adaptation scaffold with no training authorization.

## Ordered implementation / training sequence (historical, read-only archive)

This block is preserved as timeline history and remains superseded by the 2026-08-09 post-sim control commitment above.

### Stage 0
- Freeze current artifact and manifest.
- Candidate-specific one-input adapter can be prepared behind existing causal producer.
- No implementation is executed here; this step is architecture lock.

### Stage 1
- CPU/artifact pass:
  - exact hashes and shapes,
  - ONNX finite checks,
  - `82/82` parity,
  - parity max abs `<= 1e-5`.
- Failure = no-go.

### Stage 2
- Strict sim regression pass:
  - `61 clips x 10 rollouts x 500 steps`,
  - completion `>= 71.31147540983606`,
  - survival `>= 81.22950819672134`,
  - failure count `<= 38`,
  - no new zero-completion clip,
  - perfect `>= 23`,
  - original six `60/60`,
  - no new failing named clip outside the exact recorded `38`.
- If passed, this makes the candidate eligible to request separate collection-only authorization; this memo grants no collection or production authority.
- On failure, failure diagnoses only; no data collection. Diagnose identity, adapter, or rollout drift.

### Stage 3
- After separate authorization, causal tranche and teacher admission gates above must hold before any label use.
- This includes six-family coverage and heldout discipline.

### Stage 4
- Exact SONIC training from official init:
  - evidence-gated BC/distillation,
  - fixed-horizon PPO/DAgger.
- Native weights are never parameter warm-start; official genuine SONIC initialization is required (`initialization_only: true`).
- Initially preserve official encoder prior; because causal-history semantics changed, adapt/unfreeze explicitly scoped later encoder layers if heldout evidence requires it, with retention/safety gates.
- Do not connect this to native weight mapping.
- Pinned lineage required with:
  - exact source/seed config provenance,
  - at least `50` independently recorded optimizer updates,
  - final hash must differ from initialization.
- Stop any round on safety or retention regression.
- Stop campaign as no-go after `2` consecutive rounds with `< 1 pp` heldout completion gain while still below gates.

### Stage 5
- Software/simulation promotion sequence:
  - paired SONIC ONNX shapes and signatures exact,
  - parity `>= 82` cases at `atol=1e-5` and `rtol=1e-5`,
  - retain stage-2 floors and original-six `60/60`,
  - each heldout causal window must pass `10/10` nominal and `10/10` disturbance/recovery at `500`,
  - zero NaN/Inf/joint-limit/anchor/EE/target-clamp failures,
  - every inference `<20 ms`,
  - PICO age <= `100ms`, LowState age <= `40ms`,
  - exact `20ms` continuity and fail-closed behavior.
- Any failure => no-go -> stop, diagnose identity/adapter/rollout drift, then back to data collection + retraining.

### Stage 6
- Even on full pass, this memo remains **no hardware/gantry authorization**.
- No live shadow-to-gantry promotion is authorized in this scope.

## Native124 role in this plan

- More native124 training remains permissible only if causal support bottleneck is measured and bounded.
- Bones-only expansion alone cannot close distribution gap.
- `model21204_alpha25` is a non-actuating simulator/offline tracker only, not an immediate teacher.
- Native124 final policy must stay explicitly subordinate until SONIC support and heldout causal evidence are met.
- Official genuine SONIC path is prioritized for final unrestricted deployment.

## Evidence reversal section (historical archive)

- Choose native successor only if every equivalent Stage-5 causal heldout, safety, retention, parity, and latency numerical gate passes on on-policy student states, and the product drops SONIC token/planner/MotionBricks/VLA/GR00T contract.
- Drop candidate entirely as teacher if it fails window admission or worsens paired student outcomes.
- Allow native-to-SONIC weight warm-start only after a proved semantics-preserving mapping with exact forward/on-policy parity (no `124->994` padding/relabel).
- Reconsider genuine SONIC final route only after repeated controlled exact-SONIC runs on qualified causal data fail while native passes the same gates.
- Replace teacher with a qualified privileged/WBC/MPC expert only if that expert passes the same gates and is better.

## Crisp GO / NO-GO summary (historical archive)

**GO**
- Use one input-selected native124 as offline tracker behind causal producer.
- Keep immediate telemetry contracts strict (latency, freshness, finite, fail-closed).
- These are recommended next separately authorized software tasks/decision gates.
- No DAgger data collection is authorized by this memo.

**NO-GO**
- No direct final native124 deployment.
- No active teleop, no hardware actuation, no gantry deployment.
- No post-hoc relabeling of 124 actor to SONIC.
- Final unrestricted architecture is causal SONIC path only.

## Evidence source list (authorized)

- `tools/codex_claude_gauntlet/runs/20260808T165727Z-sonic-true23-policy-decision/request.json`
  - keys: `task`, `quality_bar`, `scope`.
- `artifacts/g1_native124_multimotion/scaling_all61/feasible_v1/selected_alpha25/selection.json`
  - keys: `checkpoint.sha256`, `onnx.sha256`, `all61_gate`, `qualification_scope`, `reference_feasibility.modified_clips`, `original_six_retention_gate`.
- `artifacts/g1_native124_multimotion/scaling_all61/feasible_v1/selected_alpha25/export_report.json`
  - keys: `parity`, `probe_suite`, `actor_contract`.
- `artifacts/g1_native124_multimotion/scaling_all61/feasible_v1/alpha25_scorecard/scorecard.json`
  - keys: `all.mean_completion_score`, `all.mean_survival_score`, `results[].clip`, `results[].ee_body_pos_failures`.
- `artifacts/g1_native124_multimotion/scaling_all61/feasible_v1/old_protocol_core_alpha25/summary.json`
  - keys: `clip_count=6`, `evaluated_count=6`, `perfect_count=6`, `results[].failure_counts=0`.
- `gear_sonic/utils/g1_23dof_native124_policy.py` (line refs `28-39,110-140,210-229`)
- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_g1_23dof_rev_1_0_low_latency_warm_start.yaml` (line refs `2-4,14-18`)
- `gear_sonic/envs/mjlab/sonic_true23.py` (line refs `1-11,41-67,515-544,981-1007`)
- `gear_sonic/utils/g1_23dof_mjlab_bridge.py` (line refs `1-10,280-333`)
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_true23_live_shadow.cpp` (line refs `3-9,49-59,398-466,1030-1035,1377-1543`)
- `G1_TRUE23_TELEOP_HANDOFF.md` (line refs `3-38,103-107,965-982,999-1113`) for historical context only.

## Bottom-line constraint

No architecture in this memo authorizes live hardware, hardware execution, or gantry movement.
GO means recommended next authorized software work only.
This memo is read-only; proposed future training and rollout paths are proposals, not executed or authorized actions here.
All paths remain scoped to offline/simulator software proof, not deployment.
