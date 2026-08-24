# SONIC True23 Stage 1 Decision Memo

## 2026-08-09 Stage-1 PPO update-delta attribution (current)

- The non-training 2x2 actor ablation is complete at report SHA
  `a44fcf7ca73e8bd03e99cb43568659c4a507f2327bdacf1924960f12dc47c1af`.
  Exact policies are baseline, update-1 last-hidden-block only, update-1 final
  head only, and full update 1. Their deterministic results are respectively
  `155/q163`, `152/q160`, `152/q160`, and `151/q159`; all fail right-wrist
  `ee_body_pos`.
- On common nonterminal q9 `9..158`, isolated block and head effects are
  `-2.1493` and `-2.1858` weighted reward, with an additional `-3.0220`
  block-head interaction. Both trainable pieces independently move the policy
  toward the same wrist cliff. Neither is a Stage-1 candidate.
- Stage 1 permits no Adam step, checkpoint mutation, or module-specific
  continuation. The next bounded diagnostic is an in-memory symmetric scalar
  interpolation of the exact full actor delta at alpha
  `-0.25, 0, 0.25, 0.5, 1`, using fresh deterministic environments and strict
  endpoint/provenance/action/reward/terminal gates. Its only purpose is to
  distinguish excessive update magnitude from a harmful gradient direction.
- Hardware, support, promotion, deployment, networking, and actuation remain blocked.

## 2026-08-09 full-support PPO Stage-1 result

- The authorized 128-environment, 160-step experiment completed exactly once.
  Result SHA is
  `bb0ff6d4e6517174b0cd0a51eab60f958ca4bc6cec5cfbe74ce9e672a4bef30a`;
  rollout-evidence SHA is
  `66b72261e160df34cd9defca27eb229068b0efce0b1b27f7183669f2f81e9bc6`.
- Pre-Adam evidence passes: `20,480` total transitions, `18,152` first-episode
  samples, first-generation survivor counts `128/96/94/76/74/73` at q9
  `88/126/133/154/163/168`, real wrist-barrier/death signal, exact frozen
  actor/critic-MLP state, critic-normalizer count `20,480`, and zero semantic or
  finite faults. The RTX 3070 memory guard had `6.78 GB` free.
- Exactly one update and `8` Adam steps execute. Update-1 checkpoint SHA is
  `10ae7bb7b67133c9b1c1330ea4093715f63531636291902d4b7212573d43d959`;
  frozen tensors/std remain exact and all 12 Adam states are at step `8`.
- Deterministic update 1 is rejected: `151` transitions/q9 `159` versus update
  0 at `155`/q9 `163`, both `ee_body_pos`. Candidate is null; Stage 1 remains
  unqualified. The experiment proves coverage was supplied, not that PPO found
  a better policy.
- Do not run update 2. Paired trace SHA
  `0fb71869eb1249958f7f4137df52b9b0bd92b4666d07c74a614ee11622a878d3`
  reproduces q9 `163/159` and shows model 1 is worse by `7.2195` on common
  preterminal q9 `9..158`; its less-negative whole-episode return is only early
  truncation. Material action divergence starts q9 `132`, a new `107.27 N`
  landing event appears q9 `138`, contact state diverges q9 `156/158`, and the
  right wrist terminates at `0.258851 m` on q9 `159` while model 0 is
  `0.199231 m` there.
- Stage 1 now permits only non-training attribution of the last hidden block
  versus final head. Any later optimizer/reward/std/critic change requires a
  new frozen contract based on that attribution.
- Hardware, support, promotion, deployment, networking, and actuation remain blocked.

## 2026-08-09 task-space PPO temporal-support decision

- Stage 1 PPO update 5 is rejected and must not be resumed. Immutable pilot
  result SHA is
  `f78ee6758fd7e947339422a51e1ebb85fe0de5200d9b75bd04ac962847d74d0e`:
  update 0 completes `155` transitions to q9 `163`; update 5 completes `125` to
  q9 `133`; both terminate on `ee_body_pos`; no candidate is admitted.
- Immutable paired trace SHA
  `608e126d61d14225149706d90a876445489982c7d88a49a02088d76f453fb22d`
  separates objective behavior from harness error. The old rollout schedule
  supplies no samples beyond q9 `88`. Update 5 improves the observed q9 `9..88`
  prefix slightly, but action drift crosses `0.25` immediately outside support
  at q9 `90`, contact/EE drift is material by q9 `115`, wrist-barrier cost starts
  q9 `126`, and failure occurs q9 `133`.
- Stage 1 therefore permits one new bounded simulator experiment only: exact
  update-0 actor and critic, fresh Adam/counters, one 128x160 rollout, a strict
  pre-Adam first-episode coverage/state/reward/memory gate, exactly one PPO
  update, then deterministic update-1 evaluation. The update must not occur if
  the late-phase coverage proof fails. A one-frame-or-better deterministic
  improvement is a simulator candidate only, never support or deployment.
- Teacher intervention, recovery BC, update-5 resume, wider decoder training,
  and hardware execution remain blocked.
- No hardware, gantry, deployment, networking, or robot actuation authorization.

## 2026-08-09 genuine SONIC closed-loop superseding result

- The deterministic final-affine BC candidate is not Stage-1-qualified.
  Decoder SHA
  `011740f86483323fc0f1c39ab25b784cf9411b401e56fee8b7a716664e921ee1`
  executes 155 genuine SONIC actions q9 `9..163`, then terminates on right-wrist
  `ee_body_pos`; retry-2 report SHA is
  `0d0142a53e9a4f54012ddac6eab6954ef96895d7148532dbd9386c4f549d0a1c`.
- Integration and safety semantics pass up to termination: no action mismatch,
  clipping, nonfinite value, soft-limit breach, velocity-limit breach, or hard
  force violation. The failure is accumulated off-manifold state drift, not a
  23-DoF structural impossibility and not a V2 double application.
- Stage 1 remains no-training/no-label-admission. Cutoff100 is quarantined at
  report SHA `14b358ad26e8c5b007696b4e8a89fb25e56c8d1a12d7400f22166ea2d2fb16d3`:
  student q9 `9..108`, then selected teacher q9 `109..168`, ending on
  `ee_body_pos`. PT/ONNX parity, V2-once action semantics, causal selected124
  state, and hard-safety gates pass; zero labels were published.
- Cutoff75 is quarantined too: report SHA
  `1163f92dc7d3d35614932abb1ba659303c035b536a21ddfb363d5967c5d1b190`.
  After student q9 `9..83`, the selected teacher controls q9 `84..183`; the
  right wrist then breaches z error at `0.261582 m`. All action/parity/state and
  hard-safety gates remain clean, and no labels are admitted.
- Cutoff50 passes: report SHA
  `8d2089903013721739652d005ae360db6ff94023c91e2c97cff8702f0cd83442`.
  Student q9 `9..58` and selected teacher q9 `59..518` complete `510/510` to
  timeout, with zero required failure counts and final right-wrist z error
  `0.173643 m` versus `0.25 m`.
- Stage 1 recovery campaign is now failed and quarantined. Immutable report SHA
  is `be8151713528f2cb9bcbc9b22c1fc8e2b913cfd5a07b6e246a38a112f17829b3`.
  Nominal seeds 0/1 and disturbance seed 0 pass; disturbance seed `1542545985`
  terminates at transition `258`, q9 `267`, with exact qvel/action/parity seams,
  hard safety `0`, soft warnings `118`, and zero published labels/rows. The
  remaining 16 runs are intentionally unexecuted by fail-fast. Recovery suffix
  collection, BC admission, DAgger, promotion, and deployment remain blocked.
- Stop teacher cutoffs and intervention BC. The next bounded Stage-1 experiment
  is genuine-SONIC task-space PPO from verified model250 actor weights overlaid
  with the current v2 head, fresh critic/Adam/counters, frozen encoder and early
  decoder, trainable final hidden block plus all 23 output rows, V2 exactly once,
  and deterministic checkpoint qualification. This authorizes implementation
  and simulator training only, never hardware or deployment.
- A passing diagnostic does not itself authorize training. It only permits a
  new recovery-window contract and qualification cohort. Existing support-v1
  and six-family tranche rules must not be weakened or misapplied.
- No hardware, gantry, deployment, networking, or robot actuation authorization.

## 2026-08-09 superseding full-clip/BC result

- Stop native ankle PPO permanently for this branch. Selected source is useful
  only on initial fixed-start DadDance slice; continuous teacher fails at q9
  `876`, and all four later fixed-phase restart shards fail. Failure is chiefly
  wrist/root drift, not an ankle-z or hard-safety breach.
- Admit one immutable teacher-controlled bootstrap only: `510` rows q9 `9..518`,
  NPZ SHA
  `136768fd1595265d9743d5a9e5f7ef38e431de9a57f9ff85246123a7d649f475`,
  manifest SHA
  `5ab761c3b82d62c3a4524f1f195f6f187b91e56b41e854d186339ddcca86f4a1`.
  Admission: BC `510`; support/on-policy/DAgger `0`; deployment false.
- Supersede earlier residual/CEM/PPO proposal with deterministic genuine-SONIC
  head adaptation. Freeze encoder and decoder layers 0–7. Fit final affine only
  using blocked, purged PCA-ridge residual against plain raw native23 labels.
  Keep V2 outside decoder and apply once.
- Candidate is weights-only/offline-BC lineage. Source model250 PT, optimizer,
  critic, std, and resume state are absent and must not be invented.
- Required next evidence: exact ONNX ABI/frozen-trunk/parity; then a q9 `9`
  no-warmup genuine SONIC rollout controlling all `510` actions. Any termination,
  action clipping, nonfinite value, limit/history mismatch, or failed safety gate
  quarantines run and blocks DAgger.
- No hardware, gantry, deployment, networking, or actuation authorization.

## 2026-08-09 superseding reset-seam result

- Prior Stage-1 diagnostic plan is superseded. No CEM/impulse diagnostic or more PPO is next.
- Training acted first at q9 `9`. Old deterministic evaluator substituted two default-target actions and first invoked the actor at q9 `11`. Its shared q9 `85` failure family was an evaluator reset-seam mismatch, not a valid capacity comparison.
- Exact no-update q9 `9` reports:
  - source: SHA `a7abcc5f48070fc22c66446f4f75e00a9394db58a0912a6760ddb073de108954`;
  - model55: SHA `48e12ea0a5692ea737d40e2db3aa701684014a314974d9299003cbb5e8c7b060`.
- Both run `500/500` to timeout q9 `508`, hard `0`, soft `0`. Original selected source is quantitatively cleaner; reject model55 as an improvement and stop ankle PPO.
- Next candidate is immutable selected source actor SHA `17302f7076cb480fe4ffc253e7b8228fcbaa033ccb3bf7aac1ed34940b8648ec`, checkpoint SHA `9cb0a06db441b8ceb51404b45ba25a81bd4120114aa6b97d6f660cac3f742f81`, ONNX SHA `321504108e677fb4b70d1398ff9a20e168def2231eb574e6d8fc1f39385d7b9b`.
- Required next gates: canonical 500-step PT/ONNX parity, continuous full-clip coverage, fixed-phase restart evidence, teacher-controlled bootstrap kept distinct from learner-on-policy DAgger, then genuine SONIC `267 -> 64 -> 994 -> 23` evaluation.
- No hardware, gantry, deployment, networking, or actuation authorization.
- Canonical source qualifier now passes: report SHA `10684c42485a0526783405624526c42cbe903af4805c56a06646f3afd65b149b`; q9 `9..508`; `500/500`; timeout only; hard/soft `0/0`; action semantics mismatch `0`; `500` checkpoint-vs-ONNX checks with worst absolute error `1.1920928955078125e-06`; frozen actor/critic/optimizer proof passes. This is one nominal slice, not full-clip or teacher-support qualification.

## Historical facts (superseded where inconsistent above)

- Primary decision: do not repeat, extend, or tune PPO.
  Next action is exactly one bounded simulator-only boundary-causality diagnostic.

- Approved training architecture is **support-gated SONIC DAgger only**.
  This memo does not authorize any diagnostic execution or training execution.

- Source checkpoint SHA-256: `9cb0a06db441b8ceb51404b45ba25a81bd4120114aa6b97d6f660cac3f742f81`
  ONNX SHA-256: `321504108e677fb4b70d1398ff9a20e168def2231eb574e6d8fc1f39385d7b9b`

- Clean peak run evidence: 60 iterations, 720/720 Adam steps, exact checkpoints
  `initial_model.pt`, `model_0.pt`, `model_5.pt`, `model_10.pt`, `model_15.pt`, `model_20.pt`, `model_25.pt`, `model_30.pt`, `model_35.pt`, `model_40.pt`, `model_45.pt`, `model_50.pt`, `model_55.pt`, `model_59.pt`.

- Trainable rows only: 4, 5, 10, 11 (mapped to four ankles).
  Source std frozen for these rows: row4≈0.2859, row5≈0.3193, row10≈0.3116, row11≈0.3479.
  Trunk, normalizer, std, and all other output rows are frozen.

- Deterministic table (exact): for models 45, 50, 55, 59: `passed=false`, `74 completed nonterminal policy transitions`, first `done` at `q9=85`, `ee_body_pos` goal, zero hard-safety violations, each no candidate.
  Trainable L2 deltas may be used only as context: `0.0099597`, `0.0117119`, `0.0119158`, `0.0114532` (explicitly not action-space magnitudes).

- PPO peak near iterations 52–55 is a **non-qualifying** pre-update stochastic rollout artifact:
  - TensorBoard episode stats are rollout averages across episodes and policies.
  - Saved checkpoint is post-update.
  - Deterministic evaluation calls `stochastic_output=False`.
  - Therefore deterministic reports are authoritative and these stochastic peaks cannot qualify a checkpoint.

- Prior, separate `model99` localization findings:
  - `q9=86`, `ee_body_pos`; right wrist abs z error `0.257398m` (over 0.25m).
  - Ankle z errors: `0.061218m`, `0.057810m`.
  - Right ankle pitch projection: `1.4758rad` terminal, `1.5555rad` largest.
  - Left-knee row3 gate false; keep frozen.
  - Do **not** conflate with the clean peak run.

- Proved via this evidence:
  1. Equal deterministic family boundary.
  2. Integration checks and changed-row constraints.
  3. Hash/parity/frozen-row/determinism checks.

- Not proved:
  - Inadequate action-mean update.
  - Exploration/mean mismatch.
  - Insufficient four-row authority.
  - Specific implementation bug remains possible; not eliminated, but weakened by invariants above.

- Rejected alternatives (adversarial compare/reject): identical PPO; LR/variance sweep; left-knee unlock; ungated row expansion; task-space residual.

- Prioritized diagnostic (exact and bounded):
  - <=5,600 simulated transitions.
  - No Adam steps, no parameter updates.
  - One environment.
  - Sources: baseline source + source model checkpoints 45, 50, 55, 59.
  - Fixed env seed: `20260805`.
  - Capture `model59` pre-action states / 124-D observations for `q9=82,83,84,85`.
  - A) 16 deterministic forward pairs (4 checkpoints × 4 observations): finite/repeatable means + source-normalized mean movement.
  - B) At `q9=82`, baseline plus ±0.05 raw-action impulse per row 4,5,10,11 (9 one-step branches). For each branch: intended row mapping/sign response `>=1e-4 rad`, and every unintended target change `<=1e-6 rad`; otherwise implementation=no-go.
  - C) From identical `q9=82` snapshot: four deterministic branches; 32 masked-stochastic `model59` branches using Torch seeds `20260805`…`20260836`; and constrained four-row CEM branches with seeds `20260805`, `20260806`, `20260807`, population 64, 3 generations, 8 elites, horizon `q9=82..90`.

- Collected metrics:
  first termination/q9/name; all four EE z errors; projection; action/target row isolation; hard/soft counts; max force/velocity ratios; stochastic safe-success count; CEM safe-success by seed; source-normalized mean displacement; oracle-direction progress
  `rho = dot(mu59 - mu0, a* - mu0) / ||a* - mu0||^2` over 4 captured states.

- Safe-branch acceptance:
  survives all `q9=82..90` transitions, no termination, zero hard violations, zero soft-margin/saturation violations, all finite, max force and velocity ratios `<=0.8`, wrist abs-z `<=0.20m`, ankle abs-z `<=0.10m`, projection `<=0.05rad`, no clipping.

- Classification gates:
  - Integrity failure => implementation branch, stop and fix, repeat same diagnostic.
  - Four-row authority GO only if a CEM safe branch exists for 3/3 seeds.
  - 0/3 seeds => insufficient-authority evidence.
  - 1-2/3 seeds => inconclusive.
  - Both outcomes: no training.
  - Exploration/mean mismatch only if `>=8/32` stochastic safe successes while all four deterministic means fail.
  - Inadequate update-magnitude only if authority GO plus stochastic `<8/32` and `rho < 0.25` on at least 3/4 states.
  - Otherwise unresolved/no training.

- Finite CEM failure is evidence, not mathematical proof.

- Qualification gates required after any future DAgger phase:
  hash/row/fallback invariants; full deterministic 500-step DadDance episode including two warmups under existing evaluator; zero hard safety, soft warning pass, finite, deterministic replay/RNG pass, no non-timeout termination, and boundary margins above threshold.
  Simulator candidates only.

- Teacher label admissibility:
  candidate must be causal, learner-state, direct true23 coordinates, inside support, final safe target within `1e-6 rad`, no clipping/projection.

- Reject teacher label families: projected 29->23/Bones, generic Bones data/seeds, no-causal labels.

- Explicit stop conditions and quarantine: no identical PPO, no extra generic seeds, no LR/std tuning, no left-knee unlock, no row expansion, no generic Bones, no unsafe projection.

- No result from this memo authorizes promotion, deployment, hardware use, gantry activity, networking, or actuation.

## Hypotheses

- Primary unresolved hypothesis set:
  - Implementation branch active if diagnostic integrity checks fail.
  - Causal-supervised deficiency or authority gap if evidence in Diagnostic gates indicates authority fail or insufficient safe-stochastic support.
  - Exploration/mean mismatch only if stochastic-safe success pattern is high while deterministic mean families fail.
  - Inadequate update magnitude only if authority GO plus low `rho` criteria are met.

- If no diagnostic criteria are satisfied: status remains unresolved, no-training action is taken.

## Stage-1 Authorized Training Architecture (only if diagnostic authority GO passes)

- Policy form:
  `mu = mu_source + g(o) * M * r_theta(o)`
- Source checkpoint, normalizer, trunk, std rows, and unapproved rows remain frozen.
- `g` is hard support gate with exact source fallback outside support.
- `M` is diagnostic-authorized rows only.
- Labels are direct true23 action labels from safe constrained oracle.
- Supervision is deterministic, on learner states, no PPO/stochastic qualification.
- Four-row authority PASS keeps `M={4,5,10,11}`.
- Authority FAIL does not authorize row expansion; any future expansion requires separate row-authority evidence and keeps left knee excluded until it passes its existing criterion.
- Bug/inconclusive outcomes keep no training in effect.
