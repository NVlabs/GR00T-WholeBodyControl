# SONIC VLA action pipeline note

## Question from checklist

Checklist item: inspect the VLA pipeline in the SONIC repo and answer:

- how the pipeline outputs motion-token and hand-joint actions from the same VLA backbone;
- whether the output is discrete and/or continuous;
- whether there is an MLP mapping mechanism.

## Short answer

In the current `GR00T-WholeBodyControl` + `Isaac-GR00T` code path, the VLA does **not** expose a categorical/discrete-token output head. It predicts a continuous action chunk for the `unitree_g1_sonic` embodiment. That chunk is split into:

- `motion_token`: 64 floats in SONIC latent token space;
- `left_hand_joints`: 7 floats;
- `right_hand_joints`: 7 floats.

So the deployed action vector is 78 continuous values per step over a 40-step action horizon. The “token” terminology is real for SONIC’s controller-side latent space and FSQ-based action-transform module, but the VLA bridge passes float vectors, not token IDs or categorical logits. Meeting-safe phrasing: **latent token space**.

## Repository snapshots used

- `GR00T-WholeBodyControl`: branch `analysis/sonic-vla-pipeline`, commit `0a87181c9106d0e49293400714b157676e0ec664`.
- `Isaac-GR00T`: branch `main`, commit `3df8b3825d67f755e69141446f4315f281b9b7e6`.

## End-to-end action flow

```text
camera + robot state
  -> GR00T-WholeBodyControl run_vla_inference.py
  -> Isaac-GR00T PolicyClient / PolicyServer
  -> GR00T N1.7 backbone + diffusion action head
  -> continuous action_pred [B, 40, max_action_dim]
  -> processor splits by unitree_g1_sonic modality keys
  -> action.motion_token / action.left_hand_joints / action.right_hand_joints
  -> SONIC VLA client strips prefixes and publishes ZMQ pose protocol v4
  -> C++ SONIC deploy consumes token_state + hand joints
```

The public docs describe this same deployment shape: the VLA inference guide says the pipeline consists of an Isaac-GR00T `PolicyServer`, the Python VLA inference client, C++ deploy, camera server, and optional exporter (`docs/source/tutorials/vla_inference.md:6-15`). It also states the SONIC embodiment uses 64D motion token + 7D left hand + 7D right hand (`docs/source/tutorials/vla_inference.md:77-80`). The workflow guide states the VLA operates at 2.5 Hz and SONIC decodes at 50 Hz, with a 64D latent token over a 40-step chunk (`docs/source/tutorials/vla_workflow.md:21-41`).

## Action schema and dimensions

`Isaac-GR00T` defines the `unitree_g1_sonic` action modality as one 40-step action chunk with three modality slices: `motion_token`, `left_hand_joints`, and `right_hand_joints` (`gr00t/configs/data/embodiment_configs.py:67-113`).

`GR00T-WholeBodyControl` defines the dataset/export feature schema:

| Field | Source key | Dim | Dtype | Citation |
|---|---:|---:|---:|---|
| `motion_token` | `action.motion_token` | 64 | `float64` in dataset schema | `gear_sonic/data/features_sonic_vla.py:101-111`, `264-268` |
| `left_hand_joints` | `teleop.left_hand_joints` | 7 | `float32` | `gear_sonic/data/features_sonic_vla.py:134-138`, `296-300` |
| `right_hand_joints` | `teleop.right_hand_joints` | 7 | `float32` | `gear_sonic/data/features_sonic_vla.py:139-143`, `301-304` |

The inference transport casts the deployed values to float32. `pack_latent_action_message()` documents and enforces shape `[64]` for `motion_token` and `[7]` for each hand; it stores the motion vector under `token_state` and publishes `pack_pose_message(..., version=4)` (`gear_sonic/scripts/run_vla_inference.py:126-184`).

## Where the same VLA backbone/action head produces all outputs

In `Isaac-GR00T`, the SONIC embodiment does not create separate model heads for motion-token and hand-joint groups. The processor concatenates normalized action groups in modality-config order during training (`gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py:508-547`) and splits the predicted action tensor back into groups during inference/decode (`gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py:292-314`, `316-353`).

The model path is:

- the GR00T model prepares backbone and action-head inputs, runs the VLM backbone, then calls the action head (`gr00t/model/gr00t_n1d7/gr00t_n1d7.py:539-600`);
- the action head creates a continuous action tensor initialized from Gaussian noise with shape `(batch_size, action_horizon, action_dim)` (`gr00t/model/gr00t_n1d7/gr00t_n1d7.py:332-339`);
- each denoising step embeds the current continuous actions, conditions on vision/language/state features through DiT, decodes predicted velocity, and updates the continuous action tensor by Euler integration (`gr00t/model/gr00t_n1d7/gr00t_n1d7.py:382-425`).

There is therefore one continuous trajectory tensor; `motion_token` and hand joints are slices of it. The split is schema-driven, not separate output heads.

## What happens in `run_vla_inference.py`

The Python inference client:

1. Reads camera and robot state, constructs `video`, `state`, language, and `q`, then calls `prepare_observation_for_eval()` (`gear_sonic/scripts/run_vla_inference.py:206-269`).
2. `prepare_observation_for_eval()` splits the whole-body `q` vector into state groups: legs, waist, arms, and hands (`gear_sonic/utils/inference/vla_utils.py:32-62`).
3. Calls `policy.get_action(observation)` and strips any task-progress fields (`gear_sonic/scripts/run_vla_inference.py:272-294`).
4. Checks the motion-token bound by `abs(action[motion_key]).max() > 1.25`, then calls `concat_action()` (`gear_sonic/scripts/run_vla_inference.py:284-294`). `concat_action()` only strips the `action.` prefix; it does not transform or discretize the values (`gear_sonic/utils/inference/vla_utils.py:14-29`).
5. During the 50 Hz publish loop, extracts `motion_token`, `left_hand_joints`, and `right_hand_joints`, indexes the current horizon step, and publishes them via ZMQ (`gear_sonic/scripts/run_vla_inference.py:623-664`).

This is the critical bridge evidence: after `PolicyClient` returns, the bridge treats all fields as float arrays. No categorical decode appears in this file.

## Data collection/export side

The exporter records the same schema it later fine-tunes on:

- `action.motion_token` is copied from C++/proprio `token_state`, defaulting to zeros if absent (`gear_sonic/scripts/run_data_exporter.py:662-665`).
- `teleop.left_hand_joints` and `teleop.right_hand_joints` are copied from the current hand message, defaulting to 7D zeros (`gear_sonic/scripts/run_data_exporter.py:751-767`).

That means the training data teaches GR00T to reproduce a continuous SONIC latent vector plus continuous hand joints, not token class IDs.

## MLP mapping mechanisms

There are two different “MLP mapping” mechanisms, and conflating them is the trap.

### 1. SONIC controller-side Action Transform Module

SONIC’s `UniversalTokenModule` is explicitly an encoder → FSQ quantizer → decoder module. Its docstring says multiple encoders (`g1`, `smpl`, `teleop`) map different inputs into a shared latent space, optional FSQ discretizes the latent, and decoders map tokens plus proprioception back to actions (`gear_sonic/trl/modules/universal_token_modules.py:33-67`). The implementation sets `token_dim = num_fsq_levels` and `token_total_dim = token_dim * max_num_tokens` (`gear_sonic/trl/modules/universal_token_modules.py:219-240`). The `all_mlp_v1` config sets `num_fsq_levels: 32` and `max_num_tokens: 2`, hence a flat 64D latent, and uses MLP encoders/decoders (`gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml:3-13`, `29-63`).

The forward path quantizes encoder latents when a quantizer exists (`gear_sonic/trl/modules/universal_token_modules.py:681-690`), optionally adds residuals pre/post quantization, assembles tokens, flattens them, and decodes actions (`gear_sonic/trl/modules/universal_token_modules.py:741-906`).

Interpretation: SONIC has MLP encoders/decoders around an FSQ latent token bottleneck. This explains why the 64D `motion_token` belongs to a latent token space.

### 2. GR00T action-head projections

GR00T uses embodiment-conditioned MLP projections around the diffusion action model:

- `state_encoder = CategorySpecificMLP(...)`
- `action_encoder = MultiEmbodimentActionEncoder(...)`
- `action_decoder = CategorySpecificMLP(..., output_dim=action_dim)`

These are instantiated in `Gr00tN1d7ActionHead` (`gr00t/model/gr00t_n1d7/gr00t_n1d7.py:66-82`). `CategorySpecificMLP` is a two-layer category-specific MLP (`gr00t/model/modules/embodiment_conditioned_mlp.py:143-161`). `MultiEmbodimentActionEncoder` maps the continuous action tensor and timestep to action features via category-specific linear layers and sinusoidal time encoding (`gr00t/model/modules/embodiment_conditioned_mlp.py:177-225`).

Interpretation: this is not a standalone “backbone embedding → discrete token ID” MLP. It is the standard GR00T action-head machinery: continuous action trajectories are embedded for diffusion and decoded back to continuous `action_dim` outputs.

## Discrete vs continuous boundary

What is discrete/quantized:

- SONIC’s internal universal-token learning can quantize an encoder latent with FSQ (`gear_sonic/trl/modules/universal_token_modules.py:33-67`, `681-690`).
- The SONIC paper also describes MLP encoders mapping commands into a shared latent space, FSQ quantization, and MLP decoders back to motor commands.

What is continuous in the VLA bridge:

- The dataset schema stores `action.motion_token` as a 64-vector, not an ID (`gear_sonic/data/features_sonic_vla.py:264-268`).
- The inference bridge casts `motion_token`, `left_hand_joints`, and `right_hand_joints` to `np.float32` and publishes them (`gear_sonic/scripts/run_vla_inference.py:148-184`, `623-664`).
- GR00T predicts a continuous action tensor via diffusion/Euler integration (`gr00t/model/gr00t_n1d7/gr00t_n1d7.py:332-425`).

So: say “the VLA outputs continuous values in SONIC latent token space plus continuous hand joints.” Avoid “the VLA outputs discrete tokens” unless we inspect a checkpoint or branch that adds categorical token heads.

## Answer to checklist wording

> how they output discrete + continuous hand tokens from the same backbone

A concise phrasing is:

> The current SONIC VLA path uses one GR00T continuous action head conditioned by the VLM backbone. For `unitree_g1_sonic`, the action schema slices the generated trajectory into a 64D SONIC latent motion-token vector and two 7D hand-joint vectors. The motion vector lives in SONIC’s latent token space, but the bridge passes it as floats; the hand outputs are continuous joint commands, not hand tokens.

> is there an MLP mapping mechanism?

Yes, but not as a categorical-token mapper from the VLA backbone:

- SONIC has MLP encoders/decoders plus FSQ in its controller-side universal token module.
- GR00T has embodiment-conditioned MLPs for state/action projection and action decoding in the diffusion head.
- No code path was found where the VLA backbone directly feeds an MLP classifier/regressor that emits discrete token IDs. The deployed VLA output is continuous and schema-split.

## Open checks before making stronger claims

- Inspect the exact fine-tuned checkpoint processor/statistics used by the team. The code says what the path supports; checkpoint metadata confirms what was trained.
- If someone refers to older GR00T N1.5 paper/demo behavior, note that the paper describes VLA outputting teleoperation-format control signals into a kinematic planner. The current repo path has a latent-action VLA bridge that publishes `motion_token` directly.
- If the team wants discrete token outputs, that would require an explicit categorical/FSQ-index interface or a quantization/dequantization adapter. No such adapter was found in the current VLA inference path.
