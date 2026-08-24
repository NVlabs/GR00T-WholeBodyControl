# MJLab/RSL-RL checkpoint boundary for SONIC true23

A stock MJLab G1 checkpoint is teacher or diagnostic material. It is not a
SONIC `*.promotion.pt` checkpoint and must not be relabelled as one.

The formats differ materially:

- Stock MJLab G1 uses an RSL-RL normalized, monolithic observation-to-action
  MLP. Current upstream G1 configurations use hidden widths 512, 256, and 128
  with ELU activation.
- SONIC true23 requires a separate 267-input teleoperation encoder, an exact
  FSQ token contract, ten frames of padded-29 proprioception history, and a
  994-input/23-output SiLU decoder.
- RSL-RL checkpoints save actor, critic, optimizer, iteration, and optional
  information. An iteration number alone does not prove which initial weights,
  dataset, task configuration, source tree, joint order, or observation
  contract produced the final weights.
- Upstream RSL-RL exports one observation-to-action ONNX model at opset 18.
  The SONIC candidate is a hash-bound static float32 encoder/decoder pair at
  opset 13.

Use the safe diagnostic audit:

```powershell
python -m gear_sonic.scripts.inspect_g1_23dof_mjlab_checkpoint `
  --checkpoint <model_iteration.pt> `
  --output <audit.json>
```

The audit loads with `torch.load(..., weights_only=True)`, reports architecture
compatibility, and never writes a promotion checkpoint.

## Truthful bridge

A promotable MJLab path must be designed into training before the first PPO
learning iteration. A custom runner must train the exact SONIC
encoder/FSQ/H10/true23 decoder contract and record:

- the approved released warm start and exact initial policy-state hash;
- at least 50 completed outer PPO iterations (successful `alg.update()` calls);
- a final policy-state hash different from initialization;
- the resolved MJLab task and RSL-RL configuration;
- pinned MJLab, RSL-RL, task, runner, robot-asset, and source hashes;
- exact source and processed motion-dataset manifests;
- explicit 23-joint observation and action semantic ordering.

One outer PPO iteration contains multiple epoch/minibatch optimizer steps.
Checkpoint `update_count` records outer iterations, not those internal steps.

That runner needs a distinct externally-trained evidence schema and a reviewed,
checked-in approval. Only then may it emit a weights-only promotion checkpoint.
The resulting ONNX candidate must still pass parity, full deterministic MuJoCo
replay, live shadow mode, and supported first-actuation testing. Post-hoc key
renaming or copying an RSL-RL iteration into SONIC evidence is forbidden.
