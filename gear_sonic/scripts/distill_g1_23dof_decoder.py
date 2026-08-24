"""Distill the released 29-DoF SONIC decoder into the 23-output head.

Reinforcement learning from the reshaped warm start does not work: measured at
the shipped 0.25 m thresholds it survives 4.4 control steps, and PPO from there
optimizes episode termination rather than tracking. The released 29-DoF decoder,
by contrast, demonstrably tracks these motions.

Both decoders consume the same 994-value input, so the 23-output head can be
regressed directly onto the teacher's outputs at the 23 retained joint indices.
That is a supervised problem with a known target: no reward design, no
exploration, no termination tuning.

The teacher assumes six joints this body does not have (waist roll/pitch and
both wrists' pitch/yaw). Its targets for the retained joints were produced under
that assumption, so distillation alone cannot be expected to close the
embodiment gap - it produces a starting point far better than the reshaped head,
which reinforcement learning can then correct.

Distillation authorizes nothing. The result is an initialization checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gear_sonic.utils.g1_23dof_contract import (
    SOURCE_IL29_EXCLUDED_INDICES,
    SOURCE_IL29_KEEP_INDICES,
)

DECODER_INPUT_DIM = 994
TEACHER_OUTPUT_DIM = 29
STUDENT_OUTPUT_DIM = 23
_DECODER_PREFIX = "actor_module.decoders.g1_dyn."


def find_teacher(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    matches = glob.glob(
        "/root/.cache/huggingface/hub/models--nvidia--GEAR-SONIC/"
        "snapshots/*/model_decoder.onnx"
    )
    if not matches:
        raise SystemExit("released model_decoder.onnx not found in the HF cache")
    return Path(sorted(matches)[-1])


def build_student(state: dict[str, torch.Tensor]) -> tuple[nn.Sequential, list[str]]:
    """Rebuild the g1_dyn decoder as a plain MLP from the checkpoint tensors."""
    keys = sorted(
        (k for k in state if k.startswith(_DECODER_PREFIX) and k.endswith(".weight")),
        key=lambda k: int(k.split(".module.")[1].split(".")[0]),
    )
    layers: list[nn.Module] = []
    names: list[str] = []
    for position, key in enumerate(keys):
        weight = state[key]
        bias_key = key[: -len("weight")] + "bias"
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            linear.weight.copy_(weight)
            if bias_key in state:
                linear.bias.copy_(state[bias_key])
        layers.append(linear)
        names.append(key)
        if position < len(keys) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers), names


def sample_inputs(count: int, generator: torch.Generator) -> torch.Tensor:
    """Inputs spanning the decoder's operating range.

    Observation components are normalized, so standard normal samples with a
    spread of scales cover the space the decoder actually sees without needing
    a full environment rollout.
    """
    scales = torch.tensor([0.25, 0.5, 1.0, 1.5, 2.0])
    picks = scales[torch.randint(len(scales), (count, 1), generator=generator)]
    return torch.randn(count, DECODER_INPUT_DIM, generator=generator) * picks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-onnx", type=Path)
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    import onnxruntime as ort

    args = _parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    keep = list(SOURCE_IL29_KEEP_INDICES)
    if len(keep) != STUDENT_OUTPUT_DIM:
        raise SystemExit(
            f"expected {STUDENT_OUTPUT_DIM} retained indices, got {len(keep)}"
        )
    print(f"retained teacher outputs: {keep}")
    print(f"dropped (absent on this body): {list(SOURCE_IL29_EXCLUDED_INDICES)}")

    teacher_path = find_teacher(args.teacher_onnx)
    session = ort.InferenceSession(
        str(teacher_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    print(f"teacher: {teacher_path}")

    checkpoint = torch.load(args.student, map_location="cpu", weights_only=False)
    state = checkpoint["policy_state_dict"]
    student, layer_names = build_student(state)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    student.to(device)
    print(f"student layers: {len(layer_names)}  device: {device}")

    generator = torch.Generator().manual_seed(20260805)
    print(f"generating {args.samples} teacher targets...", flush=True)
    inputs = sample_inputs(args.samples, generator)
    targets = np.empty((args.samples, STUDENT_OUTPUT_DIM), dtype=np.float32)
    block = 512
    for start in range(0, args.samples, block):
        chunk = inputs[start : start + block].numpy().astype(np.float32)
        out = np.concatenate(
            [session.run(None, {input_name: row[None, :]})[0] for row in chunk],
            axis=0,
        )
        targets[start : start + len(chunk)] = out[:, keep]
        if start % (block * 40) == 0:
            print(f"  {start}/{args.samples}", flush=True)

    inputs = inputs.to(device)
    target_tensor = torch.from_numpy(targets).to(device)

    split = int(args.samples * 0.9)
    train_x, val_x = inputs[:split], inputs[split:]
    train_y, val_y = target_tensor[:split], target_tensor[split:]

    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    print(f"\n{'epoch':>6}{'train_mse':>14}{'val_mse':>14}{'val_max_err':>14}")
    history = []
    for epoch in range(args.epochs):
        student.train()
        order = torch.randperm(len(train_x), device=device)
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(student(train_x[index]), train_y[index])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(index)
        student.eval()
        with torch.no_grad():
            prediction = student(val_x)
            val = float(loss_fn(prediction, val_y))
            worst = float((prediction - val_y).abs().max())
        train_mse = total / len(train_x)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val})
        print(f"{epoch:>6}{train_mse:>14.6f}{val:>14.6f}{worst:>14.6f}", flush=True)

    # Write the distilled weights back into the checkpoint's decoder tensors.
    linear_layers = [m for m in student if isinstance(m, nn.Linear)]
    updated = dict(state)
    for name, layer in zip(layer_names, linear_layers, strict=True):
        updated[name] = layer.weight.detach().cpu()
        bias_key = name[: -len("weight")] + "bias"
        if bias_key in updated:
            updated[bias_key] = layer.bias.detach().cpu()

    report = {
        "kind": "g1_true23_decoder_distillation_v1",
        "teacher_onnx": str(teacher_path),
        "teacher_output_dim": TEACHER_OUTPUT_DIM,
        "student_output_dim": STUDENT_OUTPUT_DIM,
        "retained_indices": keep,
        "excluded_indices": list(SOURCE_IL29_EXCLUDED_INDICES),
        "samples": args.samples,
        "epochs": args.epochs,
        "final_train_mse": history[-1]["train_mse"],
        "final_val_mse": history[-1]["val_mse"],
        "history": history,
        "initialization_only": True,
        "deployment_ready": False,
        "caveat": (
            "Teacher assumes 29 actuated joints; six are absent on this body. "
            "Distillation transfers behaviour produced under that assumption."
        ),
    }

    out = dict(checkpoint)
    out["policy_state_dict"] = updated
    out["g1_23dof_distillation_report"] = report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    if args.report:
        args.report.write_text(json.dumps(report, indent=1))

    print(f"\n[OK] distilled decoder -> {args.output}")
    print(f"     final val MSE {history[-1]['val_mse']:.6f}")
    print("[NOTE] Initialization only. Train and validate before deployment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
