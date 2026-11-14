"""
Turn SONIC ``.pt2`` files into ExecuTorch ``.pte`` files that run on TensorRT.

``eval_agent_trl.py ... +export_onnx_only=true +export_pt2=true`` writes the ``.pt2``
files next to the ONNX files. Run this script on the device you deploy to: the TensorRT
engine inside a ``.pte`` only runs on the GPU type and TensorRT version it was built
with. It needs only ``pip install "torch_tensorrt[executorch]==2.15.*"``, not the rest
of gear_sonic.

Usage:

    python gear_sonic/scripts/pt2_to_pte.py exported/model_step_000123_encoder.pt2 \
        exported/model_step_000123_decoder.pt2

Each ``<name>.pt2`` becomes ``<name>.pte`` in the same folder.
"""

import argparse
from pathlib import Path

import torch
from torch.export.passes import move_to_device_pass
import torch_tensorrt
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule


def convert(pt2_path: Path) -> Path:
    # The training side exports on the CPU; the engine is built on the GPU.
    program = move_to_device_pass(torch.export.load(pt2_path), "cuda")
    (input_name,) = program.graph_signature.user_inputs
    placeholder = next(node for node in program.graph.nodes if node.name == input_name)
    example_input = torch.zeros(placeholder.meta["val"].shape, device="cuda")

    with torch.no_grad():
        # Keep FP32 layers in full FP32 so the outputs stay close to the PyTorch model.
        # TensorRT otherwise may pick faster but less exact TF32 kernels.
        trt_module = torch_tensorrt.dynamo.compile(
            program, inputs=[example_input], disable_tf32=True
        )
    # The C++ loader checks only the input and the output, so make sure TensorRT took the whole
    # model rather than leaving parts of it to run outside the engine.
    calls = [node for node in trt_module.graph.nodes if node.op in ("call_module", "call_function")]
    one_engine = (
        len(calls) == 1
        and calls[0].op == "call_module"
        and isinstance(trt_module.get_submodule(calls[0].target), TorchTensorRTModule)
    )
    if not one_engine:
        raise SystemExit(f"TensorRT could not convert all of {pt2_path}, so no .pte was written")
    pte_path = pt2_path.with_suffix(".pte")
    # Same as the TensorRT deploy path, which replays its engines as CUDA graphs.
    torch_tensorrt.save(trt_module, str(pte_path), output_format="executorch", use_cuda_graphs=True)
    return pte_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("pt2_files", nargs="+", type=Path, help=".pt2 files to convert")
    for pt2_path in parser.parse_args().pt2_files:
        print(f"Saved {convert(pt2_path)}")  # noqa: T201


if __name__ == "__main__":
    main()
