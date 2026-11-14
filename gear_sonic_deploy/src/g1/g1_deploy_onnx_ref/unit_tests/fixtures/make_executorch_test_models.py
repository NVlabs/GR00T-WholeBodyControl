"""Write the tiny ``.pte`` programs that ``test_executorch_engine.cpp`` loads.

They run on the CPU only, so the tests need no TensorRT. Run this in an environment with
``executorch`` installed, for example from ``pip install "torch_tensorrt[executorch]"``:

    python make_executorch_test_models.py
"""

from pathlib import Path

from executorch.exir import to_edge_transform_and_lower
import torch


class SliceAdd(torch.nn.Module):
    """The valid case: 3 floats in, 2 floats out."""

    def forward(self, x):
        return x[:, :2] + 1


class G1Policy(torch.nn.Module):
    """A valid policy for PolicyEngine: 30 floats in, one action per G1 motor (29) out."""

    def forward(self, x):
        return x[:, :29] + 1


class TwoInputs(torch.nn.Module):
    def forward(self, x, y):
        return x + y


class TwoOutputs(torch.nn.Module):
    def forward(self, x):
        return x + 1, x + 2


class AddOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


class IntInput(torch.nn.Module):
    def forward(self, x):
        return x.float() + 1


class IntOutput(torch.nn.Module):
    def forward(self, x):
        return torch.argmax(x, dim=1)


def main():
    x = torch.zeros(1, 3)
    models = {
        "slice_add": (SliceAdd(), (x,)),
        "g1_policy": (G1Policy(), (torch.zeros(1, 30),)),
        "two_inputs": (TwoInputs(), (x, x)),
        "int_input": (IntInput(), (torch.zeros(1, 3, dtype=torch.int32),)),
        "two_outputs": (TwoOutputs(), (x,)),
        "int_output": (IntOutput(), (x,)),
        "channels_last": (
            AddOne(),
            (torch.zeros(1, 2, 2, 2).to(memory_format=torch.channels_last),),
        ),
    }
    for name, (model, example_inputs) in models.items():
        program = to_edge_transform_and_lower(torch.export.export(model, example_inputs))
        (Path(__file__).parent / f"{name}.pte").write_bytes(program.to_executorch().buffer)


if __name__ == "__main__":
    main()
