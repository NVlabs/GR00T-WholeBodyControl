"""Write the tiny ``.pte`` programs that ``test_executorch_engine.cpp`` loads.

They are CPU programs with no TensorRT engine inside. Run this in an environment with
``executorch`` installed, for example from ``pip install "torch_tensorrt[executorch]==2.15.*"``:

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


class ChannelsLastIn(torch.nn.Module):
    def forward(self, x):
        # Contiguous output, so the input layout is the only thing wrong with this program.
        return (x + 1).contiguous()


class ChannelsLastOut(torch.nn.Module):
    def forward(self, x):
        return (x + 1).contiguous(memory_format=torch.channels_last)


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
        "channels_last_input": (
            ChannelsLastIn(),
            (torch.zeros(1, 2, 2, 2).to(memory_format=torch.channels_last),),
        ),
        "channels_last_output": (ChannelsLastOut(), (torch.zeros(1, 2, 2, 2),)),
    }
    out = Path(__file__).parent
    for name, (model, example_inputs) in models.items():
        program = to_edge_transform_and_lower(torch.export.export(model, example_inputs))
        (out / f"{name}.pte").write_bytes(program.to_executorch().buffer)
    # Point the delegate data field of the program table far past the end of the file.
    # The structure check rejects that. This program has no delegate, so without the check
    # nothing reads the field and the file loads, which makes the test fail.
    damaged = bytearray((out / "slice_add.pte").read_bytes())
    table = int.from_bytes(damaged[:4], "little")
    field_offsets = table - int.from_bytes(damaged[table : table + 4], "little", signed=True)
    damaged[field_offsets + 11] ^= 0xFF  # high byte of the offset of field 3, backend_delegate_data
    (out / "damaged.pte").write_bytes(damaged)


if __name__ == "__main__":
    main()
