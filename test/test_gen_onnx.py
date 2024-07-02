try:
    import numpy as np
    import onnx
    import torch
    import torch.nn as nn
    import torch.onnx
    from onnx2pytorch import ConvertModel
except ImportError as imperr:
    print(f"Import error: {imperr}")

import pathlib

data = pathlib.Path(__file__).parent / "data"


def create_cifar10_model():
    import sys

    sys.path.append("/home/vscode/_Masterarbeit_SS24/hannah-env/ai8x-training/")
    sys.path.append("/home/vscode/_Masterarbeit_SS24/hannah-env/ai8x-training/models/")
    # i need to import the class from the file ai85net-nas-cifar.py
    # i tried to add the path to the sys.path, but it still doesn't work
    # i also tried to import the class from the file, but it doesn't work either

    def torch_aten_exp2(g, input):
        # print("TYPE=", type(input), "INPUT=", input)
        # if input is a torch.Value and is a float
        # then we can just return 2**input
        # exp_constant = None
        # exp_constant = 0
        # if exp_constant is not None:
        #     return g.op("Const", torch.tensor(2.0**exp_constant, dtype=float))
        return g.op("Pow", torch.tensor(2.0), input)

    torch.onnx.register_custom_op_symbolic("aten::exp2", torch_aten_exp2, 1)

    import ai8x

    ai8x.set_device(device=85, simulate=False, round_avg=False, verbose=True)

    # Instantiate the model
    ai85net = __import__("ai85net-nas-cifar")
    AI85NASCifarNet = ai85net.AI85NASCifarNet  # noqa: N806
    model = AI85NASCifarNet(bias=True)

    checkpoint = torch.load(
        data / "ai85-cifar10-qat8-q.pth.tar",
        map_location="cpu",
    )
    augmented_checkpoint_state_dict = checkpoint["state_dict"]
    shape = {
        "conv1_1": ([1] * 64, [1] * 64),
        "conv1_2": ([1] * 32, [1] * 32),
        "conv1_3": ([1] * 64, [1] * 64),
        "conv2_1": ([1] * 32, [1] * 32),
        "conv2_2": ([1] * 64, [1] * 64),
        "conv3_1": ([1] * 128, [1] * 128),
        "conv3_2": ([1] * 128, [1] * 128),
        "conv4_1": ([1] * 64, [1] * 64),
        "conv4_2": ([1] * 128, [1] * 128),
        "conv5_1": ([1] * 128, [1] * 128),
    }
    for conv in shape:
        for i, bn_type in enumerate(["running_mean", "running_var"]):
            augmented_checkpoint_state_dict[f"{conv}.bn.{bn_type}"] = torch.Tensor(
                shape[conv][i]
            )

    model.load_state_dict(augmented_checkpoint_state_dict)

    dummy_input = torch.randn(1, 3, 32, 32)

    # Export the model to an ONNX file
    torch.onnx.export(
        model,
        dummy_input,
        data / "cifar10.onnx",
        input_names=["input"],
        output_names=["output"],
    )
    return model


def create_model():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            return x

    # Instantiate the model
    model = SimpleModel()

    # Create a dummy input tensor with shape (1, 1, 5, 5)
    dummy_input = torch.randn(1, 1, 5, 5)

    # Assuming 'model.conv' is a pre-existing Conv2d layer
    model.conv.weight.data = torch.tensor(
        [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.float32
    )
    model.conv.bias.data = torch.tensor([1.5], dtype=torch.float32)
    print(dummy_input.size())
    print(model.conv.weight.data.size())
    print(model.conv.bias.data.size())

    # The correct sizes should be
    #     input: (batch_size, in_channels , height, width)
    #     weight: (out_channels, in_channels , kernel_height, kernel_width)

    # Export the model to an ONNX file
    torch.onnx.export(
        model,
        dummy_input,
        data / "simple_model.onnx",
        input_names=["input"],
        output_names=["output"],
    )
    return model


def load_model():
    try:
        # Load the ONNX model
        onnx_model = onnx.load("simple_model.onnx")
        onnx.checker.check_model(onnx_model)  # Optional: check the model
        return onnx_model
    except FileNotFoundError:
        return None


def execute_model(model):
    # Prepare input data as torch tensor
    input_data = torch.tensor(
        [
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        ],
        dtype=torch.float32,
    )
    output_data = torch.tensor(
        [
            [
                [47, 47, 47],
                [47, 47, 47],
                [47, 47, 47],
            ]
        ],
        dtype=torch.float32,
    )
    pytorch_model = ConvertModel(model)

    # Run inference
    with torch.no_grad():
        outputs = pytorch_model(input_data)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(outputs, output_data, rtol=1e-03, atol=1e-05)

    print("result looks good!")


def test_create_cifar10_model():
    model = create_cifar10_model()
    assert model


if __name__ == "__main__":
    create_cifar10_model()
    # model = load_model()
    # if not model:
    #     create_model()
    #     model = load_model()
    # execute_model(model)
