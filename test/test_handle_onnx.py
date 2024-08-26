import io
import warnings

import onnx
import pytest

try:
    import google.protobuf.message
except ImportError:
    google = None

try:
    import torch
    import torch.nn as nn
    import torch.onnx
except ImportError:
    torch = None


from netdeployonnx.devices import TestDevice

warnings.filterwarnings("ignore", category=pytest.PytestCollectionWarning)


@pytest.fixture(scope="module")
def device():
    return TestDevice()


@pytest.mark.parametrize(
    "run_type, data, expected_error",
    [
        ("onnxb", b"", None),
        ("onnx", onnx.ModelProto(), None),
        # ("onnxb", "", ValueError),
        ("onnx", "", ValueError),
        ("tflite", b"", NotImplementedError),
        ("", b"", NotImplementedError),
        (None, b"", ValueError),
    ],
)
@pytest.mark.asyncio
async def test_device_run_types(
    device, run_type: str, data: (str, bytes), expected_error
):
    if expected_error:
        with pytest.raises(expected_error):
            await device.run_async(run_type, b"", "", b"")


@pytest.mark.asyncio
async def test_device_run_with_empty_data(device):
    datatype = "onnxb"
    data = b""
    with pytest.raises(ValueError):  # model empty
        result = await device.run_async(datatype, data, "", b"")
        assert isinstance(result, dict)
        assert "result" in result
        assert result["result"] is None


@pytest.mark.asyncio
async def test_device_run_with_false_data(device):
    datatype = "onnxb"
    data = b"1"
    with pytest.raises(google.protobuf.message.DecodeError):
        result = await device.run_async(datatype, data, "", b"")
        assert isinstance(result, dict)
        assert "result" in result
        assert result["result"] is None


@pytest.mark.skipif(torch is None, reason="PyTorch is not importable.")
@pytest.mark.asyncio
async def test_device_run_with_good_data(device):
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

    data_stream = io.BytesIO()

    # Export the model to an ONNX file
    torch.onnx.export(
        model,
        dummy_input,
        data_stream,
        input_names=["input"],
        output_names=["output"],
    )

    # run_id = "runid"
    datatype = "onnxb"
    data = data_stream.getvalue()

    result = await device.run_async(datatype, data, "", b"")

    assert isinstance(result, dict)
    assert "result" in result
    assert result["result"] == []
    assert "deployment_execution_times" in result
