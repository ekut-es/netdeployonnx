from pathlib import Path
import numpy as np

import onnx


def fix_w2_in_files(onnx_filename):
    data_folder = Path(__file__).parent  # this folder
    model = onnx.load(data_folder / onnx_filename)
    new_filename = (data_folder / onnx_filename).stem + "_fixed.onnx"

    # just modify w2
    assert "w2" in [input.name for input in model.graph.input]
    for input in model.graph.input:
        if input.name == "w2":
            print(f"Fixing {onnx_filename}")
            old_tensor = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
            print(old_tensor)
            while np.prod(old_tensor) > 32768:
                old_tensor[0] //= 2
            input.type.tensor_type.shape.ClearField("dim")
            input.type.tensor_type.shape.dim.extend(
                [onnx.TensorShapeProto.Dimension(dim_value=dim) for dim in old_tensor]
            )
            new_shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
            print(f"New shape: {new_shape}")
            break

    new_model = onnx.helper.make_model(model.graph, producer_name="onnx-edit")
    print(f"Saving to {new_filename}")
    onnx.save(new_model, data_folder / new_filename)


def main():
    for filename in [
        "ai8x_net_0.onnx",
        "ai8x_net_1.onnx",
        "ai8x_net_2.onnx",
        "ai8x_net_3.onnx",
        "ai8x_net_4.onnx",
        "ai8x_net_5.onnx",
        "ai8x_net_6.onnx",
        "ai8x_net_7.onnx",
        "ai8x_net_8.onnx",
        "ai8x_net_9.onnx",
    ]:
        fix_w2_in_files(filename)


if __name__ == "__main__":
    main()
