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
            while np.prod(old_tensor+[32]) > 32768:
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

def reduce_layers(onnx_filename):
    data_folder = Path(__file__).parent  # this folder
    model = onnx.load(data_folder / onnx_filename)
    new_filename = (data_folder / onnx_filename).stem + "_short.onnx"

    print(len(model.graph.node))

    print("available", len(model.graph.node), [n.name for n in model.graph.node])

    nodes_to_add = []
    # reduce layer height
    for node in model.graph.node:
        if (node.name.startswith("/conv2") or
            node.name.startswith("/conv1") or
            node.name.startswith("Identity")
        ):
            nodes_to_add.append(node)
        if node.name in ["onnx::Pow_394", "onnx::Pow_384", "Identity"]:
            nodes_to_add.append(node)
        if node.name == "/conv2_2/clamp/Clip_1":
            node.ClearField('output')
            node.output.extend(["output"])

    graph = onnx.helper.make_graph(
        nodes_to_add,
        model.graph.name,
        model.graph.input,
        model.graph.output,
        model.graph.initializer,
    )

    print("input", len(graph.input), [i.name for i in graph.input])
    print("node", len(graph.node), [n.name for n in graph.node])
    print("init",len(graph.initializer), [i.name for i in graph.initializer])

    new_model = onnx.helper.make_model(graph, producer_name="onnx-edit")
    print(f"Saving to {new_filename}")
    onnx.save(new_model, data_folder / new_filename)


def main():
    if 0:
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
    if 1:
        for filename in [
            "cifar10.onnx"
        ]:
            reduce_layers(filename)


if __name__ == "__main__":
    main()
