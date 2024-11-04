#
# Copyright (c) 2024 netdeployonnx contributors.
#
# This file is part of netdeployonx.
# See https://github.com/ekut-es/netdeployonnx for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
            # we need to modify the weights output channels
            node.ClearField('output')
            node.output.extend(["output"])
    # initializer = [i for i in model.graph.initializer if i.name == "onnx::Conv_357" or i.name == "onnx::Conv_358"]
    # target_shape = [16,32, 1, 1]
    # # we cant just reshape, but just remove channels 16...64
    # orig_array = onnx.numpy_helper.to_array(initializer[0])
    # print(orig_array.shape)
    # reduced_array = orig_array[:target_shape[0], :, :, :]
    # print (reduced_array.shape)
    # assert reduced_array.shape == tuple(target_shape)
    # model.graph.initializer.remove(initializer[0])
    # model.graph.initializer.append(onnx.helper.make_tensor(
    #     name="onnx::Conv_357",
    #     data_type=initializer[0].data_type,
    #     dims=target_shape,
    #     vals=reduced_array.tobytes(),
    #     raw=True,
    #     ))
    # model.graph.initializer.remove(initializer[1])
    # model.graph.initializer.append(onnx.helper.make_tensor(
    #     name="onnx::Conv_359",
    #     data_type=initializer[1].data_type,
    #     dims=initializer[1].dims,
    #     vals=reduced_array.tobytes(),
    #     raw=True,
    #     ))


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
