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

import onnx

from netdeployonnx.devices.max78000.graph_transformer import (
    Graph,
    run_optimizer,
)


def transform_graph(graph: onnx.GraphProto) -> any:
    graph = Graph(graph)
    last_pass = False
    while True:
        changes: int = run_optimizer(graph, last_pass=last_pass)
        if changes == 0:
            if last_pass:
                break
            else:
                last_pass = True
    return graph


def transform_graph_onnx(onnx_filename: str):
    data_folder = Path(__file__).parent.parent / "test" / "data"
    model = onnx.load(data_folder / onnx_filename)
    new_filename = (
        data_folder / f"transformed_{(data_folder / onnx_filename).stem}.onnx"
    )

    # print(f"transforming {(data_folder / onnx_filename)} to {new_filename}")
    try:
        model = onnx.helper.make_model(transform_graph(model.graph).onnx())
    except Exception as ex:
        print(f"Failed to transform {onnx_filename}")
        raise ex
        return

    new_model = onnx.helper.make_model(model.graph, producer_name="onnx-edit")
    print(f"Saving to {new_filename}")
    res = onnx.save(new_model, new_filename)
    print(f"Saving result: {res}")


def main():
    for filename in [
        # "ai8x_net_0.onnx",
        # "ai8x_net_1.onnx",
        # "ai8x_net_2.onnx",
        # "ai8x_net_3.onnx",
        # "ai8x_net_4.onnx",
        # "ai8x_net_5.onnx",
        # "ai8x_net_6.onnx",
        # "ai8x_net_7.onnx",
        # "ai8x_net_8.onnx",
        # "ai8x_net_9.onnx",
        # "ai8x_net_0_fixed.onnx",
        # "ai8x_net_1_fixed.onnx",
        # "ai8x_net_2_fixed.onnx",
        # "ai8x_net_3_fixed.onnx",
        # "ai8x_net_4_fixed.onnx",
        # "ai8x_net_5_fixed.onnx",
        # "ai8x_net_6_fixed.onnx",
        # "ai8x_net_7_fixed.onnx",
        # "ai8x_net_8_fixed.onnx",
        # "ai8x_net_9_fixed.onnx",
        # "cifar10_short.onnx",
        # "cifar10.onnx",
        "ai85-bayer2rgb-qat8-q.pth.onnx",
        "ai85-cifar10-qat8-q.pth.onnx",
        "ai85-cifar100-qat8-q.pth.onnx",
        "ai85-faceid_112-qat-q.pth.onnx",
        "ai85-kws20_v3-qat8-q.pth.onnx",
    ]:
        transform_graph_onnx(filename)


if __name__ == "__main__":
    main()
