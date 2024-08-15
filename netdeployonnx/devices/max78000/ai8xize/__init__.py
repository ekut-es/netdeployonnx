
import numpy as np
import onnx
import networkx as nx

from netdeployonnx.devices.max78000 import MAX78000
from netdeployonnx.devices.max78000.ai8xize.config import (
    AI8XizeConfig,
    AI8XizeConfigLayer,
)
from netdeployonnx.devices.max78000.ai8xize.wrap_ai8ize import (
    layout_transform as wrap_ai8ize_layout_transform,
)
from netdeployonnx.devices.max78000.cnn_registers import (
    register_class_by_address,
)
from netdeployonnx.devices.max78000.core import CNNx16Core


c10_layers = [
    AI8XizeConfigLayer(**layer)
    for layer in [
        {
            "out_offset": 0x4000,
            "processors": 0x0000000000000007,  # 1_1
            "operation": "conv2d",
            "kernel_size": "3x3",
            "pad": 1,
            "activate": "ReLU",
            "data_format": "HWC",
        },
        {
            "out_offset": 0x0000,
            "processors": 0xFFFFFFFFFFFFFFFF,  # 1_2
            "operation": "conv2d",
            "kernel_size": "1x1",
            "pad": 0,
            "activate": "ReLU",
            "output_shift": -1,
        },
        {
            "out_offset": 0x4000,
            "processors": 0x00000000FFFFFFFF,  # 1_3
            "operation": "conv2d",
            "kernel_size": "3x3",
            "pad": 1,
            "activate": "ReLU",
            "output_shift": -1,
        },
        {
            "max_pool": 2,
            "pool_stride": 2,
            "out_offset": 0x0000,
            "processors": 0xFFFFFFFFFFFFFFFF,  # 2_1
            "operation": "conv2d",
            "kernel_size": "3x3",
            "pad": 1,
            "activate": "ReLU",
            "output_shift": -3,
        },
        {
            "out_offset": 0x4000,
            "processors": 0xFFFFFFFF00000000,  # 2_2
            "operation": "conv2d",
            "kernel_size": "1x1",
            "pad": 0,
            "activate": "ReLU",
        },
        {
            "max_pool": 2,
            "pool_stride": 2,
            "out_offset": 0x0000,
            "processors": 0xFFFFFFFFFFFFFFFF,  # 3_1
            "operation": "conv2d",
            "kernel_size": "3x3",
            "pad": 1,  # do 0?
            "activate": "ReLU",
            "output_shift": -3,
        },
        {
            "out_offset": 0x4000,
            "processors": 0xFFFFFFFFFFFFFFFF,  # 3_2
            "operation": "conv2d",
            "kernel_size": "1x1",
            "pad": 0,
            "activate": "ReLU",
            "output_shift": -1,
        },
        {
            "max_pool": 2,
            "pool_stride": 2,
            "out_offset": 0x0000,
            "processors": 0xFFFFFFFFFFFFFFFF,  # 4_1
            "operation": "conv2d",
            "kernel_size": "3x3",
            "pad": 1,
            "activate": "ReLU",
            "output_shift": -3,
        },
        {
            "out_offset": 0x4000,
            "processors": 0xFFFFFFFFFFFFFFFF,  # 4_2
            "operation": "conv2d",
            "kernel_size": "3x3",
            "pad": 1,
            "activate": "ReLU",
            "output_shift": -2,
        },
        {
            "max_pool": 2,
            "pool_stride": 2,
            "out_offset": 0x0000,
            "processors": 0xFFFFFFFFFFFFFFFF,  # 5_1
            "operation": "conv2d",
            "kernel_size": "1x1",
            "pad": 0,
            "activate": "ReLU",
            "output_shift": -1,
        },
        {
            "flatten": True,
            "out_offset": 0x4000,
            "processors": 0xFFFFFFFFFFFFFFFF,
            "operation": "MLP",
            "output_width": 32,
            "activate": "none",
            "output_shift": 1,
        },
    ]
]

def get_nx_graph(model: onnx.ModelProto) -> nx.DiGraph:
    # Initialize a directed graph
    nx_graph = nx.DiGraph()

    def resolve_input_name(model: onnx.ModelProto, input_name: str) -> np.ndarray:
        # Resolve input name
        for initializer in model.graph.initializer:
            if initializer.name == input_name:
                return onnx.numpy_helper.to_array(initializer)
        return None

    # Add nodes and edges
    for node in model.graph.node:
        kwargs = {}
        if node.op_type in ["Conv", "Gemm"]:
            kwargs["weights"] = resolve_input_name(model, node.input[1])
            kwargs["bias"] = resolve_input_name(model, node.input[2])
        nx_graph.add_node(node.name, op_type=node.op_type, name=node.name, **kwargs)
        for input_name in node.input:
            nx_graph.add_edge(input_name, node.name)
        for output_name in node.output:
            nx_graph.add_edge(node.name, output_name)
    return nx_graph

class MAX78000_ai8xize(MAX78000):  # noqa: N801
    @classmethod
    def create_device_from_name_and_ports(
        cls,
        model_name: str,
        communication_port: str,
        energy_port: str,
    ) -> MAX78000:
        return MAX78000_ai8xize(
            model_name,
            "Maxim Integrated",
            "?",
            communication_port,
            energy_port,
        )

    async def layout_transform(self, model: onnx.ModelProto) -> any:
        cfg = self.generate_config_from_model(model)
        list_of_results: list[any] = wrap_ai8ize_layout_transform(cfg, model)

        core = CNNx16Core()

        for apb in list_of_results:
            set_lregs_to_core(apb._lregs, core)

        return core

    @staticmethod
    def following_node(graph: any, node: any, op_type: str, max_depth: int) -> bool:
        node_name: str = node.get("name")
        if node_name is None: return False
        for depth, layer in enumerate(nx.bfs_layers(graph, node_name)):
            if depth == max_depth:
                break
            for nodename in layer:
                other_node = graph.nodes[nodename]
                if other_node.get('op_type') == op_type:
                    return True
        return False


    def generate_config_from_model(self, model: onnx.ModelProto) -> dict:
        # the cfg is expected in the order of the nodes in the onnx model
        layers = []
        out_offset = 0
        nx_graph = get_nx_graph(model)
        seq = 0
        for node in model.graph.node:
            nxnode: any = nx_graph.nodes[node.name]
            op_type:dict = nxnode.get("op_type", None)
            if op_type in ["Conv", "Gemm"]:
                layerconfig = {}
                if out_offset == 0:
                    out_offset = 0x4000
                else:
                    out_offset = 0
                layerconfig["out_offset"] = out_offset
                layerconfig["processors"] = 0xFFFFFFFFFFFFFFFF
                layerconfig["operation"] = "conv2d" if op_type == "Conv" else "MLP"
                if self.following_node(nx_graph, nxnode, "Relu", 8):
                    layerconfig["activate"] = "ReLU"
                if self.following_node(nx_graph, nxnode, "Flatten", 8):
                    layerconfig["flatten"] = True
                if op_type == "Conv":
                    weights_shape = nxnode.get("weights").shape
                    layerconfig["kernel_size"] = "3x3" if weights_shape[2] == 3 else "1x1"

                layers.append(AI8XizeConfigLayer(**layerconfig))
        c10_layers = layers
        cfg = AI8XizeConfig(
            arch="ai85nascifarnet", dataset="CIFAR10", layers=c10_layers
        )
        return dict(cfg.model_dump(exclude_defaults=True))


def set_lregs_to_core(lregs: list[any], core: CNNx16Core):
    for lreg in lregs:
        (quad, layeridx, reg, val, force_write, no_verify, comment) = lreg
        globalreg, local_reg = reg
        layer = core[quad, layeridx]
        local_reg &= 0xFFFF
        if local_reg in register_class_by_address:
            register_class = register_class_by_address[local_reg & 0xFFFF]
            register = register_class()
            register.value = val
            layer.set_from_register(register)
        else:
            raise ValueError(
                f"did not find register class for local_reg={local_reg:04X}"
                f"global_reg={globalreg:04X}"
            )
