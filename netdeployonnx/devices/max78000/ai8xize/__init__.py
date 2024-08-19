import math
from collections import defaultdict

import networkx as nx
import numpy as np
import onnx
from izer import tornadocnn as tc  # noqa: E402

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
from netdeployonnx.devices.max78000.core import (
    CNNx16_Processor,
    CNNx16Core,
)
from netdeployonnx.devices.max78000.graph_transformer import (
    Graph,
    run_optimizer,
)


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
        cfg, input_shape = self.generate_config_from_model(model)
        layer0_is_not_gemm = cfg.get("layers", [{}])[0].get("operation") != "MLP"
        if layer0_is_not_gemm:
            # if the first layer is a CONV layer, then the input shape should be
            # in_chan x H x W
            assert len(input_shape) == 3, f"unexpected input shape: {input_shape}"
        sample_input = np.zeros(input_shape, dtype=np.int64)
        list_of_results: list[any] = wrap_ai8ize_layout_transform(
            cfg, model, sample_input
        )

        core = CNNx16Core()

        for apb in list_of_results:
            set_lregs_to_core(apb._lregs, core)
            set_bias_to_core(apb._bias, core)
            set_weights_to_core(apb.kernel_mem, core)

        return core

    @staticmethod
    def following_node(graph: any, node: any, op_type: str, max_depth: int) -> bool:
        node_name: str = node.get("name")
        if node_name is None:
            return False
        for depth, layer in enumerate(nx.bfs_layers(graph, node_name)):
            if depth == max_depth:
                break
            for nodename in layer:
                other_node = graph.nodes[nodename]
                if other_node.get("op_type") == op_type:
                    return True
        return False

    def transform_graph(self, graph: onnx.GraphProto) -> any:
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

    def generate_config_from_model(  # noqa: C901
        self, model: onnx.ModelProto
    ) -> tuple[dict, list[int]]:
        # the cfg is expected in the order of the nodes in the onnx model
        layers: list[AI8XizeConfigLayer] = []
        input_shape: list[int] = None
        trf_graph = self.transform_graph(model.graph)

        nx_graph = trf_graph.nxgraph_for_ai8x()
        for node in trf_graph.onnx().node:
            nxnode: any = nx_graph.nodes[node.name]
            op_type: dict = nxnode.get("op_type", None)
            if op_type.startswith("Conv") or op_type.startswith("Gemm"):
                if nxnode.get("input") is not None:
                    input_shape = list(nxnode.get("input").shape)
                    if len(input_shape) == 4:
                        input_shape = input_shape[1:]
                    # assert (
                    #     len(input_shape) == 3
                    # ), f"unexpected input shape: {input_shape}"
                    # TODO: re-enable this check
                ly = AI8XizeConfigLayer(processors=0, out_offset=0)
                ly.name = node.name

                if len(layers) == 0:
                    ly.data_format = "HWC"
                ly.out_offset = 0x4000 if len(layers) % 2 == 0 else 0
                if len(layers) == 0:  # TODO: generalize this
                    input_channels = nxnode.get("input").shape[1]
                    ly.processors = 2 ** (input_channels) - 1
                elif len(layers) == 2:  # TODO: generalize this
                    ly.processors = 0x00000000FFFFFFFF
                elif len(layers) == 4:  # TODO: generalize this
                    ly.processors = 0xFFFFFFFF00000000
                else:
                    ly.processors = 0xFFFFFFFFFFFFFFFF
                ly.operation = "conv2d" if op_type.startswith("Conv") else "MLP"
                if "Relu" in node.name:
                    ly.activate = "ReLU"
                if "Reshape" in node.name:
                    # TODO check if it is flatten
                    ly.flatten = True
                if "MaxPool" in node.name:
                    ly.max_pool = nxnode.get("_maxpool_kernel_shape", [1, 1])
                    if isinstance(ly.max_pool, list) and len(ly.max_pool) == 2:
                        assert ly.max_pool[0] == ly.max_pool[1]
                        ly.max_pool = ly.max_pool[0]
                    elif isinstance(ly.max_pool, int):
                        pass
                    else:
                        raise ValueError(f"unexpected max_pool value: {ly.max_pool}")
                    ly.pool_stride = nxnode.get("_maxpool_strides", [1, 1])
                    if isinstance(ly.pool_stride, list) and len(ly.pool_stride) == 2:
                        assert ly.pool_stride[0] == ly.pool_stride[1]
                        ly.pool_stride = ly.pool_stride[0]
                    elif isinstance(ly.pool_stride, int):
                        pass
                    else:
                        raise ValueError(
                            f"unexpected pool_stride value: {ly.pool_stride}"
                        )
                if op_type.startswith("Conv"):
                    weights_shape = nxnode.get("weights").shape
                    ly.kernel_size = "3x3" if weights_shape[2] == 3 else "1x1"
                pads = nxnode.get("pads", [0, 0, 0, 0])
                assert len(pads) == 4 and all(p == pads[0] for p in pads)
                ly.pad = pads[0]

                _squeeze_factor = nxnode.get("_squeeze_factor", [1])
                assert isinstance(_squeeze_factor, list) and len(_squeeze_factor) == 1
                ly.output_shift = int(math.log2(_squeeze_factor[0])) + 2

                layers.append(ly)

        layers[-1].output_width = 32  # for our output
        layers[-1].output_shift -= 1  # TODO: check if this is correct
        c10_layers = layers
        cfg = AI8XizeConfig(
            arch="ai85nascifarnet", dataset="CIFAR10", layers=c10_layers
        )
        return dict(cfg.model_dump(exclude_unset=True)), input_shape


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


def set_bias_to_core(bias: list[tuple[int, int, int]], core: CNNx16Core):
    collected_bias_per_quad = defaultdict(dict)
    maxoffs_per_quad = [0] * 4
    for quad, offs, val in bias:
        if offs not in collected_bias_per_quad[quad]:
            collected_bias_per_quad[quad][offs] = val
            maxoffs_per_quad[quad] = max(maxoffs_per_quad[quad], offs)
    quad_bias = [[] for _ in range(4)]
    # iterate over the quads
    for quad, bias_collection in collected_bias_per_quad.items():
        # iterate over the layers
        for i in range(
            min(maxoffs_per_quad[quad] + 1, tc.dev.BIAS_SIZE)
        ):  # iterate over all necessary bytes of bias
            # append the bias value to the quad_bias list, if it exists
            # if it doesn't exist, append 0
            quad_bias[quad].append(bias_collection.get(i, 0))
        # set the bias values to the core
        core[quad].bias = bytes(quad_bias[quad])


def assign_collected_weights_to_processor(
    collected_weights: dict[int, bytes], processor: CNNx16_Processor
):
    for addr, weights_array in collected_weights.items():
        assert addr % 4 == 0, "lower than 8 bit resolution not implemented"
        addr_div_4 = addr // 4  # because the address is in 2-bit resolution
        if weights_array is None:
            continue
        # in case our weights array is not 4-byte aligned, we need to pad it
        array_4byte_aligned = len(weights_array) % 4
        if array_4byte_aligned != 0:
            weights_array += b"\x00" * (4 - array_4byte_aligned)
        if addr_div_4 not in processor.kernels:
            processor.kernels[addr_div_4] = weights_array
        else:
            from warnings import warn

            warn(
                f"overwriting kernel at address {addr:08X} in" f" processor {processor}"
            )
            processor.kernels[addr_div_4] = weights_array


def set_weights_to_core(weights: list[list[list[any]]], core: CNNx16Core):
    apb_base = 0

    for group in range(len(weights)):
        for proc in range(len(weights[group])):
            processor: CNNx16_Processor = core[group].processors[proc]
            memory_array = weights[group][proc]
            for mem in range(len(memory_array)):
                weights_array = memory_array[mem]
                collected_weights: dict[int, bytes] = {}
                assert isinstance(weights_array, list)
                for weights_entry in weights_array:
                    if not isinstance(weights_entry, tuple):
                        raise ValueError(f"unexpected type: {type(weights_entry)}")
                    assert len(weights_entry) == 2
                    naddr, weights_array = weights_entry

                    # copied and modified from apbaccess.py in ai8xize (https://github.com/analogdevicesinc/ai8x-synthesis/blob/0f3dd3a3af464e1615722929a27363280281b31a/izer/apbaccess.py#L166)
                    if mem >= tc.dev.MASK_INSTANCES_EACH:
                        phys_addr = (
                            apb_base
                            # + tc.dev.C_GROUP_OFFS * group
                            # + tc.dev.C_MRAM_BASE
                            + proc * tc.dev.MASK_OFFS * 16
                            + tc.dev.MASK_WIDTH_SMALL * 16
                            + (mem - tc.dev.MASK_INSTANCES_EACH)
                            * 16
                            * (tc.dev.MASK_WIDTH_LARGE - tc.dev.MASK_WIDTH_SMALL)
                            // tc.dev.MASK_INSTANCES_EACH
                            + naddr * 16
                        )
                    else:
                        phys_addr = (
                            apb_base
                            # + tc.dev.C_GROUP_OFFS * group
                            # + tc.dev.C_MRAM_BASE
                            + proc * tc.dev.MASK_OFFS * 16
                            + mem
                            * 16
                            * tc.dev.MASK_WIDTH_SMALL
                            // tc.dev.MASK_INSTANCES_EACH
                            + naddr * 16
                        )
                    phys_addr %= tc.dev.MASK_OFFS * 16
                    if phys_addr not in collected_weights:
                        # check if there is an address 16 bytes earlier in the list
                        if phys_addr - 16 in collected_weights:
                            addr_list = sorted(
                                (
                                    item[0]
                                    for item in collected_weights.items()
                                    if item[1] is not None
                                ),
                            )
                            iterated_addr = addr_list[-1] if addr_list else None
                            if iterated_addr in collected_weights:
                                collected_weights[iterated_addr] += (
                                    weights_array.tobytes()
                                )
                                collected_weights[phys_addr] = None
                            else:
                                raise ValueError(
                                    f"unexpected address: {iterated_addr:08X}"
                                )
                        else:
                            collected_weights[phys_addr] = weights_array.tobytes()
                    else:
                        raise ValueError(f"unexpected address: {phys_addr:08X}")

                assign_collected_weights_to_processor(collected_weights, processor)
