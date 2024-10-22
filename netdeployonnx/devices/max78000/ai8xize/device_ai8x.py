import logging
import math
import re
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
    CNNx16_Quadrant,
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

    def maximum_network_size_okay(self, bytecount: int):
        weights_start_addr = 0x10028000  # check max78000.map / customlinkerfile.ld
        if weights_start_addr == 0x10028000:
            return bytecount < 44 * self.FLASH_PAGE_SIZE
        elif weights_start_addr == 0x10030000:
            # when the weights start at 0x10030000 => 40 (see firmware)
            return bytecount < 40 * self.FLASH_PAGE_SIZE
        elif weights_start_addr == 0x10040000:
            return bytecount < 32 * self.FLASH_PAGE_SIZE
        else:
            return False

        return bytecount < 40 * self.FLASH_PAGE_SIZE

    async def layout_transform(self, model: onnx.ModelProto) -> any:  # noqa: C901
        DEBUG = True  # noqa: N806
        SAVE_MODELS = False  # noqa: N806

        if DEBUG and SAVE_MODELS:
            # saving model
            import os
            from pathlib import Path

            test = Path(__file__).parent.parent.parent.parent.parent / "test"
            n = len(os.listdir(test / "data"))
            filname = test / "data" / f"ai8x_test_{n}_possibly_notworking.onnx"
            with open(filname, "wb") as fx:  # noqa: ASYNC230
                fx.write(model.SerializeToString())
            print(f"saved as {filname}")

        izer_config, locked_config, input_shape, transformed_model = (
            self.generate_config_from_model(model)
        )
        if DEBUG:
            # from pprint import pprint

            # pprint(izer_config)
            import yaml

            print(yaml.dump(izer_config))
            print("input_shape:", input_shape)
        # print(onnx.printer.to_text(transformed_model.graph))
        layer0_is_not_gemm = (
            izer_config.get("layers", [{}])[0].get("operation") != "MLP"
        )
        if layer0_is_not_gemm:
            # if the first layer is a CONV layer, then the input shape should be
            # in_chan x H x W
            assert len(input_shape) == 3, f"unexpected input shape: {input_shape}"
        sample_input = np.zeros(input_shape, dtype=np.int64)
        list_of_results: list[any] = wrap_ai8ize_layout_transform(
            izer_config, transformed_model, sample_input
        )

        if DEBUG and SAVE_MODELS:
            # saving model
            import os
            from pathlib import Path

            test = Path(__file__).parent.parent.parent.parent.parent / "test"
            n = len(os.listdir(test / "data"))
            filname = test / "data" / f"ai8x_test_{n}_working.onnx"
            with open(filname, "wb") as fx:  # noqa: ASYNC230
                fx.write(model.SerializeToString())
            print(f"saved as {filname}")

        core = CNNx16Core()

        for apb in list_of_results:
            set_lregs_to_core(apb._lregs, core)
            set_bias_to_core(apb._bias, core)
            set_weights_to_core(apb.kernel_mem, core)

        def set_when_available(
            quadrant: CNNx16_Quadrant, locked_config: dict, setting_name: str
        ):
            if setting_name in locked_config:
                setattr(quadrant, setting_name, int(locked_config.get(setting_name, 0)))

        regaccess = re.compile(r"([\w]+)\.([\w]+)")
        cheeky_variables = re.compile(r"^__")
        for key in locked_config:
            if regaccess.match(key):
                core.specialconfig[key] = locked_config.get(key)
                continue
            if cheeky_variables.match(key):
                core.specialconfig[key] = locked_config.get(key)
                continue
            if not hasattr(core[0], key):
                logging.warning(
                    f"Core / Quadrant does not have the option {key}"
                    " and is not in the format 'REGISTER.FIELD'"
                )
                continue
        # to modify sram fetch
        for quad in range(4):
            # SRAM reg
            set_when_available(core[quad], locked_config, "lightsleep_bram")
            set_when_available(core[quad], locked_config, "lightsleep_tram")
            set_when_available(core[quad], locked_config, "lightsleep_mram")
            set_when_available(core[quad], locked_config, "lightsleep_dram")
            set_when_available(core[quad], locked_config, "memory_deep_sleep")
            set_when_available(core[quad], locked_config, "write_pulse_width")
            set_when_available(core[quad], locked_config, "write_neg_voltage_enable")
            set_when_available(core[quad], locked_config, "write_neg_voltage")
            set_when_available(core[quad], locked_config, "read_assist_voltage")
            set_when_available(core[quad], locked_config, "read_margin")
            set_when_available(core[quad], locked_config, "read_margin_enable")
            set_when_available(core[quad], locked_config, "extended_access_time_enable")

            # CTRL reg
            set_when_available(core[quad], locked_config, "pool_en")
            set_when_available(core[quad], locked_config, "maxpool_en")

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

    @staticmethod
    def processor_format_helper(amount_on: int, shift: int = 0) -> int:
        assert amount_on >= 0, "proc amount less than 0 is not possible"
        assert shift >= 0, "proc shift less than 0 is not possible"
        processor_count = min(64, amount_on)
        processor_shift = min(64 - processor_count, shift)

        return (2 ** (processor_count) - 1) << processor_shift

    def generate_config_from_model(  # noqa: C901
        self, model: onnx.ModelProto
    ) -> tuple[dict, dict, list[int], onnx.ModelProto]:
        # the cfg is expected in the order of the nodes in the onnx model
        INSTANCE_WIDTH = 0x800  # noqa: N806 , F841    tc.dev.INSTANCE_WIDTH
        DEVICE_MEMORY_INSTANCE_SIZE = (
            INSTANCE_WIDTH * 16
        )  # atleast for ai85 (tc.dev.INSTANCE_WIDTH*16)  # noqa: E501, N806
        PING_PONG_VALUE = DEVICE_MEMORY_INSTANCE_SIZE // 2  # noqa: N806

        locked_config = {item.key: item.value for item in model.metadata_props}
        layers: list[AI8XizeConfigLayer] = []
        input_shape: list[int] = None
        trf_graph = self.transform_graph(model.graph)

        relevant_channels = 1

        nx_graph = trf_graph.nxgraph_for_ai8x()

        onnx_graph = trf_graph.onnx()
        layer_information = get_layer_information(
            model=onnx.helper.make_model(graph=onnx_graph),
        )
        layerlevel = -1  # so we start at 0
        for node in onnx_graph.node:
            nxnode: any = nx_graph.nodes[node.name]
            op_type: dict = nxnode.get("op_type", None)
            if (
                op_type.startswith("Conv")
                or op_type.startswith("Gemm")
                or op_type.startswith("None")  # average pool standalone (faceid)
            ):
                layerlevel += 1
                weights_shape = nxnode.get("weights", np.zeros(shape=[0])).shape
                is_1d = len(weights_shape) == 3  # just guessing
                assert len(weights_shape) in [2, 3, 4], (
                    f"weights shape has to be in 2D, 3D or 4D,"
                    f"but is {weights_shape}"
                )

                if nxnode.get("input") is not None and input_shape is None:
                    input_shape = list(nxnode.get("input").shape)
                    if len(input_shape) == 4:
                        input_shape = input_shape[1:]
                        input_dim = input_shape[-2:]
                    assert (
                        len(input_shape) == 3
                    ), f"unexpected input shape: {input_shape}"

                ly = AI8XizeConfigLayer(processors=0, out_offset=0)
                ly.name = node.name

                # extract all the rest
                if op_type.startswith("Conv"):
                    if is_1d:
                        ly.operation = "Conv1d"  # capitalization is needed for testing, .but can be lower()  # noqa: E501
                    else:
                        if nxnode.get("transpose", None):
                            ly.operation = "convtranspose2d"  # capitalization is needed for testing
                        else:
                            ly.operation = (
                                "Conv2d"  # capitalization is needed for testing
                            )

                else:
                    ly.operation = "MLP"
                if "Relu" in node.name:
                    ly.activate = "ReLU"
                if "Reshape" in node.name:
                    ly.flatten = True
                if "Flatten" in node.name:
                    ly.flatten = True
                if "MaxPool" in node.name:
                    ly.max_pool = nxnode.get("_maxpool_kernel_shape", [1, 1])
                    if isinstance(ly.max_pool, list) and len(ly.max_pool) in [1, 2]:
                        if len(ly.max_pool) == 2:
                            assert ly.max_pool[0] == ly.max_pool[1]
                        ly.max_pool = ly.max_pool[0]
                    elif isinstance(ly.max_pool, int):
                        # ly.max_pool = ly.max_pool[0] # why not this?
                        pass
                    else:
                        raise ValueError(f"unexpected max_pool value: {ly.max_pool}")
                    ly.pool_stride = nxnode.get("_maxpool_strides", [1, 1])
                    if isinstance(ly.pool_stride, list) and len(ly.pool_stride) in [
                        1,
                        2,
                    ]:
                        if len(ly.pool_stride) == 2:
                            assert ly.pool_stride[0] == ly.pool_stride[1]
                        ly.pool_stride = ly.pool_stride[0]
                    elif isinstance(ly.pool_stride, int):
                        pass
                    else:
                        raise ValueError(
                            f"unexpected pool_stride value: {ly.pool_stride}"
                        )
                if "AveragePool" in node.name:
                    ly.avg_pool = nxnode.get("_averagepool_kernel_shape", [1, 1])
                    if isinstance(ly.avg_pool, list) and len(ly.avg_pool) in [1, 2]:
                        if len(ly.avg_pool) == 2:
                            assert ly.avg_pool[0] == ly.avg_pool[1]
                        ly.avg_pool = ly.avg_pool[0]
                    elif isinstance(ly.avg_pool, int):
                        # ly.avg_pool = ly.avg_pool[0] # why not this?
                        pass
                    else:
                        raise ValueError(f"unexpected avg_pool value: {ly.avg_pool}")
                    ly.pool_stride = nxnode.get("_averagepool_strides", [1, 1])
                    if isinstance(ly.pool_stride, list) and len(ly.pool_stride) in [
                        1,
                        2,
                    ]:
                        if len(ly.pool_stride) == 2:
                            assert ly.pool_stride[0] == ly.pool_stride[1]
                        ly.pool_stride = ly.pool_stride[0]
                    elif isinstance(ly.pool_stride, int):
                        pass
                    else:
                        raise ValueError(
                            f"unexpected pool_stride value: {ly.pool_stride}"
                        )

                if op_type.startswith("Conv"):
                    # invalid to set for MLP/Gemm
                    kernel_size = nxnode.get("kernel_shape", [1])
                    # we only have square kernels for 2d
                    # Ackchyually, ERROR: Unsupported value `1x1` for `kernel_size` (found in layer sequence 0 in YAML configuration).
                    if is_1d:
                        ly.kernel_size = kernel_size[0]  # mental (int)
                    else:
                        ly.kernel_size = f"{kernel_size[0]}x{kernel_size[0]}"

                layer_input_shape = weights_shape  # noqa: F841
                # assert np.prod(layer_input_shape) < INSTANCE_WIDTH * 16 * 9, (
                #     f"input shape {layer_input_shape}={np.prod(layer_input_shape)} "
                #     f"is too large for the core REF={INSTANCE_WIDTH * 16}"
                # )

                pads = nxnode.get("pads", [0, 0, 0, 0])
                # 2 for 1d, 4 for OIWH/OIHW
                assert len(pads) in [2, 4] and all(p == pads[0] for p in pads)
                ly.pad = pads[0]

                _squeeze_factor = nxnode.get("_squeeze_factor", [1])
                assert isinstance(_squeeze_factor, list) and len(_squeeze_factor) == 1
                # the thing is: output shift is solved via processor shifting
                # ly.output_shift = int(math.log2(_squeeze_factor[0])) + 2

                # now calc inputs

                input_dim = [1, 1]

                if len(layers) == 0:
                    ly.data_format = "HWC"
                    if is_1d:  # noqa: SIM108
                        relevant_channels = input_shape[1]
                    else:
                        relevant_channels = input_shape[0]
                ly.out_offset = PING_PONG_VALUE if len(layers) % 2 == 0 else 0

                if not op_type.lower().startswith("conv1d") and "Flatten" in node.name:
                    # basically, if this is not conv1d and its flatten
                    # izer.py:555 does this:
                    # input_channels[ll] //= pooled_dim[ll][0] * pooled_dim[ll][1]
                    # and the pooled_dim is calculated by
                    # [(input_dim[0] + pool_stride[0] - pool[0] - pool_dilation[0] + 1) // pool_stride[0], ...]  # noqa: E501
                    pooled_dimensions = input_dim[0] * input_dim[1]
                    relevant_channels //= pooled_dimensions

                if op_type.startswith("Gemm") and "Flatten" in node.name:
                    # we need to modify the proc count
                    relevant_channels = weights_shape[1]

                # reduce by input channels
                passes_float = relevant_channels / 64
                passes = math.ceil(passes_float)
                assert passes_float > 0, "passes_float cant be 0 or smaller"
                assert passes > 0, "passes cant be 0 or smaller (div by 0)"
                # multipy by output channels
                processor_count = (
                    relevant_channels // passes
                )  # future relevant_channels are output_channels / passes

                if is_1d:  # noqa: SIM108
                    relevant_channels = weights_shape[0]
                else:
                    relevant_channels = weights_shape[0]

                # i dont know when to shift the processors
                # TODO: generalize this
                # maybe with _factor_factor
                # processor_shift = 32 if len(layers) == 4 else 0
                _factor_factor = nxnode.get("_factor_factor", [1])
                _full_factor = nxnode.get("_full_factor", _factor_factor)[0]
                _resulting_shift = int(math.log2(1 if is_1d else _full_factor))
                if _resulting_shift < 0:
                    # output shift is barely used if not at all
                    # ly.output_shift = _resulting_shift
                    # ly.output_processors = self.processor_format_helper(
                    #     amount_on=processor_count,
                    #     shift=-_resulting_shift,
                    # )
                    _resulting_shift = 0  # set it to 0 for the processors
                ly.processors = self.processor_format_helper(
                    amount_on=processor_count, shift=_resulting_shift
                )
                logging.warning(
                    f"proc: {processor_count}, pass:{passes}, "
                    f"rel_chan:{relevant_channels}, weights:{weights_shape}"
                    f"factor:{_factor_factor}, fullfactor:{_full_factor}"
                    "results in proc count/shift:"
                    f"{processor_count}/{int(math.log2(_full_factor))}"
                )

                layers.append(ly)

        # 2024-10-20 dont shift at all
        # try to use 32 if possible
        # if layers[-1].activation in [None] and layers[-1].activate in [
        #     None
        # ]:  # TODO: or 'none',
        #     # do we always want 32 bit? (yes, because of --softmax)
        #     layers[-1].output_width = 32
        # layers[-1].output_shift -= 1  # TODO: check if this is correct

        transformed_model = onnx.helper.make_model(
            trf_graph.onnx(),
            producer_name="netdeployonnx",
            doc_string=f"transformed by netdeployonnx for "
            f"{self.__class__.__module__}.{self.__class__.__name__}",
        )

        cfg = AI8XizeConfig(
            arch="unknown-arch", dataset="unknown-dataset", layers=layers
        )
        return (
            dict(cfg.model_dump(exclude_unset=True)),
            locked_config,
            input_shape,
            transformed_model,
        )


def set_lregs_to_core(lregs: list[any], core: CNNx16Core):
    for lreg in lregs:
        (quad, layeridx, reg, val, force_write, no_verify, comment) = lreg
        # logging.debug(f"used layer {quad}.{layeridx}")
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


class LayerInformation:
    def __init__(self, layer_count: int):
        self.input_dim = [] * layer_count


def get_layer_information(
    model: onnx.ModelProto,
) -> LayerInformation:
    layerinfo = LayerInformation(0)
    # weights = [
    #     []
    #     for layer in range(layer_count)
    # ]

    # for layeridx in range(layer_count):
    #     layerinfo.input_dim[layeridx] = input_dim[:-2]
    #     assert len(layerinfo.input_dim[layeridx]) == 2, "this dim is 2, even for 1D"

    return layerinfo
