import asyncio
from typing import Any

import onnx

import netdeployonnx.devices.max78000.cnn_constants as cnn_constants
from netdeployonnx.devices import Device, Metrics
from netdeployonnx.devices.max78000.cnn_constants import (
    CNNx16_n_CTRL_CLK_EN,
    CNNx16_n_CTRL_CNN_EN,
    CNNx16_n_CTRL_EXT_SYNC_BIT2,
    CNNx16_n_CTRL_MEXPRESS,
)
from netdeployonnx.devices.max78000.core import CNNx16Core
from netdeployonnx.devices.max78000.graph_synthesizer import synth_to_core_ir
from netdeployonnx.devices.max78000.graph_transformer import transform_graph


class MAX78000Metrics(Metrics):
    def __init__(self, tty_port: str):
        super().__init__()
        self.tty_port = tty_port

    def _get_network_stats(self) -> dict[str, float]:
        stats: dict[str, float] = {}

        measurements = {
            "convolution": (0.1, 0.1),
            "input_loading": (0.1, 0.1),
            "weights_loading": (0.1, 0.1),
            "all": (0.1, 0.1),
        }

        for measurement_name, (nano_watt, nano_s) in measurements.items():
            stats[f"nW_per_{measurement_name}"] = round(nano_watt, 2)
            stats[f"ns_per_{measurement_name}"] = round(nano_s, 2)
            stats[f"nJ_per_{measurement_name}"] = round(nano_s * nano_watt, 2)

        return stats

    def as_dict(self) -> dict:
        d = super().as_dict()
        d.update(self._get_network_stats())
        return d


class MAX78000(Device):
    def __init__(
        self,
        name: str = "MAX78000",
        manufacturer: str = "Maxim Integrated",
        firmware_version: str = "?",
        communication_port: str = "",
        energy_port: str = "",
    ):
        super().__init__(
            name,
            manufacturer,
            firmware_version,
            comm_port=communication_port,
            energy_port=energy_port,
        )

    @classmethod
    def create_device_from_name_and_ports(
        cls,
        model_name: str,
        communication_port: str,
        energy_port: str,
    ) -> Device:
        return MAX78000(
            model_name,
            "Maxim Integrated",
            "?",
            communication_port,
            energy_port,
        )

    async def layout_transform(self, model: onnx.ModelProto) -> Any:
        # now start the layout transformation to IR

        # first we need to retransform the graph so we have a workable graph
        transformed_graph: onnx.ModelProto = transform_graph(model.graph)

        # then to Immediate Representation
        core_ir: CNNx16Core = synth_to_core_ir(transformed_graph)

        return core_ir

    def cnn_enable(self, layout: Any) -> Any:
        """ """
        if layout is None:
            return []
        # TODO: do something based on the layout
        return [
            # ("ACTION",main_pb2.ActionEnum.RUN_CNN_ENABLE),
            ("ACTION", 4, 0),
        ]

    def cnn_init(self, layout: "CNNx16Core") -> Any:
        """
        Initialize the CNN core
        """
        if layout is None:
            return []
        ret = layout.instructions_init()
        for quadrant in layout.quadrants.values():
            if not quadrant.unused:
                ret += quadrant.instructions_init()
        ret.append("")  # TODO: this is only required for the synth to c right now
        return ret

    def cnn_configure(self, layout: Any) -> Any:
        """
        Configure the CNN core
        """
        ret = []
        if layout is None:
            return []

        for core in []:
            ret += layout.quadrants[core].instructions_configure()
        for layer in range(16):
            for core in range(4):
                ret += layout[core, layer].instructions_configure()
        return ret

    def cnn_start(self, layout: Any) -> Any:
        """
        Start the computation
        """
        if layout is None:
            return []
        # TOOD: check if we need to start all quadrants
        return [
            (
                "CNNx16_0_CTRL",
                CNNx16_n_CTRL_MEXPRESS
                | CNNx16_n_CTRL_EXT_SYNC_BIT2
                | CNNx16_n_CTRL_CLK_EN,
            ),  # Enable, but hold back master
            (
                "CNNx16_1_CTRL",
                CNNx16_n_CTRL_MEXPRESS
                | CNNx16_n_CTRL_EXT_SYNC_BIT2
                | CNNx16_n_CTRL_CLK_EN
                | CNNx16_n_CTRL_CNN_EN,
            ),  # Start SM
            (
                "CNNx16_2_CTRL",
                CNNx16_n_CTRL_MEXPRESS
                | CNNx16_n_CTRL_EXT_SYNC_BIT2  #  noqa: F405
                | CNNx16_n_CTRL_CLK_EN
                | CNNx16_n_CTRL_CNN_EN,
            ),  # Start SM
            (
                "CNNx16_3_CTRL",
                CNNx16_n_CTRL_MEXPRESS
                | CNNx16_n_CTRL_EXT_SYNC_BIT2
                | CNNx16_n_CTRL_CLK_EN
                | CNNx16_n_CTRL_CNN_EN,
            ),  # Start SM
            (""),
            (
                "CNNx16_0_CTRL",
                CNNx16_n_CTRL_MEXPRESS | CNNx16_n_CTRL_CLK_EN | CNNx16_n_CTRL_CNN_EN,
            ),  # Start Master
        ]

    def cnn_load_bias(self, layout: Any) -> Any:
        """
        Load the bias values
        """
        ret = []
        if layout is None:
            return []

        for quad in range(4):
            bias_addr_name = f"CNNx16_{quad}_BIAS"
            bias_addr = cnn_constants.memory[bias_addr_name]
            ret.append((bias_addr, layout[quad].bias))
        return ret

    def cnn_load_weights(self, layout: Any) -> Any:
        """
        Load the weights
        """
        ret = []

        if layout is None:
            return []

        for quad in range(4):
            for proc in range(16):
                mram_addr = cnn_constants.memory[f"CNNx16_{quad}_P{proc}_MRAM"]
                for kernel_addr, kernel_data in (
                    layout[quad].processors[proc].kernels.items()
                ):
                    ret.append((kernel_addr + mram_addr, kernel_data))
        return ret

    async def compile_instructions(
        self, layout: Any
    ) -> list[dict[str, list["RegisterAccess | MemoryAccess"]]]:  # noqa: F821
        """
        Compile the instructions for the given layout
        """
        instructions = []
        assert isinstance(layout, CNNx16Core) or layout is None
        for stage, instructions_per_stage in {
            "cnn_enable": self.cnn_enable(layout),
            "cnn_init": self.cnn_init(layout),
            "cnn_load_weights": self.cnn_load_weights(layout),
            "cnn_load_bias": self.cnn_load_bias(layout),
            "cnn_configure": self.cnn_configure(layout),
            "load_input": [],
            "cnn_start": self.cnn_start(layout),
            "done": [],
        }.items():
            instructions.append(
                {"stage": stage, "instructions": instructions_per_stage}
            )
        return instructions

    async def acquire_metrics(self) -> Any:
        """
        Start collecting metrics from the device
        """
        # TODO: this is a placeholder, get from config
        return MAX78000Metrics("/dev/ttyACM0")

    async def execute(self, instructions: Any, metrics: Metrics) -> Any:
        if type(instructions) is not list:
            # something failed in self.compile_instructions?
            raise ValueError("instructions must be a list")

        await asyncio.sleep(0.1)

        assert isinstance(instructions, list)
        assert len(instructions) > 0 and isinstance(instructions[0], dict)

        return [
            1,
            2,
            3,
        ]
