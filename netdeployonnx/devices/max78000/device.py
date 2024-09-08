import asyncio
import logging
from collections.abc import Iterable
from contextlib import asynccontextmanager, suppress
from typing import Any

import onnx
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

import netdeployonnx.devices.max78000.cnn_constants as cnn_constants
from netdeployonnx.devices import Device, Metrics
from netdeployonnx.devices.max78000.cnn_constants import (
    CNNx16_n_CTRL_CLK_EN,
    CNNx16_n_CTRL_CNN_EN,
    CNNx16_n_CTRL_EXT_SYNC_BIT2,
    CNNx16_n_CTRL_MEXPRESS,
)
from netdeployonnx.devices.max78000.core import CNNx16Core
from netdeployonnx.devices.max78000.device_transport import serialhandler
from netdeployonnx.devices.max78000.device_transport.commands import Commands
from netdeployonnx.devices.max78000.device_transport.protobuffers import main_pb2
from netdeployonnx.devices.max78000.graph_synthesizer import synth_to_core_ir
from netdeployonnx.devices.max78000.graph_transformer import transform_graph

try:
    from itertools import batched
except ImportError:
    from itertools import islice

    def batched(iterable, n):
        # batched('ABCDEFG', 3) → ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        iterator = iter(iterable)
        while batch := tuple(islice(iterator, n)):
            yield batch


class MAX78000Metrics(Metrics):
    def __init__(self, tty_port: str):
        super().__init__()
        self.tty_port = tty_port
        self.collected_answers = []

    def _get_network_stats(self) -> dict[str, float]:
        stats: dict[str, float] = {}

        one_row = self.collected_answers[-1] if len(self.collected_answers) > 0 else ""

        res = one_row.rstrip().split(",")

        if len(res) == 12:
            IDX_ENERGY_USED = 0  # noqa: N806, F841
            IDX_TIME = 1  # noqa: N806
            IDX_IDLE_POWER = 2  # noqa: N806, F841
            IDX_ACTIVE_POWER = 3  # noqa: N806, F841
            IDX_USED_POWER = 4  # noqa: N806

            def extract_stage(
                stage: list[str],
            ) -> tuple[float, float, float, float, float]:
                used_energy = float(stage[0])
                used_time = float(stage[1])
                idle_power = float(stage[2])
                active_power = float(stage[3])
                diff_power = active_power - idle_power

                return used_energy, used_time, idle_power, active_power, diff_power

            # TIMES_OPERATION = 100

            measure_kernels = extract_stage(res[0:4])
            measure_input = extract_stage(res[4:8])
            measure_input_convolution = extract_stage(res[8:12])
            # only possible for non-FIFO mode
            calculated_convolutions = [
                measure_input_convolution[idx] - measure_input[idx]
                for idx in range(len(measure_input_convolution))
            ]

            X_TO_MICRO_WATTS = 1e6  # noqa: N806
            X_TO_MICRO_SECONDS = 1e6  # noqa: N806
            X_TO_MICRO_JOULES = 1e6  # noqa: N806

            measurements = {
                "weights_loading": (
                    measure_kernels[IDX_USED_POWER] * X_TO_MICRO_WATTS,
                    measure_kernels[IDX_TIME] * X_TO_MICRO_SECONDS,
                    measure_kernels[IDX_ENERGY_USED] * X_TO_MICRO_JOULES,
                ),
                "input_loading": (
                    measure_input[IDX_USED_POWER] * X_TO_MICRO_WATTS,
                    measure_input[IDX_TIME] * X_TO_MICRO_SECONDS,
                    measure_input[IDX_ENERGY_USED] * X_TO_MICRO_JOULES,
                ),
                "convolution": (
                    calculated_convolutions[IDX_USED_POWER] * X_TO_MICRO_WATTS,
                    calculated_convolutions[IDX_TIME] * X_TO_MICRO_SECONDS,
                    calculated_convolutions[IDX_ENERGY_USED] * X_TO_MICRO_JOULES,
                ),
                "all": (
                    sum(
                        [
                            measure_kernels[IDX_USED_POWER],
                            measure_input_convolution[IDX_USED_POWER],
                        ]
                    )
                    * X_TO_MICRO_WATTS,
                    sum(
                        [
                            measure_kernels[IDX_TIME],
                            measure_input_convolution[IDX_TIME],
                        ]
                    )
                    * X_TO_MICRO_SECONDS,
                    sum(
                        [
                            measure_kernels[IDX_ENERGY_USED],
                            measure_input_convolution[IDX_ENERGY_USED],
                        ]
                    )
                    * X_TO_MICRO_JOULES,
                ),
            }

            for measurement_name, (
                micro_watt,
                micro_s,
                micro_joules,
            ) in measurements.items():
                stats[f"uW_per_{measurement_name}"] = round(max(0, micro_watt), 2)
                stats[f"us_per_{measurement_name}"] = round(max(0, micro_s), 2)
                # stats[f"uJ_per_{measurement_name}"] = (
                #     round(micro_s * micro_watt, 2) * 1e-6
                # )
                stats[f"uJ_per_{measurement_name}"] = round(max(0, micro_joules), 2)

        return stats

    @asynccontextmanager
    async def get_serial(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await serialhandler.open_serial_connection(
            url=self.tty_port, baudrate=1_500_000
        )
        yield reader, writer
        writer.close()
        try:  # noqa: SIM105
            await asyncio.wait_for(writer.wait_closed(), timeout=0.5)
        except TimeoutError:
            # if it does not return, i dont fcare
            pass

    async def collect(self, timeout: float = 1) -> str:  # noqa: ASYNC109
        async with self.get_serial() as (reader, writer):
            try:
                data = await asyncio.wait_for(reader.read(100), timeout=timeout)
                answer = data.decode()
                self.collected_answers.append(answer)
                return answer
            except asyncio.TimeoutError:
                return ""

    async def set_mode(self, mode: str) -> None:
        """
        set the mode of the measurement device
        v - voltage mode
        i - current mode
        w - power mode
        t - triggered mode (used with the ai8xize.py --energy option)
        s - system mode (used with the ai8xize.py --energy option)
        """
        modes = {
            "power": "w",
            "voltage": "v",
            "current": "i",
            "triggered": "t",
            "system": "s",
        }
        if mode not in modes:
            raise ValueError(f"mode {mode} not supported")
        character: str = modes[mode]
        async with self.get_serial() as (reader, writer):
            writer.write(character.encode())
            await writer.drain()

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
        self.commands = Commands()
        # get communication task
        self.handle_serial_task = None
        self.handle_serial_task_closed = None

    def __del__(self):
        # make sure the task is cancelled
        if self.handle_serial_task:
            self.handle_serial_task.cancel()

    async def get_handle_serial_task(self, loop: None) -> asyncio.Task:
        if self.handle_serial_task is None:
            if self.handle_serial_task_closed is not None:
                # we have a problem, it has to be closed
                raise RuntimeError("handle_serial_task_closed is not None")
            else:
                self.handle_serial_task_closed = asyncio.Future()
            self.handle_serial_task = loop.create_task(
                serialhandler.handle_serial(
                    self.commands,
                    tty=self.port,
                    timeout=4,
                    closed_future=self.handle_serial_task_closed,
                )
            )
        return self.handle_serial_task

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
            ("ACTION", main_pb2.ActionEnum.RUN_CNN_ENABLE, 0),
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
            "load_input": [],  # TODO: add input loading
            "cnn_start": self.cnn_start(layout),
            # TODO: fetch results
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

    def transform_instructions(
        self,
        commands: Commands,
        instructions: list[tuple[any]],
    ) -> list[main_pb2.ProtocolMessage]:
        msg = commands.new_message()
        messages = [msg]
        action_not_set: bool = True
        for instruction in instructions:
            if isinstance(instruction, str):
                # this is just a comment, so we ignore it (or maybe log / debug it?)
                logging.debug(f"comment: {instruction}")
            elif isinstance(instruction, tuple):  # either reg or mem access
                if False:
                    ...
                elif len(instruction) == 2:
                    instruction_dest, instruction_value = instruction
                    if isinstance(instruction_value, int):  # reg access
                        reg_addr: int = cnn_constants.registers.get(instruction_dest, 0)
                        msg.payload.registers.append(
                            main_pb2.SetRegister(
                                address=reg_addr,
                                preserve_mask=0,
                                set_mask=instruction_value,
                                size=main_pb2.Regsize.UINT32,
                                readable=False,  # we dont care
                            )
                        )
                    elif isinstance(instruction_value, (list, bytes)):  # mem access
                        msg.payload.memory.append(
                            main_pb2.SetMemoryContent(
                                address=instruction_dest,
                                data=instruction_value,
                                setAddr=True,  # TODO: do we set it?
                            )
                        )
                elif len(instruction) == 3:
                    # we may have an action instead of an instruction
                    if action_not_set:
                        # use available msg
                        action_not_set = False
                    else:
                        # create a new message and add
                        msg = commands.new_message()
                        messages.append(msg)
                    action, action_value, action_mask = instruction
                    msg.action.execute_measurement = action_value
                    logging.debug(f"action: {action} {action_value} {action_mask}")
                else:
                    raise ValueError(f"invalid instruction: {instruction}")
            else:
                raise ValueError(f"invalid instruction: {instruction}")

        return messages

    async def execute(self, instructions: Any, metrics: Metrics) -> Any:  # noqa: C901
        if type(instructions) is not list:
            # something failed in self.compile_instructions?
            raise ValueError("instructions must be a list")
        assert isinstance(instructions, list)
        assert len(instructions) > 0 and isinstance(instructions[0], dict)
        result = []

        await metrics.set_mode("triggered")

        task = await self.get_handle_serial_task(loop=asyncio.get_event_loop())
        await asyncio.sleep(0)  # make sure the task is started
        assert task

        commands = self.commands
        messages_per_stage: list[tuple[str, list[main_pb2.ProtocolMessage]]] = []
        for stage in instructions:
            if isinstance(stage, dict):
                stage_instructions = stage.get("instructions", [])
                stage_name = stage.get("stage", "")
                messages_per_stage.append(
                    (
                        stage_name,
                        self.transform_instructions(commands, stage_instructions),
                    )
                )

        try:
            tasks = {}
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
            ) as progress:
                for stagename, stagemsg in messages_per_stage:
                    tasks[stagename] = progress.add_task(stagename, total=1)
                for stagename, stage_messages in messages_per_stage:
                    for stagemsg in stage_messages:
                        messages = commands.split_message(stagemsg)
                        if stagename == "cnn_load_weights":
                            messages = messages[40:]
                        progress.reset(tasks[stagename], total=len(messages))
                        batchsize = 10
                        for batch in batched(enumerate(messages), batchsize):
                            await asyncio.wait_for(
                                commands.send_batch(
                                    submessage for index_submessage, submessage in batch
                                ),
                                timeout=2 * batchsize,  # TODO: change to 1 per msg
                            )  # these can throw a CancelledError
                            readback: list = []
                            if isinstance(readback, Iterable):
                                result.extend(readback)
                            progress.advance(tasks[stagename], len(batch))
        except asyncio.exceptions.CancelledError:
            raise Exception("message cancelled")

        await metrics.collect()
        # collect is needed so that .as_dict works
        # metrics is done by run_onnx

        return result  # maybe we have a readback?

    async def free(self):
        "free the device; this means close the handle_serial task"
        with suppress(Exception):
            if self.handle_serial_task:
                self.commands.set_exit_request = True
                await self.handle_serial_task
                self.handle_serial_task = None
                await self.handle_serial_task_closed
                self.handle_serial_task_closed = None
        await asyncio.sleep(0.1)
