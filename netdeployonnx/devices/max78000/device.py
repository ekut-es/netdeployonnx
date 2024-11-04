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
import asyncio
import logging
import struct
from collections.abc import Iterable
from contextlib import asynccontextmanager, suppress
from typing import Any

import onnx
from tqdm import tqdm

import netdeployonnx.devices.max78000.cnn_constants as cnn_constants
from netdeployonnx.devices import Device, Metrics
from netdeployonnx.devices.max78000.core import CNNx16Core
from netdeployonnx.devices.max78000.device_transport import serialhandler
from netdeployonnx.devices.max78000.device_transport.commands import Commands, crc
from netdeployonnx.devices.max78000.device_transport.protobuffers import main_pb2
from netdeployonnx.devices.max78000.graph_synthesizer import synth_to_core_ir
from netdeployonnx.devices.max78000.graph_transformer import transform_graph

try:
    from itertools import batched
    # batched was introduced in python 3.11
except ImportError:
    from itertools import islice

    # backport
    # TODO: use a library
    def batched(iterable, n):
        # batched('ABCDEFG', 3) → ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        iterator = iter(iterable)
        while batch := tuple(islice(iterator, n)):
            yield batch


def str_to_bool(s: str) -> bool | None:
    return {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "t": True,
        "f": False,
        "yes": True,
        "no": False,
    }.get(s.lower(), None)


class MAX78000Metrics(Metrics):
    def __init__(self, tty_port: str):
        super().__init__()
        self.tty_port = tty_port
        self.collected_answers = []
        self.collected_answers_index = 0
        self.reader = None
        self.writer = None

    def _get_network_stats(self) -> list[dict[str, float]]:
        results = []

        one_row = self.collected_answers[-1] if len(self.collected_answers) > 0 else ""
        for one_row in self.collected_answers[self.collected_answers_index :]:
            stats: dict[str, float] = {}
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

                    # energy, time and power cannot be negative, but we dont abs,
                    # so we dont remove information

                    # used_energy = abs(used_energy)
                    # used_time = abs(used_time)
                    # idle_power = abs(idle_power)
                    # active_power = abs(active_power)

                    return used_energy, used_time, idle_power, active_power, diff_power

                # TIMES_OPERATION = 100

                measure_kernels = extract_stage(res[0:4])
                measure_input = extract_stage(res[4:8])
                measure_input_inference = extract_stage(res[8:12])
                # only possible for non-FIFO mode
                calculated_inferences = [
                    measure_input_inference[idx] - measure_input[idx]
                    for idx in range(len(measure_input_inference))
                ]

                X_TO_MICRO_WATTS = 1e6  # noqa: N806
                X_TO_MICRO_SECONDS = 1e6  # noqa: N806
                X_TO_MICRO_JOULES = 1e6  # noqa: N806

                measurements = {
                    "weights_loading": (
                        measure_kernels[IDX_USED_POWER] * X_TO_MICRO_WATTS,
                        measure_kernels[IDX_ACTIVE_POWER] * X_TO_MICRO_WATTS,
                        measure_kernels[IDX_TIME] * X_TO_MICRO_SECONDS,
                        measure_kernels[IDX_ENERGY_USED] * X_TO_MICRO_JOULES,
                    ),
                    "input_loading": (
                        measure_input[IDX_USED_POWER] * X_TO_MICRO_WATTS,
                        measure_input[IDX_ACTIVE_POWER] * X_TO_MICRO_WATTS,
                        measure_input[IDX_TIME] * X_TO_MICRO_SECONDS,
                        measure_input[IDX_ENERGY_USED] * X_TO_MICRO_JOULES,
                    ),
                    "inference": (
                        calculated_inferences[IDX_USED_POWER] * X_TO_MICRO_WATTS,
                        calculated_inferences[IDX_ACTIVE_POWER] * X_TO_MICRO_WATTS,
                        calculated_inferences[IDX_TIME] * X_TO_MICRO_SECONDS,
                        calculated_inferences[IDX_ENERGY_USED] * X_TO_MICRO_JOULES,
                    ),
                    "load_and_inference": (
                        measure_input_inference[IDX_USED_POWER] * X_TO_MICRO_WATTS,
                        measure_input_inference[IDX_ACTIVE_POWER] * X_TO_MICRO_WATTS,
                        measure_input_inference[IDX_TIME] * X_TO_MICRO_SECONDS,
                        measure_input_inference[IDX_ENERGY_USED] * X_TO_MICRO_JOULES,
                    ),
                    "all": (
                        sum(
                            [
                                measure_kernels[IDX_USED_POWER],
                                measure_input_inference[IDX_USED_POWER],
                            ]
                        )
                        * X_TO_MICRO_WATTS,
                        sum(
                            [
                                measure_kernels[IDX_ACTIVE_POWER],
                                measure_input_inference[IDX_ACTIVE_POWER],
                            ]
                        )
                        * X_TO_MICRO_WATTS,
                        sum(
                            [
                                measure_kernels[IDX_TIME],
                                measure_input_inference[IDX_TIME],
                            ]
                        )
                        * X_TO_MICRO_SECONDS,
                        sum(
                            [
                                measure_kernels[IDX_ENERGY_USED],
                                measure_input_inference[IDX_ENERGY_USED],
                            ]
                        )
                        * X_TO_MICRO_JOULES,
                    ),
                }

                for measurement_name, (
                    micro_watt,
                    micro_watt_abs,
                    micro_s,
                    micro_joules,
                ) in measurements.items():
                    # disable clipping, so no information is removed
                    stats[f"uW_per_{measurement_name}"] = round((micro_watt), 2)
                    stats[f"us_per_{measurement_name}"] = round((micro_s), 2)
                    stats[f"uW_abs_per_{measurement_name}"] = round((micro_watt_abs), 2)
                    # stats[f"uJ_per_{measurement_name}"] = (
                    #     round(micro_s * micro_watt, 2) * 1e-6
                    # )
                    stats[f"uJ_per_{measurement_name}"] = round((micro_joules), 2)
                results.append(stats)
            else:
                print("we found size of res=", len(res))

        return results

    @asynccontextmanager
    async def get_serial(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        # reader, writer = await serialhandler.open_serial_connection(
        #     url=self.tty_port, baudrate=1_500_000
        # )
        yield self.reader, self.writer
        # self.writer.close()
        # try:  # noqa: SIM105
        #     await asyncio.wait_for(self.writer.wait_closed(), timeout=0.5)
        # except TimeoutError:
        #     # if it does not return, i dont fcare
        #     pass

    async def collect(self, timeout: float = 2) -> str:  # noqa: ASYNC109
        async with self.get_serial() as (reader, writer):
            try:
                data = await asyncio.wait_for(reader.read(2000), timeout=timeout)
                answer = data.decode()
                for line in answer.split("\n"):
                    if line.strip():
                        self.collected_answers.append(line.strip())
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

    async def start(self):
        reader, writer = await serialhandler.open_serial_connection(
            url=self.tty_port, baudrate=1_500_000
        )
        self.reader = reader
        self.writer = writer

    async def stop(self):
        await super().stop()
        self.writer.close()
        try:  # noqa: SIM105
            await asyncio.wait_for(self.writer.wait_closed(), timeout=0.5)
        except TimeoutError:
            # if it does not return, i dont fcare
            pass

    def as_dict(self) -> dict:
        d = super().as_dict()
        d["metrics"] = self._get_network_stats()
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
        self.FLASH_PAGE_SIZE = 8192
        self.BIAS_SIZE = 256

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
                    timeout=15,
                    closed_future=self.handle_serial_task_closed,
                )
            )
            await asyncio.sleep(0.01)
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

    async def layout_transform(self, model: onnx.ModelProto) -> CNNx16Core:
        # now start the layout transformation to IR

        # first we need to retransform the graph so we have a workable graph
        transformed_graph: onnx.ModelProto = transform_graph(model.graph)

        # then to Immediate Representation
        core_ir: CNNx16Core = synth_to_core_ir(transformed_graph)

        for quadrant in range(4):
            core_ir[quadrant].layer_count = core_ir[quadrant].max_used_layer
        return core_ir

    def cnn_enable(self, layout: CNNx16Core) -> Any:
        """ """
        if layout is None:
            return []

        cnnclksel = 0
        cnnclkdiv = 4  #  4 is 1
        if hasattr(layout, "specialconfig"):
            cnnclksel = int(layout.specialconfig.get("GCR_pclkdiv.cnnclksel", "0"), 0)
            cnnclkdiv = int(layout.specialconfig.get("GCR_pclkdiv.cnnclkdiv", "4"), 0)
            assert cnnclksel in [0, 1], "cnnclksel has to be in [0...1]"
            assert cnnclkdiv in range(7 + 1), "cnnclksel has to be in [0...7]"  # 0..7
        default_enable = [
            # (
            #     "CONFIG",
            #     "execute_reset",
            #     True,
            # ),
            (
                "ACTION",
                main_pb2.ActionEnum.RUN_CNN_ENABLE,
                (cnnclkdiv << 1) + cnnclksel,  # this is PCLK + CLKDIV 1
            ),
        ]
        return default_enable

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

    def cnn_configure(self, layout: CNNx16Core) -> Any:
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

    def cnn_start(self, layout: CNNx16Core) -> Any:
        """
        Start the computation
        """
        if layout is None:
            return []
        # TODO: check if we need to start all quadrants
        # as this works, we may want to transition to
        # measurements instead of just starting a measurement
        need_to_flash = True
        if hasattr(layout, "specialconfig"):
            # if not available, retain value
            need_to_flash = str_to_bool(
                layout.specialconfig.get("__reflash", str(need_to_flash))
            )
        reboot_start = [  # noqa: F841
            (
                "CONFIG",
                "execute_reset",
                True,
            ),
        ]

        default_start = [
            (
                "ACTION",
                main_pb2.ActionEnum.MEASUREMENT,
                0,
            ),
        ]
        full_start = []
        # check if we reflashed?
        # if need_to_flash:
        #     full_start.extend(reboot_start)
        full_start.extend(default_start)
        return full_start

    def cnn_load_bias(self, layout: CNNx16Core) -> Any:
        """
        Load the bias values
        """
        ret = []
        if layout is None:
            return []
        need_to_flash = True
        if hasattr(layout, "specialconfig"):
            # if not available, retain value
            need_to_flash = str_to_bool(
                layout.specialconfig.get("__reflash", str(need_to_flash))
            )

        need_to_direct_write = False
        if need_to_flash:
            # we layout our bias page
            bias_page = b""
            BIAS_SIZE = self.BIAS_SIZE  # noqa: N806

            def pad_BIAS(data: bytes) -> bytes:  # noqa: N802
                return (data + b"\0" * BIAS_SIZE)[:BIAS_SIZE]

            for quad in range(4):
                bias_addr_name = f"CNNx16_{quad}_BIAS"
                bias_addr = cnn_constants.memory[bias_addr_name]
                bias_page += pad_BIAS(layout[quad].bias)
            if len(bias_page) < self.FLASH_PAGE_SIZE:
                # pad last page
                bias_page = bias_page + b"\0" * self.FLASH_PAGE_SIZE
                bias_page = bias_page[: self.FLASH_PAGE_SIZE]
            # now write the bias as flash instr
            ret.append((main_pb2.Variable.BIAS_0, [bias_page]))
        elif need_to_direct_write:
            for quad in range(4):
                bias_addr_name = f"CNNx16_{quad}_BIAS"
                bias_addr = cnn_constants.memory[bias_addr_name]
                ret.append((bias_addr, layout[quad].bias))
        else:
            # dont need to flash?
            # leave it.
            pass
        return ret

    def maximum_network_size_okay(self, bytecount: int):
        # we assume 32?
        return bytecount < 32 * self.FLASH_PAGE_SIZE

    def cnn_load_weights(self, layout: CNNx16Core) -> Any:  # noqa: C901
        """
        Load the weights
        """
        ret = []

        if layout is None:
            return []

        # TODO: we should flash the entries
        # but before, check if we need to flash (or check atleast the force option)

        need_to_flash = True
        if hasattr(layout, "specialconfig"):
            # if not available, retain value
            need_to_flash = str_to_bool(
                layout.specialconfig.get("__reflash", str(need_to_flash))
            )
        need_to_direct_write = False
        init_pattern = False
        if need_to_flash:
            weight_pages = []
            # we layout our weight page
            data_block = b""
            for quad in range(4):
                for proc in range(16):
                    mram_addr = cnn_constants.memory[f"CNNx16_{quad}_P{proc}_MRAM"]
                    for kernel_addr, kernel_data in (
                        layout[quad].processors[proc].kernels.items()
                    ):
                        data_block += struct.pack("<I", mram_addr + kernel_addr)
                        data_block += struct.pack("<I", len(kernel_data) // 4)
                        assert isinstance(
                            kernel_data, bytes
                        ), "expected bytes for kernel_data"
                        assert (
                            len(kernel_data) % 4
                        ) == 0, "assumed kernel_data is multiple of 4"
                        data_block += kernel_data  # assuming kernel_data is bytes
            while len(data_block) > 0:
                weight_page = data_block[: self.FLASH_PAGE_SIZE]
                if len(weight_page) < self.FLASH_PAGE_SIZE:
                    # pad last page
                    weight_page = weight_page + b"\0" * self.FLASH_PAGE_SIZE
                    weight_page = weight_page[: self.FLASH_PAGE_SIZE]
                data_block = data_block[self.FLASH_PAGE_SIZE :]
                weight_pages.append(weight_page)
            # now write the bias as flash instr
            ret.append((main_pb2.Variable.WEIGHTS, weight_pages))
            assert self.maximum_network_size_okay(
                sum([len(p) for p in weight_pages])
            ), "Weights too big"
        elif need_to_direct_write:
            for quad in range(4):
                for proc in range(16):
                    mram_addr = cnn_constants.memory[f"CNNx16_{quad}_P{proc}_MRAM"]
                    for kernel_addr, kernel_data in (
                        layout[quad].processors[proc].kernels.items()
                    ):
                        ret.append((kernel_addr + mram_addr, kernel_data))
        elif init_pattern:
            ret.append(
                (
                    "ACTION",
                    main_pb2.ActionEnum.INIT_WEIGHTS_PATTERN1,
                    0,
                )
            )
        else:
            # we dont need to flash, so everything okay
            pass
        return ret

    def cnn_load_input(self, layout: CNNx16Core) -> Any:
        ret = []
        pages = []
        if 0:
            input_data = b"\0" * 1000  # is it always 1000?
            page = b""
            page += struct.pack("<I", 0x50400000)
            page += struct.pack("<I", len(input_data) // 4)
            pages.append(page)

        if pages:
            ret.append((main_pb2.Variable.INPUT, pages))
        return ret

    def cnn_fetch_results(self, layout: CNNx16Core) -> Any:
        # fetch results means get data via action
        ret = []
        # either RUN_CNN_UNLOAD (but this does not send a getmemory / readmemory)
        # ret.append(
        #     ("READ", 0x50404000, 16)
        # )  # CNN-memory: CNNx16_0_SRAM = 0x5040_0000 + 0x04000
        # ret.append(
        #     ("READ", 0x5040C000, 16)
        # )  # CNN-memory: CNNx16_0_SRAM = 0x5040_0000 + 0x0C000
        # ret.append(
        #     ("READ", 0x50414000, 8)
        # )  # CNN-memory: CNNx16_0_SRAM = 0x5040_0000 + 0x14000
        return ret

    async def compile_instructions(
        self, layout: CNNx16Core
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
            "load_input": self.cnn_load_input(layout),
            "cnn_start": self.cnn_start(layout),
            "done": self.cnn_fetch_results(layout),
        }.items():
            instructions.append(
                {"stage": stage, "instructions": instructions_per_stage}
            )
        return instructions

    async def acquire_metrics(self) -> Any:
        """
        Start collecting metrics from the device
        """
        metrics = MAX78000Metrics(self.energy_port)
        await metrics.start()
        return metrics

    def transform_instructions(  # noqa: C901
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
                    # this should be a memaccessor or a register, but what about a flash?  # noqa: E501
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
                    # mem access is either list or bytes
                    elif isinstance(instruction_value, (list, bytes)):
                        # instead of directly writing, we do flashing
                        direct_memaccess = (
                            instruction_dest not in main_pb2.Variable.values()
                        )
                        if direct_memaccess:
                            msg.payload.memory.append(
                                main_pb2.SetMemoryContent(
                                    address=instruction_dest,
                                    data=instruction_value,
                                    setAddr=True,  # TODO: do we set it?
                                )
                            )
                        else:
                            variable = instruction_dest
                            # we do flashing, but one page at a time
                            for page_offset, page_data in enumerate(instruction_value):
                                assert len(page_data) == 8192  # flash page size
                                msg.payload.flash.append(
                                    main_pb2.SetFlash(
                                        var=variable,
                                        address_offset=page_offset
                                        * self.FLASH_PAGE_SIZE,
                                        start_flash=False,
                                        data=page_data,
                                    )
                                )
                                # flash page
                                msg.payload.flash.append(
                                    main_pb2.SetFlash(
                                        var=variable,
                                        address_offset=page_offset
                                        * self.FLASH_PAGE_SIZE,
                                        crc=crc(page_data),
                                        start_flash=True,
                                    )
                                )
                elif len(instruction) == 3:
                    # we may have an action instead of an instruction
                    MSG_TYPE, action_value, action_arg = instruction  # noqa: N806
                    if MSG_TYPE == "ACTION":
                        action = MSG_TYPE
                        if action_not_set:
                            # use available msg
                            action_not_set = False
                        else:
                            # create a new message and add
                            msg = commands.new_message()
                            messages.append(msg)
                        action, action_value, action_arg = instruction
                        msg.action.execute_measurement = action_value
                        msg.action.action_argument = action_arg
                        logging.debug(f"action: {action} {action_value} {action_arg}")
                    elif MSG_TYPE == "CONFIG":
                        messages.append(msg)  # add previous message
                        msg = commands.new_message()
                        setattr(msg.configuration, action_value, action_arg)
                        messages.append(msg)  # add our msg
                        msg = commands.new_message()  # create new
                    elif MSG_TYPE == "READ":
                        read_from_addr = action_value
                        read_amount = action_arg

                        messages.append(msg)  # add previous message
                        msg = commands.new_message()
                        msg.payload.read.append(
                            main_pb2.ReadMemoryContent(
                                address=read_from_addr, len=read_amount
                            )
                        )
                        messages.append(msg)  # add our msg
                        msg = commands.new_message()  # create new

                    else:
                        raise NotImplementedError(f"unknown MSG_TYPE {MSG_TYPE}")
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
            batch_size = 10
            stage_bar = tqdm(messages_per_stage, desc="Stages", position=0)
            for stagename, stage_messages in stage_bar:
                stage_bar.set_description(f"Stage: {stagename}")

                for stagemsg in stage_messages:
                    try:
                        messages = commands.split_message(stagemsg)
                    except Exception:
                        logging.exception("could not split message correctly")
                        messages = []

                    with tqdm(
                        range(0, len(messages)),
                        desc="Progress",
                        position=1,
                        leave=False,
                    ) as pbar:
                        for i in range(0, len(messages), batch_size):
                            # Process batch from i to i+batch_size
                            timeout_batch = 2 * batch_size
                            batch = messages[i : i + batch_size]
                            try:
                                for resend_attempts in range(3):
                                    if resend_attempts > 1:
                                        logging.warning(
                                            f"resending batch index {i} for stage "
                                            f"{stagename}"
                                        )
                                    readback = await asyncio.wait_for(
                                        commands.send_batch(batch),
                                        timeout=timeout_batch,
                                    )
                                    if isinstance(readback, Iterable):
                                        result.extend(readback)
                                    elif isinstance(readback, bool):
                                        # check if the batch was a success?
                                        if not readback:
                                            # we have a problem, we may need to resend
                                            logging.info(
                                                "batch result is False, send again."
                                            )
                                        else:
                                            logging.info("batch returned all good")
                                            break
                            except asyncio.CancelledError:
                                pass
                            pbar.update(batch_size)
                stage_bar.update(1)
        except asyncio.exceptions.CancelledError:
            raise Exception("message cancelled")
        logging.debug("collecting metrics...")
        collected_data = await metrics.collect()  # noqa: F841
        # collect is needed so that .as_dict works
        # metrics is done by run_onnx
        logging.debug(f"got {collected_data}")
        logging.debug(f"which is translated to {metrics.as_dict()}")

        return self.prepare_result_for_return(result)

    def prepare_result_for_return(self, result: Any) -> list[any]:
        # as this always depends on the cnn_fetch_results, we should have a overwritable method that does that  # noqa: E501
        # actually, we should get the output shape from the network
        all_bytes = b"".join(
            entry.get("data", b"") if isinstance(entry, dict) else b""
            for entry in sorted(
                result,
                key=lambda x: x["addr"] if isinstance(x, dict) and "addr" in x else x,
            )
        )
        if (len(all_bytes) % 4) == 0:
            num_ints = len(all_bytes) // 4
            result = list(struct.unpack(f"{num_ints}i", all_bytes))
            return result
        # should return json serializable
        return ["unknown byte format? expected 10 integers = 40 bytes"]

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
