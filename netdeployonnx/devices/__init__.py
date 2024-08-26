import abc
import io
import json
import pickle
import time
from contextlib import asynccontextmanager
from typing import Any

import onnx


def convert(input_type: str, input_data: bytes) -> Any:
    if input_type == "bytes" or input_type == "":
        return input_data
    elif input_type == "str":
        return input_data.decode("utf-8")
    elif input_type == "json":
        return json.loads(input_data.decode("utf-8"))
    elif input_type == "dict":
        return pickle.loads(input_data)
    elif input_type == "none":
        return None
    else:
        raise ValueError(
            f"input_type must be one of bytes, str, int, float, but is {input_type}"
        )


class Metrics:
    def __init__(self):
        self.timestamps: tuple[str, float] = []

        self.timestamp("__init__")

    async def stop(self):
        """
        Stop collecting metrics
        """
        self.timestamp("stop")

    def timestamp(self, name: str):
        """
        Add a timestamp to the metrics
        """
        self.timestamps.append((name, time.time()))

    def _get_execution_times(self) -> dict[str, float]:
        """
        Return the execution times as a dictionary from the list of timestamps.

        Returns:
            Dict[str, float]: A dictionary containing the duration of each step
            and the total duration. Keys are step names, and values are durations
            in seconds.
        """
        timestamps: list[tuple[str, float]] = self.timestamps
        sorted_timestamps = sorted(timestamps, key=lambda x: x[1])

        # Calculate total duration
        total_duration = round(sorted_timestamps[-1][1] - sorted_timestamps[0][1], 2)

        result: dict[str, float] = {"total": total_duration}
        prev_timestamp: float | None = None

        for name, timestamp in sorted_timestamps:
            if name == "stop":  # Ignore the stop timestamp
                continue
            if prev_timestamp is not None:
                result[name] = round(timestamp - prev_timestamp, 2)
            prev_timestamp = timestamp

        return result

    def as_dict(self) -> dict:
        """
        Return the metrics as a dictionary
        """
        ret = {}
        ret.update({"deployment_execution_times": self._get_execution_times()})
        return ret


class Device(abc.ABC):
    def __init__(
        self,
        model: str,
        manufacturer: str,
        firmware_version: str,
        comm_port: str = "",
        energy_port: str = "",
    ):
        self.model = model
        self.manufacturer = manufacturer
        self.firmware_version = firmware_version
        self.port = comm_port
        # energy_port is optional, as not every device has an energy monitoring
        # (f.ex. FTHR has none)
        self.energy_port = energy_port

    @abc.abstractmethod
    async def layout_transform(self, model: "onnx.ModelProto") -> Any:
        """
        Transform the model to the device layout
        """
        raise NotImplementedError("layout_transform not implemented")

    @abc.abstractmethod
    async def compile_instructions(
        self, layout: Any
    ) -> list[dict[str, list["RegisterAccess | MemoryAccess"]]]:  # noqa: F821
        """
        Compile the layout to instructions for the device
        """
        raise NotImplementedError("compile_instructions not implemented")

    @abc.abstractmethod
    async def execute(self, instructions: Any, metrics: Metrics) -> Any:
        """
        Execute the model on the device
        """
        raise NotImplementedError("layout_transform not implemented")

    @abc.abstractmethod
    async def acquire_metrics(self) -> Metrics:
        """
        Start collecting metrics from the device
        """
        raise NotImplementedError("acquire_metrics not implemented")

    @classmethod
    @abc.abstractmethod
    def create_device_from_name_and_ports(
        cls,
        model_name: str,
        communication_port: str,
        energy_port: str,
    ) -> "Device":
        raise NotImplementedError("create_device_from_name_and_ports not implemented")

    @asynccontextmanager
    async def collect_metrics(self) -> Metrics:
        """
        Collect metrics from the device
        """
        metrics_object = await self.acquire_metrics()
        yield metrics_object
        await metrics_object.stop()

    def _check_model(self, model: "onnx.ModelProto"):
        """
        Check the model to ensure it is compatible with the device
        """
        # check if the model is valid
        onnx.checker.check_model(model)
        # check if the model is not too big
        if model.ByteSize() > 1024 * 1024 * 10:  # 10MB could be too big
            raise ValueError("Model is too big")
        # check if the model is not too complex
        if len(model.graph.node) > 300:
            raise ValueError("Model is too complex")
        # check if the model is not too deep
        # if len(model.graph.node) > 10:
        #     raise ValueError("Model is too deep")

    async def run_onnx(self, model: "onnx.ModelProto", input: Any) -> dict:
        """ """
        try:
            async with self.collect_metrics() as collected_metrics:
                self._check_model(model)
                collected_metrics.timestamp("_check_model")
                # now we can start transforming
                # the model to the device

                # layout is an IR
                layout = await self.layout_transform(model)
                collected_metrics.timestamp("layout_transform")

                # now we can compile the instructions
                # this could either be the path to a directory / file to be compiled
                # or the instructions themselves as an array / collection
                # just the execute function has to handle this
                instructions = await self.compile_instructions(layout)
                collected_metrics.timestamp("compile_instructions")

                # now we can run the instructions
                result = await self.execute(instructions, collected_metrics)
                collected_metrics.timestamp("execute")
            metrics = {
                "result": result,
            }
            metrics.update(collected_metrics.as_dict())
        except Exception as e:
            import traceback

            traceback.print_exc()
            metrics = {
                "result": None,
                "exception": f"{type(e).__name__}: {e}",
            }
        return metrics

    async def run_async(
        self,
        datatype: str,
        data: (bytes, "onnx.ModelProto"),
        input_type: str,
        input_data: bytes,
    ) -> dict:
        """
        Run the model with the provided data
        """
        converted_input_data: Any = convert(input_type, input_data)
        if not isinstance(datatype, str):
            raise ValueError("datatype must be a string")
        if datatype == "onnxb":
            if not isinstance(data, bytes):
                raise ValueError("data must be bytes")
            if len(data) == 0:
                raise ValueError("Model is empty")
            textio = io.BytesIO(data)
            model = onnx.load(textio)
            return await self.run_onnx(model, converted_input_data)
        elif datatype == "onnx":
            if isinstance(data, onnx.ModelProto):
                model = data
                raise NotImplementedError("Not implemented")
            else:
                raise ValueError("data must be an onnx model")
            return await self.run_onnx(model, converted_input_data)
        else:
            raise NotImplementedError("Only ONNX (onnxb/onnx) is supported")


class DummyDevice(Device):
    async def layout_transform(self, model: onnx.ModelProto) -> Any:
        return None

    async def compile_instructions(
        self, layout: Any
    ) -> list[dict[str, list["RegisterAccess | MemoryAccess"]]]:  # noqa: F821
        instructions = []
        for stage in [
            "cnn_enable",
            "cnn_init",
            "cnn_load_weights",
            "cnn_load_bias",
            "cnn_configure",
            "load_input",
            "cnn_start",
            "done",
        ]:
            instructions.append({"stage": stage, "instructions": []})
        return instructions

    @classmethod
    def create_device_from_name_and_ports(
        cls,
        model_name: str,
        communication_port: str,
        energy_port: str,
    ) -> "Device":
        return DummyDevice(model_name, "No Manufacturer", "unknown_firmware_version")

    async def execute(self, instructions: Any, metrics: Metrics) -> Any:
        return ["empty dummy result"]

    async def acquire_metrics(self) -> Metrics:
        """
        Start collecting metrics from the device
        """
        return Metrics()


class TestDevice(DummyDevice):
    """
    Test device for testing purposes
    """

    def __init__(self):
        super().__init__("Test", "test", ".")

    @classmethod
    def create_device_from_name_and_ports(
        cls, model_name: str, communication_port: str, energy_port: str
    ) -> Device:
        return TestDevice()
