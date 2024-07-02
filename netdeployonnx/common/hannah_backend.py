import abc
import asyncio
import copy
import io
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import torch

from netdeployonnx.common.wrapper import NetClient, get_netclient_from_connect

try:
    from hannah.backends.base import (
        ClassifierModule,
        InferenceBackendBase,
        ProfilingResult,
    )
except ImportError:

    class ClassifierModule: ...

    class ProfilingResult: ...

    class AbstractBackend(abc.ABC): ...

    class InferenceBackendBase(AbstractBackend): ...


try:
    from netdeployonnx.common.wrapper import NetClient, get_netclient_from_connect
except ImportError:
    NetClient = None
    get_netclient_from_connect = None
try:
    import asyncio

    import grpc
    import onnx
except ImportError:
    grpc = None
    onnx = None


# InferenceBackendBase: https://es-git.cs.uni-tuebingen.de/es/ai/hannah/hannah/-/blob/main/hannah/backends/base.py?ref_type=heads
class GRPCBackend(InferenceBackendBase):
    def __init__(self, *args, client_connect: str = "localhost:28329", **kwargs):
        self.client_connect = client_connect
        self.auth: Path | str | bytes | None = kwargs.pop("auth", None)
        self.device_filter: list[dict] = kwargs.pop("device_filter", [])
        self.keepalive_timeout: float = kwargs.pop("keepalive_timeout", 4)
        self._client = None
        self.modelbytes = None
        super().__init__(*args, **kwargs)

    @property
    def client(self) -> NetClient:
        if self._client is None:
            try:
                # either it is a path
                if isinstance(self.auth, (str, Path)):
                    if os.path.exists(self.auth):
                        with open(self.auth, "rb") as f:
                            auth = f.read()
                    else:
                        raise FileNotFoundError(f"File {self.auth} not found")
                elif isinstance(self.auth, bytes):
                    auth = self.auth
                else:
                    auth = None
                self._client = get_netclient_from_connect(
                    self.client_connect,
                    auth,
                    keepalive_timeout=self.keepalive_timeout,
                )
            except Exception:
                raise  # ConnectionError(f"Could not connect to client: {ex}")
        return self._client

    def __del__(self):
        if self._client is not None:
            self._client.close()

    def prepare(self, module: ClassifierModule):
        """
        Prepare the model for execution on the target device

        Args:
          module: the classifier module to be exported

        """
        self.module = module
        quantized_model = copy.deepcopy(module.model)
        quantized_model.cpu()
        quantized_model.train(False)

        dummy_input = module.example_input_array.cpu()
        bytesio = io.BytesIO()

        torch.onnx.export(
            quantized_model,
            dummy_input,
            bytesio,
            verbose=False,
            opset_version=11,
        )
        self.modelbytes = bytesio.getvalue()

    def run(self, *inputs) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """
        Run a batch on the target device

        Args:
          inputs: a list of torch tensors representing the inputs to be run on the target device, each tensor represents a whole batched input, so for models taking 1 parameter, the list will contain 1 tensor of shape (batch_size, *input_shape)

        Returns: the output(s) of the model as a torch tensor or a Sequence of torch tensors for models producing multiple outputs

        """  # noqa: E501
        return self._run(*inputs, profiling=False)

    def profile(self, *inputs: torch.Tensor) -> ProfilingResult:
        """Do a profiling run on the target device

        Args:
            inputs: a list of torch tensors representing the inputs to be run on the target device, each tensor represents a whole batched input, so for models taking 1 parameter, the list will contain 1 tensor of shape (batch_size, *input_shape)

        Returns: a ProfilingResult object containing the outputs of the model, the metrics obtained from the profiling run and the raw profile in a backend-specific format
        """  # noqa: E501
        return self._run(*inputs, profiling=True)

    async def _run_async(
        self, *inputs: torch.Tensor, profiling: bool = False
    ) -> Union[torch.Tensor, Sequence[torch.Tensor], ProfilingResult]:
        with get_netclient_from_connect(
            self.client_connect,
            self.auth,
            keepalive_timeout=self.keepalive_timeout,
        ) as client:
            async with client.connect(filters=self.device_filter) as device:
                result_dict = await device.deploy(
                    modelbytes=self.modelbytes, profiling=profiling
                )
                if profiling:
                    return ProfilingResult(
                        output=result_dict["result"],
                        metrics=result_dict["metrics"],
                        profile=result_dict["deployment_execution_times"],
                    )
                else:
                    return result_dict["result"]
        raise ConnectionError("Could not connect to client")

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        return loop

    def _run(
        self, *inputs: torch.Tensor, profiling: bool = False
    ) -> Union[torch.Tensor, Sequence[torch.Tensor], ProfilingResult]:
        try:
            loop = self._get_loop()
            async_result = loop.run_until_complete(
                self._run_async(*inputs, profiling=profiling)
            )
            return async_result
        except TimeoutError:
            raise
        except ValueError:
            raise
        except ConnectionError:
            raise
        except Exception as ex:
            raise ex  # reraise

    @classmethod
    def available(cls) -> bool:
        """
        Check if the backend is available

        Returns: True if the backend is available, False otherwise

        """
        try:
            # TODO: check server availability?
            return grpc is not None and onnx is not None and NetClient is not None
        except Exception:
            pass
        return False


class GRPCBackend_(InferenceBackendBase):  # noqa: N801
    """Inference Backend for grpc-based systems"""

    def __init__(
        self,
        repeat=10,
        warmup=2,
        url="localhost:8001",
    ):
        self.repeat = repeat
        self.warmup = warmup
        self.client = grpc.insecure_channel(url)
        self.model = None  # TODO:remove

    def prepare(self, model):
        memory_stream = io.BytesIO()
        dummy_input = model.example_input_array
        self.model = model  # TODO:remove
        torch.onnx.export(model, dummy_input, memory_stream, verbose=False)

        self.modelbytes = memory_stream

    def run(self, *inputs):
        return self._run(*inputs)

    def profile(self, *inputs):
        result = self._run(*inputs, profile=True)
        return result

    def _run(self, *inputs, profile=False):
        logging.info("running grpc backend on batch")
        # run on grpc
        output = self.model(*inputs)

        result = [output]
        duration = 0
        if profile:
            return ProfilingResult(
                outputs=result, metrics={"duration": duration}, profile=None
            )
        return result
