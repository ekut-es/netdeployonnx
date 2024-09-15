import asyncio
import functools
import json
import pickle
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Union

import anyio

from netdeployonnx.common.device_pb2 import (
    CheckPayloadRequest,
    CheckPayloadResponse,
    DeviceHandle,
    DeviceInfo,
    FreeDeviceHandleRequest,
    GetDeviceHandleRequest,
    ListDevicesRequest,
    ListDevicesResponse,
    Payload,
    Payload_Datatype,
    RunPayloadRequest,
)
from netdeployonnx.common.device_pb2_grpc import DeviceServiceStub
from netdeployonnx.server import (
    DeviceService,
)

try:
    import grpc
except ImportError:
    grpc = None


class NetClient:
    def __init__(self, client: DeviceService, channel: "grpc.Channel"):
        self.client = client
        self.channel = channel

    @asynccontextmanager
    async def connect(self, model: str = None, **kwargs):
        if model:
            kwargs.update({"model": model})
        filters = kwargs.pop("filters", None)
        if not filters:
            filters = [DeviceInfo(**kwargs)]

        handle = self.client.GetDeviceHandle(
            GetDeviceHandleRequest(filters=filters)
        ).deviceHandle.handle

        if not handle.startswith("devhandle"):
            response: ListDevicesResponse = self.client.ListDevices(
                ListDevicesRequest()
            )
            device_list = response.devices
            raise ValueError(
                "could not get handle, maybe device not found?\n"
                f"Possible devices: [{device_list}]"
            )

        yield RemoteDevice(client=self.client, handle=handle)

        # free handle
        self.client.FreeDeviceHandle(
            FreeDeviceHandleRequest(deviceHandle=DeviceHandle(handle=handle))
        )
        await asyncio.sleep(0)  # just for good measure and to sync

    async def clear_handles(self):
        raise NotImplementedError("close not implemented")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self.channel:
            self.channel.close()


class RemoteDevice:
    def __init__(self, client: DeviceService, handle: str):
        self.client = client
        self.handle = handle

    async def deploy(  # noqa: C901
        self,
        onnx_file: Union[Path, str] = "",
        modelbytes: bytes = None,
        timeout: float = 4,
        profiling: bool = True,
    ) -> Union[dict, bytes, None]:
        if modelbytes is None:
            # load the net
            async with await anyio.open_file(onnx_file, "rb") as f:
                modelbytes = await f.read()

        assert len(modelbytes) < 4e6, "model too large" # TODO: transfer model in chunks

        response = self.client.RunPayloadAsynchronous(
            RunPayloadRequest(
                deviceHandle=DeviceHandle(handle=self.handle),
                payload=Payload(data=modelbytes, datatype=Payload_Datatype.onnxb),
            )
        )

        if not response.run_id.startswith("run"):
            raise ValueError("could not run")

        async def wait_for_result(run_id: str, interval: float = 0.01):
            # we stop with either return or timeout
            while True:
                result = self.client.CheckPayloadAsynchronous(
                    CheckPayloadRequest(run_id=run_id)
                )
                if result != CheckPayloadResponse():
                    return result
                # wait for it to return
                await asyncio.sleep(interval)

        try:
            result: CheckPayloadResponse = await asyncio.wait_for(
                wait_for_result(response.run_id), timeout=timeout
            )
        except TimeoutError:
            raise TimeoutError("waiting for CheckPayloadResponse")
        if result.payload.datatype == Payload_Datatype.exception:
            exc, traceback = pickle.loads(result.payload.data)
            for frame in traceback:
                # print(
                #     f"  File '{frame.filename}', line {frame.lineno}, in {frame.name}"
                # )
                # print(f"    {frame.line}")
                print(frame)
            raise exc
        else:
            if result.payload.datatype == Payload_Datatype.none:
                return None
            elif result.payload.datatype == Payload_Datatype.json:
                return json.loads(result.payload.data)
            elif result.payload.datatype == Payload_Datatype.dict:
                return pickle.loads(result.payload.data)
            raise ValueError(f"Unknown datatype: {result.payload.datatype}")
        raise ValueError("Unknown error")


def get_netclient_from_connect(
    connect: str, auth: Union[bytes, None], keepalive_timeout: float = 4
) -> NetClient:
    """
    Get a NetClient object from a connect string and password
    Args:
        connect: the connect string
        auth: the password or root certificate
    """
    if grpc:
        if auth:
            credentials = grpc.ssl_channel_credentials(root_certificates=auth)
            create_channel_method = functools.partial(
                grpc.secure_channel, connect, credentials=credentials
            )
        else:  # insecure
            warnings.warn("No auth for grpc provided")
            create_channel_method = functools.partial(
                grpc.insecure_channel,
                connect,
            )
        try:
            channel = create_channel_method(
                options=[
                    (
                        "grpc.keepalive_timeout_ms",
                        int(keepalive_timeout * 1000),  # timeout from s to ms
                    )
                ],
            )
            stub = DeviceServiceStub(channel)
            return NetClient(stub, channel)
        except:
            import traceback

            traceback.print_exc()
            raise
    else:
        raise ImportError("grpc not imported, please install grpcio")