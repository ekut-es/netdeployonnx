import time
from pathlib import Path
from unittest import mock

import pytest

from netdeployonnx.common import device_pb2
from netdeployonnx.common.wrapper import NetClient
from netdeployonnx.devices import DummyDevice
from netdeployonnx.devices.max78000 import MAX78000
from netdeployonnx.devices.max78000.ai8xize import MAX78000_ai8xize
from netdeployonnx.server import DeviceService


def run_debugger():
    import debugpy

    debugpy.listen(4567)
    debugpy.wait_for_client()
    debugpy.breakpoint()


@pytest.fixture(scope="module")
def grpc_add_to_server():
    from netdeployonnx.common import device_pb2_grpc

    return device_pb2_grpc.add_DeviceServiceServicer_to_server


@pytest.fixture(scope="module")
def grpc_servicer():
    with mock.patch("netdeployonnx.server.list_devices") as mock_list_devices:
        mock_list_devices.return_value = {
            "1": MAX78000("EvKit_V1", "MAXIM", "?"),
            "2": MAX78000("FTHR_RevA", "MAXIM", "?"),
            "3": DummyDevice("Test", "test", "."),
        }
        return DeviceService()


@pytest.fixture(scope="module")
def grpc_stub_cls(grpc_channel):
    from netdeployonnx.common.device_pb2_grpc import DeviceServiceStub

    return DeviceServiceStub


def test_net_deploy(grpc_stub: DeviceService):
    """
    Test that the device information can be retrieved by handle
    """
    with (
        mock.patch("netdeployonnx.server.list_devices") as mock_list_devices,
        mock.patch("netdeployonnx.devices.max78000.MAX78000.execute") as mock_execute,
    ):
        mock_execute.return_value = {"result": "ok"}
        mock_list_devices.return_value = {
            "1": MAX78000("EvKit_V1", "MAXIM", "?"),
            "2": MAX78000("FTHR_RevA", "MAXIM", "?"),
            "3": DummyDevice("Test", "test", "."),
        }

        handle = grpc_stub.GetDeviceHandle(
            device_pb2.GetDeviceHandleRequest(
                filters=[device_pb2.DeviceInfo(model="EvKit_V1")]
            )
        ).deviceHandle.handle
        print("handle=", handle)
        response = grpc_stub.GetDeviceInfo(
            device_pb2.GetDeviceInfoRequest(
                deviceHandle=device_pb2.DeviceHandle(handle=handle)
            )
        )
        assert isinstance(response, device_pb2.GetDeviceInfoResponse)
        assert isinstance(response.device, device_pb2.DeviceInfo)
        print("response.device=", response.device, "#")
        assert response.device.model == "EvKit_V1"
        assert response.device.manufacturer == "MAXIM"
        assert response.device.firmware_version == "?"

        handle = grpc_stub.GetDeviceHandle(
            device_pb2.GetDeviceHandleRequest(
                filters=[device_pb2.DeviceInfo(model="EvKit_V1")]
            )
        ).deviceHandle.handle

        # load the cifar10 net
        data_folder = Path(__file__).parent / "data"
        with open(data_folder / "cifar10.onnx", "rb") as fx:
            data = fx.read()

        response = grpc_stub.RunPayloadAsynchronous(
            device_pb2.RunPayloadRequest(
                deviceHandle=device_pb2.DeviceHandle(handle=handle),
                payload=device_pb2.Payload(
                    data=data, datatype=device_pb2.Payload_Datatype.onnxb
                ),
            )
        )

        run_id = response.run_id
        assert len(run_id) > 0
        time.sleep(0.05)  # wait for it to return
        result: device_pb2.CheckPayloadResponse = grpc_stub.CheckPayloadAsynchronous(
            device_pb2.CheckPayloadRequest(
                run_id=run_id,
            )
        )
        assert result
        assert result.payload
        assert (
            result.payload.datatype != device_pb2.Payload_Datatype.exception
        ), result.payload.data

        # free handle
        grpc_stub.FreeDeviceHandle(
            device_pb2.FreeDeviceHandleRequest(
                deviceHandle=device_pb2.DeviceHandle(handle=handle)
            )
        )


@pytest.mark.asyncio
async def test_wrapper_for_deployment(grpc_stub):
    with (
        mock.patch("netdeployonnx.server.list_devices") as mock_list_devices,
        mock.patch("netdeployonnx.devices.max78000.MAX78000.execute") as mock_execute,
    ):
        mock_execute.return_value = {"result": "ok"}
        mock_list_devices.return_value = {
            "1": MAX78000_ai8xize("EvKit_V1", "MAXIM", "?"),
            "2": MAX78000("FTHR_RevA", "MAXIM", "?"),
            "3": DummyDevice("Test", "test", "."),
        }
        client = NetClient(grpc_stub, mock.MagicMock())
        async with client.connect("EvKit_V1") as remote_device:
            # load the cifar10 net
            data_folder = Path(__file__).parent / "data"
            run_stats_async = await remote_device.deploy(data_folder / "cifar10.onnx")

            assert isinstance(run_stats_async, dict), "result is no dictionary"
            assert "exception" not in run_stats_async, run_stats_async["exception"]
            from pprint import pprint

            pprint(run_stats_async)
