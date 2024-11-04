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
import json
import time
import uuid
from unittest import mock

import pytest

from netdeployonnx.common import device_pb2, device_pb2_grpc
from netdeployonnx.common.device_pb2_grpc import DeviceServiceStub
from netdeployonnx.devices import DummyDevice
from netdeployonnx.devices.max78000 import MAX78000
from netdeployonnx.server import DeviceService


def run_debugger():
    import debugpy

    debugpy.listen(4567)
    debugpy.wait_for_client()
    debugpy.breakpoint()


@pytest.fixture(scope="module")
def grpc_add_to_server():
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
    return DeviceServiceStub


@pytest.mark.parametrize(
    "list_devices, filter_model, filter_manuf, expect",
    [
        ([], [], [], None),  # no devices -> None
        (
            [1],
            [],
            [],
            uuid.UUID,
        ),  # one device, no filters -> exact one result -> return the one and only
        ([1, 2, 3], [], [], None),  # no filters -> too many results -> None
        ([1, 2, 3], ["EvKit_V1"], [], uuid.UUID),  # select by model
        ([1, 2, 3], ["FTHR_RevA"], [], uuid.UUID),  # select by model
        ([1, 2, 3], [], ["MAXIM"], None),  # return None because of too many filters
        ([1, 2, 3], [], ["test"], uuid.UUID),  # select by manuf
    ],
)
def test_device_handle_filter(
    grpc_stub, list_devices, filter_model, filter_manuf, expect
):
    """
    Test that a device handle can be retrieved with filters
    """
    retval = {
        "1": DummyDevice("EvKit_V1", "MAXIM", "?") if 1 in list_devices else None,
        "2": DummyDevice("FTHR_RevA", "MAXIM", "?") if 2 in list_devices else None,
        "3": DummyDevice("Test", "test", ".") if 3 in list_devices else None,
    }
    retval = dict(
        filter(lambda x: x[1] is not None, retval.items())
    )  # remove None values
    with mock.patch("netdeployonnx.server.list_devices") as mock_devices:
        mock_devices.return_value = retval
        filters_req = [device_pb2.DeviceInfo(model=f) for f in filter_model]
        filters_req += [device_pb2.DeviceInfo(manufacturer=f) for f in filter_manuf]
        response = grpc_stub.GetDeviceHandle(
            device_pb2.GetDeviceHandleRequest(filters=filters_req)
        )
        assert type(response) is device_pb2.GetDeviceHandleResponse
        if expect is None:
            assert response.deviceHandle == device_pb2.DeviceHandle()
        else:
            assert type(response.deviceHandle) is device_pb2.DeviceHandle
            assert isinstance(response.deviceHandle.handle, str)
            assert response.deviceHandle.handle.startswith("devhandle-")
            assert expect(response.deviceHandle.handle.replace("devhandle-", ""))


def test_list_devices(grpc_stub):
    """
    Test that the list of devices can be retrieved
    """
    with mock.patch("netdeployonnx.server.list_devices") as mock_list_devices:
        mock_list_devices.return_value = {
            "1": DummyDevice("EvKit_V1", "MAXIM", "?"),
            "2": DummyDevice("FTHR_RevA", "MAXIM", "?"),
            "3": DummyDevice("Test", "test", "."),
        }
        response = grpc_stub.ListDevices(device_pb2.ListDevicesRequest())
        assert len(response.devices) == 3
        assert response.devices[0].model == "EvKit_V1"
        assert response.devices[1].model == "FTHR_RevA"
        assert response.devices[2].model == "Test"


def test_device_info(grpc_stub):
    """
    Test that the device information can be retrieved by handle
    """
    with mock.patch("netdeployonnx.server.list_devices") as mock_list_devices:
        mock_list_devices.return_value = {
            "1": DummyDevice("EvKit_V1", "MAXIM", "?", "1"),
            "2": DummyDevice("FTHR_RevA", "MAXIM", "?"),
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
        assert response.device.port == "1"
        assert response.device.model == "EvKit_V1"
        assert response.device.manufacturer == "MAXIM"
        assert response.device.firmware_version == "?"


def test_free_device_handle(grpc_stub):
    """
    Test that a device handle can be freed and that the device information is no longer
        available
    """
    with mock.patch("netdeployonnx.server.list_devices") as mock_list_devices:
        mock_list_devices.return_value = {
            "1": DummyDevice("EvKit_V1", "MAXIM", "?"),
        }

        handle = grpc_stub.GetDeviceHandle(
            device_pb2.GetDeviceHandleRequest(
                filters=[device_pb2.DeviceInfo(model="EvKit_V1")]
            )
        ).deviceHandle.handle
        response = grpc_stub.GetDeviceInfo(
            device_pb2.GetDeviceInfoRequest(
                deviceHandle=device_pb2.DeviceHandle(handle=handle)
            )
        )
        assert response.device is not None

        response = grpc_stub.FreeDeviceHandle(
            device_pb2.FreeDeviceHandleRequest(
                deviceHandle=device_pb2.DeviceHandle(handle=handle)
            )
        )
        assert response is not None
        assert response.ok

        response = grpc_stub.GetDeviceInfo(
            device_pb2.GetDeviceInfoRequest(
                deviceHandle=device_pb2.DeviceHandle(handle=handle)
            )
        )
        assert isinstance(response, device_pb2.GetDeviceInfoResponse)
        assert isinstance(response.device, device_pb2.DeviceInfo)
        assert response.device == device_pb2.DeviceInfo()


@pytest.mark.parametrize(
    "wait_time, expected_result",
    [
        (0.0, b"test1243"),
        (0.1, b"test1243"),
        (0.5, b"test1243"),
        (0.5, b"test1243"),
    ],
)
def test_run_payload_async(
    grpc_stub, wait_time: float, expected_result: bytes, twice: bool = True
):
    """
    Test that a payload can be run on a device asynchronously
    """
    with mock.patch("netdeployonnx.server.Device.run_async") as mock_run_async:
        run_id = ""
        expected_result = {
            "any_result": "any_result"
        }  # TODO: stop overwriting expected_result
        mock_run_async.return_value = expected_result
        expected_result = json.dumps(mock_run_async.return_value).encode("utf8")
        with mock.patch("netdeployonnx.server.list_devices") as mock_list_devices:
            mock_list_devices.return_value = {
                "1": DummyDevice("EvKit_V1", "MAXIM", "?"),
            }

            handle = grpc_stub.GetDeviceHandle(
                device_pb2.GetDeviceHandleRequest(
                    filters=[device_pb2.DeviceInfo(model="EvKit_V1")]
                )
            ).deviceHandle.handle
            response = grpc_stub.RunPayloadAsynchronous(
                device_pb2.RunPayloadRequest(
                    deviceHandle=device_pb2.DeviceHandle(handle=handle),
                    payload=device_pb2.Payload(data=b"test"),
                )
            )
            run_id = response.run_id
            assert len(run_id) > 0
            time.sleep(wait_time)  # wait for it to return
            result: device_pb2.CheckPayloadResponse = (
                grpc_stub.CheckPayloadAsynchronous(
                    device_pb2.CheckPayloadRequest(
                        run_id=run_id,
                    )
                )
            )

            if twice:
                time.sleep(wait_time)  # wait for it to return
                result_2nd: device_pb2.CheckPayloadResponse = (
                    grpc_stub.CheckPayloadAsynchronous(
                        device_pb2.CheckPayloadRequest(
                            run_id=run_id,
                        )
                    )
                )

                assert result_2nd.payload == device_pb2.Payload()  # empty payload

            assert result.payload.data == expected_result
        assert len(run_id) > 0
        mock_run_async.assert_called_once()
        mock_run_async.assert_awaited_once()
        print(mock_run_async)
        mock_run_async.assert_awaited_with("none", b"test", "none", b"")


def test_run_payload_async_wrong_returntype(
    grpc_stub, expected_result={"test": "asdf"}
):
    """
    Test that a payload can be run on a device asynchronously with wrong return type
    """
    with mock.patch("netdeployonnx.server.Device.run_async") as mock_run_async:
        mock_run_async.return_value = expected_result
        expected_result = json.dumps(expected_result).encode("utf8")
        with mock.patch("netdeployonnx.server.list_devices") as mock_list_devices:
            mock_list_devices.return_value = {
                "1": DummyDevice("EvKit_V1", "MAXIM", "?"),
            }

            handle = grpc_stub.GetDeviceHandle(
                device_pb2.GetDeviceHandleRequest(
                    filters=[device_pb2.DeviceInfo(model="EvKit_V1")]
                )
            ).deviceHandle.handle
            response = grpc_stub.RunPayloadSynchronous(
                device_pb2.RunPayloadRequest(
                    deviceHandle=device_pb2.DeviceHandle(handle=handle),
                    payload=device_pb2.Payload(data=b"test"),
                )
            )
            assert response.payload.data == expected_result
        mock_run_async.assert_called_once()
        # runid, datatype, data
        mock_run_async.assert_called_with("none", b"test", "none", b"")


def test_run_payload_async_invalid_runid(grpc_stub):
    """
    Test that a payload can be run on a device asynchronously
    """
    with mock.patch("netdeployonnx.server.Device.run_async") as mock_run_async:
        run_id = ""
        mock_run_async.return_value = b"test1243"
        with mock.patch("netdeployonnx.server.list_devices") as mock_list_devices:
            mock_list_devices.return_value = {
                "1": DummyDevice("EvKit_V1", "MAXIM", "?"),
            }

            handle = grpc_stub.GetDeviceHandle(  # noqa: F841
                device_pb2.GetDeviceHandleRequest(
                    filters=[device_pb2.DeviceInfo(model="EvKit_V1")]
                )
            ).deviceHandle.handle
            run_id = "run_asdf"
            assert len(run_id) > 0
            result: device_pb2.CheckPayloadResponse = (
                grpc_stub.CheckPayloadAsynchronous(
                    device_pb2.CheckPayloadRequest(
                        run_id=run_id,
                    )
                )
            )
            assert result.payload.data == b""
            assert result.payload.datatype == device_pb2.Payload_Datatype.none
            assert result.payload == device_pb2.Payload()  # empty payload
        mock_run_async.assert_not_called()
        mock_run_async.assert_not_awaited()
