# device_grpc.py
import asyncio
import concurrent.futures
import json
import logging
import pickle
import time
import uuid

import grpc

from netdeployonnx.common import device_pb2_grpc
from netdeployonnx.common.device_pb2 import (
    CheckPayloadRequest,
    CheckPayloadResponse,
    DeviceHandle,
    DeviceInfo,
    FreeDeviceHandleRequest,
    FreeDeviceHandleResponse,
    GetDeviceHandleResponse,
    GetDeviceInfoResponse,
    ListDevicesResponse,
    Payload,
    Payload_Datatype,
    RunPayloadResponse,
)
from netdeployonnx.config import AppConfig
from netdeployonnx.devices import Device, DummyDevice


def get_device_by_devinfo(devinfo: DeviceInfo) -> Device:
    def fields_match(dev: Device, devinfo: DeviceInfo) -> bool:
        return all(
            [
                dev.model == devinfo.model,
                dev.manufacturer == devinfo.manufacturer,
                dev.firmware_version == devinfo.firmware_version,
                dev.port == devinfo.port,
            ]
        )

    # try a better variant to get the device for each field
    for dev in list_devices().values():
        if fields_match(dev, devinfo):
            return dev
    raise ValueError("No device found for the provided device info")


def list_devices() -> dict[str, Device]:
    devices = [
        DummyDevice("EvKit_V1", "MAXIM", "?"),
        DummyDevice("FTHR_RevA", "MAXIM", "?"),
        DummyDevice("Test", "test", "."),
    ]
    return {dev.port: dev for dev in devices}


class DeviceService(device_pb2_grpc.DeviceServiceServicer):
    def __init__(self):
        self.device_handles: dict[(str | uuid.UUID), Device] = {}
        self.run_queue: dict[str, concurrent.futures.Future] = {}

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        try:
            loop = asyncio.get_event_loop()
            return loop
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def ListDevices(self, request, context):  # noqa: N802
        devices = list_devices()
        return ListDevicesResponse(
            devices=[
                DeviceInfo(
                    port=dev.port,
                    model=dev.model,
                    manufacturer=dev.manufacturer,
                    firmware_version=dev.firmware_version,
                )
                for devid, dev in devices.items()
            ]
        )

    def RunPayloadSynchronous(self, request, context):  # noqa: N802
        # Run the payload on the device with the provided handle
        run_payload_req = self.RunPayloadAsynchronous(request, context)
        run_id = run_payload_req.run_id
        while self.run_queue[run_id].done() is False:
            time.sleep(0.01)
            # check for errors
            if self.run_queue[run_id].exception() is not None:
                break
        ret_payload = self.CheckPayloadAsynchronous(
            CheckPayloadRequest(run_id=run_id), context
        ).payload
        return RunPayloadResponse(payload=ret_payload)

    def RunPayloadAsynchronous(self, request, context):  # noqa: N802
        # Run the payload on the device with the provided handle
        handle = request.deviceHandle.handle
        if handle in self.device_handles:
            payload = request.payload
            device: Device = self.device_handles[handle]

            # put the reqid in a queue
            run_id = "run" + str(uuid.uuid4())
            if run_id not in self.run_queue:
                self.run_queue[run_id] = self.loop.create_task(
                    device.run_async(
                        Payload_Datatype.Name(payload.datatype), payload.data
                    )
                )
                # we need to make sure that our task has the
                # possibility to run atleast once
                self.loop.run_until_complete(asyncio.sleep(0))
            return RunPayloadResponse(run_id=run_id)
        return RunPayloadResponse(payload=None)

    def CheckPayloadAsynchronous(self, request, context):  # noqa: N802
        # Check the status of the payload with the provided run_id
        run_id = request.run_id
        if run_id in self.run_queue:
            # queue update -> start a thread sync
            self.loop.run_until_complete(asyncio.sleep(0))
            if self.run_queue[run_id].done():
                try:
                    result = self.run_queue[run_id].result()
                    assert isinstance(
                        result, dict
                    ), f"return value is not dict but {type(result)}"
                    if "json":
                        payload = Payload(
                            datatype="json",
                            data=json.dumps(result).encode("utf-8"),
                        )
                    else:
                        payload = Payload(
                            datatype=Payload_Datatype.pickle,
                            data=pickle.dumps(result),
                        )
                except Exception as ex:
                    payload = Payload(
                        datatype=Payload_Datatype.exception,
                        data=pickle.dumps(ex),
                    )
                finally:
                    del self.run_queue[run_id]  # we dont need that anymore
                return CheckPayloadResponse(payload=payload)
            else:
                return CheckPayloadResponse()  # not done yet
        return CheckPayloadResponse()

    def GetDeviceInfo(self, request, context):  # noqa: N802
        """
        Get the device information for the device with the provided handle
        """
        handle = request.deviceHandle.handle
        if handle in self.device_handles:
            device = self.device_handles[handle]
            return GetDeviceInfoResponse(
                device=DeviceInfo(
                    port=device.port,
                    model=device.model,
                    manufacturer=device.manufacturer,
                    firmware_version=device.firmware_version,
                )
            )
        else:
            return GetDeviceInfoResponse(device=None)

    def GetDeviceHandle(self, request, context):  # noqa: N802
        """
        Get the device handle for the device that matches the provided filters
        """
        filters: list[DeviceInfo] = request.filters
        result: list[DeviceInfo] = [
            DeviceInfo(
                port=dev.port,
                model=dev.model,
                manufacturer=dev.manufacturer,
                firmware_version=dev.firmware_version,
            )
            for devid, dev in list_devices().items()
        ]
        # distill the result to only the devices that match the filters
        for f in filters:

            def filterfunc(dev):
                """
                Check if the device matches the filter
                """
                for field in f.DESCRIPTOR.fields:
                    filter_val = getattr(f, field.name)
                    # check if the field is set
                    if filter_val:
                        dev_val = getattr(dev, field.name)
                        # check if the field value matches the device value
                        if filter_val not in dev_val:
                            return False
                return True

            result = filter(filterfunc, result)

        result = list(result)
        if len(result) == 1:
            new_handle = f"devhandle-{uuid.uuid4()}"
            # this could fail
            self.device_handles[new_handle] = get_device_by_devinfo(result[0])
            return GetDeviceHandleResponse(
                deviceHandle=DeviceHandle(handle=new_handle),
            )
        else:
            return GetDeviceHandleResponse(deviceHandle=None)

    def FreeDeviceHandle(  # noqa: N802
        self, request: FreeDeviceHandleRequest, context
    ) -> FreeDeviceHandleResponse:
        worked: bool = False
        handle: str = request.deviceHandle.handle
        if handle in self.device_handles:
            try:
                del self.device_handles[handle]
                worked = True
            except Exception as ex:
                worked = False
                logging.exception("Failed to free device handle", ex)
        return FreeDeviceHandleResponse(ok=worked)


def listen(config: AppConfig):
    print(f"Listening on {config.server.host}:{config.server.port}")
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    device_pb2_grpc.add_DeviceServiceServicer_to_server(DeviceService(), server)
    server.add_insecure_port(f"{config.server.host}:{config.server.port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    listen(AppConfig())
