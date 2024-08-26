from pathlib import Path

import grpc

from netdeployonnx.common import device_pb2, device_pb2_grpc
from netdeployonnx.config import AppConfig


def connect(config: AppConfig):
    with grpc.insecure_channel(f"{config.client.host}:{config.client.port}") as channel:
        stub = device_pb2_grpc.DeviceServiceStub(channel)

        # Get device handle
        response = stub.GetDeviceHandle(device_pb2.GetDeviceHandleRequest())
        deviceHandle = response.deviceHandle
        print(f"Device handle: {deviceHandle.handle}")

        # load the cifar10 net
        data_folder = Path(__file__).parent.parent.parent / "test" / "data"
        with open(data_folder / "cifar10.onnx", "rb") as fx:
            data = fx.read()


        # Run payload
        payload = device_pb2.RunPayloadRequest(
            deviceHandle=deviceHandle,
            payload=device_pb2.Payload(
                data=data,
                datatype=device_pb2.Payload_Datatype.onnxb,
            )
        )
        async_response:RunPayloadResponse = stub.RunPayloadAsynchronous(payload)
        print("RUN-ID:", async_response.run_id)
        while True:
            response = stub.CheckPayloadAsynchronous(device_pb2.CheckPayloadRequest(
                run_id=async_response.run_id,
            ))
            if response.payload != device_pb2.Payload():
                break
            print("checking soon...")
            import time; time.sleep(1)
        print(f"Payload result: {response.payload}")

        # Get device info
        response = stub.GetDeviceInfo(device_pb2.GetDeviceInfoRequest(deviceHandle=deviceHandle))
        print(f"Device model: {response.device.model}")
        print(f"Device manufacturer: {response.device.manufacturer}")
        print(f"Device firmware version: {response.device.firmware_version}")
        freed = stub.FreeDeviceHandle(device_pb2.FreeDeviceHandleRequest(deviceHandle=deviceHandle))
        if freed.ok:
            print("device freed.")


if __name__ == "__main__":
    connect(AppConfig())
