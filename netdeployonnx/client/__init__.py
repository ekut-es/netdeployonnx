import grpc

from netdeployonnx.common import device_pb2, device_pb2_grpc
from netdeployonnx.config import AppConfig


def connect(config: AppConfig):
    with grpc.insecure_channel(f"{config.client.host}:{config.client.port}") as channel:
        stub = device_pb2_grpc.DeviceServiceStub(channel)

        # Get device handle
        response = stub.GetDeviceHandle(device_pb2.GetDeviceHandleRequest())
        print(f"Device handle: {response.deviceHandle.handle}")

        # Run payload
        payload = device_pb2.RunPayloadRequest(
            payload=device_pb2.Payload(
                data=b"Hello, world!",
                datatype=device_pb2.Payload_Datatype.json,
            )
        )
        response = stub.RunPayload(payload)
        print(f"Payload result: {response.result}")

        # Get device info
        response = stub.GetDeviceInfo(device_pb2.GetDeviceInfoRequest())
        print(f"Device model: {response.model}")
        print(f"Device manufacturer: {response.manufacturer}")
        print(f"Device firmware version: {response.firmware_version}")


if __name__ == "__main__":
    connect(AppConfig())
