import time
from pathlib import Path

import grpc

from netdeployonnx.client.experiment import do_experiments
from netdeployonnx.common import device_pb2, device_pb2_grpc
from netdeployonnx.config import AppConfig


def connect(config: AppConfig, run_experiments=False):
    """
    Either does the experiments or deploys once a cifar10_short.onnx as a sample
    """
    if run_experiments:
        do_experiments(
            client_connect=f"{config.client.host}:{config.client.port}",
            device_filter=[
                # {"model": "VirtualDevice"},
                {"model": "EVKit_V1"},
            ],
            keepalive_timeout=15,
            function_timeout=30,
        )
    else:
        print("or here")
        # sample deployment
        sample_connect(config)


def sample_connect(config: AppConfig):
    with grpc.insecure_channel(f"{config.client.host}:{config.client.port}") as channel:
        stub = device_pb2_grpc.DeviceServiceStub(channel)

        # Get device handle
        response = stub.GetDeviceHandle(
            device_pb2.GetDeviceHandleRequest(
                filters=[device_pb2.DeviceInfo(**{"model": "EVKit_V1"})]
            )
        )
        device_handle = response.deviceHandle
        print(f"Device handle: {device_handle.handle}")
        assert device_handle.handle.startswith(
            "devhandle"
        ), "device id does not start with devhandle"

        # load the cifar10 net
        data_folder = Path(__file__).parent.parent.parent / "test" / "data"
        with open(data_folder / "cifar10_short.onnx", "rb") as fx:
            # with open(data_folder / "ai8x_net_0.onnx", "rb") as fx:
            data = fx.read()

        # Run payload
        payload = device_pb2.RunPayloadRequest(
            deviceHandle=device_handle,
            payload=device_pb2.Payload(
                data=data,
                datatype=device_pb2.Payload_Datatype.onnxb,
            ),
        )
        async_response: device_pb2.RunPayloadResponse = stub.RunPayloadAsynchronous(
            payload
        )
        print("RUN-ID:", async_response.run_id)
        assert async_response.run_id.startswith("run"), "run_id does not start with run"
        while True:
            response = stub.CheckPayloadAsynchronous(
                device_pb2.CheckPayloadRequest(
                    run_id=async_response.run_id,
                )
            )
            if response.payload != device_pb2.Payload():
                break
            print("waiting for callback...")
            time.sleep(1)
        if response.payload.datatype == device_pb2.Payload_Datatype.exception:
            # unpickle
            import pickle

            exception, traceback = pickle.loads(response.payload.data)
            print(f"Exception: {exception}")
            for frame in traceback:
                print(frame)
                # print(
                #     f"  File '{frame.filename}', line {frame.lineno}, in {frame.name}"
                # )
                # print(f"    {frame.line}")
        else:
            print(f"Result: {response.payload}")

        # Get device info
        response = stub.GetDeviceInfo(
            device_pb2.GetDeviceInfoRequest(deviceHandle=device_handle)
        )
        print(f"Device model: {response.device.model}")
        print(f"Device manufacturer: {response.device.manufacturer}")
        print(f"Device firmware version: {response.device.firmware_version}")
        freed = stub.FreeDeviceHandle(
            device_pb2.FreeDeviceHandleRequest(deviceHandle=device_handle)
        )
        if freed.ok:
            print("device freed.")


if __name__ == "__main__":
    connect(AppConfig())
