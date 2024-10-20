import time
from pathlib import Path

import grpc

from netdeployonnx.client.experiment import do_experiments
from netdeployonnx.common import device_pb2, device_pb2_grpc
from netdeployonnx.config import AppConfig


def connect(
    config: AppConfig, networkfile: Path, run_experiments=False, no_flash: bool = False
):
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
        # sample deployment
        sample_connect(config, networkfile, no_flash=no_flash)


def sample_connect(config: AppConfig, networkfile: Path, no_flash: bool = False):
    with grpc.insecure_channel(f"{config.client.host}:{config.client.port}") as channel:
        stub = device_pb2_grpc.DeviceServiceStub(channel)

        # load the given net
        if networkfile.exists():
            with open(networkfile, "rb") as fx:
                # with open(data_folder / "ai8x_net_0.onnx", "rb") as fx:
                data = fx.read()
                fx.seek(0)
                # test against onnx?
                try:
                    import io

                    import onnx

                    model = onnx.load(fx)
                    model.metadata_props.append(
                        onnx.StringStringEntryProto(
                            key="__reflash",
                            value=str(
                                not no_flash
                            ),  # if no_flash is true, reflash is false
                        )
                    )
                    buffer = io.BytesIO()
                    onnx.save(model, buffer)
                    data = buffer.getvalue()  # overwrite
                except ImportError:
                    if no_flash:
                        raise Exception("cannot set __reflash without onnx")
                    pass
        else:
            raise Exception(f"networkfile '{networkfile}' not found")

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
            time.sleep(0.5)  # has to be about 0.5
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
