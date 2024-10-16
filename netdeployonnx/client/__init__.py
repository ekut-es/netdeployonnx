import datetime
import io
import time
from pathlib import Path

import grpc
import onnx
import yaml

from netdeployonnx.client.grpc_backend import (
    ClassifierModule,
    GRPCBackend,
    LightningModel,
    ProfilingResult,
)
from netdeployonnx.common import device_pb2, device_pb2_grpc
from netdeployonnx.config import AppConfig


def run_experiment(*args, **kwargs):
    ret = {
        "args": list(args),
        "kwargs": {k: v for k, v in kwargs.items() if k not in ["onnx_model"]},
    }
    try:
        b = GRPCBackend(
            client_connect=kwargs.get("client_connect"),
            device_filter=list(kwargs.get("device_filter")),
            keepalive_timeout=kwargs.get("keepalive_timeout"),
            function_timeout=kwargs.get("function_timeout"),
        )

        data_folder = kwargs.get(
            "data_folder", Path(__file__).parent.parent.parent / "test" / "data"
        )
        model_file: str = kwargs.get("model_file", "cifar10_short.onnx")
        if "onnx_model" not in kwargs:
            if "onnx_modelbytes" in kwargs:
                with open(data_folder / model_file, "rb") as fx:
                    modelbytes = fx.read()
            else:
                modelbytes = kwargs.get("modelbytes")
            onnx_model = onnx.load(io.BytesIO(modelbytes))
        else:
            onnx_model = kwargs.get("onnx_model")
        b.prepare(module=ClassifierModule(model=LightningModel(onnx_model=onnx_model)))
        results: ProfilingResult = b.profile(kwargs.get("model_input", []))
        ret.update(
            {
                "outputs": results.outputs,
                "metrics": results.metrics,
                "profile": results.profile,
            }
        )
    except Exception as ex:
        ret.update(
            {
                "exception": str(ex),
            }
        )
    return ret


def experiment_sram_clockspeed(*args, **kwargs):
    results = []
    data_folder = Path(__file__).parent.parent.parent / "test" / "data"
    with open(data_folder / "cifar10_short.onnx", "rb") as fx:
        onnx_model = onnx.load(fx)

    for read_margin_enable in range(3, -1, -1):
        # repeat x times
        configs = [{"read_margin_enable": read_margin_enable}] * 10
        for config in configs:
            # execute each config, for that prepare metadata
            del onnx_model.metadata_props[:]
            for key, value in config.items():
                onnx_model.metadata_props.append(
                    onnx.StringStringEntryProto(key=key, value=str(value))
                )
            # overwrite model
            kwargs["onnx_model"] = onnx_model
            # copy on call
            results.append(run_experiment(*list(args), **dict(kwargs)))

    return results


def experiment_network_size(*args, **kwargs): ...


def experiment_baseclock_clockspeed(*args, **kwargs): ...
def experiment_avgpool_vs_maxpool(*args, **kwargs): ...
def experiment_input_loading(*args, **kwargs): ...


def do_experiments(*args, **kwargs):
    experiments = {
        "sram_clockspeed": experiment_sram_clockspeed,
        "network_size": experiment_network_size,
        "baseclock_clockspeed": experiment_baseclock_clockspeed,
        "avgpool_vs_maxpool": experiment_avgpool_vs_maxpool,
        "input_loading": experiment_input_loading,
    }
    data_collector = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": [],
    }

    for experiment_name, experiment in experiments.items():
        results = experiment(*args, **kwargs)
        data_collector["experiments"].append(
            {
                "name": experiment_name,
                "results": results,
            }
        )

    # save to yaml
    with open("results.yaml", "w") as fx:
        yaml.dump(data_collector, fx)


def connect(config: AppConfig):
    do_experiments(
        client_connect=f"{config.client.host}:{config.client.port}",
        device_filter=[
            # {"model": "VirtualDevice"},
            {"model": "EVKit_V1"},
        ],
        keepalive_timeout=15,
        function_timeout=30,
    )


def sample_connect(config: AppConfig):
    with grpc.insecure_channel(f"{config.client.host}:{config.client.port}") as channel:
        stub = device_pb2_grpc.DeviceServiceStub(channel)

        # Get device handle
        response = stub.GetDeviceHandle(device_pb2.GetDeviceHandleRequest())
        device_handle = response.deviceHandle
        print(f"Device handle: {device_handle.handle}")

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
