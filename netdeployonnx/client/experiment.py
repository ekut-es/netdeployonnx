import datetime
import io
from pathlib import Path

import onnx
import yaml
from tqdm import tqdm

from netdeployonnx.client.grpc_backend import (
    ClassifierModule,
    GRPCBackend,
    LightningModel,
    ProfilingResult,
)

# use pyproject group: experiments_analysis


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

    configs = []
    for read_margin in range(3, -1, -1):
        # repeat x times
        configs.extend(
            [
                {
                    "read_margin": read_margin,
                    "read_margin_enable": 1,
                    "write_neg_voltage_enable": 1,
                }
            ]
            # scale by sample_points
            * kwargs.get("sample_points", 25)
        )
    for config in tqdm(configs, desc="sram clockspeed"):
        # execute each config, for that prepare metadata
        del onnx_model.metadata_props[:]
        for key, value in config.items():
            onnx_model.metadata_props.append(
                onnx.StringStringEntryProto(key=key, value=str(value))
            )
        # overwrite model
        kwargs["onnx_model"] = onnx_model
        kwargs["config"] = config
        # copy on call
        results.append(run_experiment(*list(args), **dict(kwargs)))

    return results


def experiment_network_size(*args, **kwargs):
    results = []
    data_folder = Path(__file__).parent.parent.parent / "test" / "data"
    networks = ["cifar10_short.onnx", "cifar10.onnx"]
    configs = []
    for network in networks:
        # repeat x times
        configs.extend(
            [
                {
                    "network_name": network,
                }
            ]
            # scale by sample_points
            * kwargs.get("sample_points", 25)
        )
    for config in tqdm(configs, desc="network size"):
        network = config.get("network_name")
        with open(data_folder / network, "rb") as fx:
            onnx_model = onnx.load(fx)
        del onnx_model.metadata_props[:]

        # overwrite model
        kwargs["onnx_model"] = onnx_model
        kwargs["config"] = config
        # copy on call
        results.append(run_experiment(*list(args), **dict(kwargs)))

    return results


def experiment_cnn_clockdividers(*args, **kwargs):
    # pooling enabled vs not enabled
    results = []
    data_folder = Path(__file__).parent.parent.parent / "test" / "data"
    with open(data_folder / "cifar10_short.onnx", "rb") as fx:
        onnx_model = onnx.load(fx)

    configs = []
    for cnnclksel in range(0, 1 + 1):
        # CNN Peripheral Clock Select = cnnclksel
        # 0 is PCLK
        # 1 is ISO
        for cnnclkdiv in range(0, 4 + 1):
            # 0 is cnn_clock/2
            # 1 is cnn_clock/4
            # 2 is cnn_clock/8
            # 3 is cnn_clock/16
            # 4 is cnn_clock/1

            # repeat x times
            configs.extend(
                [
                    {
                        "GCR_pclkdiv.cnnclksel": cnnclksel,
                        "GCR.pclkdiv.cnnclkdiv": cnnclkdiv,
                    }
                ]
                # scale by sample_points
                * kwargs.get("sample_points", 25)
            )

    for config in tqdm(configs, desc="clockselectors"):
        # execute each config, for that prepare metadata
        del onnx_model.metadata_props[:]
        for key, value in config.items():
            onnx_model.metadata_props.append(
                onnx.StringStringEntryProto(key=key, value=str(value))
            )
        # overwrite model
        kwargs["onnx_model"] = onnx_model
        kwargs["config"] = config
        # copy on call
        results.append(run_experiment(*list(args), **dict(kwargs)))

    return results


def experiment_pooling(*args, **kwargs):
    # pooling enabled vs not enabled
    results = []
    data_folder = Path(__file__).parent.parent.parent / "test" / "data"
    with open(data_folder / "cifar10_short.onnx", "rb") as fx:
        onnx_model = onnx.load(fx)

    configs = []
    for pooling_enable in range(0, 1 + 1):
        for maxpooling_enable in range(0, 1 + 1):
            # repeat x times
            configs.extend(
                [
                    {
                        "pool_en": pooling_enable,
                        "maxpool_en": maxpooling_enable,  # calcmax
                    }
                ]
                # scale by sample_points
                * kwargs.get("sample_points", 25)
            )

    for config in tqdm(configs, desc="pooling"):
        # execute each config, for that prepare metadata
        del onnx_model.metadata_props[:]
        for key, value in config.items():
            onnx_model.metadata_props.append(
                onnx.StringStringEntryProto(key=key, value=str(value))
            )
        # overwrite model
        kwargs["onnx_model"] = onnx_model
        kwargs["config"] = config
        # copy on call
        results.append(run_experiment(*list(args), **dict(kwargs)))

    return results


def do_experiments(*args, **kwargs):
    experiments = {
        "sram_clockspeed": experiment_sram_clockspeed,
        "network_size": experiment_network_size,
        "experiment_cnn_clockdividers": experiment_cnn_clockdividers,
        "experiment_pooling": experiment_pooling,
    }
    data_collector = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": [],
    }

    kwargs["samplepoints"] = 25

    for experiment_name, experiment in experiments.items():
        results = experiment(*args, **kwargs)
        results = results if results else []
        data_collector["experiments"].append(
            {
                "name": experiment_name,
                "results": results,
            }
        )

    # save to yaml
    with open("results.yaml", "w") as fx:
        yaml.dump(data_collector, fx)
