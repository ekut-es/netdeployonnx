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
import datetime
import io
import logging
import traceback
from pathlib import Path

import onnx
import yaml
from tqdm import tqdm

from netdeployonnx.client.grpc_backend import (
    ClassifierModule,
    GRPCBackend,
    ProfilingResult,
)


class LightningModel:
    def __init__(self, onnx_model):
        self.onnx_model = onnx_model

    def cpu(self):
        return self

    def train(self, *args):
        return self


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
        logging.info(f"ran experiment with results: {ret}")
    except Exception as ex:
        logging.error("error during experiment")
        ret.update(
            {
                "exception": {
                    "msg": str(ex),
                    "traceback": traceback.format_exc(),
                }
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
                    "network_name": "cifar10_short.onnx",
                    "__reflash": False,
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
    networks = [
        "cifar10_short.onnx",
        "cifar10.onnx",
        "ai85-bayer2rgb-qat8-q.pth.onnx",
        "ai85-cifar10-qat8-q.pth.onnx",
        "ai85-cifar100-qat8-q.pth.onnx",
        # "ai85-faceid_112-qat-q.pth.onnx", # unfortunately, this does not really work
        "ai85-kws20_v3-qat8-q.pth.onnx",
    ]
    configs = []
    for network in networks:
        # repeat x times
        # for i in range(kwargs.get("sample_points", 10)):
        for i in range(10):  # 10 times is enough
            configs.extend(
                [
                    {
                        "network_name": network,
                        "__reflash": i == 0,  # first time = reflash
                    }
                ]
            )
    pbar = tqdm(configs, desc="network size")
    for config in pbar:
        network = config.get("network_name")
        pbar.set_description(desc=f"network size [{network}]")
        with open(data_folder / network, "rb") as fx:
            onnx_model = onnx.load(fx)
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
                        "network_name": "cifar10_short.onnx",
                        "__reflash": False,
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
        break

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
                        "network_name": "cifar10_short.onnx",
                        "__reflash": False,
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


def force_flash_cifar10_short(*args, **kwargs):
    results = []
    data_folder = Path(__file__).parent.parent.parent / "test" / "data"
    networks = ["cifar10_short.onnx"]
    # TODO: kws20.v3, cifar100, bayer2rgb, faceid
    configs = []
    for network in networks:
        # repeat x times
        configs.extend(
            [
                {
                    "network_name": network,
                    "__reflash": True,
                }
            ]
        )
    for config in tqdm(configs, desc="force flash cifar10short"):
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


def experiment_measure_per_layer(*args, **kwargs):
    # per layer (just reduce max layer?!)
    results = []
    data_folder = Path(__file__).parent.parent.parent / "test" / "data"
    with open(data_folder / "cifar10_short.onnx", "rb") as fx:
        onnx_model = onnx.load(fx)

    configs = []
    for layers in range(0, 5 + 1):
        # repeat x times
        configs.extend(
            [
                {
                    "layer_count": layers,
                    "network_name": "cifar10_short.onnx",
                    "__reflash": False,
                }
            ]
            # scale by sample_points
            * kwargs.get("sample_points", 10)
        )

    for config in tqdm(configs, desc="layercnt"):
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


def write_results(data_collector):
    # save to yaml
    with open("results.yaml", "w") as fx:
        yaml.dump(data_collector, fx)


def do_experiments(*args, **kwargs):
    experiments = {
        "force_flash_cifar10_short": force_flash_cifar10_short,
        "sram_clockspeed": experiment_sram_clockspeed,
        "experiment_cnn_clockdividers": experiment_cnn_clockdividers,
        # "experiment_pooling": experiment_pooling,
        "experiment_measure_per_layer": experiment_measure_per_layer,
        # "network_size": experiment_network_size,
    }
    data_collector = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": [],
    }

    kwargs["samplepoints"] = 25
    try:
        for experiment_name, experiment in experiments.items():
            results = experiment(*args, **kwargs)
            results = results if results else []
            data_collector["experiments"].append(
                {
                    "name": experiment_name,
                    "results": results,
                }
            )
    finally:
        write_results(data_collector)
