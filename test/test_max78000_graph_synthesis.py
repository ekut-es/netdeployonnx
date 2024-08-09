# test_max78000_graph_synthesis.py
import asyncio
import logging
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
import pytest

from netdeployonnx.devices.max78000.core import CNNx16_Quadrant, CNNx16Core
from netdeployonnx.devices.max78000.graph_synthesizer import (
    grid_optimizer_maximize_utilization,  # noqa: F401
    synth_to_core_ir,
    synthesizer_draft,  # noqa: F401
)
from netdeployonnx.devices.max78000.synthesizer.grid import print_table

from .data.cifar10_weights import kernels as cifar10_kernels
from .test_max78000_synthesize_cifar10 import MAX78000, cifar10_layout, synthesize_to_c

logging.basicConfig(
    level=logging.INFO,
    format="[+{relativeCreated:2.2f}ms] {levelname}: ({funcName:10s}) {message}",
    style="{",
)


async def synth_to_file(layout):
    device = MAX78000()
    test_name = "test.test_max78000_synthesize_cifar10"
    with mock.patch(f"{test_name}.check_bias_from_instructions") as mock_check_bias:
        mock_check_bias.return_value = True
        with mock.patch(
            f"{test_name}.check_kernels_from_instructions"
        ) as mock_check_kernel:
            mock_check_kernel.return_value = True
            await synthesize_to_c(
                await device.compile_instructions(layout), "my_cifar10_cnn.c"
            )


@pytest.mark.skip(reason="dont synth to file all the time")
def test_max78000_graph_synth_to_file():
    # Test the transformer
    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10_transformed.onnx")

    # Transform the graph
    layout: CNNx16Core = synth_to_core_ir(model.graph)
    assert layout is not None

    # now to file?
    asyncio.run(synth_to_file(layout))


synth_compare_parameters = [
    ("kernels", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
    ("enable_mask", [0, 2, 4]),
    ("enable_processor", [0, 2, 4]),
    ("writeptr", [3, 10]),  # layer 3 and 10 dont follow a rule?
    ("maxpool_en", []),
    ("pool_en", []),
    ("relu_en", []),
    ("flatten_en", []),
    ("master", []),
    ("stride", []),
    ("shift_by", [1, 2, 3, 5, 6, 7, 8, 9, 10]),  # TODO: fix
    ("sram_load_source", []),
    # TODO: just not now
    ("big_data_write", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15]),
    ("expansion_mode_inputchan", [10]),
    ("expansion_mode_processors", [1, 3, 10]),
    ("expansion_mode_writeptr", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),  # TODO:
    ("nonscaled_nonquantized_sum_feed_en", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ("oned_mask_width", []),
    ("bias_addr", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),  # TODO:
    ("bias_en", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),  # TODO:
    ("writeptr_mask_offset", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ("writeptr_multipass_offset", [5, 6, 8, 9]),  # TODO:
    ("writeptr_timeslot_offset", [1, 4, 6, 9, 10]),  # TODO:
    ("mask_start", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ("mask_maxaddr", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ("tram_maxaddr", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),  # TODO:
    ("readptr", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),  # TODO:
    # (None, []), # TODO: this one is broken
]


@pytest.mark.parametrize(
    "explicit_var_test,ignore_layers",
    synth_compare_parameters,
)
def test_max78000_graph_synth_compare(explicit_var_test, ignore_layers):
    global global_max_correct_per_layer, global_correct_per_layer
    global_max_correct_per_layer = 0
    global_correct_per_layer = 0

    def per_layer(layer, ground_truth_layer):
        global global_max_correct_per_layer, global_correct_per_layer
        correct_per_layer = 0
        max_correct_per_layer = 0
        orig_value = None
        value = None
        if layer.idx in ignore_layers:
            return "[skip]"
        for fieldname, field in ground_truth_layer.model_fields.items():
            if explicit_var_test and explicit_var_test != fieldname:
                continue  # skip if explicit variable test is selected
            value = getattr(layer, fieldname)
            orig_value = getattr(ground_truth_layer, fieldname)
            default_value = None
            # skip if the type of the orig value is not in list
            if type(orig_value) in [CNNx16_Quadrant]:
                continue
            # only if the value was changed
            if orig_value != default_value:
                # compare
                if value == orig_value:
                    correct_per_layer += 1
                # else:
                #     print(f"Error: Q{ground_truth_layer.quadrant.idx}" \
                #            "Ly{ground_truth_layer.idx}: {fieldname} different")
                max_correct_per_layer += 1
        global_max_correct_per_layer += max_correct_per_layer
        global_correct_per_layer += correct_per_layer
        if explicit_var_test:
            ov = str(orig_value)[:10]
            v = str(value)[:10]
            return f"[orig:{ov} vs {v}]" if orig_value != value else "[ ok ]"
        else:
            return (
                f"[incor:{max_correct_per_layer-correct_per_layer}"
                f"/{max_correct_per_layer}]"
            )
        # return f"[correct:{correct_per_layer}/{max_correct_per_layer}]"

    # Test the transformer
    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10_transformed.onnx")

    # Transform the graph
    layout = synth_to_core_ir(model.graph)
    assert layout is not None
    ground_truth_layout = asyncio.run(cifar10_layout())
    assert ground_truth_layout is not None
    print_table(
        data_accessor=(
            lambda r, c: per_layer(layout[c, r - 1], ground_truth_layout[c, r - 1])
        ),
        COLUMN_WIDTH=-1,  # noqa: N803
        COLUMNS=4,  # noqa: N803
        ROWS=16,  # noqa: N803
    )
    assert global_correct_per_layer == global_max_correct_per_layer, "atleast one error"
    print(
        f"[incor:{global_max_correct_per_layer-global_correct_per_layer}"
        f"/{global_max_correct_per_layer}]"
    )


def test_max78000_graph_synth_compare_list_missing():
    global missing_list
    missing_list = set()

    def per_layer(layer, ground_truth_layer):
        global missing_list
        for fieldname, field in ground_truth_layer.model_fields.items():
            value = getattr(layer, fieldname)
            orig_value = getattr(ground_truth_layer, fieldname)
            default_value = None
            # skip if the type of the orig value is not in list
            if type(orig_value) in [CNNx16_Quadrant]:
                continue
            # only if the value was changed
            if orig_value != default_value:
                # compare
                if value != orig_value:
                    missing_list |= {fieldname}

    # Test the transformer
    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10_transformed.onnx")

    # Transform the graph
    layout = synth_to_core_ir(model.graph)
    assert layout is not None
    ground_truth_layout = asyncio.run(cifar10_layout())
    assert ground_truth_layout is not None
    for quad in range(4):
        for layer in range(16):
            per_layer(layout[quad, layer], ground_truth_layout[quad, layer])

    # get the params from test_max78000_graph_synth_compare
    ignores = set([name for name, _ in synth_compare_parameters if name is not None])
    print(missing_list - ignores)


@pytest.fixture(scope="module")
def kernel_data_per_layer():
    return make_kernel_data_per_layer()


def make_kernel_data_per_layer():
    import os
    import sys
    import warnings

    warnings.warn("sketchy shit with importing: TODO: remove")
    sys.path.append(Path(__file__).parent.parent.parent.parent / "ai8x-synthesis")
    print(os.path.exists(sys.path[-1]))
    from izer import checkpoint

    # Test the transformer
    data_folder = Path(__file__).parent / "data"

    layers, weights, bias, output_shift, input_channels, output_channels = (
        checkpoint.load(
            data_folder / "ai85-cifar10-qat8-q.pth.tar",
            "ai85nascifarnet",
            quantization=[8] * 11,
            bias_quantization=[8] * 11,
            output_shift=[0] * 11,
            kernel_size=[
                [3, 3],
                [1, 1],
                [3, 3],
                [3, 3],
                [1, 1],
                [3, 3],
                [1, 1],
                [3, 3],
                [3, 3],
                [1, 1],
                [1, 1],
            ],
            operator=[
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                4,
            ],
            bypass=[None] * 11,
            weight_source=[None] * 11,
            conv_groups=[1] * 11,
        )
    )
    assert len(weights) == 11
    return weights


@pytest.mark.skip(reason="dont synth to file all the time")
def test_print_cifar10_kernels():
    with open("cifar10_kernels.txt", "w") as fx:
        addr_data = {}
        for idx, kernel in enumerate(cifar10_kernels):
            addr, data = kernel["addr"], kernel["data"]
            addr_data[addr] = data
        for addr, data in addr_data.items():
            fx.write(f"kernel_0x{addr:08x} = [\n\t")
            blocksize = 16
            blocks = [data[i : i + blocksize] for i in range(0, len(data), blocksize)]
            for block in blocks:
                fx.write(
                    " ".join([f"{x:02X}[{x - 256 if x > 127 else x}]" for x in block])
                )
                fx.write("\n\t")
            fx.write("]" + "\n" * 10)


@pytest.mark.skip(reason="dont synth to file all the time")
def test_print_kernel_data(kernel_data_per_layer):
    with open("kernel_data_per_layer.txt", "w") as fx:
        import json

        for idx, layer in enumerate(kernel_data_per_layer):
            layer_shape = "_".join([str(x) for x in layer.shape])
            layerdata = (
                layer.reshape(-1) if layer.shape[-2:] == (1, 1) else layer
            ).tolist()
            fx.write(f"layer_{idx}_{layer_shape} = ")
            fx.write(json.dumps(layerdata))
            fx.write("\n" * 10)


@pytest.mark.skip(reason="dont synth to file all the time")
def test_print_kernel_map(kernel_data_per_layer):
    # i want to print a map thats like:
    # layer_0_192_3_3:
    #   0: (quadrant, mram_offset, length) # should be 3x3
    # layer_1_2048_1_1:
    #   0: (quadrant, mram_offset, length) # should be 1x1
    #   ...

    addr_data = {}
    for idx, kernel in enumerate(cifar10_kernels):
        addr, data = kernel["addr"], kernel["data"]
        addr_data[addr] = data

    result = {}
    for layeridx, layer in enumerate(kernel_data_per_layer):
        layer_map = {}
        for mask_idx in range(layer.shape[0]):
            this_data: bytes = layer[mask_idx].view(np.uint8).reshape(-1).tobytes()
            # search this_data in addr_data
            found_locations = []
            for addr, data in addr_data.items():
                if this_data in data:
                    found_locations.append(addr)
            quadrant = found_locations
            mram_offset = 0
            length = 0
            layer_map[mask_idx] = str((quadrant, mram_offset, length))
        result[f"layer_{layeridx}" + "_".join([str(x) for x in layer.shape])] = (
            layer_map
        )

    # with open("kernel_map.yaml", "w") as fx:
    #     fx.write(yaml.dump(result, Dumper=yaml.Dumper))


@pytest.mark.skip(reason="fix this test")
@pytest.mark.parametrize(
    "inc,outc,expected",
    [
        (3, 3, [0, 3, 6, 1, 4, 7, 2, 5, 8]),
        (3, 2, [0, 3, 1, 4, 2, 5]),
        (3, 5, [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]),
    ],
)
def test_memoryquadrant_kernel_reorder(inc, outc, expected):
    class MemoryQuadrant: ...  # TODO: fix this test

    m = MemoryQuadrant(processors=0xFFFF, per_layer_quantization=[8] * 16, tensor=None)
    assert list(m.kernel_reorder_generator(inc, outc)) == expected
    ...


# def izer():
#     for processor in range(64):
#         col_minmax = 0, 768 # comes from 768x72bit => 6kb
#         # find out which processor owns which part of the full memory
#         mem_map = {
#             processor: (mincol, maxcol)
#             for p in enabled_procs
#         }
#         # now itereate over the procs again and find out which memory is used
#         for p in range:
#             ...
