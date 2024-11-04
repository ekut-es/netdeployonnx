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
import contextlib
import io
import json
import logging
import warnings
from dataclasses import dataclass
from typing import Any
from unittest import mock

from netdeployonnx.devices.max78000.ai8xize.apb_writer import get_custom_writer

warnings.filterwarnings("ignore", category=DeprecationWarning)

import fs  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import yaml  # noqa: E402
from fs.memoryfs import MemoryFS  # noqa: E402
from izer.izer import main as izer_main  # noqa: E402

_board_name = "assets.from_template.board_name"  # "EvKit_V1"
_config_file_yaml = "config_file.yaml"
_checkpoint_file_onnx = "checkpoint_file.onnx"
_sample_input_npy = "tests/sample_cifar-10.npy"
_vscode_settings_json = "assets/vscode/defaults/settings.json"
_test_dir = "sdk/Examples/MAX78000/CNN_2LY"
_api_filename = "_api_filename"
_makefile = "assets/makefile/Makefile"
_projects_makefile = "assets/makefile/projects.mk"


def get_parser():
    @dataclass
    class Options:
        allow_streaming = False
        apb_base = None
        api_filename = _api_filename
        autogen = "None"
        autogen_list = "autogen_list"
        avg_pool_rounding = False
        balance_speed = True
        bias_input = None
        board_name = _board_name
        boost = None
        c_filename = "main"
        calcx4 = False
        checkpoint_file = _checkpoint_file_onnx  # 'trained/ai85-cifar10-qat8-q.pth.tar'
        clock_divider = None
        clock_trim = None
        compact_data = True
        compact_weights = False
        config_file = _config_file_yaml
        debug = False
        debug_computation = False
        debug_latency = False
        debug_new_streaming = True
        debug_snoop = False
        debugwait = 2
        deepsleep = False
        define = ""
        define_default_arm = "-DMXC_ASSERT_ENABLE -DARM_MATH_CM4"
        define_default_riscv = "-DMXC_ASSERT_ENABLE -DRV32"
        device = 85
        display_checkpoint = False  # was True
        display_progress = True
        eclipse_includes = ""
        eclipse_openocd_args = (
            "-f interface/cmsis-dap.cfg -f target/##__TARGET_LC__##.cfg"
        )
        eclipse_variables = ""
        embedded_code = False  # was True, TODO: check
        enable_delay = None
        energy = False
        ext_rdy = False
        fast_fifo = False
        fast_fifo_quad = False
        fifo = False
        fifo_go = False
        fifo_wait = True
        fixed_input = False
        forever = False
        generate_kat = True
        ignore_activation = False
        ignore_bias_groups = False
        ignore_bn = False
        ignore_energy_warning = False
        ignore_hw_limits = False
        ignore_mlator_warning = False
        ignore_streaming = False
        increase_delta1 = 0
        increase_delta2 = 0
        increase_start = 2
        init_tram = False
        input_csv = None
        input_csv_format = 888
        input_csv_period = 80
        input_csv_retrace = 5
        input_fifo = False
        input_filename = "input"
        input_offset = None
        input_pix_clk = 9
        input_split = 1
        input_sync = False
        kernel_format = "{0:4}"
        legacy_kernels = False
        legacy_test = False
        link_layer = False
        log = True
        log_filename = "log.txt"
        log_intermediate = False
        log_pooling = False
        max_count = None
        max_proc = None
        mexpress = True
        mlator = False
        mlator_noverify = False
        new_kernel_loader = True
        no_bias = []
        no_deduplicate_weights = False
        no_error_stop = False
        no_timer = False
        no_version_check = True
        no_warn_zero = False
        one_shot = False
        output_bias_filename = "bias"
        output_config_filename = "config"
        output_data_filename = "data"
        output_filename = "output"
        output_pass_filename = None
        output_weights_filename = "weights"
        output_width = None
        override_delta1 = None
        override_delta2 = None
        override_rollover = None
        override_start = None
        overwrite = False
        overwrite_ok = False
        pipeline = None
        pll = None
        powerdown = False
        prefix = "cifar-10"
        pretend_zero_sram = False
        queue_name = None
        rd_ahead = False
        ready_sel = None
        ready_sel_aon = None
        ready_sel_fifo = None
        repeat_layers = 1
        reshape_inputs = False
        result_filename = "sampleoutput.h"
        result_numpy = None
        result_output = False
        riscv = False
        riscv_cache = False
        riscv_debug = False
        riscv_exclusive = False
        riscv_flash = False
        rtl_preload = False
        rtl_preload_weights = False
        runtest_filename = "run_test.sv"
        sample_filename = "sampledata.h"
        sample_input = _sample_input_npy
        simple1b = False
        skip_checkpoint_layers = 0
        skip_yaml_layers = 0
        slow_load = 0
        snoop_loop = False
        softmax = True
        start_layer = 0
        stop_after = None
        stop_start = False
        streaming_layers = None
        synthesize_input = None
        synthesize_words = 8
        test_bist = False
        test_dir = _test_dir
        timeout = None
        timer = 0
        top_level = "cnn"
        unload = True
        unroll_8bit = 1
        unroll_mlator = 8
        unroll_wide = 8
        upstream = "MaximIntegratedAI/ai8x-synthesis"
        verbose = False
        verbose_all = False
        verify_kernels = False
        verify_writes = False
        version_check_interval = 24
        weight_filename = "weights.h"
        weight_input = None
        weight_start = 0
        wfi = True
        write_zero_registers = False
        yamllint = None
        zero_sram = False
        zero_unused = False

    return Options()


def open_yamlcfg(cfg: dict[str, any]) -> io.StringIO:
    buffer = io.StringIO()
    yaml.dump(cfg, buffer)
    buffer.seek(0)
    return buffer


def open_checkpoint(model: onnx.ModelProto) -> io.BytesIO:
    buffer = io.BytesIO()
    buffer.write(model.SerializeToString())
    buffer.seek(0)
    return buffer


def open_sample_input(sample_input: np.ndarray) -> io.BytesIO:
    buffer = io.BytesIO()
    np.save(buffer, sample_input)
    buffer.seek(0)
    return buffer


def open_vscode_settings_json(defaults) -> io.BytesIO:
    buffer = io.BytesIO()
    buffer.write(json.dumps(defaults).encode("utf8"))  # why tf utf8
    buffer.seek(0)
    return buffer


def wrap_vfs_open(vfs: MemoryFS):
    def func(path, *args, **kwargs):
        path = str(path)
        return vfs.open(path, *args, **kwargs)

    return func


def wrap_vfs_makedirs(vfs: MemoryFS):
    def func(path, *args, **kwargs):
        path = str(path)
        kwargs.pop("exist_ok", None)
        return vfs.makedirs(path, *args, **kwargs)

    return func


def custom_locate(name: str):
    if name == "izer.backend.max7800x":
        from izer.backend import max7800x as izer_max7800x_backend

        return izer_max7800x_backend
    raise ImportError(f"Cannot locate {name}")


def no_makefile(*args, **kwargs):
    return None


def no_from_template(*args, **kwargs):
    return None


def no_vscode(*args, **kwargs):
    return None


virtual_files = {
    _config_file_yaml: open_yamlcfg,
    _checkpoint_file_onnx: open_checkpoint,
    _sample_input_npy: open_sample_input,
    _vscode_settings_json: open_vscode_settings_json,
    # _makefile: (lambda x: io.StringIO("")),
    # _projects_makefile: (lambda x: io.StringIO("")),
}


def prepare_vfs(*vfs_args):
    vfs = MemoryFS()
    for i, (name, func) in enumerate(virtual_files.items()):
        directory_path = fs.path.dirname(name)
        if not vfs.exists(directory_path):
            vfs.makedirs(directory_path)
        ret = func(vfs_args[i])
        if isinstance(ret, io.StringIO):
            with vfs.open(name, "w") as f:
                f.write(ret.getvalue())
        elif isinstance(ret, io.BytesIO):
            with vfs.open(name, "wb") as f:
                f.write(ret.getvalue())
        else:
            raise ValueError(f"Unsupported return type {type(ret)}")
    return vfs


def _get_inouts(node: onnx.NodeProto) -> tuple[list[str], list[str]]:
    inputs = []
    outputs = []
    for inp in node.input[
        :-1
    ]:  # dont use the last input, because we dont want to use bias
        inputs.append(inp)
    for out in node.output[:]:  # dont care about more
        outputs.append(out)
    return inputs, outputs


def _process_channels(  # noqa: C901
    model: onnx.ModelProto, _input: str, initializers: set[str]
) -> np.ndarray:
    """
    trampoline for izer.onnxcp.process_channels
    Remove kernel_shape attribute from Conv node when checking for bias
    """
    # remove kernel_shape attribute from Conv node
    probable_nodes = [
        node
        for node in model.graph.node
        if node.op_type in ["Gemm", "Conv"] and _input in node.input
    ]
    # check if a bias has a kernel_shape and remove it
    # or update the kernel_shape of a weight
    for node in probable_nodes:
        if node.op_type == "Conv":  # if it is a conv node
            if (
                len(node.input) >= 3 and _input == node.input[2]
            ):  # if the input is the bias
                # remove kernel_shape from this bias
                with contextlib.suppress(ValueError):
                    removal_idx = [attr.name for attr in node.attribute].index(
                        "kernel_shape"
                    )
                    node.attribute.pop(removal_idx)
                    # remove kernel_shape so that we can use the function and do not add
                    # the kernel shape a second time
            elif len(node.input) >= 2 and _input == node.input[1]:
                # if its a weight, make sure its atleast of length 2
                with contextlib.suppress(ValueError):
                    kernel_shape_idx = [attr.name for attr in node.attribute].index(
                        "kernel_shape"
                    )
                    if len(node.attribute[kernel_shape_idx].ints) == 1:
                        # just check the weights of the node
                        size = node.attribute[kernel_shape_idx].ints[0]
                        node.attribute[kernel_shape_idx].ints[:] = [size, 1]

    # since hannah produces different models, where the initializers are empty,
    # but the weights are in the inputs, we need to check both
    if _input in [input.name for input in model.graph.input]:
        # the thing is now, it has to be a weight
        all_weights = [
            node.input[1]
            for node in model.graph.node
            if len(node.input) > 1 and node.op_type in ["Conv", "Gemm"]
        ]
        if _input in all_weights:
            for _input_ in model.graph.input:
                if _input == _input_.name:
                    # as it is an input and not an initializer, we dont have a value her
                    # so thats why we do numpy.zeros
                    w = np.zeros(
                        shape=[
                            dim.dim_value for dim in _input_.type.tensor_type.shape.dim
                        ],
                    ).astype(np.int64)
                    # we have to return early here, because
                    # we dont want to check the initializers and then return None
                    return w

    # proceed with original code
    if _input in initializers:
        for _init in model.graph.initializer:
            if _input == _init.name:
                w = onnx.numpy_helper.to_array(_init).astype(np.int64)
                break
    else:
        w = None
    return w


def eprint_hooked(*args, error=True, notice=False, prefix=True, exit_code=1, **kwargs):
    message = "{}".format(*args)
    if notice:
        logging.info(message)
    elif error:
        logging.error(message)
        raise SystemExit(message)
    else:
        logging.warning(message)


def layout_transform(
    modelcfg: dict,
    model: onnx.ModelProto,
    sample_input: np.ndarray,
    **kwargs,
) -> list[int]:
    """
    Idea: we let ai8xize do the job
    """

    vscode_settings_defaults = {
        "PROGRAM_FILE": "path/to/program",
        "SYMBOL_FILE": "path/to/symbol",
        "M4_OCD_INTERFACE_FILE": "path/to/m4_ocd_interface",
        "M4_OCD_TARGET_FILE": "path/to/m4_ocd_target",
        "RV_OCD_INTERFACE_FILE": "path/to/rv_ocd_interface",
        "RV_OCD_TARGET_FILE": "path/to/rv_ocd_target",
        "C_CPP.DEFAULT.DEFINES": ["define1", "define2"],
        "C_CPP.DEFAULT.INCLUDEPATH": ["path/to/include"],
        "C_CPP.DEFAULT.BROWSE.PATH": ["path/to/browse"],
        "V_ARM_GCC": "version_arm_gcc",
        "V_XPACK_GCC": "version_xpack_gcc",
        "OCD_PATH": "path/to/ocd",
        "ARM_GCC_PATH": "path/to/arm_gcc",
        "XPACK_GCC_PATH": "path/to/xpack_gcc",
        "MAKE_PATH": "path/to/make",
        "MSYS_PATH": "path/to/msys",
    }

    vfs = prepare_vfs(
        modelcfg,
        model,
        sample_input,
        vscode_settings_defaults,
    )

    refs = []

    # init the state
    with (
        mock.patch("izer.commandline.get_parser", get_parser),
        mock.patch("os.path.exists", vfs.exists),
        mock.patch("os.path.isdir", vfs.isdir),
        mock.patch("os.makedirs", wrap_vfs_makedirs(vfs)),
        mock.patch("builtins.open", wrap_vfs_open(vfs)),
        mock.patch("izer.izer.locate", custom_locate),
        mock.patch("izer.apbaccess.apbwriter", get_custom_writer(refs)),
        mock.patch("izer.assets.makefile", no_makefile),
        mock.patch("izer.assets.from_template", no_from_template),
        mock.patch("izer.assets.vscode", no_vscode),
        # functional
        # mock.patch("izer.onnxcp.get_inouts", _get_inouts),
        mock.patch("izer.onnxcp.process_channels", _process_channels),
        mock.patch("izer.eprint.eprint", eprint_hooked),
        mock.patch("izer.izer.eprint", eprint_hooked),
        mock.patch("izer.backend.max7800x.eprint", eprint_hooked),
        # patch rich away as it may interfere with our threadrunners
        # mock.patch("izer.console.Progress", mock.MagicMock()),
    ):
        izer_main()

    return refs


def get_layer_information(
    current_information: dict,
    model: onnx.ModelProto,
    layer_count: int,
    input_dim: tuple,
    **kwargs,
) -> list[dict[str, Any]]:
    """
    Idea: we let ai8xize do the job
    """

    raise NotImplementedError("this method is not fully functional")

    vscode_settings_defaults = {
        "PROGRAM_FILE": "path/to/program",
        "SYMBOL_FILE": "path/to/symbol",
        "M4_OCD_INTERFACE_FILE": "path/to/m4_ocd_interface",
        "M4_OCD_TARGET_FILE": "path/to/m4_ocd_target",
        "RV_OCD_INTERFACE_FILE": "path/to/rv_ocd_interface",
        "RV_OCD_TARGET_FILE": "path/to/rv_ocd_target",
        "C_CPP.DEFAULT.DEFINES": ["define1", "define2"],
        "C_CPP.DEFAULT.INCLUDEPATH": ["path/to/include"],
        "C_CPP.DEFAULT.BROWSE.PATH": ["path/to/browse"],
        "V_ARM_GCC": "version_arm_gcc",
        "V_XPACK_GCC": "version_xpack_gcc",
        "OCD_PATH": "path/to/ocd",
        "ARM_GCC_PATH": "path/to/arm_gcc",
        "XPACK_GCC_PATH": "path/to/xpack_gcc",
        "MAKE_PATH": "path/to/make",
        "MSYS_PATH": "path/to/msys",
    }

    respective_layer = current_information.get("layers", []) + [None] * layer_count

    vfs = prepare_vfs(
        {
            "arch": "unknown",
            "dataset": "unknown",
            "layers": [
                respective_layer[i]
                if respective_layer[i]
                # else dummy layer
                else {
                    "name": str(i),
                    "processors": 1,
                    "operator": "conv2d",  # just pretend
                }
                for i in range(layer_count)
            ],
        },
        model,
        np.zeros(input_dim, dtype=int),
        vscode_settings_defaults,
    )

    state = None

    # init the state
    with (
        mock.patch("izer.commandline.get_parser", get_parser),
        mock.patch("os.path.exists", vfs.exists),
        mock.patch("os.path.isdir", vfs.isdir),
        mock.patch("os.makedirs", wrap_vfs_makedirs(vfs)),
        mock.patch("builtins.open", wrap_vfs_open(vfs)),
        mock.patch("izer.izer.locate", custom_locate),
        mock.patch("izer.apbaccess.apbwriter", get_custom_writer([])),
        mock.patch("izer.assets.makefile", no_makefile),
        mock.patch("izer.assets.from_template", no_from_template),
        mock.patch("izer.assets.vscode", no_vscode),
        # functional
        # mock.patch("izer.onnxcp.get_inouts", _get_inouts),
        mock.patch("izer.onnxcp.process_channels", _process_channels),
        mock.patch("izer.eprint.eprint", eprint_hooked),
        mock.patch("izer.izer.eprint", eprint_hooked),
        mock.patch("izer.backend.max7800x.eprint", eprint_hooked),
        mock.patch("izer.izer.locate"),
        mock.patch("izer.izer.state") as state_hooked,
        # patch rich away as it may interfere with our threadrunners
        # mock.patch("izer.console.Progress", mock.MagicMock()),
    ):
        state_hooked.layer_name = [""] * 1000  # until overwritten
        # init state with state variables
        izer_main()
        # now quickly save the state
        state = state_hooked
    return state
