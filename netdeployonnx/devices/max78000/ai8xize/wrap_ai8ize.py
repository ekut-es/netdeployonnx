import io
import json
import warnings
from dataclasses import dataclass
from unittest import mock

warnings.filterwarnings("ignore", category=DeprecationWarning)

import fs  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import yaml  # noqa: E402
from fs.memoryfs import MemoryFS  # noqa: E402
from izer import tornadocnn as tc  # noqa: E402
from izer.apbaccess import APBBlockLevel, APBTopLevel  # noqa: E402
from izer.izer import main as izer_main  # noqa: E402
from izer.izer import onnxcp as izer_onnxcp  # noqa: E402

_board_name = "assets.from_template.board_name"  # "EvKit_V1"
_config_file_yaml = "config_file.yaml"
_checkpoint_file_onnx = "checkpoint_file.onnx"
_sample_input_npy = "tests/sample_cifar-10.npy"
_vscode_settings_json = "assets/vscode/defaults/settings.json"
_test_dir = "sdk/Examples/MAX78000/CNN_2LY"
_api_filename = "_api_filename"
_makefile = "assets/makefile/Makefile"
_projects_makefile = "assets/makefile/projects.mk"


class CustomAPBBlocklevel(APBBlockLevel):
    def write(
        self,
        addr,
        val,
        comment="",
        indent="  ",
        no_verify=False,
        fifo=None,
        base=None,
        fifo_wait=True,
    ):
        """
        Write address `addr` and data `val` to the .c file.
        if `no_verify` is `True`, do not check the result
        of the write operation, even if
        `verify_writes` is globally enabled.
        An optional `comment` can be added to the output.
        """
        ...


class CustomAPBTopLevel(APBTopLevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_defines = []
        self._lregs = []
        self._bias = []

    def output_define(
        self,
        array,
        define_name,
        fmt,
        columns,
        weights=True,
    ):
        # holdup waitaminute
        self._output_defines.append((array, define_name, fmt, columns, weights))

    def write_lreg(
        self,
        group,
        layer,
        reg,
        val,
        force_write=False,
        no_verify=False,
        comment="",
    ):
        reg = tc.lreg_addr(group, reg, layer=layer), tc.lreg_addr(0, reg, layer=0)
        self._lregs.append((group, layer, reg, val, force_write, no_verify, comment))

    def write_bias(
        self,
        group,
        offs,
        bias,
    ):
        self._bias.append((group, offs, bias))


def get_custom_writer(refs):
    def custom_writer(*args, debug_mem=False, **kwargs):
        if not debug_mem:
            import izer.state as state

            APBClass = (  # noqa: N806
                CustomAPBBlocklevel
                if state.block_mode or debug_mem
                else CustomAPBTopLevel
            )
        else:
            raise NotImplementedError("debug_mem is not implemented")
        obj = APBClass(
            *args,
            **kwargs,
        )
        refs.append(obj)
        return obj

    return custom_writer


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
        embedded_code = True
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


def _process_channels(
    model: onnx.ModelProto, _input: str, initializers: set[str]
) -> np.ndarray:
    """
    trampoline for izer.onnxcp.process_channels
    Remove kernel_shape attribute from Conv node when checking for bias
    """
    # remove kernel_shape attribute from Conv node
    probable_nodes = [node for node in model.graph.node if _input in node.input]
    for node in probable_nodes:
        if node.op_type == "Conv":  # if it is a conv node
            if _input == node.input[2]:  # if the input is the bias
                # remove kernel_shape from this bias
                removal_idx = [attr.name for attr in node.attribute].index(
                    "kernel_shape"
                )
                node.attribute.pop(removal_idx)
                # remove kernel_shape so that we can use the function and do not add
                # the kernel shape a second time

    # proceed with original code
    if _input in initializers:
        for _init in model.graph.initializer:
            if _input == _init.name:
                w = onnx.numpy_helper.to_array(_init).astype(np.int64)
                break
    else:
        w = None
    return w


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
    ):
        izer_main()

    return refs
