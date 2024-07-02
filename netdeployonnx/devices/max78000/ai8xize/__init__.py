import onnx

from netdeployonnx.devices.max78000 import MAX78000
from netdeployonnx.devices.max78000.ai8xize.wrap_ai8ize import (
    layout_transform as wrap_ai8ize_layout_transform,
)
from netdeployonnx.devices.max78000.cnn_registers import (
    register_class_by_address,
)
from netdeployonnx.devices.max78000.core import CNNx16Core


class MAX78000_ai8xize(MAX78000):  # noqa: N801
    async def layout_transform(self, model: onnx.ModelProto) -> any:
        cfg = {
            "arch": "ai85nascifarnet",
            "dataset": "CIFAR10",
            "layers": [
                {
                    "out_offset": 0x4000,
                    "processors": 0x0000000000000007,  # 1_1
                    "operation": "conv2d",
                    "kernel_size": "3x3",
                    "pad": 1,
                    "activate": "ReLU",
                    "data_format": "HWC",
                },
                {
                    "out_offset": 0x0000,
                    "processors": 0xFFFFFFFFFFFFFFFF,  # 1_2
                    "operation": "conv2d",
                    "kernel_size": "1x1",
                    "pad": 0,
                    "activate": "ReLU",
                    "output_shift": -1,
                },
                {
                    "out_offset": 0x4000,
                    "processors": 0x00000000FFFFFFFF,  # 1_3
                    "operation": "conv2d",
                    "kernel_size": "3x3",
                    "pad": 1,
                    "activate": "ReLU",
                    "output_shift": -1,
                },
                {
                    "max_pool": 2,
                    "pool_stride": 2,
                    "out_offset": 0x0000,
                    "processors": 0xFFFFFFFFFFFFFFFF,  # 2_1
                    "operation": "conv2d",
                    "kernel_size": "3x3",
                    "pad": 1,
                    "activate": "ReLU",
                    "output_shift": -3,
                },
                {
                    "out_offset": 0x4000,
                    "processors": 0xFFFFFFFF00000000,  # 2_2
                    "operation": "conv2d",
                    "kernel_size": "1x1",
                    "pad": 0,
                    "activate": "ReLU",
                },
                {
                    "max_pool": 2,
                    "pool_stride": 2,
                    "out_offset": 0x0000,
                    "processors": 0xFFFFFFFFFFFFFFFF,  # 3_1
                    "operation": "conv2d",
                    "kernel_size": "3x3",
                    "pad": 1,  # do 0?
                    "activate": "ReLU",
                    "output_shift": -3,
                },
                {
                    "out_offset": 0x4000,
                    "processors": 0xFFFFFFFFFFFFFFFF,  # 3_2
                    "operation": "conv2d",
                    "kernel_size": "1x1",
                    "pad": 0,
                    "activate": "ReLU",
                    "output_shift": -1,
                },
                {
                    "max_pool": 2,
                    "pool_stride": 2,
                    "out_offset": 0x0000,
                    "processors": 0xFFFFFFFFFFFFFFFF,  # 4_1
                    "operation": "conv2d",
                    "kernel_size": "3x3",
                    "pad": 1,
                    "activate": "ReLU",
                    "output_shift": -3,
                },
                {
                    "out_offset": 0x4000,
                    "processors": 0xFFFFFFFFFFFFFFFF,  # 4_2
                    "operation": "conv2d",
                    "kernel_size": "3x3",
                    "pad": 1,
                    "activate": "ReLU",
                    "output_shift": -2,
                },
                {
                    "max_pool": 2,
                    "pool_stride": 2,
                    "out_offset": 0x0000,
                    "processors": 0xFFFFFFFFFFFFFFFF,  # 5_1
                    "operation": "conv2d",
                    "kernel_size": "1x1",
                    "pad": 0,
                    "activate": "ReLU",
                    "output_shift": -1,
                },
                {
                    "flatten": True,
                    "out_offset": 0x4000,
                    "processors": 0xFFFFFFFFFFFFFFFF,
                    "operation": "MLP",
                    "output_width": 32,
                    "activate": "none",
                    "output_shift": 1,
                },
            ],
        }
        list_of_results: list[any] = wrap_ai8ize_layout_transform(cfg, model)

        core = CNNx16Core()

        for apb in list_of_results:
            set_lregs_to_core(apb._lregs, core)

        return core


def set_lregs_to_core(lregs: list[any], core: CNNx16Core):
    for lreg in lregs:
        (quad, layeridx, reg, val, force_write, no_verify, comment) = lreg
        globalreg, local_reg = reg
        layer = core[quad, layeridx]
        local_reg &= 0xFFFF
        if local_reg in register_class_by_address:
            register_class = register_class_by_address[local_reg & 0xFFFF]
            register = register_class()
            register.value = val
            layer.set_from_register(register)
        else:
            raise ValueError(
                f"did not find register class for local_reg={local_reg:04X}"
                f"global_reg={globalreg:04X}"
            )
