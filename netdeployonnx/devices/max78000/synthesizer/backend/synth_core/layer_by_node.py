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
import numpy as np

from netdeployonnx.devices.max78000.core import CNNx16_Layer
from netdeployonnx.devices.max78000.synthesizer.grid import Node, NodeGrid


def calc_output_shape(
    input_shape: list[int],
    kernel_channels: int,
    kernel_shape: list[int],
    padding: list[int],
    stride: list[int],
) -> list[int]:
    """
    Calculate the output shape of a node
    """
    batch_size = input_shape[0]
    input_wh = input_shape[2], input_shape[3]
    return [
        batch_size,
        kernel_channels,
        (input_wh[0] - kernel_shape[0] + 2 * padding[0]) // stride[0] + 1,
        (input_wh[1] - kernel_shape[1] + 2 * padding[1]) // stride[1] + 1,
    ]


def configure_layer_by_node(
    layer: CNNx16_Layer, grid: NodeGrid, node: Node, input_shape: list[int]
) -> list[int]:
    """
    Configure a layer by a node
    """
    intermed_shape = [x for x in input_shape]
    if "Conv" in node.op_type:
        assert len(node.input) == 3, "convX does not have 3 inputs"
        X, W, B = list(node.input.values())  # noqa: N806
        attrs = node.attributes
        assert isinstance(W, np.ndarray)
        assert isinstance(B, np.ndarray)
        wdims = W.shape
        bdims = B.shape  # noqa: F841
        assert isinstance(wdims, tuple) and len(wdims) == 4, "wdims must be 4-tuple"
        pad = attrs.get("pads", [0 for w in wdims])
        stride = attrs.get("strides", (1, 1))

        kernel_shape = wdims[2:]

        pooling = attrs.get("_maxpool_kernel_shape", (0, 0))
        pooling_stride = attrs.get("_maxpool_strides", (1, 1))
        pooling_pad = attrs.get("_maxpool_pads", (0, 0, 0, 0))
        pool_ceil = attrs.get("_maxpool_ceil_mode", 0)  # noqa: F841

        # input is [1,3,32,32] = NCHW
        # first dim: [64,3,3,3] = KCkHkW
        # this means, we drag a 3x3 window over the input
        # and do that 64 times, but we add padding to the input
        # so we get an output shape [1,64,32,32]
        # no grouped convolutions or other confusing stuff
        assert input_shape[1] == wdims[1], "input channels must match kernel channels"
        # print(f"input for layer {layer.idx} is {inp} * {wdims}")
        # print(attrs)

        for fused_layers in ["pool"] if "_maxpool_pads" in attrs else []:
            intermed_shape = [
                intermed_shape[0],
                wdims[0],
                (intermed_shape[2] - pooling[0] + 2 * pooling_pad[2])
                // pooling_stride[0]
                + 1,
                (intermed_shape[3] - pooling[1] + 2 * pooling_pad[3])
                // pooling_stride[1]
                + 1,
            ]

        output_shape = calc_output_shape(
            input_shape=intermed_shape,
            kernel_channels=wdims[0],
            kernel_shape=kernel_shape,
            padding=pad,
            stride=stride,
        )

        out_expand = 1

    elif "Gemm" in node.op_type:
        # General Matrix Multiplication
        # this is a fully connected layer
        assert len(node.input) == 3, "gemm does not have 3 inputs"
        shape, weight, bias = node.input.values()
        assert isinstance(weight, np.ndarray)
        assert isinstance(bias, np.ndarray)
        wdims = weight.shape
        bdims = bias.shape  # noqa: F841

        pad = (0, 0, 0, 0)
        stride = (0, 0)
        out_expand = 10
        kernel_shape = (1, 1)

        pooling = (0, 0)
        pooling_stride = (0, 0)
        pooling_pad = (0, 0, 0, 0)

        intermed_shape[2] = 1
        intermed_shape[3] = 1
        output_shape = calc_output_shape(
            input_shape=intermed_shape,
            kernel_channels=wdims[0],
            kernel_shape=kernel_shape,
            padding=pad,
            stride=(1, 1),
        )
    else:
        raise NotImplementedError(f"op_type {node.op_type} not implemented")

    # TODO: remove magic number (tc.dev.MAX_PROC)
    expand = (input_shape[1] + 64 - 1) // 64

    configure_layer_sizes(
        layer,
        row=intermed_shape[2],
        col=intermed_shape[3],
        row_pad=pad[2],
        col_pad=pad[3],
        stride=stride[0],  # pooling stride?
        pooling_stride=pooling_stride[0],
        pooling="pool" in node.op_type.lower(),
        maxpooling=True,  # basically, we want everytime max pooling if we dont have
        # any avgpooling
    )
    configure_layer_control(
        layer,
        input_channels=intermed_shape[1],
        flatten=False,  # flatten="reshape" in node.op_type.lower(),
        relu="relu" in node.op_type.lower(),
        in_expand=expand,
        out_expand=out_expand,
    )
    configure_layer_kernels(
        layer,
        kernel_shape=kernel_shape,
        max_size=(0x1000 if layer.idx != 10 else 0x4000)
        if node.op_type != "Gemm"
        else 0,
    )
    configure_layer_post(layer)

    return output_shape


def configure_layer_sizes(
    layer: CNNx16_Layer,
    row: int = 0,
    col: int = 0,
    row_pad: int = 0,
    col_pad: int = 0,
    stride: int = 0,
    pooling_stride: int = 0,
    pooling: bool = False,
    maxpooling: bool = False,  # if false this is avg-pooling
):
    """
    Configure the sizes of a layer
        - col_count
        - row_count
        - col_pad
        - row_pad
    """
    layer.maxpool_en = maxpooling
    layer.pool_en = pooling
    if pooling:
        pool_dilation = 1  # its max78000, so always 1
        layer.row_pooling = (2 - 1) * pool_dilation
        layer.col_pooling = (2 - 1) * pool_dilation
    if stride > 1:
        layer.stride = stride - 1
    if pooling_stride > 1:
        layer.stride = pooling_stride - 1
    if pooling_stride == 0:
        pooling_stride = 1
    # print(f"stride {stride} pooling {pooling} expand {expand}")
    row_mul = pooling_stride
    col_mul = pooling_stride
    assert row_mul >= 1, "row_mul must be positive"
    assert col_mul >= 1, "col_mul must be positive"
    layer.row_count = row * row_mul + 2 * row_pad - 1
    layer.col_count = col * col_mul + 2 * col_pad - 1
    layer.row_pad = row_pad
    layer.col_pad = col_pad

    # print(
    #     f"Ly{layer.idx}",
    #     layer.row_count,
    #     layer.col_count,
    #     layer.row_pad,
    #     layer.col_pad,
    #     f"stride {stride}" if stride > 1 else "",
    #     "pool" if pooling else "",
    # )


def configure_layer_kernels(
    layer: CNNx16_Layer,
    max_size: int,
    kernel_shape: tuple[int, int],
    local_source: bool = False,
):
    """
    Configure the kernels of a layer
        - masks
        - wptr
        - bptr
        - rptr
        - tptr
    """

    # https://github.com/analogdevicesinc/MaximAI_Documentation/blob/ffc43ad693f4e79dc3cfbcdaa6f6f09499d4a135/Guides/YAML%20Quickstart.md
    # Note 5: Note 5: The input of each layer is taken from the output offset of the
    # previous layer. To avoid overwriting an input that has not been consumed, use
    # ping-ponging between out_offset=0 and half the memory (0x4000) or less in
    # consecutive layers
    layer.writeptr = max_size if layer.idx % 2 == 0 else 0

    layer.writeptr_mask_offset = 0
    layer.writeptr_timeslot_offset = 0

    # TODO: search for gaps / "Local output must be used:"
    layer.sram_load_source = not local_source  # if not local_source

    layer.bias_addr = 0
    layer.bias_en = False

    layer.readptr = 0

    # when the kernel is 1x1?
    layer.oned_mask_width = 1 if (kernel_shape == (1, 1)) else 0


def configure_layer_control(
    layer: CNNx16_Layer,
    input_channels: int,
    flatten: bool,
    relu: bool,
    in_expand: int = 1,
    out_expand: int = 1,
):
    """
    Configure the actions
        - enable
        - relu,
        - padding
        - oned / flatten
        - striding
    """
    # depending on the input channels and the data format, this is for 3 channels either
    # 0x7 (Height x Weight x Channels) or 0x111 (Channels x Height x Weight)
    # 0b111                               0b0001_0001_0001
    if flatten:
        # enable everything
        layer.flatten_en = True
        layer.enable_mask = 0xFFFF
        layer.enable_processor = 0xFFFF
    layer.enable_mask = (2 ** min(input_channels, 16)) - 1
    layer.enable_processor = (2 ** min(input_channels, 16)) - 1
    layer.relu_en = relu

    # from ai8x-synth: Channels 64+ handled by processors 0+ (expand)
    if in_expand >= 1:
        layer.expansion_mode_inputchan = in_expand - 1
    if out_expand >= 1:
        quant = 8
        layer.expansion_mode_processors = 63 * quant

    layer.master = True  # maxim always enables master (0x20 on LCTRL0)

    layer.deconv_en = False


def configure_layer_post(layer: CNNx16_Layer):
    """
    Configure the post-processing of a layer
    """
    # expansion mode, multipass, oned / onexone, ?
    ...
