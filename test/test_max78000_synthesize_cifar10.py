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
import asyncio
import logging
import math
import pathlib
import re
import struct

import anyio
import onnx
import pytest
from PIL import Image, ImageColor, ImageDraw

import netdeployonnx.devices.max78000.cnn_constants as cnn_constants
from netdeployonnx.devices.max78000 import (
    MAX78000,
    CNNx16_Layer,
    CNNx16_Quadrant,
)

from .data.cifar10_layout import cifar10_layout

logging.basicConfig(
    level=logging.INFO,
    format="[+{relativeCreated:2.2f}ms] {levelname}: ({funcName:10s}) {message}",
    style="{",
)


async def synthesize_function(return_type, name, args, instrs=[]):
    """
    synthesizes a function with the given return type, name, arguments, and instructions
    """
    lines = []
    lines.append(f"{return_type} {name}({', '.join(args)}) \n{{")
    for indent, text in instrs:
        indent_ = " " * indent
        lines.append(f"{indent_}{text}")
    lines.append("}")
    lines.append("")
    lines.append("")
    return lines


async def transform_instructions(synth_funcname, instructions):  # noqa: C901
    """
    transforms instructions to be synthesized
    """
    transformed_instructions_list = []

    def reg_32_write(reg: str, val: int) -> str:
        # get address from regname
        reg_addr = cnn_constants.registers.get(reg, 0)
        return f"*((volatile uint32_t *)0x{reg_addr:08x}) = 0x{val:08x};"

    logging.info(f"synthesizing instructions for function: {synth_funcname}")  # noqa

    if synth_funcname in ["cnn_configure"]:
        logging.info("reordering instructions")
        # reorder the layer control 2 to layer control 1
        for running_idx in range(len(instructions)):
            lctrl0_instr = instructions[running_idx]
            if isinstance(lctrl0_instr, tuple) and "LCTRL0" in lctrl0_instr[0]:
                # now find the next LCTRL1
                # search max 16 instructions ahead
                for next_idx in range(running_idx + 1, running_idx + 9):
                    if next_idx >= len(instructions):
                        break
                    lctrl1_instr = instructions[next_idx]
                    if isinstance(lctrl1_instr, tuple) and "LCTRL1" in lctrl1_instr[0]:
                        # now move the LCTRL1 just after the LCTRL0
                        instructions.insert(running_idx + 1, lctrl1_instr)
                        instructions.pop(next_idx + 1)
                        logging.debug(
                            f"INFO: reordered LCTRL0 and LCTRL1 at {running_idx}"
                            f" and {next_idx}"
                        )
                        break

        # reorder the enable after post
        for running_idx in range(len(instructions)):
            post_instr = instructions[running_idx]
            if isinstance(post_instr, tuple) and "POST" in post_instr[0]:
                # now find the next ENABLE
                # search max 16 instructions before
                for eninstr_idx in range(running_idx - 1, running_idx - 16, -1):
                    if eninstr_idx >= len(instructions):
                        break
                    if eninstr_idx < 0:
                        break
                    if isinstance(instructions[eninstr_idx], str):
                        break
                    en_instr = instructions[eninstr_idx]
                    if isinstance(en_instr, tuple) and "_EN" in en_instr[0]:
                        # now move the ENABLE just after the POST
                        val = instructions.pop(eninstr_idx)  # remove the EN
                        instructions.insert(running_idx, en_instr)
                        logging.debug(
                            f"INFO: reordered POST and EN; ({eninstr_idx}) is"
                            " now at {running_idx+1}"
                        )
                        break

        # reorder ONED after MCNT
        for running_idx in range(len(instructions)):
            oned_instr = instructions[running_idx]
            if isinstance(oned_instr, tuple) and "ONED" in oned_instr[0]:
                # now find the next MCNT
                # search max 16 instructions ahead
                for mcnt_idx in range(running_idx, running_idx + 16):
                    if mcnt_idx >= len(instructions):
                        break
                    if mcnt_idx < 0:
                        break
                    if isinstance(instructions[mcnt_idx], str):
                        break
                    mcnt_instr = instructions[mcnt_idx]
                    if isinstance(mcnt_instr, tuple) and "MCNT" in mcnt_instr[0]:
                        # now move the ONED just after the MCNT
                        val = instructions.pop(running_idx)  # noqa: F841
                        instructions.insert(mcnt_idx, oned_instr)
                        logging.debug(
                            f"INFO: reordered MCNT and ONED; ({running_idx}) is"
                            " now at {mcnt_idx+1}"
                        )

    for instruction in instructions:
        if isinstance(instruction, str):  # just a direct entry
            transformed_instructions_list.append((4, f"{instruction}"))
        elif isinstance(instruction, tuple):  # either reg or mem access
            if len(instruction) not in [2, 3]:
                raise ValueError(f"invalid instruction: {instruction}")
            if len(instruction) == 2:
                instruction_dest, instruction_value = instruction
                if isinstance(instruction_value, int):  # reg access
                    transformed_instr = reg_32_write(
                        instruction_dest, instruction_value
                    )
                elif isinstance(instruction_value, (list, bytes)):  # mem access
                    if synth_funcname in ["cnn_load_bias"]:
                        check_bias_from_instructions(instructions)
                        quad = 0
                        transformed_instr = (
                            f"memcpy_8to32((uint32_t *)0x{instruction_dest:08X},"
                            f" bias_{quad}, sizeof(uint8_t) *"
                            f" {len(instruction_value)});"
                        )
                    elif synth_funcname in ["cnn_load_weights"]:
                        check_kernels_from_instructions(instructions)
                        transformed_instr = (
                            f"write_at_address(0x{instruction_dest:08X},"
                            f" {len(instruction_value): 5d},"
                            f" (uint32_t *)kernel_quad_layer);"
                        )
                    else:
                        raise NotImplementedError(
                            "invalid synth_funcname for mem access"
                        )
                    instruction = (f"0x{instruction_dest:08X}", b"...")  # noqa
                else:
                    raise ValueError(f"invalid instruction: {instruction}")
                transformed_instructions_list.append(
                    (4, f"{transformed_instr} // {instruction}")
                )
            elif len(instruction) == 3:
                # we may have an action instead of an instruction
                action, action_value, action_mask = instruction
                transformed_instructions_list.append((4, f"// action: {action}"))

    if synth_funcname in ["cnn_init", "cnn_start", "cnn_configure"]:
        transformed_instructions_list += [
            (4, ""),
            (4, "return CNN_OK;"),
        ]
    return transformed_instructions_list


async def synthesize_to_c(instruction_dict_list: [], output_file_path: str) -> None:
    """
    Synthesize the instructions to a C file

        How this method works:
        - It synthesizes the functions in the order of the synth_order
        - It transforms the instructions to be synthesized
        - It synthesizes the functions
        - It writes the synthesized functions to the output

    """
    lines = [
        "#include <stdlib.h>",
        "#include <stdint.h>",
        "#include <string.h>",
        "#include <stdio.h>",
        "",
        "",
    ]

    # synthesize_to_c function order:
    synth_order = [
        "CNN_ISR",
        "cnn_continue",
        "cnn_stop",
        "cnn_load_weights",
        "cnn_load_bias",
        "cnn_init",
        "cnn_configure",
        "cnn_start",
        "cnn_unload",
        "cnn_enable",
        # boost
        # boost disable
        "cnn_disable",
    ]

    for synth_funcname in synth_order:
        instruction_dict = next(
            (
                instruction_dict
                for instruction_dict in instruction_dict_list
                if instruction_dict.get("stage") == synth_funcname
            ),
            None,
        )
        if instruction_dict:
            instructions = instruction_dict.get("instructions", [])
            transformed_instructions = await transform_instructions(
                synth_funcname, instructions
            )
            lines += await synthesize_function(
                "int",
                synth_funcname,
                [
                    (
                        "uint32_t clock_source, uint32_t clock_divider"
                        if synth_funcname == "cnn_enable"
                        else "void"
                    )
                ],
                transformed_instructions,
            )
        else:
            lines += await synthesize_function(
                "void" if synth_funcname in ["CNN_ISR"] else "int",
                synth_funcname,
                ["uint32_t *out_buf" if synth_funcname == "cnn_unload" else "void"],
                [
                    (4, "// not synthesized"),
                ],
            )

    async with await anyio.open_file(output_file_path, "w") as synthesized_file:
        synth = ""
        for line in lines:
            synth += f"{line}\n"
        await synthesized_file.write(synth)
    logging.info(f"wrote synthesized file to ./{output_file_path}")


def check_bias_from_layout(layout):
    from .data.cifar10_bias import correct_bias

    for quad, correct_bias_per_quad in enumerate(correct_bias):
        for idx, bias in enumerate(correct_bias_per_quad):
            assert (
                layout[quad].bias[idx] == bias
            ), f"bias faulty at quad {quad} idx {idx} expected {bias} got"
            f" {layout[quad].bias[idx]}"
    else:
        logging.info("bias all good")
        assert True


def check_bias_from_instructions(instructions):
    from .data.cifar10_bias import correct_bias_instructions

    for idx, instr in enumerate(correct_bias_instructions):
        assert (
            instr == instructions[idx]
        ), f"bias faulty at idx {idx} expected 0x{instr[0]:08X} got"
        f" 0x{instructions[idx][0]:08X}"
    else:
        logging.info("bias all good")
        assert True


def set_kernel(layout):
    from .data.cifar10_weights import kernels as kernel_list_correct  # noqa

    reverse_maddr = {
        cnn_constants.memory.get(f"CNNx16_{quad}_L{layer}_MRAM"): (quad, layer)
        for quad in range(4)
        for layer in range(16)
    }

    for kernel_entry in kernel_list_correct:
        kernel_addr, kernel_data = kernel_entry["addr"], kernel_entry["data"]
        offs = kernel_addr % 0x4000
        mem_addr = kernel_addr - offs

        if mem_addr in reverse_maddr:
            quad, layer = reverse_maddr[mem_addr]
            layout[quad, layer].kernels[offs] = kernel_data


def check_kernels_from_instructions(kernel_list: list[tuple[int, bytes]]):
    from .data.cifar10_weights import kernels as kernel_list_correct  # noqa

    kernel_list_correct_transformed = [
        (kernel["addr"], kernel["data"]) for kernel in kernel_list_correct
    ]
    # print(find_memsect(kernel_list_correct_transformed[0][0]), \
    #   find_memsect(kernel_list[0][0]))
    # print(f"{kernel_list_correct_transformed[0][0]:08X}, {kernel_list[0][0]:08X}")
    assert len(kernel_list_correct_transformed) == len(
        kernel_list
    ), "Kernel list is not the same length"
    for i in range(len(kernel_list)):
        if kernel_list_correct_transformed[i] != kernel_list[i]:
            print(f"Kernel list is not the same @ index {i}")
            print(f"correct: {kernel_list_correct_transformed[i][0]:08X}, [...]")
            print(f"actual: {kernel_list[i][0]:08X}, [...]")
        assert (
            kernel_list_correct_transformed[i] == kernel_list[i]
        ), f"Kernel list is not the same @ index {i}"

    logging.info("kernel all good")


def write_and_check_kernels(kernel_list: list[tuple[int, bytes]]):
    def find_memsect(addr):
        for memorysect, addrx in cnn_constants.memory.items():
            if addrx <= addr and addr < (addrx + 0x4000):
                return memorysect, addr - addrx
        raise ValueError(f"Address {addr} not in any memory section")

    check_kernels_from_instructions(kernel_list)

    kernels = []

    # get all kernels and get it into the format [addr, length, data[...], data[...+4]]
    for kernel_at_address in kernel_list:
        addr = kernel_at_address[0]
        data = kernel_at_address[1]
        memsect, offs = find_memsect(addr)
        logging.debug(f"Kernel at [{memsect}+{offs:04X}] with {len(data)} bytes")
        kernels.append(addr)
        assert len(data) % 4 == 0, "Data is not a multiple of 4"
        kernels.append(len(data) // 4)
        # now get 4 bytes each
        for i in range(0, len(data), 4):
            packed = struct.unpack(">I", (data[i : i + 4] + b"\0" * 4)[:4])
            kernels.append(packed[0])

    kernels.append(
        0
    )  # so addr 0 is the end of the kernel list and the while unpacker will stop

    # format the array kernels to a c array (like in weights.h)
    kernels_c = ""
    values_per_line, linewidth = 7, 96
    max_lines = math.ceil(len(kernels) / values_per_line)
    for lineidx in range(max_lines):
        line = ", ".join(
            [
                f"0x{val:08x}"
                for val in kernels[
                    lineidx * values_per_line : min(
                        (lineidx + 1) * values_per_line, len(kernels) - 1
                    )
                ]
            ]
        )
        if lineidx == max_lines - 1:
            line += ", 0x00000000"
        else:
            line += ","
        linestart = " " * (8 if lineidx == 0 else 12)
        lineadd = " " * (linewidth - len(line) - len(linestart))
        kernels_c += f"{linestart}{line}{lineadd}\\\n"

    # write the kernel to the layout
    try:
        # check if the kernel is in the weights.h file
        with open("../embedded/code/cifar-10/weights.h") as weights_file:
            weights = weights_file.read()
            spacr = " " * 91
            findpattern = f"    {{{spacr}\\\n"
            start_idx = weights.find(findpattern)
            assert start_idx >= 0, "Did not find the start"
            start_idx += len(findpattern)
            for i in range(len(kernels_c)):
                # print(f"'{kernels_c[i]} is equal to '{weights[start_idx + i]}'?")
                assert (
                    kernels_c[i] == weights[start_idx + i]
                ), f"different at line {lineidx} => @{i}"
            assert True
            logging.info("kernels are all good compared to ..../cifar10/weights.h from")
    except AssertionError as asserr:
        print("Kernel not in weights.h", asserr)


def draw_layer(
    layer: "CNNx16_Layer",
    draw: ImageDraw,
    x,
    y,
    rect_width,
    rect_height,
    back_color="white",
):
    # width = rect_width - 10
    # height = rect_height - 10
    draw.text(
        (x + rect_width - 40, y + 3),
        f"R{layer.row_count}",
        fill="black" if layer.row_count else back_color,
    )
    draw.text(
        (x + rect_width - 20, y + 3),
        f"C{layer.col_count}",
        fill="black" if layer.col_count else back_color,
    )
    draw.text(
        (x + rect_width - 50, y + 13),
        f"Pad: R{layer.row_pad}",
        fill="black" if layer.row_pad else back_color,
    )
    draw.text(
        (x + rect_width - 15, y + 13),
        f"C:{layer.col_pad}",
        fill="black" if layer.col_pad else back_color,
    )
    draw.text(
        (x + rect_width - 50, y + 23),
        f"Pool:R{layer.row_pooling}",
        fill="black" if layer.row_pooling else back_color,
    )
    draw.text(
        (x + rect_width - 15, y + 23),
        f"C:{layer.col_pooling}",
        fill="black" if layer.col_pooling else back_color,
    )

    draw.text((x + 3, y + 40), f"STRID: {layer.stride}", fill="black")
    draw.text((x + 3, y + 50), f"WPTR : 0x{layer.writeptr:04x}", fill="black")
    draw.text((x + 3, y + 60), f"RPTR : 0x{layer.readptr:04x}", fill="black")
    draw.text((x + 3, y + 70), f"M_MAX: 0x{layer.mask_maxaddr:04x}", fill="black")
    draw.text((x + 3, y + 80), f"M_SAD: 0x{layer.mask_start:04x}", fill="black")
    draw.text((x + 3, y + 90), f"Master: {layer.master}", fill="black")

    draw.text((x + 3, y + 100), f"SRAM: {layer.sram_load_source}", fill="black")
    draw.text((x + 3, y + 110), f"RELU: {layer.relu_en}", fill="black")
    draw.text((x + 3, y + 120), f"POOL: {layer.pool_en}", fill="black")
    draw.text((x + 3, y + 130), f"MPOL: {layer.maxpool_en}", fill="black")
    draw.text((x + 3, y + 140), f"BIAS: {layer.bias}", fill="black")
    draw.text((x + 3, y + 150), f"BIASADDR: {layer.bias_addr:03X}", fill="black")
    draw.text(
        (x + 3, y + 160), f"EXP: {layer.expansion_mode_processors:04b}", fill="black"
    )


def draw_quadrant(quadrant: CNNx16_Quadrant, image):
    draw = ImageDraw.Draw(image)
    memory_height = 400
    width = image.size[0] - 2 * 10  # subtract 50 for padding each
    height = image.size[1] - 2 * 10 - memory_height  # subtract 50 for padding each
    rect_width = width / 16
    rect_height = height / 4
    for layer in range(16):
        x = layer * rect_width + 10
        y = quadrant.idx * rect_height + 10
        fill_color = "gray" if quadrant[layer].unused else "white"
        fill_color = (
            "silver" if quadrant[layer].writeptr_multipass_offset else fill_color
        )
        draw.rectangle(
            [(x, y), (x + rect_width, y + rect_height)],
            outline="black",
            fill=fill_color,
        )
        draw.text((x + 3, y + 3), f"Q{quadrant.idx} L{layer}", fill="black")
        draw_layer(
            quadrant[layer], draw, x, y, rect_width, rect_height, back_color=fill_color
        )
    return rect_width, rect_height


async def render_layout(layout):  # noqa: C901
    try:
        # Create a new image with a white background
        width, height = 4 * 1280, 720 * 2
        border_width, border_height = 10, 10
        image = Image.new("RGBA", (width, height), "white")

        for quadrant in range(4):
            quadrant_width, quadrant_height = draw_quadrant(layout[quadrant], image)

        memsect_width = image.size[0] - 2 * border_width  # subtract 50 for padding each
        memsect_height = 400
        draw = ImageDraw.Draw(image)
        memory_types = ["BIAS", "SRAM", "MRAM", "TRAM"]
        memsect_h = memsect_height / len(memory_types)
        for memrow, memtype in enumerate(memory_types):
            x = border_width
            y = height - memsect_height - border_height + memsect_h * memrow  # 50 pad
            draw.rectangle(
                [(x, y), (x + memsect_width, y + memsect_h)],
                outline="black",
                fill="white",
            )

        from scripts.analyze_cnn import weights_str

        for i, x in enumerate(weights_str.split("\n")):
            if x.strip().startswith("WEIGHTS:"):
                regex_res = re.search(r"0x([0-9A-Fa-f]+) => \((\d+)\)", x)
                addr = int(regex_res.group(1), 16)
                size = int(regex_res.group(2)) * 4  # because its 4 bytes

                # find closes mram
                for memory, memaddr in cnn_constants.memory.items():
                    dist = addr - memaddr
                    if (
                        memaddr <= addr and dist < 0x4000
                    ):  # max size per memsection is 16KB / 0x4000
                        break
                if not memory.endswith("MRAM"):
                    raise Exception(f"Invalid memory type: {memory}")
                match = re.match(r"CNNx16_(\d)_L(\d{1,2})_MRAM", memory)
                if not match:
                    raise Exception(f"Invalid memory type: {memory}")
                quad, layeridx = match.group(1), match.group(2)
                quad, layeridx = int(quad), int(layeridx)
                print(
                    f"Found weights at {addr:04x} [{size}] in {memory} for"
                    " layer {layeridx} in quad {quad}"
                )
                offs_addr = addr - memaddr

                x = border_width + (layeridx * memsect_width / 16)
                y = (
                    height
                    - border_height
                    - memsect_height
                    + memsect_h * 2
                    + quad * memsect_h / 4
                )

                x_addr = x + (offs_addr / 0x4000) * (memsect_width / 16)
                x_size = x_addr + (size / 0x4000) * (memsect_width / 16)
                draw.rectangle(
                    [x_addr, y, x_size, y + memsect_h / 8],
                    fill=(185, 112, 33, 58),
                    outline="black",
                )

        # bias
        for quad, (addr, size) in enumerate(
            {
                0x50108000: 202,
                0x50508000: 224,
                0x50908000: 224,
                0x50D08000: 192,
            }.items()
        ):
            # find closes bias
            offs_addr = addr % 0x8000
            x = border_width
            y = (
                height
                - border_height
                - memsect_height
                + memsect_h * 0
                + quad * memsect_h / 4
            )

            x_addr = x + (offs_addr / 0x1FF) * (memsect_width)
            x_size = x_addr + (size / 0x1FF) * (memsect_width)
            draw.rectangle(
                [x_addr, y, x_size, y + memsect_h / 8],
                fill=(23, 23, 180, 58),
                outline="black",
            )

        colors = ["red", "green", "blue", "cyan", "orange"]
        colors = [ImageColor.getrgb(color) for color in colors]
        # now get rgba with alpha = 0.5
        colors = [color[:3] + (200,) for color in colors]
        for conn_idx, (conn_attr, connection_type) in enumerate(
            {
                "writeptr": "SRAM",
                "readptr": "SRAM",
                "mask_start": "MRAM",
                "tram_maxaddr": "TRAM",
                "bias_addr": "BIAS",
            }.items()
        ):
            # draw a line from each quadrant to the memory, each using its own color
            bias_idx = 0
            mask_idx = 0
            tram_idx = 0

            def adjust_brightness(color, factor):
                return tuple(int(c * factor) for c in color)

            for quadrant in range(4):
                for layeridx in range(16):
                    layer = layout[quadrant, layeridx]
                    addr_offset = layer.idx * 0x4000

                    max_addr = 16 * 0x4000 if connection_type == "TRAM" else 0x20000
                    max_addr = 16 * 0x4000 if connection_type == "MRAM" else max_addr
                    max_addr = 0x1FF if connection_type == "BIAS" else max_addr

                    memidx = memory_types.index(connection_type)
                    x0 = border_width + (layeridx * quadrant_width) + quadrant_width / 2
                    y0 = (
                        border_height
                        + (quadrant * quadrant_height)
                        + quadrant_height / 2
                    )
                    y1 = (
                        (height - border_height)
                        - memsect_height
                        + memsect_h / 2
                        + memidx * memsect_height / 4
                    )
                    ycoord = y1 - memsect_h / 2 + memsect_h / 4 * quadrant
                    # now draw boxes for invalid memory
                    if connection_type in ["MRAM", "TRAM"]:
                        # see datasheet
                        normal_memory_size = 6144 if connection_type == "TRAM" else 6912
                        gray_out_start_addr = addr_offset + normal_memory_size
                        gray_out_end_addr = addr_offset + 0x4000
                        grayout_x0 = (
                            border_width
                            + (gray_out_start_addr) / (max_addr) * memsect_width
                        )
                        grayout_x1 = (
                            border_width
                            + (gray_out_end_addr) / (max_addr) * memsect_width
                        )

                        draw.rectangle(
                            [
                                grayout_x0,
                                ycoord + 1,
                                grayout_x1,
                                ycoord + memsect_h / 4 - 1,
                            ],
                            fill="gray",
                        )

                    # skip if unused
                    if layer.unused:
                        continue
                    addr = getattr(layer, conn_attr)
                    # skip if not set
                    if addr is None:
                        continue
                    if connection_type == "BIAS":
                        # skip only if bias is not set
                        if not layer.bias:
                            continue
                    if connection_type == "MRAM":
                        if addr == 0 and layer.mask_maxaddr == 0:
                            continue
                    else:
                        # skip if addr is 0
                        if addr == 0:
                            continue

                    if connection_type == "TRAM" or connection_type == "MRAM":
                        addr = addr_offset + addr

                    x1 = border_width + (addr) / (max_addr) * memsect_width

                    mask_size_bits = (
                        8 if layer.mask_size == 0 else 2 ** (layer.mask_size - 1)
                    )
                    output_size = min(layer.output_row * layer.output_col, 64)
                    box_size = (mask_size_bits * output_size) // 8
                    box_size = min(box_size, max_addr)
                    box_size = box_size / max_addr * memsect_width

                    if connection_type == "BIAS":
                        # draw a little rectangle for the size of the bias
                        # now subtract

                        bias_colors = [colors[conn_idx] for _ in range(4)]
                        # increase the alpha for each bias
                        bias_colors = [
                            color[:3] + (128,) for bi, color in enumerate(bias_colors)
                        ]
                        bias_colors = [
                            adjust_brightness(
                                color, 0.3 + 0.1 * ((ci + 1) * (ci + 1)) % 1.0
                            )
                            for ci, color in enumerate(bias_colors)
                        ]
                        bias_h = memsect_h / 8  # divide by quadrants
                        draw.rectangle(
                            [x1, ycoord + bias_h, x1 + box_size, ycoord + 2 * bias_h],
                            fill=bias_colors[bias_idx % len(bias_colors)],
                            outline="gray",
                        )
                        bias_idx += 1

                    if connection_type == "MRAM":
                        # draw a little rectangle for the size of the mask
                        mask_size = layer.mask_maxaddr - layer.mask_start
                        mask_size = min(mask_size, max_addr)
                        mask_size = mask_size / max_addr * memsect_width
                        # same base color, but different brightnesses, changing from
                        # dark to bright to semidark to semibright
                        mask_colors = [colors[conn_idx] for _ in range(4)]
                        mask_colors = [
                            adjust_brightness(
                                color, 0.3 + 0.1 * ((ci + 1) * (ci + 1)) % 1.0
                            )
                            for ci, color in enumerate(mask_colors)
                        ]
                        mask_h = memsect_h / 8  # divide by quadrants
                        draw.rectangle(
                            [x1, ycoord + mask_h, x1 + mask_size, ycoord + 2 * mask_h],
                            fill=mask_colors[mask_idx % len(mask_colors)],
                            outline="gray",
                        )
                        mask_idx += 1

                    if connection_type == "TRAM":
                        # draw a little rectangle for the size of the bias
                        tram_colors = [colors[conn_idx] for _ in range(4)]
                        # increase the alpha for each tram
                        tram_colors = [
                            color[:3] + (128,) for bi, color in enumerate(tram_colors)
                        ]
                        tram_colors = [
                            adjust_brightness(
                                color, 0.3 + 0.1 * ((ci + 1) * (ci + 1)) % 1.0
                            )
                            for ci, color in enumerate(tram_colors)
                        ]
                        tram_h = memsect_h / 4  # divide by quadrants

                        # output_size = min(layer.output_row * layer.output_col, 3)
                        # box_size = (mask_size_bits * output_size) // 8
                        # box_size = min(box_size, max_addr)
                        # box_size = box_size / max_addr * memsect_width

                        draw.rectangle(
                            [x1, ycoord + tram_h, x1 + box_size, ycoord + 2 * tram_h],
                            fill=tram_colors[tram_idx % len(tram_colors)],
                            outline="gray",
                        )
                        tram_idx += 1

                    draw.line(
                        [x0, y0, x1, ycoord + memsect_h / 8],
                        fill=colors[conn_idx],
                        width=1,
                    )

        for quadrant in range(4):
            for memidx, memtype in enumerate(memory_types):
                # label at the right end
                ycoord_memsection_start = (height - border_height) - memsect_height
                draw.text(
                    (
                        border_width + memsect_width - 50,
                        ycoord_memsection_start
                        + memidx * memsect_h
                        + quadrant * memsect_h / 4,
                    ),
                    f"Q{quadrant} {memtype}",
                    fill="black",
                )
                if quadrant == 0:
                    continue
                # draw a horizontal line to separate all the quadrants
                draw.line(
                    [
                        border_width,
                        ycoord_memsection_start
                        + memidx * memsect_h
                        + quadrant * memsect_h / 4,
                        border_width + memsect_width,
                        ycoord_memsection_start
                        + memidx * memsect_h
                        + quadrant * memsect_h / 4,
                    ],
                    fill="black",
                    width=1,
                )

        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        # Save the image
        background.save("canvas.png")
    except Exception:
        import traceback

        traceback.print_exc()


async def async_main():
    device = MAX78000()

    data_folder = pathlib.Path("test/data")
    # Load the ONNX model
    model = onnx.load(data_folder / "cifar10.onnx")
    onnx.checker.check_model(model)
    layout_ir = await device.layout_transform(model)

    layout = await cifar10_layout()

    assert layout_ir == layout

    await synthesize_to_c(await device.compile_instructions(layout), "cifar10_cnn.c")
    # await render_layout(layout)


@pytest.mark.skip("TODO: broken")
@pytest.mark.asyncio
async def test_run_async_main():
    await async_main()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    assert False, "run with pytest"
    main()
