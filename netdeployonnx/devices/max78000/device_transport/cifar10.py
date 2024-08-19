"""
# Configuring 11 layers
# Input data: HWC
# Layer 0: 3x32x32, no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 64x32x32 output
# Layer 1: 64x32x32, no pooling, conv2d with kernel size 1x1, stride 1/1, pad 0/0, ReLU, 32x32x32 output
# Layer 2: 32x32x32, no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 64x32x32 output
# Layer 3: 64x32x32, max pool 2x2 with stride 2/2, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 32x16x16 output
# Layer 4: 32x16x16, no pooling, conv2d with kernel size 1x1, stride 1/1, pad 0/0, ReLU, 64x16x16 output
# Layer 5: 64x16x16, max pool 2x2 with stride 2/2, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 128x8x8 output
# Layer 6: 128x8x8, no pooling, conv2d with kernel size 1x1, stride 1/1, pad 0/0, ReLU, 128x8x8 output
# Layer 7: 128x8x8, max pool 2x2 with stride 2/2, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 64x4x4 output
# Layer 8: 64x4x4, no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 128x4x4 output
# Layer 9: 128x4x4, max pool 2x2 with stride 2/2, conv2d with kernel size 1x1, stride 1/1, pad 0/0, ReLU, 128x2x2 output
# Layer 10: 128x2x2 flattened to 512x1x1, no pooling, linear, no activation, 10x1x1 output
"""

import os
import struct

from protobuffers import main_pb2


def invert_byte_order(data: bytes) -> bytes:
    """
    Inverts the byte order of each 4-byte chunk in the given bytes object.

    Args:
        data (bytes): The input bytes object.

    Returns:
        bytes: A new bytes object with the byte order of each 4-byte chunk inverted.

    Example:
        >>> data = b'\\x01\\x02\\x03\\x04\\x05\\x06\\x07\\x08'
        >>> invert_byte_order(data)
        b'\\x04\\x03\\x02\\x01\\x08\\x07\\x06\\x05'
    """
    return b"".join(data[i : i + 4][::-1] for i in range(0, len(data), 4))


def load_input():
    # load_input(); // Load data input
    ret = []
    with open("cifar10_sampledata.dat", "rb") as data:
        inp = data.read()
        # memcpy32
        ret.append((0x50400000, invert_byte_order(inp)))

    # ret = [main_pb2.ActionEnum.RUN_CNN_LOAD_INPUT]
    return ret


def cnn_load_weights():
    # cnn_load_weights(); // Load kernels
    ret = []
    with open("cifar10_weights.dat", "rb") as data:
        file_stat = os.fstat(data.fileno())
        file_size = file_stat.st_size
        while file_size > data.tell():
            # print(file_size, data.tell())
            addr = struct.unpack(">I", data.read(4))[0]
            if addr == 0:
                break
            size = struct.unpack(">I", data.read(4))[0]
            # print(addr, size, f"=> 0x{addr:X} [0x{size:X}]")
            inp = data.read(size * 4)
            # set address and copy uint32_t
            ret.append((addr, invert_byte_order(inp), True))
    # data = [
    #     0xfe08f8f6,
    #     0xfb080715,
    #     0x03f20dff,
    #     0xfb05f4f6,
    #     0xecee02f5,
    # ]
    # data_bytes = b"".join([struct.pack(">I", val) for val in data])
    # ret.append((0x20003448, invert_byte_order(data_bytes)))

    # ret = [main_pb2.ActionEnum.RUN_CNN_LOAD_WEIGHTS]
    return ret


def cnn_load_bias():
    # cnn_load_bias();
    ret = []
    with open("cifar10_biases.dat", "rb") as data:
        for addr, size in {
            0x50108000: 202,
            0x50508000: 224,
            0x50908000: 224,
            0x50D08000: 192,
        }.items():
            inp = data.read(size)
            # memcpy_8to32
            # we need to space them out
            result = b""
            for i in range(len(inp)):
                result += bytes([inp[i], 0, 0, 0])
            ret.append((addr, result))

    # ret = [main_pb2.ActionEnum.RUN_CNN_LOAD_BIAS]
    return ret


def cnn_init():
    regs = []

    regs.append((0x50001000, 0x00000000))  # AON control
    regs.append((0x50100000, 0x00100008))  # Stop SM
    regs.append((0x50100004, 0x0000040E))  # SRAM control
    regs.append((0x50100008, 0x0000000A))  # Layer count
    regs.append((0x50500000, 0x00100008))  # Stop SM
    regs.append((0x50500004, 0x0000040E))  # SRAM control
    regs.append((0x50500008, 0x0000000A))  # Layer count
    regs.append((0x50900000, 0x00100008))  # Stop SM
    regs.append((0x50900004, 0x0000040E))  # SRAM control
    regs.append((0x50900008, 0x0000000A))  # Layer count
    regs.append((0x50D00000, 0x00100008))  # Stop SM
    regs.append((0x50D00004, 0x0000040E))  # SRAM control
    regs.append((0x50D00008, 0x0000000A))  # Layer count

    # regs = [main_pb2.ActionEnum.RUN_CNN_INIT]
    return regs


def cnn_configure():
    regs = []

    # Layer 0 quadrant 0
    regs.append((0x50100010, 0x00010021))  # Rows
    regs.append((0x50100090, 0x00010021))  # Columns
    regs.append((0x50100310, 0x00001000))  # SRAM write ptr
    regs.append((0x50100410, 0x00002000))  # Write ptr mask offs
    regs.append((0x50100590, 0x00002B20))  # Layer control
    regs.append((0x50100A10, 0x0001F800))  # Layer control 2
    regs.append((0x50100610, 0x000001F8))  # Mask offset and count
    regs.append((0x50100690, 0x0000001F))  # TRAM ptr max
    regs.append((0x50100710, 0x00070007))  # Mask and processor enables

    # Layer 0 quadrant 1
    regs.append((0x50500010, 0x00010021))  # Rows
    regs.append((0x50500090, 0x00010021))  # Columns
    regs.append((0x50500310, 0x00001000))  # SRAM write ptr
    regs.append((0x50500410, 0x00002000))  # Write ptr mask offs
    regs.append((0x50500590, 0x00000B20))  # Layer control
    regs.append((0x50500A10, 0x0001F800))  # Layer control 2
    regs.append((0x50500610, 0x000001F8))  # Mask offset and count
    regs.append((0x50500690, 0x0000001F))  # TRAM ptr max
    regs.append((0x50500790, 0x00001080))  # Post processing register

    # Layer 0 quadrant 2
    regs.append((0x50900010, 0x00010021))  # Rows
    regs.append((0x50900090, 0x00010021))  # Columns
    regs.append((0x50900310, 0x00001000))  # SRAM write ptr
    regs.append((0x50900410, 0x00002000))  # Write ptr mask offs
    regs.append((0x50900590, 0x00000B20))  # Layer control
    regs.append((0x50900A10, 0x0001F800))  # Layer control 2
    regs.append((0x50900610, 0x000001F8))  # Mask offset and count
    regs.append((0x50900690, 0x0000001F))  # TRAM ptr max

    # Layer 0 quadrant 3
    regs.append((0x50D00010, 0x00010021))  # Rows
    regs.append((0x50D00090, 0x00010021))  # Columns
    regs.append((0x50D00310, 0x00001000))  # SRAM write ptr
    regs.append((0x50D00410, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D00590, 0x00000B20))  # Layer control
    regs.append((0x50D00A10, 0x0001F800))  # Layer control 2
    regs.append((0x50D00610, 0x000001F8))  # Mask offset and count
    regs.append((0x50D00690, 0x0000001F))  # TRAM ptr max

    # Layer 1 quadrant 0
    regs.append((0x50100014, 0x0000001F))  # Rows
    regs.append((0x50100094, 0x0000001F))  # Columns
    regs.append((0x50100394, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50100414, 0x00002000))  # Write ptr mask offs
    regs.append((0x50100514, 0x00001000))  # SRAM read ptr
    regs.append((0x50100594, 0x0000EB20))  # Layer control
    regs.append((0x50100A14, 0x0000F800))  # Layer control 2
    regs.append((0x50100614, 0x120012F8))  # Mask offset and count
    regs.append((0x50100114, 0x00000100))  # 1D
    regs.append((0x50100794, 0x00022000))  # Post processing register
    regs.append((0x50100714, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 1 quadrant 1
    regs.append((0x50500014, 0x0000001F))  # Rows
    regs.append((0x50500094, 0x0000001F))  # Columns
    regs.append((0x50500394, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50500414, 0x00002000))  # Write ptr mask offs
    regs.append((0x50500514, 0x00001000))  # SRAM read ptr
    regs.append((0x50500594, 0x00000B20))  # Layer control
    regs.append((0x50500A14, 0x0000F800))  # Layer control 2
    regs.append((0x50500614, 0x120012F8))  # Mask offset and count
    regs.append((0x50500114, 0x00000100))  # 1D
    regs.append((0x50500794, 0x000230C0))  # Post processing register
    regs.append((0x50500714, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 1 quadrant 2
    regs.append((0x50900014, 0x0000001F))  # Rows
    regs.append((0x50900094, 0x0000001F))  # Columns
    regs.append((0x50900394, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50900414, 0x00002000))  # Write ptr mask offs
    regs.append((0x50900514, 0x00001000))  # SRAM read ptr
    regs.append((0x50900594, 0x00000B20))  # Layer control
    regs.append((0x50900A14, 0x0000F800))  # Layer control 2
    regs.append((0x50900614, 0x120012F8))  # Mask offset and count
    regs.append((0x50900114, 0x00000100))  # 1D
    regs.append((0x50900794, 0x00022000))  # Post processing register
    regs.append((0x50900714, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 1 quadrant 3
    regs.append((0x50D00014, 0x0000001F))  # Rows
    regs.append((0x50D00094, 0x0000001F))  # Columns
    regs.append((0x50D00394, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50D00414, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D00514, 0x00001000))  # SRAM read ptr
    regs.append((0x50D00594, 0x00000B20))  # Layer control
    regs.append((0x50D00A14, 0x0000F800))  # Layer control 2
    regs.append((0x50D00614, 0x120012F8))  # Mask offset and count
    regs.append((0x50D00114, 0x00000100))  # 1D
    regs.append((0x50D00794, 0x00022000))  # Post processing register
    regs.append((0x50D00714, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 2 quadrant 0
    regs.append((0x50100018, 0x00010021))  # Rows
    regs.append((0x50100098, 0x00010021))  # Columns
    regs.append((0x50100318, 0x00001000))  # SRAM write ptr
    regs.append((0x50100418, 0x00002000))  # Write ptr mask offs
    regs.append((0x50100598, 0x00006B20))  # Layer control
    regs.append((0x50100A18, 0x0001F800))  # Layer control 2
    regs.append((0x50100618, 0x02200418))  # Mask offset and count
    regs.append((0x50100698, 0x0000001F))  # TRAM ptr max
    regs.append((0x50100798, 0x00022000))  # Post processing register
    regs.append((0x50100718, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 2 quadrant 1
    regs.append((0x50500018, 0x00010021))  # Rows
    regs.append((0x50500098, 0x00010021))  # Columns
    regs.append((0x50500318, 0x00001000))  # SRAM write ptr
    regs.append((0x50500418, 0x00002000))  # Write ptr mask offs
    regs.append((0x50500598, 0x00000B20))  # Layer control
    regs.append((0x50500A18, 0x0001F800))  # Layer control 2
    regs.append((0x50500618, 0x02200418))  # Mask offset and count
    regs.append((0x50500698, 0x0000001F))  # TRAM ptr max
    regs.append((0x50500798, 0x00022000))  # Post processing register
    regs.append((0x50500718, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 2 quadrant 2
    regs.append((0x50900018, 0x00010021))  # Rows
    regs.append((0x50900098, 0x00010021))  # Columns
    regs.append((0x50900318, 0x00001000))  # SRAM write ptr
    regs.append((0x50900418, 0x00002000))  # Write ptr mask offs
    regs.append((0x50900598, 0x00000B20))  # Layer control
    regs.append((0x50900A18, 0x0001F800))  # Layer control 2
    regs.append((0x50900618, 0x02200418))  # Mask offset and count
    regs.append((0x50900698, 0x0000001F))  # TRAM ptr max
    regs.append((0x50900798, 0x00023080))  # Post processing register

    # Layer 2 quadrant 3
    regs.append((0x50D00018, 0x00010021))  # Rows
    regs.append((0x50D00098, 0x00010021))  # Columns
    regs.append((0x50D00318, 0x00001000))  # SRAM write ptr
    regs.append((0x50D00418, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D00598, 0x00000B20))  # Layer control
    regs.append((0x50D00A18, 0x0001F800))  # Layer control 2
    regs.append((0x50D00618, 0x02200418))  # Mask offset and count
    regs.append((0x50D00698, 0x0000001F))  # TRAM ptr max
    regs.append((0x50D00798, 0x00022000))  # Post processing register

    # Layer 3 quadrant 0
    regs.append((0x5010001C, 0x00010021))  # Rows
    regs.append((0x5010009C, 0x00010021))  # Columns
    regs.append((0x5010019C, 0x00000001))  # Pooling rows
    regs.append((0x5010021C, 0x00000001))  # Pooling columns
    regs.append((0x5010029C, 0x00000001))  # Stride
    regs.append((0x5010031C, 0x00010000))  # SRAM write ptr
    regs.append((0x5010041C, 0x00002000))  # Write ptr mask offs
    regs.append((0x5010051C, 0x00001000))  # SRAM read ptr
    regs.append((0x5010059C, 0x0000EBA0))  # Layer control
    regs.append((0x50100A1C, 0x0000F800))  # Layer control 2
    regs.append((0x5010061C, 0x04200518))  # Mask offset and count
    regs.append((0x5010069C, 0x0000000F))  # TRAM ptr max
    regs.append((0x5010079C, 0x00026000))  # Post processing register
    regs.append((0x5010071C, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 3 quadrant 1
    regs.append((0x5050001C, 0x00010021))  # Rows
    regs.append((0x5050009C, 0x00010021))  # Columns
    regs.append((0x5050019C, 0x00000001))  # Pooling rows
    regs.append((0x5050021C, 0x00000001))  # Pooling columns
    regs.append((0x5050029C, 0x00000001))  # Stride
    regs.append((0x5050031C, 0x00010000))  # SRAM write ptr
    regs.append((0x5050041C, 0x00002000))  # Write ptr mask offs
    regs.append((0x5050051C, 0x00001000))  # SRAM read ptr
    regs.append((0x5050059C, 0x00000BA0))  # Layer control
    regs.append((0x50500A1C, 0x0000F800))  # Layer control 2
    regs.append((0x5050061C, 0x04200518))  # Mask offset and count
    regs.append((0x5050069C, 0x0000000F))  # TRAM ptr max
    regs.append((0x5050079C, 0x00026000))  # Post processing register
    regs.append((0x5050071C, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 3 quadrant 2
    regs.append((0x5090001C, 0x00010021))  # Rows
    regs.append((0x5090009C, 0x00010021))  # Columns
    regs.append((0x5090019C, 0x00000001))  # Pooling rows
    regs.append((0x5090021C, 0x00000001))  # Pooling columns
    regs.append((0x5090029C, 0x00000001))  # Stride
    regs.append((0x5090031C, 0x00010000))  # SRAM write ptr
    regs.append((0x5090041C, 0x00002000))  # Write ptr mask offs
    regs.append((0x5090051C, 0x00001000))  # SRAM read ptr
    regs.append((0x5090059C, 0x00000BA0))  # Layer control
    regs.append((0x50900A1C, 0x0000F800))  # Layer control 2
    regs.append((0x5090061C, 0x04200518))  # Mask offset and count
    regs.append((0x5090069C, 0x0000000F))  # TRAM ptr max
    regs.append((0x5090079C, 0x000270C0))  # Post processing register
    regs.append((0x5090071C, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 3 quadrant 3
    regs.append((0x50D0001C, 0x00010021))  # Rows
    regs.append((0x50D0009C, 0x00010021))  # Columns
    regs.append((0x50D0019C, 0x00000001))  # Pooling rows
    regs.append((0x50D0021C, 0x00000001))  # Pooling columns
    regs.append((0x50D0029C, 0x00000001))  # Stride
    regs.append((0x50D0031C, 0x00010000))  # SRAM write ptr
    regs.append((0x50D0041C, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D0051C, 0x00001000))  # SRAM read ptr
    regs.append((0x50D0059C, 0x00000BA0))  # Layer control
    regs.append((0x50D00A1C, 0x0000F800))  # Layer control 2
    regs.append((0x50D0061C, 0x04200518))  # Mask offset and count
    regs.append((0x50D0069C, 0x0000000F))  # TRAM ptr max
    regs.append((0x50D0079C, 0x00026000))  # Post processing register
    regs.append((0x50D0071C, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 4 quadrant 0
    regs.append((0x50100020, 0x0000000F))  # Rows
    regs.append((0x501000A0, 0x0000000F))  # Columns
    regs.append((0x50100320, 0x00001000))  # SRAM write ptr
    regs.append((0x501003A0, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50100420, 0x00002000))  # Write ptr mask offs
    regs.append((0x501005A0, 0x0000CB20))  # Layer control
    regs.append((0x50100A20, 0x0001F800))  # Layer control 2
    regs.append((0x50100620, 0x000001F8))  # Mask offset and count
    regs.append((0x50100120, 0x00000100))  # 1D

    # Layer 4 quadrant 1
    regs.append((0x50500020, 0x0000000F))  # Rows
    regs.append((0x505000A0, 0x0000000F))  # Columns
    regs.append((0x50500320, 0x00001000))  # SRAM write ptr
    regs.append((0x505003A0, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50500420, 0x00002000))  # Write ptr mask offs
    regs.append((0x505005A0, 0x00000B20))  # Layer control
    regs.append((0x50500A20, 0x0001F800))  # Layer control 2
    regs.append((0x50500620, 0x000001F8))  # Mask offset and count
    regs.append((0x50500120, 0x00000100))  # 1D

    # Layer 4 quadrant 2
    regs.append((0x50900020, 0x0000000F))  # Rows
    regs.append((0x509000A0, 0x0000000F))  # Columns
    regs.append((0x50900320, 0x00001000))  # SRAM write ptr
    regs.append((0x509003A0, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50900420, 0x00002000))  # Write ptr mask offs
    regs.append((0x509005A0, 0x00000B20))  # Layer control
    regs.append((0x50900A20, 0x0001F800))  # Layer control 2
    regs.append((0x50900620, 0x000001F8))  # Mask offset and count
    regs.append((0x50900120, 0x00000100))  # 1D
    regs.append((0x50900720, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 4 quadrant 3
    regs.append((0x50D00020, 0x0000000F))  # Rows
    regs.append((0x50D000A0, 0x0000000F))  # Columns
    regs.append((0x50D00320, 0x00001000))  # SRAM write ptr
    regs.append((0x50D003A0, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50D00420, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D005A0, 0x00000B20))  # Layer control
    regs.append((0x50D00A20, 0x0001F800))  # Layer control 2
    regs.append((0x50D00620, 0x000001F8))  # Mask offset and count
    regs.append((0x50D00120, 0x00000100))  # 1D
    regs.append((0x50D007A0, 0x00001080))  # Post processing register
    regs.append((0x50D00720, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 5 quadrant 0
    regs.append((0x50100024, 0x00010011))  # Rows
    regs.append((0x501000A4, 0x00010011))  # Columns
    regs.append((0x501001A4, 0x00000001))  # Pooling rows
    regs.append((0x50100224, 0x00000001))  # Pooling columns
    regs.append((0x501002A4, 0x00000001))  # Stride
    regs.append((0x50100424, 0x00002000))  # Write ptr mask offs
    regs.append((0x501004A4, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50100524, 0x00001000))  # SRAM read ptr
    regs.append((0x501005A4, 0x0000EBA0))  # Layer control
    regs.append((0x50100A24, 0x0001F810))  # Layer control 2
    regs.append((0x50100624, 0x05200918))  # Mask offset and count
    regs.append((0x501006A4, 0x00000007))  # TRAM ptr max
    regs.append((0x501007A4, 0x00026000))  # Post processing register
    regs.append((0x50100724, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 5 quadrant 1
    regs.append((0x50500024, 0x00010011))  # Rows
    regs.append((0x505000A4, 0x00010011))  # Columns
    regs.append((0x505001A4, 0x00000001))  # Pooling rows
    regs.append((0x50500224, 0x00000001))  # Pooling columns
    regs.append((0x505002A4, 0x00000001))  # Stride
    regs.append((0x50500424, 0x00002000))  # Write ptr mask offs
    regs.append((0x505004A4, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50500524, 0x00001000))  # SRAM read ptr
    regs.append((0x505005A4, 0x00000BA0))  # Layer control
    regs.append((0x50500A24, 0x0001F810))  # Layer control 2
    regs.append((0x50500624, 0x05200918))  # Mask offset and count
    regs.append((0x505006A4, 0x00000007))  # TRAM ptr max
    regs.append((0x505007A4, 0x00027000))  # Post processing register
    regs.append((0x50500724, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 5 quadrant 2
    regs.append((0x50900024, 0x00010011))  # Rows
    regs.append((0x509000A4, 0x00010011))  # Columns
    regs.append((0x509001A4, 0x00000001))  # Pooling rows
    regs.append((0x50900224, 0x00000001))  # Pooling columns
    regs.append((0x509002A4, 0x00000001))  # Stride
    regs.append((0x50900424, 0x00002000))  # Write ptr mask offs
    regs.append((0x509004A4, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50900524, 0x00001000))  # SRAM read ptr
    regs.append((0x509005A4, 0x00000BA0))  # Layer control
    regs.append((0x50900A24, 0x0001F810))  # Layer control 2
    regs.append((0x50900624, 0x05200918))  # Mask offset and count
    regs.append((0x509006A4, 0x00000007))  # TRAM ptr max
    regs.append((0x509007A4, 0x00026000))  # Post processing register
    regs.append((0x50900724, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 5 quadrant 3
    regs.append((0x50D00024, 0x00010011))  # Rows
    regs.append((0x50D000A4, 0x00010011))  # Columns
    regs.append((0x50D001A4, 0x00000001))  # Pooling rows
    regs.append((0x50D00224, 0x00000001))  # Pooling columns
    regs.append((0x50D002A4, 0x00000001))  # Stride
    regs.append((0x50D00424, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D004A4, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50D00524, 0x00001000))  # SRAM read ptr
    regs.append((0x50D005A4, 0x00000BA0))  # Layer control
    regs.append((0x50D00A24, 0x0001F810))  # Layer control 2
    regs.append((0x50D00624, 0x05200918))  # Mask offset and count
    regs.append((0x50D006A4, 0x00000007))  # TRAM ptr max
    regs.append((0x50D007A4, 0x00026000))  # Post processing register
    regs.append((0x50D00724, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 6 quadrant 0
    regs.append((0x50100028, 0x00000007))  # Rows
    regs.append((0x501000A8, 0x00000007))  # Columns
    regs.append((0x50100328, 0x00001000))  # SRAM write ptr
    regs.append((0x501003A8, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50100428, 0x00002000))  # Write ptr mask offs
    regs.append((0x501004A8, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x501005A8, 0x0000EB20))  # Layer control
    regs.append((0x50100A28, 0x0001F811))  # Layer control 2
    regs.append((0x50100628, 0x52205A18))  # Mask offset and count
    regs.append((0x50100128, 0x00000100))  # 1D
    regs.append((0x501007A8, 0x00022000))  # Post processing register
    regs.append((0x50100728, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 6 quadrant 1
    regs.append((0x50500028, 0x00000007))  # Rows
    regs.append((0x505000A8, 0x00000007))  # Columns
    regs.append((0x50500328, 0x00001000))  # SRAM write ptr
    regs.append((0x505003A8, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50500428, 0x00002000))  # Write ptr mask offs
    regs.append((0x505004A8, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x505005A8, 0x00000B20))  # Layer control
    regs.append((0x50500A28, 0x0001F811))  # Layer control 2
    regs.append((0x50500628, 0x52205A18))  # Mask offset and count
    regs.append((0x50500128, 0x00000100))  # 1D
    regs.append((0x505007A8, 0x00022000))  # Post processing register
    regs.append((0x50500728, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 6 quadrant 2
    regs.append((0x50900028, 0x00000007))  # Rows
    regs.append((0x509000A8, 0x00000007))  # Columns
    regs.append((0x50900328, 0x00001000))  # SRAM write ptr
    regs.append((0x509003A8, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50900428, 0x00002000))  # Write ptr mask offs
    regs.append((0x509004A8, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x509005A8, 0x00000B20))  # Layer control
    regs.append((0x50900A28, 0x0001F811))  # Layer control 2
    regs.append((0x50900628, 0x52205A18))  # Mask offset and count
    regs.append((0x50900128, 0x00000100))  # 1D
    regs.append((0x509007A8, 0x00023000))  # Post processing register
    regs.append((0x50900728, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 6 quadrant 3
    regs.append((0x50D00028, 0x00000007))  # Rows
    regs.append((0x50D000A8, 0x00000007))  # Columns
    regs.append((0x50D00328, 0x00001000))  # SRAM write ptr
    regs.append((0x50D003A8, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50D00428, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D004A8, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50D005A8, 0x00000B20))  # Layer control
    regs.append((0x50D00A28, 0x0001F811))  # Layer control 2
    regs.append((0x50D00628, 0x52205A18))  # Mask offset and count
    regs.append((0x50D00128, 0x00000100))  # 1D
    regs.append((0x50D007A8, 0x00022000))  # Post processing register
    regs.append((0x50D00728, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 7 quadrant 0
    regs.append((0x5010002C, 0x00010009))  # Rows
    regs.append((0x501000AC, 0x00010009))  # Columns
    regs.append((0x501001AC, 0x00000001))  # Pooling rows
    regs.append((0x5010022C, 0x00000001))  # Pooling columns
    regs.append((0x501002AC, 0x00000001))  # Stride
    regs.append((0x5010042C, 0x00002000))  # Write ptr mask offs
    regs.append((0x5010052C, 0x00001000))  # SRAM read ptr
    regs.append((0x501005AC, 0x0000EBA0))  # Layer control
    regs.append((0x50100A2C, 0x0001F801))  # Layer control 2
    regs.append((0x5010062C, 0x0A200E18))  # Mask offset and count
    regs.append((0x501006AC, 0x00000003))  # TRAM ptr max
    regs.append((0x501007AC, 0x0002708A))  # Post processing register
    regs.append((0x5010072C, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 7 quadrant 1
    regs.append((0x5050002C, 0x00010009))  # Rows
    regs.append((0x505000AC, 0x00010009))  # Columns
    regs.append((0x505001AC, 0x00000001))  # Pooling rows
    regs.append((0x5050022C, 0x00000001))  # Pooling columns
    regs.append((0x505002AC, 0x00000001))  # Stride
    regs.append((0x5050042C, 0x00002000))  # Write ptr mask offs
    regs.append((0x5050052C, 0x00001000))  # SRAM read ptr
    regs.append((0x505005AC, 0x00000BA0))  # Layer control
    regs.append((0x50500A2C, 0x0001F801))  # Layer control 2
    regs.append((0x5050062C, 0x0A200E18))  # Mask offset and count
    regs.append((0x505006AC, 0x00000003))  # TRAM ptr max
    regs.append((0x505007AC, 0x00026000))  # Post processing register
    regs.append((0x5050072C, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 7 quadrant 2
    regs.append((0x5090002C, 0x00010009))  # Rows
    regs.append((0x509000AC, 0x00010009))  # Columns
    regs.append((0x509001AC, 0x00000001))  # Pooling rows
    regs.append((0x5090022C, 0x00000001))  # Pooling columns
    regs.append((0x509002AC, 0x00000001))  # Stride
    regs.append((0x5090042C, 0x00002000))  # Write ptr mask offs
    regs.append((0x5090052C, 0x00001000))  # SRAM read ptr
    regs.append((0x509005AC, 0x00000BA0))  # Layer control
    regs.append((0x50900A2C, 0x0001F801))  # Layer control 2
    regs.append((0x5090062C, 0x0A200E18))  # Mask offset and count
    regs.append((0x509006AC, 0x00000003))  # TRAM ptr max
    regs.append((0x509007AC, 0x00026000))  # Post processing register
    regs.append((0x5090072C, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 7 quadrant 3
    regs.append((0x50D0002C, 0x00010009))  # Rows
    regs.append((0x50D000AC, 0x00010009))  # Columns
    regs.append((0x50D001AC, 0x00000001))  # Pooling rows
    regs.append((0x50D0022C, 0x00000001))  # Pooling columns
    regs.append((0x50D002AC, 0x00000001))  # Stride
    regs.append((0x50D0042C, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D0052C, 0x00001000))  # SRAM read ptr
    regs.append((0x50D005AC, 0x00000BA0))  # Layer control
    regs.append((0x50D00A2C, 0x0001F801))  # Layer control 2
    regs.append((0x50D0062C, 0x0A200E18))  # Mask offset and count
    regs.append((0x50D006AC, 0x00000003))  # TRAM ptr max
    regs.append((0x50D007AC, 0x00026000))  # Post processing register
    regs.append((0x50D0072C, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 8 quadrant 0
    regs.append((0x50100030, 0x00010005))  # Rows
    regs.append((0x501000B0, 0x00010005))  # Columns
    regs.append((0x50100330, 0x00001000))  # SRAM write ptr
    regs.append((0x50100430, 0x00002000))  # Write ptr mask offs
    regs.append((0x501004B0, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x501005B0, 0x0000EB20))  # Layer control
    regs.append((0x50100A30, 0x0001F810))  # Layer control 2
    regs.append((0x50100630, 0x0E201218))  # Mask offset and count
    regs.append((0x501006B0, 0x00000003))  # TRAM ptr max
    regs.append((0x501007B0, 0x00024000))  # Post processing register
    regs.append((0x50100730, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 8 quadrant 1
    regs.append((0x50500030, 0x00010005))  # Rows
    regs.append((0x505000B0, 0x00010005))  # Columns
    regs.append((0x50500330, 0x00001000))  # SRAM write ptr
    regs.append((0x50500430, 0x00002000))  # Write ptr mask offs
    regs.append((0x505004B0, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x505005B0, 0x00000B20))  # Layer control
    regs.append((0x50500A30, 0x0001F810))  # Layer control 2
    regs.append((0x50500630, 0x0E201218))  # Mask offset and count
    regs.append((0x505006B0, 0x00000003))  # TRAM ptr max
    regs.append((0x505007B0, 0x00024000))  # Post processing register
    regs.append((0x50500730, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 8 quadrant 2
    regs.append((0x50900030, 0x00010005))  # Rows
    regs.append((0x509000B0, 0x00010005))  # Columns
    regs.append((0x50900330, 0x00001000))  # SRAM write ptr
    regs.append((0x50900430, 0x00002000))  # Write ptr mask offs
    regs.append((0x509004B0, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x509005B0, 0x00000B20))  # Layer control
    regs.append((0x50900A30, 0x0001F810))  # Layer control 2
    regs.append((0x50900630, 0x0E201218))  # Mask offset and count
    regs.append((0x509006B0, 0x00000003))  # TRAM ptr max
    regs.append((0x509007B0, 0x00024000))  # Post processing register
    regs.append((0x50900730, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 8 quadrant 3
    regs.append((0x50D00030, 0x00010005))  # Rows
    regs.append((0x50D000B0, 0x00010005))  # Columns
    regs.append((0x50D00330, 0x00001000))  # SRAM write ptr
    regs.append((0x50D00430, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D004B0, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50D005B0, 0x00000B20))  # Layer control
    regs.append((0x50D00A30, 0x0001F810))  # Layer control 2
    regs.append((0x50D00630, 0x0E201218))  # Mask offset and count
    regs.append((0x50D006B0, 0x00000003))  # TRAM ptr max
    regs.append((0x50D007B0, 0x00025000))  # Post processing register
    regs.append((0x50D00730, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 9 quadrant 0
    regs.append((0x50100034, 0x00000003))  # Rows
    regs.append((0x501000B4, 0x00000003))  # Columns
    regs.append((0x501001B4, 0x00000001))  # Pooling rows
    regs.append((0x50100234, 0x00000001))  # Pooling columns
    regs.append((0x501002B4, 0x00000001))  # Stride
    regs.append((0x501003B4, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50100434, 0x00002000))  # Write ptr mask offs
    regs.append((0x501004B4, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50100534, 0x00001000))  # SRAM read ptr
    regs.append((0x501005B4, 0x0000EBA0))  # Layer control
    regs.append((0x50100A34, 0x0001F811))  # Layer control 2
    regs.append((0x50100634, 0xA320AB18))  # Mask offset and count
    regs.append((0x50100134, 0x00000100))  # 1D
    regs.append((0x501007B4, 0x0002300A))  # Post processing register
    regs.append((0x50100734, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 9 quadrant 1
    regs.append((0x50500034, 0x00000003))  # Rows
    regs.append((0x505000B4, 0x00000003))  # Columns
    regs.append((0x505001B4, 0x00000001))  # Pooling rows
    regs.append((0x50500234, 0x00000001))  # Pooling columns
    regs.append((0x505002B4, 0x00000001))  # Stride
    regs.append((0x505003B4, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50500434, 0x00002000))  # Write ptr mask offs
    regs.append((0x505004B4, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50500534, 0x00001000))  # SRAM read ptr
    regs.append((0x505005B4, 0x00000BA0))  # Layer control
    regs.append((0x50500A34, 0x0001F811))  # Layer control 2
    regs.append((0x50500634, 0xA320AB18))  # Mask offset and count
    regs.append((0x50500134, 0x00000100))  # 1D
    regs.append((0x505007B4, 0x00022000))  # Post processing register
    regs.append((0x50500734, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 9 quadrant 2
    regs.append((0x50900034, 0x00000003))  # Rows
    regs.append((0x509000B4, 0x00000003))  # Columns
    regs.append((0x509001B4, 0x00000001))  # Pooling rows
    regs.append((0x50900234, 0x00000001))  # Pooling columns
    regs.append((0x509002B4, 0x00000001))  # Stride
    regs.append((0x509003B4, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50900434, 0x00002000))  # Write ptr mask offs
    regs.append((0x509004B4, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50900534, 0x00001000))  # SRAM read ptr
    regs.append((0x509005B4, 0x00000BA0))  # Layer control
    regs.append((0x50900A34, 0x0001F811))  # Layer control 2
    regs.append((0x50900634, 0xA320AB18))  # Mask offset and count
    regs.append((0x50900134, 0x00000100))  # 1D
    regs.append((0x509007B4, 0x00022000))  # Post processing register
    regs.append((0x50900734, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 9 quadrant 3
    regs.append((0x50D00034, 0x00000003))  # Rows
    regs.append((0x50D000B4, 0x00000003))  # Columns
    regs.append((0x50D001B4, 0x00000001))  # Pooling rows
    regs.append((0x50D00234, 0x00000001))  # Pooling columns
    regs.append((0x50D002B4, 0x00000001))  # Stride
    regs.append((0x50D003B4, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50D00434, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D004B4, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50D00534, 0x00001000))  # SRAM read ptr
    regs.append((0x50D005B4, 0x00000BA0))  # Layer control
    regs.append((0x50D00A34, 0x0001F811))  # Layer control 2
    regs.append((0x50D00634, 0xA320AB18))  # Mask offset and count
    regs.append((0x50D00134, 0x00000100))  # 1D
    regs.append((0x50D007B4, 0x00022000))  # Post processing register
    regs.append((0x50D00734, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 10 quadrant 0
    regs.append((0x50100338, 0x00001000))  # SRAM write ptr
    regs.append((0x501003B8, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50100438, 0x00002000))  # Write ptr mask offs
    regs.append((0x501005B8, 0x0001E920))  # Layer control
    regs.append((0x50100A38, 0x00004807))  # Layer control 2
    regs.append((0x50100638, 0xAC20AE98))  # Mask offset and count
    regs.append((0x50100138, 0x00000100))  # 1D
    regs.append((0x501007B8, 0x00003000))  # Post processing register
    regs.append((0x50100738, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 10 quadrant 1
    regs.append((0x50500338, 0x00001000))  # SRAM write ptr
    regs.append((0x505003B8, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50500438, 0x00002000))  # Write ptr mask offs
    regs.append((0x505005B8, 0x00010920))  # Layer control
    regs.append((0x50500A38, 0x00004807))  # Layer control 2
    regs.append((0x50500638, 0xAC20AE98))  # Mask offset and count
    regs.append((0x50500138, 0x00000100))  # 1D
    regs.append((0x505007B8, 0x00002000))  # Post processing register
    regs.append((0x50500738, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 10 quadrant 2
    regs.append((0x50900338, 0x00001000))  # SRAM write ptr
    regs.append((0x509003B8, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50900438, 0x00002000))  # Write ptr mask offs
    regs.append((0x509005B8, 0x00010920))  # Layer control
    regs.append((0x50900A38, 0x00004807))  # Layer control 2
    regs.append((0x50900638, 0xAC20AE98))  # Mask offset and count
    regs.append((0x50900138, 0x00000100))  # 1D
    regs.append((0x509007B8, 0x00002000))  # Post processing register
    regs.append((0x50900738, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 10 quadrant 3
    regs.append((0x50D00338, 0x00001000))  # SRAM write ptr
    regs.append((0x50D003B8, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50D00438, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D005B8, 0x00010920))  # Layer control
    regs.append((0x50D00A38, 0x00004807))  # Layer control 2
    regs.append((0x50D00638, 0xAC20AE98))  # Mask offset and count
    regs.append((0x50D00138, 0x00000100))  # 1D
    regs.append((0x50D007B8, 0x00002000))  # Post processing register
    regs.append((0x50D00738, 0xFFFFFFFF))  # Mask and processor enables

    # regs = [main_pb2.ActionEnum.RUN_CNN_CONFIGURE]

    return regs


def cnn_start():
    regs = []
    # cnn_time = 0

    regs.append((0x50100000, 0x00100808))  # Enable quadrant 0
    regs.append((0x50500000, 0x00100809))  # Enable quadrant 1
    regs.append((0x50900000, 0x00100809))  # Enable quadrant 2
    regs.append((0x50D00000, 0x00100809))  # Enable quadrant 3

    # CNN_START)) # Allow capture of processing time
    regs.append((0x50100000, 0x00100009))  # Master enable quadrant 0

    # regs = [main_pb2.ActionEnum.RUN_CNN_START]
    return regs


def cnn_unload():
    """
    # Custom unload for this network, layer 10: 32-bit data, shape: (10, 1, 1)
    addr = (volatile uint32_t *)0x50404000
    *out_buf++ = *addr++
    *out_buf++ = *addr++
    *out_buf++ = *addr++
    *out_buf++ = *addr++
    addr = (volatile uint32_t *)0x5040c000
    *out_buf++ = *addr++
    *out_buf++ = *addr++
    *out_buf++ = *addr++
    *out_buf++ = *addr++
    addr = (volatile uint32_t *)0x50414000
    *out_buf++ = *addr++
    *out_buf++ = *addr++
    """
    return []


def cnn_enable():
    ret = [main_pb2.ActionEnum.RUN_CNN_ENABLE]
    return ret


def cnn_disable():
    ret = [main_pb2.ActionEnum.RUN_CNN_DISABLE]
    return ret
