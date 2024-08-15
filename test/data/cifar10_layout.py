from struct import pack

import netdeployonnx.devices.max78000.cnn_constants as cnn_constants
from netdeployonnx.devices.max78000.core import CNNx16Core


def set_bias(layout):
    from .cifar10_bias import bias_group, bias_order, biases  # noqa

    for layeridx, allowed_quads, len_bias, flatten in bias_order:
        quad = bias_group[layeridx]
        # struct pack for unsigned char is 'B', but preserve whatever
        serialized_bias = b"".join([pack(">b", bias) for bias in biases[layeridx]])
        layout[quad].bias += serialized_bias


def set_kernel(layout):
    from .cifar10_weights import kernels as kernel_list_correct  # noqa

    reverse_maddr = {
        cnn_constants.memory.get(f"CNNx16_{quad}_P{proc}_MRAM"): (quad, proc)
        for quad in range(4)
        for proc in range(16)
    }

    for kernel_entry in kernel_list_correct:
        kernel_addr, kernel_data = kernel_entry["addr"], kernel_entry["data"]
        offs = kernel_addr % 0x4000
        mem_addr = kernel_addr - offs

        if mem_addr in reverse_maddr:
            quad, proc = reverse_maddr[mem_addr]
            layout[quad].processors[proc].kernels[offs] = kernel_data


async def cifar10_layout():  # noqa: C901
    layout = CNNx16Core()
    set_bias(layout)
    set_kernel(layout)
    for quad in range(4):
        for layer in range(11):
            if layer == 0:
                layout[quad, layer].row_pad = 1
                layout[quad, layer].col_pad = 1
                layout[quad, layer].row_count = 33
                layout[quad, layer].col_count = 33

                layout[quad, layer].writeptr = 0x1000
                layout[quad, layer].writeptr_mask_offset = 0x2000

                layout[quad, layer].mask_maxaddr = 0x1F8  # 504 / 8-> 63
                layout[quad, layer].tram_maxaddr = 0x1F  # 31
                layout[quad, layer].expansion_mode_processors = 0x1F8 # LCTRL1

                layout[quad, layer].master = True
                layout[quad, layer].sram_load_source = True
                layout[quad, layer].relu_en = True
                layout[quad, layer].maxpool_en = True

                if quad in [0]:
                    layout[quad, layer].enable_processor = 0x0007
                    layout[quad, layer].enable_mask = 0x0007
                    layout[quad, layer].nonscaled_nonquantized_sum_feed_en = 2
                if quad in [1]:
                    layout[quad, layer].bias_addr = 0x80
            if layer == 1:
                layout[quad, layer].row_count = 31
                layout[quad, layer].col_count = 31

                layout[quad, layer].writeptr_timeslot_offset = 1
                layout[quad, layer].writeptr_mask_offset = 0x2000
                layout[quad, layer].readptr = 0x1000

                layout[quad, layer].master = True
                layout[quad, layer].sram_load_source = True
                layout[quad, layer].relu_en = True
                layout[quad, layer].maxpool_en = True
                if quad in [0]:
                    layout[quad, layer].nonscaled_nonquantized_sum_feed_en = 0xE
                if quad in [0, 1, 2, 3]:
                    layout[quad, layer].shift_by = -1
                if quad == 1:
                    layout[quad, layer].bias_addr = 0xC0

                layout[quad, layer].expansion_mode_processors = 0xF8 # 248 => 31
                layout[quad, layer].mask_maxaddr = 0x12F8 # len:248 => 32
                layout[quad, layer].mask_start = 0x1200 # 1200 => 576

                layout[quad, layer].enable_mask = 0xFFFF
                layout[quad, layer].enable_processor = 0xFFFF

                layout[quad, layer].oned_mask_width = 1
            if layer == 2:
                layout[quad, layer].row_pad = 1
                layout[quad, layer].col_pad = 1
                layout[quad, layer].row_count = 33
                layout[quad, layer].col_count = 33

                layout[quad, layer].writeptr = 0x1000
                layout[quad, layer].writeptr_mask_offset = 0x2000
                layout[quad, layer].tram_maxaddr = 0x1F

                layout[quad, layer].master = True
                layout[quad, layer].sram_load_source = True
                layout[quad, layer].relu_en = True
                layout[quad, layer].maxpool_en = True
                if quad in [0]:
                    layout[quad, layer].nonscaled_nonquantized_sum_feed_en = 0x6

                layout[quad, layer].shift_by = -1
                if quad in [2]:
                    layout[quad, layer].bias_addr = 0x80

                layout[quad, layer].expansion_mode_processors = 0x1F8
                layout[quad, layer].mask_maxaddr = 0x418
                layout[quad, layer].mask_start = 0x0220
                if quad in [0, 1]:
                    layout[quad, layer].enable_mask = 0xFFFF
                    layout[quad, layer].enable_processor = 0xFFFF
            if layer == 3:
                layout[quad, layer].row_pad = 1
                layout[quad, layer].col_pad = 1
                layout[quad, layer].row_count = 33
                layout[quad, layer].col_count = 33

                layout[quad, layer].writeptr = 0x10000
                layout[quad, layer].writeptr_mask_offset = 0x2000
                layout[quad, layer].readptr = 0x1000
                layout[quad, layer].tram_maxaddr = 0xF

                layout[quad, layer].row_pooling = 1
                layout[quad, layer].col_pooling = 1
                layout[quad, layer].stride = True

                layout[quad, layer].master = True
                layout[quad, layer].sram_load_source = True
                layout[quad, layer].pool_en = True
                layout[quad, layer].relu_en = True
                layout[quad, layer].maxpool_en = True
                if quad in [0]:
                    layout[quad, layer].nonscaled_nonquantized_sum_feed_en = 0xE

                layout[quad, layer].shift_by = -3
                if quad in [2]:
                    layout[quad, layer].bias_addr = 0xC0

                layout[quad, layer].expansion_mode_processors = 0xF8
                layout[quad, layer].mask_maxaddr = 0x518
                layout[quad, layer].mask_start = 0x0420

                layout[quad, layer].enable_mask = 0xFFFF
                layout[quad, layer].enable_processor = 0xFFFF
            if layer == 4:
                layout[quad, layer].row_count = 15
                layout[quad, layer].col_count = 15

                layout[quad, layer].writeptr = 0x1000
                layout[quad, layer].writeptr_timeslot_offset = 1
                layout[quad, layer].writeptr_mask_offset = 0x2000

                layout[quad, layer].master = True
                layout[quad, layer].sram_load_source = True
                layout[quad, layer].relu_en = True
                layout[quad, layer].maxpool_en = True
                if quad in [0]:
                    layout[quad, layer].nonscaled_nonquantized_sum_feed_en = 0xC

                layout[quad, layer].expansion_mode_processors = 0x1F8
                layout[quad, layer].mask_maxaddr = 0x1F8
                layout[quad, layer].mask_start = 0

                if quad in [2, 3]:
                    layout[quad, layer].enable_mask = 0xFFFF
                    layout[quad, layer].enable_processor = 0xFFFF
                if quad == 3:
                    layout[quad, layer].bias_addr = 0x80
                layout[quad, layer].oned_mask_width = 1
            if layer == 5:
                layout[quad, layer].row_count = 17
                layout[quad, layer].col_count = 17
                layout[quad, layer].row_pad = 1
                layout[quad, layer].col_pad = 1

                layout[quad, layer].row_pooling = 1
                layout[quad, layer].col_pooling = 1
                layout[quad, layer].stride = True

                layout[quad, layer].writeptr_mask_offset = 0x2000
                layout[quad, layer].writeptr_multipass_offset = 1
                layout[quad, layer].readptr = 0x1000
                layout[quad, layer].tram_maxaddr = 0x7

                layout[quad, layer].mask_maxaddr = 0x918
                layout[quad, layer].mask_start = 0x520

                layout[quad, layer].master = True
                layout[quad, layer].sram_load_source = True
                layout[quad, layer].pool_en = True
                layout[quad, layer].relu_en = True
                layout[quad, layer].maxpool_en = True
                if quad in [0]:
                    layout[quad, layer].nonscaled_nonquantized_sum_feed_en = 0xE

                layout[quad, layer].shift_by = -3
                if quad in [1]:
                    layout[quad, layer].bias_en = True
                    layout[quad, layer].bias_addr = 0x0

                layout[quad, layer].expansion_mode_processors = 0x1F8
                layout[quad, layer].expansion_mode_writeptr = 1

                if quad in [0, 1, 2, 3]:
                    layout[quad, layer].enable_mask = 0xFFFF
                    layout[quad, layer].enable_processor = 0xFFFF
            if layer == 6:
                layout[quad, layer].row_count = 7
                layout[quad, layer].col_count = 7

                layout[quad, layer].writeptr = 0x1000
                layout[quad, layer].writeptr_timeslot_offset = 1
                layout[quad, layer].writeptr_mask_offset = 0x2000
                layout[quad, layer].writeptr_multipass_offset = 1

                layout[quad, layer].mask_maxaddr = 0x5A18
                layout[quad, layer].mask_start = 0x5220

                layout[quad, layer].master = True
                layout[quad, layer].sram_load_source = True
                layout[quad, layer].relu_en = True
                layout[quad, layer].maxpool_en = True
                if quad in [0]:
                    layout[quad, layer].nonscaled_nonquantized_sum_feed_en = 0xE

                layout[quad, layer].shift_by = -1
                if quad in [2]:
                    layout[quad, layer].bias_en = True
                    layout[quad, layer].bias_addr = 0x00

                layout[quad, layer].expansion_mode_processors = 0x1F8
                layout[quad, layer].expansion_mode_writeptr = 1
                layout[quad, layer].expansion_mode_inputchan = 1

                layout[quad, layer].enable_mask = 0xFFFF
                layout[quad, layer].enable_processor = 0xFFFF
                layout[quad, layer].oned_mask_width = 1
            if layer == 7:
                layout[quad, layer].row_pad = 1
                layout[quad, layer].col_pad = 1
                layout[quad, layer].row_count = 9
                layout[quad, layer].col_count = 9
                layout[quad, layer].row_pooling = 1
                layout[quad, layer].col_pooling = 1
                layout[quad, layer].stride = True

                layout[quad, layer].writeptr_mask_offset = 0x2000
                layout[quad, layer].readptr = 0x1000
                layout[quad, layer].tram_maxaddr = 0x3

                layout[quad, layer].mask_maxaddr = 0xE18
                layout[quad, layer].mask_start = 0xA20

                layout[quad, layer].master = True
                layout[quad, layer].sram_load_source = True
                layout[quad, layer].pool_en = True
                layout[quad, layer].relu_en = True
                layout[quad, layer].maxpool_en = True
                if quad in [0]:
                    layout[quad, layer].nonscaled_nonquantized_sum_feed_en = 0xE

                layout[quad, layer].shift_by = -3
                if quad in [0]:
                    layout[quad, layer].bias_addr = 0x8A

                layout[quad, layer].expansion_mode_processors = 0x1F8
                layout[quad, layer].expansion_mode_inputchan = 1

                layout[quad, layer].enable_mask = 0xFFFF
                layout[quad, layer].enable_processor = 0xFFFF
            if layer == 8:
                layout[quad, layer].row_pad = 1
                layout[quad, layer].col_pad = 1
                layout[quad, layer].row_count = 5
                layout[quad, layer].col_count = 5

                layout[quad, layer].writeptr = 0x1000
                layout[quad, layer].writeptr_mask_offset = 0x2000
                layout[quad, layer].writeptr_multipass_offset = 1
                layout[quad, layer].tram_maxaddr = 0x3

                layout[quad, layer].mask_maxaddr = 0x1218
                layout[quad, layer].mask_start = 0xE20

                layout[quad, layer].master = True
                layout[quad, layer].sram_load_source = True
                layout[quad, layer].relu_en = True
                layout[quad, layer].maxpool_en = True
                if quad in [0]:
                    layout[quad, layer].nonscaled_nonquantized_sum_feed_en = 0xE

                layout[quad, layer].shift_by = -2
                if quad in [3]:
                    layout[quad, layer].bias_en = True
                    layout[quad, layer].bias_addr = 0x00

                layout[quad, layer].expansion_mode_processors = 0x1F8
                layout[quad, layer].expansion_mode_writeptr = 1

                layout[quad, layer].enable_mask = 0xFFFF
                layout[quad, layer].enable_processor = 0xFFFF
            if layer == 9:
                layout[quad, layer].row_count = 3
                layout[quad, layer].col_count = 3
                layout[quad, layer].row_pooling = 1
                layout[quad, layer].col_pooling = 1
                layout[quad, layer].stride = True

                layout[quad, layer].writeptr_timeslot_offset = 1
                layout[quad, layer].writeptr_mask_offset = 0x2000
                layout[quad, layer].writeptr_multipass_offset = 1

                layout[quad, layer].readptr = 0x1000

                layout[quad, layer].mask_maxaddr = 0xAB18
                layout[quad, layer].mask_start = 0xA320

                layout[quad, layer].master = True
                layout[quad, layer].sram_load_source = True
                layout[quad, layer].pool_en = True
                layout[quad, layer].relu_en = True
                layout[quad, layer].maxpool_en = True
                if quad in [0]:
                    layout[quad, layer].nonscaled_nonquantized_sum_feed_en = 0xE

                layout[quad, layer].shift_by = -1
                if quad in [0]:
                    layout[quad, layer].bias_addr = 0x0A

                layout[quad, layer].expansion_mode_processors = 0x1F8
                layout[quad, layer].expansion_mode_writeptr = 1
                layout[quad, layer].expansion_mode_inputchan = 1

                layout[quad, layer].enable_mask = 0xFFFF
                layout[quad, layer].enable_processor = 0xFFFF
                layout[quad, layer].oned_mask_width = 1
            if layer == 10:
                layout[quad, layer].shift_by = 1

                layout[quad, layer].writeptr = 0x1000
                layout[quad, layer].writeptr_timeslot_offset = 1
                layout[quad, layer].writeptr_mask_offset = 0x2000

                layout[quad, layer].mask_maxaddr = 0xAE98
                layout[quad, layer].mask_start = 0xAC20

                layout[quad, layer].big_data_write = True
                layout[quad, layer].sram_load_source = True
                layout[quad, layer].maxpool_en = True
                layout[quad, layer].master = True

                if quad in [0]:
                    layout[quad, layer].nonscaled_nonquantized_sum_feed_en = 0xE

                if quad in [0]:
                    layout[quad, layer].bias_en = True
                    layout[quad, layer].bias_addr = 0x00

                layout[quad, layer].expansion_mode_processors = 0x48
                layout[quad, layer].expansion_mode_inputchan = 7

                layout[quad, layer].enable_mask = 0xFFFF
                layout[quad, layer].enable_processor = 0xFFFF
                layout[quad, layer].oned_mask_width = 1

    return layout
