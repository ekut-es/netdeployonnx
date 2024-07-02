import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from netdeployonnx.devices.max78000 import cnn_constants
from netdeployonnx.devices.max78000.core import CNNx16_Layer, CNNx16Core
from netdeployonnx.devices.max78000.synthesizer.grid import Node
from netdeployonnx.devices.max78000.synthesizer.memory import MemoryQuadrant, layout_mem

from .test_max78000_graph_synthesis import cifar10_kernels
from .test_max78000_layer_by_node import cifar10_refnet


@pytest.fixture(scope="function")
def core():
    return CNNx16Core()


@pytest.fixture(scope="function")
def layer(core) -> CNNx16_Layer:
    return core[0, 0]


def make_testnode(
    nodetype: str, inputs: list[str], outputs: list[str], **kwargs
) -> Node:
    node = MagicMock()
    node.op_type = nodetype
    node.input = [val for name, val in kwargs.items() if name in inputs]
    node.output = [val for name, val in kwargs.items() if name in outputs]
    node.attributes = {
        name: val
        for name, val in kwargs.items()
        if name not in outputs and name not in inputs
    }

    return node


@pytest.fixture(scope="function")
def grid():
    grd = MagicMock()
    return grd


def consolidated_mem(datablocks: dict[int, bytes]):
    """
    Consolidate a list of numpy arrays into a single numpy array
    """
    data = b"\0" * 4096
    for datablock_addr, datablock in datablocks.items():
        rel_addr = datablock_addr & 0x3FFF  # 4k page
        data = data[:rel_addr] + datablock + data[rel_addr + len(datablock) :]
    return data


addr_data = {kernel["addr"]: kernel["data"] for kernel in cifar10_kernels}
mram_data = {
    memname: consolidated_mem(
        {
            # select all data from addr_data that is in the 4k page
            addr: addr_data[addr]
            for addr in addr_data
            if addr & 0xFFFFC000 == blockaddr
        }
    )
    for memname, blockaddr in cnn_constants.memory.items()
    if "MRAM" in memname
}


@pytest.mark.parametrize(
    "refnet,layeridx,per_processor_offset,lenx",
    [
        (cifar10_refnet, 0, [0] * 64, (64) * 9),
        # (cifar10_refnet, 1, [(64) * 9] * 3 + [256] * 61, ((32 + 8) // 9) * 9),
        # (cifar10_refnet, 7, (1, 128, 8, 8), (64, 128, 3, 3), 0, 0),
    ],
)
def test_memory_quadrant(
    core: CNNx16Core,
    refnet: dict[int, list[Node]],
    layeridx: int,
    per_processor_offset: list[int],
    lenx: int,
):
    # find out if we have an expansion
    layout_mem(core, layeridx, refnet[layeridx])

    for quad in core:
        for locprocid, proc in quad.processors.items():
            offset = per_processor_offset[proc.global_idx]
            memname = f"CNNx16_{quad.idx}_P{locprocid}_MRAM"
            generated: bytes = proc.get_memory_bytes(mexpress=True)
            # assert generated, f"no data for {memname}"
            if generated:
                reference: bytes = mram_data[memname][offset : offset + lenx]
                logging.info(f"len {lenx}: {len(generated)} vs {len(reference)}")
                assert len(generated) == lenx, "not comparing full length"
                assert len(generated) == len(reference), "not comparing both length"
                comp = np.where(
                    np.frombuffer(generated[:lenx], dtype=np.uint8)
                    != np.frombuffer(reference[:lenx], dtype=np.uint8)
                )[0]
                if len(comp) > 0:
                    # find where in mem this is
                    for addr, data in addr_data.items():
                        if generated[:lenx] in data:
                            memname2 = {v: k for k, v in cnn_constants.memory.items()}[
                                addr & 0xFFFFC000
                            ]  # its a 4k page
                            offs = data.index(generated[:lenx])
                            try:
                                offs2 = reference.index(generated[:lenx])
                            except Exception:
                                offs2 = None
                            # shoudl be at offs3
                            offs3 = addr & 0x3FFF
                            logging.info(
                                f"found in {memname}[should be {memname2}]"
                                f" @ offs={offs}/{offs2}/{offs3},"
                                f" but tried offset={offset}"
                            )
                            assert memname != memname2, "why TF did we not find it?"
                            break
                assert len(comp) == 0, f"Differences found at indices: {comp[:10]}"
                logging.info(
                    f"no differences found for {memname}"
                    f"(lens: {len(generated)} vs {len(reference)}"
                )

    # packed = memquad.get_kernel_data(
    #     inch=mul_shape[1], outch=mul_shape[0], packed72bit=True
    # )
    # packed = packed[:lenx]
    # assert len(packed) == np.prod(shape), "our output is false length"
    # assert len(packed) == len(reference), "not the same length"
    # assert packed == reference, "not the same value"


@pytest.mark.parametrize(
    "processor_usage_mask,reference_bytes",
    [
        (
            # basically kernel_data[0][:3] with ll ==0 and in kernels.py:573
            0x0007,
            b"\xfe\x08\xf8\xf6\xfb\x08\x07\x15\x03\xf2\x0d\xff\xfb\x05\xf4\xf6"
            b"\xec\xee\x02\xf5\x09\xed\xf4\x03\x0c\x06\x07"
            b"\x0a\x23\xe1\x15\xe5\xf4\x01\x14\xee\x0b\x18\xfc\x27\xca\x23\x16"
            b"\xca\xe8\xf4\x00\xff\x11\x0d\x16\x0c\x24\x01"
            b"\xfc\xe4\xeb\x0d\x12\xf8\xf6\x1d\x0d\x0e\x2b\x19\xfa\xf5\xed\xf7"
            b"\xf6\xf7\xf8\xe9\x2d\xea\xee\x33\x04\x02\xe5",
        ),
    ],
)
def test_memoryquadrant_get_kernel_data(processor_usage_mask, reference_bytes):
    import numpy as np

    m = MemoryQuadrant(
        processors=processor_usage_mask, per_layer_quantization=[8] * 16, tensor=None
    )
    m.data.append(
        np.array(
            # content is basically kernel[0][:27] => (192[:27],3,3)
            [
                [[-2, 8, -8], [-10, -5, 8], [7, 21, 3]],
                [[-6, -8, -9], [-24, -20, -4], [2, 14, 11]],
                [[-13, 0, 3], [-10, -9, 10], [-2, 13, 16]],
                [[-14, 13, -1], [-5, 5, -12], [-10, -20, -18]],
                [[-11, 26, 9], [5, 0, -4], [3, -16, -9]],
                [[-6, 29, 14], [0, 8, 13], [6, -1, -1]],
                [[2, -11, 9], [-19, -12, 3], [12, 6, 7]],
                [[14, 12, 4], [-2, -1, 1], [13, -16, -11]],
                [[12, 11, 8], [5, -9, -26], [3, -21, -29]],
                [[10, 35, -31], [21, -27, -12], [1, 20, -18]],
                [[-20, 28, 16], [-5, -43, -1], [10, 35, 0]],
                [[-15, 44, 1], [1, -21, -19], [0, 7, -23]],
                [[11, 24, -4], [39, -54, 35], [22, -54, -24]],
                [[21, 10, -10], [55, -17, -18], [45, -33, -46]],
                [[-54, -12, 6], [-59, -5, 70], [-45, 29, 72]],
                [[-12, 0, -1], [17, 13, 22], [12, 36, 1]],
                [[-6, -8, 19], [-9, -69, -32], [-25, -25, -5]],
                [[8, 25, -5], [21, -25, 2], [-4, 8, 1]],
                [[-4, -28, -21], [13, 18, -8], [-10, 29, 13]],
                [[5, 23, 14], [-20, 35, -3], [-34, 17, -26]],
                [[6, 4, 19], [-5, 20, 6], [-26, -6, -29]],
                [[14, 43, 25], [-6, -11, -19], [-9, -10, -9]],
                [[-37, 4, 5], [-12, -31, -23], [-11, 26, -3]],
                [[-3, 11, 1], [3, -2, -2], [-1, 21, 5]],
                [[-8, -23, 45], [-22, -18, 51], [4, 2, -27]],
                [[4, -48, -6], [2, -23, 10], [34, 19, -2]],
                [[22, -26, 14], [2, -26, 17], [12, 13, -30]],
            ],
        )
    )
    inputch = 3
    kernel_shape = (3, 3)
    packed = m.get_kernel_data(9, 3, packed72bit=True)
    packed = packed[: (27 // inputch) * (kernel_shape[0] * kernel_shape[1])]
    assert len(packed) == len(reference_bytes), "len is not the same"
    # assert packed == reference_bytes, "sad"


@pytest.mark.parametrize(
    "inc,outc,expected",
    [
        (3, 3, [0, 3, 6, 1, 4, 7, 2, 5, 8]),
        (3, 2, [0, 3, 1, 4, 2, 5]),
        (3, 5, [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]),
    ],
)
def test_memoryquadrant_kernel_reorder(inc, outc, expected):
    m = MemoryQuadrant(processors=0xFFFF, per_layer_quantization=[8] * 16, tensor=None)
    assert list(m.kernel_reorder_generator(inc, outc)) == expected


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
