"""
This file contains the Core, Quadrant and Layer for the MAX78000

Explicit copyright (C) 2024 notice:
If no explicit written permission is given, this file and closely related files are not
    allowed for any other purposes than reading by humans.
All rights are reserved by the original author.

"""

from collections import defaultdict
from collections.abc import Iterator
from typing import Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, conint

import netdeployonnx.devices.max78000.cnn_constants as cnn_constants
from netdeployonnx.devices.max78000.cnn_constants import (
    CNNx16_n_IFRM_IFRM_REG_VALUEMASK,
    CNNx16_n_Ly_CCNT_CCNT_MAX_VALUEMASK,
    CNNx16_n_Ly_CCNT_CCNT_PAD_VALUEMASK,
    CNNx16_n_Ly_EN_MASK_EN_VALUEMASK,
    CNNx16_n_Ly_EN_PRO_EN_VALUEMASK,
    CNNx16_n_Ly_LCTRL0_CNNSI_EN_VALUEMASK,
    CNNx16_n_Ly_LCTRL0_SSLAVE_VALUEMASK,
    CNNx16_n_Ly_LCTRL1_INPCHEXP_VALUEMASK,
    CNNx16_n_Ly_LCTRL1_WPTR_INC_VALUEMASK,
    CNNx16_n_Ly_LCTRL1_XPCH_MAX_VALUEMASK,
    CNNx16_n_Ly_MCNT_MCNT_MAX_VALUEMASK,
    CNNx16_n_Ly_MCNT_MCNT_SAD_VALUEMASK,
    CNNx16_n_Ly_ONED_EWISE_CNT_VALUEMASK,
    CNNx16_n_Ly_ONED_EWISE_FUNC_VALUEMASK,
    CNNx16_n_Ly_ONED_ONED_SAD_VALUEMASK,
    CNNx16_n_Ly_ONED_ONED_WIDTH_VALUEMASK,
    CNNx16_n_Ly_ONED_TSCNT_MAX_VALUEMASK,
    CNNx16_n_Ly_PCCNT_PCCNT_MAX_VALUEMASK,
    CNNx16_n_Ly_POST_BPTR_SAD_VALUEMASK,
    CNNx16_n_Ly_POST_MASK_SIZE_VALUEMASK,
    CNNx16_n_Ly_POST_XPMP_CNT_VALUEMASK,
    CNNx16_n_Ly_PRCNT_PRCNT_MAX_VALUEMASK,
    CNNx16_n_Ly_RCNT_RCNT_MAX_VALUEMASK,
    CNNx16_n_Ly_RCNT_RCNT_PAD_VALUEMASK,
    CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_VALUEMASK,
    CNNx16_n_Ly_STRIDE_STRIDE_VALUEMASK,
    CNNx16_n_Ly_TPTR_TPTR_MAX_VALUEMASK,
    CNNx16_n_Ly_TPTR_TPTR_SAD_VALUEMASK,
    CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_VALUEMASK,
    CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_VALUEMASK,
    CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_VALUEMASK,
    CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_VALUEMASK,
    CNNx16_n_MLAT_MLATDAT_VALUEMASK,
)
from netdeployonnx.devices.max78000.cnn_registers import CNNx16_BaseReg

PROCESSORS_PER_QUADRANT = 16
MAX_LAYERS_PER_QUADRANT = 16
MAX_PROCESSORS = 4 * PROCESSORS_PER_QUADRANT


class CNNx16_Processor(BaseModel):  # noqa: N801
    """
    CNNx16 Processor
    This is a pydantic model for a CNNx16 processor, because we need to verify
        some of the properties
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    # PROCESSOR
    idx: conint(ge=0, le=15)
    quadrant: "CNNx16_Quadrant"
    _unused: bool = True

    # layeridx -> kernel_data
    layer: dict[int, list[np.ndarray]] = defaultdict(lambda: [])

    enabled: bool = False

    # per processor data
    kernels: dict[int, bytes] = {}  # relative address -> kernel_data

    def __init__(self, quadrant: "CNNx16_Quadrant", idx: int):
        super().__init__(quadrant=quadrant, idx=idx)

    @property
    def unused(self):
        """
        returns True if the proc is unused
        """
        return self._unused

    def __setattr__(self, name, value):
        # dont care init vars
        if name not in ["unused", "idx", "quadrant"] and value not in [0, None]:
            if self.unused:
                self._unused = False
        if name == "bias_addr":
            self.bias_en = True
        super().__setattr__(name, value)

    @property
    def group(self) -> int:
        return self.idx // 4

    @property
    def global_idx(self) -> int:
        return self.idx + self.quadrant.idx * PROCESSORS_PER_QUADRANT

    def add_kernel_for_layer(self, layeridx: int, kernel_data: np.ndarray):
        self.layer[layeridx].append(kernel_data)
        # do we need to do anything?

    def get_memory_bytes(self, /, mexpress: bool) -> bytes:
        # do we have memexpress enabled?
        data: bytes = b""
        kernel_data = b""
        for layeridx in self.layer:
            kernel_list = self.layer[layeridx]
            if not kernel_list:
                # TODO: maybe return filler?
                # it seems like we dont need a filler (as it would consume memory)
                # but an address skip instruction. so what if we add Nones?
                # or just a memory map? we could have another method that
                # provides the memory map
                continue
            for kernel_collection in kernel_list:
                # check for the shape, assume X, Y, Y
                assert (
                    kernel_collection.shape[2] == kernel_collection.shape[1]
                ), "Only square kernels supported"

                for kernelidx, kernel in enumerate(kernel_collection[:,]):
                    # we have to reorder the kernel
                    kernel_data += kernel.astype(dtype=np.uint8).tobytes()[::-1]
                    if len(kernel_data) >= 9:
                        if mexpress:
                            data += kernel_data[:9][::-1]
                        else:
                            raise NotImplementedError("untested")
                            data += b"\x00\x00\x00" + kernel_data[:9][::-1]
                        kernel_data = kernel_data[9:]
            if len(kernel_data) > 0:
                data += kernel_data[:9][::-1]  # finally copy the last residue?
                data += b"\x00" * (9 - len(kernel_data))  # add some weird buffer?
                kernel_data = kernel_data[9:]  # now clear the buffer
            if len(kernel_data) > 0:
                raise NotImplementedError("Kernel data residue?")
        assert mexpress and len(data) % 9 == 0, "Not a multiple of 9, but mexpress"
        return data


class CNNx16_Layer(BaseModel):  # noqa: N801
    """
    CNNx16 Layer
    This is a pydantic model for a CNNx16 layer, because we need to verify some of the
        properties
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    # LAYER
    idx: conint(ge=0, le=15)
    quadrant: "CNNx16_Quadrant"
    _unused: bool = True
    layer_field_dict: dict = {}

    # RCNT
    row_count: conint(ge=0, le=CNNx16_n_Ly_RCNT_RCNT_MAX_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "RCNT", "field": "RCNT_MAX"}
    )
    row_pad: conint(ge=0, le=CNNx16_n_Ly_RCNT_RCNT_PAD_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "RCNT", "field": "RCNT_PAD"}
    )

    # CCNT
    col_count: conint(ge=0, le=CNNx16_n_Ly_CCNT_CCNT_MAX_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "CCNT", "field": "CCNT_MAX"}
    )
    col_pad: conint(ge=0, le=CNNx16_n_Ly_CCNT_CCNT_PAD_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "CCNT", "field": "CCNT_PAD"}
    )

    # ONED
    elementwise_channels: conint(ge=0, le=CNNx16_n_Ly_ONED_EWISE_CNT_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "ONED", "field": "EWISE_CNT"}
    )
    elementwise_2d_conv: bool = Field(
        default=False,
        json_schema_extra={
            "register": "ONED",
            "field": "2D_CONV",
            "altname": "TWO_D_CONV",
        },
    )
    prepool: bool = Field(
        default=False, json_schema_extra={"register": "ONED", "field": "PREPOOL"}
    )
    elementwise_func: conint(ge=0, le=CNNx16_n_Ly_ONED_EWISE_FUNC_VALUEMASK) = Field(
        default=0,  # 0 is sub, 1 is add, 2 is or, 3 is xor
        json_schema_extra={"register": "ONED", "field": "EWISE_FUNC"},
    )
    elementwise_en: bool = Field(
        default=False, json_schema_extra={"register": "ONED", "field": "EWISE_EN"}
    )
    oned_processing_en: bool = Field(
        default=False, json_schema_extra={"register": "ONED", "field": "ONED_EN"}
    )
    oned_mask_width: conint(ge=0, le=CNNx16_n_Ly_ONED_ONED_WIDTH_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "ONED", "field": "ONED_WIDTH"}
    )
    oned_mask_start: conint(ge=0, le=CNNx16_n_Ly_ONED_ONED_SAD_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "ONED", "field": "ONED_SAD"}
    )
    timeslot_max_count: conint(ge=0, le=CNNx16_n_Ly_ONED_TSCNT_MAX_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "ONED", "field": "TSCNT_MAX"}
    )

    # PRCNT
    row_pooling: conint(ge=0, le=CNNx16_n_Ly_PRCNT_PRCNT_MAX_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "PRCNT", "field": "PRCNT_MAX"}
    )

    # PCCNT
    col_pooling: conint(ge=0, le=CNNx16_n_Ly_PCCNT_PCCNT_MAX_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "PCCNT", "field": "PCCNT_MAX"}
    )

    # STRIDE
    stride: conint(ge=0, le=CNNx16_n_Ly_STRIDE_STRIDE_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "STRIDE", "field": "STRIDE"}
    )

    # WPTR_BASE
    writeptr: conint(ge=0, le=CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "WPTR_BASE", "field": "WPTR_BASE"}
    )

    # WPTR_TOFF
    writeptr_timeslot_offset: conint(
        ge=0, le=CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_VALUEMASK
    ) = Field(
        default=0, json_schema_extra={"register": "WPTR_TOFF", "field": "WPTR_TOFF"}
    )

    # WPTR_MOFF
    writeptr_mask_offset: conint(ge=0, le=CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_VALUEMASK) = (
        Field(
            default=0, json_schema_extra={"register": "WPTR_MOFF", "field": "WPTR_MOFF"}
        )
    )

    # WPTR_CHOFF
    writeptr_multipass_offset: conint(
        ge=0, le=CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_VALUEMASK
    ) = Field(
        default=0, json_schema_extra={"register": "WPTR_CHOFF", "field": "WPTR_CHOFF"}
    )

    # RPTR_BASE
    readptr: conint(ge=0, le=CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "RPTR_BASE", "field": "RPTR_BASE"}
    )

    # LCTRL0
    big_data_write: bool = Field(
        default=False, json_schema_extra={"register": "LCTRL0", "field": "BIGDWRT"}
    )
    nonscaled_nonquantized_sum_feed_en: conint(
        ge=0, le=CNNx16_n_Ly_LCTRL0_CNNSI_EN_VALUEMASK
    ) = Field(default=0, json_schema_extra={"register": "LCTRL0", "field": "CNNSI_EN"})
    sram_load_source: bool = Field(
        default=False, json_schema_extra={"register": "LCTRL0", "field": "SRAMLSRC"}
    )
    input_frame_colpad: bool = Field(
        default=False, json_schema_extra={"register": "LCTRL0", "field": "CPAD_ONLY"}
    )
    relu_en: bool = Field(
        default=False, json_schema_extra={"register": "LCTRL0", "field": "ACT_EN"}
    )
    maxpool_en: bool = Field(
        default=False, json_schema_extra={"register": "LCTRL0", "field": "MAXPL_EN"}
    )
    pool_en: bool = Field(
        default=False, json_schema_extra={"register": "LCTRL0", "field": "POOL_EN"}
    )
    parallel: bool = Field(
        default=False, json_schema_extra={"register": "LCTRL0", "field": "PARALLEL"}
    )
    master: bool = Field(
        default=False, json_schema_extra={"register": "LCTRL0", "field": "MASTER"}
    )
    mslave: bool = Field(
        default=False, json_schema_extra={"register": "LCTRL0", "field": "MSLAVE"}
    )
    sslave: conint(ge=0, le=CNNx16_n_Ly_LCTRL0_SSLAVE_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "LCTRL0", "field": "SSLAVE"}
    )

    # MCNT
    mask_start: conint(ge=0, le=CNNx16_n_Ly_MCNT_MCNT_SAD_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "MCNT", "field": "MCNT_SAD"}
    )
    mask_maxaddr: conint(ge=0, le=CNNx16_n_Ly_MCNT_MCNT_MAX_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "MCNT", "field": "MCNT_MAX"}
    )

    # TPTR
    tram_start: conint(ge=0, le=CNNx16_n_Ly_TPTR_TPTR_SAD_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "TPTR", "field": "TPTR_SAD"}
    )
    tram_maxaddr: conint(ge=0, le=CNNx16_n_Ly_TPTR_TPTR_MAX_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "TPTR", "field": "TPTR_MAX"}
    )

    # EN
    enable_mask: conint(ge=0, le=CNNx16_n_Ly_EN_MASK_EN_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "EN", "field": "MASK_EN"}
    )
    enable_processor: conint(ge=0, le=CNNx16_n_Ly_EN_PRO_EN_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "EN", "field": "PRO_EN"}
    )

    # POST
    deconv_en: bool = Field(
        default=False, json_schema_extra={"register": "POST", "field": "DECONV_EN"}
    )
    flatten_en: bool = Field(
        default=False, json_schema_extra={"register": "POST", "field": "FLATTEN_EN"}
    )
    out_abs: bool = Field(
        default=False, json_schema_extra={"register": "POST", "field": "OUT_ABS"}
    )
    onexone_en: bool = Field(
        default=False, json_schema_extra={"register": "POST", "field": "ONEXONE_EN"}
    )
    ts_en: bool = Field(
        default=False, json_schema_extra={"register": "POST", "field": "TS_EN"}
    )
    mask_size: conint(ge=0, le=CNNx16_n_Ly_POST_MASK_SIZE_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "POST", "field": "MASK_SIZE"}
    )
    multipass: conint(ge=0, le=CNNx16_n_Ly_POST_XPMP_CNT_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "POST", "field": "XPMP_CNT"}
    )
    shift_by: conint(ge=-15, le=15) = Field(
        default=0,  # positive values means right shift, negative means left shift
        json_schema_extra={"register": "POST", "field": "SCALE_REG"},
    )
    bias_en: bool = Field(
        default=False, json_schema_extra={"register": "POST", "field": "BPTR_EN"}
    )
    bias_addr: conint(ge=0, le=CNNx16_n_Ly_POST_BPTR_SAD_VALUEMASK) = Field(
        default=0, json_schema_extra={"register": "POST", "field": "BPTR_SAD"}
    )

    # LCTRL1
    expansion_mode_processors: conint(
        ge=0, le=CNNx16_n_Ly_LCTRL1_XPCH_MAX_VALUEMASK
    ) = Field(default=0, json_schema_extra={"register": "LCTRL1", "field": "XPCH_MAX"})
    expansion_mode_writeptr: conint(ge=0, le=CNNx16_n_Ly_LCTRL1_WPTR_INC_VALUEMASK) = (
        Field(default=0, json_schema_extra={"register": "LCTRL1", "field": "WPTR_INC"})
    )
    expansion_mode_inputchan: conint(ge=0, le=CNNx16_n_Ly_LCTRL1_INPCHEXP_VALUEMASK) = (
        Field(default=0, json_schema_extra={"register": "LCTRL1", "field": "INPCHEXP"})
    )

    def __init__(self, quadrant: "CNNx16_Quadrant", idx: int):
        super().__init__(quadrant=quadrant, idx=idx)
        self.layer_field_dict: dict = {
            layer_field.json_schema_extra.get(
                "altname",
                layer_field.json_schema_extra.get("field"),
            ): layer_fieldname
            for layer_fieldname, layer_field in self.model_fields.items()
            if layer_field.json_schema_extra
        }

    @property
    def unused(self):
        """
        returns True if the layer is unused
        """
        return self._unused

    def write_register(
        self, register_name: str, fields: Union[dict, int]
    ) -> list[tuple[str, int]]:
        """
        Write a register with the given fields
        """
        cnn_constant_vars = vars(cnn_constants)
        generic_register_name = f"CNNx16_n_Ly_{register_name}"
        value = 0
        if isinstance(fields, dict):
            for field, field_value in fields.items():
                value |= (
                    field_value
                    << cnn_constant_vars[f"{generic_register_name}_{field}_POS"]
                ) & cnn_constant_vars[f"{generic_register_name}_{field}_MASK"]
        elif isinstance(fields, int):
            value = fields
        else:
            raise ValueError(f"Invalid fields type: {type(fields)}")
        return [(f"CNNx16_{self.quadrant.idx}_L{self.idx}_{register_name}", value)]

    def __setattr__(self, name, value):
        # dont care init vars
        if name not in ["unused", "idx", "quadrant"] and value not in [0, None]:
            if self.unused:
                self._unused = False
        if name == "bias_addr":
            self.bias_en = True
        super().__setattr__(name, value)

    def set_from_register(self, register: CNNx16_BaseReg):
        """
        Set the layer properties from a register
        """
        assert hasattr(register, "model_fields"), "should be a pydantic model"
        if "Ly" in type(register).__name__:
            for fieldname, field in register.model_fields.items():
                if "reserved" in fieldname.lower():
                    continue
                value = getattr(register, fieldname)
                if fieldname == "SCALE_REG":
                    val = abs(getattr(self, "shift_by"))
                    sign = -1 if value < 0 else 1
                    value = value if val == 0 else sign * value
                if fieldname == "SCALE_SHFT":
                    fieldname = "SCALE_REG"
                    val = abs(getattr(self, "shift_by"))
                    value = -val if value else val
                # print(f"setting {self.layer_field_dict[fieldname]} to {value}")
                setattr(self, self.layer_field_dict[fieldname], value)
        # else:
        #     print("skipping?!", type(register).__name__)

    def instructions_configure(self) -> list:  # noqa: C901
        ret = []
        if self.unused:
            return ret
        ret.append(f"// Layer {self.idx} quadrant {self.quadrant.idx}")

        if self.row_count or self.row_pad:
            ret += self.write_register(
                "RCNT",
                {
                    "RCNT_MAX": self.row_count,
                    "RCNT_PAD": self.row_pad,
                },
            )
        if self.col_count or self.col_pad:
            ret += self.write_register(
                "CCNT",
                {
                    "CCNT_MAX": self.col_count,
                    "CCNT_PAD": self.col_pad,
                },
            )

        if (
            self.elementwise_channels
            or self.elementwise_2d_conv
            or self.prepool
            or self.elementwise_func
            or self.elementwise_en
            or self.oned_processing_en
            or self.oned_mask_width
            or self.oned_mask_start
            or self.timeslot_max_count
        ):
            ret += self.write_register(
                "ONED",
                {
                    "EWISE_CNT": self.elementwise_channels,
                    "2D_CONV": self.elementwise_2d_conv,
                    "PREPOOL": self.prepool,
                    "EWISE_FUNC": self.elementwise_func,
                    "EWISE_EN": self.elementwise_en,
                    "ONED_EN": self.oned_processing_en,
                    "ONED_WIDTH": self.oned_mask_width,
                    "ONED_SAD": self.oned_mask_start,
                    "TSCNT_MAX": self.timeslot_max_count,
                },
            )

        if self.row_pooling:
            ret += self.write_register(
                "PRCNT",
                {
                    "PRCNT_MAX": self.row_pooling,
                },
            )

        if self.col_pooling:
            ret += self.write_register(
                "PCCNT",
                {
                    "PCCNT_MAX": self.col_pooling,
                },
            )

        if self.stride:
            ret += self.write_register("STRIDE", self.stride)

        if self.writeptr:
            ret += self.write_register("WPTR_BASE", self.writeptr)

        if self.writeptr_timeslot_offset:
            ret += self.write_register("WPTR_TOFF", self.writeptr_timeslot_offset)

        if self.writeptr_mask_offset:
            ret += self.write_register("WPTR_MOFF", self.writeptr_mask_offset)

        if self.writeptr_multipass_offset:
            ret += self.write_register("WPTR_CHOFF", self.writeptr_multipass_offset)

        if self.readptr:
            ret += self.write_register("RPTR_BASE", self.readptr)

        if (
            self.big_data_write
            or self.nonscaled_nonquantized_sum_feed_en
            or self.sram_load_source
            or self.input_frame_colpad
            or self.relu_en
            or self.maxpool_en
            or self.pool_en
            or self.parallel
            or self.master
            or self.mslave
            or self.sslave
        ):
            ret += self.write_register(
                "LCTRL0",
                {
                    "BIGDWRT": self.big_data_write,
                    "CNNSI_EN": self.nonscaled_nonquantized_sum_feed_en,
                    "SRAMLSRC": self.sram_load_source,
                    "CPAD_ONLY": self.input_frame_colpad,
                    "ACT_EN": self.relu_en,
                    "MAXPL_EN": self.maxpool_en,
                    "POOL_EN": self.pool_en,
                    "PARALLEL": self.parallel,
                    "MASTER": self.master,
                    "MSLAVE": self.mslave,
                    "SSLAVE": self.sslave,
                },
            )

        if self.mask_start or self.mask_maxaddr:
            ret += self.write_register(
                "MCNT",
                {
                    "MCNT_SAD": self.mask_start,
                    "MCNT_MAX": self.mask_maxaddr,
                },
            )

        if self.tram_start or self.tram_maxaddr:
            ret += self.write_register(
                "TPTR",
                {
                    "TPTR_SAD": self.tram_start,
                    "TPTR_MAX": self.tram_maxaddr,
                },
            )

        if self.enable_mask or self.enable_processor:
            ret += self.write_register(
                "EN",
                {
                    "MASK_EN": self.enable_mask,
                    "PRO_EN": self.enable_processor,
                },
            )

        if (
            self.deconv_en
            or self.flatten_en
            or self.out_abs
            or self.onexone_en
            or self.ts_en
            or self.mask_size
            or self.multipass
            or self.shift_by
            or self.bias_en
            or self.bias_addr
        ):
            ret += self.write_register(
                "POST",
                {
                    "DECONV_EN": self.deconv_en,  # 0 is deconv, 1 is conv
                    "FLATTEN_EN": self.flatten_en,  # 0 is flatten, 1 is unflatten
                    "OUT_ABS": self.out_abs,  # 0 is signed, 1 is unsigned
                    "ONEXONE_EN": self.onexone_en,  # 0 is 1x1, 1 is not 1x1
                    "TS_EN": self.ts_en,  # 0 is disabled, 1 is enabled
                    # 0 is 8bits, 1 is 1bit, 2 is 2 bit, 3 is 4 bit
                    "MASK_SIZE": self.mask_size,
                    # multipass count, max 15, only used when flatten_en is 1
                    "XPMP_CNT": self.multipass,
                    "SCALE_SHFT": (
                        0 if self.shift_by >= 0 else 1
                    ),  # 0 is left, 1 is right
                    "SCALE_REG": abs(self.shift_by),  # max 15
                    "BPTR_EN": (
                        self.bias_en if self.bias_addr == 0 else True
                    ),  # used when sad > 0 or when enabled
                    "BPTR_SAD": self.bias_addr,  # bias addr: 11 bits
                },
            )

        if (
            self.expansion_mode_processors
            or self.expansion_mode_writeptr
            or self.expansion_mode_inputchan
        ):
            ret += self.write_register(
                "LCTRL1",
                {
                    # Expansion Mode Maximum Processors
                    "XPCH_MAX": self.expansion_mode_processors,
                    # Write pointer increment (after all output channels are writted)
                    "WPTR_INC": self.expansion_mode_writeptr,
                    # Input channel expansion count
                    "INPCHEXP": self.expansion_mode_inputchan,
                },
            )
        # check for streams

        # TODO: remove, this is only for the synth to c
        ret.append("")
        return ret

    @property
    def output_row(self) -> int:
        """row count accounting for pooling and padding"""
        return self.row_count + self.row_pad - self.row_pooling - 1

    @property
    def output_col(self) -> int:
        """col count accounting for pooling and padding"""
        return self.col_count + self.col_pad - self.col_pooling - 1


class CNNx16_Quadrant(BaseModel):  # noqa: N801
    """
    CNNx16 Quadrant
    """

    # MODEL
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    idx: conint(ge=0, le=3)
    layers: dict[int, CNNx16_Layer]  # layeridx -> layer
    processors: dict[int, CNNx16_Processor] = {}  # processor_id & 0xF -> layer

    # per quadrant data
    bias: bytes = b""  # TODO: check if unused and deletable

    # CTRL

    # SRAM

    # LCNT_MAX

    # TEST

    # IFRM
    input_frame_size: conint(ge=0, le=CNNx16_n_IFRM_IFRM_REG_VALUEMASK) = 0

    # MLAT
    mlat_data: conint(ge=0, le=CNNx16_n_MLAT_MLATDAT_VALUEMASK) = 0

    def __init__(self, idx: int, layer_count: int = MAX_LAYERS_PER_QUADRANT):
        """
        Initialize the quadrant
        """
        assert layer_count <= MAX_LAYERS_PER_QUADRANT
        super().__init__(
            idx=idx,
            layers={
                layeridx: CNNx16_Layer(self, layeridx)
                for layeridx in range(layer_count)
            },
            processors={
                processoridx: CNNx16_Processor(self, processoridx)
                for processoridx in range(
                    PROCESSORS_PER_QUADRANT
                )  # TODO: remove magic numbers
            },
        )

    @property
    def unused(self):
        """
        returns True if all layers in the quadrant are unused
        """
        return all([self.layers[layer].unused for layer in self.layers])

    @property
    def max_used_layer(self):
        """
        returns the maximum used layer index in the quadrant
        """
        if self.unused:
            return -1
        return max([layer.idx for layer in self.layers.values() if not layer.unused])

    def __getitem__(self, key: int) -> CNNx16_Layer:
        if isinstance(key, int):
            return self.layers[key]
        else:
            raise ValueError("Invalid key type")

    def __iter__(self) -> Iterator[CNNx16_Layer]:
        return iter(self.layers.values())

    def write_register(
        self, register_name: str, fields: Union[dict, int]
    ) -> list[tuple[str, int]]:
        """
        Write a register with the given fields
        """
        cnn_constant_vars = vars(cnn_constants)
        generic_register_name = f"CNNx16_n_{register_name}"
        value = 0
        if isinstance(fields, dict):
            for field, field_value in fields.items():
                value |= (
                    field_value
                    << cnn_constant_vars[f"{generic_register_name}_{field}_POS"]
                ) & cnn_constant_vars[f"{generic_register_name}_{field}_MASK"]
        elif isinstance(fields, int):
            value = fields
        else:
            raise ValueError(f"Invalid fields type: {type(fields)}")
        return [(f"CNNx16_{self.idx}_{register_name}", value)]

    # are there quadrant-only instructions?
    def instructions_init(self):
        """
        return instructions like ctrl, sram, layercount, etc.
        """
        ret = []
        if self.unused:
            return ret
        if 1:
            ret += self.write_register(
                "CTRL",
                {
                    "MEXPRESS": True,
                    "CLK_EN": True,
                },
            )
        if 1:
            ret += self.write_register(
                "SRAM",
                {
                    "RMARGIN": 3,
                    "WNEG_EN": True,
                },
            )
        if 1:
            ret += self.write_register("LCNT_MAX", self.max_used_layer)
        return ret

    def instructions_configure(self):
        """
        return instructions like layercount, etc.
        """
        if self.unused:
            return []
        ret = []

        # MLAT
        # CTRL
        # SRAM
        # LCNT_MAX
        # TEST
        # IFRM
        return ret

    def add_quadkernel_for_layer(
        self, layeridx: int, processor_id: int, kernel_data: np.ndarray, **kwargs
    ):
        """
        Set the kernel data for a layer
        """
        self.processors[processor_id % 16].add_kernel_for_layer(
            layeridx,
            kernel_data,
            **kwargs,
        )


class CNNx16Core:
    # do not forget the AON and FIFO control

    def __init__(self, quadrant_count: int = 4):
        self.quadrants = {
            quadrant: CNNx16_Quadrant(quadrant) for quadrant in range(quadrant_count)
        }

    def __getitem__(self, key: Union[tuple[int, int], int]) -> CNNx16_Layer:
        if isinstance(key, int):
            core = key
            return self.quadrants[core]
        elif isinstance(key, tuple):
            core, layer = key
            return self.quadrants[core].layers[layer]
        else:
            raise ValueError("Invalid key type")

    def __iter__(self) -> Iterator[CNNx16_Quadrant]:
        return iter(self.quadrants.values())

    def write_register(
        self, register_name: str, fields: Union[dict, int]
    ) -> list[tuple[str, int]]:
        """
        Write a register with the given fields
        """
        cnn_constant_vars = vars(cnn_constants)
        generic_register_name = f"CNNx16_n_{register_name}"
        value = 0
        if isinstance(fields, dict):
            for field, field_value in fields.items():
                value |= (
                    field_value
                    << cnn_constant_vars[f"{generic_register_name}_{field}_POS"]
                ) & cnn_constant_vars[f"{generic_register_name}_{field}_MASK"]
        elif isinstance(fields, int):
            value = fields
        else:
            raise ValueError(f"Invalid fields type: {type(fields)}")
        return [(f"CNNx16_{register_name}", value)]

    def instructions_init(self):
        """
        Initialize the CNN core / Do AOD + FIFO control
        """
        ret = []

        if 1:
            ret += self.write_register(
                "AOD_CTRL", 0
            )  # Disable Always on domain control
        return ret

    @classmethod
    def processor_quadrant(cls, processor_id: int) -> int:
        """
        Get the quadrant for a processor
        """
        assert 0 <= processor_id < MAX_PROCESSORS
        return processor_id // PROCESSORS_PER_QUADRANT
