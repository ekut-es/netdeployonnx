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
from pydantic import BaseModel, Field


class CNNx16_BaseReg(BaseModel):  # noqa: N801
    _regname: str
    _value: int = 0

    def __init__(self, **data):
        super().__init__(**data)
        self._regname = self.__class__.__name__
        self._update_value()

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, value: int):
        self._value = value
        self._update_fields()

    def _update_value(self):
        self._value = 0
        for name, field in self.model_fields.items():
            if 'bit' in field.json_schema_extra:
                bit = field.json_schema_extra['bit']
                bits = field.json_schema_extra['bits']
                mask = (1 << bits) - 1
                field_value = getattr(self, name)
                self._value |= (field_value & mask) << bit

    def _update_fields(self):
        for name, field in self.model_fields.items():
            if 'bit' in field.json_schema_extra:
                bit = field.json_schema_extra['bit']
                bits = field.json_schema_extra['bits']
                mask = (1 << bits) - 1
                field_value = (self._value >> bit) & mask
                setattr(self, name, field_value)


class Reg_CNNx16_n_CTRL(CNNx16_BaseReg):  # noqa: N801
    CNN_EN:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 1, 'doc': 'CNN EnableSetting this bit to 1 initiates CNN processing. Processing is triggered by this field to1. Software must set this field to 0 and back to 1 to perform subsequent CNNprocessing operations.0: CNN processing stopped1: Start CNN processingNote: This field must be written to 0 before writing it to 1 for subsequent CNNprocessing to start.'})  # noqa: E501
    RDY_SEL:int = Field(default=3, json_schema_extra={'bit': 1, 'bits': 2, 'doc': 'APB Wait State SelectionThis field determines the number of wait states added during an APB access.0b00: 0 wait states0b01: 1 wait state0b10: 2 wait states0b11: 3 wait statesNote: Write operations load the data one clock before the end of the cycle.'})  # noqa: E501
    CLK_EN:int = Field(default=0, json_schema_extra={'bit': 3, 'bits': 1, 'doc': 'Data Processing Clock EnableSetting this bit enables the clocks to the . This field does not affect the clocks to theAPB registers. See the CNNx16_n_CTRL.apbclken bit for the description of the APBclock behavior.0: Clocks disabled to the Data Processing registers1: Clocks enabled to the Data Processing registers'})  # noqa: E501
    CALCMAX:int = Field(default=0, json_schema_extra={'bit': 4, 'bits': 1, 'doc': 'Max Pooling EnableThis bit globally enables max pooling for all layers when theCNNx16_n_CTRL.pool_en bit is set.Note that this bit will be in effect, per layer, when the globalCNNx16_n_CTRL.pool_en bit is 0, and the per-layer CNNx16_n_Ly_LCTRL0.pool_enbits are set.'})  # noqa: E501
    POOL_EN:int = Field(default=0, json_schema_extra={'bit': 5, 'bits': 1, 'doc': 'Pooling EnableThis bit globally enables pooling for all layers.0: Global pooling disabled1: Global pooling enabled for all layers.Note: If this bit is set and the CNNx16_n_CTRL.calcmax bit is not set, the per-layerCNNx16_n_Ly_LCTRL0.maxpl_en bits are in effect.'})  # noqa: E501
    BIGDATA:int = Field(default=0, json_schema_extra={'bit': 6, 'bits': 1, 'doc': 'Big Data EnableThis bit globally selects the input data format that uses four data bytes per read forthe groups of 4 processors. In other words, the four bytes allocated for a group offour processors is multiplexed to the first processor of the group.'})  # noqa: E501
    APBCLK_EN:int = Field(default=0, json_schema_extra={'bit': 7, 'bits': 1, 'doc': 'APB Clock Always OnSetting this bit forces the APB clock always on. When this bit is set to 0, clocks areonly generated to the APB registers during write operations.0: APB clocks are only on when APB register writes occur1: APB clocks to the CNN APB registers is always on'})  # noqa: E501
    ONESHOT:int = Field(default=0, json_schema_extra={'bit': 8, 'bits': 1, 'doc': 'One Shot Layer ModeWith this bit set, only one layer is processed when the CNNx16_n_CTRL.cnn_en bit isset. To advance to the next layer, the CNNx16_n_CTRL.cnn_en bit must be reset to 0and then set to 1. The low to high transition causes the CNN state machine toadvance through the next layer. Memories can be interrogated between layerswhen CNNx16_n_CTRL.cnn_en is 0.0: One-shot layer mode disabled1: One-shot layer mode enabled'})  # noqa: E501
    EXT_SYNC:int = Field(default=0, json_schema_extra={'bit': 9, 'bits': 3, 'doc': "CNNx16_n External Sync SelectEach of these bits enable the external sync input from one of the CNN_x16_nprocessors. These bits allow the CNNx16_n processors to optionally operate insynchronization with one of the other CNNx16_n processors. In the general case,when all 64 processors are operating on a single convolution, CNNx16_0 processor 0is selected by all four of the CNNx16_n's as the master byte settingext_sync = 0b001. Combinations of processors can be configured as long as thegroups are made up of sequential processors, without gaps."})  # noqa: E501
    CNN_IRQ:int = Field(default=0, json_schema_extra={'bit': 12, 'bits': 1, 'doc': 'CNN Interrupt EnableThis read/write bit indicates when the CNN interrupt request is active. It can bewritten to zero to reset the interrupt request. This interrupt signals the completionof CNN processing and should be masked in the interrupt control logic if notrequired. It can also be written to 1 to force an interrupt.0: CNN interrupt not active.1: CNN interrupt active, write 0 to clear the CNN interrupt.'})  # noqa: E501
    POOLRND:int = Field(default=0, json_schema_extra={'bit': 13, 'bits': 1, 'doc': 'Average Pooling EnableWhen this bit is set, and average pooling is enabled, pooled values are rounded upfor remainders greater or equal to 0.5 and down for remainders less than 0.5.0: Average Pooling Disabled1: Average Pooling Enabled'})  # noqa: E501
    STREAM_EN:int = Field(default=0, json_schema_extra={'bit': 14, 'bits': 1, 'doc': 'Streaming Mode EnableWhen set, the streaming mode is enabled for the CNNx16_n processor array.Streaming behavior is defined by the CNN×16_n Processor Stream RegistersSee Streaming Mode Configuration for additional information.Note: Unexpected behavior is likely when all four CNNx16_n processor arrays are notconfigured identically for streaming/non-streaming operation. Each CNN_x16_nprocessor array should be configured identically for streaming or non-streamingoperation. See Streaming Mode Configuration for further details.'})  # noqa: E501
    FIFO_EN:int = Field(default=0, json_schema_extra={'bit': 15, 'bits': 1, 'doc': 'CNNx16_n_FIFO EnableWhen set, data for the first (input) layer is taken from the CNN_FIFO_WRn FIFOregister. One 4 byte-wide FIFO is dedicated for each of the four processor arrays.The FIFOs are accessed through the APB memory map, and each can be used in asingle byte-wide channel mode, a single 4 byte-wide channel mode, or 4 single byte-wide channel mode. The mode is determined by the data configuration written tothe FIFO through the APB, theCNNx16_n_CTRL.bigdata/CNNx16_n_Ly_LCTRL0.parallel configuration, and thechannel enables.'})  # noqa: E501
    MLAT_LD:int = Field(default=0, json_schema_extra={'bit': 16, 'bits': 1, 'doc': 'Mlator Load DataWriting the bit from a 0 to a 1 forces the CNNx16_n_Ly_WPTR_BASE address to beloaded into the Mlator address counter and selects the counter as the SRAMaddress source. SRAM reads are only possible when CNNx16_n_CTRL.cnn_en is resetto 0.'})  # noqa: E501
    MLATCH_SEL:int = Field(default=0, json_schema_extra={'bit': 17, 'bits': 2, 'doc': 'SRAM Packed Channel SelectSelects which of the four channels in the SRAM read data is packed.0: Selects channel 0, SRAM data bits 7:01: Selects channel 1 SRAM data bits 15:82: Selects channel 2 SRAM data bits 23:163: Select channel 3 SRAM data bits 31:24'})  # noqa: E501
    LILBUF:int = Field(default=0, json_schema_extra={'bit': 19, 'bits': 1, 'doc': "Stream Mode Circular Buffer EnableSetting this bit restricts the associated read buffer's bounds to a circular bufferstarting at the CNNx16_n_Ly_RPTR_BASE address and terminating at theCNNx16_n_Sz_FBUF address. When set, the circular buffer is used on all streaminglayers."})  # noqa: E501
    MEXPRESS:int = Field(default=0, json_schema_extra={'bit': 20, 'bits': 1, 'doc': 'Mask Memory Packed Memory EnableEnable loading of the mask memories using packed data. With this bit set, a changein the state of the two least significant bits of the MRAM address triggers a reload ofthe address counter.'})  # noqa: E501
    SIMPLE1B:int = Field(default=0, json_schema_extra={'bit': 21, 'bits': 1, 'doc': 'Simple 1-Bit WeightsEnable simple logic for 1-bit weights. Setting this bit disables the wide accumulatorsused to calculate the convolution products and activates simple one-bit logicfunctions.'})  # noqa: E501
    FFIFO_EN:int = Field(default=0, json_schema_extra={'bit': 22, 'bits': 1, 'doc': "Fast FIFO EnableThis field enables the tightly coupled data path between the MAX78000's CNN_TXfast FIFO and the CNN data SRAMs. The CNN_TX_FIFO is 32 bits wide, with 8 bitsbeing dedicated to each of 4 channels. Channel routing is controlled by the state ofthe CNNx16_n_CTRL.fifogrp control bit."})  # noqa: E501
    FIFOGRP:int = Field(default=0, json_schema_extra={'bit': 23, 'bits': 1, 'doc': "FIFO Group OutputEnables sending all 'little data' channels to the first 4 processors. When this bit isnot set, each byte of FIFO data is directed to the first little data channel of eachCNNx16_n processor."})  # noqa: E501
    FCLK_DLY:int = Field(default=0, json_schema_extra={'bit': 24, 'bits': 6, 'doc': 'FIFO Clock DelayThis field selects the clock delay of the FIFO clock relative to the primary CNN clock.A setting of 0 adds in the minimum delay, and a value of 0x3F adds the maximumdelay.'})  # noqa: E501
    TIMESHFT:int = Field(default=0, json_schema_extra={'bit': 30, 'bits': 1, 'doc': 'Pooling Stage Time ShiftWhen set, one wait state is added to the pooling stage to allow design time closureat a higher clock frequency.'})  # noqa: E501
    QUPAC:int = Field(default=0, json_schema_extra={'bit': 31, 'bits': 1, 'doc': 'QuPac ModeThe quad processors pack bit enables parallel processing of the same data by eachof the 16 processors in the CNNx16_n.0: Normal processing mode1: x16 parallel processing modeNote: FIFO mode must be enabled, CNNx16_n_CTRL.ffifoen set to 1, prior toenabling QuPac mode.'})  # noqa: E501


class Reg_CNNx16_n_IFRM(CNNx16_BaseReg):  # noqa: N801
    IFRM_REG:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 17, 'doc': 'Streaming Input Frame Size Byte CountWhen streaming is enabled (CNNx16_n_CTRL.stream_en = 1), this fielddetermines the number of bytes read from the data FIFOs. An internal countercounts the number of input frame bytes read from the input FIFOs andcompares the count to the value in this register. Once the value in this field isreached, the terminal count value is retained, incrementing of this counter isinhibited, and processing of the input data is terminated.'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 17, 'bits': 15, 'doc': 'Reserved'})  # noqa: E501


class Reg_CNNx16_n_LCNT_MAX(CNNx16_BaseReg):  # noqa: N801
    LCNT:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 5, 'doc': 'Layer Count MaximumSet this field to the maximum layer number for processing by the CNNx16_n.When the CNNx16_n is enabled, processing starts at layer 0 and completesprocessing at the layer number set by this field.0-31: Set to the last layer for processing by the CNNx16_nNote: The CNNx16_n must be inactive(CNNx16_n_CTRL.cnn_en=0) when settingthis field.'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 5, 'bits': 27, 'doc': 'Reserved'})  # noqa: E501


class Reg_CNNx16_n_Ly_CCNT(CNNx16_BaseReg):  # noqa: N801
    CCNT_MAX:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 10, 'doc': 'Layer Column Count MaximumWhen the CNN is enabled, these bits determine the maximum per layer columncount to be processed. Processing begins with column 0 and completes when theprocessing of the column determined by CNNx16_n_Ly_CCNT.ccnt_max is complete.The value programmed into this register, through the CNN APB interface, is theeffective image column value including pad, but excluding columns not processeddue to stride restrictions.'})  # noqa: E501
    RESERVED2:int = Field(default=0, json_schema_extra={'bit': 10, 'bits': 6, 'doc': 'Reserved'})  # noqa: E501
    CCNT_PAD:int = Field(default=0, json_schema_extra={'bit': 16, 'bits': 2, 'doc': 'Layer Column Pad CountThis field determines the number of pad columns included at the beginning and endof the frame. Note that the CNNx16_n_Ly_CCNT.ccnt_max is inclusive of two timesthis value, as the same number of pad columns are included at the beginning andend of the row.'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 18, 'bits': 14, 'doc': 'Reserved'})  # noqa: E501


class Reg_CNNx16_n_Ly_EN(CNNx16_BaseReg):  # noqa: N801
    PRO_EN:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 16, 'doc': "CNNx16_n Processor EnableEach bit in this per layer field controls the enable state of one processor in theCNNx16_n. Bit 0 controls processor 0's (the master processor) enable and bit 15controls processor 15's enable."})  # noqa: E501
    MASK_EN:int = Field(default=0, json_schema_extra={'bit': 16, 'bits': 16, 'doc': "CNNx16_n Processor Mask EnableEach bit in this per layer field enables the state of one processor's kernel logic inthe CNNx16_n. Bit 0 controls processor 0's mask application (the masterprocessor), and bit 15 controls processor 15's mask application."})  # noqa: E501


class Reg_CNNx16_n_Ly_LCTRL0(CNNx16_BaseReg):  # noqa: N801
    SSLAVE:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 4, 'doc': 'CNNx16_n Processor Group Slave ModeWithin a CNNx16_n, this field enables each of the 4 groups of four processors toshare the receptive field with the first processor of the x4 group (channels0,4,8,12). When set, the receptive field of the first processor in the associated x4group to be passed to the remaining 3 processors in the x4 group. Masksassociated with the slaved group of three processors can be applied to the fieldof the first processor and additional output channels generated. Use of thetimeslot counter is required to write the additional output channels to memory.'})  # noqa: E501
    MSLAVE:int = Field(default=0, json_schema_extra={'bit': 4, 'bits': 1, 'doc': 'CNNx16_n Processor 0 Mask LeaderWhen this bit is set, all 16 processors within the CNNx16_n share the receptivefield with processor 0. This allows the generation of additional output channelsusing the mask values distributed across the 16 processors. Each processorapplies a 3x3 mask of the selected width to be applied to the processor 0 field.Note: Use of timeslots is required to write the parallel generated output channelsto memory.'})  # noqa: E501
    MASTER:int = Field(default=0, json_schema_extra={'bit': 5, 'bits': 1, 'doc': 'CNNx16_n Processor Group Master SelectEnables a CNNx16_n group of processors to independently calculate a sum-of-products result for all adjacent ascending CNNx16_n processor groups notconfigured for master operation.'})  # noqa: E501
    PARALLEL:int = Field(default=0, json_schema_extra={'bit': 6, 'bits': 1, 'doc': "Parallel Mode EnableSetting this bit to one enables a single input channel's data to use 4 bytes ofmemory instead of one. In parallel mode, data is read in byte order from byte 0to byte 3 and then by memory depth. The purpose of the mode is to allowadditional memory to be used for the input layer data to support larger images.When set, the receptive field for the data will be generated in the first processorin each group of 4 processors, provided the processor is enabled."})  # noqa: E501
    POOL_EN:int = Field(default=0, json_schema_extra={'bit': 7, 'bits': 1, 'doc': 'Enable PoolingSetting this bit enables pooling for the associated layer. Pool dimensions aredetermined by the pool row and column count maximums(CNNx16_n_Ly_PRCNT.prcnt_max and CNNx16_n_Ly_PCCNT.pccnt_max).0: Disabled1: EnabledNote: The type of pooling performed is determined by theCNNx16_n_Ly_LCTRL0.maxpl_en field as either average pooling or max pooling.'})  # noqa: E501
    MAXPL_EN:int = Field(default=0, json_schema_extra={'bit': 8, 'bits': 1, 'doc': 'Max Pooling EnableWhen set to 1, Max Pooling is selected as the pooling mode. In Max Poolingmode, the maximum value in the pool is selected for the field and written intothe TRAM.When this field is set to 0, average pooling mode is selected. In Average PoolingMode, the average of all pooled values is calculated and selected for use in thefield.0: Average pooling mode selected.1: Max pooling mode selectedNote: This field is only used if the CNNx16_n_Ly_LCTRL0.pool_en field is set to 1.'})  # noqa: E501
    ACT_EN:int = Field(default=0, json_schema_extra={'bit': 9, 'bits': 1, 'doc': 'ReLU Activation EnableWhen set, ReLU activation is enabled for each output channel. Activation isapplied to the scaled and quantized data.0: ReLU not enabled1: ReLU activation enabled for each output channel and is applied to thescaled and quantized data.'})  # noqa: E501
    CPAD_ONLY:int = Field(default=0, json_schema_extra={'bit': 10, 'bits': 1, 'doc': 'Input Frame Column PadWhen set, padding is applied only to the input frame columns. In this case, rowpadding is ignored. This bit is intended to be used for parallel processing.Note: When this field is set, row padding is ignored.'})  # noqa: E501
    SRAMLSRC:int = Field(default=0, json_schema_extra={'bit': 11, 'bits': 1, 'doc': 'SRAM CNNx16_n SRAM Global Write Source DataWhen set, SRAM data is sourced from the global data busses. The devicesupports 4 global data busses, one from each of the CNNx16_n processors.When all four CNNx16_n processors are used together to form a single sum-of-products value, all four CNNx16_n outputs an identical address, and based onthe natural priority of the decode, processor 0 sources the SRAM write data.'})  # noqa: E501
    CNNSI_EN:int = Field(default=0, json_schema_extra={'bit': 12, 'bits': 4, 'doc': 'CNN 26-Bit Non-Scaled Non-quantized Sum of Products FeedWhen set, this field enables the associated CNNx16_n 26-bit non-scaled, non-quantized sum of products data into the output accumulator, allowing sums of16, 32, 48, or 64 products. Each bit is associated with one of the remaining3 CNNx16_n processors. In processor 0, bit 1 is associated withCNNx16_n processor 1, bit 2 with CNNx16_n processor 2, and bit 3 withCNNx16_n processor 3. In processor 1, bit 1 is associated withCNNx16_n processor 2, bit 2 with CNNx16_n processor 3, and bit 3 withCNNx16_n processor 0, and so on.When these bits are set to zero, the internal state of the 26-bit data bus isforced to zero.0b0000: 26-bit data bus set to 00b0001-0b1111: See description'})  # noqa: E501
    BIGDWRT:int = Field(default=0, json_schema_extra={'bit': 16, 'bits': 1, 'doc': 'Big Data WriteEnables writing out the current output channel in 32-bit accumulator form. Thisbit allows the full resolution of the output layer to be written to assist withsoftware-defined softmax operation.'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 17, 'bits': 15, 'doc': 'Reserved'})  # noqa: E501


class Reg_CNNx16_n_Ly_LCTRL1(CNNx16_BaseReg):  # noqa: N801
    INPCHEXP:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 4, 'doc': 'Multi-Pass Input Channel Expansion CountThis field determines the number of sequential memory locations associatedwith a single convolution.For example, if a value of 15 (0xf) is programmed into this field, 16 channels areassumed to be sequentially stored in memory (channel then byte order) for eachof the 64 processors. This allows for up to 1024 input channels to be processedin a single convolution. Similarly, if this field is set to 1, 2 channels are assumedto be sequentially stored in memory (channel then byte order) as channels forthe convolution, totaling 128 channels if all 64 processors are used. A value ofzero disables the multi-pass feature allowing for a single channel for eachprocessor.'})  # noqa: E501
    WPTR_INC:int = Field(default=0, json_schema_extra={'bit': 4, 'bits': 4, 'doc': 'Write Pointer IncrementThis field determines the increment for the write pointer counter after all outputchannels within a given stride are written to memory. Non-zero values in thisfield are used for multi-pass operations.Note: The Write Pointer is always incremented by 1 or by 4 in parallel mode. Thevalue in this field is added to the internal write pointer increment.'})  # noqa: E501
    XPCH_MAX:int = Field(default=0, json_schema_extra={'bit': 8, 'bits': 9, 'doc': 'Expansion Mode Maximum ProcessorsThis field selects the maximum channel processor number used in channelexpansion mode. This allows for fewer than 64 processors to be used in a multi-pass or channel expansion configuration.Note: Processor management was shown to be important for mask managementwhen input channel processing draws on a relatively small number of channelsfor a potentially large number of masks.0-64: Selects the number of processors to use for channel expansion modeNote: This field contains 9 bits; the upper 3 bits are reserved. Only values up to64 are supported.'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 17, 'bits': 15, 'doc': 'Reserved'})  # noqa: E501


class Reg_CNNx16_n_Ly_MCNT(CNNx16_BaseReg):  # noqa: N801
    MCNT_MAX:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 16, 'doc': 'Mask RAM Layer Maximum AddressMask RAM address counter increments sequentially by word and bit addressduring layer processing to this address. Counter restarts each stride andincrements once per output channel:• mcnt_max[15:4]→SRAM word (72-bits) address• mcnt_max[2:0]→bit address.'})  # noqa: E501
    MCNT_SAD:int = Field(default=0, json_schema_extra={'bit': 16, 'bits': 16, 'doc': 'Layer Mask RAM start AddressMask RAM address counter increments sequentially from this word and bitaddress during layer processing. Counter restarts each stride and incrementsonce per output channel:• mcnt_sad[15:4]→SRAM word (72bits) address• mcnt_sad[2:0]→bit address'})  # noqa: E501


class Reg_CNNx16_n_Ly_ONED(CNNx16_BaseReg):  # noqa: N801
    TSCNT_MAX:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 4, 'doc': 'Maximum Time Slot CountMaximum timeslot count register. When CNNx16_n_Ly_POST.tsen is set, this valuedetermines the number of timeslots required to output data that has beengenerated in parallel by the CNNx16_n processors. This count is used for passthrough, 1x1, elementwise, and mask sharing (CNNx16_n_Ly_LCTRL0.mslave andCNNx16_n_Ly_LCTRL0.sslave) operations.'})  # noqa: E501
    ONED_SAD:int = Field(default=0, json_schema_extra={'bit': 4, 'bits': 4, 'doc': 'One Dimensional Convolution Start Mask AddressOne dimensional convolution start mask address (offset within 9-byte mask width)used in conjunction with the mask start address (CNNx16_n_Ly_MCNT.mcnt_sad) todetermine that 1D convolution mask starting address.'})  # noqa: E501
    ONED_WIDTH:int = Field(default=0, json_schema_extra={'bit': 8, 'bits': 4, 'doc': 'One Dimensional Convolution Mask WidthOne dimensional convolution mask width (0-9 are valid values). One based valuewith a width > 0 enabling 1D convolution operation.'})  # noqa: E501
    ONED_EN:int = Field(default=0, json_schema_extra={'bit': 12, 'bits': 1, 'doc': 'One Dimensional Processing EnableEnables 1D input data processing. If the CNNx16_n_Ly_RCNT.rcnt_max field is non-zero, the row count is used to index the input image; otherwise, the column count,CNNx16_n_Ly_CCNT.ccnt_max, value is used.'})  # noqa: E501
    EWISE_EN:int = Field(default=0, json_schema_extra={'bit': 13, 'bits': 1, 'doc': 'Element-Wise EnableSet this field to 1 to enable the element-wise operations. Prior to enabling element-wise operations, both the CNNx16_n_Ly_ONED.tscnt_max field and theCNNx16_n_Ly_POST.ts_en fields must be set.0: Element-wise operation disabled1: Enable element-wise operation'})  # noqa: E501
    EWISE_FUNC:int = Field(default=0, json_schema_extra={'bit': 14, 'bits': 2, 'doc': 'Element-Wise Function SelectSelects the element-wise function performed on the input data if .0b00: Subtract0b01: Add0b10: Bitwise OR0b11: Bitwise XOR'})  # noqa: E501
    PREPOOL:int = Field(default=0, json_schema_extra={'bit': 16, 'bits': 1, 'doc': 'Pre-Pooling of Input DataSet this field to 1 to enable pre-pooling of the input data prior to the element-wiseoperation selected with CNNx16_n_Ly_ONED.ewise_func.0: Input data is not pre-pooled prior to the element-wise function.1: Pre-pooling of input data prior to element-wise operation enabled'})  # noqa: E501
    TWO_D_CONV:int = Field(default=0, json_schema_extra={'bit': 17, 'bits': 1, 'doc': '2D Convolution of Element-Wise ResultSet this field to 1 to enable 2D convolution of the element-wise result. Standard 2Dprocessing applies.0: 2D convolution disabled1: Enable 2D convolution'})  # noqa: E501
    EWISE_CNT:int = Field(default=0, json_schema_extra={'bit': 18, 'bits': 4, 'doc': 'Element-Wise Channel CountDetermines the number of element-wise channels to be processed.0: 1 channel1: 2 channels2: 3 channels...14: 15 channels15: 16 channels'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 22, 'bits': 10, 'doc': 'Reserved nan'})  # noqa: E501


class Reg_CNNx16_n_Ly_PCCNT(CNNx16_BaseReg):  # noqa: N801
    PCCNT_MAX:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 4, 'doc': 'Pool Column Count MaxWhen the CNN is enabled with one of the CNNx16_n_CTRL.pool_en (global) orCNNx16_n_Ly_LCTRL0.pool_en (per layer) bits set, these bits determine the perlayer pool column count to be processed. Processing begins with pool column 0 andcompletes when the processing of the pooling column determined by pccnt_max iscomplete, or the effective pool column count exceeds the image column countspecified in CNNx16_n_Ly_CCNT.ccnt_max. These count values are added to thecolumn count to determine the effective address of the pooled data.'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 4, 'bits': 28, 'doc': 'Reserved'})  # noqa: E501


class Reg_CNNx16_n_Ly_POST(CNNx16_BaseReg):  # noqa: N801
    BPTR_SAD:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 12, 'doc': 'Bias Pointer Start AddressBias register file address byte counter increments once each mask countincrement.Note: The x16 Bias values can be enabled and used independently across the fouroutput processors.'})  # noqa: E501
    BPTR_EN:int = Field(default=0, json_schema_extra={'bit': 12, 'bits': 1, 'doc': 'Bias EnableThis field enables the addition of a scaled bias, stored in each bias location, tothe result of the convolution. Bias values are automatically scaled by a shift leftof seven bits in hardware.'})  # noqa: E501
    SCALE_REG:int = Field(default=0, json_schema_extra={'bit': 13, 'bits': 4, 'doc': 'Scale Shift NumberThis field sets the number of bits to shift the pre-activation sum-of-productsresult of the convolution. Valid values are 0 to 16-bits, and the direction of theshift is controlled by CNNx16_n_Ly_POST.scale_shift.'})  # noqa: E501
    SCALE_SHFT:int = Field(default=0, json_schema_extra={'bit': 17, 'bits': 1, 'doc': 'Scale Shift ControlThis bit selects the shift direction of the pre-activation sum-of-products result ofthe convolution.0: Left Shift1: Right Shift'})  # noqa: E501
    XPMP_CNT:int = Field(default=0, json_schema_extra={'bit': 18, 'bits': 4, 'doc': 'Expanded Multi-Pass (MP) CountAdds 4 additional bits to the MP counter for flattening (MLP) operations. Thisfield is only used when the CNNx16_n_Ly_POST.flatten_en bit is set to 1. Thisfield is appended to the CNNx16_n_Ly_LCTRL1.inpchexp bits and makes up the 4most significant bits of the count.'})  # noqa: E501
    MASK_SIZE:int = Field(default=0, json_schema_extra={'bit': 22, 'bits': 2, 'doc': 'Mask Size SelectionThis field determines the mask size multiplied with each 8-bit data value in theconvolution.0b00: 8-bits0b01: 1-bit0b10: 2-bits0b11: 4-bits'})  # noqa: E501
    TS_EN:int = Field(default=0, json_schema_extra={'bit': 24, 'bits': 1, 'doc': 'Timeslot Mode EnableThis bit is used to enable the timeslot counter. When enabled, the number oftimeslots programmed into the timeslot counter,CNNx16_n_Ly_ONED.tscnt_max, are added to each output channel slot. Thetimeslot counter allows pass through, 1×1, and elementwise values calculated inparallel to be written sequentially to memory.0: Timeslot Mode Disabled1: Timeslot Mode Enabled'})  # noqa: E501
    ONEXONE_EN:int = Field(default=0, json_schema_extra={'bit': 25, 'bits': 1, 'doc': 'Pass-Thru/1×1 Convolution Mode EnableThis bit forces all sixteen processors in the CNNx16_n to either directly passthrough the result of the pooling logic or compute a 1 data byte by 1 byte(1,2,4,8 bit) weight product. Control of a pass through or 1×1 convolution ismade for each of the 16 processors using the CNNx16_n_Ly_EN.mask_en controlfield.0: Pass-Thru Enabled1: 1×1 Enabled'})  # noqa: E501
    OUT_ABS:int = Field(default=0, json_schema_extra={'bit': 26, 'bits': 1, 'doc': 'Absolute Value Output EnableConvert the scaled, quantized convolution output to an absolute value. This bit,along with the activation enable bit, CNNx16_n_Ly_LCTRL0.act_en) determinethe final processing operation prior to writing the out channel to memory.0: Output is not converted to an absolute value1: Output is converted to an absolute valueNote: The CNNx16_n_Ly_POST.out_abs has priority over the activation enablebit, CNNx16_n_Ly_LCTRL0.act_en.'})  # noqa: E501
    FLATTEN_EN:int = Field(default=0, json_schema_extra={'bit': 27, 'bits': 1, 'doc': 'Flatten EnableEnables flattening all of the input channel data supporting a series of1×1 convolutions emulating a fully connected network. Setting this bit forces theuse of an extended multi-pass count allowing for up to 256 neurons. This bit isused in conjunction with the CNNx16_n_Ly_LCTRL1.inpchexp,CNNx16_n_Ly_POST.xpmp_cnt, and CNNx16_n_Ly_POST.onexone_en fields toenable Multi-Layer Processing.'})  # noqa: E501
    DECONV_EN:int = Field(default=0, json_schema_extra={'bit': 28, 'bits': 1, 'doc': 'Deconvolution EnableVirtually expands the input image size by adding a 0 byte after each actual inputdata byte is shifted into the TRAM, and a row of 0s following each row of columnexpanded input data is shifted into the TRAM. The receptive field remains at 3×3scanned across the expanded data.0: Deconvolution disabled1: Deconvolution enabled'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 29, 'bits': 3, 'doc': 'Reserved'})  # noqa: E501


class Reg_CNNx16_n_Ly_PRCNT(CNNx16_BaseReg):  # noqa: N801
    PRCNT_MAX:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 4, 'doc': 'Pool Row Count MaxWhen the CNN is enabled with one of the CNNx16_n_CTRL.pool_en (global) orCNNx16_n_Ly_LCTRL0.pool_en (per layer) bits set, this field determines the perlayer pool row count to be processed. Processing begins with pool row 0 andcompletes when the processing of the pooling row determined by prcnt_max iscomplete, or the effective pool row count exceeds the image row count specified inCNNx16_n_Ly_RCNT.rcnt_max. These count values are added to the row count todetermine the effective address of the pooled data.'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 4, 'bits': 28, 'doc': 'Reserved'})  # noqa: E501


class Reg_CNNx16_n_Ly_RCNT(CNNx16_BaseReg):  # noqa: N801
    RCNT_MAX:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 10, 'doc': 'Layer Row Count MaximumThis field sets the maximum row count to be processed. Processing begins with row0 and completes when the processing of the row determined by this field iscomplete.Set this field to include two times the CNNx16_n_Ly_RCNT.rcnt_pad value.aavvRROO_ mmVVtt = (2 × aavvRROO_OOVVaa) + RROOmmvvVVaa VVff aaVVwwvvNote: The value programmed into this field is the effective image row value,including pad, but excluding rows not processed due to stride restrictions.'})  # noqa: E501
    RESERVED2:int = Field(default=0, json_schema_extra={'bit': 10, 'bits': 6, 'doc': 'Reserved'})  # noqa: E501
    RCNT_PAD:int = Field(default=0, json_schema_extra={'bit': 16, 'bits': 2, 'doc': 'Pad RowsThis field sets the number of pad rows included at the beginning and end of theframe.Note: The CNNx16_n_Ly_RCNT.rcnt_max is inclusive of two times this value, as thesame number of pad rows are included at the beginning and end of the frame.'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 18, 'bits': 14, 'doc': 'Reserved nan'})  # noqa: E501


class Reg_CNNx16_n_Ly_RPTR_BASE(CNNx16_BaseReg):  # noqa: N801
    RPTR_BASE:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 17, 'doc': 'Read Pointer Base AddressWhen the CNN is enabled, this per layer register sets the CNN convolution resultSRAM read pointer base address. The base address can be set to any location inthe SRAM dedicated to a specific CNN input channel processor.Note: This field is limited to the 8,192 bytes of SRAM in the MAX78000. Do notwrite values greater than the memory available.'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 17, 'bits': 15, 'doc': 'Reserved'})  # noqa: E501


class Reg_CNNx16_n_Ly_STRIDE(CNNx16_BaseReg):  # noqa: N801
    STRIDE:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 2, 'doc': 'StrideThis field determines the stride length across and down the image. Processingbegins with row 0 and column 0. A stride of one is applied until the top row or leftcolumn of the receptive field lands in the unpadded image. At that point, the stridevalue programmed through the APB interface is applied to the column and/or rowof the field in the unpadded image. The stride is applied as long as the field remainswithin the bounds of the padded image.'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 2, 'bits': 30, 'doc': 'Reserved'})  # noqa: E501


class Reg_CNNx16_n_Ly_TPTR(CNNx16_BaseReg):  # noqa: N801
    TPTR_MAX:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 11, 'doc': 'TRAM Max AddressThis field determines the rollover point of the TRAM address pointer. This field,tptr_max, is used together with the TRAM address pointer start address value(CNNx16_n_Ly_TPTR.tptr_sad) to reflect the usable input image row size,including pad.'})  # noqa: E501
    RESERVED2:int = Field(default=0, json_schema_extra={'bit': 11, 'bits': 5, 'doc': 'Reserved'})  # noqa: E501
    TPTR_SAD:int = Field(default=0, json_schema_extra={'bit': 16, 'bits': 11, 'doc': 'TRAM Start AddressThis field determines the per layer TRAM pointer start address for initialprocessing and rollover events.'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 27, 'bits': 5, 'doc': 'Reserved'})  # noqa: E501


class Reg_CNNx16_n_Ly_WPTR_BASE(CNNx16_BaseReg):  # noqa: N801
    WPTR_BASE:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 17, 'doc': 'Write Pointer BaseThis per layer register sets the CNN convolution result SRAM write pointer baseaddress. The base address can be set to any location in any data SRAM in the CNN.Bit 16 allows the write pointer to not point to any SRAM.'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 17, 'bits': 15, 'doc': 'Reserved'})  # noqa: E501


class Reg_CNNx16_n_Ly_WPTR_CHOFF(CNNx16_BaseReg):  # noqa: N801
    WPTR_CHOFF:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 17, 'doc': 'Write Pointer Multi-Pass Offset nanWhen the CNN is enabled with the CNNx16_n_Ly_EN.mask_en bit set and aprogrammed maximum mask counter value(CNNx16_n_Ly_MCNT.mcnt_max  CNNx16_n_Ly_MCNT.mcnt_sad) that is greaterthan the maximum number of available processors programmed into theCNNx16_n_Ly_LCTRL1.xpch_max register (output channel multi-pass is enabled),the rounded mask counter value is divided by the CNNx16_n_Ly_LCTRL1.xpch_maxvalue and multiplied by the CNNx16_n_Ly_WPTR_CHOFF.wptr_choff to create amulti-pass channel offset value. During a convolution result SRAM data write, thismulti-pass channel offset value is added to the SRAM write address pointer. Thisoffset can be used to determine SRAM write addresses based on a multi-pass countvalue. The offset can be set to any location in any data SRAM in the CNN.'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 17, 'bits': 15, 'doc': 'Reserved nan'})  # noqa: E501


class Reg_CNNx16_n_Ly_WPTR_MOFF(CNNx16_BaseReg):  # noqa: N801
    WPTR_MOFF:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 17, 'doc': 'Write Pointer Mask OffsetWhen the CNN is enabled with the CNNx16_n_Ly_EN.mask_en bit set and a maskcount value loaded into the CNNx16_n_Ly_MCNT.mcnt_max register that is greaterthan the CNNx16_n_Ly_MCNT.mcnt_sad value, this per layer register sets the CNNconvolution result SRAM write pointer mask count offset value. During aconvolution result, SRAM data write, this mask count offset value is multiplied bythe mask counter and added to the SRAM write address pointer. This offset can beused to determine SRAM write addresses based on a mask count. The offset can beset to any location in any data SRAM in the CNN.'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 17, 'bits': 15, 'doc': 'Reserved'})  # noqa: E501


class Reg_CNNx16_n_Ly_WPTR_TOFF(CNNx16_BaseReg):  # noqa: N801
    WPTR_TOFF:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 17, 'doc': 'Write Pointer Timeslot OffsetWhen the CNN is enabled with the CNNx16_n_Ly_POST.ts_en bit set and a non-zerotime slot count value loaded into CNNx16_n_Ly_ONED.tscnt_max field, this perlayer register sets the CNN convolution result SRAM write pointer timeslot offsetvalue. During a convolution result, SRAM data write, this timeslot offset value ismultiplied by the timeslot counter and added to the SRAM write address pointer.This offset can be used to determine SRAM write addresses based on the timeslot.The offset can be set to any location in any data SRAM in the CNN.'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 17, 'bits': 15, 'doc': 'Reserved'})  # noqa: E501


class Reg_CNNx16_n_MLAT(CNNx16_BaseReg):  # noqa: N801
    MLATDAT:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 32, 'doc': 'Packed Channel DataThis register accumulates four bytes of output channel data to be read by thehost MCU. Channel data is usually stored in the memory depth of a single byteof the SRAM data word, and four channels make up the memory data word. TheMlator automatically fetches four bytes in the memory depth to generate apacked 4-byte word for efficient reading by the MCU. The target channel isselected using the CNNx16_n_CTRL.mlatch_sel bits, and the target SRAMaddress is determined byte the CNNx16_n_Ly_WPTR_BASE register. Setting theCNNx16_n_CTRL.mlat_ld bit to 1 loads theCNNx16_n_Ly_WPTR_BASE.wptr_base value into the address counter andinitiates the read of the first 4 bytes of channel data. When the current 4 bytesare accumulated in the CNNx16_n_MLAT.mlatdat is read through the APBinterface, the next 4 bytes are read in sequence. Reading continues until haltedby the MCU. Four clock cycles are required after the completion of the last reador setting of the CNNx16_n_CTRL.mlat_ld bit to 1 to accumulate the 4 bytes ofdata. It is up to the MCU software to ensure there is adequate time betweenreads to accumulate the 4 bytes of data.'})  # noqa: E501


class Reg_CNNx16_n_SRAM(CNNx16_BaseReg):  # noqa: N801
    EXTACC:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 1, 'doc': 'SRAM Extended Access Time EnableSet this bit to 1 to enable SRAM access time maximum. Enabling longer SRAMaccess time increases the power consumption of the SRAM.0: SRAM Power Optimized, reduced performance1: SRAM Extended Access, higher powerNote, this setting can be used to extend access time but force the memories toconsume additional power when active. This bit applies to all SRAMs in theCNNx16_n processor.'})  # noqa: E501
    RMARGIN_EN:int = Field(default=1, json_schema_extra={'bit': 1, 'bits': 1, 'doc': 'SRAM Read Margin EnableSet this field to 1 to use the SRAM Access Time setting in theCNNx16_n_CTRL.ram_acc_time field.0: SRAM access time is set by the hardware1: SRAM access time is controlled using the CNNx16_n_CTRL.ram_acc_timefield.'})  # noqa: E501
    RMARGIN:int = Field(default=3, json_schema_extra={'bit': 2, 'bits': 4, 'doc': 'SRAM Read MarginWhen CNNx16_n_SRAM.rm_en is set, this field determines the length of thememory access time.0b0000: Slowest SRAM access time0b0001:0b0010:0b0011: Fastest SRAM access time (Reset Default)0b0100-0b1111: ReservedNote: The value of this field has no effect unless the CNNx16_n_CTRL.rm_en fieldis set to 1.'})  # noqa: E501
    RA:int = Field(default=0, json_schema_extra={'bit': 6, 'bits': 2, 'doc': 'Read Assist VoltageThis field controls the Read Assist value for the SRAM bit lines.0: VDD1: VDD – 20mV2: VDD – 40mV3: VDD – 60mVThese bits determine the WL underdrive (Read Assist) value. ra[1:0] = 00 limitsthe WL voltage to VDD, ra[1:0] = 01 limits the WL to VDD-20mV, ra[1:0] = 10limits WL to VDD-40mV, and ra[1:0] = 11 limits the WL voltage to VDD-60mV.'})  # noqa: E501
    WNEG_VOL:int = Field(default=0, json_schema_extra={'bit': 8, 'bits': 2, 'doc': 'Write Negative VoltageThis field sets the SRAM negative voltage level applied to the bit lines. This fieldis only used when the CNNx16_n_SRAM.wneg_en field is set to 1.0: VDD – 80mV1: VDD – 120mV2: VDD – 180mV3: VDD – 220mV'})  # noqa: E501
    WNEG_EN:int = Field(default=0, json_schema_extra={'bit': 10, 'bits': 1, 'doc': 'Write Negative Voltage EnableThis bit enables the CNNx16_n_SRAM.wneg_vol. If this field is 0, the systemcontrols the negative voltage applied to the bit lines.0: Write negative voltage time is controlled by the system.1: Write negative voltage time is controlled by the setting in theCNNx16_n_SRAM.wneg_vol field.'})  # noqa: E501
    WPULSE:int = Field(default=0, json_schema_extra={'bit': 11, 'bits': 3, 'doc': 'Write Pulse WidthThis field determines the bit line pulse width applied to the memory duringwrites.0b000: Use the minimum bit line pulse width0b111: Use the maximum bit line pulse widthNote: Values of wpulse between 0 and 7 incrementally set the bit line pulse widthbetween the minimum and maximum values.'})  # noqa: E501
    RESERVED7:int = Field(default=0, json_schema_extra={'bit': 14, 'bits': 1, 'doc': 'Reserved'})  # noqa: E501
    DS:int = Field(default=0, json_schema_extra={'bit': 15, 'bits': 1, 'doc': "CNNx16_n Memory Deep Sleep EnableSet this field to 1 to put the CNNx16_n's SRAM, TRAM, MRAM, and bias memoryinto a deep sleep state. In deep sleep, the memories contents are retained, butperipheral logic is powered down, and the memory cannot be accessed. Allmemory outputs are set to output 0.0: Memories are not in deep sleep state.1: Put the CNNx16_n's memories into a deep sleep state.Note: This field has no effect If the CNNx16_n_SRAM.pd field is set to 1(memories powered down)."})  # noqa: E501
    PD:int = Field(default=0, json_schema_extra={'bit': 16, 'bits': 1, 'doc': "CNNx16_n Memory Power Down EnableSet this field to 1 to put the CNNx16_n's SRAM, TRAM, MRAM, and bias memoryinto a power-down mode. All memory contents are lost in power downstate.0: CNNx16 memories are not powered down1: Power down the CNNx16_n's memories"})  # noqa: E501
    RESERVED4:int = Field(default=0, json_schema_extra={'bit': 17, 'bits': 2, 'doc': 'Reserved'})  # noqa: E501
    LSDRAM:int = Field(default=0, json_schema_extra={'bit': 19, 'bits': 1, 'doc': 'Data RAM Light SleepSetting this bit forces the Data RAMs into light sleep when the SRAM is notselected by the CNN system or the APB.'})  # noqa: E501
    LSMRAM:int = Field(default=0, json_schema_extra={'bit': 20, 'bits': 1, 'doc': 'MRAM Light SleepSetting this bit forces the MRAMs into light sleep when the MRAM is notselected by the CNN system or the APB.'})  # noqa: E501
    LSTRAM:int = Field(default=0, json_schema_extra={'bit': 21, 'bits': 1, 'doc': 'TRAM Light SleepSetting this bit forces the TRAMs into light sleep when the TRAM is not selectedby the CNN system or the APB.'})  # noqa: E501
    LSBRAM:int = Field(default=0, json_schema_extra={'bit': 22, 'bits': 1, 'doc': 'Bias Memory Light SleepSetting this bit forces the bias memory into light sleep when the bias memory isnot selected by the CNN system or the APB.'})  # noqa: E501


class Reg_CNNx16_n_Sz_FBUF(CNNx16_BaseReg):  # noqa: N801
    FBUF_MAX:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 17, 'doc': 'Per Stream Circular Buffer Max CountWhen streaming is enabled (CNNx16_n_CTRL.stream_en = 1), the per-layerSRAM read pointer base register value (CNNx16_n_Ly_RPTR_BASE.rptr_base) issubtracted from the internally generated read pointer counter and compared tothe value stored in this field. When the adjusted read pointer is equal to thisfield, the rollover point is detected, and the CNN_FIF0_STAT.rptr_base value isloaded into the read pointer counter.'})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 17, 'bits': 15, 'doc': 'Reserved'})  # noqa: E501


class Reg_CNNx16_n_Sz_STRM0(CNNx16_BaseReg):  # noqa: N801
    STRM_ISVAL:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 14, 'doc': "Per-Stream Start CountThe per-stream start count is based on the previous layer's tptr_inc count. Whenstreaming is enabled (CNNx16_n_CTRL.stream_en = 1), each time a byte in theprior or input layer is written into the TRAM, the internal stream start counter isincremented. When the counter reaches this field's value,CNNx16_n_Sz_STRM0.strm_isval, processing of the current layer is enabled. Thismechanism allows adequate receptive field data to be accumulated beforeconvolution processing in a stream layer begins. Once the counter value reachesthis field's value, CNNx16_n_Sz_STRM0.strm_isval, counting is halted, and theterminal count value remains in the counter."})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 14, 'bits': 18, 'doc': 'Reserved nan'})  # noqa: E501


class Reg_CNNx16_n_Sz_STRM1(CNNx16_BaseReg):  # noqa: N801
    STRM_INVOL:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 4, 'doc': "Per Stream Input Volume OffsetThis field is based on the stream count. When streaming is enabled(CNNx16_n_CTRL.stream_en = 1), this field determines the input volume offsetapplied to each stream. The value programmed into this field is multiplied by thestream count to calculate the stream count input volume offset.The CNN supports up to 16 independent input volumes. The input volumes aresplit between multi-pass and streaming. The streaming input volume offsetallows the streaming input volume selection to 'skip over' the input volumesused for multi-pass processing."})  # noqa: E501
    STRM_DSVAL1:int = Field(default=0, json_schema_extra={'bit': 4, 'bits': 5, 'doc': "Per Stream In-Row Delta CountThis field is based on the previous layer's tptr_inc count. When streaming isenabled (CNNx16_n_CTRL.stream_en = 1), this field determines the number ofbytes written into the TRAM of the prior layer between active processing of thecurrent layer. This APB accessible R/W register is used to set the processingcadence of each column across an image row. When the internal delta countcounter reaches the value stored in this field,CNNx16_n_Sz_STRM1.strm_dsval1, processing of the current layer is enabledfor one stride (input pooling plus output channel generation), and the deltacounter is reset to zero."})  # noqa: E501
    RESERVED2:int = Field(default=0, json_schema_extra={'bit': 9, 'bits': 7, 'doc': 'Reserved'})  # noqa: E501
    STRM_DSVAL2:int = Field(default=0, json_schema_extra={'bit': 16, 'bits': 12, 'doc': "Per Stream Multi-Row Delta CountThis field is based on the previous layer's CNNx16_n_Ly_TPTR count. Whenstreaming is enabled (CNNx16_n_CTRL.stream_en = 1), this field determines thenumber of bytes written into the TRAM of the prior layer between activeprocessing of the current layer. This APB accessible R/W register is used to setthe processing cadence across an image row boundary. When the internal deltacount counter reaches the value stored in this field,CNNx16_n_Sz_STRM1.strm_dsval2, processing of the current layer is enabledfor one stride (input pooling plus output channel generation), and the deltacounter is reset to zero."})  # noqa: E501
    RESERVED0:int = Field(default=0, json_schema_extra={'bit': 28, 'bits': 4, 'doc': 'Reserved'})  # noqa: E501


class Reg_CNNx16_n_TEST(CNNx16_BaseReg):  # noqa: N801
    SBISTRUN:int = Field(default=0, json_schema_extra={'bit': 0, 'bits': 1, 'doc': 'SRAM BIST RunSetting this bit to 1 will run the BIST for all SRAM instances in the CNNx16_nprocessor. The BIST will run to completion, and the status is reported in:• CNNx16_n_TEST.sallbdone• CNNx16_n_TEST.sallbfail• CNNx16_n_TEST.bistdone• CNNx16_n_TEST.bistfailIf more detailed status is required from the BIST execution, theCNNx16_n_TEST.bistsel field can be used to extract status from an individualBIST controller.This bit must be written to 0 to reset the BIST operation prior to writing it to 1.Writing 1 consecutively does not run the BIST again.'})  # noqa: E501
    SRAMZ:int = Field(default=0, json_schema_extra={'bit': 1, 'bits': 1, 'doc': 'SRAM ZeroizeSetting this bit to 1 will force the BIST to initialize all SRAM memory locations to0. Completion status can be found in the• CNNx16_n_TEST.sallzdone and• CNNx16_n_TEST.zerodoneThis bit must be written to 0 to reset the operation prior to writing it to 1.Writing 1 consecutively does not run the memory zeroization again.'})  # noqa: E501
    MBISTRUN:int = Field(default=0, json_schema_extra={'bit': 2, 'bits': 1, 'doc': 'MRAM BIST RunSetting this bit to 1 will run the BIST for all MRAM instances in the CNNx16_nprocessor. The BIST will run to completion, and the status is reported in:• CNNx16_n_TEST.mallbdone• CNNx16_n_TEST.mallbfail• CNNx16_n_TEST.bistdone• CNNx16_n_TEST.bistfailIf more detailed status is required from the BIST execution, theCNNx16_n_TEST.bistsel field can be used to extract status from an individualBIST controller.This bit must be written to 0 to reset the BIST operation prior to writing it to 1.Writing 1 consecutively does not run the BIST again.'})  # noqa: E501
    MRAMZ:int = Field(default=0, json_schema_extra={'bit': 3, 'bits': 1, 'doc': 'MRAM ZeroizeSetting this bit to 1 will force the BIST to initialize all MRAM memory locations to0. Completion status can be found in the:• CNNx16_n_TEST.mallzdone and• CNNx16_n_TEST.zerodonThis bit must be written to 0 to reset the operation prior to writing it to 1.Writing 1 consecutively does not run the memory zeroization again.'})  # noqa: E501
    TBISTRUN:int = Field(default=0, json_schema_extra={'bit': 4, 'bits': 1, 'doc': 'TRAM BIST RunSetting this bit to 1 will run the BIST for all TRAM instances in the CNNx16_nprocessor. The BIST will run to completion and the status reported in:• CNNx16_n_TEST.tallbdone• CNNx16_n_TEST.tallbfail• CNNx16_n_TEST.bistdone• CNNx16_n_TEST.bistfailIf more detailed status is required from the BIST execution, theCNNx16_n_TEST.bistsel field can be used to extract status from an individualBIST controller.This bit must be written to 0 to reset the BIST operation prior to writing it to 1.Writing 1 consecutively does not run the BIST again.'})  # noqa: E501
    TRAMZ:int = Field(default=0, json_schema_extra={'bit': 5, 'bits': 1, 'doc': 'TRAM ZeroizeSetting this bit to 1 will force the BIST to initialize all TRAM memory locationsto 0. Completion status can be found in the CNNx16_n_TEST.tallzdone andCNNx16_n_TEST.zerodone status bit. This bit is edge-triggered and must betoggled from zero to one to run the BIST.'})  # noqa: E501
    BBISTRUN:int = Field(default=0, json_schema_extra={'bit': 6, 'bits': 1, 'doc': 'Run Bias Memory BISTSetting this bit to 1 will run the BIST for all bias memory instances in theCNNx16_n processor. The BIST will run to completion, and the status is reportedin:• CNNx16_n_TEST.ballbdone• CNNx16_n_TEST.ballbfail• CNNx16_n_TEST.bistdone• CNNx16_n_TEST.bistfailIf more detailed status is required from the BIST execution, theCNNx16_n_TEST.bistsel field can be used to extract status from an individualBIST controller.This bit must be written to 0 to reset the BIST operation prior to writing it to 1.Writing 1 consecutively does not run the BIST again.'})  # noqa: E501
    BRAMZ:int = Field(default=0, json_schema_extra={'bit': 7, 'bits': 1, 'doc': 'BIAS Memory ZeroizeSetting this bit to 1 will force the BIST to initialize all Bias memory locations to 0.Completion status can be found in the:• CNNx16_n_TEST.sallzdone and• CNNx16_n_TEST.zerodoneThis bit must be written to 0 to reset the operation prior to writing it to 1.Writing 1 consecutively does not run the memory zeroization again.'})  # noqa: E501
    BISTSEL:int = Field(default=0, json_schema_extra={'bit': 8, 'bits': 6, 'doc': 'BIST Controller Status SelectionThe bits select an individual BIST controller status to be reported in theassociated 32 bits of the memory read data bus. CNNx16_n_TEST.bistsel[5]selects the SRAM or bias memory BIST group controller statuses, with:CNNx16_n_TEST.bistsel[2:0] selecting the individual SRAM/bias memoryinstance with CNNx16_n_TEST.bistsel[2:0] = 100 selecting the bias memoryInstance. Control bit CNNx16_n_TEST.bistsel[4] selects the MRAM BISTcontroller statuses, with CNNx16_n_TEST.bistsel[3:0] selecting the individualRAM instance. Control bit CNNx16_n_TEST.bistsel[3] selects the TRAM BISTcontroller statuses, with CNNx16_n_TEST.bistsel[3:0] selecting the individualRAM instance.'})  # noqa: E501
    SALLBFAIL:int = Field(default=0, json_schema_extra={'bit': 14, 'bits': 1, 'doc': 'SRAM BIST ResultWhen an SRAM BIST operation was started by setting CNNx16_n_TEST.sbistrunto one, and the operation is completed by hardware (CNNx16_n_TEST.sallbdoneis set by hardware to 1), this field indicates the result of the SRAM BISToperation. Reset this field by writing a 0 to the CNNx16_n_TEST.sbistrun field0: SRAM BIST Passed1: SRAM BIST Failed, indicating an error occurred in one of the four SRAMs.Note: This field is only valid after an SRAM BIST operation has started andcompleted (CNNx16_n_TEST.sallbdone = 1).'})  # noqa: E501
    MALLBFAIL:int = Field(default=0, json_schema_extra={'bit': 15, 'bits': 1, 'doc': 'MRAM BIST ResultWhen a MRAM BIST operation was started by setting CNNx16_n_TEST.mbistrunto one, and the operation is completed by hardware(CNNx16_n_TEST.mallbdone is set by hardware to 1), this field indicates theresult of the SRAM BIST operation. Reset this field by writing a 0 to theCNNx16_n_TEST.mbistrun field.0: MRAM BIST Passed1: MRAM BIST Failed, indicating an error occurred in one of the 16 MRAMs.Note: This field is only valid after a MRAM BIST operation has started andcompleted (CNNx16_n_TEST.mallbdone = 1).'})  # noqa: E501
    TALLBFAIL:int = Field(default=0, json_schema_extra={'bit': 16, 'bits': 1, 'doc': 'SRAM BIST ResultWhen a TRAM BIST operation was started by setting CNNx16_n_TEST.tbistrunbit to 1, and the operation is completed by hardware (CNNx16_n_TEST.tallbdoneis set to 1 by hardware), this field indicates the result of the TRAM BISToperation. Reset this field by writing a 0 to the CNNx16_n_TEST.tbistrun field.0: TRAM BIST Passed1: TRAM BIST Failed, indicating an error occurred in one of the four SRAMs.Note: This field is only valid after a TRAM BIST operation has started andcompletes (CNNx16_n_TEST.tallbdone = 1).'})  # noqa: E501
    BALLBFAIL:int = Field(default=0, json_schema_extra={'bit': 17, 'bits': 1, 'doc': 'BRAM BIST ResultWhen a BRAM BIST operation was started by setting CNNx16_n_TEST.bbistrunbit to 1, and the operation is completed by hardware(CNNx16_n_TEST.ballbdone is set to 1 by hardware), this field indicates theresult of the BRAM BIST operation. Reset this field by writing a 0 to theCNNx16_n_TEST.bbistrun field.0: BRAM BIST Passed1: BRAM BIST Failed, indicating an error occurred in one of the BRAMs.Note: This field is only valid after a BRAM BIST operation has started andcompleted (CNNx16_n_TEST.ballbdone = 1).'})  # noqa: E501
    SALLBDONE:int = Field(default=0, json_schema_extra={'bit': 18, 'bits': 1, 'doc': 'SRAM BIST CompleteThis field indicates an SRAM BIST run is completed. This field is reset byhardware when software writes the CNNx16_n_TEST.sbistrun field to 0.1: SRAM BIST complete'})  # noqa: E501
    MALLBDONE:int = Field(default=0, json_schema_extra={'bit': 19, 'bits': 1, 'doc': 'MRAM BIST CompleteThis field indicates a MRAM BIST run is completed. This field is reset byhardware when software writes the CNNx16_n_TEST.mbistrun field to 0.1: MRAM BIST complete'})  # noqa: E501
    TALLBDONE:int = Field(default=0, json_schema_extra={'bit': 20, 'bits': 1, 'doc': 'TRAM BIST CompleteThis field indicates a TRAM BIST run is completed. This field is reset by hardwarewhen software writes the CNNx16_n_TEST.tbistrun field to 0.1: TRAM BIST complete'})  # noqa: E501
    BALLBDONE:int = Field(default=0, json_schema_extra={'bit': 21, 'bits': 1, 'doc': 'BRAM BIST CompleteThis field indicates a BRAM BIST run is completed. This field is reset by hardwarewhen software writes the CNNx16_n_TEST.bbistrun field to 0.1: BRAM BIST complete'})  # noqa: E501
    SALLZDONE:int = Field(default=0, json_schema_extra={'bit': 22, 'bits': 1, 'doc': 'SRAM Zeroization CompleteThis field indicates an SRAM zeroization is completed. This field is reset byhardware when software writes the CNNx16_n_TEST.szerorun field to 0.1: SRAM zeroization complete'})  # noqa: E501
    MALLZDONE:int = Field(default=0, json_schema_extra={'bit': 23, 'bits': 1, 'doc': 'MRAM Zeroization CompleteThis field indicates a MRAM zeroization is completed. This field is reset byhardware when software writes the CNNx16_n_TEST.mzerorun field to 0.1: MRAM zeroization complete'})  # noqa: E501
    TALLZDONE:int = Field(default=0, json_schema_extra={'bit': 24, 'bits': 1, 'doc': 'TRAM Zeroization CompleteThis field indicates a TRAM zeroization is completed. This field is reset byhardware when software writes the CNNx16_n_TEST.tzerorun field to 0.1: TRAM zeroization complete'})  # noqa: E501
    BALLZDONE:int = Field(default=0, json_schema_extra={'bit': 25, 'bits': 1, 'doc': 'BRAM Zeroization CompleteThis field indicates an SRAM zeroization is completed. This field is reset byhardware when software writes the CNNx16_n_TEST.bzerorun field to 0.1: BRAM zeroization complete'})  # noqa: E501
    BISTFAIL:int = Field(default=0, json_schema_extra={'bit': 26, 'bits': 1, 'doc': 'BIST Run Failure DetectedThis field is set to 1 by hardware if a BIST run operation was run and a BISTfailure occurred.This bit is read only. Clear this bit by setting each of the BIST run bits to 0.0: If the CNNx16_n_TEST.bistdone bit reads 1, this bit indicates no BIST failureswere detected.1: BIST failure detected'})  # noqa: E501
    BISTDONE:int = Field(default=0, json_schema_extra={'bit': 27, 'bits': 1, 'doc': 'BIST Run CompleteThis field is set to 1 by hardware when any of the BIST run bits are set to 1 bysoftware, and the hardware completes the BIST operation.This bit is read only. Clear this bit by setting each of the BIST run bits to 0.1: BIST operation complete'})  # noqa: E501
    ZERODONE:int = Field(default=0, json_schema_extra={'bit': 28, 'bits': 1, 'doc': 'BIST Zeroization CompleteThis field is set to 1 by hardware when any of the zero run bits are set to 1 bysoftware, and the hardware completes the zeroization.This bit is read only. Clear this bit by setting each of the zero run bits to 0.1: BIST zeroization complete'})  # noqa: E501

register_class_by_address = {
    0x0000: Reg_CNNx16_n_CTRL,
    0x0004: Reg_CNNx16_n_SRAM,
    0x0008: Reg_CNNx16_n_LCNT_MAX,
    0x000C: Reg_CNNx16_n_TEST,
    0x0010: Reg_CNNx16_n_Ly_RCNT,
    0x0090: Reg_CNNx16_n_Ly_CCNT,
    0x0110: Reg_CNNx16_n_Ly_ONED,
    0x0190: Reg_CNNx16_n_Ly_PRCNT,
    0x0210: Reg_CNNx16_n_Ly_PCCNT,
    0x0290: Reg_CNNx16_n_Ly_STRIDE,
    0x0310: Reg_CNNx16_n_Ly_WPTR_BASE,
    0x0390: Reg_CNNx16_n_Ly_WPTR_TOFF,
    0x0410: Reg_CNNx16_n_Ly_WPTR_MOFF,
    0x0490: Reg_CNNx16_n_Ly_WPTR_CHOFF,
    0x0510: Reg_CNNx16_n_Ly_RPTR_BASE,
    0x0590: Reg_CNNx16_n_Ly_LCTRL0,
    0x0610: Reg_CNNx16_n_Ly_MCNT,
    0x0690: Reg_CNNx16_n_Ly_TPTR,
    0x0710: Reg_CNNx16_n_Ly_EN,
    0x0790: Reg_CNNx16_n_Ly_POST,
    0x0810: Reg_CNNx16_n_Sz_STRM0,
    0x0890: Reg_CNNx16_n_Sz_STRM1,
    0x0910: Reg_CNNx16_n_Sz_FBUF,
    0x0990: Reg_CNNx16_n_IFRM,
    0x0A10: Reg_CNNx16_n_Ly_LCTRL1,
    0x1000: Reg_CNNx16_n_MLAT,
    }
