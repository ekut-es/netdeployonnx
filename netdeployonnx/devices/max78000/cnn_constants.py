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
registers = {
    "CNNx16_FIF0_CTRL":                  0x5000_0000, # 50000000
    "CNNx16_FIF0_STAT":                  0x5000_0004, # 50000004
    "CNNx16_FIFO_WR0":                   0x5000_0008, # 50000008
    "CNNx16_FIFO_WR1":                   0x5000_000C, # 5000000C
    "CNNx16_FIFO_WR2":                   0x5000_0010, # 50000010
    "CNNx16_FIFO_WR3":                   0x5000_0014, # 50000014
    "CNNx16_AOD_CTRL":                   0x5000_1000, # 50001000
    "CNNx16_0_CTRL":                  0x5010_0000, # 50100000
    "CNNx16_0_SRAM":                  0x5010_0004, # 50100004
    "CNNx16_0_LCNT_MAX":              0x5010_0008, # 50100008
    "CNNx16_0_TEST":                  0x5010_000C, # 5010000C
    "CNNx16_0_L0_RCNT":               0x5010_0010, # 50100010
    "CNNx16_0_L1_RCNT":               0x5010_0014, # 50100014
    "CNNx16_0_L2_RCNT":               0x5010_0018, # 50100018
    "CNNx16_0_L3_RCNT":               0x5010_001C, # 5010001C
    "CNNx16_0_L4_RCNT":               0x5010_0020, # 50100020
    "CNNx16_0_L5_RCNT":               0x5010_0024, # 50100024
    "CNNx16_0_L6_RCNT":               0x5010_0028, # 50100028
    "CNNx16_0_L7_RCNT":               0x5010_002C, # 5010002C
    "CNNx16_0_L8_RCNT":               0x5010_0030, # 50100030
    "CNNx16_0_L9_RCNT":               0x5010_0034, # 50100034
    "CNNx16_0_L10_RCNT":              0x5010_0038, # 50100038
    "CNNx16_0_L11_RCNT":              0x5010_003C, # 5010003C
    "CNNx16_0_L12_RCNT":              0x5010_0040, # 50100040
    "CNNx16_0_L13_RCNT":              0x5010_0044, # 50100044
    "CNNx16_0_L14_RCNT":              0x5010_0048, # 50100048
    "CNNx16_0_L15_RCNT":              0x5010_004C, # 5010004C
    "CNNx16_0_L0_CCNT":               0x5010_0090, # 50100090
    "CNNx16_0_L1_CCNT":               0x5010_0094, # 50100094
    "CNNx16_0_L2_CCNT":               0x5010_0098, # 50100098
    "CNNx16_0_L3_CCNT":               0x5010_009C, # 5010009C
    "CNNx16_0_L4_CCNT":               0x5010_00A0, # 501000A0
    "CNNx16_0_L5_CCNT":               0x5010_00A4, # 501000A4
    "CNNx16_0_L6_CCNT":               0x5010_00A8, # 501000A8
    "CNNx16_0_L7_CCNT":               0x5010_00AC, # 501000AC
    "CNNx16_0_L8_CCNT":               0x5010_00B0, # 501000B0
    "CNNx16_0_L9_CCNT":               0x5010_00B4, # 501000B4
    "CNNx16_0_L10_CCNT":              0x5010_00B8, # 501000B8
    "CNNx16_0_L11_CCNT":              0x5010_00BC, # 501000BC
    "CNNx16_0_L12_CCNT":              0x5010_00C0, # 501000C0
    "CNNx16_0_L13_CCNT":              0x5010_00C4, # 501000C4
    "CNNx16_0_L14_CCNT":              0x5010_00C8, # 501000C8
    "CNNx16_0_L15_CCNT":              0x5010_00CC, # 501000CC
    "CNNx16_0_L0_ONED":               0x5010_0110, # 50100110
    "CNNx16_0_L1_ONED":               0x5010_0114, # 50100114
    "CNNx16_0_L2_ONED":               0x5010_0118, # 50100118
    "CNNx16_0_L3_ONED":               0x5010_011C, # 5010011C
    "CNNx16_0_L4_ONED":               0x5010_0120, # 50100120
    "CNNx16_0_L5_ONED":               0x5010_0124, # 50100124
    "CNNx16_0_L6_ONED":               0x5010_0128, # 50100128
    "CNNx16_0_L7_ONED":               0x5010_012C, # 5010012C
    "CNNx16_0_L8_ONED":               0x5010_0130, # 50100130
    "CNNx16_0_L9_ONED":               0x5010_0134, # 50100134
    "CNNx16_0_L10_ONED":              0x5010_0138, # 50100138
    "CNNx16_0_L11_ONED":              0x5010_013C, # 5010013C
    "CNNx16_0_L12_ONED":              0x5010_0140, # 50100140
    "CNNx16_0_L13_ONED":              0x5010_0144, # 50100144
    "CNNx16_0_L14_ONED":              0x5010_0148, # 50100148
    "CNNx16_0_L15_ONED":              0x5010_014C, # 5010014C
    "CNNx16_0_L0_PRCNT":              0x5010_0190, # 50100190
    "CNNx16_0_L1_PRCNT":              0x5010_0194, # 50100194
    "CNNx16_0_L2_PRCNT":              0x5010_0198, # 50100198
    "CNNx16_0_L3_PRCNT":              0x5010_019C, # 5010019C
    "CNNx16_0_L4_PRCNT":              0x5010_01A0, # 501001A0
    "CNNx16_0_L5_PRCNT":              0x5010_01A4, # 501001A4
    "CNNx16_0_L6_PRCNT":              0x5010_01A8, # 501001A8
    "CNNx16_0_L7_PRCNT":              0x5010_01AC, # 501001AC
    "CNNx16_0_L8_PRCNT":              0x5010_01B0, # 501001B0
    "CNNx16_0_L9_PRCNT":              0x5010_01B4, # 501001B4
    "CNNx16_0_L10_PRCNT":             0x5010_01B8, # 501001B8
    "CNNx16_0_L11_PRCNT":             0x5010_01BC, # 501001BC
    "CNNx16_0_L12_PRCNT":             0x5010_01C0, # 501001C0
    "CNNx16_0_L13_PRCNT":             0x5010_01C4, # 501001C4
    "CNNx16_0_L14_PRCNT":             0x5010_01C8, # 501001C8
    "CNNx16_0_L15_PRCNT":             0x5010_01CC, # 501001CC
    "CNNx16_0_L0_PCCNT":              0x5010_0210, # 50100210
    "CNNx16_0_L1_PCCNT":              0x5010_0214, # 50100214
    "CNNx16_0_L2_PCCNT":              0x5010_0218, # 50100218
    "CNNx16_0_L3_PCCNT":              0x5010_021C, # 5010021C
    "CNNx16_0_L4_PCCNT":              0x5010_0220, # 50100220
    "CNNx16_0_L5_PCCNT":              0x5010_0224, # 50100224
    "CNNx16_0_L6_PCCNT":              0x5010_0228, # 50100228
    "CNNx16_0_L7_PCCNT":              0x5010_022C, # 5010022C
    "CNNx16_0_L8_PCCNT":              0x5010_0230, # 50100230
    "CNNx16_0_L9_PCCNT":              0x5010_0234, # 50100234
    "CNNx16_0_L10_PCCNT":             0x5010_0238, # 50100238
    "CNNx16_0_L11_PCCNT":             0x5010_023C, # 5010023C
    "CNNx16_0_L12_PCCNT":             0x5010_0240, # 50100240
    "CNNx16_0_L13_PCCNT":             0x5010_0244, # 50100244
    "CNNx16_0_L14_PCCNT":             0x5010_0248, # 50100248
    "CNNx16_0_L15_PCCNT":             0x5010_024C, # 5010024C
    "CNNx16_0_L0_STRIDE":             0x5010_0290, # 50100290
    "CNNx16_0_L1_STRIDE":             0x5010_0294, # 50100294
    "CNNx16_0_L2_STRIDE":             0x5010_0298, # 50100298
    "CNNx16_0_L3_STRIDE":             0x5010_029C, # 5010029C
    "CNNx16_0_L4_STRIDE":             0x5010_02A0, # 501002A0
    "CNNx16_0_L5_STRIDE":             0x5010_02A4, # 501002A4
    "CNNx16_0_L6_STRIDE":             0x5010_02A8, # 501002A8
    "CNNx16_0_L7_STRIDE":             0x5010_02AC, # 501002AC
    "CNNx16_0_L8_STRIDE":             0x5010_02B0, # 501002B0
    "CNNx16_0_L9_STRIDE":             0x5010_02B4, # 501002B4
    "CNNx16_0_L10_STRIDE":            0x5010_02B8, # 501002B8
    "CNNx16_0_L11_STRIDE":            0x5010_02BC, # 501002BC
    "CNNx16_0_L12_STRIDE":            0x5010_02C0, # 501002C0
    "CNNx16_0_L13_STRIDE":            0x5010_02C4, # 501002C4
    "CNNx16_0_L14_STRIDE":            0x5010_02C8, # 501002C8
    "CNNx16_0_L15_STRIDE":            0x5010_02CC, # 501002CC
    "CNNx16_0_L0_WPTR_BASE":          0x5010_0310, # 50100310
    "CNNx16_0_L1_WPTR_BASE":          0x5010_0314, # 50100314
    "CNNx16_0_L2_WPTR_BASE":          0x5010_0318, # 50100318
    "CNNx16_0_L3_WPTR_BASE":          0x5010_031C, # 5010031C
    "CNNx16_0_L4_WPTR_BASE":          0x5010_0320, # 50100320
    "CNNx16_0_L5_WPTR_BASE":          0x5010_0324, # 50100324
    "CNNx16_0_L6_WPTR_BASE":          0x5010_0328, # 50100328
    "CNNx16_0_L7_WPTR_BASE":          0x5010_032C, # 5010032C
    "CNNx16_0_L8_WPTR_BASE":          0x5010_0330, # 50100330
    "CNNx16_0_L9_WPTR_BASE":          0x5010_0334, # 50100334
    "CNNx16_0_L10_WPTR_BASE":         0x5010_0338, # 50100338
    "CNNx16_0_L11_WPTR_BASE":         0x5010_033C, # 5010033C
    "CNNx16_0_L12_WPTR_BASE":         0x5010_0340, # 50100340
    "CNNx16_0_L13_WPTR_BASE":         0x5010_0344, # 50100344
    "CNNx16_0_L14_WPTR_BASE":         0x5010_0348, # 50100348
    "CNNx16_0_L15_WPTR_BASE":         0x5010_034C, # 5010034C
    "CNNx16_0_L0_WPTR_TOFF":          0x5010_0390, # 50100390
    "CNNx16_0_L1_WPTR_TOFF":          0x5010_0394, # 50100394
    "CNNx16_0_L2_WPTR_TOFF":          0x5010_0398, # 50100398
    "CNNx16_0_L3_WPTR_TOFF":          0x5010_039C, # 5010039C
    "CNNx16_0_L4_WPTR_TOFF":          0x5010_03A0, # 501003A0
    "CNNx16_0_L5_WPTR_TOFF":          0x5010_03A4, # 501003A4
    "CNNx16_0_L6_WPTR_TOFF":          0x5010_03A8, # 501003A8
    "CNNx16_0_L7_WPTR_TOFF":          0x5010_03AC, # 501003AC
    "CNNx16_0_L8_WPTR_TOFF":          0x5010_03B0, # 501003B0
    "CNNx16_0_L9_WPTR_TOFF":          0x5010_03B4, # 501003B4
    "CNNx16_0_L10_WPTR_TOFF":         0x5010_03B8, # 501003B8
    "CNNx16_0_L11_WPTR_TOFF":         0x5010_03BC, # 501003BC
    "CNNx16_0_L12_WPTR_TOFF":         0x5010_03C0, # 501003C0
    "CNNx16_0_L13_WPTR_TOFF":         0x5010_03C4, # 501003C4
    "CNNx16_0_L14_WPTR_TOFF":         0x5010_03C8, # 501003C8
    "CNNx16_0_L15_WPTR_TOFF":         0x5010_03CC, # 501003CC
    "CNNx16_0_L0_WPTR_MOFF":          0x5010_0410, # 50100410
    "CNNx16_0_L1_WPTR_MOFF":          0x5010_0414, # 50100414
    "CNNx16_0_L2_WPTR_MOFF":          0x5010_0418, # 50100418
    "CNNx16_0_L3_WPTR_MOFF":          0x5010_041C, # 5010041C
    "CNNx16_0_L4_WPTR_MOFF":          0x5010_0420, # 50100420
    "CNNx16_0_L5_WPTR_MOFF":          0x5010_0424, # 50100424
    "CNNx16_0_L6_WPTR_MOFF":          0x5010_0428, # 50100428
    "CNNx16_0_L7_WPTR_MOFF":          0x5010_042C, # 5010042C
    "CNNx16_0_L8_WPTR_MOFF":          0x5010_0430, # 50100430
    "CNNx16_0_L9_WPTR_MOFF":          0x5010_0434, # 50100434
    "CNNx16_0_L10_WPTR_MOFF":         0x5010_0438, # 50100438
    "CNNx16_0_L11_WPTR_MOFF":         0x5010_043C, # 5010043C
    "CNNx16_0_L12_WPTR_MOFF":         0x5010_0440, # 50100440
    "CNNx16_0_L13_WPTR_MOFF":         0x5010_0444, # 50100444
    "CNNx16_0_L14_WPTR_MOFF":         0x5010_0448, # 50100448
    "CNNx16_0_L15_WPTR_MOFF":         0x5010_044C, # 5010044C
    "CNNx16_0_L0_WPTR_CHOFF":         0x5010_0490, # 50100490
    "CNNx16_0_L1_WPTR_CHOFF":         0x5010_0494, # 50100494
    "CNNx16_0_L2_WPTR_CHOFF":         0x5010_0498, # 50100498
    "CNNx16_0_L3_WPTR_CHOFF":         0x5010_049C, # 5010049C
    "CNNx16_0_L4_WPTR_CHOFF":         0x5010_04A0, # 501004A0
    "CNNx16_0_L5_WPTR_CHOFF":         0x5010_04A4, # 501004A4
    "CNNx16_0_L6_WPTR_CHOFF":         0x5010_04A8, # 501004A8
    "CNNx16_0_L7_WPTR_CHOFF":         0x5010_04AC, # 501004AC
    "CNNx16_0_L8_WPTR_CHOFF":         0x5010_04B0, # 501004B0
    "CNNx16_0_L9_WPTR_CHOFF":         0x5010_04B4, # 501004B4
    "CNNx16_0_L10_WPTR_CHOFF":        0x5010_04B8, # 501004B8
    "CNNx16_0_L11_WPTR_CHOFF":        0x5010_04BC, # 501004BC
    "CNNx16_0_L12_WPTR_CHOFF":        0x5010_04C0, # 501004C0
    "CNNx16_0_L13_WPTR_CHOFF":        0x5010_04C4, # 501004C4
    "CNNx16_0_L14_WPTR_CHOFF":        0x5010_04C8, # 501004C8
    "CNNx16_0_L15_WPTR_CHOFF":        0x5010_04CC, # 501004CC
    "CNNx16_0_L0_RPTR_BASE":          0x5010_0510, # 50100510
    "CNNx16_0_L1_RPTR_BASE":          0x5010_0514, # 50100514
    "CNNx16_0_L2_RPTR_BASE":          0x5010_0518, # 50100518
    "CNNx16_0_L3_RPTR_BASE":          0x5010_051C, # 5010051C
    "CNNx16_0_L4_RPTR_BASE":          0x5010_0520, # 50100520
    "CNNx16_0_L5_RPTR_BASE":          0x5010_0524, # 50100524
    "CNNx16_0_L6_RPTR_BASE":          0x5010_0528, # 50100528
    "CNNx16_0_L7_RPTR_BASE":          0x5010_052C, # 5010052C
    "CNNx16_0_L8_RPTR_BASE":          0x5010_0530, # 50100530
    "CNNx16_0_L9_RPTR_BASE":          0x5010_0534, # 50100534
    "CNNx16_0_L10_RPTR_BASE":         0x5010_0538, # 50100538
    "CNNx16_0_L11_RPTR_BASE":         0x5010_053C, # 5010053C
    "CNNx16_0_L12_RPTR_BASE":         0x5010_0540, # 50100540
    "CNNx16_0_L13_RPTR_BASE":         0x5010_0544, # 50100544
    "CNNx16_0_L14_RPTR_BASE":         0x5010_0548, # 50100548
    "CNNx16_0_L15_RPTR_BASE":         0x5010_054C, # 5010054C
    "CNNx16_0_L0_LCTRL0":             0x5010_0590, # 50100590
    "CNNx16_0_L1_LCTRL0":             0x5010_0594, # 50100594
    "CNNx16_0_L2_LCTRL0":             0x5010_0598, # 50100598
    "CNNx16_0_L3_LCTRL0":             0x5010_059C, # 5010059C
    "CNNx16_0_L4_LCTRL0":             0x5010_05A0, # 501005A0
    "CNNx16_0_L5_LCTRL0":             0x5010_05A4, # 501005A4
    "CNNx16_0_L6_LCTRL0":             0x5010_05A8, # 501005A8
    "CNNx16_0_L7_LCTRL0":             0x5010_05AC, # 501005AC
    "CNNx16_0_L8_LCTRL0":             0x5010_05B0, # 501005B0
    "CNNx16_0_L9_LCTRL0":             0x5010_05B4, # 501005B4
    "CNNx16_0_L10_LCTRL0":            0x5010_05B8, # 501005B8
    "CNNx16_0_L11_LCTRL0":            0x5010_05BC, # 501005BC
    "CNNx16_0_L12_LCTRL0":            0x5010_05C0, # 501005C0
    "CNNx16_0_L13_LCTRL0":            0x5010_05C4, # 501005C4
    "CNNx16_0_L14_LCTRL0":            0x5010_05C8, # 501005C8
    "CNNx16_0_L15_LCTRL0":            0x5010_05CC, # 501005CC
    "CNNx16_0_L0_MCNT":               0x5010_0610, # 50100610
    "CNNx16_0_L1_MCNT":               0x5010_0614, # 50100614
    "CNNx16_0_L2_MCNT":               0x5010_0618, # 50100618
    "CNNx16_0_L3_MCNT":               0x5010_061C, # 5010061C
    "CNNx16_0_L4_MCNT":               0x5010_0620, # 50100620
    "CNNx16_0_L5_MCNT":               0x5010_0624, # 50100624
    "CNNx16_0_L6_MCNT":               0x5010_0628, # 50100628
    "CNNx16_0_L7_MCNT":               0x5010_062C, # 5010062C
    "CNNx16_0_L8_MCNT":               0x5010_0630, # 50100630
    "CNNx16_0_L9_MCNT":               0x5010_0634, # 50100634
    "CNNx16_0_L10_MCNT":              0x5010_0638, # 50100638
    "CNNx16_0_L11_MCNT":              0x5010_063C, # 5010063C
    "CNNx16_0_L12_MCNT":              0x5010_0640, # 50100640
    "CNNx16_0_L13_MCNT":              0x5010_0644, # 50100644
    "CNNx16_0_L14_MCNT":              0x5010_0648, # 50100648
    "CNNx16_0_L15_MCNT":              0x5010_064C, # 5010064C
    "CNNx16_0_L0_TPTR":               0x5010_0690, # 50100690
    "CNNx16_0_L1_TPTR":               0x5010_0694, # 50100694
    "CNNx16_0_L2_TPTR":               0x5010_0698, # 50100698
    "CNNx16_0_L3_TPTR":               0x5010_069C, # 5010069C
    "CNNx16_0_L4_TPTR":               0x5010_06A0, # 501006A0
    "CNNx16_0_L5_TPTR":               0x5010_06A4, # 501006A4
    "CNNx16_0_L6_TPTR":               0x5010_06A8, # 501006A8
    "CNNx16_0_L7_TPTR":               0x5010_06AC, # 501006AC
    "CNNx16_0_L8_TPTR":               0x5010_06B0, # 501006B0
    "CNNx16_0_L9_TPTR":               0x5010_06B4, # 501006B4
    "CNNx16_0_L10_TPTR":              0x5010_06B8, # 501006B8
    "CNNx16_0_L11_TPTR":              0x5010_06BC, # 501006BC
    "CNNx16_0_L12_TPTR":              0x5010_06C0, # 501006C0
    "CNNx16_0_L13_TPTR":              0x5010_06C4, # 501006C4
    "CNNx16_0_L14_TPTR":              0x5010_06C8, # 501006C8
    "CNNx16_0_L15_TPTR":              0x5010_06CC, # 501006CC
    "CNNx16_0_L0_EN":                 0x5010_0710, # 50100710
    "CNNx16_0_L1_EN":                 0x5010_0714, # 50100714
    "CNNx16_0_L2_EN":                 0x5010_0718, # 50100718
    "CNNx16_0_L3_EN":                 0x5010_071C, # 5010071C
    "CNNx16_0_L4_EN":                 0x5010_0720, # 50100720
    "CNNx16_0_L5_EN":                 0x5010_0724, # 50100724
    "CNNx16_0_L6_EN":                 0x5010_0728, # 50100728
    "CNNx16_0_L7_EN":                 0x5010_072C, # 5010072C
    "CNNx16_0_L8_EN":                 0x5010_0730, # 50100730
    "CNNx16_0_L9_EN":                 0x5010_0734, # 50100734
    "CNNx16_0_L10_EN":                0x5010_0738, # 50100738
    "CNNx16_0_L11_EN":                0x5010_073C, # 5010073C
    "CNNx16_0_L12_EN":                0x5010_0740, # 50100740
    "CNNx16_0_L13_EN":                0x5010_0744, # 50100744
    "CNNx16_0_L14_EN":                0x5010_0748, # 50100748
    "CNNx16_0_L15_EN":                0x5010_074C, # 5010074C
    "CNNx16_0_L0_POST":               0x5010_0790, # 50100790
    "CNNx16_0_L1_POST":               0x5010_0794, # 50100794
    "CNNx16_0_L2_POST":               0x5010_0798, # 50100798
    "CNNx16_0_L3_POST":               0x5010_079C, # 5010079C
    "CNNx16_0_L4_POST":               0x5010_07A0, # 501007A0
    "CNNx16_0_L5_POST":               0x5010_07A4, # 501007A4
    "CNNx16_0_L6_POST":               0x5010_07A8, # 501007A8
    "CNNx16_0_L7_POST":               0x5010_07AC, # 501007AC
    "CNNx16_0_L8_POST":               0x5010_07B0, # 501007B0
    "CNNx16_0_L9_POST":               0x5010_07B4, # 501007B4
    "CNNx16_0_L10_POST":              0x5010_07B8, # 501007B8
    "CNNx16_0_L11_POST":              0x5010_07BC, # 501007BC
    "CNNx16_0_L12_POST":              0x5010_07C0, # 501007C0
    "CNNx16_0_L13_POST":              0x5010_07C4, # 501007C4
    "CNNx16_0_L14_POST":              0x5010_07C8, # 501007C8
    "CNNx16_0_L15_POST":              0x5010_07CC, # 501007CC
    "CNNx16_0_S0_STRM0":              0x5010_0810, # 50100810
    "CNNx16_0_S1_STRM0":              0x5010_0814, # 50100814
    "CNNx16_0_S2_STRM0":              0x5010_0818, # 50100818
    "CNNx16_0_S3_STRM0":              0x5010_081C, # 5010081C
    "CNNx16_0_S4_STRM0":              0x5010_0820, # 50100820
    "CNNx16_0_S5_STRM0":              0x5010_0824, # 50100824
    "CNNx16_0_S6_STRM0":              0x5010_0828, # 50100828
    "CNNx16_0_S7_STRM0":              0x5010_082C, # 5010082C
    "CNNx16_0_S0_STRM1":              0x5010_0890, # 50100890
    "CNNx16_0_S1_STRM1":              0x5010_0894, # 50100894
    "CNNx16_0_S2_STRM1":              0x5010_0898, # 50100898
    "CNNx16_0_S3_STRM1":              0x5010_089C, # 5010089C
    "CNNx16_0_S4_STRM1":              0x5010_08A0, # 501008A0
    "CNNx16_0_S5_STRM1":              0x5010_08A4, # 501008A4
    "CNNx16_0_S6_STRM1":              0x5010_08A8, # 501008A8
    "CNNx16_0_S7_STRM1":              0x5010_08AC, # 501008AC
    "CNNx16_0_S0_FBUF":               0x5010_0910, # 50100910
    "CNNx16_0_S1_FBUF":               0x5010_0914, # 50100914
    "CNNx16_0_S2_FBUF":               0x5010_0918, # 50100918
    "CNNx16_0_S3_FBUF":               0x5010_091C, # 5010091C
    "CNNx16_0_S4_FBUF":               0x5010_0920, # 50100920
    "CNNx16_0_S5_FBUF":               0x5010_0924, # 50100924
    "CNNx16_0_S6_FBUF":               0x5010_0928, # 50100928
    "CNNx16_0_S7_FBUF":               0x5010_092C, # 5010092C
    "CNNx16_0_IFRM":                  0x5010_0990, # 50100990
    "CNNx16_0_L0_LCTRL1":             0x5010_0A10, # 50100A10
    "CNNx16_0_L1_LCTRL1":             0x5010_0A14, # 50100A14
    "CNNx16_0_L2_LCTRL1":             0x5010_0A18, # 50100A18
    "CNNx16_0_L3_LCTRL1":             0x5010_0A1C, # 50100A1C
    "CNNx16_0_L4_LCTRL1":             0x5010_0A20, # 50100A20
    "CNNx16_0_L5_LCTRL1":             0x5010_0A24, # 50100A24
    "CNNx16_0_L6_LCTRL1":             0x5010_0A28, # 50100A28
    "CNNx16_0_L7_LCTRL1":             0x5010_0A2C, # 50100A2C
    "CNNx16_0_L8_LCTRL1":             0x5010_0A30, # 50100A30
    "CNNx16_0_L9_LCTRL1":             0x5010_0A34, # 50100A34
    "CNNx16_0_L10_LCTRL1":            0x5010_0A38, # 50100A38
    "CNNx16_0_L11_LCTRL1":            0x5010_0A3C, # 50100A3C
    "CNNx16_0_L12_LCTRL1":            0x5010_0A40, # 50100A40
    "CNNx16_0_L13_LCTRL1":            0x5010_0A44, # 50100A44
    "CNNx16_0_L14_LCTRL1":            0x5010_0A48, # 50100A48
    "CNNx16_0_L15_LCTRL1":            0x5010_0A4C, # 50100A4C
    "CNNx16_0_MLAT":                  0x5010_1000, # 50101000
    "CNNx16_1_CTRL":                  0x5050_0000, # 50500000
    "CNNx16_1_SRAM":                  0x5050_0004, # 50500004
    "CNNx16_1_LCNT_MAX":              0x5050_0008, # 50500008
    "CNNx16_1_TEST":                  0x5050_000C, # 5050000C
    "CNNx16_1_L0_RCNT":               0x5050_0010, # 50500010
    "CNNx16_1_L1_RCNT":               0x5050_0014, # 50500014
    "CNNx16_1_L2_RCNT":               0x5050_0018, # 50500018
    "CNNx16_1_L3_RCNT":               0x5050_001C, # 5050001C
    "CNNx16_1_L4_RCNT":               0x5050_0020, # 50500020
    "CNNx16_1_L5_RCNT":               0x5050_0024, # 50500024
    "CNNx16_1_L6_RCNT":               0x5050_0028, # 50500028
    "CNNx16_1_L7_RCNT":               0x5050_002C, # 5050002C
    "CNNx16_1_L8_RCNT":               0x5050_0030, # 50500030
    "CNNx16_1_L9_RCNT":               0x5050_0034, # 50500034
    "CNNx16_1_L10_RCNT":              0x5050_0038, # 50500038
    "CNNx16_1_L11_RCNT":              0x5050_003C, # 5050003C
    "CNNx16_1_L12_RCNT":              0x5050_0040, # 50500040
    "CNNx16_1_L13_RCNT":              0x5050_0044, # 50500044
    "CNNx16_1_L14_RCNT":              0x5050_0048, # 50500048
    "CNNx16_1_L15_RCNT":              0x5050_004C, # 5050004C
    "CNNx16_1_L0_CCNT":               0x5050_0090, # 50500090
    "CNNx16_1_L1_CCNT":               0x5050_0094, # 50500094
    "CNNx16_1_L2_CCNT":               0x5050_0098, # 50500098
    "CNNx16_1_L3_CCNT":               0x5050_009C, # 5050009C
    "CNNx16_1_L4_CCNT":               0x5050_00A0, # 505000A0
    "CNNx16_1_L5_CCNT":               0x5050_00A4, # 505000A4
    "CNNx16_1_L6_CCNT":               0x5050_00A8, # 505000A8
    "CNNx16_1_L7_CCNT":               0x5050_00AC, # 505000AC
    "CNNx16_1_L8_CCNT":               0x5050_00B0, # 505000B0
    "CNNx16_1_L9_CCNT":               0x5050_00B4, # 505000B4
    "CNNx16_1_L10_CCNT":              0x5050_00B8, # 505000B8
    "CNNx16_1_L11_CCNT":              0x5050_00BC, # 505000BC
    "CNNx16_1_L12_CCNT":              0x5050_00C0, # 505000C0
    "CNNx16_1_L13_CCNT":              0x5050_00C4, # 505000C4
    "CNNx16_1_L14_CCNT":              0x5050_00C8, # 505000C8
    "CNNx16_1_L15_CCNT":              0x5050_00CC, # 505000CC
    "CNNx16_1_L0_ONED":               0x5050_0110, # 50500110
    "CNNx16_1_L1_ONED":               0x5050_0114, # 50500114
    "CNNx16_1_L2_ONED":               0x5050_0118, # 50500118
    "CNNx16_1_L3_ONED":               0x5050_011C, # 5050011C
    "CNNx16_1_L4_ONED":               0x5050_0120, # 50500120
    "CNNx16_1_L5_ONED":               0x5050_0124, # 50500124
    "CNNx16_1_L6_ONED":               0x5050_0128, # 50500128
    "CNNx16_1_L7_ONED":               0x5050_012C, # 5050012C
    "CNNx16_1_L8_ONED":               0x5050_0130, # 50500130
    "CNNx16_1_L9_ONED":               0x5050_0134, # 50500134
    "CNNx16_1_L10_ONED":              0x5050_0138, # 50500138
    "CNNx16_1_L11_ONED":              0x5050_013C, # 5050013C
    "CNNx16_1_L12_ONED":              0x5050_0140, # 50500140
    "CNNx16_1_L13_ONED":              0x5050_0144, # 50500144
    "CNNx16_1_L14_ONED":              0x5050_0148, # 50500148
    "CNNx16_1_L15_ONED":              0x5050_014C, # 5050014C
    "CNNx16_1_L0_PRCNT":              0x5050_0190, # 50500190
    "CNNx16_1_L1_PRCNT":              0x5050_0194, # 50500194
    "CNNx16_1_L2_PRCNT":              0x5050_0198, # 50500198
    "CNNx16_1_L3_PRCNT":              0x5050_019C, # 5050019C
    "CNNx16_1_L4_PRCNT":              0x5050_01A0, # 505001A0
    "CNNx16_1_L5_PRCNT":              0x5050_01A4, # 505001A4
    "CNNx16_1_L6_PRCNT":              0x5050_01A8, # 505001A8
    "CNNx16_1_L7_PRCNT":              0x5050_01AC, # 505001AC
    "CNNx16_1_L8_PRCNT":              0x5050_01B0, # 505001B0
    "CNNx16_1_L9_PRCNT":              0x5050_01B4, # 505001B4
    "CNNx16_1_L10_PRCNT":             0x5050_01B8, # 505001B8
    "CNNx16_1_L11_PRCNT":             0x5050_01BC, # 505001BC
    "CNNx16_1_L12_PRCNT":             0x5050_01C0, # 505001C0
    "CNNx16_1_L13_PRCNT":             0x5050_01C4, # 505001C4
    "CNNx16_1_L14_PRCNT":             0x5050_01C8, # 505001C8
    "CNNx16_1_L15_PRCNT":             0x5050_01CC, # 505001CC
    "CNNx16_1_L0_PCCNT":              0x5050_0210, # 50500210
    "CNNx16_1_L1_PCCNT":              0x5050_0214, # 50500214
    "CNNx16_1_L2_PCCNT":              0x5050_0218, # 50500218
    "CNNx16_1_L3_PCCNT":              0x5050_021C, # 5050021C
    "CNNx16_1_L4_PCCNT":              0x5050_0220, # 50500220
    "CNNx16_1_L5_PCCNT":              0x5050_0224, # 50500224
    "CNNx16_1_L6_PCCNT":              0x5050_0228, # 50500228
    "CNNx16_1_L7_PCCNT":              0x5050_022C, # 5050022C
    "CNNx16_1_L8_PCCNT":              0x5050_0230, # 50500230
    "CNNx16_1_L9_PCCNT":              0x5050_0234, # 50500234
    "CNNx16_1_L10_PCCNT":             0x5050_0238, # 50500238
    "CNNx16_1_L11_PCCNT":             0x5050_023C, # 5050023C
    "CNNx16_1_L12_PCCNT":             0x5050_0240, # 50500240
    "CNNx16_1_L13_PCCNT":             0x5050_0244, # 50500244
    "CNNx16_1_L14_PCCNT":             0x5050_0248, # 50500248
    "CNNx16_1_L15_PCCNT":             0x5050_024C, # 5050024C
    "CNNx16_1_L0_STRIDE":             0x5050_0290, # 50500290
    "CNNx16_1_L1_STRIDE":             0x5050_0294, # 50500294
    "CNNx16_1_L2_STRIDE":             0x5050_0298, # 50500298
    "CNNx16_1_L3_STRIDE":             0x5050_029C, # 5050029C
    "CNNx16_1_L4_STRIDE":             0x5050_02A0, # 505002A0
    "CNNx16_1_L5_STRIDE":             0x5050_02A4, # 505002A4
    "CNNx16_1_L6_STRIDE":             0x5050_02A8, # 505002A8
    "CNNx16_1_L7_STRIDE":             0x5050_02AC, # 505002AC
    "CNNx16_1_L8_STRIDE":             0x5050_02B0, # 505002B0
    "CNNx16_1_L9_STRIDE":             0x5050_02B4, # 505002B4
    "CNNx16_1_L10_STRIDE":            0x5050_02B8, # 505002B8
    "CNNx16_1_L11_STRIDE":            0x5050_02BC, # 505002BC
    "CNNx16_1_L12_STRIDE":            0x5050_02C0, # 505002C0
    "CNNx16_1_L13_STRIDE":            0x5050_02C4, # 505002C4
    "CNNx16_1_L14_STRIDE":            0x5050_02C8, # 505002C8
    "CNNx16_1_L15_STRIDE":            0x5050_02CC, # 505002CC
    "CNNx16_1_L0_WPTR_BASE":          0x5050_0310, # 50500310
    "CNNx16_1_L1_WPTR_BASE":          0x5050_0314, # 50500314
    "CNNx16_1_L2_WPTR_BASE":          0x5050_0318, # 50500318
    "CNNx16_1_L3_WPTR_BASE":          0x5050_031C, # 5050031C
    "CNNx16_1_L4_WPTR_BASE":          0x5050_0320, # 50500320
    "CNNx16_1_L5_WPTR_BASE":          0x5050_0324, # 50500324
    "CNNx16_1_L6_WPTR_BASE":          0x5050_0328, # 50500328
    "CNNx16_1_L7_WPTR_BASE":          0x5050_032C, # 5050032C
    "CNNx16_1_L8_WPTR_BASE":          0x5050_0330, # 50500330
    "CNNx16_1_L9_WPTR_BASE":          0x5050_0334, # 50500334
    "CNNx16_1_L10_WPTR_BASE":         0x5050_0338, # 50500338
    "CNNx16_1_L11_WPTR_BASE":         0x5050_033C, # 5050033C
    "CNNx16_1_L12_WPTR_BASE":         0x5050_0340, # 50500340
    "CNNx16_1_L13_WPTR_BASE":         0x5050_0344, # 50500344
    "CNNx16_1_L14_WPTR_BASE":         0x5050_0348, # 50500348
    "CNNx16_1_L15_WPTR_BASE":         0x5050_034C, # 5050034C
    "CNNx16_1_L0_WPTR_TOFF":          0x5050_0390, # 50500390
    "CNNx16_1_L1_WPTR_TOFF":          0x5050_0394, # 50500394
    "CNNx16_1_L2_WPTR_TOFF":          0x5050_0398, # 50500398
    "CNNx16_1_L3_WPTR_TOFF":          0x5050_039C, # 5050039C
    "CNNx16_1_L4_WPTR_TOFF":          0x5050_03A0, # 505003A0
    "CNNx16_1_L5_WPTR_TOFF":          0x5050_03A4, # 505003A4
    "CNNx16_1_L6_WPTR_TOFF":          0x5050_03A8, # 505003A8
    "CNNx16_1_L7_WPTR_TOFF":          0x5050_03AC, # 505003AC
    "CNNx16_1_L8_WPTR_TOFF":          0x5050_03B0, # 505003B0
    "CNNx16_1_L9_WPTR_TOFF":          0x5050_03B4, # 505003B4
    "CNNx16_1_L10_WPTR_TOFF":         0x5050_03B8, # 505003B8
    "CNNx16_1_L11_WPTR_TOFF":         0x5050_03BC, # 505003BC
    "CNNx16_1_L12_WPTR_TOFF":         0x5050_03C0, # 505003C0
    "CNNx16_1_L13_WPTR_TOFF":         0x5050_03C4, # 505003C4
    "CNNx16_1_L14_WPTR_TOFF":         0x5050_03C8, # 505003C8
    "CNNx16_1_L15_WPTR_TOFF":         0x5050_03CC, # 505003CC
    "CNNx16_1_L0_WPTR_MOFF":          0x5050_0410, # 50500410
    "CNNx16_1_L1_WPTR_MOFF":          0x5050_0414, # 50500414
    "CNNx16_1_L2_WPTR_MOFF":          0x5050_0418, # 50500418
    "CNNx16_1_L3_WPTR_MOFF":          0x5050_041C, # 5050041C
    "CNNx16_1_L4_WPTR_MOFF":          0x5050_0420, # 50500420
    "CNNx16_1_L5_WPTR_MOFF":          0x5050_0424, # 50500424
    "CNNx16_1_L6_WPTR_MOFF":          0x5050_0428, # 50500428
    "CNNx16_1_L7_WPTR_MOFF":          0x5050_042C, # 5050042C
    "CNNx16_1_L8_WPTR_MOFF":          0x5050_0430, # 50500430
    "CNNx16_1_L9_WPTR_MOFF":          0x5050_0434, # 50500434
    "CNNx16_1_L10_WPTR_MOFF":         0x5050_0438, # 50500438
    "CNNx16_1_L11_WPTR_MOFF":         0x5050_043C, # 5050043C
    "CNNx16_1_L12_WPTR_MOFF":         0x5050_0440, # 50500440
    "CNNx16_1_L13_WPTR_MOFF":         0x5050_0444, # 50500444
    "CNNx16_1_L14_WPTR_MOFF":         0x5050_0448, # 50500448
    "CNNx16_1_L15_WPTR_MOFF":         0x5050_044C, # 5050044C
    "CNNx16_1_L0_WPTR_CHOFF":         0x5050_0490, # 50500490
    "CNNx16_1_L1_WPTR_CHOFF":         0x5050_0494, # 50500494
    "CNNx16_1_L2_WPTR_CHOFF":         0x5050_0498, # 50500498
    "CNNx16_1_L3_WPTR_CHOFF":         0x5050_049C, # 5050049C
    "CNNx16_1_L4_WPTR_CHOFF":         0x5050_04A0, # 505004A0
    "CNNx16_1_L5_WPTR_CHOFF":         0x5050_04A4, # 505004A4
    "CNNx16_1_L6_WPTR_CHOFF":         0x5050_04A8, # 505004A8
    "CNNx16_1_L7_WPTR_CHOFF":         0x5050_04AC, # 505004AC
    "CNNx16_1_L8_WPTR_CHOFF":         0x5050_04B0, # 505004B0
    "CNNx16_1_L9_WPTR_CHOFF":         0x5050_04B4, # 505004B4
    "CNNx16_1_L10_WPTR_CHOFF":        0x5050_04B8, # 505004B8
    "CNNx16_1_L11_WPTR_CHOFF":        0x5050_04BC, # 505004BC
    "CNNx16_1_L12_WPTR_CHOFF":        0x5050_04C0, # 505004C0
    "CNNx16_1_L13_WPTR_CHOFF":        0x5050_04C4, # 505004C4
    "CNNx16_1_L14_WPTR_CHOFF":        0x5050_04C8, # 505004C8
    "CNNx16_1_L15_WPTR_CHOFF":        0x5050_04CC, # 505004CC
    "CNNx16_1_L0_RPTR_BASE":          0x5050_0510, # 50500510
    "CNNx16_1_L1_RPTR_BASE":          0x5050_0514, # 50500514
    "CNNx16_1_L2_RPTR_BASE":          0x5050_0518, # 50500518
    "CNNx16_1_L3_RPTR_BASE":          0x5050_051C, # 5050051C
    "CNNx16_1_L4_RPTR_BASE":          0x5050_0520, # 50500520
    "CNNx16_1_L5_RPTR_BASE":          0x5050_0524, # 50500524
    "CNNx16_1_L6_RPTR_BASE":          0x5050_0528, # 50500528
    "CNNx16_1_L7_RPTR_BASE":          0x5050_052C, # 5050052C
    "CNNx16_1_L8_RPTR_BASE":          0x5050_0530, # 50500530
    "CNNx16_1_L9_RPTR_BASE":          0x5050_0534, # 50500534
    "CNNx16_1_L10_RPTR_BASE":         0x5050_0538, # 50500538
    "CNNx16_1_L11_RPTR_BASE":         0x5050_053C, # 5050053C
    "CNNx16_1_L12_RPTR_BASE":         0x5050_0540, # 50500540
    "CNNx16_1_L13_RPTR_BASE":         0x5050_0544, # 50500544
    "CNNx16_1_L14_RPTR_BASE":         0x5050_0548, # 50500548
    "CNNx16_1_L15_RPTR_BASE":         0x5050_054C, # 5050054C
    "CNNx16_1_L0_LCTRL0":             0x5050_0590, # 50500590
    "CNNx16_1_L1_LCTRL0":             0x5050_0594, # 50500594
    "CNNx16_1_L2_LCTRL0":             0x5050_0598, # 50500598
    "CNNx16_1_L3_LCTRL0":             0x5050_059C, # 5050059C
    "CNNx16_1_L4_LCTRL0":             0x5050_05A0, # 505005A0
    "CNNx16_1_L5_LCTRL0":             0x5050_05A4, # 505005A4
    "CNNx16_1_L6_LCTRL0":             0x5050_05A8, # 505005A8
    "CNNx16_1_L7_LCTRL0":             0x5050_05AC, # 505005AC
    "CNNx16_1_L8_LCTRL0":             0x5050_05B0, # 505005B0
    "CNNx16_1_L9_LCTRL0":             0x5050_05B4, # 505005B4
    "CNNx16_1_L10_LCTRL0":            0x5050_05B8, # 505005B8
    "CNNx16_1_L11_LCTRL0":            0x5050_05BC, # 505005BC
    "CNNx16_1_L12_LCTRL0":            0x5050_05C0, # 505005C0
    "CNNx16_1_L13_LCTRL0":            0x5050_05C4, # 505005C4
    "CNNx16_1_L14_LCTRL0":            0x5050_05C8, # 505005C8
    "CNNx16_1_L15_LCTRL0":            0x5050_05CC, # 505005CC
    "CNNx16_1_L0_MCNT":               0x5050_0610, # 50500610
    "CNNx16_1_L1_MCNT":               0x5050_0614, # 50500614
    "CNNx16_1_L2_MCNT":               0x5050_0618, # 50500618
    "CNNx16_1_L3_MCNT":               0x5050_061C, # 5050061C
    "CNNx16_1_L4_MCNT":               0x5050_0620, # 50500620
    "CNNx16_1_L5_MCNT":               0x5050_0624, # 50500624
    "CNNx16_1_L6_MCNT":               0x5050_0628, # 50500628
    "CNNx16_1_L7_MCNT":               0x5050_062C, # 5050062C
    "CNNx16_1_L8_MCNT":               0x5050_0630, # 50500630
    "CNNx16_1_L9_MCNT":               0x5050_0634, # 50500634
    "CNNx16_1_L10_MCNT":              0x5050_0638, # 50500638
    "CNNx16_1_L11_MCNT":              0x5050_063C, # 5050063C
    "CNNx16_1_L12_MCNT":              0x5050_0640, # 50500640
    "CNNx16_1_L13_MCNT":              0x5050_0644, # 50500644
    "CNNx16_1_L14_MCNT":              0x5050_0648, # 50500648
    "CNNx16_1_L15_MCNT":              0x5050_064C, # 5050064C
    "CNNx16_1_L0_TPTR":               0x5050_0690, # 50500690
    "CNNx16_1_L1_TPTR":               0x5050_0694, # 50500694
    "CNNx16_1_L2_TPTR":               0x5050_0698, # 50500698
    "CNNx16_1_L3_TPTR":               0x5050_069C, # 5050069C
    "CNNx16_1_L4_TPTR":               0x5050_06A0, # 505006A0
    "CNNx16_1_L5_TPTR":               0x5050_06A4, # 505006A4
    "CNNx16_1_L6_TPTR":               0x5050_06A8, # 505006A8
    "CNNx16_1_L7_TPTR":               0x5050_06AC, # 505006AC
    "CNNx16_1_L8_TPTR":               0x5050_06B0, # 505006B0
    "CNNx16_1_L9_TPTR":               0x5050_06B4, # 505006B4
    "CNNx16_1_L10_TPTR":              0x5050_06B8, # 505006B8
    "CNNx16_1_L11_TPTR":              0x5050_06BC, # 505006BC
    "CNNx16_1_L12_TPTR":              0x5050_06C0, # 505006C0
    "CNNx16_1_L13_TPTR":              0x5050_06C4, # 505006C4
    "CNNx16_1_L14_TPTR":              0x5050_06C8, # 505006C8
    "CNNx16_1_L15_TPTR":              0x5050_06CC, # 505006CC
    "CNNx16_1_L0_EN":                 0x5050_0710, # 50500710
    "CNNx16_1_L1_EN":                 0x5050_0714, # 50500714
    "CNNx16_1_L2_EN":                 0x5050_0718, # 50500718
    "CNNx16_1_L3_EN":                 0x5050_071C, # 5050071C
    "CNNx16_1_L4_EN":                 0x5050_0720, # 50500720
    "CNNx16_1_L5_EN":                 0x5050_0724, # 50500724
    "CNNx16_1_L6_EN":                 0x5050_0728, # 50500728
    "CNNx16_1_L7_EN":                 0x5050_072C, # 5050072C
    "CNNx16_1_L8_EN":                 0x5050_0730, # 50500730
    "CNNx16_1_L9_EN":                 0x5050_0734, # 50500734
    "CNNx16_1_L10_EN":                0x5050_0738, # 50500738
    "CNNx16_1_L11_EN":                0x5050_073C, # 5050073C
    "CNNx16_1_L12_EN":                0x5050_0740, # 50500740
    "CNNx16_1_L13_EN":                0x5050_0744, # 50500744
    "CNNx16_1_L14_EN":                0x5050_0748, # 50500748
    "CNNx16_1_L15_EN":                0x5050_074C, # 5050074C
    "CNNx16_1_L0_POST":               0x5050_0790, # 50500790
    "CNNx16_1_L1_POST":               0x5050_0794, # 50500794
    "CNNx16_1_L2_POST":               0x5050_0798, # 50500798
    "CNNx16_1_L3_POST":               0x5050_079C, # 5050079C
    "CNNx16_1_L4_POST":               0x5050_07A0, # 505007A0
    "CNNx16_1_L5_POST":               0x5050_07A4, # 505007A4
    "CNNx16_1_L6_POST":               0x5050_07A8, # 505007A8
    "CNNx16_1_L7_POST":               0x5050_07AC, # 505007AC
    "CNNx16_1_L8_POST":               0x5050_07B0, # 505007B0
    "CNNx16_1_L9_POST":               0x5050_07B4, # 505007B4
    "CNNx16_1_L10_POST":              0x5050_07B8, # 505007B8
    "CNNx16_1_L11_POST":              0x5050_07BC, # 505007BC
    "CNNx16_1_L12_POST":              0x5050_07C0, # 505007C0
    "CNNx16_1_L13_POST":              0x5050_07C4, # 505007C4
    "CNNx16_1_L14_POST":              0x5050_07C8, # 505007C8
    "CNNx16_1_L15_POST":              0x5050_07CC, # 505007CC
    "CNNx16_1_S0_STRM0":              0x5050_0810, # 50500810
    "CNNx16_1_S1_STRM0":              0x5050_0814, # 50500814
    "CNNx16_1_S2_STRM0":              0x5050_0818, # 50500818
    "CNNx16_1_S3_STRM0":              0x5050_081C, # 5050081C
    "CNNx16_1_S4_STRM0":              0x5050_0820, # 50500820
    "CNNx16_1_S5_STRM0":              0x5050_0824, # 50500824
    "CNNx16_1_S6_STRM0":              0x5050_0828, # 50500828
    "CNNx16_1_S7_STRM0":              0x5050_082C, # 5050082C
    "CNNx16_1_S0_STRM1":              0x5050_0890, # 50500890
    "CNNx16_1_S1_STRM1":              0x5050_0894, # 50500894
    "CNNx16_1_S2_STRM1":              0x5050_0898, # 50500898
    "CNNx16_1_S3_STRM1":              0x5050_089C, # 5050089C
    "CNNx16_1_S4_STRM1":              0x5050_08A0, # 505008A0
    "CNNx16_1_S5_STRM1":              0x5050_08A4, # 505008A4
    "CNNx16_1_S6_STRM1":              0x5050_08A8, # 505008A8
    "CNNx16_1_S7_STRM1":              0x5050_08AC, # 505008AC
    "CNNx16_1_S0_FBUF":               0x5050_0910, # 50500910
    "CNNx16_1_S1_FBUF":               0x5050_0914, # 50500914
    "CNNx16_1_S2_FBUF":               0x5050_0918, # 50500918
    "CNNx16_1_S3_FBUF":               0x5050_091C, # 5050091C
    "CNNx16_1_S4_FBUF":               0x5050_0920, # 50500920
    "CNNx16_1_S5_FBUF":               0x5050_0924, # 50500924
    "CNNx16_1_S6_FBUF":               0x5050_0928, # 50500928
    "CNNx16_1_S7_FBUF":               0x5050_092C, # 5050092C
    "CNNx16_1_IFRM":                  0x5050_0990, # 50500990
    "CNNx16_1_L0_LCTRL1":             0x5050_0A10, # 50500A10
    "CNNx16_1_L1_LCTRL1":             0x5050_0A14, # 50500A14
    "CNNx16_1_L2_LCTRL1":             0x5050_0A18, # 50500A18
    "CNNx16_1_L3_LCTRL1":             0x5050_0A1C, # 50500A1C
    "CNNx16_1_L4_LCTRL1":             0x5050_0A20, # 50500A20
    "CNNx16_1_L5_LCTRL1":             0x5050_0A24, # 50500A24
    "CNNx16_1_L6_LCTRL1":             0x5050_0A28, # 50500A28
    "CNNx16_1_L7_LCTRL1":             0x5050_0A2C, # 50500A2C
    "CNNx16_1_L8_LCTRL1":             0x5050_0A30, # 50500A30
    "CNNx16_1_L9_LCTRL1":             0x5050_0A34, # 50500A34
    "CNNx16_1_L10_LCTRL1":            0x5050_0A38, # 50500A38
    "CNNx16_1_L11_LCTRL1":            0x5050_0A3C, # 50500A3C
    "CNNx16_1_L12_LCTRL1":            0x5050_0A40, # 50500A40
    "CNNx16_1_L13_LCTRL1":            0x5050_0A44, # 50500A44
    "CNNx16_1_L14_LCTRL1":            0x5050_0A48, # 50500A48
    "CNNx16_1_L15_LCTRL1":            0x5050_0A4C, # 50500A4C
    "CNNx16_1_MLAT":                  0x5050_1000, # 50501000
    "CNNx16_2_CTRL":                  0x5090_0000, # 50900000
    "CNNx16_2_SRAM":                  0x5090_0004, # 50900004
    "CNNx16_2_LCNT_MAX":              0x5090_0008, # 50900008
    "CNNx16_2_TEST":                  0x5090_000C, # 5090000C
    "CNNx16_2_L0_RCNT":               0x5090_0010, # 50900010
    "CNNx16_2_L1_RCNT":               0x5090_0014, # 50900014
    "CNNx16_2_L2_RCNT":               0x5090_0018, # 50900018
    "CNNx16_2_L3_RCNT":               0x5090_001C, # 5090001C
    "CNNx16_2_L4_RCNT":               0x5090_0020, # 50900020
    "CNNx16_2_L5_RCNT":               0x5090_0024, # 50900024
    "CNNx16_2_L6_RCNT":               0x5090_0028, # 50900028
    "CNNx16_2_L7_RCNT":               0x5090_002C, # 5090002C
    "CNNx16_2_L8_RCNT":               0x5090_0030, # 50900030
    "CNNx16_2_L9_RCNT":               0x5090_0034, # 50900034
    "CNNx16_2_L10_RCNT":              0x5090_0038, # 50900038
    "CNNx16_2_L11_RCNT":              0x5090_003C, # 5090003C
    "CNNx16_2_L12_RCNT":              0x5090_0040, # 50900040
    "CNNx16_2_L13_RCNT":              0x5090_0044, # 50900044
    "CNNx16_2_L14_RCNT":              0x5090_0048, # 50900048
    "CNNx16_2_L15_RCNT":              0x5090_004C, # 5090004C
    "CNNx16_2_L0_CCNT":               0x5090_0090, # 50900090
    "CNNx16_2_L1_CCNT":               0x5090_0094, # 50900094
    "CNNx16_2_L2_CCNT":               0x5090_0098, # 50900098
    "CNNx16_2_L3_CCNT":               0x5090_009C, # 5090009C
    "CNNx16_2_L4_CCNT":               0x5090_00A0, # 509000A0
    "CNNx16_2_L5_CCNT":               0x5090_00A4, # 509000A4
    "CNNx16_2_L6_CCNT":               0x5090_00A8, # 509000A8
    "CNNx16_2_L7_CCNT":               0x5090_00AC, # 509000AC
    "CNNx16_2_L8_CCNT":               0x5090_00B0, # 509000B0
    "CNNx16_2_L9_CCNT":               0x5090_00B4, # 509000B4
    "CNNx16_2_L10_CCNT":              0x5090_00B8, # 509000B8
    "CNNx16_2_L11_CCNT":              0x5090_00BC, # 509000BC
    "CNNx16_2_L12_CCNT":              0x5090_00C0, # 509000C0
    "CNNx16_2_L13_CCNT":              0x5090_00C4, # 509000C4
    "CNNx16_2_L14_CCNT":              0x5090_00C8, # 509000C8
    "CNNx16_2_L15_CCNT":              0x5090_00CC, # 509000CC
    "CNNx16_2_L0_ONED":               0x5090_0110, # 50900110
    "CNNx16_2_L1_ONED":               0x5090_0114, # 50900114
    "CNNx16_2_L2_ONED":               0x5090_0118, # 50900118
    "CNNx16_2_L3_ONED":               0x5090_011C, # 5090011C
    "CNNx16_2_L4_ONED":               0x5090_0120, # 50900120
    "CNNx16_2_L5_ONED":               0x5090_0124, # 50900124
    "CNNx16_2_L6_ONED":               0x5090_0128, # 50900128
    "CNNx16_2_L7_ONED":               0x5090_012C, # 5090012C
    "CNNx16_2_L8_ONED":               0x5090_0130, # 50900130
    "CNNx16_2_L9_ONED":               0x5090_0134, # 50900134
    "CNNx16_2_L10_ONED":              0x5090_0138, # 50900138
    "CNNx16_2_L11_ONED":              0x5090_013C, # 5090013C
    "CNNx16_2_L12_ONED":              0x5090_0140, # 50900140
    "CNNx16_2_L13_ONED":              0x5090_0144, # 50900144
    "CNNx16_2_L14_ONED":              0x5090_0148, # 50900148
    "CNNx16_2_L15_ONED":              0x5090_014C, # 5090014C
    "CNNx16_2_L0_PRCNT":              0x5090_0190, # 50900190
    "CNNx16_2_L1_PRCNT":              0x5090_0194, # 50900194
    "CNNx16_2_L2_PRCNT":              0x5090_0198, # 50900198
    "CNNx16_2_L3_PRCNT":              0x5090_019C, # 5090019C
    "CNNx16_2_L4_PRCNT":              0x5090_01A0, # 509001A0
    "CNNx16_2_L5_PRCNT":              0x5090_01A4, # 509001A4
    "CNNx16_2_L6_PRCNT":              0x5090_01A8, # 509001A8
    "CNNx16_2_L7_PRCNT":              0x5090_01AC, # 509001AC
    "CNNx16_2_L8_PRCNT":              0x5090_01B0, # 509001B0
    "CNNx16_2_L9_PRCNT":              0x5090_01B4, # 509001B4
    "CNNx16_2_L10_PRCNT":             0x5090_01B8, # 509001B8
    "CNNx16_2_L11_PRCNT":             0x5090_01BC, # 509001BC
    "CNNx16_2_L12_PRCNT":             0x5090_01C0, # 509001C0
    "CNNx16_2_L13_PRCNT":             0x5090_01C4, # 509001C4
    "CNNx16_2_L14_PRCNT":             0x5090_01C8, # 509001C8
    "CNNx16_2_L15_PRCNT":             0x5090_01CC, # 509001CC
    "CNNx16_2_L0_PCCNT":              0x5090_0210, # 50900210
    "CNNx16_2_L1_PCCNT":              0x5090_0214, # 50900214
    "CNNx16_2_L2_PCCNT":              0x5090_0218, # 50900218
    "CNNx16_2_L3_PCCNT":              0x5090_021C, # 5090021C
    "CNNx16_2_L4_PCCNT":              0x5090_0220, # 50900220
    "CNNx16_2_L5_PCCNT":              0x5090_0224, # 50900224
    "CNNx16_2_L6_PCCNT":              0x5090_0228, # 50900228
    "CNNx16_2_L7_PCCNT":              0x5090_022C, # 5090022C
    "CNNx16_2_L8_PCCNT":              0x5090_0230, # 50900230
    "CNNx16_2_L9_PCCNT":              0x5090_0234, # 50900234
    "CNNx16_2_L10_PCCNT":             0x5090_0238, # 50900238
    "CNNx16_2_L11_PCCNT":             0x5090_023C, # 5090023C
    "CNNx16_2_L12_PCCNT":             0x5090_0240, # 50900240
    "CNNx16_2_L13_PCCNT":             0x5090_0244, # 50900244
    "CNNx16_2_L14_PCCNT":             0x5090_0248, # 50900248
    "CNNx16_2_L15_PCCNT":             0x5090_024C, # 5090024C
    "CNNx16_2_L0_STRIDE":             0x5090_0290, # 50900290
    "CNNx16_2_L1_STRIDE":             0x5090_0294, # 50900294
    "CNNx16_2_L2_STRIDE":             0x5090_0298, # 50900298
    "CNNx16_2_L3_STRIDE":             0x5090_029C, # 5090029C
    "CNNx16_2_L4_STRIDE":             0x5090_02A0, # 509002A0
    "CNNx16_2_L5_STRIDE":             0x5090_02A4, # 509002A4
    "CNNx16_2_L6_STRIDE":             0x5090_02A8, # 509002A8
    "CNNx16_2_L7_STRIDE":             0x5090_02AC, # 509002AC
    "CNNx16_2_L8_STRIDE":             0x5090_02B0, # 509002B0
    "CNNx16_2_L9_STRIDE":             0x5090_02B4, # 509002B4
    "CNNx16_2_L10_STRIDE":            0x5090_02B8, # 509002B8
    "CNNx16_2_L11_STRIDE":            0x5090_02BC, # 509002BC
    "CNNx16_2_L12_STRIDE":            0x5090_02C0, # 509002C0
    "CNNx16_2_L13_STRIDE":            0x5090_02C4, # 509002C4
    "CNNx16_2_L14_STRIDE":            0x5090_02C8, # 509002C8
    "CNNx16_2_L15_STRIDE":            0x5090_02CC, # 509002CC
    "CNNx16_2_L0_WPTR_BASE":          0x5090_0310, # 50900310
    "CNNx16_2_L1_WPTR_BASE":          0x5090_0314, # 50900314
    "CNNx16_2_L2_WPTR_BASE":          0x5090_0318, # 50900318
    "CNNx16_2_L3_WPTR_BASE":          0x5090_031C, # 5090031C
    "CNNx16_2_L4_WPTR_BASE":          0x5090_0320, # 50900320
    "CNNx16_2_L5_WPTR_BASE":          0x5090_0324, # 50900324
    "CNNx16_2_L6_WPTR_BASE":          0x5090_0328, # 50900328
    "CNNx16_2_L7_WPTR_BASE":          0x5090_032C, # 5090032C
    "CNNx16_2_L8_WPTR_BASE":          0x5090_0330, # 50900330
    "CNNx16_2_L9_WPTR_BASE":          0x5090_0334, # 50900334
    "CNNx16_2_L10_WPTR_BASE":         0x5090_0338, # 50900338
    "CNNx16_2_L11_WPTR_BASE":         0x5090_033C, # 5090033C
    "CNNx16_2_L12_WPTR_BASE":         0x5090_0340, # 50900340
    "CNNx16_2_L13_WPTR_BASE":         0x5090_0344, # 50900344
    "CNNx16_2_L14_WPTR_BASE":         0x5090_0348, # 50900348
    "CNNx16_2_L15_WPTR_BASE":         0x5090_034C, # 5090034C
    "CNNx16_2_L0_WPTR_TOFF":          0x5090_0390, # 50900390
    "CNNx16_2_L1_WPTR_TOFF":          0x5090_0394, # 50900394
    "CNNx16_2_L2_WPTR_TOFF":          0x5090_0398, # 50900398
    "CNNx16_2_L3_WPTR_TOFF":          0x5090_039C, # 5090039C
    "CNNx16_2_L4_WPTR_TOFF":          0x5090_03A0, # 509003A0
    "CNNx16_2_L5_WPTR_TOFF":          0x5090_03A4, # 509003A4
    "CNNx16_2_L6_WPTR_TOFF":          0x5090_03A8, # 509003A8
    "CNNx16_2_L7_WPTR_TOFF":          0x5090_03AC, # 509003AC
    "CNNx16_2_L8_WPTR_TOFF":          0x5090_03B0, # 509003B0
    "CNNx16_2_L9_WPTR_TOFF":          0x5090_03B4, # 509003B4
    "CNNx16_2_L10_WPTR_TOFF":         0x5090_03B8, # 509003B8
    "CNNx16_2_L11_WPTR_TOFF":         0x5090_03BC, # 509003BC
    "CNNx16_2_L12_WPTR_TOFF":         0x5090_03C0, # 509003C0
    "CNNx16_2_L13_WPTR_TOFF":         0x5090_03C4, # 509003C4
    "CNNx16_2_L14_WPTR_TOFF":         0x5090_03C8, # 509003C8
    "CNNx16_2_L15_WPTR_TOFF":         0x5090_03CC, # 509003CC
    "CNNx16_2_L0_WPTR_MOFF":          0x5090_0410, # 50900410
    "CNNx16_2_L1_WPTR_MOFF":          0x5090_0414, # 50900414
    "CNNx16_2_L2_WPTR_MOFF":          0x5090_0418, # 50900418
    "CNNx16_2_L3_WPTR_MOFF":          0x5090_041C, # 5090041C
    "CNNx16_2_L4_WPTR_MOFF":          0x5090_0420, # 50900420
    "CNNx16_2_L5_WPTR_MOFF":          0x5090_0424, # 50900424
    "CNNx16_2_L6_WPTR_MOFF":          0x5090_0428, # 50900428
    "CNNx16_2_L7_WPTR_MOFF":          0x5090_042C, # 5090042C
    "CNNx16_2_L8_WPTR_MOFF":          0x5090_0430, # 50900430
    "CNNx16_2_L9_WPTR_MOFF":          0x5090_0434, # 50900434
    "CNNx16_2_L10_WPTR_MOFF":         0x5090_0438, # 50900438
    "CNNx16_2_L11_WPTR_MOFF":         0x5090_043C, # 5090043C
    "CNNx16_2_L12_WPTR_MOFF":         0x5090_0440, # 50900440
    "CNNx16_2_L13_WPTR_MOFF":         0x5090_0444, # 50900444
    "CNNx16_2_L14_WPTR_MOFF":         0x5090_0448, # 50900448
    "CNNx16_2_L15_WPTR_MOFF":         0x5090_044C, # 5090044C
    "CNNx16_2_L0_WPTR_CHOFF":         0x5090_0490, # 50900490
    "CNNx16_2_L1_WPTR_CHOFF":         0x5090_0494, # 50900494
    "CNNx16_2_L2_WPTR_CHOFF":         0x5090_0498, # 50900498
    "CNNx16_2_L3_WPTR_CHOFF":         0x5090_049C, # 5090049C
    "CNNx16_2_L4_WPTR_CHOFF":         0x5090_04A0, # 509004A0
    "CNNx16_2_L5_WPTR_CHOFF":         0x5090_04A4, # 509004A4
    "CNNx16_2_L6_WPTR_CHOFF":         0x5090_04A8, # 509004A8
    "CNNx16_2_L7_WPTR_CHOFF":         0x5090_04AC, # 509004AC
    "CNNx16_2_L8_WPTR_CHOFF":         0x5090_04B0, # 509004B0
    "CNNx16_2_L9_WPTR_CHOFF":         0x5090_04B4, # 509004B4
    "CNNx16_2_L10_WPTR_CHOFF":        0x5090_04B8, # 509004B8
    "CNNx16_2_L11_WPTR_CHOFF":        0x5090_04BC, # 509004BC
    "CNNx16_2_L12_WPTR_CHOFF":        0x5090_04C0, # 509004C0
    "CNNx16_2_L13_WPTR_CHOFF":        0x5090_04C4, # 509004C4
    "CNNx16_2_L14_WPTR_CHOFF":        0x5090_04C8, # 509004C8
    "CNNx16_2_L15_WPTR_CHOFF":        0x5090_04CC, # 509004CC
    "CNNx16_2_L0_RPTR_BASE":          0x5090_0510, # 50900510
    "CNNx16_2_L1_RPTR_BASE":          0x5090_0514, # 50900514
    "CNNx16_2_L2_RPTR_BASE":          0x5090_0518, # 50900518
    "CNNx16_2_L3_RPTR_BASE":          0x5090_051C, # 5090051C
    "CNNx16_2_L4_RPTR_BASE":          0x5090_0520, # 50900520
    "CNNx16_2_L5_RPTR_BASE":          0x5090_0524, # 50900524
    "CNNx16_2_L6_RPTR_BASE":          0x5090_0528, # 50900528
    "CNNx16_2_L7_RPTR_BASE":          0x5090_052C, # 5090052C
    "CNNx16_2_L8_RPTR_BASE":          0x5090_0530, # 50900530
    "CNNx16_2_L9_RPTR_BASE":          0x5090_0534, # 50900534
    "CNNx16_2_L10_RPTR_BASE":         0x5090_0538, # 50900538
    "CNNx16_2_L11_RPTR_BASE":         0x5090_053C, # 5090053C
    "CNNx16_2_L12_RPTR_BASE":         0x5090_0540, # 50900540
    "CNNx16_2_L13_RPTR_BASE":         0x5090_0544, # 50900544
    "CNNx16_2_L14_RPTR_BASE":         0x5090_0548, # 50900548
    "CNNx16_2_L15_RPTR_BASE":         0x5090_054C, # 5090054C
    "CNNx16_2_L0_LCTRL0":             0x5090_0590, # 50900590
    "CNNx16_2_L1_LCTRL0":             0x5090_0594, # 50900594
    "CNNx16_2_L2_LCTRL0":             0x5090_0598, # 50900598
    "CNNx16_2_L3_LCTRL0":             0x5090_059C, # 5090059C
    "CNNx16_2_L4_LCTRL0":             0x5090_05A0, # 509005A0
    "CNNx16_2_L5_LCTRL0":             0x5090_05A4, # 509005A4
    "CNNx16_2_L6_LCTRL0":             0x5090_05A8, # 509005A8
    "CNNx16_2_L7_LCTRL0":             0x5090_05AC, # 509005AC
    "CNNx16_2_L8_LCTRL0":             0x5090_05B0, # 509005B0
    "CNNx16_2_L9_LCTRL0":             0x5090_05B4, # 509005B4
    "CNNx16_2_L10_LCTRL0":            0x5090_05B8, # 509005B8
    "CNNx16_2_L11_LCTRL0":            0x5090_05BC, # 509005BC
    "CNNx16_2_L12_LCTRL0":            0x5090_05C0, # 509005C0
    "CNNx16_2_L13_LCTRL0":            0x5090_05C4, # 509005C4
    "CNNx16_2_L14_LCTRL0":            0x5090_05C8, # 509005C8
    "CNNx16_2_L15_LCTRL0":            0x5090_05CC, # 509005CC
    "CNNx16_2_L0_MCNT":               0x5090_0610, # 50900610
    "CNNx16_2_L1_MCNT":               0x5090_0614, # 50900614
    "CNNx16_2_L2_MCNT":               0x5090_0618, # 50900618
    "CNNx16_2_L3_MCNT":               0x5090_061C, # 5090061C
    "CNNx16_2_L4_MCNT":               0x5090_0620, # 50900620
    "CNNx16_2_L5_MCNT":               0x5090_0624, # 50900624
    "CNNx16_2_L6_MCNT":               0x5090_0628, # 50900628
    "CNNx16_2_L7_MCNT":               0x5090_062C, # 5090062C
    "CNNx16_2_L8_MCNT":               0x5090_0630, # 50900630
    "CNNx16_2_L9_MCNT":               0x5090_0634, # 50900634
    "CNNx16_2_L10_MCNT":              0x5090_0638, # 50900638
    "CNNx16_2_L11_MCNT":              0x5090_063C, # 5090063C
    "CNNx16_2_L12_MCNT":              0x5090_0640, # 50900640
    "CNNx16_2_L13_MCNT":              0x5090_0644, # 50900644
    "CNNx16_2_L14_MCNT":              0x5090_0648, # 50900648
    "CNNx16_2_L15_MCNT":              0x5090_064C, # 5090064C
    "CNNx16_2_L0_TPTR":               0x5090_0690, # 50900690
    "CNNx16_2_L1_TPTR":               0x5090_0694, # 50900694
    "CNNx16_2_L2_TPTR":               0x5090_0698, # 50900698
    "CNNx16_2_L3_TPTR":               0x5090_069C, # 5090069C
    "CNNx16_2_L4_TPTR":               0x5090_06A0, # 509006A0
    "CNNx16_2_L5_TPTR":               0x5090_06A4, # 509006A4
    "CNNx16_2_L6_TPTR":               0x5090_06A8, # 509006A8
    "CNNx16_2_L7_TPTR":               0x5090_06AC, # 509006AC
    "CNNx16_2_L8_TPTR":               0x5090_06B0, # 509006B0
    "CNNx16_2_L9_TPTR":               0x5090_06B4, # 509006B4
    "CNNx16_2_L10_TPTR":              0x5090_06B8, # 509006B8
    "CNNx16_2_L11_TPTR":              0x5090_06BC, # 509006BC
    "CNNx16_2_L12_TPTR":              0x5090_06C0, # 509006C0
    "CNNx16_2_L13_TPTR":              0x5090_06C4, # 509006C4
    "CNNx16_2_L14_TPTR":              0x5090_06C8, # 509006C8
    "CNNx16_2_L15_TPTR":              0x5090_06CC, # 509006CC
    "CNNx16_2_L0_EN":                 0x5090_0710, # 50900710
    "CNNx16_2_L1_EN":                 0x5090_0714, # 50900714
    "CNNx16_2_L2_EN":                 0x5090_0718, # 50900718
    "CNNx16_2_L3_EN":                 0x5090_071C, # 5090071C
    "CNNx16_2_L4_EN":                 0x5090_0720, # 50900720
    "CNNx16_2_L5_EN":                 0x5090_0724, # 50900724
    "CNNx16_2_L6_EN":                 0x5090_0728, # 50900728
    "CNNx16_2_L7_EN":                 0x5090_072C, # 5090072C
    "CNNx16_2_L8_EN":                 0x5090_0730, # 50900730
    "CNNx16_2_L9_EN":                 0x5090_0734, # 50900734
    "CNNx16_2_L10_EN":                0x5090_0738, # 50900738
    "CNNx16_2_L11_EN":                0x5090_073C, # 5090073C
    "CNNx16_2_L12_EN":                0x5090_0740, # 50900740
    "CNNx16_2_L13_EN":                0x5090_0744, # 50900744
    "CNNx16_2_L14_EN":                0x5090_0748, # 50900748
    "CNNx16_2_L15_EN":                0x5090_074C, # 5090074C
    "CNNx16_2_L0_POST":               0x5090_0790, # 50900790
    "CNNx16_2_L1_POST":               0x5090_0794, # 50900794
    "CNNx16_2_L2_POST":               0x5090_0798, # 50900798
    "CNNx16_2_L3_POST":               0x5090_079C, # 5090079C
    "CNNx16_2_L4_POST":               0x5090_07A0, # 509007A0
    "CNNx16_2_L5_POST":               0x5090_07A4, # 509007A4
    "CNNx16_2_L6_POST":               0x5090_07A8, # 509007A8
    "CNNx16_2_L7_POST":               0x5090_07AC, # 509007AC
    "CNNx16_2_L8_POST":               0x5090_07B0, # 509007B0
    "CNNx16_2_L9_POST":               0x5090_07B4, # 509007B4
    "CNNx16_2_L10_POST":              0x5090_07B8, # 509007B8
    "CNNx16_2_L11_POST":              0x5090_07BC, # 509007BC
    "CNNx16_2_L12_POST":              0x5090_07C0, # 509007C0
    "CNNx16_2_L13_POST":              0x5090_07C4, # 509007C4
    "CNNx16_2_L14_POST":              0x5090_07C8, # 509007C8
    "CNNx16_2_L15_POST":              0x5090_07CC, # 509007CC
    "CNNx16_2_S0_STRM0":              0x5090_0810, # 50900810
    "CNNx16_2_S1_STRM0":              0x5090_0814, # 50900814
    "CNNx16_2_S2_STRM0":              0x5090_0818, # 50900818
    "CNNx16_2_S3_STRM0":              0x5090_081C, # 5090081C
    "CNNx16_2_S4_STRM0":              0x5090_0820, # 50900820
    "CNNx16_2_S5_STRM0":              0x5090_0824, # 50900824
    "CNNx16_2_S6_STRM0":              0x5090_0828, # 50900828
    "CNNx16_2_S7_STRM0":              0x5090_082C, # 5090082C
    "CNNx16_2_S0_STRM1":              0x5090_0890, # 50900890
    "CNNx16_2_S1_STRM1":              0x5090_0894, # 50900894
    "CNNx16_2_S2_STRM1":              0x5090_0898, # 50900898
    "CNNx16_2_S3_STRM1":              0x5090_089C, # 5090089C
    "CNNx16_2_S4_STRM1":              0x5090_08A0, # 509008A0
    "CNNx16_2_S5_STRM1":              0x5090_08A4, # 509008A4
    "CNNx16_2_S6_STRM1":              0x5090_08A8, # 509008A8
    "CNNx16_2_S7_STRM1":              0x5090_08AC, # 509008AC
    "CNNx16_2_S0_FBUF":               0x5090_0910, # 50900910
    "CNNx16_2_S1_FBUF":               0x5090_0914, # 50900914
    "CNNx16_2_S2_FBUF":               0x5090_0918, # 50900918
    "CNNx16_2_S3_FBUF":               0x5090_091C, # 5090091C
    "CNNx16_2_S4_FBUF":               0x5090_0920, # 50900920
    "CNNx16_2_S5_FBUF":               0x5090_0924, # 50900924
    "CNNx16_2_S6_FBUF":               0x5090_0928, # 50900928
    "CNNx16_2_S7_FBUF":               0x5090_092C, # 5090092C
    "CNNx16_2_IFRM":                  0x5090_0990, # 50900990
    "CNNx16_2_L0_LCTRL1":             0x5090_0A10, # 50900A10
    "CNNx16_2_L1_LCTRL1":             0x5090_0A14, # 50900A14
    "CNNx16_2_L2_LCTRL1":             0x5090_0A18, # 50900A18
    "CNNx16_2_L3_LCTRL1":             0x5090_0A1C, # 50900A1C
    "CNNx16_2_L4_LCTRL1":             0x5090_0A20, # 50900A20
    "CNNx16_2_L5_LCTRL1":             0x5090_0A24, # 50900A24
    "CNNx16_2_L6_LCTRL1":             0x5090_0A28, # 50900A28
    "CNNx16_2_L7_LCTRL1":             0x5090_0A2C, # 50900A2C
    "CNNx16_2_L8_LCTRL1":             0x5090_0A30, # 50900A30
    "CNNx16_2_L9_LCTRL1":             0x5090_0A34, # 50900A34
    "CNNx16_2_L10_LCTRL1":            0x5090_0A38, # 50900A38
    "CNNx16_2_L11_LCTRL1":            0x5090_0A3C, # 50900A3C
    "CNNx16_2_L12_LCTRL1":            0x5090_0A40, # 50900A40
    "CNNx16_2_L13_LCTRL1":            0x5090_0A44, # 50900A44
    "CNNx16_2_L14_LCTRL1":            0x5090_0A48, # 50900A48
    "CNNx16_2_L15_LCTRL1":            0x5090_0A4C, # 50900A4C
    "CNNx16_2_MLAT":                  0x5090_1000, # 50901000
    "CNNx16_3_CTRL":                  0x50D0_0000, # 50D00000
    "CNNx16_3_SRAM":                  0x50D0_0004, # 50D00004
    "CNNx16_3_LCNT_MAX":              0x50D0_0008, # 50D00008
    "CNNx16_3_TEST":                  0x50D0_000C, # 50D0000C
    "CNNx16_3_L0_RCNT":               0x50D0_0010, # 50D00010
    "CNNx16_3_L1_RCNT":               0x50D0_0014, # 50D00014
    "CNNx16_3_L2_RCNT":               0x50D0_0018, # 50D00018
    "CNNx16_3_L3_RCNT":               0x50D0_001C, # 50D0001C
    "CNNx16_3_L4_RCNT":               0x50D0_0020, # 50D00020
    "CNNx16_3_L5_RCNT":               0x50D0_0024, # 50D00024
    "CNNx16_3_L6_RCNT":               0x50D0_0028, # 50D00028
    "CNNx16_3_L7_RCNT":               0x50D0_002C, # 50D0002C
    "CNNx16_3_L8_RCNT":               0x50D0_0030, # 50D00030
    "CNNx16_3_L9_RCNT":               0x50D0_0034, # 50D00034
    "CNNx16_3_L10_RCNT":              0x50D0_0038, # 50D00038
    "CNNx16_3_L11_RCNT":              0x50D0_003C, # 50D0003C
    "CNNx16_3_L12_RCNT":              0x50D0_0040, # 50D00040
    "CNNx16_3_L13_RCNT":              0x50D0_0044, # 50D00044
    "CNNx16_3_L14_RCNT":              0x50D0_0048, # 50D00048
    "CNNx16_3_L15_RCNT":              0x50D0_004C, # 50D0004C
    "CNNx16_3_L0_CCNT":               0x50D0_0090, # 50D00090
    "CNNx16_3_L1_CCNT":               0x50D0_0094, # 50D00094
    "CNNx16_3_L2_CCNT":               0x50D0_0098, # 50D00098
    "CNNx16_3_L3_CCNT":               0x50D0_009C, # 50D0009C
    "CNNx16_3_L4_CCNT":               0x50D0_00A0, # 50D000A0
    "CNNx16_3_L5_CCNT":               0x50D0_00A4, # 50D000A4
    "CNNx16_3_L6_CCNT":               0x50D0_00A8, # 50D000A8
    "CNNx16_3_L7_CCNT":               0x50D0_00AC, # 50D000AC
    "CNNx16_3_L8_CCNT":               0x50D0_00B0, # 50D000B0
    "CNNx16_3_L9_CCNT":               0x50D0_00B4, # 50D000B4
    "CNNx16_3_L10_CCNT":              0x50D0_00B8, # 50D000B8
    "CNNx16_3_L11_CCNT":              0x50D0_00BC, # 50D000BC
    "CNNx16_3_L12_CCNT":              0x50D0_00C0, # 50D000C0
    "CNNx16_3_L13_CCNT":              0x50D0_00C4, # 50D000C4
    "CNNx16_3_L14_CCNT":              0x50D0_00C8, # 50D000C8
    "CNNx16_3_L15_CCNT":              0x50D0_00CC, # 50D000CC
    "CNNx16_3_L0_ONED":               0x50D0_0110, # 50D00110
    "CNNx16_3_L1_ONED":               0x50D0_0114, # 50D00114
    "CNNx16_3_L2_ONED":               0x50D0_0118, # 50D00118
    "CNNx16_3_L3_ONED":               0x50D0_011C, # 50D0011C
    "CNNx16_3_L4_ONED":               0x50D0_0120, # 50D00120
    "CNNx16_3_L5_ONED":               0x50D0_0124, # 50D00124
    "CNNx16_3_L6_ONED":               0x50D0_0128, # 50D00128
    "CNNx16_3_L7_ONED":               0x50D0_012C, # 50D0012C
    "CNNx16_3_L8_ONED":               0x50D0_0130, # 50D00130
    "CNNx16_3_L9_ONED":               0x50D0_0134, # 50D00134
    "CNNx16_3_L10_ONED":              0x50D0_0138, # 50D00138
    "CNNx16_3_L11_ONED":              0x50D0_013C, # 50D0013C
    "CNNx16_3_L12_ONED":              0x50D0_0140, # 50D00140
    "CNNx16_3_L13_ONED":              0x50D0_0144, # 50D00144
    "CNNx16_3_L14_ONED":              0x50D0_0148, # 50D00148
    "CNNx16_3_L15_ONED":              0x50D0_014C, # 50D0014C
    "CNNx16_3_L0_PRCNT":              0x50D0_0190, # 50D00190
    "CNNx16_3_L1_PRCNT":              0x50D0_0194, # 50D00194
    "CNNx16_3_L2_PRCNT":              0x50D0_0198, # 50D00198
    "CNNx16_3_L3_PRCNT":              0x50D0_019C, # 50D0019C
    "CNNx16_3_L4_PRCNT":              0x50D0_01A0, # 50D001A0
    "CNNx16_3_L5_PRCNT":              0x50D0_01A4, # 50D001A4
    "CNNx16_3_L6_PRCNT":              0x50D0_01A8, # 50D001A8
    "CNNx16_3_L7_PRCNT":              0x50D0_01AC, # 50D001AC
    "CNNx16_3_L8_PRCNT":              0x50D0_01B0, # 50D001B0
    "CNNx16_3_L9_PRCNT":              0x50D0_01B4, # 50D001B4
    "CNNx16_3_L10_PRCNT":             0x50D0_01B8, # 50D001B8
    "CNNx16_3_L11_PRCNT":             0x50D0_01BC, # 50D001BC
    "CNNx16_3_L12_PRCNT":             0x50D0_01C0, # 50D001C0
    "CNNx16_3_L13_PRCNT":             0x50D0_01C4, # 50D001C4
    "CNNx16_3_L14_PRCNT":             0x50D0_01C8, # 50D001C8
    "CNNx16_3_L15_PRCNT":             0x50D0_01CC, # 50D001CC
    "CNNx16_3_L0_PCCNT":              0x50D0_0210, # 50D00210
    "CNNx16_3_L1_PCCNT":              0x50D0_0214, # 50D00214
    "CNNx16_3_L2_PCCNT":              0x50D0_0218, # 50D00218
    "CNNx16_3_L3_PCCNT":              0x50D0_021C, # 50D0021C
    "CNNx16_3_L4_PCCNT":              0x50D0_0220, # 50D00220
    "CNNx16_3_L5_PCCNT":              0x50D0_0224, # 50D00224
    "CNNx16_3_L6_PCCNT":              0x50D0_0228, # 50D00228
    "CNNx16_3_L7_PCCNT":              0x50D0_022C, # 50D0022C
    "CNNx16_3_L8_PCCNT":              0x50D0_0230, # 50D00230
    "CNNx16_3_L9_PCCNT":              0x50D0_0234, # 50D00234
    "CNNx16_3_L10_PCCNT":             0x50D0_0238, # 50D00238
    "CNNx16_3_L11_PCCNT":             0x50D0_023C, # 50D0023C
    "CNNx16_3_L12_PCCNT":             0x50D0_0240, # 50D00240
    "CNNx16_3_L13_PCCNT":             0x50D0_0244, # 50D00244
    "CNNx16_3_L14_PCCNT":             0x50D0_0248, # 50D00248
    "CNNx16_3_L15_PCCNT":             0x50D0_024C, # 50D0024C
    "CNNx16_3_L0_STRIDE":             0x50D0_0290, # 50D00290
    "CNNx16_3_L1_STRIDE":             0x50D0_0294, # 50D00294
    "CNNx16_3_L2_STRIDE":             0x50D0_0298, # 50D00298
    "CNNx16_3_L3_STRIDE":             0x50D0_029C, # 50D0029C
    "CNNx16_3_L4_STRIDE":             0x50D0_02A0, # 50D002A0
    "CNNx16_3_L5_STRIDE":             0x50D0_02A4, # 50D002A4
    "CNNx16_3_L6_STRIDE":             0x50D0_02A8, # 50D002A8
    "CNNx16_3_L7_STRIDE":             0x50D0_02AC, # 50D002AC
    "CNNx16_3_L8_STRIDE":             0x50D0_02B0, # 50D002B0
    "CNNx16_3_L9_STRIDE":             0x50D0_02B4, # 50D002B4
    "CNNx16_3_L10_STRIDE":            0x50D0_02B8, # 50D002B8
    "CNNx16_3_L11_STRIDE":            0x50D0_02BC, # 50D002BC
    "CNNx16_3_L12_STRIDE":            0x50D0_02C0, # 50D002C0
    "CNNx16_3_L13_STRIDE":            0x50D0_02C4, # 50D002C4
    "CNNx16_3_L14_STRIDE":            0x50D0_02C8, # 50D002C8
    "CNNx16_3_L15_STRIDE":            0x50D0_02CC, # 50D002CC
    "CNNx16_3_L0_WPTR_BASE":          0x50D0_0310, # 50D00310
    "CNNx16_3_L1_WPTR_BASE":          0x50D0_0314, # 50D00314
    "CNNx16_3_L2_WPTR_BASE":          0x50D0_0318, # 50D00318
    "CNNx16_3_L3_WPTR_BASE":          0x50D0_031C, # 50D0031C
    "CNNx16_3_L4_WPTR_BASE":          0x50D0_0320, # 50D00320
    "CNNx16_3_L5_WPTR_BASE":          0x50D0_0324, # 50D00324
    "CNNx16_3_L6_WPTR_BASE":          0x50D0_0328, # 50D00328
    "CNNx16_3_L7_WPTR_BASE":          0x50D0_032C, # 50D0032C
    "CNNx16_3_L8_WPTR_BASE":          0x50D0_0330, # 50D00330
    "CNNx16_3_L9_WPTR_BASE":          0x50D0_0334, # 50D00334
    "CNNx16_3_L10_WPTR_BASE":         0x50D0_0338, # 50D00338
    "CNNx16_3_L11_WPTR_BASE":         0x50D0_033C, # 50D0033C
    "CNNx16_3_L12_WPTR_BASE":         0x50D0_0340, # 50D00340
    "CNNx16_3_L13_WPTR_BASE":         0x50D0_0344, # 50D00344
    "CNNx16_3_L14_WPTR_BASE":         0x50D0_0348, # 50D00348
    "CNNx16_3_L15_WPTR_BASE":         0x50D0_034C, # 50D0034C
    "CNNx16_3_L0_WPTR_TOFF":          0x50D0_0390, # 50D00390
    "CNNx16_3_L1_WPTR_TOFF":          0x50D0_0394, # 50D00394
    "CNNx16_3_L2_WPTR_TOFF":          0x50D0_0398, # 50D00398
    "CNNx16_3_L3_WPTR_TOFF":          0x50D0_039C, # 50D0039C
    "CNNx16_3_L4_WPTR_TOFF":          0x50D0_03A0, # 50D003A0
    "CNNx16_3_L5_WPTR_TOFF":          0x50D0_03A4, # 50D003A4
    "CNNx16_3_L6_WPTR_TOFF":          0x50D0_03A8, # 50D003A8
    "CNNx16_3_L7_WPTR_TOFF":          0x50D0_03AC, # 50D003AC
    "CNNx16_3_L8_WPTR_TOFF":          0x50D0_03B0, # 50D003B0
    "CNNx16_3_L9_WPTR_TOFF":          0x50D0_03B4, # 50D003B4
    "CNNx16_3_L10_WPTR_TOFF":         0x50D0_03B8, # 50D003B8
    "CNNx16_3_L11_WPTR_TOFF":         0x50D0_03BC, # 50D003BC
    "CNNx16_3_L12_WPTR_TOFF":         0x50D0_03C0, # 50D003C0
    "CNNx16_3_L13_WPTR_TOFF":         0x50D0_03C4, # 50D003C4
    "CNNx16_3_L14_WPTR_TOFF":         0x50D0_03C8, # 50D003C8
    "CNNx16_3_L15_WPTR_TOFF":         0x50D0_03CC, # 50D003CC
    "CNNx16_3_L0_WPTR_MOFF":          0x50D0_0410, # 50D00410
    "CNNx16_3_L1_WPTR_MOFF":          0x50D0_0414, # 50D00414
    "CNNx16_3_L2_WPTR_MOFF":          0x50D0_0418, # 50D00418
    "CNNx16_3_L3_WPTR_MOFF":          0x50D0_041C, # 50D0041C
    "CNNx16_3_L4_WPTR_MOFF":          0x50D0_0420, # 50D00420
    "CNNx16_3_L5_WPTR_MOFF":          0x50D0_0424, # 50D00424
    "CNNx16_3_L6_WPTR_MOFF":          0x50D0_0428, # 50D00428
    "CNNx16_3_L7_WPTR_MOFF":          0x50D0_042C, # 50D0042C
    "CNNx16_3_L8_WPTR_MOFF":          0x50D0_0430, # 50D00430
    "CNNx16_3_L9_WPTR_MOFF":          0x50D0_0434, # 50D00434
    "CNNx16_3_L10_WPTR_MOFF":         0x50D0_0438, # 50D00438
    "CNNx16_3_L11_WPTR_MOFF":         0x50D0_043C, # 50D0043C
    "CNNx16_3_L12_WPTR_MOFF":         0x50D0_0440, # 50D00440
    "CNNx16_3_L13_WPTR_MOFF":         0x50D0_0444, # 50D00444
    "CNNx16_3_L14_WPTR_MOFF":         0x50D0_0448, # 50D00448
    "CNNx16_3_L15_WPTR_MOFF":         0x50D0_044C, # 50D0044C
    "CNNx16_3_L0_WPTR_CHOFF":         0x50D0_0490, # 50D00490
    "CNNx16_3_L1_WPTR_CHOFF":         0x50D0_0494, # 50D00494
    "CNNx16_3_L2_WPTR_CHOFF":         0x50D0_0498, # 50D00498
    "CNNx16_3_L3_WPTR_CHOFF":         0x50D0_049C, # 50D0049C
    "CNNx16_3_L4_WPTR_CHOFF":         0x50D0_04A0, # 50D004A0
    "CNNx16_3_L5_WPTR_CHOFF":         0x50D0_04A4, # 50D004A4
    "CNNx16_3_L6_WPTR_CHOFF":         0x50D0_04A8, # 50D004A8
    "CNNx16_3_L7_WPTR_CHOFF":         0x50D0_04AC, # 50D004AC
    "CNNx16_3_L8_WPTR_CHOFF":         0x50D0_04B0, # 50D004B0
    "CNNx16_3_L9_WPTR_CHOFF":         0x50D0_04B4, # 50D004B4
    "CNNx16_3_L10_WPTR_CHOFF":        0x50D0_04B8, # 50D004B8
    "CNNx16_3_L11_WPTR_CHOFF":        0x50D0_04BC, # 50D004BC
    "CNNx16_3_L12_WPTR_CHOFF":        0x50D0_04C0, # 50D004C0
    "CNNx16_3_L13_WPTR_CHOFF":        0x50D0_04C4, # 50D004C4
    "CNNx16_3_L14_WPTR_CHOFF":        0x50D0_04C8, # 50D004C8
    "CNNx16_3_L15_WPTR_CHOFF":        0x50D0_04CC, # 50D004CC
    "CNNx16_3_L0_RPTR_BASE":          0x50D0_0510, # 50D00510
    "CNNx16_3_L1_RPTR_BASE":          0x50D0_0514, # 50D00514
    "CNNx16_3_L2_RPTR_BASE":          0x50D0_0518, # 50D00518
    "CNNx16_3_L3_RPTR_BASE":          0x50D0_051C, # 50D0051C
    "CNNx16_3_L4_RPTR_BASE":          0x50D0_0520, # 50D00520
    "CNNx16_3_L5_RPTR_BASE":          0x50D0_0524, # 50D00524
    "CNNx16_3_L6_RPTR_BASE":          0x50D0_0528, # 50D00528
    "CNNx16_3_L7_RPTR_BASE":          0x50D0_052C, # 50D0052C
    "CNNx16_3_L8_RPTR_BASE":          0x50D0_0530, # 50D00530
    "CNNx16_3_L9_RPTR_BASE":          0x50D0_0534, # 50D00534
    "CNNx16_3_L10_RPTR_BASE":         0x50D0_0538, # 50D00538
    "CNNx16_3_L11_RPTR_BASE":         0x50D0_053C, # 50D0053C
    "CNNx16_3_L12_RPTR_BASE":         0x50D0_0540, # 50D00540
    "CNNx16_3_L13_RPTR_BASE":         0x50D0_0544, # 50D00544
    "CNNx16_3_L14_RPTR_BASE":         0x50D0_0548, # 50D00548
    "CNNx16_3_L15_RPTR_BASE":         0x50D0_054C, # 50D0054C
    "CNNx16_3_L0_LCTRL0":             0x50D0_0590, # 50D00590
    "CNNx16_3_L1_LCTRL0":             0x50D0_0594, # 50D00594
    "CNNx16_3_L2_LCTRL0":             0x50D0_0598, # 50D00598
    "CNNx16_3_L3_LCTRL0":             0x50D0_059C, # 50D0059C
    "CNNx16_3_L4_LCTRL0":             0x50D0_05A0, # 50D005A0
    "CNNx16_3_L5_LCTRL0":             0x50D0_05A4, # 50D005A4
    "CNNx16_3_L6_LCTRL0":             0x50D0_05A8, # 50D005A8
    "CNNx16_3_L7_LCTRL0":             0x50D0_05AC, # 50D005AC
    "CNNx16_3_L8_LCTRL0":             0x50D0_05B0, # 50D005B0
    "CNNx16_3_L9_LCTRL0":             0x50D0_05B4, # 50D005B4
    "CNNx16_3_L10_LCTRL0":            0x50D0_05B8, # 50D005B8
    "CNNx16_3_L11_LCTRL0":            0x50D0_05BC, # 50D005BC
    "CNNx16_3_L12_LCTRL0":            0x50D0_05C0, # 50D005C0
    "CNNx16_3_L13_LCTRL0":            0x50D0_05C4, # 50D005C4
    "CNNx16_3_L14_LCTRL0":            0x50D0_05C8, # 50D005C8
    "CNNx16_3_L15_LCTRL0":            0x50D0_05CC, # 50D005CC
    "CNNx16_3_L0_MCNT":               0x50D0_0610, # 50D00610
    "CNNx16_3_L1_MCNT":               0x50D0_0614, # 50D00614
    "CNNx16_3_L2_MCNT":               0x50D0_0618, # 50D00618
    "CNNx16_3_L3_MCNT":               0x50D0_061C, # 50D0061C
    "CNNx16_3_L4_MCNT":               0x50D0_0620, # 50D00620
    "CNNx16_3_L5_MCNT":               0x50D0_0624, # 50D00624
    "CNNx16_3_L6_MCNT":               0x50D0_0628, # 50D00628
    "CNNx16_3_L7_MCNT":               0x50D0_062C, # 50D0062C
    "CNNx16_3_L8_MCNT":               0x50D0_0630, # 50D00630
    "CNNx16_3_L9_MCNT":               0x50D0_0634, # 50D00634
    "CNNx16_3_L10_MCNT":              0x50D0_0638, # 50D00638
    "CNNx16_3_L11_MCNT":              0x50D0_063C, # 50D0063C
    "CNNx16_3_L12_MCNT":              0x50D0_0640, # 50D00640
    "CNNx16_3_L13_MCNT":              0x50D0_0644, # 50D00644
    "CNNx16_3_L14_MCNT":              0x50D0_0648, # 50D00648
    "CNNx16_3_L15_MCNT":              0x50D0_064C, # 50D0064C
    "CNNx16_3_L0_TPTR":               0x50D0_0690, # 50D00690
    "CNNx16_3_L1_TPTR":               0x50D0_0694, # 50D00694
    "CNNx16_3_L2_TPTR":               0x50D0_0698, # 50D00698
    "CNNx16_3_L3_TPTR":               0x50D0_069C, # 50D0069C
    "CNNx16_3_L4_TPTR":               0x50D0_06A0, # 50D006A0
    "CNNx16_3_L5_TPTR":               0x50D0_06A4, # 50D006A4
    "CNNx16_3_L6_TPTR":               0x50D0_06A8, # 50D006A8
    "CNNx16_3_L7_TPTR":               0x50D0_06AC, # 50D006AC
    "CNNx16_3_L8_TPTR":               0x50D0_06B0, # 50D006B0
    "CNNx16_3_L9_TPTR":               0x50D0_06B4, # 50D006B4
    "CNNx16_3_L10_TPTR":              0x50D0_06B8, # 50D006B8
    "CNNx16_3_L11_TPTR":              0x50D0_06BC, # 50D006BC
    "CNNx16_3_L12_TPTR":              0x50D0_06C0, # 50D006C0
    "CNNx16_3_L13_TPTR":              0x50D0_06C4, # 50D006C4
    "CNNx16_3_L14_TPTR":              0x50D0_06C8, # 50D006C8
    "CNNx16_3_L15_TPTR":              0x50D0_06CC, # 50D006CC
    "CNNx16_3_L0_EN":                 0x50D0_0710, # 50D00710
    "CNNx16_3_L1_EN":                 0x50D0_0714, # 50D00714
    "CNNx16_3_L2_EN":                 0x50D0_0718, # 50D00718
    "CNNx16_3_L3_EN":                 0x50D0_071C, # 50D0071C
    "CNNx16_3_L4_EN":                 0x50D0_0720, # 50D00720
    "CNNx16_3_L5_EN":                 0x50D0_0724, # 50D00724
    "CNNx16_3_L6_EN":                 0x50D0_0728, # 50D00728
    "CNNx16_3_L7_EN":                 0x50D0_072C, # 50D0072C
    "CNNx16_3_L8_EN":                 0x50D0_0730, # 50D00730
    "CNNx16_3_L9_EN":                 0x50D0_0734, # 50D00734
    "CNNx16_3_L10_EN":                0x50D0_0738, # 50D00738
    "CNNx16_3_L11_EN":                0x50D0_073C, # 50D0073C
    "CNNx16_3_L12_EN":                0x50D0_0740, # 50D00740
    "CNNx16_3_L13_EN":                0x50D0_0744, # 50D00744
    "CNNx16_3_L14_EN":                0x50D0_0748, # 50D00748
    "CNNx16_3_L15_EN":                0x50D0_074C, # 50D0074C
    "CNNx16_3_L0_POST":               0x50D0_0790, # 50D00790
    "CNNx16_3_L1_POST":               0x50D0_0794, # 50D00794
    "CNNx16_3_L2_POST":               0x50D0_0798, # 50D00798
    "CNNx16_3_L3_POST":               0x50D0_079C, # 50D0079C
    "CNNx16_3_L4_POST":               0x50D0_07A0, # 50D007A0
    "CNNx16_3_L5_POST":               0x50D0_07A4, # 50D007A4
    "CNNx16_3_L6_POST":               0x50D0_07A8, # 50D007A8
    "CNNx16_3_L7_POST":               0x50D0_07AC, # 50D007AC
    "CNNx16_3_L8_POST":               0x50D0_07B0, # 50D007B0
    "CNNx16_3_L9_POST":               0x50D0_07B4, # 50D007B4
    "CNNx16_3_L10_POST":              0x50D0_07B8, # 50D007B8
    "CNNx16_3_L11_POST":              0x50D0_07BC, # 50D007BC
    "CNNx16_3_L12_POST":              0x50D0_07C0, # 50D007C0
    "CNNx16_3_L13_POST":              0x50D0_07C4, # 50D007C4
    "CNNx16_3_L14_POST":              0x50D0_07C8, # 50D007C8
    "CNNx16_3_L15_POST":              0x50D0_07CC, # 50D007CC
    "CNNx16_3_S0_STRM0":              0x50D0_0810, # 50D00810
    "CNNx16_3_S1_STRM0":              0x50D0_0814, # 50D00814
    "CNNx16_3_S2_STRM0":              0x50D0_0818, # 50D00818
    "CNNx16_3_S3_STRM0":              0x50D0_081C, # 50D0081C
    "CNNx16_3_S4_STRM0":              0x50D0_0820, # 50D00820
    "CNNx16_3_S5_STRM0":              0x50D0_0824, # 50D00824
    "CNNx16_3_S6_STRM0":              0x50D0_0828, # 50D00828
    "CNNx16_3_S7_STRM0":              0x50D0_082C, # 50D0082C
    "CNNx16_3_S0_STRM1":              0x50D0_0890, # 50D00890
    "CNNx16_3_S1_STRM1":              0x50D0_0894, # 50D00894
    "CNNx16_3_S2_STRM1":              0x50D0_0898, # 50D00898
    "CNNx16_3_S3_STRM1":              0x50D0_089C, # 50D0089C
    "CNNx16_3_S4_STRM1":              0x50D0_08A0, # 50D008A0
    "CNNx16_3_S5_STRM1":              0x50D0_08A4, # 50D008A4
    "CNNx16_3_S6_STRM1":              0x50D0_08A8, # 50D008A8
    "CNNx16_3_S7_STRM1":              0x50D0_08AC, # 50D008AC
    "CNNx16_3_S0_FBUF":               0x50D0_0910, # 50D00910
    "CNNx16_3_S1_FBUF":               0x50D0_0914, # 50D00914
    "CNNx16_3_S2_FBUF":               0x50D0_0918, # 50D00918
    "CNNx16_3_S3_FBUF":               0x50D0_091C, # 50D0091C
    "CNNx16_3_S4_FBUF":               0x50D0_0920, # 50D00920
    "CNNx16_3_S5_FBUF":               0x50D0_0924, # 50D00924
    "CNNx16_3_S6_FBUF":               0x50D0_0928, # 50D00928
    "CNNx16_3_S7_FBUF":               0x50D0_092C, # 50D0092C
    "CNNx16_3_IFRM":                  0x50D0_0990, # 50D00990
    "CNNx16_3_L0_LCTRL1":             0x50D0_0A10, # 50D00A10
    "CNNx16_3_L1_LCTRL1":             0x50D0_0A14, # 50D00A14
    "CNNx16_3_L2_LCTRL1":             0x50D0_0A18, # 50D00A18
    "CNNx16_3_L3_LCTRL1":             0x50D0_0A1C, # 50D00A1C
    "CNNx16_3_L4_LCTRL1":             0x50D0_0A20, # 50D00A20
    "CNNx16_3_L5_LCTRL1":             0x50D0_0A24, # 50D00A24
    "CNNx16_3_L6_LCTRL1":             0x50D0_0A28, # 50D00A28
    "CNNx16_3_L7_LCTRL1":             0x50D0_0A2C, # 50D00A2C
    "CNNx16_3_L8_LCTRL1":             0x50D0_0A30, # 50D00A30
    "CNNx16_3_L9_LCTRL1":             0x50D0_0A34, # 50D00A34
    "CNNx16_3_L10_LCTRL1":            0x50D0_0A38, # 50D00A38
    "CNNx16_3_L11_LCTRL1":            0x50D0_0A3C, # 50D00A3C
    "CNNx16_3_L12_LCTRL1":            0x50D0_0A40, # 50D00A40
    "CNNx16_3_L13_LCTRL1":            0x50D0_0A44, # 50D00A44
    "CNNx16_3_L14_LCTRL1":            0x50D0_0A48, # 50D00A48
    "CNNx16_3_L15_LCTRL1":            0x50D0_0A4C, # 50D00A4C
    "CNNx16_3_MLAT":                  0x50D0_1000, # 50D01000
} # registers


memory = {
    "CNNx16_0_BIAS":                  0x5010_8000, # 50108000
    "CNNx16_0_P0_TRAM":               0x5011_0000, # 50110000
    "CNNx16_0_P1_TRAM":               0x5011_4000, # 50114000
    "CNNx16_0_P2_TRAM":               0x5011_8000, # 50118000
    "CNNx16_0_P3_TRAM":               0x5011_C000, # 5011C000
    "CNNx16_0_P4_TRAM":               0x5012_0000, # 50120000
    "CNNx16_0_P5_TRAM":               0x5012_4000, # 50124000
    "CNNx16_0_P6_TRAM":               0x5012_8000, # 50128000
    "CNNx16_0_P7_TRAM":               0x5012_C000, # 5012C000
    "CNNx16_0_P8_TRAM":               0x5013_0000, # 50130000
    "CNNx16_0_P9_TRAM":               0x5013_4000, # 50134000
    "CNNx16_0_P10_TRAM":              0x5013_8000, # 50138000
    "CNNx16_0_P11_TRAM":              0x5013_C000, # 5013C000
    "CNNx16_0_P12_TRAM":              0x5014_0000, # 50140000
    "CNNx16_0_P13_TRAM":              0x5014_4000, # 50144000
    "CNNx16_0_P14_TRAM":              0x5014_8000, # 50148000
    "CNNx16_0_P15_TRAM":              0x5014_C000, # 5014C000
    "CNNx16_0_P0_MRAM":               0x5018_0000, # 50180000
    "CNNx16_0_P1_MRAM":               0x5018_4000, # 50184000
    "CNNx16_0_P2_MRAM":               0x5018_8000, # 50188000
    "CNNx16_0_P3_MRAM":               0x5018_C000, # 5018C000
    "CNNx16_0_P4_MRAM":               0x5019_0000, # 50190000
    "CNNx16_0_P5_MRAM":               0x5019_4000, # 50194000
    "CNNx16_0_P6_MRAM":               0x5019_8000, # 50198000
    "CNNx16_0_P7_MRAM":               0x5019_C000, # 5019C000
    "CNNx16_0_P8_MRAM":               0x501A_0000, # 501A0000
    "CNNx16_0_P9_MRAM":               0x501A_4000, # 501A4000
    "CNNx16_0_P10_MRAM":              0x501A_8000, # 501A8000
    "CNNx16_0_P11_MRAM":              0x501A_C000, # 501AC000
    "CNNx16_0_P12_MRAM":              0x501B_0000, # 501B0000
    "CNNx16_0_P13_MRAM":              0x501B_4000, # 501B4000
    "CNNx16_0_P14_MRAM":              0x501B_8000, # 501B8000
    "CNNx16_0_P15_MRAM":              0x501B_C000, # 501BC000
    "CNNx16_0_SRAM":                  0x5040_0000, # 50400000
    "CNNx16_1_BIAS":                  0x5050_8000, # 50508000
    "CNNx16_1_P0_TRAM":               0x5051_0000, # 50510000
    "CNNx16_1_P1_TRAM":               0x5051_4000, # 50514000
    "CNNx16_1_P2_TRAM":               0x5051_8000, # 50518000
    "CNNx16_1_P3_TRAM":               0x5051_C000, # 5051C000
    "CNNx16_1_P4_TRAM":               0x5052_0000, # 50520000
    "CNNx16_1_P5_TRAM":               0x5052_4000, # 50524000
    "CNNx16_1_P6_TRAM":               0x5052_8000, # 50528000
    "CNNx16_1_P7_TRAM":               0x5052_C000, # 5052C000
    "CNNx16_1_P8_TRAM":               0x5053_0000, # 50530000
    "CNNx16_1_P9_TRAM":               0x5053_4000, # 50534000
    "CNNx16_1_P10_TRAM":              0x5053_8000, # 50538000
    "CNNx16_1_P11_TRAM":              0x5053_C000, # 5053C000
    "CNNx16_1_P12_TRAM":              0x5054_0000, # 50540000
    "CNNx16_1_P13_TRAM":              0x5054_4000, # 50544000
    "CNNx16_1_P14_TRAM":              0x5054_8000, # 50548000
    "CNNx16_1_P15_TRAM":              0x5054_C000, # 5054C000
    "CNNx16_1_P0_MRAM":               0x5058_0000, # 50580000
    "CNNx16_1_P1_MRAM":               0x5058_4000, # 50584000
    "CNNx16_1_P2_MRAM":               0x5058_8000, # 50588000
    "CNNx16_1_P3_MRAM":               0x5058_C000, # 5058C000
    "CNNx16_1_P4_MRAM":               0x5059_0000, # 50590000
    "CNNx16_1_P5_MRAM":               0x5059_4000, # 50594000
    "CNNx16_1_P6_MRAM":               0x5059_8000, # 50598000
    "CNNx16_1_P7_MRAM":               0x5059_C000, # 5059C000
    "CNNx16_1_P8_MRAM":               0x505A_0000, # 505A0000
    "CNNx16_1_P9_MRAM":               0x505A_4000, # 505A4000
    "CNNx16_1_P10_MRAM":              0x505A_8000, # 505A8000
    "CNNx16_1_P11_MRAM":              0x505A_C000, # 505AC000
    "CNNx16_1_P12_MRAM":              0x505B_0000, # 505B0000
    "CNNx16_1_P13_MRAM":              0x505B_4000, # 505B4000
    "CNNx16_1_P14_MRAM":              0x505B_8000, # 505B8000
    "CNNx16_1_P15_MRAM":              0x505B_C000, # 505BC000
    "CNNx16_1_SRAM":                  0x5080_0000, # 50800000
    "CNNx16_2_BIAS":                  0x5090_8000, # 50908000
    "CNNx16_2_P0_TRAM":               0x5091_0000, # 50910000
    "CNNx16_2_P1_TRAM":               0x5091_4000, # 50914000
    "CNNx16_2_P2_TRAM":               0x5091_8000, # 50918000
    "CNNx16_2_P3_TRAM":               0x5091_C000, # 5091C000
    "CNNx16_2_P4_TRAM":               0x5092_0000, # 50920000
    "CNNx16_2_P5_TRAM":               0x5092_4000, # 50924000
    "CNNx16_2_P6_TRAM":               0x5092_8000, # 50928000
    "CNNx16_2_P7_TRAM":               0x5092_C000, # 5092C000
    "CNNx16_2_P8_TRAM":               0x5093_0000, # 50930000
    "CNNx16_2_P9_TRAM":               0x5093_4000, # 50934000
    "CNNx16_2_P10_TRAM":              0x5093_8000, # 50938000
    "CNNx16_2_P11_TRAM":              0x5093_C000, # 5093C000
    "CNNx16_2_P12_TRAM":              0x5094_0000, # 50940000
    "CNNx16_2_P13_TRAM":              0x5094_4000, # 50944000
    "CNNx16_2_P14_TRAM":              0x5094_8000, # 50948000
    "CNNx16_2_P15_TRAM":              0x5094_C000, # 5094C000
    "CNNx16_2_P0_MRAM":               0x5098_0000, # 50980000
    "CNNx16_2_P1_MRAM":               0x5098_4000, # 50984000
    "CNNx16_2_P2_MRAM":               0x5098_8000, # 50988000
    "CNNx16_2_P3_MRAM":               0x5098_C000, # 5098C000
    "CNNx16_2_P4_MRAM":               0x5099_0000, # 50990000
    "CNNx16_2_P5_MRAM":               0x5099_4000, # 50994000
    "CNNx16_2_P6_MRAM":               0x5099_8000, # 50998000
    "CNNx16_2_P7_MRAM":               0x5099_C000, # 5099C000
    "CNNx16_2_P8_MRAM":               0x509A_0000, # 509A0000
    "CNNx16_2_P9_MRAM":               0x509A_4000, # 509A4000
    "CNNx16_2_P10_MRAM":              0x509A_8000, # 509A8000
    "CNNx16_2_P11_MRAM":              0x509A_C000, # 509AC000
    "CNNx16_2_P12_MRAM":              0x509B_0000, # 509B0000
    "CNNx16_2_P13_MRAM":              0x509B_4000, # 509B4000
    "CNNx16_2_P14_MRAM":              0x509B_8000, # 509B8000
    "CNNx16_2_P15_MRAM":              0x509B_C000, # 509BC000
    "CNNx16_2_SRAM":                  0x50C0_0000, # 50C00000
    "CNNx16_3_BIAS":                  0x50D0_8000, # 50D08000
    "CNNx16_3_P0_TRAM":               0x50D1_0000, # 50D10000
    "CNNx16_3_P1_TRAM":               0x50D1_4000, # 50D14000
    "CNNx16_3_P2_TRAM":               0x50D1_8000, # 50D18000
    "CNNx16_3_P3_TRAM":               0x50D1_C000, # 50D1C000
    "CNNx16_3_P4_TRAM":               0x50D2_0000, # 50D20000
    "CNNx16_3_P5_TRAM":               0x50D2_4000, # 50D24000
    "CNNx16_3_P6_TRAM":               0x50D2_8000, # 50D28000
    "CNNx16_3_P7_TRAM":               0x50D2_C000, # 50D2C000
    "CNNx16_3_P8_TRAM":               0x50D3_0000, # 50D30000
    "CNNx16_3_P9_TRAM":               0x50D3_4000, # 50D34000
    "CNNx16_3_P10_TRAM":              0x50D3_8000, # 50D38000
    "CNNx16_3_P11_TRAM":              0x50D3_C000, # 50D3C000
    "CNNx16_3_P12_TRAM":              0x50D4_0000, # 50D40000
    "CNNx16_3_P13_TRAM":              0x50D4_4000, # 50D44000
    "CNNx16_3_P14_TRAM":              0x50D4_8000, # 50D48000
    "CNNx16_3_P15_TRAM":              0x50D4_C000, # 50D4C000
    "CNNx16_3_P0_MRAM":               0x50D8_0000, # 50D80000
    "CNNx16_3_P1_MRAM":               0x50D8_4000, # 50D84000
    "CNNx16_3_P2_MRAM":               0x50D8_8000, # 50D88000
    "CNNx16_3_P3_MRAM":               0x50D8_C000, # 50D8C000
    "CNNx16_3_P4_MRAM":               0x50D9_0000, # 50D90000
    "CNNx16_3_P5_MRAM":               0x50D9_4000, # 50D94000
    "CNNx16_3_P6_MRAM":               0x50D9_8000, # 50D98000
    "CNNx16_3_P7_MRAM":               0x50D9_C000, # 50D9C000
    "CNNx16_3_P8_MRAM":               0x50DA_0000, # 50DA0000
    "CNNx16_3_P9_MRAM":               0x50DA_4000, # 50DA4000
    "CNNx16_3_P10_MRAM":              0x50DA_8000, # 50DA8000
    "CNNx16_3_P11_MRAM":              0x50DA_C000, # 50DAC000
    "CNNx16_3_P12_MRAM":              0x50DB_0000, # 50DB0000
    "CNNx16_3_P13_MRAM":              0x50DB_4000, # 50DB4000
    "CNNx16_3_P14_MRAM":              0x50DB_8000, # 50DB8000
    "CNNx16_3_P15_MRAM":              0x50DB_C000, # 50DBC000
    "CNNx16_3_SRAM":                  0x5100_0000, # 51000000
} # memory


# REGISTER FIELDS
CNNx16_n_CTRL_APBCLK_EN_POS = 7
CNNx16_n_CTRL_APBCLK_EN = 0x00000080
CNNx16_n_CTRL_APBCLK_EN_MASK = 0x00000080
CNNx16_n_CTRL_APBCLK_EN_VALUEMASK = 0x00000001
CNNx16_n_CTRL_BIGDATA_POS = 6
CNNx16_n_CTRL_BIGDATA = 0x00000040
CNNx16_n_CTRL_BIGDATA_MASK = 0x00000040
CNNx16_n_CTRL_BIGDATA_VALUEMASK = 0x00000001
CNNx16_n_CTRL_CALCMAX_POS = 4
CNNx16_n_CTRL_CALCMAX = 0x00000010
CNNx16_n_CTRL_CALCMAX_MASK = 0x00000010
CNNx16_n_CTRL_CALCMAX_VALUEMASK = 0x00000001
CNNx16_n_CTRL_CLK_EN_POS = 3
CNNx16_n_CTRL_CLK_EN = 0x00000008
CNNx16_n_CTRL_CLK_EN_MASK = 0x00000008
CNNx16_n_CTRL_CLK_EN_VALUEMASK = 0x00000001
CNNx16_n_CTRL_CNN_EN_POS = 0
CNNx16_n_CTRL_CNN_EN = 0x00000001
CNNx16_n_CTRL_CNN_EN_MASK = 0x00000001
CNNx16_n_CTRL_CNN_EN_VALUEMASK = 0x00000001
CNNx16_n_CTRL_CNN_IRQ_POS = 12
CNNx16_n_CTRL_CNN_IRQ = 0x00001000
CNNx16_n_CTRL_CNN_IRQ_MASK = 0x00001000
CNNx16_n_CTRL_CNN_IRQ_VALUEMASK = 0x00000001
CNNx16_n_CTRL_EXT_SYNC_POS = 9
CNNx16_n_CTRL_EXT_SYNC_BIT0 = 0x00000200
CNNx16_n_CTRL_EXT_SYNC_BIT1 = 0x00000400
CNNx16_n_CTRL_EXT_SYNC_BIT2 = 0x00000800
CNNx16_n_CTRL_EXT_SYNC_MASK = 0x00000E00
CNNx16_n_CTRL_EXT_SYNC_VALUEMASK = 0x00000007
CNNx16_n_CTRL_FCLK_DLY_POS = 24
CNNx16_n_CTRL_FCLK_DLY_BIT0 = 0x01000000
CNNx16_n_CTRL_FCLK_DLY_BIT1 = 0x02000000
CNNx16_n_CTRL_FCLK_DLY_BIT2 = 0x04000000
CNNx16_n_CTRL_FCLK_DLY_BIT3 = 0x08000000
CNNx16_n_CTRL_FCLK_DLY_BIT4 = 0x10000000
CNNx16_n_CTRL_FCLK_DLY_BIT5 = 0x20000000
CNNx16_n_CTRL_FCLK_DLY_MASK = 0x3F000000
CNNx16_n_CTRL_FCLK_DLY_VALUEMASK = 0x0000003F
CNNx16_n_CTRL_FFIFO_EN_POS = 22
CNNx16_n_CTRL_FFIFO_EN = 0x00400000
CNNx16_n_CTRL_FFIFO_EN_MASK = 0x00400000
CNNx16_n_CTRL_FFIFO_EN_VALUEMASK = 0x00000001
CNNx16_n_CTRL_FIFOGRP_POS = 23
CNNx16_n_CTRL_FIFOGRP = 0x00800000
CNNx16_n_CTRL_FIFOGRP_MASK = 0x00800000
CNNx16_n_CTRL_FIFOGRP_VALUEMASK = 0x00000001
CNNx16_n_CTRL_FIFO_EN_POS = 15
CNNx16_n_CTRL_FIFO_EN = 0x00008000
CNNx16_n_CTRL_FIFO_EN_MASK = 0x00008000
CNNx16_n_CTRL_FIFO_EN_VALUEMASK = 0x00000001
CNNx16_n_CTRL_LILBUF_POS = 19
CNNx16_n_CTRL_LILBUF = 0x00080000
CNNx16_n_CTRL_LILBUF_MASK = 0x00080000
CNNx16_n_CTRL_LILBUF_VALUEMASK = 0x00000001
CNNx16_n_CTRL_MEXPRESS_POS = 20
CNNx16_n_CTRL_MEXPRESS = 0x00100000
CNNx16_n_CTRL_MEXPRESS_MASK = 0x00100000
CNNx16_n_CTRL_MEXPRESS_VALUEMASK = 0x00000001
CNNx16_n_CTRL_MLATCH_SEL_POS = 17
CNNx16_n_CTRL_MLATCH_SEL_BIT0 = 0x00020000
CNNx16_n_CTRL_MLATCH_SEL_BIT1 = 0x00040000
CNNx16_n_CTRL_MLATCH_SEL_MASK = 0x00060000
CNNx16_n_CTRL_MLATCH_SEL_VALUEMASK = 0x00000003
CNNx16_n_CTRL_MLAT_LD_POS = 16
CNNx16_n_CTRL_MLAT_LD = 0x00010000
CNNx16_n_CTRL_MLAT_LD_MASK = 0x00010000
CNNx16_n_CTRL_MLAT_LD_VALUEMASK = 0x00000001
CNNx16_n_CTRL_ONESHOT_POS = 8
CNNx16_n_CTRL_ONESHOT = 0x00000100
CNNx16_n_CTRL_ONESHOT_MASK = 0x00000100
CNNx16_n_CTRL_ONESHOT_VALUEMASK = 0x00000001
CNNx16_n_CTRL_POOLRND_POS = 13
CNNx16_n_CTRL_POOLRND = 0x00002000
CNNx16_n_CTRL_POOLRND_MASK = 0x00002000
CNNx16_n_CTRL_POOLRND_VALUEMASK = 0x00000001
CNNx16_n_CTRL_POOL_EN_POS = 5
CNNx16_n_CTRL_POOL_EN = 0x00000020
CNNx16_n_CTRL_POOL_EN_MASK = 0x00000020
CNNx16_n_CTRL_POOL_EN_VALUEMASK = 0x00000001
CNNx16_n_CTRL_QUPAC_POS = 31
CNNx16_n_CTRL_QUPAC = 0x80000000
CNNx16_n_CTRL_QUPAC_MASK = 0x80000000
CNNx16_n_CTRL_QUPAC_VALUEMASK = 0x00000001
CNNx16_n_CTRL_RDY_SEL_POS = 1
CNNx16_n_CTRL_RDY_SEL_BIT0 = 0x00000002
CNNx16_n_CTRL_RDY_SEL_BIT1 = 0x00000004
CNNx16_n_CTRL_RDY_SEL_MASK = 0x00000006
CNNx16_n_CTRL_RDY_SEL_VALUEMASK = 0x00000003
CNNx16_n_CTRL_SIMPLE1B_POS = 21
CNNx16_n_CTRL_SIMPLE1B = 0x00200000
CNNx16_n_CTRL_SIMPLE1B_MASK = 0x00200000
CNNx16_n_CTRL_SIMPLE1B_VALUEMASK = 0x00000001
CNNx16_n_CTRL_STREAM_EN_POS = 14
CNNx16_n_CTRL_STREAM_EN = 0x00004000
CNNx16_n_CTRL_STREAM_EN_MASK = 0x00004000
CNNx16_n_CTRL_STREAM_EN_VALUEMASK = 0x00000001
CNNx16_n_CTRL_TIMESHFT_POS = 30
CNNx16_n_CTRL_TIMESHFT = 0x40000000
CNNx16_n_CTRL_TIMESHFT_MASK = 0x40000000
CNNx16_n_CTRL_TIMESHFT_VALUEMASK = 0x00000001
CNNx16_n_IFRM_IFRM_REG_POS = 0
CNNx16_n_IFRM_IFRM_REG_BIT0 = 0x00000001
CNNx16_n_IFRM_IFRM_REG_BIT1 = 0x00000002
CNNx16_n_IFRM_IFRM_REG_BIT2 = 0x00000004
CNNx16_n_IFRM_IFRM_REG_BIT3 = 0x00000008
CNNx16_n_IFRM_IFRM_REG_BIT4 = 0x00000010
CNNx16_n_IFRM_IFRM_REG_BIT5 = 0x00000020
CNNx16_n_IFRM_IFRM_REG_BIT6 = 0x00000040
CNNx16_n_IFRM_IFRM_REG_BIT7 = 0x00000080
CNNx16_n_IFRM_IFRM_REG_BIT8 = 0x00000100
CNNx16_n_IFRM_IFRM_REG_BIT9 = 0x00000200
CNNx16_n_IFRM_IFRM_REG_BIT10 = 0x00000400
CNNx16_n_IFRM_IFRM_REG_BIT11 = 0x00000800
CNNx16_n_IFRM_IFRM_REG_BIT12 = 0x00001000
CNNx16_n_IFRM_IFRM_REG_BIT13 = 0x00002000
CNNx16_n_IFRM_IFRM_REG_BIT14 = 0x00004000
CNNx16_n_IFRM_IFRM_REG_BIT15 = 0x00008000
CNNx16_n_IFRM_IFRM_REG_BIT16 = 0x00010000
CNNx16_n_IFRM_IFRM_REG_MASK = 0x0001FFFF
CNNx16_n_IFRM_IFRM_REG_VALUEMASK = 0x0001FFFF
CNNx16_n_LCNT_MAX_LCNT_POS = 0
CNNx16_n_LCNT_MAX_LCNT_BIT0 = 0x00000001
CNNx16_n_LCNT_MAX_LCNT_BIT1 = 0x00000002
CNNx16_n_LCNT_MAX_LCNT_BIT2 = 0x00000004
CNNx16_n_LCNT_MAX_LCNT_BIT3 = 0x00000008
CNNx16_n_LCNT_MAX_LCNT_BIT4 = 0x00000010
CNNx16_n_LCNT_MAX_LCNT_MASK = 0x0000001F
CNNx16_n_LCNT_MAX_LCNT_VALUEMASK = 0x0000001F
CNNx16_n_Ly_CCNT_CCNT_MAX_POS = 0
CNNx16_n_Ly_CCNT_CCNT_MAX_BIT0 = 0x00000001
CNNx16_n_Ly_CCNT_CCNT_MAX_BIT1 = 0x00000002
CNNx16_n_Ly_CCNT_CCNT_MAX_BIT2 = 0x00000004
CNNx16_n_Ly_CCNT_CCNT_MAX_BIT3 = 0x00000008
CNNx16_n_Ly_CCNT_CCNT_MAX_BIT4 = 0x00000010
CNNx16_n_Ly_CCNT_CCNT_MAX_BIT5 = 0x00000020
CNNx16_n_Ly_CCNT_CCNT_MAX_BIT6 = 0x00000040
CNNx16_n_Ly_CCNT_CCNT_MAX_BIT7 = 0x00000080
CNNx16_n_Ly_CCNT_CCNT_MAX_BIT8 = 0x00000100
CNNx16_n_Ly_CCNT_CCNT_MAX_BIT9 = 0x00000200
CNNx16_n_Ly_CCNT_CCNT_MAX_MASK = 0x000003FF
CNNx16_n_Ly_CCNT_CCNT_MAX_VALUEMASK = 0x000003FF
CNNx16_n_Ly_CCNT_CCNT_PAD_POS = 16
CNNx16_n_Ly_CCNT_CCNT_PAD_BIT0 = 0x00010000
CNNx16_n_Ly_CCNT_CCNT_PAD_BIT1 = 0x00020000
CNNx16_n_Ly_CCNT_CCNT_PAD_MASK = 0x00030000
CNNx16_n_Ly_CCNT_CCNT_PAD_VALUEMASK = 0x00000003
CNNx16_n_Ly_EN_MASK_EN_POS = 16
CNNx16_n_Ly_EN_MASK_EN_BIT0 = 0x00010000
CNNx16_n_Ly_EN_MASK_EN_BIT1 = 0x00020000
CNNx16_n_Ly_EN_MASK_EN_BIT2 = 0x00040000
CNNx16_n_Ly_EN_MASK_EN_BIT3 = 0x00080000
CNNx16_n_Ly_EN_MASK_EN_BIT4 = 0x00100000
CNNx16_n_Ly_EN_MASK_EN_BIT5 = 0x00200000
CNNx16_n_Ly_EN_MASK_EN_BIT6 = 0x00400000
CNNx16_n_Ly_EN_MASK_EN_BIT7 = 0x00800000
CNNx16_n_Ly_EN_MASK_EN_BIT8 = 0x01000000
CNNx16_n_Ly_EN_MASK_EN_BIT9 = 0x02000000
CNNx16_n_Ly_EN_MASK_EN_BIT10 = 0x04000000
CNNx16_n_Ly_EN_MASK_EN_BIT11 = 0x08000000
CNNx16_n_Ly_EN_MASK_EN_BIT12 = 0x10000000
CNNx16_n_Ly_EN_MASK_EN_BIT13 = 0x20000000
CNNx16_n_Ly_EN_MASK_EN_BIT14 = 0x40000000
CNNx16_n_Ly_EN_MASK_EN_BIT15 = 0x80000000
CNNx16_n_Ly_EN_MASK_EN_MASK = 0xFFFF0000
CNNx16_n_Ly_EN_MASK_EN_VALUEMASK = 0x0000FFFF
CNNx16_n_Ly_EN_PRO_EN_POS = 0
CNNx16_n_Ly_EN_PRO_EN_BIT0 = 0x00000001
CNNx16_n_Ly_EN_PRO_EN_BIT1 = 0x00000002
CNNx16_n_Ly_EN_PRO_EN_BIT2 = 0x00000004
CNNx16_n_Ly_EN_PRO_EN_BIT3 = 0x00000008
CNNx16_n_Ly_EN_PRO_EN_BIT4 = 0x00000010
CNNx16_n_Ly_EN_PRO_EN_BIT5 = 0x00000020
CNNx16_n_Ly_EN_PRO_EN_BIT6 = 0x00000040
CNNx16_n_Ly_EN_PRO_EN_BIT7 = 0x00000080
CNNx16_n_Ly_EN_PRO_EN_BIT8 = 0x00000100
CNNx16_n_Ly_EN_PRO_EN_BIT9 = 0x00000200
CNNx16_n_Ly_EN_PRO_EN_BIT10 = 0x00000400
CNNx16_n_Ly_EN_PRO_EN_BIT11 = 0x00000800
CNNx16_n_Ly_EN_PRO_EN_BIT12 = 0x00001000
CNNx16_n_Ly_EN_PRO_EN_BIT13 = 0x00002000
CNNx16_n_Ly_EN_PRO_EN_BIT14 = 0x00004000
CNNx16_n_Ly_EN_PRO_EN_BIT15 = 0x00008000
CNNx16_n_Ly_EN_PRO_EN_MASK = 0x0000FFFF
CNNx16_n_Ly_EN_PRO_EN_VALUEMASK = 0x0000FFFF
CNNx16_n_Ly_LCTRL0_ACT_EN_POS = 9
CNNx16_n_Ly_LCTRL0_ACT_EN = 0x00000200
CNNx16_n_Ly_LCTRL0_ACT_EN_MASK = 0x00000200
CNNx16_n_Ly_LCTRL0_ACT_EN_VALUEMASK = 0x00000001
CNNx16_n_Ly_LCTRL0_BIGDWRT_POS = 16
CNNx16_n_Ly_LCTRL0_BIGDWRT = 0x00010000
CNNx16_n_Ly_LCTRL0_BIGDWRT_MASK = 0x00010000
CNNx16_n_Ly_LCTRL0_BIGDWRT_VALUEMASK = 0x00000001
CNNx16_n_Ly_LCTRL0_CNNSI_EN_POS = 12
CNNx16_n_Ly_LCTRL0_CNNSI_EN_BIT0 = 0x00001000
CNNx16_n_Ly_LCTRL0_CNNSI_EN_BIT1 = 0x00002000
CNNx16_n_Ly_LCTRL0_CNNSI_EN_BIT2 = 0x00004000
CNNx16_n_Ly_LCTRL0_CNNSI_EN_BIT3 = 0x00008000
CNNx16_n_Ly_LCTRL0_CNNSI_EN_MASK = 0x0000F000
CNNx16_n_Ly_LCTRL0_CNNSI_EN_VALUEMASK = 0x0000000F
CNNx16_n_Ly_LCTRL0_CPAD_ONLY_POS = 10
CNNx16_n_Ly_LCTRL0_CPAD_ONLY = 0x00000400
CNNx16_n_Ly_LCTRL0_CPAD_ONLY_MASK = 0x00000400
CNNx16_n_Ly_LCTRL0_CPAD_ONLY_VALUEMASK = 0x00000001
CNNx16_n_Ly_LCTRL0_MASTER_POS = 5
CNNx16_n_Ly_LCTRL0_MASTER = 0x00000020
CNNx16_n_Ly_LCTRL0_MASTER_MASK = 0x00000020
CNNx16_n_Ly_LCTRL0_MASTER_VALUEMASK = 0x00000001
CNNx16_n_Ly_LCTRL0_MAXPL_EN_POS = 8
CNNx16_n_Ly_LCTRL0_MAXPL_EN = 0x00000100
CNNx16_n_Ly_LCTRL0_MAXPL_EN_MASK = 0x00000100
CNNx16_n_Ly_LCTRL0_MAXPL_EN_VALUEMASK = 0x00000001
CNNx16_n_Ly_LCTRL0_MSLAVE_POS = 4
CNNx16_n_Ly_LCTRL0_MSLAVE = 0x00000010
CNNx16_n_Ly_LCTRL0_MSLAVE_MASK = 0x00000010
CNNx16_n_Ly_LCTRL0_MSLAVE_VALUEMASK = 0x00000001
CNNx16_n_Ly_LCTRL0_PARALLEL_POS = 6
CNNx16_n_Ly_LCTRL0_PARALLEL = 0x00000040
CNNx16_n_Ly_LCTRL0_PARALLEL_MASK = 0x00000040
CNNx16_n_Ly_LCTRL0_PARALLEL_VALUEMASK = 0x00000001
CNNx16_n_Ly_LCTRL0_POOL_EN_POS = 7
CNNx16_n_Ly_LCTRL0_POOL_EN = 0x00000080
CNNx16_n_Ly_LCTRL0_POOL_EN_MASK = 0x00000080
CNNx16_n_Ly_LCTRL0_POOL_EN_VALUEMASK = 0x00000001
CNNx16_n_Ly_LCTRL0_SRAMLSRC_POS = 11
CNNx16_n_Ly_LCTRL0_SRAMLSRC = 0x00000800
CNNx16_n_Ly_LCTRL0_SRAMLSRC_MASK = 0x00000800
CNNx16_n_Ly_LCTRL0_SRAMLSRC_VALUEMASK = 0x00000001
CNNx16_n_Ly_LCTRL0_SSLAVE_POS = 0
CNNx16_n_Ly_LCTRL0_SSLAVE_BIT0 = 0x00000001
CNNx16_n_Ly_LCTRL0_SSLAVE_BIT1 = 0x00000002
CNNx16_n_Ly_LCTRL0_SSLAVE_BIT2 = 0x00000004
CNNx16_n_Ly_LCTRL0_SSLAVE_BIT3 = 0x00000008
CNNx16_n_Ly_LCTRL0_SSLAVE_MASK = 0x0000000F
CNNx16_n_Ly_LCTRL0_SSLAVE_VALUEMASK = 0x0000000F
CNNx16_n_Ly_LCTRL1_INPCHEXP_POS = 0
CNNx16_n_Ly_LCTRL1_INPCHEXP_BIT0 = 0x00000001
CNNx16_n_Ly_LCTRL1_INPCHEXP_BIT1 = 0x00000002
CNNx16_n_Ly_LCTRL1_INPCHEXP_BIT2 = 0x00000004
CNNx16_n_Ly_LCTRL1_INPCHEXP_BIT3 = 0x00000008
CNNx16_n_Ly_LCTRL1_INPCHEXP_MASK = 0x0000000F
CNNx16_n_Ly_LCTRL1_INPCHEXP_VALUEMASK = 0x0000000F
CNNx16_n_Ly_LCTRL1_WPTR_INC_POS = 4
CNNx16_n_Ly_LCTRL1_WPTR_INC_BIT0 = 0x00000010
CNNx16_n_Ly_LCTRL1_WPTR_INC_BIT1 = 0x00000020
CNNx16_n_Ly_LCTRL1_WPTR_INC_BIT2 = 0x00000040
CNNx16_n_Ly_LCTRL1_WPTR_INC_BIT3 = 0x00000080
CNNx16_n_Ly_LCTRL1_WPTR_INC_MASK = 0x000000F0
CNNx16_n_Ly_LCTRL1_WPTR_INC_VALUEMASK = 0x0000000F
CNNx16_n_Ly_LCTRL1_XPCH_MAX_POS = 8
CNNx16_n_Ly_LCTRL1_XPCH_MAX_BIT0 = 0x00000100
CNNx16_n_Ly_LCTRL1_XPCH_MAX_BIT1 = 0x00000200
CNNx16_n_Ly_LCTRL1_XPCH_MAX_BIT2 = 0x00000400
CNNx16_n_Ly_LCTRL1_XPCH_MAX_BIT3 = 0x00000800
CNNx16_n_Ly_LCTRL1_XPCH_MAX_BIT4 = 0x00001000
CNNx16_n_Ly_LCTRL1_XPCH_MAX_BIT5 = 0x00002000
CNNx16_n_Ly_LCTRL1_XPCH_MAX_BIT6 = 0x00004000
CNNx16_n_Ly_LCTRL1_XPCH_MAX_BIT7 = 0x00008000
CNNx16_n_Ly_LCTRL1_XPCH_MAX_BIT8 = 0x00010000
CNNx16_n_Ly_LCTRL1_XPCH_MAX_MASK = 0x0001FF00
CNNx16_n_Ly_LCTRL1_XPCH_MAX_VALUEMASK = 0x000001FF
CNNx16_n_Ly_MCNT_MCNT_MAX_POS = 0
CNNx16_n_Ly_MCNT_MCNT_MAX_BIT0 = 0x00000001
CNNx16_n_Ly_MCNT_MCNT_MAX_BIT1 = 0x00000002
CNNx16_n_Ly_MCNT_MCNT_MAX_BIT2 = 0x00000004
CNNx16_n_Ly_MCNT_MCNT_MAX_BIT3 = 0x00000008
CNNx16_n_Ly_MCNT_MCNT_MAX_BIT4 = 0x00000010
CNNx16_n_Ly_MCNT_MCNT_MAX_BIT5 = 0x00000020
CNNx16_n_Ly_MCNT_MCNT_MAX_BIT6 = 0x00000040
CNNx16_n_Ly_MCNT_MCNT_MAX_BIT7 = 0x00000080
CNNx16_n_Ly_MCNT_MCNT_MAX_BIT8 = 0x00000100
CNNx16_n_Ly_MCNT_MCNT_MAX_BIT9 = 0x00000200
CNNx16_n_Ly_MCNT_MCNT_MAX_BIT10 = 0x00000400
CNNx16_n_Ly_MCNT_MCNT_MAX_BIT11 = 0x00000800
CNNx16_n_Ly_MCNT_MCNT_MAX_BIT12 = 0x00001000
CNNx16_n_Ly_MCNT_MCNT_MAX_BIT13 = 0x00002000
CNNx16_n_Ly_MCNT_MCNT_MAX_BIT14 = 0x00004000
CNNx16_n_Ly_MCNT_MCNT_MAX_BIT15 = 0x00008000
CNNx16_n_Ly_MCNT_MCNT_MAX_MASK = 0x0000FFFF
CNNx16_n_Ly_MCNT_MCNT_MAX_VALUEMASK = 0x0000FFFF
CNNx16_n_Ly_MCNT_MCNT_SAD_POS = 16
CNNx16_n_Ly_MCNT_MCNT_SAD_BIT0 = 0x00010000
CNNx16_n_Ly_MCNT_MCNT_SAD_BIT1 = 0x00020000
CNNx16_n_Ly_MCNT_MCNT_SAD_BIT2 = 0x00040000
CNNx16_n_Ly_MCNT_MCNT_SAD_BIT3 = 0x00080000
CNNx16_n_Ly_MCNT_MCNT_SAD_BIT4 = 0x00100000
CNNx16_n_Ly_MCNT_MCNT_SAD_BIT5 = 0x00200000
CNNx16_n_Ly_MCNT_MCNT_SAD_BIT6 = 0x00400000
CNNx16_n_Ly_MCNT_MCNT_SAD_BIT7 = 0x00800000
CNNx16_n_Ly_MCNT_MCNT_SAD_BIT8 = 0x01000000
CNNx16_n_Ly_MCNT_MCNT_SAD_BIT9 = 0x02000000
CNNx16_n_Ly_MCNT_MCNT_SAD_BIT10 = 0x04000000
CNNx16_n_Ly_MCNT_MCNT_SAD_BIT11 = 0x08000000
CNNx16_n_Ly_MCNT_MCNT_SAD_BIT12 = 0x10000000
CNNx16_n_Ly_MCNT_MCNT_SAD_BIT13 = 0x20000000
CNNx16_n_Ly_MCNT_MCNT_SAD_BIT14 = 0x40000000
CNNx16_n_Ly_MCNT_MCNT_SAD_BIT15 = 0x80000000
CNNx16_n_Ly_MCNT_MCNT_SAD_MASK = 0xFFFF0000
CNNx16_n_Ly_MCNT_MCNT_SAD_VALUEMASK = 0x0000FFFF
CNNx16_n_Ly_ONED_2D_CONV_POS = 17
CNNx16_n_Ly_ONED_2D_CONV = 0x00020000
CNNx16_n_Ly_ONED_2D_CONV_MASK = 0x00020000
CNNx16_n_Ly_ONED_2D_CONV_VALUEMASK = 0x00000001
CNNx16_n_Ly_ONED_EWISE_CNT_POS = 18
CNNx16_n_Ly_ONED_EWISE_CNT_BIT0 = 0x00040000
CNNx16_n_Ly_ONED_EWISE_CNT_BIT1 = 0x00080000
CNNx16_n_Ly_ONED_EWISE_CNT_BIT2 = 0x00100000
CNNx16_n_Ly_ONED_EWISE_CNT_BIT3 = 0x00200000
CNNx16_n_Ly_ONED_EWISE_CNT_MASK = 0x003C0000
CNNx16_n_Ly_ONED_EWISE_CNT_VALUEMASK = 0x0000000F
CNNx16_n_Ly_ONED_EWISE_EN_POS = 13
CNNx16_n_Ly_ONED_EWISE_EN = 0x00002000
CNNx16_n_Ly_ONED_EWISE_EN_MASK = 0x00002000
CNNx16_n_Ly_ONED_EWISE_EN_VALUEMASK = 0x00000001
CNNx16_n_Ly_ONED_EWISE_FUNC_POS = 14
CNNx16_n_Ly_ONED_EWISE_FUNC_BIT0 = 0x00004000
CNNx16_n_Ly_ONED_EWISE_FUNC_BIT1 = 0x00008000
CNNx16_n_Ly_ONED_EWISE_FUNC_MASK = 0x0000C000
CNNx16_n_Ly_ONED_EWISE_FUNC_VALUEMASK = 0x00000003
CNNx16_n_Ly_ONED_ONED_EN_POS = 12
CNNx16_n_Ly_ONED_ONED_EN = 0x00001000
CNNx16_n_Ly_ONED_ONED_EN_MASK = 0x00001000
CNNx16_n_Ly_ONED_ONED_EN_VALUEMASK = 0x00000001
CNNx16_n_Ly_ONED_ONED_SAD_POS = 4
CNNx16_n_Ly_ONED_ONED_SAD_BIT0 = 0x00000010
CNNx16_n_Ly_ONED_ONED_SAD_BIT1 = 0x00000020
CNNx16_n_Ly_ONED_ONED_SAD_BIT2 = 0x00000040
CNNx16_n_Ly_ONED_ONED_SAD_BIT3 = 0x00000080
CNNx16_n_Ly_ONED_ONED_SAD_MASK = 0x000000F0
CNNx16_n_Ly_ONED_ONED_SAD_VALUEMASK = 0x0000000F
CNNx16_n_Ly_ONED_ONED_WIDTH_POS = 8
CNNx16_n_Ly_ONED_ONED_WIDTH_BIT0 = 0x00000100
CNNx16_n_Ly_ONED_ONED_WIDTH_BIT1 = 0x00000200
CNNx16_n_Ly_ONED_ONED_WIDTH_BIT2 = 0x00000400
CNNx16_n_Ly_ONED_ONED_WIDTH_BIT3 = 0x00000800
CNNx16_n_Ly_ONED_ONED_WIDTH_MASK = 0x00000F00
CNNx16_n_Ly_ONED_ONED_WIDTH_VALUEMASK = 0x0000000F
CNNx16_n_Ly_ONED_PREPOOL_POS = 16
CNNx16_n_Ly_ONED_PREPOOL = 0x00010000
CNNx16_n_Ly_ONED_PREPOOL_MASK = 0x00010000
CNNx16_n_Ly_ONED_PREPOOL_VALUEMASK = 0x00000001
CNNx16_n_Ly_ONED_TSCNT_MAX_POS = 0
CNNx16_n_Ly_ONED_TSCNT_MAX_BIT0 = 0x00000001
CNNx16_n_Ly_ONED_TSCNT_MAX_BIT1 = 0x00000002
CNNx16_n_Ly_ONED_TSCNT_MAX_BIT2 = 0x00000004
CNNx16_n_Ly_ONED_TSCNT_MAX_BIT3 = 0x00000008
CNNx16_n_Ly_ONED_TSCNT_MAX_MASK = 0x0000000F
CNNx16_n_Ly_ONED_TSCNT_MAX_VALUEMASK = 0x0000000F
CNNx16_n_Ly_PCCNT_PCCNT_MAX_POS = 0
CNNx16_n_Ly_PCCNT_PCCNT_MAX_BIT0 = 0x00000001
CNNx16_n_Ly_PCCNT_PCCNT_MAX_BIT1 = 0x00000002
CNNx16_n_Ly_PCCNT_PCCNT_MAX_BIT2 = 0x00000004
CNNx16_n_Ly_PCCNT_PCCNT_MAX_BIT3 = 0x00000008
CNNx16_n_Ly_PCCNT_PCCNT_MAX_MASK = 0x0000000F
CNNx16_n_Ly_PCCNT_PCCNT_MAX_VALUEMASK = 0x0000000F
CNNx16_n_Ly_POST_BPTR_EN_POS = 12
CNNx16_n_Ly_POST_BPTR_EN = 0x00001000
CNNx16_n_Ly_POST_BPTR_EN_MASK = 0x00001000
CNNx16_n_Ly_POST_BPTR_EN_VALUEMASK = 0x00000001
CNNx16_n_Ly_POST_BPTR_SAD_POS = 0
CNNx16_n_Ly_POST_BPTR_SAD_BIT0 = 0x00000001
CNNx16_n_Ly_POST_BPTR_SAD_BIT1 = 0x00000002
CNNx16_n_Ly_POST_BPTR_SAD_BIT2 = 0x00000004
CNNx16_n_Ly_POST_BPTR_SAD_BIT3 = 0x00000008
CNNx16_n_Ly_POST_BPTR_SAD_BIT4 = 0x00000010
CNNx16_n_Ly_POST_BPTR_SAD_BIT5 = 0x00000020
CNNx16_n_Ly_POST_BPTR_SAD_BIT6 = 0x00000040
CNNx16_n_Ly_POST_BPTR_SAD_BIT7 = 0x00000080
CNNx16_n_Ly_POST_BPTR_SAD_BIT8 = 0x00000100
CNNx16_n_Ly_POST_BPTR_SAD_BIT9 = 0x00000200
CNNx16_n_Ly_POST_BPTR_SAD_BIT10 = 0x00000400
CNNx16_n_Ly_POST_BPTR_SAD_BIT11 = 0x00000800
CNNx16_n_Ly_POST_BPTR_SAD_MASK = 0x00000FFF
CNNx16_n_Ly_POST_BPTR_SAD_VALUEMASK = 0x00000FFF
CNNx16_n_Ly_POST_DECONV_EN_POS = 28
CNNx16_n_Ly_POST_DECONV_EN = 0x10000000
CNNx16_n_Ly_POST_DECONV_EN_MASK = 0x10000000
CNNx16_n_Ly_POST_DECONV_EN_VALUEMASK = 0x00000001
CNNx16_n_Ly_POST_FLATTEN_EN_POS = 27
CNNx16_n_Ly_POST_FLATTEN_EN = 0x08000000
CNNx16_n_Ly_POST_FLATTEN_EN_MASK = 0x08000000
CNNx16_n_Ly_POST_FLATTEN_EN_VALUEMASK = 0x00000001
CNNx16_n_Ly_POST_MASK_SIZE_POS = 22
CNNx16_n_Ly_POST_MASK_SIZE_BIT0 = 0x00400000
CNNx16_n_Ly_POST_MASK_SIZE_BIT1 = 0x00800000
CNNx16_n_Ly_POST_MASK_SIZE_MASK = 0x00C00000
CNNx16_n_Ly_POST_MASK_SIZE_VALUEMASK = 0x00000003
CNNx16_n_Ly_POST_ONEXONE_EN_POS = 25
CNNx16_n_Ly_POST_ONEXONE_EN = 0x02000000
CNNx16_n_Ly_POST_ONEXONE_EN_MASK = 0x02000000
CNNx16_n_Ly_POST_ONEXONE_EN_VALUEMASK = 0x00000001
CNNx16_n_Ly_POST_OUT_ABS_POS = 26
CNNx16_n_Ly_POST_OUT_ABS = 0x04000000
CNNx16_n_Ly_POST_OUT_ABS_MASK = 0x04000000
CNNx16_n_Ly_POST_OUT_ABS_VALUEMASK = 0x00000001
CNNx16_n_Ly_POST_SCALE_REG_POS = 13
CNNx16_n_Ly_POST_SCALE_REG_BIT0 = 0x00002000
CNNx16_n_Ly_POST_SCALE_REG_BIT1 = 0x00004000
CNNx16_n_Ly_POST_SCALE_REG_BIT2 = 0x00008000
CNNx16_n_Ly_POST_SCALE_REG_BIT3 = 0x00010000
CNNx16_n_Ly_POST_SCALE_REG_MASK = 0x0001E000
CNNx16_n_Ly_POST_SCALE_REG_VALUEMASK = 0x0000000F
CNNx16_n_Ly_POST_SCALE_SHFT_POS = 17
CNNx16_n_Ly_POST_SCALE_SHFT = 0x00020000
CNNx16_n_Ly_POST_SCALE_SHFT_MASK = 0x00020000
CNNx16_n_Ly_POST_SCALE_SHFT_VALUEMASK = 0x00000001
CNNx16_n_Ly_POST_TS_EN_POS = 24
CNNx16_n_Ly_POST_TS_EN = 0x01000000
CNNx16_n_Ly_POST_TS_EN_MASK = 0x01000000
CNNx16_n_Ly_POST_TS_EN_VALUEMASK = 0x00000001
CNNx16_n_Ly_POST_XPMP_CNT_POS = 18
CNNx16_n_Ly_POST_XPMP_CNT_BIT0 = 0x00040000
CNNx16_n_Ly_POST_XPMP_CNT_BIT1 = 0x00080000
CNNx16_n_Ly_POST_XPMP_CNT_BIT2 = 0x00100000
CNNx16_n_Ly_POST_XPMP_CNT_BIT3 = 0x00200000
CNNx16_n_Ly_POST_XPMP_CNT_MASK = 0x003C0000
CNNx16_n_Ly_POST_XPMP_CNT_VALUEMASK = 0x0000000F
CNNx16_n_Ly_PRCNT_PRCNT_MAX_POS = 0
CNNx16_n_Ly_PRCNT_PRCNT_MAX_BIT0 = 0x00000001
CNNx16_n_Ly_PRCNT_PRCNT_MAX_BIT1 = 0x00000002
CNNx16_n_Ly_PRCNT_PRCNT_MAX_BIT2 = 0x00000004
CNNx16_n_Ly_PRCNT_PRCNT_MAX_BIT3 = 0x00000008
CNNx16_n_Ly_PRCNT_PRCNT_MAX_MASK = 0x0000000F
CNNx16_n_Ly_PRCNT_PRCNT_MAX_VALUEMASK = 0x0000000F
CNNx16_n_Ly_RCNT_RCNT_MAX_POS = 0
CNNx16_n_Ly_RCNT_RCNT_MAX_BIT0 = 0x00000001
CNNx16_n_Ly_RCNT_RCNT_MAX_BIT1 = 0x00000002
CNNx16_n_Ly_RCNT_RCNT_MAX_BIT2 = 0x00000004
CNNx16_n_Ly_RCNT_RCNT_MAX_BIT3 = 0x00000008
CNNx16_n_Ly_RCNT_RCNT_MAX_BIT4 = 0x00000010
CNNx16_n_Ly_RCNT_RCNT_MAX_BIT5 = 0x00000020
CNNx16_n_Ly_RCNT_RCNT_MAX_BIT6 = 0x00000040
CNNx16_n_Ly_RCNT_RCNT_MAX_BIT7 = 0x00000080
CNNx16_n_Ly_RCNT_RCNT_MAX_BIT8 = 0x00000100
CNNx16_n_Ly_RCNT_RCNT_MAX_BIT9 = 0x00000200
CNNx16_n_Ly_RCNT_RCNT_MAX_MASK = 0x000003FF
CNNx16_n_Ly_RCNT_RCNT_MAX_VALUEMASK = 0x000003FF
CNNx16_n_Ly_RCNT_RCNT_PAD_POS = 16
CNNx16_n_Ly_RCNT_RCNT_PAD_BIT0 = 0x00010000
CNNx16_n_Ly_RCNT_RCNT_PAD_BIT1 = 0x00020000
CNNx16_n_Ly_RCNT_RCNT_PAD_MASK = 0x00030000
CNNx16_n_Ly_RCNT_RCNT_PAD_VALUEMASK = 0x00000003
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_POS = 0
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_BIT0 = 0x00000001
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_BIT1 = 0x00000002
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_BIT2 = 0x00000004
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_BIT3 = 0x00000008
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_BIT4 = 0x00000010
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_BIT5 = 0x00000020
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_BIT6 = 0x00000040
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_BIT7 = 0x00000080
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_BIT8 = 0x00000100
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_BIT9 = 0x00000200
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_BIT10 = 0x00000400
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_BIT11 = 0x00000800
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_BIT12 = 0x00001000
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_BIT13 = 0x00002000
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_BIT14 = 0x00004000
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_BIT15 = 0x00008000
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_BIT16 = 0x00010000
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_MASK = 0x0001FFFF
CNNx16_n_Ly_RPTR_BASE_RPTR_BASE_VALUEMASK = 0x0001FFFF
CNNx16_n_Ly_STRIDE_STRIDE_POS = 0
CNNx16_n_Ly_STRIDE_STRIDE_BIT0 = 0x00000001
CNNx16_n_Ly_STRIDE_STRIDE_BIT1 = 0x00000002
CNNx16_n_Ly_STRIDE_STRIDE_MASK = 0x00000003
CNNx16_n_Ly_STRIDE_STRIDE_VALUEMASK = 0x00000003
CNNx16_n_Ly_TPTR_TPTR_MAX_POS = 0
CNNx16_n_Ly_TPTR_TPTR_MAX_BIT0 = 0x00000001
CNNx16_n_Ly_TPTR_TPTR_MAX_BIT1 = 0x00000002
CNNx16_n_Ly_TPTR_TPTR_MAX_BIT2 = 0x00000004
CNNx16_n_Ly_TPTR_TPTR_MAX_BIT3 = 0x00000008
CNNx16_n_Ly_TPTR_TPTR_MAX_BIT4 = 0x00000010
CNNx16_n_Ly_TPTR_TPTR_MAX_BIT5 = 0x00000020
CNNx16_n_Ly_TPTR_TPTR_MAX_BIT6 = 0x00000040
CNNx16_n_Ly_TPTR_TPTR_MAX_BIT7 = 0x00000080
CNNx16_n_Ly_TPTR_TPTR_MAX_BIT8 = 0x00000100
CNNx16_n_Ly_TPTR_TPTR_MAX_BIT9 = 0x00000200
CNNx16_n_Ly_TPTR_TPTR_MAX_BIT10 = 0x00000400
CNNx16_n_Ly_TPTR_TPTR_MAX_MASK = 0x000007FF
CNNx16_n_Ly_TPTR_TPTR_MAX_VALUEMASK = 0x000007FF
CNNx16_n_Ly_TPTR_TPTR_SAD_POS = 16
CNNx16_n_Ly_TPTR_TPTR_SAD_BIT0 = 0x00010000
CNNx16_n_Ly_TPTR_TPTR_SAD_BIT1 = 0x00020000
CNNx16_n_Ly_TPTR_TPTR_SAD_BIT2 = 0x00040000
CNNx16_n_Ly_TPTR_TPTR_SAD_BIT3 = 0x00080000
CNNx16_n_Ly_TPTR_TPTR_SAD_BIT4 = 0x00100000
CNNx16_n_Ly_TPTR_TPTR_SAD_BIT5 = 0x00200000
CNNx16_n_Ly_TPTR_TPTR_SAD_BIT6 = 0x00400000
CNNx16_n_Ly_TPTR_TPTR_SAD_BIT7 = 0x00800000
CNNx16_n_Ly_TPTR_TPTR_SAD_BIT8 = 0x01000000
CNNx16_n_Ly_TPTR_TPTR_SAD_BIT9 = 0x02000000
CNNx16_n_Ly_TPTR_TPTR_SAD_BIT10 = 0x04000000
CNNx16_n_Ly_TPTR_TPTR_SAD_MASK = 0x07FF0000
CNNx16_n_Ly_TPTR_TPTR_SAD_VALUEMASK = 0x000007FF
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_POS = 0
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_BIT0 = 0x00000001
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_BIT1 = 0x00000002
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_BIT2 = 0x00000004
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_BIT3 = 0x00000008
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_BIT4 = 0x00000010
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_BIT5 = 0x00000020
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_BIT6 = 0x00000040
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_BIT7 = 0x00000080
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_BIT8 = 0x00000100
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_BIT9 = 0x00000200
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_BIT10 = 0x00000400
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_BIT11 = 0x00000800
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_BIT12 = 0x00001000
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_BIT13 = 0x00002000
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_BIT14 = 0x00004000
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_BIT15 = 0x00008000
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_BIT16 = 0x00010000
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_MASK = 0x0001FFFF
CNNx16_n_Ly_WPTR_BASE_WPTR_BASE_VALUEMASK = 0x0001FFFF
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_POS = 0
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_BIT0 = 0x00000001
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_BIT1 = 0x00000002
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_BIT2 = 0x00000004
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_BIT3 = 0x00000008
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_BIT4 = 0x00000010
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_BIT5 = 0x00000020
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_BIT6 = 0x00000040
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_BIT7 = 0x00000080
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_BIT8 = 0x00000100
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_BIT9 = 0x00000200
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_BIT10 = 0x00000400
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_BIT11 = 0x00000800
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_BIT12 = 0x00001000
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_BIT13 = 0x00002000
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_BIT14 = 0x00004000
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_BIT15 = 0x00008000
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_BIT16 = 0x00010000
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_MASK = 0x0001FFFF
CNNx16_n_Ly_WPTR_CHOFF_WPTR_CHOFF_VALUEMASK = 0x0001FFFF
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_POS = 0
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_BIT0 = 0x00000001
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_BIT1 = 0x00000002
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_BIT2 = 0x00000004
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_BIT3 = 0x00000008
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_BIT4 = 0x00000010
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_BIT5 = 0x00000020
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_BIT6 = 0x00000040
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_BIT7 = 0x00000080
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_BIT8 = 0x00000100
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_BIT9 = 0x00000200
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_BIT10 = 0x00000400
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_BIT11 = 0x00000800
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_BIT12 = 0x00001000
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_BIT13 = 0x00002000
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_BIT14 = 0x00004000
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_BIT15 = 0x00008000
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_BIT16 = 0x00010000
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_MASK = 0x0001FFFF
CNNx16_n_Ly_WPTR_MOFF_WPTR_MOFF_VALUEMASK = 0x0001FFFF
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_POS = 0
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_BIT0 = 0x00000001
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_BIT1 = 0x00000002
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_BIT2 = 0x00000004
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_BIT3 = 0x00000008
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_BIT4 = 0x00000010
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_BIT5 = 0x00000020
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_BIT6 = 0x00000040
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_BIT7 = 0x00000080
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_BIT8 = 0x00000100
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_BIT9 = 0x00000200
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_BIT10 = 0x00000400
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_BIT11 = 0x00000800
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_BIT12 = 0x00001000
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_BIT13 = 0x00002000
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_BIT14 = 0x00004000
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_BIT15 = 0x00008000
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_BIT16 = 0x00010000
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_MASK = 0x0001FFFF
CNNx16_n_Ly_WPTR_TOFF_WPTR_TOFF_VALUEMASK = 0x0001FFFF
CNNx16_n_MLAT_MLATDAT_POS = 0
CNNx16_n_MLAT_MLATDAT_BIT0 = 0x00000001
CNNx16_n_MLAT_MLATDAT_BIT1 = 0x00000002
CNNx16_n_MLAT_MLATDAT_BIT2 = 0x00000004
CNNx16_n_MLAT_MLATDAT_BIT3 = 0x00000008
CNNx16_n_MLAT_MLATDAT_BIT4 = 0x00000010
CNNx16_n_MLAT_MLATDAT_BIT5 = 0x00000020
CNNx16_n_MLAT_MLATDAT_BIT6 = 0x00000040
CNNx16_n_MLAT_MLATDAT_BIT7 = 0x00000080
CNNx16_n_MLAT_MLATDAT_BIT8 = 0x00000100
CNNx16_n_MLAT_MLATDAT_BIT9 = 0x00000200
CNNx16_n_MLAT_MLATDAT_BIT10 = 0x00000400
CNNx16_n_MLAT_MLATDAT_BIT11 = 0x00000800
CNNx16_n_MLAT_MLATDAT_BIT12 = 0x00001000
CNNx16_n_MLAT_MLATDAT_BIT13 = 0x00002000
CNNx16_n_MLAT_MLATDAT_BIT14 = 0x00004000
CNNx16_n_MLAT_MLATDAT_BIT15 = 0x00008000
CNNx16_n_MLAT_MLATDAT_BIT16 = 0x00010000
CNNx16_n_MLAT_MLATDAT_BIT17 = 0x00020000
CNNx16_n_MLAT_MLATDAT_BIT18 = 0x00040000
CNNx16_n_MLAT_MLATDAT_BIT19 = 0x00080000
CNNx16_n_MLAT_MLATDAT_BIT20 = 0x00100000
CNNx16_n_MLAT_MLATDAT_BIT21 = 0x00200000
CNNx16_n_MLAT_MLATDAT_BIT22 = 0x00400000
CNNx16_n_MLAT_MLATDAT_BIT23 = 0x00800000
CNNx16_n_MLAT_MLATDAT_BIT24 = 0x01000000
CNNx16_n_MLAT_MLATDAT_BIT25 = 0x02000000
CNNx16_n_MLAT_MLATDAT_BIT26 = 0x04000000
CNNx16_n_MLAT_MLATDAT_BIT27 = 0x08000000
CNNx16_n_MLAT_MLATDAT_BIT28 = 0x10000000
CNNx16_n_MLAT_MLATDAT_BIT29 = 0x20000000
CNNx16_n_MLAT_MLATDAT_BIT30 = 0x40000000
CNNx16_n_MLAT_MLATDAT_BIT31 = 0x80000000
CNNx16_n_MLAT_MLATDAT_MASK = 0xFFFFFFFF
CNNx16_n_MLAT_MLATDAT_VALUEMASK = 0xFFFFFFFF
CNNx16_n_SRAM_DS_POS = 15
CNNx16_n_SRAM_DS = 0x00008000
CNNx16_n_SRAM_DS_MASK = 0x00008000
CNNx16_n_SRAM_DS_VALUEMASK = 0x00000001
CNNx16_n_SRAM_EXTACC_POS = 0
CNNx16_n_SRAM_EXTACC = 0x00000001
CNNx16_n_SRAM_EXTACC_MASK = 0x00000001
CNNx16_n_SRAM_EXTACC_VALUEMASK = 0x00000001
CNNx16_n_SRAM_LSBRAM_POS = 22
CNNx16_n_SRAM_LSBRAM = 0x00400000
CNNx16_n_SRAM_LSBRAM_MASK = 0x00400000
CNNx16_n_SRAM_LSBRAM_VALUEMASK = 0x00000001
CNNx16_n_SRAM_LSDRAM_POS = 19
CNNx16_n_SRAM_LSDRAM = 0x00080000
CNNx16_n_SRAM_LSDRAM_MASK = 0x00080000
CNNx16_n_SRAM_LSDRAM_VALUEMASK = 0x00000001
CNNx16_n_SRAM_LSMRAM_POS = 20
CNNx16_n_SRAM_LSMRAM = 0x00100000
CNNx16_n_SRAM_LSMRAM_MASK = 0x00100000
CNNx16_n_SRAM_LSMRAM_VALUEMASK = 0x00000001
CNNx16_n_SRAM_LSTRAM_POS = 21
CNNx16_n_SRAM_LSTRAM = 0x00200000
CNNx16_n_SRAM_LSTRAM_MASK = 0x00200000
CNNx16_n_SRAM_LSTRAM_VALUEMASK = 0x00000001
CNNx16_n_SRAM_PD_POS = 16
CNNx16_n_SRAM_PD = 0x00010000
CNNx16_n_SRAM_PD_MASK = 0x00010000
CNNx16_n_SRAM_PD_VALUEMASK = 0x00000001
CNNx16_n_SRAM_RA_POS = 6
CNNx16_n_SRAM_RA_BIT0 = 0x00000040
CNNx16_n_SRAM_RA_BIT1 = 0x00000080
CNNx16_n_SRAM_RA_MASK = 0x000000C0
CNNx16_n_SRAM_RA_VALUEMASK = 0x00000003
CNNx16_n_SRAM_RMARGIN_POS = 2
CNNx16_n_SRAM_RMARGIN_BIT0 = 0x00000004
CNNx16_n_SRAM_RMARGIN_BIT1 = 0x00000008
CNNx16_n_SRAM_RMARGIN_BIT2 = 0x00000010
CNNx16_n_SRAM_RMARGIN_BIT3 = 0x00000020
CNNx16_n_SRAM_RMARGIN_MASK = 0x0000003C
CNNx16_n_SRAM_RMARGIN_VALUEMASK = 0x0000000F
CNNx16_n_SRAM_RMARGIN_EN_POS = 1
CNNx16_n_SRAM_RMARGIN_EN = 0x00000002
CNNx16_n_SRAM_RMARGIN_EN_MASK = 0x00000002
CNNx16_n_SRAM_RMARGIN_EN_VALUEMASK = 0x00000001
CNNx16_n_SRAM_WNEG_EN_POS = 10
CNNx16_n_SRAM_WNEG_EN = 0x00000400
CNNx16_n_SRAM_WNEG_EN_MASK = 0x00000400
CNNx16_n_SRAM_WNEG_EN_VALUEMASK = 0x00000001
CNNx16_n_SRAM_WNEG_VOL_POS = 8
CNNx16_n_SRAM_WNEG_VOL_BIT0 = 0x00000100
CNNx16_n_SRAM_WNEG_VOL_BIT1 = 0x00000200
CNNx16_n_SRAM_WNEG_VOL_MASK = 0x00000300
CNNx16_n_SRAM_WNEG_VOL_VALUEMASK = 0x00000003
CNNx16_n_SRAM_WPULSE_POS = 11
CNNx16_n_SRAM_WPULSE_BIT0 = 0x00000800
CNNx16_n_SRAM_WPULSE_BIT1 = 0x00001000
CNNx16_n_SRAM_WPULSE_BIT2 = 0x00002000
CNNx16_n_SRAM_WPULSE_MASK = 0x00003800
CNNx16_n_SRAM_WPULSE_VALUEMASK = 0x00000007
CNNx16_n_Sz_FBUF_FBUF_MAX_POS = 0
CNNx16_n_Sz_FBUF_FBUF_MAX_BIT0 = 0x00000001
CNNx16_n_Sz_FBUF_FBUF_MAX_BIT1 = 0x00000002
CNNx16_n_Sz_FBUF_FBUF_MAX_BIT2 = 0x00000004
CNNx16_n_Sz_FBUF_FBUF_MAX_BIT3 = 0x00000008
CNNx16_n_Sz_FBUF_FBUF_MAX_BIT4 = 0x00000010
CNNx16_n_Sz_FBUF_FBUF_MAX_BIT5 = 0x00000020
CNNx16_n_Sz_FBUF_FBUF_MAX_BIT6 = 0x00000040
CNNx16_n_Sz_FBUF_FBUF_MAX_BIT7 = 0x00000080
CNNx16_n_Sz_FBUF_FBUF_MAX_BIT8 = 0x00000100
CNNx16_n_Sz_FBUF_FBUF_MAX_BIT9 = 0x00000200
CNNx16_n_Sz_FBUF_FBUF_MAX_BIT10 = 0x00000400
CNNx16_n_Sz_FBUF_FBUF_MAX_BIT11 = 0x00000800
CNNx16_n_Sz_FBUF_FBUF_MAX_BIT12 = 0x00001000
CNNx16_n_Sz_FBUF_FBUF_MAX_BIT13 = 0x00002000
CNNx16_n_Sz_FBUF_FBUF_MAX_BIT14 = 0x00004000
CNNx16_n_Sz_FBUF_FBUF_MAX_BIT15 = 0x00008000
CNNx16_n_Sz_FBUF_FBUF_MAX_BIT16 = 0x00010000
CNNx16_n_Sz_FBUF_FBUF_MAX_MASK = 0x0001FFFF
CNNx16_n_Sz_FBUF_FBUF_MAX_VALUEMASK = 0x0001FFFF
CNNx16_n_Sz_STRM0_STRM_ISVAL_POS = 0
CNNx16_n_Sz_STRM0_STRM_ISVAL_BIT0 = 0x00000001
CNNx16_n_Sz_STRM0_STRM_ISVAL_BIT1 = 0x00000002
CNNx16_n_Sz_STRM0_STRM_ISVAL_BIT2 = 0x00000004
CNNx16_n_Sz_STRM0_STRM_ISVAL_BIT3 = 0x00000008
CNNx16_n_Sz_STRM0_STRM_ISVAL_BIT4 = 0x00000010
CNNx16_n_Sz_STRM0_STRM_ISVAL_BIT5 = 0x00000020
CNNx16_n_Sz_STRM0_STRM_ISVAL_BIT6 = 0x00000040
CNNx16_n_Sz_STRM0_STRM_ISVAL_BIT7 = 0x00000080
CNNx16_n_Sz_STRM0_STRM_ISVAL_BIT8 = 0x00000100
CNNx16_n_Sz_STRM0_STRM_ISVAL_BIT9 = 0x00000200
CNNx16_n_Sz_STRM0_STRM_ISVAL_BIT10 = 0x00000400
CNNx16_n_Sz_STRM0_STRM_ISVAL_BIT11 = 0x00000800
CNNx16_n_Sz_STRM0_STRM_ISVAL_BIT12 = 0x00001000
CNNx16_n_Sz_STRM0_STRM_ISVAL_BIT13 = 0x00002000
CNNx16_n_Sz_STRM0_STRM_ISVAL_MASK = 0x00003FFF
CNNx16_n_Sz_STRM0_STRM_ISVAL_VALUEMASK = 0x00003FFF
CNNx16_n_Sz_STRM1_STRM_DSVAL1_POS = 4
CNNx16_n_Sz_STRM1_STRM_DSVAL1_BIT0 = 0x00000010
CNNx16_n_Sz_STRM1_STRM_DSVAL1_BIT1 = 0x00000020
CNNx16_n_Sz_STRM1_STRM_DSVAL1_BIT2 = 0x00000040
CNNx16_n_Sz_STRM1_STRM_DSVAL1_BIT3 = 0x00000080
CNNx16_n_Sz_STRM1_STRM_DSVAL1_BIT4 = 0x00000100
CNNx16_n_Sz_STRM1_STRM_DSVAL1_MASK = 0x000001F0
CNNx16_n_Sz_STRM1_STRM_DSVAL1_VALUEMASK = 0x0000001F
CNNx16_n_Sz_STRM1_STRM_DSVAL2_POS = 16
CNNx16_n_Sz_STRM1_STRM_DSVAL2_BIT0 = 0x00010000
CNNx16_n_Sz_STRM1_STRM_DSVAL2_BIT1 = 0x00020000
CNNx16_n_Sz_STRM1_STRM_DSVAL2_BIT2 = 0x00040000
CNNx16_n_Sz_STRM1_STRM_DSVAL2_BIT3 = 0x00080000
CNNx16_n_Sz_STRM1_STRM_DSVAL2_BIT4 = 0x00100000
CNNx16_n_Sz_STRM1_STRM_DSVAL2_BIT5 = 0x00200000
CNNx16_n_Sz_STRM1_STRM_DSVAL2_BIT6 = 0x00400000
CNNx16_n_Sz_STRM1_STRM_DSVAL2_BIT7 = 0x00800000
CNNx16_n_Sz_STRM1_STRM_DSVAL2_BIT8 = 0x01000000
CNNx16_n_Sz_STRM1_STRM_DSVAL2_BIT9 = 0x02000000
CNNx16_n_Sz_STRM1_STRM_DSVAL2_BIT10 = 0x04000000
CNNx16_n_Sz_STRM1_STRM_DSVAL2_BIT11 = 0x08000000
CNNx16_n_Sz_STRM1_STRM_DSVAL2_MASK = 0x0FFF0000
CNNx16_n_Sz_STRM1_STRM_DSVAL2_VALUEMASK = 0x00000FFF
CNNx16_n_Sz_STRM1_STRM_INVOL_POS = 0
CNNx16_n_Sz_STRM1_STRM_INVOL_BIT0 = 0x00000001
CNNx16_n_Sz_STRM1_STRM_INVOL_BIT1 = 0x00000002
CNNx16_n_Sz_STRM1_STRM_INVOL_BIT2 = 0x00000004
CNNx16_n_Sz_STRM1_STRM_INVOL_BIT3 = 0x00000008
CNNx16_n_Sz_STRM1_STRM_INVOL_MASK = 0x0000000F
CNNx16_n_Sz_STRM1_STRM_INVOL_VALUEMASK = 0x0000000F
CNNx16_n_TEST_BALLBDONE_POS = 21
CNNx16_n_TEST_BALLBDONE = 0x00200000
CNNx16_n_TEST_BALLBDONE_MASK = 0x00200000
CNNx16_n_TEST_BALLBDONE_VALUEMASK = 0x00000001
CNNx16_n_TEST_BALLBFAIL_POS = 17
CNNx16_n_TEST_BALLBFAIL = 0x00020000
CNNx16_n_TEST_BALLBFAIL_MASK = 0x00020000
CNNx16_n_TEST_BALLBFAIL_VALUEMASK = 0x00000001
CNNx16_n_TEST_BALLZDONE_POS = 25
CNNx16_n_TEST_BALLZDONE = 0x02000000
CNNx16_n_TEST_BALLZDONE_MASK = 0x02000000
CNNx16_n_TEST_BALLZDONE_VALUEMASK = 0x00000001
CNNx16_n_TEST_BBISTRUN_POS = 6
CNNx16_n_TEST_BBISTRUN = 0x00000040
CNNx16_n_TEST_BBISTRUN_MASK = 0x00000040
CNNx16_n_TEST_BBISTRUN_VALUEMASK = 0x00000001
CNNx16_n_TEST_BISTDONE_POS = 27
CNNx16_n_TEST_BISTDONE = 0x08000000
CNNx16_n_TEST_BISTDONE_MASK = 0x08000000
CNNx16_n_TEST_BISTDONE_VALUEMASK = 0x00000001
CNNx16_n_TEST_BISTFAIL_POS = 26
CNNx16_n_TEST_BISTFAIL = 0x04000000
CNNx16_n_TEST_BISTFAIL_MASK = 0x04000000
CNNx16_n_TEST_BISTFAIL_VALUEMASK = 0x00000001
CNNx16_n_TEST_BISTSEL_POS = 8
CNNx16_n_TEST_BISTSEL_BIT0 = 0x00000100
CNNx16_n_TEST_BISTSEL_BIT1 = 0x00000200
CNNx16_n_TEST_BISTSEL_BIT2 = 0x00000400
CNNx16_n_TEST_BISTSEL_BIT3 = 0x00000800
CNNx16_n_TEST_BISTSEL_BIT4 = 0x00001000
CNNx16_n_TEST_BISTSEL_BIT5 = 0x00002000
CNNx16_n_TEST_BISTSEL_MASK = 0x00003F00
CNNx16_n_TEST_BISTSEL_VALUEMASK = 0x0000003F
CNNx16_n_TEST_BRAMZ_POS = 7
CNNx16_n_TEST_BRAMZ = 0x00000080
CNNx16_n_TEST_BRAMZ_MASK = 0x00000080
CNNx16_n_TEST_BRAMZ_VALUEMASK = 0x00000001
CNNx16_n_TEST_MALLBDONE_POS = 19
CNNx16_n_TEST_MALLBDONE = 0x00080000
CNNx16_n_TEST_MALLBDONE_MASK = 0x00080000
CNNx16_n_TEST_MALLBDONE_VALUEMASK = 0x00000001
CNNx16_n_TEST_MALLBFAIL_POS = 15
CNNx16_n_TEST_MALLBFAIL = 0x00008000
CNNx16_n_TEST_MALLBFAIL_MASK = 0x00008000
CNNx16_n_TEST_MALLBFAIL_VALUEMASK = 0x00000001
CNNx16_n_TEST_MALLZDONE_POS = 23
CNNx16_n_TEST_MALLZDONE = 0x00800000
CNNx16_n_TEST_MALLZDONE_MASK = 0x00800000
CNNx16_n_TEST_MALLZDONE_VALUEMASK = 0x00000001
CNNx16_n_TEST_MBISTRUN_POS = 2
CNNx16_n_TEST_MBISTRUN = 0x00000004
CNNx16_n_TEST_MBISTRUN_MASK = 0x00000004
CNNx16_n_TEST_MBISTRUN_VALUEMASK = 0x00000001
CNNx16_n_TEST_MRAMZ_POS = 3
CNNx16_n_TEST_MRAMZ = 0x00000008
CNNx16_n_TEST_MRAMZ_MASK = 0x00000008
CNNx16_n_TEST_MRAMZ_VALUEMASK = 0x00000001
CNNx16_n_TEST_SALLBDONE_POS = 18
CNNx16_n_TEST_SALLBDONE = 0x00040000
CNNx16_n_TEST_SALLBDONE_MASK = 0x00040000
CNNx16_n_TEST_SALLBDONE_VALUEMASK = 0x00000001
CNNx16_n_TEST_SALLBFAIL_POS = 14
CNNx16_n_TEST_SALLBFAIL = 0x00004000
CNNx16_n_TEST_SALLBFAIL_MASK = 0x00004000
CNNx16_n_TEST_SALLBFAIL_VALUEMASK = 0x00000001
CNNx16_n_TEST_SALLZDONE_POS = 22
CNNx16_n_TEST_SALLZDONE = 0x00400000
CNNx16_n_TEST_SALLZDONE_MASK = 0x00400000
CNNx16_n_TEST_SALLZDONE_VALUEMASK = 0x00000001
CNNx16_n_TEST_SBISTRUN_POS = 0
CNNx16_n_TEST_SBISTRUN = 0x00000001
CNNx16_n_TEST_SBISTRUN_MASK = 0x00000001
CNNx16_n_TEST_SBISTRUN_VALUEMASK = 0x00000001
CNNx16_n_TEST_SRAMZ_POS = 1
CNNx16_n_TEST_SRAMZ = 0x00000002
CNNx16_n_TEST_SRAMZ_MASK = 0x00000002
CNNx16_n_TEST_SRAMZ_VALUEMASK = 0x00000001
CNNx16_n_TEST_TALLBDONE_POS = 20
CNNx16_n_TEST_TALLBDONE = 0x00100000
CNNx16_n_TEST_TALLBDONE_MASK = 0x00100000
CNNx16_n_TEST_TALLBDONE_VALUEMASK = 0x00000001
CNNx16_n_TEST_TALLBFAIL_POS = 16
CNNx16_n_TEST_TALLBFAIL = 0x00010000
CNNx16_n_TEST_TALLBFAIL_MASK = 0x00010000
CNNx16_n_TEST_TALLBFAIL_VALUEMASK = 0x00000001
CNNx16_n_TEST_TALLZDONE_POS = 24
CNNx16_n_TEST_TALLZDONE = 0x01000000
CNNx16_n_TEST_TALLZDONE_MASK = 0x01000000
CNNx16_n_TEST_TALLZDONE_VALUEMASK = 0x00000001
CNNx16_n_TEST_TBISTRUN_POS = 4
CNNx16_n_TEST_TBISTRUN = 0x00000010
CNNx16_n_TEST_TBISTRUN_MASK = 0x00000010
CNNx16_n_TEST_TBISTRUN_VALUEMASK = 0x00000001
CNNx16_n_TEST_TRAMZ_POS = 5
CNNx16_n_TEST_TRAMZ = 0x00000020
CNNx16_n_TEST_TRAMZ_MASK = 0x00000020
CNNx16_n_TEST_TRAMZ_VALUEMASK = 0x00000001
CNNx16_n_TEST_ZERODONE_POS = 28
CNNx16_n_TEST_ZERODONE = 0x10000000
CNNx16_n_TEST_ZERODONE_MASK = 0x10000000
CNNx16_n_TEST_ZERODONE_VALUEMASK = 0x00000001



def transform_regname_to_address(name) -> int:
    name = name.rstrip("_")
    if name in registers:
        return registers[name]
    raise ValueError(f"register '{name}' not found")


def transform_memname_to_address(name) -> int:
    name = name.rstrip("_")
    if name in memory:
        return memory[name]
    raise ValueError(f"memory '{name}' not found")
