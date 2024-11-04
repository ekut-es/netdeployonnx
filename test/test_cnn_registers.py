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
from netdeployonnx.devices.max78000.cnn_registers import Reg_CNNx16_n_CTRL


def test_ctrl_field():
    reg = Reg_CNNx16_n_CTRL()
    reg.value = 0x52001243
    assert reg.CNN_EN == 1
    assert reg.RDY_SEL == 1
    assert reg.CLK_EN == 0
    assert reg.CALCMAX == 0
    assert reg.POOL_EN == 0
    assert reg.BIGDATA == 1
    assert reg.APBCLK_EN == 0
    assert reg.ONESHOT == 0
    assert reg.EXT_SYNC == 1
    assert reg.CNN_IRQ == 1
    assert reg.POOLRND == 0
    assert reg.STREAM_EN == 0
    assert reg.FIFO_EN == 0
    assert reg.MLAT_LD == 0
    assert reg.MLATCH_SEL == 0
    assert reg.LILBUF == 0
    assert reg.MEXPRESS == 0
    assert reg.SIMPLE1B == 0
    assert reg.FFIFO_EN == 0
    assert reg.FIFOGRP == 0
    assert reg.FCLK_DLY == 18
    assert reg.TIMESHFT == 1
    assert reg.QUPAC == 0
