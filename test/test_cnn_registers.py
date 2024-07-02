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
