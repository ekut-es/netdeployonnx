#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>


void CNN_ISR(void)
{
    // not synthesized
}


int cnn_continue(void)
{
    // not synthesized
}


int cnn_stop(void)
{
    // not synthesized
}


int cnn_load_weights(void)
{
}


int cnn_load_bias(void)
{
    memcpy_8to32((uint32_t *)0x50108000, bias_0, sizeof(uint8_t) * 0); // ('0x50108000', b'...')
    memcpy_8to32((uint32_t *)0x50508000, bias_0, sizeof(uint8_t) * 0); // ('0x50508000', b'...')
    memcpy_8to32((uint32_t *)0x50908000, bias_0, sizeof(uint8_t) * 0); // ('0x50908000', b'...')
    memcpy_8to32((uint32_t *)0x50D08000, bias_0, sizeof(uint8_t) * 0); // ('0x50D08000', b'...')
}


int cnn_init(void)
{
    *((volatile uint32_t *)0x00000000) = 0x00000000; // ('CNNx16_AOD_CTRL', 0)
    *((volatile uint32_t *)0x50100000) = 0x00100008; // ('CNNx16_0_CTRL', 1048584)
    *((volatile uint32_t *)0x50100004) = 0x0000040c; // ('CNNx16_0_SRAM', 1036)
    *((volatile uint32_t *)0x50100008) = 0x00000009; // ('CNNx16_0_LCNT_MAX', 9)


    return CNN_OK;
}


int cnn_configure(void)
{
    // Layer 0 quadrant 0
    *((volatile uint32_t *)0x50100010) = 0x00010021; // ('CNNx16_0_L0_RCNT', 65569)
    *((volatile uint32_t *)0x50100090) = 0x00010021; // ('CNNx16_0_L0_CCNT', 65569)
    *((volatile uint32_t *)0x50100290) = 0x00000001; // ('CNNx16_0_L0_STRIDE', 1)
    *((volatile uint32_t *)0x50100590) = 0x00000200; // ('CNNx16_0_L0_LCTRL0', 512)

    // Layer 1 quadrant 0
    *((volatile uint32_t *)0x50100014) = 0x0000001f; // ('CNNx16_0_L1_RCNT', 31)
    *((volatile uint32_t *)0x50100094) = 0x0000001f; // ('CNNx16_0_L1_CCNT', 31)
    *((volatile uint32_t *)0x50100294) = 0x00000001; // ('CNNx16_0_L1_STRIDE', 1)
    *((volatile uint32_t *)0x50100594) = 0x00000200; // ('CNNx16_0_L1_LCTRL0', 512)

    // Layer 2 quadrant 0
    *((volatile uint32_t *)0x50100018) = 0x00010021; // ('CNNx16_0_L2_RCNT', 65569)
    *((volatile uint32_t *)0x50100098) = 0x00010021; // ('CNNx16_0_L2_CCNT', 65569)
    *((volatile uint32_t *)0x50100298) = 0x00000001; // ('CNNx16_0_L2_STRIDE', 1)
    *((volatile uint32_t *)0x50100598) = 0x00000200; // ('CNNx16_0_L2_LCTRL0', 512)

    // Layer 3 quadrant 0
    *((volatile uint32_t *)0x5010001c) = 0x00010021; // ('CNNx16_0_L3_RCNT', 65569)
    *((volatile uint32_t *)0x5010009c) = 0x00010021; // ('CNNx16_0_L3_CCNT', 65569)
    *((volatile uint32_t *)0x5010019c) = 0x00000001; // ('CNNx16_0_L3_PRCNT', 1)
    *((volatile uint32_t *)0x5010021c) = 0x00000001; // ('CNNx16_0_L3_PCCNT', 1)
    *((volatile uint32_t *)0x5010029c) = 0x00000002; // ('CNNx16_0_L3_STRIDE', 2)
    *((volatile uint32_t *)0x5010059c) = 0x00000300; // ('CNNx16_0_L3_LCTRL0', 768)

    // Layer 4 quadrant 0
    *((volatile uint32_t *)0x50100020) = 0x0000000f; // ('CNNx16_0_L4_RCNT', 15)
    *((volatile uint32_t *)0x501000a0) = 0x0000000f; // ('CNNx16_0_L4_CCNT', 15)
    *((volatile uint32_t *)0x501002a0) = 0x00000001; // ('CNNx16_0_L4_STRIDE', 1)
    *((volatile uint32_t *)0x501005a0) = 0x00000200; // ('CNNx16_0_L4_LCTRL0', 512)

    // Layer 5 quadrant 0
    *((volatile uint32_t *)0x50100024) = 0x00010011; // ('CNNx16_0_L5_RCNT', 65553)
    *((volatile uint32_t *)0x501000a4) = 0x00010011; // ('CNNx16_0_L5_CCNT', 65553)
    *((volatile uint32_t *)0x501001a4) = 0x00000001; // ('CNNx16_0_L5_PRCNT', 1)
    *((volatile uint32_t *)0x50100224) = 0x00000001; // ('CNNx16_0_L5_PCCNT', 1)
    *((volatile uint32_t *)0x501002a4) = 0x00000002; // ('CNNx16_0_L5_STRIDE', 2)
    *((volatile uint32_t *)0x501005a4) = 0x00000300; // ('CNNx16_0_L5_LCTRL0', 768)

    // Layer 6 quadrant 0
    *((volatile uint32_t *)0x50100028) = 0x00000007; // ('CNNx16_0_L6_RCNT', 7)
    *((volatile uint32_t *)0x501000a8) = 0x00000007; // ('CNNx16_0_L6_CCNT', 7)
    *((volatile uint32_t *)0x501002a8) = 0x00000001; // ('CNNx16_0_L6_STRIDE', 1)
    *((volatile uint32_t *)0x501005a8) = 0x00000200; // ('CNNx16_0_L6_LCTRL0', 512)

    // Layer 7 quadrant 0
    *((volatile uint32_t *)0x5010002c) = 0x00010009; // ('CNNx16_0_L7_RCNT', 65545)
    *((volatile uint32_t *)0x501000ac) = 0x00010009; // ('CNNx16_0_L7_CCNT', 65545)
    *((volatile uint32_t *)0x501001ac) = 0x00000001; // ('CNNx16_0_L7_PRCNT', 1)
    *((volatile uint32_t *)0x5010022c) = 0x00000001; // ('CNNx16_0_L7_PCCNT', 1)
    *((volatile uint32_t *)0x501002ac) = 0x00000002; // ('CNNx16_0_L7_STRIDE', 2)
    *((volatile uint32_t *)0x501005ac) = 0x00000300; // ('CNNx16_0_L7_LCTRL0', 768)

    // Layer 8 quadrant 0
    *((volatile uint32_t *)0x50100030) = 0x00010005; // ('CNNx16_0_L8_RCNT', 65541)
    *((volatile uint32_t *)0x501000b0) = 0x00010005; // ('CNNx16_0_L8_CCNT', 65541)
    *((volatile uint32_t *)0x501002b0) = 0x00000001; // ('CNNx16_0_L8_STRIDE', 1)
    *((volatile uint32_t *)0x501005b0) = 0x00000200; // ('CNNx16_0_L8_LCTRL0', 512)

    // Layer 9 quadrant 0
    *((volatile uint32_t *)0x50100034) = 0x00000003; // ('CNNx16_0_L9_RCNT', 3)
    *((volatile uint32_t *)0x501000b4) = 0x00000003; // ('CNNx16_0_L9_CCNT', 3)
    *((volatile uint32_t *)0x501001b4) = 0x00000001; // ('CNNx16_0_L9_PRCNT', 1)
    *((volatile uint32_t *)0x50100234) = 0x00000001; // ('CNNx16_0_L9_PCCNT', 1)
    *((volatile uint32_t *)0x501002b4) = 0x00000002; // ('CNNx16_0_L9_STRIDE', 2)
    *((volatile uint32_t *)0x501005b4) = 0x00000300; // ('CNNx16_0_L9_LCTRL0', 768)


    return CNN_OK;
}


int cnn_start(void)
{
    *((volatile uint32_t *)0x50100000) = 0x00100808; // ('CNNx16_0_CTRL', 1050632)
    *((volatile uint32_t *)0x50500000) = 0x00100809; // ('CNNx16_1_CTRL', 1050633)
    *((volatile uint32_t *)0x50900000) = 0x00100809; // ('CNNx16_2_CTRL', 1050633)
    *((volatile uint32_t *)0x50d00000) = 0x00100809; // ('CNNx16_3_CTRL', 1050633)

    *((volatile uint32_t *)0x50100000) = 0x00100009; // ('CNNx16_0_CTRL', 1048585)

    return CNN_OK;
}


int cnn_unload(uint32_t *out_buf)
{
    // not synthesized
}


int cnn_enable(uint32_t clock_source, uint32_t clock_divider)
{
    // action: ACTION
}


int cnn_disable(void)
{
    // not synthesized
}
