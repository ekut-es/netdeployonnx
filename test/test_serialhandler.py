from unittest import mock

import pytest

from netdeployonnx.devices.max78000.device_transport import serialhandler


def test_packet_order():
    "test basic packet order handling"
    pos = serialhandler.PacketOrderSender(mock.MagicMock())
    for pid in range(10):
        pos.enqueue(mock.MagicMock(packet_id=pid))
    assert pos.sent_sequence == (pos.current_sequence + 10)

    # now we would send with await pos.work()

    for pid in range(10):
        assert pos.accept_acknowledge(sequence=pos.sent_sequence - 10 + pid)
    # repeat should not be accepted
    assert pos.accept_acknowledge(sequence=pos.sent_sequence - 10 + 9) is False
    # out of order should not be accepted
    assert pos.accept_acknowledge(sequence=pos.sent_sequence - 10 + 0) is False


@pytest.mark.asyncio
async def test_packet_order_resend():
    "test basic packet order handling"
    PACKET_SEND_AMOUNT = 15  # noqa: N806 because constant
    PACKET_INTERRUPTED = 8  # noqa: N806 because constant

    data_handler = mock.AsyncMock()
    pos = serialhandler.PacketOrderSender(data_handler)

    for pid in range(PACKET_SEND_AMOUNT):
        pos.enqueue(mock.MagicMock(packet_id=pid))
    assert pos.sent_sequence == (pos.current_sequence + PACKET_SEND_AMOUNT)
    await pos.work()

    for pid in range(PACKET_INTERRUPTED + 1):
        if pid == PACKET_INTERRUPTED:
            continue
        assert pos.accept_acknowledge(
            sequence=pos.sent_sequence - PACKET_SEND_AMOUNT + pid
        )
    assert (
        pos.accept_acknowledge(
            sequence=pos.sent_sequence - PACKET_SEND_AMOUNT + PACKET_INTERRUPTED + 1
        )
        is False
    )
    await pos.work()
    assert data_handler.send_msgs.await_count == 2

    call_start_ids = [
        [i for i in range(pos.MAX_QUEUE_SIZE)],
        [
            PACKET_INTERRUPTED + i
            for i in range(PACKET_SEND_AMOUNT - PACKET_INTERRUPTED)
        ],
    ]

    for callid, call_args in enumerate(data_handler.send_msgs.await_args_list):
        (sendqueue,) = call_args[0]
        assert [msg.packet_id for msg in sendqueue] == call_start_ids[callid]
