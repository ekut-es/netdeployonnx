from unittest import mock

import pytest
import torch

from netdeployonnx.common.wrapper import Payload_Datatype

try:
    from hannah.backends import GPRCBackend
except ImportError:
    GPRCBackend = None


@pytest.mark.skipif(GPRCBackend is None, reason="hannah backend not installed")
def test_backend():
    backend = GPRCBackend(client_connect="localhost:28329")
    assert backend is not None
    assert backend.available()

    def _patched_export(model, inp, bytesio, *args, **kwargs):
        return bytesio.write(b"123")

    def _insec(*args, **kwargs):
        assert args[0] == "localhost:28329"
        assert kwargs["options"]
        assert kwargs["options"]["grpc.keepalive_timeout_ms"] == 4000
        m = mock.MagicMock(name="_insec")
        m.__enter__ = mock.MagicMock(return_value=m)
        m.unary_unary = mock.MagicMock(return_value=m)
        m.return_value = m
        m.deviceHandle = mock.MagicMock(name="deviceHandle")
        m.deviceHandle.handle = b"123"
        m.run_id = "run1234"
        m.payload = m
        m.datatype = Payload_Datatype.json
        m.data = b'{"result": "ok"}'
        return m

    with (
        mock.patch("torch.onnx.export", _patched_export),
        mock.patch("grpc.insecure_channel", _insec),
    ):
        backend.prepare(mock.MagicMock())
        assert backend.run(torch.tensor([1, 2, 3]))
        with mock.patch(
            "netdeployonnx.common.hannah_backend.ProfilingResult"
        ) as profiling_result:
            assert backend.profile(torch.tensor([1, 2, 3]))
            profiling_result.assert_called_once()
