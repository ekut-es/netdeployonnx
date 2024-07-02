import contextlib

from netdeployonnx.devices.max78000.core import CNNx16_Layer
from netdeployonnx.devices.max78000.synthesizer.grid import Node, NodeGrid

_backends: dict[str, callable] = None


def load_backends():
    global _backends
    if _backends is None:
        _backends = {}

        with contextlib.suppress(ImportError):
            from netdeployonnx.devices.max78000.synthesizer.backend.synth_core import (
                configure_layer_by_node,
            )

            _backends["synth_core"] = configure_layer_by_node

        def _not_implemented_backend(*args, **kwargs):
            raise NotImplementedError("synth backend not importable")

        _backends["not_implemented"] = _not_implemented_backend
    return _backends


def configure_layer_by_node(
    layer: CNNx16_Layer,
    grid: NodeGrid,
    node: Node,
    input_shape: list[int],
    **kwargs,
) -> list[int]:
    backends = load_backends()
    backend = kwargs.pop("backend", next(iter(backends), "not_implemented"))
    if backend in backends:
        return backends[backend](layer, grid, node, input_shape)  # returns out_shape
    else:
        raise ValueError(f"Unknown backend: {backend}")
