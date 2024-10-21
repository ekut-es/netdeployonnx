import logging
import sys
from pathlib import Path

import torch

# TODO: kws20.v3, cifar100, bayer2rgb, faceid


def model_importer(
    model_name: str,
    class_name: str,
    ai8x_training_path: Path = Path("external/ai8x-training"),
) -> type:
    sys.path.append(str(ai8x_training_path))
    sys.path.append(str(ai8x_training_path / "models"))
    import ai8x

    ai8x.set_device(device=85, simulate=False, round_avg=False, verbose=True)

    # Instantiate the model
    file_module = __import__(model_name)
    if hasattr(file_module, class_name):
        return getattr(file_module, class_name)
    else:
        return None


def define_custom_onnx_parameters():
    # i need to import the class from the file ai85net-nas-cifar.py
    # i tried to add the path to the sys.path, but it still doesn't work
    # i also tried to import the class from the file, but it doesn't work either

    def torch_aten_exp2(g, input):
        # print("TYPE=", type(input), "INPUT=", input)
        # if input is a torch.Value and is a float
        # then we can just return 2**input
        # exp_constant = None
        # exp_constant = 0
        # if exp_constant is not None:
        #     return g.op("Const", torch.tensor(2.0**exp_constant, dtype=float))
        return g.op("Pow", torch.tensor(2.0), input)

    torch.onnx.register_custom_op_symbolic("aten::exp2", torch_aten_exp2, 1)


def export_ai8x_to_onnx(
    title: str,
    model_name: str,
    class_name: str,
    pth_file: Path,
    dest_path: Path,
    input_size: tuple,
    optional_class_kwargs: dict = {},
):
    try:
        if not model_name:
            raise Exception("cannot export model")

        model_class = model_importer(model_name, class_name)
        model = model_class(bias=True, **optional_class_kwargs)
        checkpoint = torch.load(
            pth_file,
            map_location="cpu",
        )
        missing_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if missing_keys:
            # logging.warn(f"missing keys: {missing_keys}")
            pass
        model.eval()  # set to evaluation mode

        dummy_input = torch.randn(*input_size)  # Example for an image input

        dest_name = f"{pth_file.stem}.onnx"
        dest_file = dest_path / dest_name

        torch.onnx.export(model, dummy_input, dest_file, export_params=True)
        logging.info(f"exported to {str(dest_file)}")
    except Exception:
        logging.exception(f"failed to export {title} to onnx because of ")


def main():
    trained_path = Path("external") / "ai8x-synthesis" / "trained"
    onnx_path = Path("test/data")

    define_custom_onnx_parameters()

    exported_models = {
        "CIFAR10": [
            "ai85net-nas-cifar",
            "AI85NASCifarNet",
            trained_path / "ai85-cifar10-qat8-q.pth.tar",
            onnx_path,
            (1, 3, 32, 32),
        ],
        "CIFAR100": [
            "ai85net-nas-cifar",
            "AI85NASCifarNet",
            trained_path / "ai85-cifar100-qat8-q.pth.tar",
            onnx_path,
            (1, 3, 32, 32),
            {
                "num_classes": 100,
            },
        ],
        "KWS20v3": [
            "ai85net-kws20-v3",
            "AI85KWS20Netv3",
            trained_path / "ai85-kws20_v3-qat8-q.pth.tar",
            onnx_path,
            (1, 128, 128),
        ],
        "bayer2rgb": [
            "ai85net-bayer2rgbnet",
            "bayer2rgbnet",
            trained_path / "ai85-bayer2rgb-qat8-q.pth.tar",
            onnx_path,
            (1, 4, 64, 64),
            {},
        ],
        "faceid": [
            "ai85net-faceid_112",
            "AI85FaceIDNet_112",
            trained_path / "ai85-faceid_112-qat-q.pth.tar",
            onnx_path,
            (1, 3, 112, 112),
            {
                "pre_layer_stride": 1,
                "bottleneck_settings": [
                    [1, 32, 48, 2, 2],
                    [1, 48, 64, 2, 4],
                    [1, 64, 64, 1, 2],
                    [1, 64, 96, 2, 4],
                    [1, 96, 128, 1, 2],
                ],
                "last_layer_width": 128,
                "emb_dimensionality": 64,
                "depthwise_bias": True,
                "reduced_depthwise_bias": True,
            },
        ],
    }

    for title, args in exported_models.items():
        export_ai8x_to_onnx(title, *args)


if __name__ == "__main__":
    main()
