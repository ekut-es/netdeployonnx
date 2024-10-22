import logging
import sys
from pathlib import Path

import numpy
import onnx
import torch

# TODO: kws20.v3, cifar100, bayer2rgb, faceid


def weights_in_kb_of_net(net_path: Path) -> int:
    with net_path.open("rb") as fx:
        model = onnx.load(fx)
        size = 0
        includes = []
        level = 0
        for node in model.graph.node:
            if node.op_type.lower() in ["conv", "gemm"]:
                level += 1
                includes.extend(node.input)
                includes.extend(node.output)
        for initi in model.graph.initializer:
            if initi.name not in includes:
                continue
            size += int(numpy.prod(initi.dims))
        return size


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
) -> Path | None:
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
        dest_file: Path = dest_path / dest_name

        torch.onnx.export(model, dummy_input, dest_file, export_params=True)
        logging.info(f"exported to {str(dest_file)}")
        return dest_file
    except Exception:
        logging.exception(f"failed to export {title} to onnx because of ")


def edit_net_input(model: onnx.ModelProto, new_input: tuple) -> onnx.ModelProto:
    graph = model.graph
    original = graph.input[0]
    graph.input.remove(original)  # remove first index, just assume len==1
    graph.input.insert(
        0,
        onnx.helper.make_tensor_value_info(
            name=original.name,
            elem_type=original.type.tensor_type.elem_type,
            shape=new_input,
            doc_string=original.doc_string,
        ),
    )
    return onnx.helper.make_model(graph=graph)


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
            {},
        ],
        "bayer2rgb": [
            "ai85net-bayer2rgbnet",
            "bayer2rgbnet",
            trained_path / "ai85-bayer2rgb-qat8-q.pth.tar",
            onnx_path,
            (1, 4, 64, 64),
            {},
        ],
        # "faceid112": [ # demo is not implemented
        #     "ai85net-faceid_112",
        #     "AI85FaceIDNet_112",
        #     trained_path / "ai85-faceid_112-qat-q.pth.tar",
        #     onnx_path,
        #     (1, 3, 112, 112),
        #     {
        #         "pre_layer_stride": 1,
        #         "bottleneck_settings": [
        #             [1, 32, 48, 2, 2],
        #             [1, 48, 64, 2, 4],
        #             [1, 64, 64, 1, 2],
        #             [1, 64, 96, 2, 4],
        #             [1, 96, 128, 1, 2],
        #         ],
        #         "last_layer_width": 128,
        #         "emb_dimensionality": 64,
        #         "depthwise_bias": True,
        #         "reduced_depthwise_bias": True,
        #     },
        # ],
        # "faceid": [
        #     "ai85net-faceid",
        #     "AI85FaceIDNet",
        #     trained_path / "ai85-faceid-qat8-q.pth.tar",#this is not exported, but defined in gen-demos-max78000.sh?
        #     onnx_path,
        #     (1, 3, 160, 120),
        #     {},
        # ],
    }

    edit_inputs_of_nets = {
        "KWS20v3": (128, 128, 1),  #  why?
    }

    nets = []
    if 1:
        for title, args in exported_models.items():
            net_path = export_ai8x_to_onnx(title, *args)
            print(f"saved {title} as {net_path}")
            if title in edit_inputs_of_nets:
                with open(net_path, "rb") as model_fx:
                    model = onnx.load(model_fx)
                mod_model = edit_net_input(model, edit_inputs_of_nets[title])
                onnx.save(mod_model, net_path)
            nets.append(net_path)

    else:
        nets.extend(
            [
                onnx_path / "cifar10_short.onnx",
                onnx_path / "cifar10.onnx",
                onnx_path / "ai85-bayer2rgb-qat8-q.pth.onnx",
                onnx_path / "ai85-cifar10-qat8-q.pth.onnx",
                onnx_path / "ai85-cifar100-qat8-q.pth.onnx",
                onnx_path / "ai85-faceid_112-qat-q.pth.onnx",
                onnx_path / "ai85-kws20_v3-qat8-q.pth.onnx",
            ]
        )

    for net in nets:
        if net:
            # find out weights:
            print(f'"{net.name}": {weights_in_kb_of_net(net)},')


if __name__ == "__main__":
    main()
