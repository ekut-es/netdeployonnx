import os
import re
from collections import defaultdict

import yaml
from matplotlib import lines
from matplotlib import pyplot as plt

import netdeployonnx.devices.max78000.cnn_constants as cnn_constants


def get_regname_for_reg_addr(reg_addr: int) -> str:
    # var_dict = dict(vars(cnn_constants))  # noqa: F841
    for variable, value in cnn_constants.registers.items():
        if value == reg_addr:
            return variable
    raise Exception(f"Register address 0x{reg_addr:08X} not found in cnn_constants")


def get_fields_for_reg_name(regname: str) -> dict[str, tuple[int, int]]:
    var_dict = dict(vars(cnn_constants))
    fields = {}
    fields_intermed = {}
    for variable, value in var_dict.items():
        if variable.startswith(regname):
            if variable.endswith("POS"):
                fields_intermed[variable] = value
    sizes = {}
    prev_var = None
    prev_val = None
    for variable, value in sorted(fields_intermed.items(), key=lambda x: x[1]):
        if prev_var:
            sizes[prev_var] = value - prev_val
        prev_var = variable
        prev_val = value
    sizes[prev_var] = 32 - prev_val
    for variable, value in fields_intermed.items():
        fields[re.sub(r"_POS$", "", variable)] = (value, sizes[variable])
    return fields


class HexInt:
    value: int

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return hex(int(self.value))


yaml.add_representer(
    HexInt, lambda self, data: self.represent_scalar("tag:yaml.org,2002:int", str(data))
)


def update_registers_x16():
    filepath = "../syshelpers/max78000_registers_X16.yaml"
    with open(filepath) as file:
        data = yaml.safe_load(file)

    for regname, reg in data.items():
        if "CNNx16_n_Ly_" in regname:
            reg["register_size"] = 32
        reg["addr"] = HexInt(reg["addr"])

    with open(filepath + "_.yaml", "w") as file:
        yaml.dump(data, file)


def main():
    """
    Reads input register and value and prints out the register names
    from the value it received.
    comma and brackets and quotes are ignored.
    it asks again and again like repl
    """
    # update_registers_x16()

    if os.path.exists("analyzed.yaml"):
        data = defaultdict(lambda: defaultdict(dict))
        # rename register addresses to register names
        # var_dict = dict(vars(cnn_constants))
        cnn_configure_lines = cnn_configure_str.split("\n")
        for line in cnn_configure_lines:
            if line.strip().startswith("regs.append"):
                regex_res = re.search(r"0x([0-9A-Fa-f]+), 0x([0-9A-Fa-f]+)", line)
                reg_addr = int(regex_res.group(1), 16)
                reg_value = int(regex_res.group(2), 16)

                regname = get_regname_for_reg_addr(reg_addr)
                line = line.replace(regex_res.group(0), f'"{regname}"')

                core_layer_res = re.search(r"CNNx16_(\d)_L(\d+)", regname)
                core = int(core_layer_res.group(1))
                layer = int(core_layer_res.group(2))

                reg_settings = {"value": reg_value}
                regsubname = regname.replace(core_layer_res.group(0), "")
                generic_regname = f"CNNx16_n_Ly{regsubname}"
                for fieldname, (pos, size) in get_fields_for_reg_name(
                    generic_regname
                ).items():
                    mask = (1 << size) - 1
                    reg_settings[fieldname] = (reg_value >> pos) & mask
                    if reg_settings[fieldname] == 0:
                        del reg_settings[fieldname]

                data[core][layer][regname] = reg_settings
            # print(line)

        with open("analyzed.yaml", "w") as file:
            data = {k: dict(core) for k, core in data.items()}
            yaml.dump(data, file)

    draw_plot("analyzed.yaml")


def draw_plot(analyze_yaml):  # noqa: C901
    # draw a scatter plot for each core and layer
    with open(analyze_yaml) as file:
        data = yaml.safe_load(file)

    fig, axs = plt.subplots(5, 1)
    # axs = [axs]
    fig.set_size_inches(10, 7)
    fig.suptitle("Analyzed CNNx16 Registers")
    # angle 45 deg for x axis
    for ax in axs:
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(-1, 0x4000)
    fig.tight_layout()

    cores_n_layers = [f"C{core}L{layer}" for core in range(4) for layer in range(16)]

    for filter_func, label in [
        ((lambda r, sr: "RPTR" in r), "Read"),
        ((lambda r, sr: "WPTR" in r), "Write"),
        ((lambda r, sr: "MCNT" in r), "MCNT"),
    ]:
        datatypedict = defaultdict(list)
        for core, layers in data.items():
            for layer, regs in layers.items():
                for regname, reg in regs.items():
                    for subreg, value in reg.items():
                        if subreg == "value":
                            continue
                        if filter_func(regname, subreg):
                            datatypedict[f"C{core}L{layer}"].append(value % 0x4000)
        for dataidx in range(4):
            datatyped = [
                (
                    datatypedict[cnl][dataidx]
                    if cnl in datatypedict and len(datatypedict[cnl]) > dataidx
                    else None
                )
                for cnl in cores_n_layers
            ]
            # Plot data
            axs[dataidx].scatter(cores_n_layers, datatyped, alpha=0.3, label=f"{label}")
            # axs[dataidx].legend()

    offs_per_core_n_layer = defaultdict(list)
    prev_layer = None
    hit_per_layer = 0
    for x in weights_str.split("\n"):
        if x.strip().startswith("WEIGHTS:"):
            regex_res = re.search(r"0x([0-9A-Fa-f]+) => \((\d+)\)", x)
            addr = int(regex_res.group(1), 16)
            size = int(regex_res.group(2)) * 4
            offs_addr = addr & 0xFFFF
            layer_addr = ((addr >> 12) & 0xFFF) - 384
            print(
                f"{layer_addr:04X} => {layer_addr // 1024}|"
                f" {(layer_addr % 256) // 4} "
            )
            core = layer_addr // 1024
            layer = (layer_addr % 256) // 4

            if prev_layer == layer:
                hit_per_layer += 1
            else:
                hit_per_layer = 0
                prev_layer = layer

            offs_per_core_n_layer[(core, layer)].append(((offs_addr % 0x4000), size))

    max_hit_per_layer = 5
    offs = [[] for _ in range(max_hit_per_layer)]
    offs_size = [[] for _ in range(max_hit_per_layer)]
    for hit_per_layer in range(max_hit_per_layer):
        for core in range(4):
            for layer in range(16):
                if (core, layer) in offs_per_core_n_layer:
                    if len(offs_per_core_n_layer[(core, layer)]) > hit_per_layer:
                        offs_addr, size = offs_per_core_n_layer[(core, layer)][
                            hit_per_layer
                        ]
                        offs[hit_per_layer].append(offs_addr % 0x4000)
                        offs_size[hit_per_layer].append((offs_addr % 0x4000) + size)
                    else:
                        offs[hit_per_layer].append(None)
                        offs_size[hit_per_layer].append(None)
                else:
                    offs[hit_per_layer].append(None)
                    offs_size[hit_per_layer].append(None)
    for hit_per_layer in range(max_hit_per_layer):
        # do an x marker for each layer with kwarg marker='x'
        axs[hit_per_layer].scatter(
            cores_n_layers,
            offs[hit_per_layer],
            alpha=0.8,
            color="black",
            marker="x",
            label="WEIGHTS",
        )
        axs[hit_per_layer].scatter(
            cores_n_layers,
            offs_size[hit_per_layer],
            alpha=0.8,
            color="gray",
            marker="+",
            label="WEIGHTS size",
        )
        # axs[hit_per_layer].legend()
        # set ylim max according to data
        # get axis data
        max_y = max(
            [y for y in axs[hit_per_layer].dataLim.bounds[1::2] if y is not None]
        )
        axs[hit_per_layer].set_ylim(-1, max_y * 1.1)
        axs[hit_per_layer].grid(True)
        # set the grid color to have more alpha
        axs[hit_per_layer].grid(color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
        # set xlim so that each x value of cores_n_layers is displayed
        axs[hit_per_layer].set_xlim(-1, len(cores_n_layers))

    # do a global legend
    fig.legend()

    plt.savefig("analyzed_2.png")


def draw_plot_memlayout(analyze_yaml):  # noqa: C901
    with open(analyze_yaml) as file:
        data = yaml.safe_load(file)

    num_cores = len(data)
    num_layers = len(data[0]) if num_cores > 0 else 0
    fig, axs = plt.subplots(1, 3)
    # axs = [axs]
    axs = list(axs)

    # add a twinax of ax1
    axs.append(axs[1].twinx())
    twinax_idx = len(axs) - 1

    # plot cores and layers
    axs[0].set_title("Cores and Layers")
    axs[0].set_xlim(-1, num_cores)
    axs[0].set_ylim(-1, num_layers)

    axs[1].set_title("Memory")
    axs[2].set_title("Weights")

    # plot MRAM and SRAM
    axs[1].set_ylim(-0x80, 0x1000)
    axs[twinax_idx].set_ylim(-0x80, 0x1000)
    axs[2].set_ylim(-1, 15)
    axs[2].set_xlim(-1, 8)

    # set a title to that
    axs[0].set_xlabel("Core")
    axs[0].set_ylabel("Layer")
    axs[1].set_ylabel("TRAM")
    axs[twinax_idx].set_ylabel("SRAM")

    # set the width of axs 1 and twinax_idx to 0.1
    fig.set_size_inches(10, 5)
    fig.tight_layout()
    # after the tight layout, we want to set the width of the two axes 1 and twinax_idx
    axs[0].set_position([0.05, 0.1, 0.40, 0.85])
    axs[1].set_position([0.5, 0.1, 0.05, 0.85])
    axs[2].set_position([0.65, 0.1, 0.3, 0.85])

    def draw_line_between(ax0, ax1, coord_ax0, coord_ax1, color, label=None):
        layer_coord = (core, layer)  # noqa: F841
        mem_coord = (0 if mem_x == "TRAM" else 1, addr)  # noqa: F841
        axs_idx = 1 if mem_x == "TRAM" else twinax_idx  # noqa: F841

        # draw an arrow from the core + layer coordinate to the other axs
        trans_figure = fig.trans_figure.inverted()
        coord1 = trans_figure.transform(ax0.transData.transform(coord_ax0))
        coord2 = trans_figure.transform(ax1.transData.transform(coord_ax1))
        line = lines.Line2D(
            (coord1[0], coord2[0]),  # xdata
            (coord1[1], coord2[1]),  # ydata
            transform=fig.trans_figure,
            color=color,
            label=label,
        )
        fig.lines.append(line)

    colors = [
        (0, 0, 0.7, 0.5),
        (0.7, 0, 0.7, 0.5),
        (0, 0.7, 0.7, 0.5),
        (0.7, 0.7, 0, 0.5),
        (0.7, 0.7, 0.7, 0.5),
    ]
    for core, layers in data.items():
        for layer, regs in layers.items():
            reg_lookup = {
                "WPTR_BASE": "SRAM",
                "WPTR_MOFF": "SRAM",
                "MCNT": "SRAM",
                "RPTR_BASE": "SRAM",
            }

            arrow_srcs = {}
            for regname, reg in regs.items():
                for reg_search, loc in reg_lookup.items():
                    if regname.endswith(reg_search):
                        for subreg in reg:
                            if subreg == "value":
                                continue
                            arrow_srcs[len(arrow_srcs)] = (loc, reg[subreg])

            # draw
            for arrow_idx, (src, (mem_x, addr)) in enumerate(arrow_srcs.items()):
                layer_coord = (core, layer)
                mem_coord = (0 if mem_x == "TRAM" else 1, addr & 0x0FFF)
                axs_idx = 1 if mem_x == "TRAM" else twinax_idx

                # draw an arrow from the core + layer coordinate to the other axs
                draw_line_between(
                    axs[0],
                    axs[axs_idx],
                    layer_coord,
                    mem_coord,
                    colors[arrow_idx],
                    label=f"{mem_x} {core} {layer}",
                )

    prev_layer = None
    hit_per_layer = 0
    for x in weights_str.split("\n"):
        if x.strip().startswith("WEIGHTS:"):
            regex_res = re.search(r"0x([0-9A-Fa-f]+) => \((\d+)\)", x)
            addr = int(regex_res.group(1), 16)
            size = int(regex_res.group(2))  # noqa: F841
            offs_addr = addr & 0xFFFF
            layer_addr = ((addr >> 12) & 0xFFF) - 384
            # print(f'{layer_addr:04X} => {(layer_addr % 256) // 4}')
            layer = (layer_addr % 256) // 4
            # print(f'0x{addr:08X} [0x{offs_addr:04X}] => {size*4}')
            if prev_layer == layer:
                hit_per_layer += 1
            else:
                hit_per_layer = 0
                prev_layer = layer
            layer_coord = (hit_per_layer, layer)
            mem_coord = (0, offs_addr & 0x0FFF)
            axs_idx = 1 if mem_x == "TRAM" else twinax_idx

            # draw an arrow from the core + layer coordinate to the other axs
            draw_line_between(
                axs[2],
                axs[axs_idx],
                layer_coord,
                mem_coord,
                colors[hit_per_layer],
                label=f"WEIGHTS {layer}",
            )

    plt.savefig("analyzed.png")
    print("Saved analyzed.png")


weights_str = """
WEIGHTS: 0x50180000 => (723)
WEIGHTS: 0x50180510 => (642)
WEIGHTS: 0x50180990 => (21)
WEIGHTS: 0x50184000 => (723)
WEIGHTS: 0x50184510 => (642)
WEIGHTS: 0x50184990 => (21)
WEIGHTS: 0x50188000 => (723)
WEIGHTS: 0x50188510 => (642)
WEIGHTS: 0x50188990 => (21)
WEIGHTS: 0x5018C100 => (579)
WEIGHTS: 0x5018C510 => (642)
WEIGHTS: 0x5018C990 => (21)
WEIGHTS: 0x50190100 => (579)
WEIGHTS: 0x50190510 => (642)
WEIGHTS: 0x50190990 => (21)
WEIGHTS: 0x50194100 => (579)
WEIGHTS: 0x50194510 => (642)
WEIGHTS: 0x50194990 => (21)
WEIGHTS: 0x50198100 => (579)
WEIGHTS: 0x50198510 => (642)
WEIGHTS: 0x50198990 => (21)
WEIGHTS: 0x5019C100 => (579)
WEIGHTS: 0x5019C510 => (642)
WEIGHTS: 0x5019C990 => (21)
WEIGHTS: 0x501A0100 => (579)
WEIGHTS: 0x501A0510 => (642)
WEIGHTS: 0x501A0990 => (21)
WEIGHTS: 0x501A4100 => (579)
WEIGHTS: 0x501A4510 => (642)
WEIGHTS: 0x501A4990 => (21)
WEIGHTS: 0x501A8100 => (579)
WEIGHTS: 0x501A8510 => (642)
WEIGHTS: 0x501A8990 => (21)
WEIGHTS: 0x501AC100 => (579)
WEIGHTS: 0x501AC510 => (642)
WEIGHTS: 0x501AC990 => (21)
WEIGHTS: 0x501B0100 => (579)
WEIGHTS: 0x501B0510 => (642)
WEIGHTS: 0x501B0990 => (21)
WEIGHTS: 0x501B4100 => (579)
WEIGHTS: 0x501B4510 => (642)
WEIGHTS: 0x501B4990 => (21)
WEIGHTS: 0x501B8100 => (579)
WEIGHTS: 0x501B8510 => (642)
WEIGHTS: 0x501B8990 => (21)
WEIGHTS: 0x501BC100 => (579)
WEIGHTS: 0x501BC510 => (642)
WEIGHTS: 0x501BC990 => (21)
WEIGHTS: 0x50580100 => (579)
WEIGHTS: 0x50580510 => (642)
WEIGHTS: 0x50580990 => (21)
WEIGHTS: 0x50584100 => (579)
WEIGHTS: 0x50584510 => (642)
WEIGHTS: 0x50584990 => (21)
WEIGHTS: 0x50588100 => (579)
WEIGHTS: 0x50588510 => (642)
WEIGHTS: 0x50588990 => (21)
WEIGHTS: 0x5058C100 => (579)
WEIGHTS: 0x5058C510 => (642)
WEIGHTS: 0x5058C990 => (21)
WEIGHTS: 0x50590100 => (579)
WEIGHTS: 0x50590510 => (642)
WEIGHTS: 0x50590990 => (21)
WEIGHTS: 0x50594100 => (579)
WEIGHTS: 0x50594510 => (642)
WEIGHTS: 0x50594990 => (21)
WEIGHTS: 0x50598100 => (579)
WEIGHTS: 0x50598510 => (642)
WEIGHTS: 0x50598990 => (21)
WEIGHTS: 0x5059C100 => (579)
WEIGHTS: 0x5059C510 => (642)
WEIGHTS: 0x5059C990 => (21)
WEIGHTS: 0x505A0100 => (579)
WEIGHTS: 0x505A0510 => (642)
WEIGHTS: 0x505A0990 => (21)
WEIGHTS: 0x505A4100 => (579)
WEIGHTS: 0x505A4510 => (642)
WEIGHTS: 0x505A4990 => (21)
WEIGHTS: 0x505A8100 => (579)
WEIGHTS: 0x505A8510 => (642)
WEIGHTS: 0x505A8990 => (21)
WEIGHTS: 0x505AC100 => (579)
WEIGHTS: 0x505AC510 => (642)
WEIGHTS: 0x505AC990 => (21)
WEIGHTS: 0x505B0100 => (579)
WEIGHTS: 0x505B0510 => (642)
WEIGHTS: 0x505B0990 => (21)
WEIGHTS: 0x505B4100 => (579)
WEIGHTS: 0x505B4510 => (642)
WEIGHTS: 0x505B4990 => (21)
WEIGHTS: 0x505B8100 => (579)
WEIGHTS: 0x505B8510 => (642)
WEIGHTS: 0x505B8990 => (21)
WEIGHTS: 0x505BC100 => (579)
WEIGHTS: 0x505BC510 => (642)
WEIGHTS: 0x505BC990 => (21)
WEIGHTS: 0x50980000 => (18)
WEIGHTS: 0x50980100 => (9)
WEIGHTS: 0x50980210 => (426)
WEIGHTS: 0x50980510 => (642)
WEIGHTS: 0x50980990 => (21)
WEIGHTS: 0x50984000 => (18)
WEIGHTS: 0x50984100 => (9)
WEIGHTS: 0x50984210 => (426)
WEIGHTS: 0x50984510 => (642)
WEIGHTS: 0x50984990 => (21)
WEIGHTS: 0x50988000 => (18)
WEIGHTS: 0x50988100 => (9)
WEIGHTS: 0x50988210 => (426)
WEIGHTS: 0x50988510 => (642)
WEIGHTS: 0x50988990 => (21)
WEIGHTS: 0x5098C000 => (18)
WEIGHTS: 0x5098C100 => (9)
WEIGHTS: 0x5098C210 => (426)
WEIGHTS: 0x5098C510 => (642)
WEIGHTS: 0x5098C990 => (21)
WEIGHTS: 0x50990000 => (18)
WEIGHTS: 0x50990100 => (9)
WEIGHTS: 0x50990210 => (426)
WEIGHTS: 0x50990510 => (642)
WEIGHTS: 0x50990990 => (21)
WEIGHTS: 0x50994000 => (18)
WEIGHTS: 0x50994100 => (9)
WEIGHTS: 0x50994210 => (426)
WEIGHTS: 0x50994510 => (642)
WEIGHTS: 0x50994990 => (21)
WEIGHTS: 0x50998000 => (18)
WEIGHTS: 0x50998100 => (9)
WEIGHTS: 0x50998210 => (426)
WEIGHTS: 0x50998510 => (642)
WEIGHTS: 0x50998990 => (21)
WEIGHTS: 0x5099C000 => (18)
WEIGHTS: 0x5099C100 => (9)
WEIGHTS: 0x5099C210 => (426)
WEIGHTS: 0x5099C510 => (642)
WEIGHTS: 0x5099C990 => (21)
WEIGHTS: 0x509A0000 => (18)
WEIGHTS: 0x509A0100 => (9)
WEIGHTS: 0x509A0210 => (426)
WEIGHTS: 0x509A0510 => (642)
WEIGHTS: 0x509A0990 => (21)
WEIGHTS: 0x509A4000 => (18)
WEIGHTS: 0x509A4100 => (9)
WEIGHTS: 0x509A4210 => (426)
WEIGHTS: 0x509A4510 => (642)
WEIGHTS: 0x509A4990 => (21)
WEIGHTS: 0x509A8000 => (18)
WEIGHTS: 0x509A8100 => (9)
WEIGHTS: 0x509A8210 => (426)
WEIGHTS: 0x509A8510 => (642)
WEIGHTS: 0x509A8990 => (21)
WEIGHTS: 0x509AC000 => (18)
WEIGHTS: 0x509AC100 => (9)
WEIGHTS: 0x509AC210 => (426)
WEIGHTS: 0x509AC510 => (642)
WEIGHTS: 0x509AC990 => (21)
WEIGHTS: 0x509B0000 => (18)
WEIGHTS: 0x509B0100 => (9)
WEIGHTS: 0x509B0210 => (426)
WEIGHTS: 0x509B0510 => (642)
WEIGHTS: 0x509B0990 => (21)
WEIGHTS: 0x509B4000 => (18)
WEIGHTS: 0x509B4100 => (9)
WEIGHTS: 0x509B4210 => (426)
WEIGHTS: 0x509B4510 => (642)
WEIGHTS: 0x509B4990 => (21)
WEIGHTS: 0x509B8000 => (18)
WEIGHTS: 0x509B8100 => (9)
WEIGHTS: 0x509B8210 => (426)
WEIGHTS: 0x509B8510 => (642)
WEIGHTS: 0x509B8990 => (21)
WEIGHTS: 0x509BC000 => (18)
WEIGHTS: 0x509BC100 => (9)
WEIGHTS: 0x509BC210 => (426)
WEIGHTS: 0x509BC510 => (642)
WEIGHTS: 0x509BC990 => (21)
WEIGHTS: 0x50D80000 => (18)
WEIGHTS: 0x50D80100 => (9)
WEIGHTS: 0x50D80210 => (426)
WEIGHTS: 0x50D80510 => (642)
WEIGHTS: 0x50D80990 => (21)
WEIGHTS: 0x50D84000 => (18)
WEIGHTS: 0x50D84100 => (9)
WEIGHTS: 0x50D84210 => (426)
WEIGHTS: 0x50D84510 => (642)
WEIGHTS: 0x50D84990 => (21)
WEIGHTS: 0x50D88000 => (18)
WEIGHTS: 0x50D88100 => (9)
WEIGHTS: 0x50D88210 => (426)
WEIGHTS: 0x50D88510 => (642)
WEIGHTS: 0x50D88990 => (21)
WEIGHTS: 0x50D8C000 => (18)
WEIGHTS: 0x50D8C100 => (9)
WEIGHTS: 0x50D8C210 => (426)
WEIGHTS: 0x50D8C510 => (642)
WEIGHTS: 0x50D8C990 => (21)
WEIGHTS: 0x50D90000 => (18)
WEIGHTS: 0x50D90100 => (9)
WEIGHTS: 0x50D90210 => (426)
WEIGHTS: 0x50D90510 => (642)
WEIGHTS: 0x50D90990 => (21)
WEIGHTS: 0x50D94000 => (18)
WEIGHTS: 0x50D94100 => (9)
WEIGHTS: 0x50D94210 => (426)
WEIGHTS: 0x50D94510 => (642)
WEIGHTS: 0x50D94990 => (21)
WEIGHTS: 0x50D98000 => (18)
WEIGHTS: 0x50D98100 => (9)
WEIGHTS: 0x50D98210 => (426)
WEIGHTS: 0x50D98510 => (642)
WEIGHTS: 0x50D98990 => (21)
WEIGHTS: 0x50D9C000 => (18)
WEIGHTS: 0x50D9C100 => (9)
WEIGHTS: 0x50D9C210 => (426)
WEIGHTS: 0x50D9C510 => (642)
WEIGHTS: 0x50D9C990 => (21)
WEIGHTS: 0x50DA0000 => (18)
WEIGHTS: 0x50DA0100 => (9)
WEIGHTS: 0x50DA0210 => (426)
WEIGHTS: 0x50DA0510 => (642)
WEIGHTS: 0x50DA0990 => (21)
WEIGHTS: 0x50DA4000 => (18)
WEIGHTS: 0x50DA4100 => (9)
WEIGHTS: 0x50DA4210 => (426)
WEIGHTS: 0x50DA4510 => (642)
WEIGHTS: 0x50DA4990 => (21)
WEIGHTS: 0x50DA8000 => (18)
WEIGHTS: 0x50DA8100 => (9)
WEIGHTS: 0x50DA8210 => (426)
WEIGHTS: 0x50DA8510 => (642)
WEIGHTS: 0x50DA8990 => (21)
WEIGHTS: 0x50DAC000 => (18)
WEIGHTS: 0x50DAC100 => (9)
WEIGHTS: 0x50DAC210 => (426)
WEIGHTS: 0x50DAC510 => (642)
WEIGHTS: 0x50DAC990 => (21)
WEIGHTS: 0x50DB0000 => (18)
WEIGHTS: 0x50DB0100 => (9)
WEIGHTS: 0x50DB0210 => (426)
WEIGHTS: 0x50DB0510 => (642)
WEIGHTS: 0x50DB0990 => (21)
WEIGHTS: 0x50DB4000 => (18)
WEIGHTS: 0x50DB4100 => (9)
WEIGHTS: 0x50DB4210 => (426)
WEIGHTS: 0x50DB4510 => (642)
WEIGHTS: 0x50DB4990 => (21)
WEIGHTS: 0x50DB8000 => (18)
WEIGHTS: 0x50DB8100 => (9)
WEIGHTS: 0x50DB8210 => (426)
WEIGHTS: 0x50DB8510 => (642)
WEIGHTS: 0x50DB8990 => (21)
WEIGHTS: 0x50DBC000 => (18)
WEIGHTS: 0x50DBC100 => (9)
WEIGHTS: 0x50DBC210 => (426)
WEIGHTS: 0x50DBC510 => (642)
WEIGHTS: 0x50DBC990 => (21)
"""


cnn_configure_str = """
def cnn_configure():
    regs = []

    # Layer 0 quadrant 0
    regs.append((0x50100010, 0x00010021))  # Rows
    regs.append((0x50100090, 0x00010021))  # Columns
    regs.append((0x50100310, 0x00001000))  # SRAM write ptr
    regs.append((0x50100410, 0x00002000))  # Write ptr mask offs
    regs.append((0x50100590, 0x00002B20))  # Layer control
    regs.append((0x50100A10, 0x0001F800))  # Layer control 2
    regs.append((0x50100610, 0x000001F8))  # Mask offset and count
    regs.append((0x50100690, 0x0000001F))  # TRAM ptr max
    regs.append((0x50100710, 0x00070007))  # Mask and processor enables

    # Layer 0 quadrant 1
    regs.append((0x50500010, 0x00010021))  # Rows
    regs.append((0x50500090, 0x00010021))  # Columns
    regs.append((0x50500310, 0x00001000))  # SRAM write ptr
    regs.append((0x50500410, 0x00002000))  # Write ptr mask offs
    regs.append((0x50500590, 0x00000B20))  # Layer control
    regs.append((0x50500A10, 0x0001F800))  # Layer control 2
    regs.append((0x50500610, 0x000001F8))  # Mask offset and count
    regs.append((0x50500690, 0x0000001F))  # TRAM ptr max
    regs.append((0x50500790, 0x00001080))  # Post processing register

    # Layer 0 quadrant 2
    regs.append((0x50900010, 0x00010021))  # Rows
    regs.append((0x50900090, 0x00010021))  # Columns
    regs.append((0x50900310, 0x00001000))  # SRAM write ptr
    regs.append((0x50900410, 0x00002000))  # Write ptr mask offs
    regs.append((0x50900590, 0x00000B20))  # Layer control
    regs.append((0x50900A10, 0x0001F800))  # Layer control 2
    regs.append((0x50900610, 0x000001F8))  # Mask offset and count
    regs.append((0x50900690, 0x0000001F))  # TRAM ptr max

    # Layer 0 quadrant 3
    regs.append((0x50D00010, 0x00010021))  # Rows
    regs.append((0x50D00090, 0x00010021))  # Columns
    regs.append((0x50D00310, 0x00001000))  # SRAM write ptr
    regs.append((0x50D00410, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D00590, 0x00000B20))  # Layer control
    regs.append((0x50D00A10, 0x0001F800))  # Layer control 2
    regs.append((0x50D00610, 0x000001F8))  # Mask offset and count
    regs.append((0x50D00690, 0x0000001F))  # TRAM ptr max

    # Layer 1 quadrant 0
    regs.append((0x50100014, 0x0000001F))  # Rows
    regs.append((0x50100094, 0x0000001F))  # Columns
    regs.append((0x50100394, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50100414, 0x00002000))  # Write ptr mask offs
    regs.append((0x50100514, 0x00001000))  # SRAM read ptr
    regs.append((0x50100594, 0x0000EB20))  # Layer control
    regs.append((0x50100A14, 0x0000F800))  # Layer control 2
    regs.append((0x50100614, 0x120012F8))  # Mask offset and count
    regs.append((0x50100114, 0x00000100))  # 1D
    regs.append((0x50100794, 0x00022000))  # Post processing register
    regs.append((0x50100714, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 1 quadrant 1
    regs.append((0x50500014, 0x0000001F))  # Rows
    regs.append((0x50500094, 0x0000001F))  # Columns
    regs.append((0x50500394, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50500414, 0x00002000))  # Write ptr mask offs
    regs.append((0x50500514, 0x00001000))  # SRAM read ptr
    regs.append((0x50500594, 0x00000B20))  # Layer control
    regs.append((0x50500A14, 0x0000F800))  # Layer control 2
    regs.append((0x50500614, 0x120012F8))  # Mask offset and count
    regs.append((0x50500114, 0x00000100))  # 1D
    regs.append((0x50500794, 0x000230C0))  # Post processing register
    regs.append((0x50500714, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 1 quadrant 2
    regs.append((0x50900014, 0x0000001F))  # Rows
    regs.append((0x50900094, 0x0000001F))  # Columns
    regs.append((0x50900394, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50900414, 0x00002000))  # Write ptr mask offs
    regs.append((0x50900514, 0x00001000))  # SRAM read ptr
    regs.append((0x50900594, 0x00000B20))  # Layer control
    regs.append((0x50900A14, 0x0000F800))  # Layer control 2
    regs.append((0x50900614, 0x120012F8))  # Mask offset and count
    regs.append((0x50900114, 0x00000100))  # 1D
    regs.append((0x50900794, 0x00022000))  # Post processing register
    regs.append((0x50900714, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 1 quadrant 3
    regs.append((0x50D00014, 0x0000001F))  # Rows
    regs.append((0x50D00094, 0x0000001F))  # Columns
    regs.append((0x50D00394, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50D00414, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D00514, 0x00001000))  # SRAM read ptr
    regs.append((0x50D00594, 0x00000B20))  # Layer control
    regs.append((0x50D00A14, 0x0000F800))  # Layer control 2
    regs.append((0x50D00614, 0x120012F8))  # Mask offset and count
    regs.append((0x50D00114, 0x00000100))  # 1D
    regs.append((0x50D00794, 0x00022000))  # Post processing register
    regs.append((0x50D00714, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 2 quadrant 0
    regs.append((0x50100018, 0x00010021))  # Rows
    regs.append((0x50100098, 0x00010021))  # Columns
    regs.append((0x50100318, 0x00001000))  # SRAM write ptr
    regs.append((0x50100418, 0x00002000))  # Write ptr mask offs
    regs.append((0x50100598, 0x00006B20))  # Layer control
    regs.append((0x50100A18, 0x0001F800))  # Layer control 2
    regs.append((0x50100618, 0x02200418))  # Mask offset and count
    regs.append((0x50100698, 0x0000001F))  # TRAM ptr max
    regs.append((0x50100798, 0x00022000))  # Post processing register
    regs.append((0x50100718, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 2 quadrant 1
    regs.append((0x50500018, 0x00010021))  # Rows
    regs.append((0x50500098, 0x00010021))  # Columns
    regs.append((0x50500318, 0x00001000))  # SRAM write ptr
    regs.append((0x50500418, 0x00002000))  # Write ptr mask offs
    regs.append((0x50500598, 0x00000B20))  # Layer control
    regs.append((0x50500A18, 0x0001F800))  # Layer control 2
    regs.append((0x50500618, 0x02200418))  # Mask offset and count
    regs.append((0x50500698, 0x0000001F))  # TRAM ptr max
    regs.append((0x50500798, 0x00022000))  # Post processing register
    regs.append((0x50500718, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 2 quadrant 2
    regs.append((0x50900018, 0x00010021))  # Rows
    regs.append((0x50900098, 0x00010021))  # Columns
    regs.append((0x50900318, 0x00001000))  # SRAM write ptr
    regs.append((0x50900418, 0x00002000))  # Write ptr mask offs
    regs.append((0x50900598, 0x00000B20))  # Layer control
    regs.append((0x50900A18, 0x0001F800))  # Layer control 2
    regs.append((0x50900618, 0x02200418))  # Mask offset and count
    regs.append((0x50900698, 0x0000001F))  # TRAM ptr max
    regs.append((0x50900798, 0x00023080))  # Post processing register

    # Layer 2 quadrant 3
    regs.append((0x50D00018, 0x00010021))  # Rows
    regs.append((0x50D00098, 0x00010021))  # Columns
    regs.append((0x50D00318, 0x00001000))  # SRAM write ptr
    regs.append((0x50D00418, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D00598, 0x00000B20))  # Layer control
    regs.append((0x50D00A18, 0x0001F800))  # Layer control 2
    regs.append((0x50D00618, 0x02200418))  # Mask offset and count
    regs.append((0x50D00698, 0x0000001F))  # TRAM ptr max
    regs.append((0x50D00798, 0x00022000))  # Post processing register

    # Layer 3 quadrant 0
    regs.append((0x5010001C, 0x00010021))  # Rows
    regs.append((0x5010009C, 0x00010021))  # Columns
    regs.append((0x5010019C, 0x00000001))  # Pooling rows
    regs.append((0x5010021C, 0x00000001))  # Pooling columns
    regs.append((0x5010029C, 0x00000001))  # Stride
    regs.append((0x5010031C, 0x00010000))  # SRAM write ptr
    regs.append((0x5010041C, 0x00002000))  # Write ptr mask offs
    regs.append((0x5010051C, 0x00001000))  # SRAM read ptr
    regs.append((0x5010059C, 0x0000EBA0))  # Layer control
    regs.append((0x50100A1C, 0x0000F800))  # Layer control 2
    regs.append((0x5010061C, 0x04200518))  # Mask offset and count
    regs.append((0x5010069C, 0x0000000F))  # TRAM ptr max
    regs.append((0x5010079C, 0x00026000))  # Post processing register
    regs.append((0x5010071C, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 3 quadrant 1
    regs.append((0x5050001C, 0x00010021))  # Rows
    regs.append((0x5050009C, 0x00010021))  # Columns
    regs.append((0x5050019C, 0x00000001))  # Pooling rows
    regs.append((0x5050021C, 0x00000001))  # Pooling columns
    regs.append((0x5050029C, 0x00000001))  # Stride
    regs.append((0x5050031C, 0x00010000))  # SRAM write ptr
    regs.append((0x5050041C, 0x00002000))  # Write ptr mask offs
    regs.append((0x5050051C, 0x00001000))  # SRAM read ptr
    regs.append((0x5050059C, 0x00000BA0))  # Layer control
    regs.append((0x50500A1C, 0x0000F800))  # Layer control 2
    regs.append((0x5050061C, 0x04200518))  # Mask offset and count
    regs.append((0x5050069C, 0x0000000F))  # TRAM ptr max
    regs.append((0x5050079C, 0x00026000))  # Post processing register
    regs.append((0x5050071C, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 3 quadrant 2
    regs.append((0x5090001C, 0x00010021))  # Rows
    regs.append((0x5090009C, 0x00010021))  # Columns
    regs.append((0x5090019C, 0x00000001))  # Pooling rows
    regs.append((0x5090021C, 0x00000001))  # Pooling columns
    regs.append((0x5090029C, 0x00000001))  # Stride
    regs.append((0x5090031C, 0x00010000))  # SRAM write ptr
    regs.append((0x5090041C, 0x00002000))  # Write ptr mask offs
    regs.append((0x5090051C, 0x00001000))  # SRAM read ptr
    regs.append((0x5090059C, 0x00000BA0))  # Layer control
    regs.append((0x50900A1C, 0x0000F800))  # Layer control 2
    regs.append((0x5090061C, 0x04200518))  # Mask offset and count
    regs.append((0x5090069C, 0x0000000F))  # TRAM ptr max
    regs.append((0x5090079C, 0x000270C0))  # Post processing register
    regs.append((0x5090071C, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 3 quadrant 3
    regs.append((0x50D0001C, 0x00010021))  # Rows
    regs.append((0x50D0009C, 0x00010021))  # Columns
    regs.append((0x50D0019C, 0x00000001))  # Pooling rows
    regs.append((0x50D0021C, 0x00000001))  # Pooling columns
    regs.append((0x50D0029C, 0x00000001))  # Stride
    regs.append((0x50D0031C, 0x00010000))  # SRAM write ptr
    regs.append((0x50D0041C, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D0051C, 0x00001000))  # SRAM read ptr
    regs.append((0x50D0059C, 0x00000BA0))  # Layer control
    regs.append((0x50D00A1C, 0x0000F800))  # Layer control 2
    regs.append((0x50D0061C, 0x04200518))  # Mask offset and count
    regs.append((0x50D0069C, 0x0000000F))  # TRAM ptr max
    regs.append((0x50D0079C, 0x00026000))  # Post processing register
    regs.append((0x50D0071C, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 4 quadrant 0
    regs.append((0x50100020, 0x0000000F))  # Rows
    regs.append((0x501000A0, 0x0000000F))  # Columns
    regs.append((0x50100320, 0x00001000))  # SRAM write ptr
    regs.append((0x501003A0, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50100420, 0x00002000))  # Write ptr mask offs
    regs.append((0x501005A0, 0x0000CB20))  # Layer control
    regs.append((0x50100A20, 0x0001F800))  # Layer control 2
    regs.append((0x50100620, 0x000001F8))  # Mask offset and count
    regs.append((0x50100120, 0x00000100))  # 1D

    # Layer 4 quadrant 1
    regs.append((0x50500020, 0x0000000F))  # Rows
    regs.append((0x505000A0, 0x0000000F))  # Columns
    regs.append((0x50500320, 0x00001000))  # SRAM write ptr
    regs.append((0x505003A0, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50500420, 0x00002000))  # Write ptr mask offs
    regs.append((0x505005A0, 0x00000B20))  # Layer control
    regs.append((0x50500A20, 0x0001F800))  # Layer control 2
    regs.append((0x50500620, 0x000001F8))  # Mask offset and count
    regs.append((0x50500120, 0x00000100))  # 1D

    # Layer 4 quadrant 2
    regs.append((0x50900020, 0x0000000F))  # Rows
    regs.append((0x509000A0, 0x0000000F))  # Columns
    regs.append((0x50900320, 0x00001000))  # SRAM write ptr
    regs.append((0x509003A0, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50900420, 0x00002000))  # Write ptr mask offs
    regs.append((0x509005A0, 0x00000B20))  # Layer control
    regs.append((0x50900A20, 0x0001F800))  # Layer control 2
    regs.append((0x50900620, 0x000001F8))  # Mask offset and count
    regs.append((0x50900120, 0x00000100))  # 1D
    regs.append((0x50900720, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 4 quadrant 3
    regs.append((0x50D00020, 0x0000000F))  # Rows
    regs.append((0x50D000A0, 0x0000000F))  # Columns
    regs.append((0x50D00320, 0x00001000))  # SRAM write ptr
    regs.append((0x50D003A0, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50D00420, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D005A0, 0x00000B20))  # Layer control
    regs.append((0x50D00A20, 0x0001F800))  # Layer control 2
    regs.append((0x50D00620, 0x000001F8))  # Mask offset and count
    regs.append((0x50D00120, 0x00000100))  # 1D
    regs.append((0x50D007A0, 0x00001080))  # Post processing register
    regs.append((0x50D00720, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 5 quadrant 0
    regs.append((0x50100024, 0x00010011))  # Rows
    regs.append((0x501000A4, 0x00010011))  # Columns
    regs.append((0x501001A4, 0x00000001))  # Pooling rows
    regs.append((0x50100224, 0x00000001))  # Pooling columns
    regs.append((0x501002A4, 0x00000001))  # Stride
    regs.append((0x50100424, 0x00002000))  # Write ptr mask offs
    regs.append((0x501004A4, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50100524, 0x00001000))  # SRAM read ptr
    regs.append((0x501005A4, 0x0000EBA0))  # Layer control
    regs.append((0x50100A24, 0x0001F810))  # Layer control 2
    regs.append((0x50100624, 0x05200918))  # Mask offset and count
    regs.append((0x501006A4, 0x00000007))  # TRAM ptr max
    regs.append((0x501007A4, 0x00026000))  # Post processing register
    regs.append((0x50100724, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 5 quadrant 1
    regs.append((0x50500024, 0x00010011))  # Rows
    regs.append((0x505000A4, 0x00010011))  # Columns
    regs.append((0x505001A4, 0x00000001))  # Pooling rows
    regs.append((0x50500224, 0x00000001))  # Pooling columns
    regs.append((0x505002A4, 0x00000001))  # Stride
    regs.append((0x50500424, 0x00002000))  # Write ptr mask offs
    regs.append((0x505004A4, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50500524, 0x00001000))  # SRAM read ptr
    regs.append((0x505005A4, 0x00000BA0))  # Layer control
    regs.append((0x50500A24, 0x0001F810))  # Layer control 2
    regs.append((0x50500624, 0x05200918))  # Mask offset and count
    regs.append((0x505006A4, 0x00000007))  # TRAM ptr max
    regs.append((0x505007A4, 0x00027000))  # Post processing register
    regs.append((0x50500724, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 5 quadrant 2
    regs.append((0x50900024, 0x00010011))  # Rows
    regs.append((0x509000A4, 0x00010011))  # Columns
    regs.append((0x509001A4, 0x00000001))  # Pooling rows
    regs.append((0x50900224, 0x00000001))  # Pooling columns
    regs.append((0x509002A4, 0x00000001))  # Stride
    regs.append((0x50900424, 0x00002000))  # Write ptr mask offs
    regs.append((0x509004A4, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50900524, 0x00001000))  # SRAM read ptr
    regs.append((0x509005A4, 0x00000BA0))  # Layer control
    regs.append((0x50900A24, 0x0001F810))  # Layer control 2
    regs.append((0x50900624, 0x05200918))  # Mask offset and count
    regs.append((0x509006A4, 0x00000007))  # TRAM ptr max
    regs.append((0x509007A4, 0x00026000))  # Post processing register
    regs.append((0x50900724, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 5 quadrant 3
    regs.append((0x50D00024, 0x00010011))  # Rows
    regs.append((0x50D000A4, 0x00010011))  # Columns
    regs.append((0x50D001A4, 0x00000001))  # Pooling rows
    regs.append((0x50D00224, 0x00000001))  # Pooling columns
    regs.append((0x50D002A4, 0x00000001))  # Stride
    regs.append((0x50D00424, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D004A4, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50D00524, 0x00001000))  # SRAM read ptr
    regs.append((0x50D005A4, 0x00000BA0))  # Layer control
    regs.append((0x50D00A24, 0x0001F810))  # Layer control 2
    regs.append((0x50D00624, 0x05200918))  # Mask offset and count
    regs.append((0x50D006A4, 0x00000007))  # TRAM ptr max
    regs.append((0x50D007A4, 0x00026000))  # Post processing register
    regs.append((0x50D00724, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 6 quadrant 0
    regs.append((0x50100028, 0x00000007))  # Rows
    regs.append((0x501000A8, 0x00000007))  # Columns
    regs.append((0x50100328, 0x00001000))  # SRAM write ptr
    regs.append((0x501003A8, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50100428, 0x00002000))  # Write ptr mask offs
    regs.append((0x501004A8, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x501005A8, 0x0000EB20))  # Layer control
    regs.append((0x50100A28, 0x0001F811))  # Layer control 2
    regs.append((0x50100628, 0x52205A18))  # Mask offset and count
    regs.append((0x50100128, 0x00000100))  # 1D
    regs.append((0x501007A8, 0x00022000))  # Post processing register
    regs.append((0x50100728, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 6 quadrant 1
    regs.append((0x50500028, 0x00000007))  # Rows
    regs.append((0x505000A8, 0x00000007))  # Columns
    regs.append((0x50500328, 0x00001000))  # SRAM write ptr
    regs.append((0x505003A8, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50500428, 0x00002000))  # Write ptr mask offs
    regs.append((0x505004A8, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x505005A8, 0x00000B20))  # Layer control
    regs.append((0x50500A28, 0x0001F811))  # Layer control 2
    regs.append((0x50500628, 0x52205A18))  # Mask offset and count
    regs.append((0x50500128, 0x00000100))  # 1D
    regs.append((0x505007A8, 0x00022000))  # Post processing register
    regs.append((0x50500728, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 6 quadrant 2
    regs.append((0x50900028, 0x00000007))  # Rows
    regs.append((0x509000A8, 0x00000007))  # Columns
    regs.append((0x50900328, 0x00001000))  # SRAM write ptr
    regs.append((0x509003A8, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50900428, 0x00002000))  # Write ptr mask offs
    regs.append((0x509004A8, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x509005A8, 0x00000B20))  # Layer control
    regs.append((0x50900A28, 0x0001F811))  # Layer control 2
    regs.append((0x50900628, 0x52205A18))  # Mask offset and count
    regs.append((0x50900128, 0x00000100))  # 1D
    regs.append((0x509007A8, 0x00023000))  # Post processing register
    regs.append((0x50900728, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 6 quadrant 3
    regs.append((0x50D00028, 0x00000007))  # Rows
    regs.append((0x50D000A8, 0x00000007))  # Columns
    regs.append((0x50D00328, 0x00001000))  # SRAM write ptr
    regs.append((0x50D003A8, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50D00428, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D004A8, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50D005A8, 0x00000B20))  # Layer control
    regs.append((0x50D00A28, 0x0001F811))  # Layer control 2
    regs.append((0x50D00628, 0x52205A18))  # Mask offset and count
    regs.append((0x50D00128, 0x00000100))  # 1D
    regs.append((0x50D007A8, 0x00022000))  # Post processing register
    regs.append((0x50D00728, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 7 quadrant 0
    regs.append((0x5010002C, 0x00010009))  # Rows
    regs.append((0x501000AC, 0x00010009))  # Columns
    regs.append((0x501001AC, 0x00000001))  # Pooling rows
    regs.append((0x5010022C, 0x00000001))  # Pooling columns
    regs.append((0x501002AC, 0x00000001))  # Stride
    regs.append((0x5010042C, 0x00002000))  # Write ptr mask offs
    regs.append((0x5010052C, 0x00001000))  # SRAM read ptr
    regs.append((0x501005AC, 0x0000EBA0))  # Layer control
    regs.append((0x50100A2C, 0x0001F801))  # Layer control 2
    regs.append((0x5010062C, 0x0A200E18))  # Mask offset and count
    regs.append((0x501006AC, 0x00000003))  # TRAM ptr max
    regs.append((0x501007AC, 0x0002708A))  # Post processing register
    regs.append((0x5010072C, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 7 quadrant 1
    regs.append((0x5050002C, 0x00010009))  # Rows
    regs.append((0x505000AC, 0x00010009))  # Columns
    regs.append((0x505001AC, 0x00000001))  # Pooling rows
    regs.append((0x5050022C, 0x00000001))  # Pooling columns
    regs.append((0x505002AC, 0x00000001))  # Stride
    regs.append((0x5050042C, 0x00002000))  # Write ptr mask offs
    regs.append((0x5050052C, 0x00001000))  # SRAM read ptr
    regs.append((0x505005AC, 0x00000BA0))  # Layer control
    regs.append((0x50500A2C, 0x0001F801))  # Layer control 2
    regs.append((0x5050062C, 0x0A200E18))  # Mask offset and count
    regs.append((0x505006AC, 0x00000003))  # TRAM ptr max
    regs.append((0x505007AC, 0x00026000))  # Post processing register
    regs.append((0x5050072C, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 7 quadrant 2
    regs.append((0x5090002C, 0x00010009))  # Rows
    regs.append((0x509000AC, 0x00010009))  # Columns
    regs.append((0x509001AC, 0x00000001))  # Pooling rows
    regs.append((0x5090022C, 0x00000001))  # Pooling columns
    regs.append((0x509002AC, 0x00000001))  # Stride
    regs.append((0x5090042C, 0x00002000))  # Write ptr mask offs
    regs.append((0x5090052C, 0x00001000))  # SRAM read ptr
    regs.append((0x509005AC, 0x00000BA0))  # Layer control
    regs.append((0x50900A2C, 0x0001F801))  # Layer control 2
    regs.append((0x5090062C, 0x0A200E18))  # Mask offset and count
    regs.append((0x509006AC, 0x00000003))  # TRAM ptr max
    regs.append((0x509007AC, 0x00026000))  # Post processing register
    regs.append((0x5090072C, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 7 quadrant 3
    regs.append((0x50D0002C, 0x00010009))  # Rows
    regs.append((0x50D000AC, 0x00010009))  # Columns
    regs.append((0x50D001AC, 0x00000001))  # Pooling rows
    regs.append((0x50D0022C, 0x00000001))  # Pooling columns
    regs.append((0x50D002AC, 0x00000001))  # Stride
    regs.append((0x50D0042C, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D0052C, 0x00001000))  # SRAM read ptr
    regs.append((0x50D005AC, 0x00000BA0))  # Layer control
    regs.append((0x50D00A2C, 0x0001F801))  # Layer control 2
    regs.append((0x50D0062C, 0x0A200E18))  # Mask offset and count
    regs.append((0x50D006AC, 0x00000003))  # TRAM ptr max
    regs.append((0x50D007AC, 0x00026000))  # Post processing register
    regs.append((0x50D0072C, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 8 quadrant 0
    regs.append((0x50100030, 0x00010005))  # Rows
    regs.append((0x501000B0, 0x00010005))  # Columns
    regs.append((0x50100330, 0x00001000))  # SRAM write ptr
    regs.append((0x50100430, 0x00002000))  # Write ptr mask offs
    regs.append((0x501004B0, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x501005B0, 0x0000EB20))  # Layer control
    regs.append((0x50100A30, 0x0001F810))  # Layer control 2
    regs.append((0x50100630, 0x0E201218))  # Mask offset and count
    regs.append((0x501006B0, 0x00000003))  # TRAM ptr max
    regs.append((0x501007B0, 0x00024000))  # Post processing register
    regs.append((0x50100730, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 8 quadrant 1
    regs.append((0x50500030, 0x00010005))  # Rows
    regs.append((0x505000B0, 0x00010005))  # Columns
    regs.append((0x50500330, 0x00001000))  # SRAM write ptr
    regs.append((0x50500430, 0x00002000))  # Write ptr mask offs
    regs.append((0x505004B0, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x505005B0, 0x00000B20))  # Layer control
    regs.append((0x50500A30, 0x0001F810))  # Layer control 2
    regs.append((0x50500630, 0x0E201218))  # Mask offset and count
    regs.append((0x505006B0, 0x00000003))  # TRAM ptr max
    regs.append((0x505007B0, 0x00024000))  # Post processing register
    regs.append((0x50500730, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 8 quadrant 2
    regs.append((0x50900030, 0x00010005))  # Rows
    regs.append((0x509000B0, 0x00010005))  # Columns
    regs.append((0x50900330, 0x00001000))  # SRAM write ptr
    regs.append((0x50900430, 0x00002000))  # Write ptr mask offs
    regs.append((0x509004B0, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x509005B0, 0x00000B20))  # Layer control
    regs.append((0x50900A30, 0x0001F810))  # Layer control 2
    regs.append((0x50900630, 0x0E201218))  # Mask offset and count
    regs.append((0x509006B0, 0x00000003))  # TRAM ptr max
    regs.append((0x509007B0, 0x00024000))  # Post processing register
    regs.append((0x50900730, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 8 quadrant 3
    regs.append((0x50D00030, 0x00010005))  # Rows
    regs.append((0x50D000B0, 0x00010005))  # Columns
    regs.append((0x50D00330, 0x00001000))  # SRAM write ptr
    regs.append((0x50D00430, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D004B0, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50D005B0, 0x00000B20))  # Layer control
    regs.append((0x50D00A30, 0x0001F810))  # Layer control 2
    regs.append((0x50D00630, 0x0E201218))  # Mask offset and count
    regs.append((0x50D006B0, 0x00000003))  # TRAM ptr max
    regs.append((0x50D007B0, 0x00025000))  # Post processing register
    regs.append((0x50D00730, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 9 quadrant 0
    regs.append((0x50100034, 0x00000003))  # Rows
    regs.append((0x501000B4, 0x00000003))  # Columns
    regs.append((0x501001B4, 0x00000001))  # Pooling rows
    regs.append((0x50100234, 0x00000001))  # Pooling columns
    regs.append((0x501002B4, 0x00000001))  # Stride
    regs.append((0x501003B4, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50100434, 0x00002000))  # Write ptr mask offs
    regs.append((0x501004B4, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50100534, 0x00001000))  # SRAM read ptr
    regs.append((0x501005B4, 0x0000EBA0))  # Layer control
    regs.append((0x50100A34, 0x0001F811))  # Layer control 2
    regs.append((0x50100634, 0xA320AB18))  # Mask offset and count
    regs.append((0x50100134, 0x00000100))  # 1D
    regs.append((0x501007B4, 0x0002300A))  # Post processing register
    regs.append((0x50100734, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 9 quadrant 1
    regs.append((0x50500034, 0x00000003))  # Rows
    regs.append((0x505000B4, 0x00000003))  # Columns
    regs.append((0x505001B4, 0x00000001))  # Pooling rows
    regs.append((0x50500234, 0x00000001))  # Pooling columns
    regs.append((0x505002B4, 0x00000001))  # Stride
    regs.append((0x505003B4, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50500434, 0x00002000))  # Write ptr mask offs
    regs.append((0x505004B4, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50500534, 0x00001000))  # SRAM read ptr
    regs.append((0x505005B4, 0x00000BA0))  # Layer control
    regs.append((0x50500A34, 0x0001F811))  # Layer control 2
    regs.append((0x50500634, 0xA320AB18))  # Mask offset and count
    regs.append((0x50500134, 0x00000100))  # 1D
    regs.append((0x505007B4, 0x00022000))  # Post processing register
    regs.append((0x50500734, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 9 quadrant 2
    regs.append((0x50900034, 0x00000003))  # Rows
    regs.append((0x509000B4, 0x00000003))  # Columns
    regs.append((0x509001B4, 0x00000001))  # Pooling rows
    regs.append((0x50900234, 0x00000001))  # Pooling columns
    regs.append((0x509002B4, 0x00000001))  # Stride
    regs.append((0x509003B4, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50900434, 0x00002000))  # Write ptr mask offs
    regs.append((0x509004B4, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50900534, 0x00001000))  # SRAM read ptr
    regs.append((0x509005B4, 0x00000BA0))  # Layer control
    regs.append((0x50900A34, 0x0001F811))  # Layer control 2
    regs.append((0x50900634, 0xA320AB18))  # Mask offset and count
    regs.append((0x50900134, 0x00000100))  # 1D
    regs.append((0x509007B4, 0x00022000))  # Post processing register
    regs.append((0x50900734, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 9 quadrant 3
    regs.append((0x50D00034, 0x00000003))  # Rows
    regs.append((0x50D000B4, 0x00000003))  # Columns
    regs.append((0x50D001B4, 0x00000001))  # Pooling rows
    regs.append((0x50D00234, 0x00000001))  # Pooling columns
    regs.append((0x50D002B4, 0x00000001))  # Stride
    regs.append((0x50D003B4, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50D00434, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D004B4, 0x00000001))  # Write ptr multi-pass channel offs
    regs.append((0x50D00534, 0x00001000))  # SRAM read ptr
    regs.append((0x50D005B4, 0x00000BA0))  # Layer control
    regs.append((0x50D00A34, 0x0001F811))  # Layer control 2
    regs.append((0x50D00634, 0xA320AB18))  # Mask offset and count
    regs.append((0x50D00134, 0x00000100))  # 1D
    regs.append((0x50D007B4, 0x00022000))  # Post processing register
    regs.append((0x50D00734, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 10 quadrant 0
    regs.append((0x50100338, 0x00001000))  # SRAM write ptr
    regs.append((0x501003B8, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50100438, 0x00002000))  # Write ptr mask offs
    regs.append((0x501005B8, 0x0001E920))  # Layer control
    regs.append((0x50100A38, 0x00004807))  # Layer control 2
    regs.append((0x50100638, 0xAC20AE98))  # Mask offset and count
    regs.append((0x50100138, 0x00000100))  # 1D
    regs.append((0x501007B8, 0x00003000))  # Post processing register
    regs.append((0x50100738, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 10 quadrant 1
    regs.append((0x50500338, 0x00001000))  # SRAM write ptr
    regs.append((0x505003B8, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50500438, 0x00002000))  # Write ptr mask offs
    regs.append((0x505005B8, 0x00010920))  # Layer control
    regs.append((0x50500A38, 0x00004807))  # Layer control 2
    regs.append((0x50500638, 0xAC20AE98))  # Mask offset and count
    regs.append((0x50500138, 0x00000100))  # 1D
    regs.append((0x505007B8, 0x00002000))  # Post processing register
    regs.append((0x50500738, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 10 quadrant 2
    regs.append((0x50900338, 0x00001000))  # SRAM write ptr
    regs.append((0x509003B8, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50900438, 0x00002000))  # Write ptr mask offs
    regs.append((0x509005B8, 0x00010920))  # Layer control
    regs.append((0x50900A38, 0x00004807))  # Layer control 2
    regs.append((0x50900638, 0xAC20AE98))  # Mask offset and count
    regs.append((0x50900138, 0x00000100))  # 1D
    regs.append((0x509007B8, 0x00002000))  # Post processing register
    regs.append((0x50900738, 0xFFFFFFFF))  # Mask and processor enables

    # Layer 10 quadrant 3
    regs.append((0x50D00338, 0x00001000))  # SRAM write ptr
    regs.append((0x50D003B8, 0x00000001))  # Write ptr time slot offs
    regs.append((0x50D00438, 0x00002000))  # Write ptr mask offs
    regs.append((0x50D005B8, 0x00010920))  # Layer control
    regs.append((0x50D00A38, 0x00004807))  # Layer control 2
    regs.append((0x50D00638, 0xAC20AE98))  # Mask offset and count
    regs.append((0x50D00138, 0x00000100))  # 1D
    regs.append((0x50D007B8, 0x00002000))  # Post processing register
    regs.append((0x50D00738, 0xFFFFFFFF))  # Mask and processor enables

    # regs = [main_pb2.ActionEnum.RUN_CNN_CONFIGURE]

    return regs
"""


if __name__ == "__main__":
    main()
