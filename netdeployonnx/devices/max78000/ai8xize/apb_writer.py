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
from izer import tornadocnn as tc  # noqa: E402
from izer.apbaccess import APBBlockLevel, APBTopLevel  # noqa: E402


class CustomAPBBlocklevel(APBBlockLevel):
    def write(
        self,
        addr,
        val,
        comment="",
        indent="  ",
        no_verify=False,
        fifo=None,
        base=None,
        fifo_wait=True,
    ):
        """
        Write address `addr` and data `val` to the .c file.
        if `no_verify` is `True`, do not check the result
        of the write operation, even if
        `verify_writes` is globally enabled.
        An optional `comment` can be added to the output.
        """
        ...


class CustomAPBTopLevel(APBTopLevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample_input = []
        self._output_defines = []
        self._lregs = []
        self._weights = []
        self._bias = []

    def output_define(
        self,
        array,
        define_name,
        fmt,
        columns,
        weights=True,
    ):
        if define_name.startswith("SAMPLE_INPUT"):
            self._sample_input.append((array, define_name, fmt, columns, weights))
        else:
            self._output_defines.append((array, define_name, fmt, columns, weights))

    def write_lreg(
        self,
        group,
        layer,
        reg,
        val,
        force_write=False,
        no_verify=False,
        comment="",
    ):
        reg = tc.lreg_addr(group, reg, layer=layer), tc.lreg_addr(0, reg, layer=0)
        self._lregs.append((group, layer, reg, val, force_write, no_verify, comment))

    def write_bias(
        self,
        group,
        offs,
        bias,
    ):
        self._bias.append((group, offs, bias))

    def write_tram(
        self,
        group,
        proc,
        offs,
        d,
        comment="",
    ):
        addr = (
            tc.dev.C_GROUP_OFFS * group
            + tc.dev.C_TRAM_BASE
            + proc * tc.dev.TRAM_OFFS * 4
            + offs * 4
        )
        raise NotImplementedError(f"write_tram is not implemented {addr}")
        # self.write(addr, d, f' // {comment}TRAM G{group} P{proc} #{offs}')

    # def write_kern(
    #         self,
    #         ll,
    #         p,
    #         idx,
    #         k,
    #         size=9,
    #         verify_only=False,
    #         calc_x4=False,
    #         kern_offs=None,
    #         count=None,
    # ):
    #     self._weights.append((ll, p, idx, k, size, verify_only, calc_x4, kern_offs, count)) # noqa E501


def get_custom_writer(refs):
    def custom_writer(*args, debug_mem=False, **kwargs):
        if not debug_mem:
            import izer.state as state

            APBClass = (  # noqa: N806
                CustomAPBBlocklevel
                if state.block_mode or debug_mem
                else CustomAPBTopLevel
            )
        else:
            raise NotImplementedError("debug_mem is not implemented")
        obj = APBClass(
            *args,
            **kwargs,
        )
        refs.append(obj)
        return obj

    return custom_writer
