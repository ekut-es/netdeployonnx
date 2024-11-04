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
import re

import netdeployonnx.devices.max78000.cnn_constants as cnn_constants


def main():
    """
    Reads input register and value and prints out the register names
    from the value it received.
    comma and brackets and quotes are ignored.
    it asks again and again like repl
    """
    exit_loop = False
    while not exit_loop:
        input_str = input(">>")
        if input_str == "exit":
            exit_loop = True
            break
        else:
            ignore_regex = r"[\"\'\[\]\(\) ]"
            # replace all
            input_str = re.sub(ignore_regex, "", input_str)
            # change CNNx16_3 to CNNx16_n
            input_str = input_str.split(",")

            orig_variable_name = input_str[0]
            variable_value = int(input_str[1], 0)
            variable_name = re.sub(r"CNNx16_(\d)", "CNNx16_n", orig_variable_name)
            varnames = []
            for variable in vars(cnn_constants):
                if variable.startswith(variable_name):
                    if variable.endswith("_POS"):
                        continue  # skip POS
                    # now we should only have _BITX and values
                    var_val = vars(cnn_constants)[variable]
                    # check if the value is in the variable_value
                    if (variable_value & var_val) == var_val:
                        varnames.append(variable)
            print(variable_name, variable_value)
            values = " |\n".join(varnames)
            print(f'("{variable_name}", ({values}))')


if __name__ == "__main__":
    main()
