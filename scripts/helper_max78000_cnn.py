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
