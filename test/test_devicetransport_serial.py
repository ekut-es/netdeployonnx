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
import random
import string
import sys
import time

import serial


def generate_data(size):
    return "".join(
        random.choices(string.ascii_letters + string.digits, k=size)
    ).encode()


def main():
    port = "/dev/ttyACM0"
    baudrate = 1500000
    size = 2048
    timeout = 0.1
    sleepx = 0.2
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    if len(sys.argv) > 2:
        port = sys.argv[2]
    if len(sys.argv) > 3:
        baudrate = int(sys.argv[3])
    ser = serial.Serial(port, baudrate, timeout=timeout)

    while True:
        start_time = time.time()
        total_bytes = 0

        while time.time() - start_time < 0.1:
            csize = random.choice([int(size / n) for n in range(1, 2)])
            data = generate_data(csize)
            ser.write(data)
            ser.flush()
            echo = ser.read(csize)

            if data != echo:
                print("Error in echo. Restarting...")
                try:
                    moredata = ser.read(len(data) * 16)
                    total_data = len(echo) + len(moredata)
                except KeyboardInterrupt:
                    break
                if total_data != len(data):
                    print(f"caused by size {len(echo)} [{total_data}]")
                else:
                    print("caused by data mismatch")
                time.sleep(sleepx - timeout)
                break
            total_bytes += csize

        elapsed_time = time.time() - start_time
        speed = total_bytes / elapsed_time if elapsed_time > 0 else 0

        print(f"Current speed: {speed:.2f} bytes/sec")
        time.sleep(sleepx)


if __name__ == "__main__":
    main()
