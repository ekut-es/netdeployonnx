import asyncio
import logging
import os
import struct
import traceback
from contextlib import contextmanager
from typing import Dict, List

from max78000_cnn import *
from protobuffers import main_pb2
from tqdm import tqdm


class hooked_obj:
    def __init__(self, future):
        self.future = future

    async def wait_call(self, timeout=0):
        await asyncio.wait_for(self.future, timeout=timeout)


@contextmanager
def hook(obj, attr, func, condition=None):
    orig = getattr(obj, attr)
    try:
        future = asyncio.Future()
        future.calls = 0

        def intermed_call(*args, **kwargs):
            try:
                future.calls += 1
                return func(*args, **kwargs)
            finally:
                if condition is None or condition(future.calls, 0):
                    if future._result == None:
                        future.set_result(True)

        setattr(obj, attr, intermed_call)
        yield hooked_obj(future)
    except Exception:
        pass

        # traceback.print_exc()
    finally:
        # setattr(obj, attr, orig)
        ...


def integer_to_bytes(value: List[int]) -> bytes:
    return b"".join(struct.pack(">I", v) for v in value)


class Commands:
    def __init__(self):
        self.dataHandler = None
        self.set_exit_request = False

    def _register(self, dataHandler: "DataHandler"):
        self.dataHandler = dataHandler

    def get_commands(self) -> dict:
        return {
            "help": self.help,
            "q": self.exit_this,
            "exit": self.exit_this,
            "setled": self.set_led,
            "clrled": self.clr_led,
            "set_mem": self.set_mem,
            "read_mem": self.read_mem,
            "mem_set": self.set_mem,
            "mem_read": self.read_mem,
            "setspeed": self.set_speed,
            "load": self.load_file,
            "testrw": self.test_set_read_memory_content,
            "reset": self.reset,
            "debug": self.debug,
            "measure": self.measure,
            "measurement_ipo": self.measurement_ipo,
            "asserter": self.asserter,
            "tsm": self.test_small_matrix,
            "c10": self.cifar10,
            "enable": self.cnn_enable,
            "start": self.cnn_start,
            "stop": self.cnn_disable,
        }

    def exit_request(self):
        self.set_exit_request = True

    def new_message(self, version=2):
        msg = main_pb2.ProtocolMessage()
        msg.version = version
        return msg

    async def send(self, msg: "upb2msg") -> int:
        """
        returns errorcode, 0 if success
        """
        if self.dataHandler:
            if msg:
                return await self.dataHandler.send_msg(msg)
            else:
                return 0
        else:
            return 1024  # no data handler

    async def help(self, *args, **kwargs):
        print("commands:")
        for command in self.get_commands():
            print("\t", command)

    async def exit_this(self, *args, **kwargs):
        raise KeyboardInterrupt()

    def convert_instr_to_messages(
        self, *args, **kwargs
    ) -> List["main_pb2.ProtocolMessage"]:
        entries = convert_instr_to_messages_dict(*args, **kwargs).items()
        # return sorted by index
        return [msg for header, msg in sorted(entries, key=lambda x: x[0])]

    def convert_instr_to_messages_dict(
        self, instr: List[[int, int or bytes] or str]
    ) -> Dict[str, "main_pb2.ProtocolMessage"]:
        msgs = {}
        msg = self.new_message()
        title = ""
        for inst in instr:
            if isinstance(inst, tuple):
                if len(inst) == 2:
                    addr, data = inst
                    setAddr = False
                else:
                    addr, data, setAddr = inst
                if isinstance(data, int):
                    # reg access
                    msg.payload.registers.append(
                        main_pb2.SetRegister(
                            address=addr,
                            preserve_mask=0,
                            set_mask=data,
                            # size=main_pb2.Regsize.UINT32, # unknown is like 32bit, save us 4 bits
                            readable=False,
                        )
                    )
                elif isinstance(data, bytes):
                    # mem access
                    msg.payload.memory.append(
                        main_pb2.SetMemoryContent(
                            address=addr,
                            data=data,
                            setAddr=setAddr,
                        )
                    )
            elif isinstance(inst, int):
                # print(inst)
                if msg:
                    msgs[f"{len(msgs)} {title}"] = msg
                msg = self.new_message()
                msg.action.execute_measurement = (
                    inst  # main_pb2.ActionEnum.RUN_CNN_ENABLE
                )
                msgs[f"{len(msgs)} {title}"] = msg
                title = ""
                msg = self.new_message()
            elif isinstance(inst, str):
                # message to send
                if msg:
                    msgs[f"{len(msgs)} {title}"] = msg
                # create new message
                title = inst
                msg = self.new_message()
        return dict(
            filter(
                lambda titlemsgtuple: titlemsgtuple[1].WhichOneof("message_type")
                != None,
                msgs.items(),
            )
        )  # TODO: filter empty messages

    def split_message(
        self, msg: main_pb2.ProtocolMessage
    ) -> List[main_pb2.ProtocolMessage]:
        msg_len = len(msg.SerializeToString())
        if msg_len > 900:
            submessages = []

            datasize = 800

            # chunk registers into parts of 200
            chunksize = datasize // 12
            for chunk_id in range(int(len(msg.payload.registers) / chunksize) + 1):
                submsg = self.new_message()
                for i in range(chunksize):
                    ind = i + chunksize * chunk_id
                    if ind < len(msg.payload.registers):
                        setreg = msg.payload.registers[ind]
                        submsg.payload.registers.append(setreg)
                submessages.append(submsg)

            # go over each memory
            for setmemory in msg.payload.memory:
                # print(f"0x{setmemory.address:08X} {len(setmemory.data)}")
                chunksize = datasize
                if len(setmemory.data) > chunksize:  # split
                    for memory_subchunk in range(
                        int(len(setmemory.data) / chunksize) + 1
                    ):
                        newsetmemory = main_pb2.SetMemoryContent(
                            address=setmemory.address + memory_subchunk * chunksize,
                            data=setmemory.data[
                                memory_subchunk * chunksize : (memory_subchunk + 1)
                                * chunksize
                            ],
                            setAddr=setmemory.setAddr
                            if memory_subchunk == 0
                            else False,
                        )
                        # print(len(newsetmemory.data))
                        submsg = self.new_message()
                        submsg.payload.memory.append(newsetmemory)
                        submessages.append(submsg)
                else:
                    submsg = self.new_message()
                    submsg.payload.memory.append(setmemory)
                    submessages.append(submsg)
            for submessage in submessages:
                if len(submessage.SerializeToString()) > 1000:
                    raise Exception("msg too big")
            return list(
                filter(lambda x: x.WhichOneof("message_type") != None, submessages)
            )
        else:
            return list(filter(lambda x: x.WhichOneof("message_type") != None, [msg]))

    async def set_led(self, *args, **kwargs):
        msg = self.new_message()
        leds = []
        if len(args) > 0:
            for arg in args:
                leds.append(int(arg))
        GPIO2 = 0x40080400  # MaximSDK/Libraries/CMSIS/Device/Maxim/MAX78000/Include/max78000.h
        SET = 0x1C  # MaximSDK/Libraries/PeriphDrivers/Source/GPIO/gpio_reva_regs.h
        CLEAR = 0x20  # MaximSDK/Libraries/PeriphDrivers/Source/GPIO/gpio_reva_regs.h
        pin = 0
        for pin in leds:
            if pin > 3:
                continue
            pinaddr = 1 << pin
            msg.payload.registers.append(
                main_pb2.SetRegister(
                    address=GPIO2 + CLEAR,
                    preserve_mask=(0xFFFFFFFF & ~pinaddr),
                    set_mask=pinaddr,
                )
            )
        await self.send(msg)
        return msg

    async def clr_led(self, *args, **kwargs):
        msg = self.new_message()
        leds = []
        if len(args) > 0:
            for arg in args:
                leds.append(int(arg))
        GPIO2 = 0x40080400  # MaximSDK/Libraries/CMSIS/Device/Maxim/MAX78000/Include/max78000.h
        SET = 0x1C  # MaximSDK/Libraries/PeriphDrivers/Source/GPIO/gpio_reva_regs.h
        CLEAR = 0x20  # MaximSDK/Libraries/PeriphDrivers/Source/GPIO/gpio_reva_regs.h
        pin = 0
        for pin in leds:
            if pin > 3:
                continue
            pinaddr = 1 << pin
            msg.payload.registers.append(
                main_pb2.SetRegister(
                    address=GPIO2 + SET,
                    preserve_mask=(0xFFFFFFFF & ~pinaddr),
                    set_mask=pinaddr,
                )
            )
        await self.send(msg)
        return msg

    async def set_mem(self, *args, **kwargs):
        memaddr, memdata = 0, b""
        if len(args) == 2:
            memaddr = int(args[0], 0)
            memdata = bytes(args[1], "utf8")
        else:
            raise ValueError("provide atleast 2 args")

        msg = self.new_message()
        if len(memdata) > 0:
            msg.payload.memory.append(
                main_pb2.SetMemoryContent(address=memaddr, data=memdata)
            )
            await self.send(msg)
            return msg

    async def read_mem(self, *args, **kwargs):
        memaddr, memlen = 0, 0
        if len(args) == 2:
            memaddr = int(args[0], 0)
            memlen = int(args[1], 0)
        elif len(args) == 1:
            memaddr = int(args[0], 0)
            memlen = 1
        else:
            raise ValueError("provide atleast 2 args")

        msg = self.new_message()
        msg.payload.read.append(main_pb2.ReadMemoryContent(address=memaddr, len=memlen))
        await self.send(msg)
        payloadMsg: main_pb2.ProtocolMessage = await self.dataHandler.receive(
            lambda msg: msg.WhichOneof("message_type") == "payload",
            timeout=1,
        )
        if payloadMsg:
            data = payloadMsg.payload.memory[0].data
            csize = 16
            for chunk in range((len(data) // csize) + 1):
                dchunk = data[chunk * csize : (chunk + 1) * csize]
                print(" ".join([f"{b:02X}" for b in dchunk]))
        else:
            print("no message")
        return msg

    async def set_speed(self, *args, **kwargs):
        msg = self.new_message()
        if len(args) > 0:
            speed = int(args[0])
            msg.configuration.tickspeed = speed
            await self.send(msg)
            # print(" ".join([f"{b:02X}" for b in msg.SerializeToString()]))
            return msg

    async def load_file(self, *args, **kwargs):
        print(f"load {args}")
        if len(args) == 0:
            await self.load_file("1")
            await self.load_file("2")
            await self.load_file("3")
            await self.load_file("4")
            return
        if len(args) > 0:
            filename = args[0]
            # print("yes")
            if not os.path.exists(filename):
                print("did not find file, but trying in syshelpers")
                filename = os.path.join(
                    "/home/vscode/_Masterarbeit_SS24/hannah-env/syshelpers", filename
                )
            if not os.path.exists(filename):
                await self.load_file(f"payload_stage{args[0]}.pbenc")
                return
            try:
                with open(filename, "rb") as filecontent:
                    data = filecontent.read()
                # print("fabulous")
                msg = main_pb2.ProtocolMessage.FromString(data)
                print("loaded successfully")
                messages = self.split_message(msg)
                for submsg in messages:
                    await self.send(submsg)
                return msg
            except:
                traceback.print_exc()

    async def test_set_read_memory_content(self, *args, **kwargs):
        timeout = 2

        async def acquire_memory_address() -> int:
            msg = self.new_message()
            msg.configuration.address_test_message_buffer = 0
            await self.send(msg)
            configMsg: main_pb2.ProtocolMessage = await self.dataHandler.receive(
                lambda msg: msg.WhichOneof("message_type") == "configuration",
                timeout=timeout,
            )
            if configMsg is not None:
                addr = configMsg.configuration.address_test_message_buffer
                assert addr >= 0x20000000 and addr < 0x30000000
                return addr
            raise Exception("did not return a configuration in time")

        def filter_setmemorycontent_message(msg: "main_pb2.ProtocolMessage") -> bool:
            return 0

        addr = await acquire_memory_address()
        print(f"found addr: {addr:08X}")
        if len(args) == 0:
            amount = 2
        else:
            amount = int(args[0])
        if 1:
            data = [
                bytes(f"hello {test_number} world", "utf8")
                for test_number in range(amount)
            ]
            received_data = []
            for test_number, test_data in enumerate(data):
                msg = self.new_message()
                msg.payload.memory.append(
                    main_pb2.SetMemoryContent(address=addr, data=test_data)
                )
                await self.send(msg)
                msg = self.new_message()
                msg.payload.read.append(
                    main_pb2.ReadMemoryContent(address=addr, len=len(test_data))
                )
                await self.send(msg)
                # now await the message
                msg: main_pb2.ProtocolMessage = await self.dataHandler.receive(
                    lambda msg: msg.WhichOneof("message_type") == "payload"
                    and len(msg.payload.memory) > 0,
                    timeout=timeout,
                )
                if msg:
                    memoryMessage = msg.payload.memory[0]
                    if memoryMessage.address == addr:
                        received_data.append(memoryMessage.data)

            # now validate
            assert (
                len(data) == len(received_data)
            ), f"did not receive the same amount of data ({len(data)} vs {len(received_data)})"
            try:
                for test_number, test_data in enumerate(data):
                    logging.debug(f"[real] {test_data}")
                    logging.debug(f"[recv] {received_data[test_number]}")
                    assert test_data == received_data[test_number]
            except Exception as ex:
                print(ex)
                return
            print("transmitted ok!", len(data))

    async def reset(self, *args, **kwargs):
        if 1:
            msg = self.new_message()
            msg.configuration.execute_reset = True
            await self.send(msg)

    async def measure(self, *args, **kwargs):
        # in mode c(measuring the energy per Round), we get:
        # [diff_power,time,idle_power,active_power]*3
        # commands:
        # 'v': voltage
        # 'i': current
        # 'w': average power
        # 'c'/'t': cnn power mode
        # 's':sys power mode
        # ctrl+v = version
        if len(args) == 0:
            return await self.measurement_ipo(*args, **kwargs)

    async def measurement_ipo(self, *args, **kwargs):
        if 1:
            msg = self.new_message()
            msg.action.execute_measurement = main_pb2.ActionEnum.MEASUREMENT_WITH_IPO
            await self.send(msg)
            return msg

    async def asserter(self, *args, **kwargs):
        assertion_types = {
            "weights": main_pb2.ActionEnum.ASSERT_WEIGHTS,
            "input": main_pb2.ActionEnum.ASSERT_INPUT,
            "output": main_pb2.ActionEnum.ASSERT_OUTPUT,
        }
        if len(args) == 1 and args[0] in assertion_types:
            msg = self.new_message()
            msg.action.execute_measurement = assertion_types[args[0]]
            await self.send(msg)
            return msg
        else:
            print("no")

    async def test_small_matrix(self, *args, **kwargs):
        if 1:
            # we write a new set of instructions
            def cnn_init(registers, data):
                # we only use one quadrant
                # set the always on to only use 1 processor
                registers += [("CNN_AOD_CTRL", 0x0000F000)]
                registers += [("CNN_AOD_CTRL", 0x00000000)]
                for n in range(0, 4):
                    registers += [(f"CNNx16_{n}_CTRL", 0x00100008 if n == 0 else 0)]
                    registers += [(f"CNNx16_{n}_SRAM", 0x0000040E if n == 0 else 0)]
                    registers += [(f"CNNx16_{n}_LCNT_MAX", 0x00000001 if n == 0 else 0)]

            def cnn_load_weights(registers, data):
                for n in range(0, 1):
                    data[f"CNNx16_{n}_L0_MRAM"] = [0x4444_7777 for i in range(4)]

            def cnn_load_bias(registers, data):
                # load bias to bias ram addr
                for n in range(0, 1):
                    data[f"CNNx16_{n}_BIAS"] = [0xABAA_AAAA for i in range(4)]

            def cnn_configure(registers, data):
                for n in range(0, 4):
                    for y in [0]:
                        registers += [(f"CNNx16_{n}_L{y}_RCNT", 8)]  # two rows
                        registers += [(f"CNNx16_{n}_L{y}_CCNT", 8)]  # two columns

                        registers += [
                            (f"CNNx16_{n}_L{y}_WPTR_BASE", 0x040)
                        ]  # write ptr in SRAM
                        registers += [
                            (f"CNNx16_{n}_L{y}_WPTR_MOFF", 0x800)
                        ]  # mask offset in SRAM
                        registers += [
                            (f"CNNx16_{n}_L{y}_ONED", (15 << 18) | (1 << 17))
                        ]  # One dim control
                        registers += [
                            (f"CNNx16_{n}_L{y}_LCTRL0", 0x2820)  # orig 0x2820
                        ]  # tow columns
                        registers += [
                            (f"CNNx16_{n}_L{y}_MCNT", 0x1F8)
                        ]  # mask coffset and count
                        registers += [
                            (f"CNNx16_{n}_L{y}_TPTR", 0x1F)
                        ]  # set tram max size
                        registers += [
                            (f"CNNx16_{n}_L{y}_EN", 0x0F)
                        ]  # set tram max size
                        registers += [
                            # bias enable
                            (f"CNNx16_{n}_L{y}_POST", (1 << 12))
                        ]
                        registers += [
                            (f"CNNx16_{n}_L{y}_LCTRL1", 0)
                        ]  # zero processors in exp mode

            def load_input(registers, data):
                data[f"CNNx16_{0}_SRAM"] = [
                    0x01020304,
                    0x05060708,
                    0x090A0B0C,
                    0x0D0E0F00,
                ] * 2

            def cnn_start(registers, data):
                for n in range(0, 4):
                    # registers += [(f'CNNx16_{n}_CTRL', 0x00100808 if n == 0 else 0x00100809)]
                    registers += [(f"CNNx16_{n}_CTRL", 0x00100808 if n == 0 else 0)]
                registers += [(f"CNNx16_{0}_CTRL_", 0x00100009)]

            logging.debug("funcdefs")

            # gen data
            regs, data = [], {}
            cnn_init(regs, data)
            cnn_load_weights(regs, data)
            cnn_load_bias(regs, data)
            cnn_configure(regs, data)
            load_input(regs, data)
            cnn_start(regs, data)
            logging.debug("base regs + data done")

            msg = self.new_message()

            for reg_name, value in regs:
                try:
                    addr = transform_regname_to_address(reg_name)
                except ValueError:
                    if reg_name == "GCFR_REG1":
                        addr = 0x4000_5800 + 0x04
                msg.payload.registers.append(
                    main_pb2.SetRegister(
                        address=addr,
                        preserve_mask=0,
                        set_mask=value,
                        # size=main_pb2.Regsize.UINT32, # unknown is like 32bit, save us 4 bits
                        readable=False,
                    )
                )
            for data_addr, value in data.items():
                addr = transform_memname_to_address(data_addr)
                try:
                    tvalue = (
                        integer_to_bytes(value)
                        if not isinstance(value, bytes)
                        else value
                    )
                except:
                    traceback.print_exc()
                msg.payload.memory.append(
                    main_pb2.SetMemoryContent(
                        address=addr,
                        data=tvalue,
                    )
                )
            # CNNx16_0_L0_MRAM CNNx16_0_BIAS CNNx16_0_SRAM
            for mname in ["CNNx16_0_SRAM"]:
                addr = transform_memname_to_address(mname)  # is that the output?
                msg.payload.read.append(
                    main_pb2.ReadMemoryContent(
                        address=addr,
                        len=128,
                    )
                )
            messages = self.split_message(msg)
            for submsg in messages:
                await self.send(submsg)
            print("sent!")
            for mname in ["CNNx16_0_SRAM"]:
                print(mname)
                payloadMsg: main_pb2.ProtocolMessage = await self.dataHandler.receive(
                    lambda msg: msg.WhichOneof("message_type") == "payload",
                    timeout=2,
                )
                if payloadMsg:
                    data = payloadMsg.payload.memory[0].data
                    csize = 16
                    for chunk in range((len(data) // csize) + 1):
                        dchunk = data[chunk * csize : (chunk + 1) * csize]
                        print(" ".join([f"{b:02X}" for b in dchunk]))
                else:
                    print("no message")
            return msg

    async def cifar10(self, *args, **kwargs):
        msgs = []
        try:
            import cifar10

            # data = cifar10.load_input()[0][1]
            # csize = 16
            # for chunk in range((len(data) // csize) + 1):
            #     dchunk = data[chunk * csize : (chunk + 1) * csize]
            #     print(" ".join([f"{b:02X}" for b in dchunk]))
            # return
            instr = []
            instr.extend(["cnn_enable()"])
            instr.extend(cifar10.cnn_enable())
            instr.extend(["cnn_init()"])
            instr.extend(cifar10.cnn_init())
            instr.extend(["cnn_load_weights()"])
            instr.extend(cifar10.cnn_load_weights())
            instr.extend(["cnn_load_bias()"])
            instr.extend(cifar10.cnn_load_bias())
            instr.extend(["cnn_configure()"])
            instr.extend(cifar10.cnn_configure())
            instr.extend(["load_input()"])
            instr.extend(cifar10.load_input())
            instr.extend(["cnn_start()"])
            instr.extend(cifar10.cnn_start())
            instr.extend(["done"])
            logging.debug("base regs + data done")

            msgs = self.convert_instr_to_messages_dict(instr)
            totalData = 0
            for msg_title, msg in msgs.items():  # tqdm(msgs):
                # print("title:", msg_title)
                mems, regs, acts = 0, 0, 0
                submessages = self.split_message(msg)
                for submessage in tqdm(submessages):
                    # print("    " * 5, msg_title, submessage.WhichOneof("message_type"))
                    mems += len(submessage.payload.memory)
                    regs += len(submessage.payload.registers)
                    acts += 1 if submessage.action.execute_measurement else 0
                    # send
                    # await asyncio.sleep(0.001)
                    error: int = 1  # init with error != 0
                    while error > 0:
                        # repeat if error code is not 0
                        error = await self.send(submessage)
                        if error == 0:
                            break
                        print("resending...", error)
                # await asyncio.sleep(0.1)
                print(f"wrote {mems} mems, {regs} regs, {acts} actions")
            # return msgs
        except:
            traceback.print_exc()

        print("reading back")
        for i in tqdm(range(15)):
            await asyncio.sleep(0.1)
        print("waited a little bit")
        target = [
            [
                0xFFFC3376,
                0xFFFC6F4D,
                0xFFFD405D,
                0x00030A96,
            ],
            [
                0xFFFCFB74,
                0x00001191,
                0xFFFDCD38,
                0xFFFC95FE,
            ],
            [0xFFFC3078, 0xFFFC09C9],
        ]
        # read back
        try:
            for i, (addr, size) in enumerate(
                {
                    0x50404000: 4 * 4,
                    0x5040C000: 4 * 4,
                    0x50414000: 2 * 4,
                }.items()
            ):
                msg = self.new_message()
                msg.payload.read.append(
                    main_pb2.ReadMemoryContent(address=addr, len=size)
                )
                await self.send(msg)

                payloadMsg: main_pb2.ProtocolMessage = await self.dataHandler.receive(
                    lambda msg: msg.WhichOneof("message_type") == "payload",
                    timeout=2,
                )
                if payloadMsg:
                    data = payloadMsg.payload.memory[0].data
                    csize = 16
                    for chunk in range((len(data) // csize) + 1):
                        dchunk = data[chunk * csize : (chunk + 1) * csize]
                        if len(dchunk):
                            print("?", " ".join([f"{b:02X}" for b in dchunk]))
                else:
                    data = b""
                    print("no message")
                target_bytes = b"".join([struct.pack("<I", val) for val in target[i]])
                print(
                    "=" if target_bytes == data else "X",
                    " ".join([f"{b:02X}" for b in target_bytes]),
                )
        except:
            traceback.print_exc()
        finally:
            # await self.cnn_disable()
            ...

    async def cnn_enable(self, *args, **kwargs):
        msg = self.new_message()
        msg.action.execute_measurement = main_pb2.ActionEnum.RUN_CNN_ENABLE
        await self.send(msg)

    async def cnn_disable(self, *args, **kwargs):
        msg = self.new_message()
        msg.action.execute_measurement = main_pb2.ActionEnum.RUN_CNN_DISABLE
        await self.send(msg)

    async def cnn_start(self, *args, **kwargs):
        msg = self.new_message()
        msg.action.execute_measurement = main_pb2.ActionEnum.RUN_CNN_START
        await self.send(msg)

    async def debug(self, *args, **kwargs):
        write_len = 1020
        if len(args) > 0:
            write_len = int(args[0])
        if "debug" in kwargs:
            kwargs["recvdata"] = b""

            def debug_search(
                datastream: bytes, discard_limit: int = 100
            ) -> tuple[list, bytes]:
                if len(datastream) > 0:
                    print(".", end="")
                kwargs["recvdata"] += datastream
                return [], b""

            cond = lambda calls, time: calls > 1000
            with hook(
                self.dataHandler,
                "search_protobuf_messages",
                debug_search,
                condition=cond,
            ) as debug_hook:
                data = self.dataHandler.debug_data[:write_len]
                reader, writer = self.dataHandler.reader, self.dataHandler.writer
                ret1 = writer.write(data)
                ret2 = await writer.drain()
                print(f"{len(data)} bytes written")
                await debug_hook.wait_call(timeout=1)

            csize = 32
            for chunk in range((len(kwargs["recvdata"]) // csize) + 1):
                dchunk = kwargs["recvdata"][chunk * csize : (chunk + 1) * csize]
                try:
                    print(" ".join([f"{b:02X}" for b in dchunk]))
                    for i in range(min(csize, len(dchunk))):
                        assert (
                            dchunk[i] == data[i + chunk * csize]
                        ), f"{dchunk[i]} != {data[i+chunk*csize]} on {i+chunk*csize}"
                except:
                    import traceback

                    traceback.print_exc()
                    break
            print(f'{len(kwargs["recvdata"])} bytes read')
        else:
            logging.warning("please enable debug mode")