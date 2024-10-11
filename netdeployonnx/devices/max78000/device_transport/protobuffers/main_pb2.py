# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protobuffers/main.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17protobuffers/main.proto\"k\n\tKeepalive\x12\r\n\x05ticks\x18\x01 \x01(\r\x12\x11\n\tnext_tick\x18\x02 \x01(\r\x12\x0f\n\x07warning\x18\x03 \x01(\r\x12\x14\n\x0cinqueue_size\x18\x04 \x01(\r\x12\x15\n\routqueue_size\x18\x05 \x01(\r\"^\n\rConfiguration\x12\x11\n\ttickspeed\x18\x02 \x01(\r\x12\x15\n\rexecute_reset\x18\x03 \x01(\x08\x12#\n\x1b\x61\x64\x64ress_test_message_buffer\x18\x04 \x01(\r\"1\n\x11ReadMemoryContent\x12\x0f\n\x07\x61\x64\x64ress\x18\x01 \x01(\r\x12\x0b\n\x03len\x18\x02 \x01(\r\"q\n\x10SetMemoryContent\x12\x0f\n\x07\x61\x64\x64ress\x18\x01 \x01(\r\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\x0c\x12\x16\n\x0eonly_do_assert\x18\x03 \x01(\x08\x12\x15\n\rsetZeroAmount\x18\x04 \x01(\r\x12\x0f\n\x07setAddr\x18\x05 \x01(\x08\"q\n\x0bSetRegister\x12\x0f\n\x07\x61\x64\x64ress\x18\x01 \x01(\r\x12\x15\n\rpreserve_mask\x18\x02 \x01(\r\x12\x10\n\x08set_mask\x18\x03 \x01(\r\x12\x16\n\x04size\x18\x04 \x01(\x0e\x32\x08.Regsize\x12\x10\n\x08readable\x18\x05 \x01(\x08\"j\n\x08SetFlash\x12\x16\n\x03var\x18\x01 \x01(\x0e\x32\t.Variable\x12\x16\n\x0e\x61\x64\x64ress_offset\x18\x02 \x01(\r\x12\x0b\n\x03\x63rc\x18\x03 \x01(\r\x12\x13\n\x0bstart_flash\x18\x04 \x01(\x08\x12\x0c\n\x04\x64\x61ta\x18\x05 \x01(\x0c\"\x89\x01\n\x07Payload\x12\x1f\n\tregisters\x18\x01 \x03(\x0b\x32\x0c.SetRegister\x12!\n\x06memory\x18\x02 \x03(\x0b\x32\x11.SetMemoryContent\x12 \n\x04read\x18\x03 \x03(\x0b\x32\x12.ReadMemoryContent\x12\x18\n\x05\x66lash\x18\x04 \x03(\x0b\x32\t.SetFlash\"2\n\x06\x41\x63tion\x12(\n\x13\x65xecute_measurement\x18\x01 \x01(\x0e\x32\x0b.ActionEnum\"\x05\n\x03\x41\x43K\"\xed\x01\n\x0fProtocolMessage\x12\x0f\n\x07version\x18\x01 \x01(\r\x12\x10\n\x08sequence\x18\x02 \x01(\r\x12\x13\n\x03\x61\x63k\x18\x03 \x01(\x0b\x32\x04.ACKH\x00\x12\x1f\n\tkeepalive\x18\x04 \x01(\x0b\x32\n.KeepaliveH\x00\x12\'\n\rconfiguration\x18\x05 \x01(\x0b\x32\x0e.ConfigurationH\x00\x12\x1b\n\x07payload\x18\x06 \x01(\x0b\x32\x08.PayloadH\x00\x12\x19\n\x06\x61\x63tion\x18\x07 \x01(\x0b\x32\x07.ActionH\x00\x12\x10\n\x08\x63hecksum\x18\n \x01(\x07\x42\x0e\n\x0cmessage_type*9\n\x07Regsize\x12\x0b\n\x07UNKNOWN\x10\x00\x12\t\n\x05UINT8\x10\x01\x12\n\n\x06UINT16\x10\x02\x12\n\n\x06UINT32\x10\x04*R\n\x08Variable\x12\n\n\x06\x42IAS_0\x10\x00\x12\n\n\x06\x42IAS_1\x10\x01\x12\n\n\x06\x42IAS_2\x10\x02\x12\n\n\x06\x42IAS_3\x10\x03\x12\x0b\n\x07WEIGHTS\x10\x04\x12\t\n\x05INPUT\x10\x05*\xb2\x03\n\nActionEnum\x12\x08\n\x04NONE\x10\x00\x12\x18\n\x14MEASUREMENT_WITH_IPO\x10\x01\x12\x12\n\x0e\x41SSERT_WEIGHTS\x10\n\x12\x10\n\x0c\x41SSERT_INPUT\x10\x0b\x12\x11\n\rASSERT_OUTPUT\x10\x0c\x12\x12\n\x0eRUN_CNN_ENABLE\x10\x14\x12\x10\n\x0cRUN_CNN_INIT\x10\x15\x12\x18\n\x14RUN_CNN_LOAD_WEIGHTS\x10\x16\x12\x15\n\x11RUN_CNN_LOAD_BIAS\x10\x17\x12\x15\n\x11RUN_CNN_CONFIGURE\x10\x18\x12\x16\n\x12RUN_CNN_LOAD_INPUT\x10\x19\x12\x11\n\rRUN_CNN_START\x10\x1a\x12\x12\n\x0eRUN_CNN_UNLOAD\x10\x1b\x12\x13\n\x0fRUN_CNN_DISABLE\x10\x1c\x12\x19\n\x15INIT_WEIGHTS_PATTERN1\x10(\x12\x19\n\x15INIT_WEIGHTS_PATTERN2\x10)\x12\x19\n\x15INIT_WEIGHTS_PATTERN3\x10*\x12\x19\n\x15INIT_WEIGHTS_PATTERN4\x10+\x12\x19\n\x15INIT_WEIGHTS_PATTERN5\x10,b\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'protobuffers.main_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _REGSIZE._serialized_start=1060
  _REGSIZE._serialized_end=1117
  _VARIABLE._serialized_start=1119
  _VARIABLE._serialized_end=1201
  _ACTIONENUM._serialized_start=1204
  _ACTIONENUM._serialized_end=1638
  _KEEPALIVE._serialized_start=27
  _KEEPALIVE._serialized_end=134
  _CONFIGURATION._serialized_start=136
  _CONFIGURATION._serialized_end=230
  _READMEMORYCONTENT._serialized_start=232
  _READMEMORYCONTENT._serialized_end=281
  _SETMEMORYCONTENT._serialized_start=283
  _SETMEMORYCONTENT._serialized_end=396
  _SETREGISTER._serialized_start=398
  _SETREGISTER._serialized_end=511
  _SETFLASH._serialized_start=513
  _SETFLASH._serialized_end=619
  _PAYLOAD._serialized_start=622
  _PAYLOAD._serialized_end=759
  _ACTION._serialized_start=761
  _ACTION._serialized_end=811
  _ACK._serialized_start=813
  _ACK._serialized_end=818
  _PROTOCOLMESSAGE._serialized_start=821
  _PROTOCOLMESSAGE._serialized_end=1058
# @@protoc_insertion_point(module_scope)
