# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import netdeployonnx.common.device_pb2 as device__pb2

GRPC_GENERATED_VERSION = '1.64.1'
GRPC_VERSION = grpc.__version__
EXPECTED_ERROR_RELEASE = '1.65.0'
SCHEDULED_RELEASE_DATE = 'June 25, 2024'
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    warnings.warn(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in device_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
        + f' This warning will become an error in {EXPECTED_ERROR_RELEASE},'
        + f' scheduled for release on {SCHEDULED_RELEASE_DATE}.',
        RuntimeWarning
    )


class DeviceServiceStub(object):
    """

    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ListDevices = channel.unary_unary(
                '/device.DeviceService/ListDevices',
                request_serializer=device__pb2.ListDevicesRequest.SerializeToString,
                response_deserializer=device__pb2.ListDevicesResponse.FromString,
                _registered_method=True)
        self.GetDeviceHandle = channel.unary_unary(
                '/device.DeviceService/GetDeviceHandle',
                request_serializer=device__pb2.GetDeviceHandleRequest.SerializeToString,
                response_deserializer=device__pb2.GetDeviceHandleResponse.FromString,
                _registered_method=True)
        self.FreeDeviceHandle = channel.unary_unary(
                '/device.DeviceService/FreeDeviceHandle',
                request_serializer=device__pb2.FreeDeviceHandleRequest.SerializeToString,
                response_deserializer=device__pb2.FreeDeviceHandleResponse.FromString,
                _registered_method=True)
        self.GetDeviceInfo = channel.unary_unary(
                '/device.DeviceService/GetDeviceInfo',
                request_serializer=device__pb2.GetDeviceInfoRequest.SerializeToString,
                response_deserializer=device__pb2.GetDeviceInfoResponse.FromString,
                _registered_method=True)
        self.RunPayloadSynchronous = channel.unary_unary(
                '/device.DeviceService/RunPayloadSynchronous',
                request_serializer=device__pb2.RunPayloadRequest.SerializeToString,
                response_deserializer=device__pb2.RunPayloadResponse.FromString,
                _registered_method=True)
        self.RunPayloadAsynchronous = channel.unary_unary(
                '/device.DeviceService/RunPayloadAsynchronous',
                request_serializer=device__pb2.RunPayloadRequest.SerializeToString,
                response_deserializer=device__pb2.RunPayloadResponse.FromString,
                _registered_method=True)
        self.CheckPayloadAsynchronous = channel.unary_unary(
                '/device.DeviceService/CheckPayloadAsynchronous',
                request_serializer=device__pb2.CheckPayloadRequest.SerializeToString,
                response_deserializer=device__pb2.CheckPayloadResponse.FromString,
                _registered_method=True)


class DeviceServiceServicer(object):
    """

    """

    def ListDevices(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetDeviceHandle(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def FreeDeviceHandle(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetDeviceInfo(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RunPayloadSynchronous(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RunPayloadAsynchronous(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CheckPayloadAsynchronous(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DeviceServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ListDevices': grpc.unary_unary_rpc_method_handler(
                    servicer.ListDevices,
                    request_deserializer=device__pb2.ListDevicesRequest.FromString,
                    response_serializer=device__pb2.ListDevicesResponse.SerializeToString,
            ),
            'GetDeviceHandle': grpc.unary_unary_rpc_method_handler(
                    servicer.GetDeviceHandle,
                    request_deserializer=device__pb2.GetDeviceHandleRequest.FromString,
                    response_serializer=device__pb2.GetDeviceHandleResponse.SerializeToString,
            ),
            'FreeDeviceHandle': grpc.unary_unary_rpc_method_handler(
                    servicer.FreeDeviceHandle,
                    request_deserializer=device__pb2.FreeDeviceHandleRequest.FromString,
                    response_serializer=device__pb2.FreeDeviceHandleResponse.SerializeToString,
            ),
            'GetDeviceInfo': grpc.unary_unary_rpc_method_handler(
                    servicer.GetDeviceInfo,
                    request_deserializer=device__pb2.GetDeviceInfoRequest.FromString,
                    response_serializer=device__pb2.GetDeviceInfoResponse.SerializeToString,
            ),
            'RunPayloadSynchronous': grpc.unary_unary_rpc_method_handler(
                    servicer.RunPayloadSynchronous,
                    request_deserializer=device__pb2.RunPayloadRequest.FromString,
                    response_serializer=device__pb2.RunPayloadResponse.SerializeToString,
            ),
            'RunPayloadAsynchronous': grpc.unary_unary_rpc_method_handler(
                    servicer.RunPayloadAsynchronous,
                    request_deserializer=device__pb2.RunPayloadRequest.FromString,
                    response_serializer=device__pb2.RunPayloadResponse.SerializeToString,
            ),
            'CheckPayloadAsynchronous': grpc.unary_unary_rpc_method_handler(
                    servicer.CheckPayloadAsynchronous,
                    request_deserializer=device__pb2.CheckPayloadRequest.FromString,
                    response_serializer=device__pb2.CheckPayloadResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'device.DeviceService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('device.DeviceService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class DeviceService(object):
    """

    """

    @staticmethod
    def ListDevices(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/device.DeviceService/ListDevices',
            device__pb2.ListDevicesRequest.SerializeToString,
            device__pb2.ListDevicesResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetDeviceHandle(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/device.DeviceService/GetDeviceHandle',
            device__pb2.GetDeviceHandleRequest.SerializeToString,
            device__pb2.GetDeviceHandleResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def FreeDeviceHandle(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/device.DeviceService/FreeDeviceHandle',
            device__pb2.FreeDeviceHandleRequest.SerializeToString,
            device__pb2.FreeDeviceHandleResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetDeviceInfo(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/device.DeviceService/GetDeviceInfo',
            device__pb2.GetDeviceInfoRequest.SerializeToString,
            device__pb2.GetDeviceInfoResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def RunPayloadSynchronous(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/device.DeviceService/RunPayloadSynchronous',
            device__pb2.RunPayloadRequest.SerializeToString,
            device__pb2.RunPayloadResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def RunPayloadAsynchronous(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/device.DeviceService/RunPayloadAsynchronous',
            device__pb2.RunPayloadRequest.SerializeToString,
            device__pb2.RunPayloadResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def CheckPayloadAsynchronous(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/device.DeviceService/CheckPayloadAsynchronous',
            device__pb2.CheckPayloadRequest.SerializeToString,
            device__pb2.CheckPayloadResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
