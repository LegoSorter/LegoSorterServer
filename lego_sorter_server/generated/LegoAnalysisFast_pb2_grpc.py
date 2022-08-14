# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import lego_sorter_server.generated.LegoAnalysisFast_pb2 as LegoAnalysisFast__pb2
import lego_sorter_server.generated.Messages_pb2 as Messages__pb2


class LegoAnalysisFastStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.DetectBricks = channel.unary_unary(
                '/analysis.LegoAnalysisFast/DetectBricks',
                request_serializer=Messages__pb2.ImageRequest.SerializeToString,
                response_deserializer=Messages__pb2.ListOfBoundingBoxes.FromString,
                )
        self.DetectAndClassifyBricks = channel.unary_unary(
                '/analysis.LegoAnalysisFast/DetectAndClassifyBricks',
                request_serializer=LegoAnalysisFast__pb2.FastImageRequest.SerializeToString,
                response_deserializer=Messages__pb2.ListOfBoundingBoxes.FromString,
                )


class LegoAnalysisFastServicer(object):
    """Missing associated documentation comment in .proto file."""

    def DetectBricks(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DetectAndClassifyBricks(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_LegoAnalysisFastServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'DetectBricks': grpc.unary_unary_rpc_method_handler(
                    servicer.DetectBricks,
                    request_deserializer=Messages__pb2.ImageRequest.FromString,
                    response_serializer=Messages__pb2.ListOfBoundingBoxes.SerializeToString,
            ),
            'DetectAndClassifyBricks': grpc.unary_unary_rpc_method_handler(
                    servicer.DetectAndClassifyBricks,
                    request_deserializer=LegoAnalysisFast__pb2.FastImageRequest.FromString,
                    response_serializer=Messages__pb2.ListOfBoundingBoxes.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'analysis.LegoAnalysisFast', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class LegoAnalysisFast(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def DetectBricks(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/analysis.LegoAnalysisFast/DetectBricks',
            Messages__pb2.ImageRequest.SerializeToString,
            Messages__pb2.ListOfBoundingBoxes.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DetectAndClassifyBricks(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/analysis.LegoAnalysisFast/DetectAndClassifyBricks',
            LegoAnalysisFast__pb2.FastImageRequest.SerializeToString,
            Messages__pb2.ListOfBoundingBoxes.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
