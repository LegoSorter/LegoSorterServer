from concurrent import futures

import grpc

from generated import LegoBrick_pb2_grpc
from LegoBrickService import LegoBrickService


class Server:

    @staticmethod
    def run():
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        LegoBrick_pb2_grpc.add_LegoBrickServicer_to_server(LegoBrickService(), server)
        server.add_insecure_port('[::]:50051')
        server.start()
        server.wait_for_termination()
