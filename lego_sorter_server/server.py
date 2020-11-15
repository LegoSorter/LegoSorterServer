from concurrent import futures

import grpc

from lego_sorter_server.generated import LegoBrick_pb2_grpc
from lego_sorter_server.LegoBrickServer import LegoBrickServer


class Server:

    @staticmethod
    def run():
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        LegoBrick_pb2_grpc.add_LegoBrickServicer_to_server(LegoBrickServer(), server)
        server.add_insecure_port('[::]:50051')
        server.start()
        server.wait_for_termination()
