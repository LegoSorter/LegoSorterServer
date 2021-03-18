import grpc

from concurrent import futures
from generated import LegoBrick_pb2_grpc
from lego_sorter_server.generated import LegoSorter_pb2_grpc
from service.LegoBrickService import LegoBrickService
from service.LegoSorterService import LegoSorterService

class Server:

    @staticmethod
    def run():
        options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=16), options=options)
        LegoBrick_pb2_grpc.add_LegoBrickServicer_to_server(LegoBrickService(), server)
        LegoSorter_pb2_grpc.add_LegoSorterServicer_to_server(LegoSorterService(), server)
        server.add_insecure_port('[::]:50051')
        server.start()
        server.wait_for_termination()
