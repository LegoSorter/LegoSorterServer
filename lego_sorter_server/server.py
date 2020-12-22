import grpc

from concurrent import futures
from generated import LegoBrick_pb2_grpc
from LegoBrickService import LegoBrickService
import tensorflow as tf

class Server:

    @staticmethod
    def run():

        options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=options)
        LegoBrick_pb2_grpc.add_LegoBrickServicer_to_server(LegoBrickService(), server)
        server.add_insecure_port('[::]:50051')
        server.start()
        server.wait_for_termination()
