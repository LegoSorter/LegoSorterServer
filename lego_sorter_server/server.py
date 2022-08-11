import grpc

from concurrent import futures

from signalrcore.hub_connection_builder import HubConnectionBuilder

from lego_sorter_server.generated import LegoSorter_pb2_grpc, LegoCapture_pb2_grpc, LegoAnalysis_pb2_grpc, \
    LegoAnalysisFast_pb2_grpc, LegoControl_pb2_grpc
from lego_sorter_server.service.LegoCaptureService import LegoCaptureService
from service.LegoAnalysisService import LegoAnalysisService
from service.LegoAnalysisFastService import LegoAnalysisFastService
from service.LegoSorterService import LegoSorterService
from service.LegoControlService import LegoControlService

from lego_sorter_server.service.BrickCategoryConfig import BrickCategoryConfig
from collections import deque

class Server:

    @staticmethod
    def run(sorterConfig: BrickCategoryConfig):
        options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        lastImages = deque([], maxlen=2)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=16), options=options)
        server2 = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=options)
        LegoSorter_pb2_grpc.add_LegoSorterServicer_to_server(LegoSorterService(sorterConfig), server)
        LegoCapture_pb2_grpc.add_LegoCaptureServicer_to_server(LegoCaptureService(), server)
        LegoAnalysis_pb2_grpc.add_LegoAnalysisServicer_to_server(LegoAnalysisService(), server)
        try:
            hub_connection: HubConnectionBuilder = HubConnectionBuilder() \
                .with_url("http://192.168.11.189:5002/hubs/sorter", options={"verify_ssl": False}) \
                .with_automatic_reconnect({
                "type": "interval",
                "keep_alive_interval": 10,
                "intervals": [1, 3, 5, 6, 7, 87, 3]
            }).build()
            hub_connection.on_open(lambda: print("connection opened and handshake received ready to send messages"))
            hub_connection.on_close(lambda: print("connection closed"))
            hub_connection.on("messageReceived", print)
            hub_connection.start()

            LegoAnalysisFast_pb2_grpc.add_LegoAnalysisFastServicer_to_server(
                LegoAnalysisFastService(hub_connection, lastImages), server)

        except:
            print("Can't connect to web gui and start LegoAnalysisFastService")
        # LegoControl_pb2_grpc.add_LegoControlServicer_to_server((LegoControlService(lastImages)), server)
        LegoControl_pb2_grpc.add_LegoControlServicer_to_server((LegoControlService(lastImages)), server2)
        server.add_insecure_port('[::]:50051')
        server.start()
        server2.add_insecure_port('[::]:50052')
        server2.start()
        server.wait_for_termination()
