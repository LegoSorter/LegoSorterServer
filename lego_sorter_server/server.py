import grpc

from concurrent import futures

from signalrcore.hub_connection_builder import HubConnectionBuilder

from lego_sorter_server.database.Models import *
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
        db.connect()
        if not DBConfiguration.table_exists():
            db.create_tables([DBConfiguration])
        if not DBSession.table_exists():
            db.create_tables([DBSession])
        if not DBImage.table_exists():
            db.create_tables([DBImage])
        if not DBImageResult.table_exists():
            db.create_tables([DBImageResult])

        web_address = DBConfiguration.get_or_none(option="web_address")
        if web_address is None:
            web_address = DBConfiguration.create(option="web_address", value="http://192.168.11.189:5002")

        server_grpc_port_1 = DBConfiguration.get(option="server_grpc_port_1")
        if server_grpc_port_1 is None:
            server_grpc_port_1 = DBConfiguration.create(option="server_grpc_port_1", value="50051")

        server_grpc_port_2 = DBConfiguration.get(option="server_grpc_port_2")
        if server_grpc_port_2 is None:
            server_grpc_port_2 = DBConfiguration.create(option="server_grpc_port_2", value="50052")

        server_grpc_max_workers_1 = DBConfiguration.get(option="server_grpc_max_workers_1")
        if server_grpc_max_workers_1 is None:
            server_grpc_max_workers_1 = DBConfiguration.create(option="server_grpc_max_workers_1", value="16")

        server_grpc_max_workers_2 = DBConfiguration.get(option="server_grpc_max_workers_2")
        if server_grpc_port_1 is None:
            server_grpc_max_workers_2 = DBConfiguration.create(option="server_grpc_max_workers_2", value="4")

        storageFastRunerExecutor_max_workers = DBConfiguration.get(option="storageFastRunerExecutor_max_workers")
        if storageFastRunerExecutor_max_workers is None:
            storageFastRunerExecutor_max_workers = DBConfiguration.create(option="storageFastRunerExecutor_max_workers", value="1")

        analyzerFastRunerExecutor_max_workers = DBConfiguration.get(option="analyzerFastRunerExecutor_max_workers")
        if analyzerFastRunerExecutor_max_workers is None:
            analyzerFastRunerExecutor_max_workers = DBConfiguration.create(option="analyzerFastRunerExecutor_max_workers", value="1")

        annotationFastRunerExecutor_max_workers = DBConfiguration.get(option="annotationFastRunerExecutor_max_workers")
        if annotationFastRunerExecutor_max_workers is None:
            annotationFastRunerExecutor_max_workers = DBConfiguration.create(option="annotationFastRunerExecutor_max_workers", value="1")

        if(int(analyzerFastRunerExecutor_max_workers.value)>2):
            analyzerFastRunerExecutor_max_workers.value = "2"
            analyzerFastRunerExecutor_max_workers.save()

        storageFastRunerExecutor_max_workers.value = "3"
        storageFastRunerExecutor_max_workers.save()
        analyzerFastRunerExecutor_max_workers.value = "3"
        analyzerFastRunerExecutor_max_workers.save()
        annotationFastRunerExecutor_max_workers.value = "2"
        annotationFastRunerExecutor_max_workers.save()

        options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        lastImages = deque([], maxlen=20)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=int(server_grpc_max_workers_1.value)), options=options)
        server2 = grpc.server(futures.ThreadPoolExecutor(max_workers=int(server_grpc_max_workers_2.value)), options=options)
        storageFastRunerExecutor = futures.ThreadPoolExecutor(max_workers=int(storageFastRunerExecutor_max_workers.value))
        analyzerFastRunerExecutor = futures.ThreadPoolExecutor(max_workers=int(analyzerFastRunerExecutor_max_workers.value))
        annotationFastRunerExecutor = futures.ThreadPoolExecutor(max_workers=int(annotationFastRunerExecutor_max_workers.value))
        LegoSorter_pb2_grpc.add_LegoSorterServicer_to_server(LegoSorterService(sorterConfig), server)
        LegoCapture_pb2_grpc.add_LegoCaptureServicer_to_server(LegoCaptureService(), server)
        LegoAnalysis_pb2_grpc.add_LegoAnalysisServicer_to_server(LegoAnalysisService(), server)
        try:
            hub_connection: HubConnectionBuilder = HubConnectionBuilder() \
                .with_url(f"{web_address.value}/hubs/sorter", options={"verify_ssl": False}) \
                .with_automatic_reconnect({
                "type": "interval",
                "keep_alive_interval": 10,
                "intervals": [1, 3, 5, 6, 7, 87, 3]
            }).build()
            hub_connection.on_open(lambda: print("connection opened and handshake received ready to send messages"))
            hub_connection.on_close(lambda: print("connection closed"))
            # hub_connection.on("messageReceived", print)
            hub_connection.start()

            LegoAnalysisFast_pb2_grpc.add_LegoAnalysisFastServicer_to_server(
                LegoAnalysisFastService(hub_connection, lastImages, storageFastRunerExecutor, analyzerFastRunerExecutor, annotationFastRunerExecutor), server)

        except Exception as exc:
            print("Can't connect to web gui and start LegoAnalysisFastService")
        # LegoControl_pb2_grpc.add_LegoControlServicer_to_server((LegoControlService(lastImages)), server)
        LegoControl_pb2_grpc.add_LegoControlServicer_to_server((LegoControlService(lastImages)), server2)
        server.add_insecure_port(f'[::]:{server_grpc_port_1.value}')
        server.start()
        server2.add_insecure_port(f'[::]:{server_grpc_port_2.value}')
        server2.start()
        server.wait_for_termination()
