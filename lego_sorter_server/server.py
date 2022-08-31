from threading import Event
from typing import Tuple
import os
import grpc
from loguru import logger
from concurrent import futures
from sqlalchemy import inspect

import uvicorn
from signalrcore.hub_connection_builder import HubConnectionBuilder

from lego_sorter_server.database import Models
from lego_sorter_server.database.Database import engine, SessionLocal
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

# from lego_sorter_server.api.ServerApi import app


class Server:

    @staticmethod
    def run(sorterConfig: BrickCategoryConfig, event: Event) -> Tuple[grpc.server, grpc.server, futures.ThreadPoolExecutor, futures.ThreadPoolExecutor, futures.ThreadPoolExecutor]:
        # inspector = inspect(engine)
        # # db.connect()
        # if not inspector.has_table("dbconfiguration"):
        #     db.create_tables([DBConfiguration])
        # if not inspector.has_table("dbsession"):
        #     db.create_tables([DBSession])
        # if not inspector.has_table("dbimage"):
        #     db.create_tables([DBImage])
        # if not inspector.has_table("dbimageresult"):
        #     db.create_tables([DBImageResult])

        db = SessionLocal()

        web_address = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "web_address").one_or_none()
        if "LEGO_SORTER_WEB_ADDRESS" in os.environ:
            if web_address is None:
                web_address = Models.DBConfiguration(option="web_address", value=os.getenv("LEGO_SORTER_WEB_ADDRESS"))
                db.add(web_address)
                db.commit()
                db.refresh(web_address)
            else:
                web_address.value = os.getenv("LEGO_SORTER_WEB_ADDRESS")
                db.commit()
        else:
            if web_address is None:
                web_address = Models.DBConfiguration(option="web_address", value="http://192.168.11.189:5002")
                db.add(web_address)
                db.commit()
                db.refresh(web_address)

        server_grpc_port_1 = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "server_grpc_port_1").one_or_none()
        if server_grpc_port_1 is None:
            server_grpc_port_1 = Models.DBConfiguration(option="server_grpc_port_1", value="50051")
            db.add(server_grpc_port_1)
            db.commit()
            db.refresh(server_grpc_port_1)

        server_grpc_port_2 = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "server_grpc_port_2").one_or_none()
        if server_grpc_port_2 is None:
            server_grpc_port_2 = Models.DBConfiguration(option="server_grpc_port_2", value="50052")
            db.add(server_grpc_port_2)
            db.commit()
            db.refresh(server_grpc_port_2)

        server_fastapi_port = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "server_fastapi_port").one_or_none()
        if server_fastapi_port is None:
            server_fastapi_port = Models.DBConfiguration(option="server_fastapi_port", value="5005")
            db.add(server_fastapi_port)
            db.commit()
            db.refresh(server_fastapi_port)

        server_fiftyone_port = db.query(Models.DBConfiguration).filter(
            Models.DBConfiguration.option == "server_fiftyone_port").one_or_none()
        if server_fiftyone_port is None:
            server_fiftyone_port = Models.DBConfiguration(option="server_fiftyone_port", value="5151")
            db.add(server_fiftyone_port)
            db.commit()
            db.refresh(server_fiftyone_port)

        server_fiftyone_address = db.query(Models.DBConfiguration).filter(
            Models.DBConfiguration.option == "server_fiftyone_address").one_or_none()
        if server_fiftyone_address is None:
            server_fiftyone_address = Models.DBConfiguration(option="server_fiftyone_address", value="0.0.0.0")
            db.add(server_fiftyone_address)
            db.commit()
            db.refresh(server_fiftyone_address)

        server_grpc_max_workers_1 = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "server_grpc_max_workers_1").one_or_none()
        if server_grpc_max_workers_1 is None:
            server_grpc_max_workers_1 = Models.DBConfiguration(option="server_grpc_max_workers_1", value="16")
            db.add(server_grpc_max_workers_1)
            db.commit()
            db.refresh(server_grpc_max_workers_1)

        server_grpc_max_workers_2 = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "server_grpc_max_workers_2").one_or_none()
        if server_grpc_max_workers_2 is None:
            server_grpc_max_workers_2 = Models.DBConfiguration(option="server_grpc_max_workers_2", value="4")
            db.add(server_grpc_max_workers_2)
            db.commit()
            db.refresh(server_grpc_max_workers_2)

        storageFastRunerExecutor_max_workers = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "storageFastRunerExecutor_max_workers").one_or_none()
        if storageFastRunerExecutor_max_workers is None:
            storageFastRunerExecutor_max_workers = Models.DBConfiguration(option="storageFastRunerExecutor_max_workers", value="1")
            db.add(storageFastRunerExecutor_max_workers)
            db.commit()
            db.refresh(storageFastRunerExecutor_max_workers)

        analyzerFastRunerExecutor_max_workers = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "analyzerFastRunerExecutor_max_workers").one_or_none()
        if analyzerFastRunerExecutor_max_workers is None:
            analyzerFastRunerExecutor_max_workers = Models.DBConfiguration(option="analyzerFastRunerExecutor_max_workers", value="1")
            db.add(analyzerFastRunerExecutor_max_workers)
            db.commit()
            db.refresh(analyzerFastRunerExecutor_max_workers)

        annotationFastRunerExecutor_max_workers = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "annotationFastRunerExecutor_max_workers").one_or_none()
        if annotationFastRunerExecutor_max_workers is None:
            annotationFastRunerExecutor_max_workers = Models.DBConfiguration(option="annotationFastRunerExecutor_max_workers", value="1")
            db.add(annotationFastRunerExecutor_max_workers)
            db.commit()
            db.refresh(annotationFastRunerExecutor_max_workers)

        if(int(analyzerFastRunerExecutor_max_workers.value)>2):
            analyzerFastRunerExecutor_max_workers.value = "2"
            db.commit()
            # analyzerFastRunerExecutor_max_workers.save()

        # storageFastRunerExecutor_max_workers.value = "3"
        # analyzerFastRunerExecutor_max_workers.value = "2"
        # annotationFastRunerExecutor_max_workers.value = "3"
        # db.commit()


        options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        lastImages = deque([], maxlen=20)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=int(server_grpc_max_workers_1.value)), options=options)
        server2 = grpc.server(futures.ThreadPoolExecutor(max_workers=int(server_grpc_max_workers_2.value)), options=options)
        storageFastRunerExecutor = futures.ThreadPoolExecutor(max_workers=int(storageFastRunerExecutor_max_workers.value))
        analyzerFastRunerExecutor = futures.ThreadPoolExecutor(max_workers=int(analyzerFastRunerExecutor_max_workers.value))
        annotationFastRunerExecutor = futures.ThreadPoolExecutor(max_workers=int(annotationFastRunerExecutor_max_workers.value))
        LegoSorter_pb2_grpc.add_LegoSorterServicer_to_server(LegoSorterService(sorterConfig), server)
        LegoCapture_pb2_grpc.add_LegoCaptureServicer_to_server(LegoCaptureService(event), server)
        LegoAnalysis_pb2_grpc.add_LegoAnalysisServicer_to_server(LegoAnalysisService(), server)
        try:
            hub_connection: HubConnectionBuilder = HubConnectionBuilder() \
                .with_url(f"{web_address.value}/hubs/sorter", options={"verify_ssl": False}) \
                .with_automatic_reconnect({
                "type": "interval",
                "keep_alive_interval": 10,
                "intervals": [1, 3, 5, 6, 7, 87, 3]
            }).build()
            hub_connection.on_open(lambda: logger.info("[hub_connection] connection opened and handshake received ready to send messages"))
            hub_connection.on_close(lambda: logger.info("[hub_connection] connection closed"))
            # hub_connection.on("messageReceived", print)
            hub_connection.start()

            LegoAnalysisFast_pb2_grpc.add_LegoAnalysisFastServicer_to_server(
                LegoAnalysisFastService(hub_connection, lastImages, storageFastRunerExecutor, analyzerFastRunerExecutor, annotationFastRunerExecutor, event), server)

        except Exception as exc:
            logger.error(exc)
            logger.warning("Can't connect to web gui and start LegoAnalysisFastService")
        # LegoControl_pb2_grpc.add_LegoControlServicer_to_server((LegoControlService(lastImages)), server)
        LegoControl_pb2_grpc.add_LegoControlServicer_to_server((LegoControlService(lastImages)), server2)
        server.add_insecure_port(f'[::]:{server_grpc_port_1.value}')
        server.start()
        server2.add_insecure_port(f'[::]:{server_grpc_port_2.value}')
        server2.start()
        # uvicorn.run(app, host="0.0.0.0", port=int(server_fastapi_port.value), log_level="info")
        db.close()
        return server, server2, storageFastRunerExecutor, analyzerFastRunerExecutor, annotationFastRunerExecutor
        # server.wait_for_termination()

    @staticmethod
    def stop(server: grpc.server, server2: grpc.server, storageFastRunerExecutor: futures.ThreadPoolExecutor, analyzerFastRunerExecutor: futures.ThreadPoolExecutor, annotationFastRunerExecutor: futures.ThreadPoolExecutor, event: Event):
        server.stop(None)
        server2.stop(None)
        event.set()
        storageFastRunerExecutor.shutdown(wait=False, cancel_futures=True)
        analyzerFastRunerExecutor.shutdown(wait=False, cancel_futures=True)
        annotationFastRunerExecutor.shutdown(wait=False, cancel_futures=True)
