from threading import Event
from typing import Tuple
import os
import grpc
from loguru import logger
from concurrent import futures

from signalrcore.hub_connection_builder import HubConnectionBuilder

from lego_sorter_server.database import Models
from lego_sorter_server.database.Database import SessionLocal
from lego_sorter_server.generated import LegoSorter_pb2_grpc, LegoCapture_pb2_grpc, LegoAnalysis_pb2_grpc, \
    LegoAnalysisFast_pb2_grpc, LegoControl_pb2_grpc
from lego_sorter_server.service.LegoCaptureService import LegoCaptureService
from service.LegoAnalysisService import LegoAnalysisService
# from service.LegoAnalysisFastService import LegoAnalysisFastService
from service.LegoSorterService import LegoSorterService
from service.LegoControlService import LegoControlService

from lego_sorter_server.service.BrickCategoryConfig import BrickCategoryConfig
from collections import deque


class Server:
    @staticmethod
    def str_to_bool_str(val, default_val):
        """Convert a string representation of truth to true (1) or false (0).
        True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
        are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
        'val' is anything else.
        """
        val = val.lower()
        if val in ('y', 'yes', 't', 'true', 'on', '1'):
            return "true"
        elif val in ('n', 'no', 'f', 'false', 'off', '0'):
            return "false"
        else:
            return default_val
            # raise ValueError("invalid truth value %r" % (val,))

    @staticmethod
    def str_to_str(val, default_val):
        if val != "":
            return val
        else:
            return default_val

    @staticmethod
    def int_to_str(val, default_val):
        try:
            if val != "":
                return str(int(val))
            else:
                return default_val
        except ValueError:
            return default_val

    @staticmethod
    def init_config(db, config_name, default_value, transform_func, env_name=None, set_env_var=False):
        conf = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == config_name).one_or_none()
        if (env_name is not None) and (env_name in os.environ):
            tmp_value = transform_func(os.getenv(env_name), default_value)
            if conf is None:
                conf = Models.DBConfiguration(option=config_name, value=tmp_value)
                db.add(conf)
                db.commit()
                db.refresh(conf)
            else:
                conf.value = tmp_value
                db.commit()
        else:
            if conf is None:
                conf = Models.DBConfiguration(option=config_name, value=default_value)
                db.add(conf)
                db.commit()
                db.refresh(conf)
        if set_env_var:
            os.environ[env_name] = conf.value
        return conf

    @staticmethod
    def init_config_int(db, config_name, default_value, env_name=None, set_env_var=False):
        return Server.init_config(db, config_name, default_value, Server.int_to_str, env_name, set_env_var)

    @staticmethod
    def init_config_str(db, config_name, default_value, env_name=None, set_env_var=False):
        return Server.init_config(db, config_name, default_value, Server.str_to_str, env_name, set_env_var)

    @staticmethod
    def init_config_bool_str(db, config_name, default_value, env_name=None, set_env_var=False):
        return Server.init_config(db, config_name, default_value, Server.str_to_bool_str, env_name, set_env_var)

    @staticmethod
    def run(sorter_config: BrickCategoryConfig, event: Event) -> Tuple[grpc.server, futures.ThreadPoolExecutor, futures.ThreadPoolExecutor, futures.ThreadPoolExecutor, futures.ThreadPoolExecutor, futures.ThreadPoolExecutor, futures.ThreadPoolExecutor]:
        # def run(sorterConfig: BrickCategoryConfig, event: Event) -> Tuple[grpc.server, grpc.server, futures.ThreadPoolExecutor, futures.ThreadPoolExecutor, futures.ThreadPoolExecutor]:

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

        # Options for Analyze Fast
        # 0 - Keras
        # 1 - TinyVit
        lego_sorter_classifier = Server.init_config_int(db, "lego_sorter_classifier", "1", "LEGO_SORTER_CLASSIFIER", True)
            # lego_sorter_classifier = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "LEGO_SORTER_CLASSIFIER").one_or_none()
            # if "LEGO_SORTER_CLASSIFIER" in os.environ:
            #     if lego_sorter_classifier is None:
            #         web_address = Models.DBConfiguration(option="LEGO_SORTER_CLASSIFIER", value=os.getenv("LEGO_SORTER_CLASSIFIER"))
            #         db.add(web_address)
            #         db.commit()
            #         db.refresh(web_address)
            #     else:
            #         lego_sorter_classifier.value = os.getenv("LEGO_SORTER_CLASSIFIER")
            #         db.commit()
            # else:
            #     if lego_sorter_classifier is None:
            #         lego_sorter_classifier = Models.DBConfiguration(option="LEGO_SORTER_CLASSIFIER", value="1")
            #         db.add(lego_sorter_classifier)
            #         db.commit()
            #         db.refresh(lego_sorter_classifier)

        # 0 - YOLOv5
        # 1 - YOLOv5 run in DeepSparse
        # 2 - YOLOv5 Google Coral Edge TPU classify 3 parts of scaled image
        # 3 - YOLOv5 run in Onnx
        # 4 - YOLOv5 Google Coral Edge TPU classify 3 parts of scaled and cropped image
        lego_sorter_detector = Server.init_config_int(db, "lego_sorter_detector", "0", "LEGO_SORTER_DETECTOR", True)
            # lego_sorter_detector = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "LEGO_SORTER_DETECTOR").one_or_none()
            # if "LEGO_SORTER_DETECTOR" in os.environ:
            #     if lego_sorter_detector is None:
            #         lego_sorter_detector = Models.DBConfiguration(option="LEGO_SORTER_DETECTOR", value=os.getenv("LEGO_SORTER_DETECTOR"))
            #         db.add(lego_sorter_detector)
            #         db.commit()
            #         db.refresh(lego_sorter_detector)
            #     else:
            #         lego_sorter_detector.value = os.getenv("LEGO_SORTER_DETECTOR")
            #         db.commit()
            # else:
            #     if lego_sorter_detector is None:
            #         lego_sorter_detector = Models.DBConfiguration(option="LEGO_SORTER_DETECTOR", value="0")
            #         db.add(lego_sorter_detector)
            #         db.commit()
            #         db.refresh(lego_sorter_detector)

        yolov5_model_path = Server.init_config_str(db, "yolov5_model_path", os.path.join("lego_sorter_server", "analysis", "detection", "models", "yolo_model", "yolov5_n.pt"), "LEGO_SORTER_YOLOV5_MODEL_PATH", True)
        keras_model_path = Server.init_config_str(db, "keras_model_path", os.path.join("lego_sorter_server", "analysis", "classification", "models", "keras_model", "447_classes.h5"), "LEGO_SORTER_KERAS_MODEL_PATH", True)
        tinyvit_model_path = Server.init_config_str(db, "tinyvit_model_path", os.path.join("lego_sorter_server", "analysis", "classification", "models", "tiny_vit_model", "tinyvit.pth"), "LEGO_SORTER_TINYVIT_MODEL_PATH", True)

        web_address = Server.init_config_str(db, "web_address", "http://127.0.0.1:5002", "LEGO_SORTER_WEB_ADDRESS")
            # web_address = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "web_address").one_or_none()
            # if "LEGO_SORTER_WEB_ADDRESS" in os.environ:
            #     if web_address is None:
            #         web_address = Models.DBConfiguration(option="web_address", value=os.getenv("LEGO_SORTER_WEB_ADDRESS"))
            #         db.add(web_address)
            #         db.commit()
            #         db.refresh(web_address)
            #     else:
            #         web_address.value = os.getenv("LEGO_SORTER_WEB_ADDRESS")
            #         db.commit()
            # else:
            #     if web_address is None:
            #         web_address = Models.DBConfiguration(option="web_address", value="http://127.0.0.1:5002")
            #         db.add(web_address)
            #         db.commit()
            #         db.refresh(web_address)

        server_grpc_port_1 = Server.init_config_int(db, "server_grpc_port_1", "50051", "LEGO_SORTER_SERVER_GRPC_PORT_1")
            # server_grpc_port_1 = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "server_grpc_port_1").one_or_none()
            # if server_grpc_port_1 is None:
            #     server_grpc_port_1 = Models.DBConfiguration(option="server_grpc_port_1", value="50051")
            #     db.add(server_grpc_port_1)
            #     db.commit()
            #     db.refresh(server_grpc_port_1)

        server_grpc_port_2 = Server.init_config_int(db, "server_grpc_port_2", "50052", "LEGO_SORTER_SERVER_GRPC_PORT_2")
            # server_grpc_port_2 = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "server_grpc_port_2").one_or_none()
            # if server_grpc_port_2 is None:
            #     server_grpc_port_2 = Models.DBConfiguration(option="server_grpc_port_2", value="50052")
            #     db.add(server_grpc_port_2)
            #     db.commit()
            #     db.refresh(server_grpc_port_2)

        server_fastapi_port = Server.init_config_int(db, "server_fastapi_port", "5005", "LEGO_SORTER_SERVER_FASTAPI_PORT")
            # server_fastapi_port = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "server_fastapi_port").one_or_none()
            # if server_fastapi_port is None:
            #     server_fastapi_port = Models.DBConfiguration(option="server_fastapi_port", value="5005")
            #     db.add(server_fastapi_port)
            #     db.commit()
            #     db.refresh(server_fastapi_port)

        server_fiftyone_port = Server.init_config_int(db, "server_fiftyone_port", "5151", "LEGO_SORTER_SERVER_FIFTYONE_PORT")
            # server_fiftyone_port = db.query(Models.DBConfiguration).filter(
            #     Models.DBConfiguration.option == "server_fiftyone_port").one_or_none()
            # if server_fiftyone_port is None:
            #     server_fiftyone_port = Models.DBConfiguration(option="server_fiftyone_port", value="5151")
            #     db.add(server_fiftyone_port)
            #     db.commit()
            #     db.refresh(server_fiftyone_port)

        server_fiftyone_address = Server.init_config_str(db, "server_fiftyone_address", "0.0.0.0", "LEGO_SORTER_SERVER_FIFTYONE_ADDRESS")
            # server_fiftyone_address = db.query(Models.DBConfiguration).filter(
            #     Models.DBConfiguration.option == "server_fiftyone_address").one_or_none()
            # if server_fiftyone_address is None:
            #     server_fiftyone_address = Models.DBConfiguration(option="server_fiftyone_address", value="0.0.0.0")
            #     db.add(server_fiftyone_address)
            #     db.commit()
            #     db.refresh(server_fiftyone_address)

        conveyor_local_address = Server.init_config_str(db, "conveyor_local_address", "http://192.168.83.45:8000", "LEGO_SORTER_CONVEYOR_LOCAL_ADDRESS")
            # conveyor_local_address = db.query(Models.DBConfiguration).filter(
            #     Models.DBConfiguration.option == "conveyor_local_address").one_or_none()
            # if conveyor_local_address is None:
            #     conveyor_local_address = Models.DBConfiguration(option="conveyor_local_address", value="http://192.168.83.45:8000")
            #     db.add(conveyor_local_address)
            #     db.commit()
            #     db.refresh(conveyor_local_address)

        sorter_local_address = Server.init_config_str(db, "sorter_local_address", "http://192.168.83.45:8001", "LEGO_SORTER_SORTER_LOCAL_ADDRESS")
            # sorter_local_address = db.query(Models.DBConfiguration).filter(
            #     Models.DBConfiguration.option == "sorter_local_address").one_or_none()
            # if sorter_local_address is None:
            #     sorter_local_address = Models.DBConfiguration(option="sorter_local_address", value="http://192.168.83.45:8001")
            #     db.add(sorter_local_address)
            #     db.commit()
            #     db.refresh(sorter_local_address)

        camera_conveyor_duty_cycle = Server.init_config_int(db, "camera_conveyor_duty_cycle", "50", "LEGO_SORTER_CAMERA_CONVEYOR_DUTY_CYCLE")
            # camera_conveyor_duty_cycle = db.query(Models.DBConfiguration).filter(
            #     Models.DBConfiguration.option == "camera_conveyor_duty_cycle").one_or_none()
            # if camera_conveyor_duty_cycle is None:
            #     camera_conveyor_duty_cycle = Models.DBConfiguration(option="camera_conveyor_duty_cycle", value="50")
            #     db.add(camera_conveyor_duty_cycle)
            #     db.commit()
            #     db.refresh(camera_conveyor_duty_cycle)

        camera_conveyor_frequency = Server.init_config_int(db, "camera_conveyor_frequency", "15", "LEGO_SORTER_CAMERA_CONVEYOR_FREQUENCY")
            # camera_conveyor_frequency = db.query(Models.DBConfiguration).filter(
            #     Models.DBConfiguration.option == "camera_conveyor_frequency").one_or_none()
            # if camera_conveyor_frequency is None:
            #     camera_conveyor_frequency = Models.DBConfiguration(option="camera_conveyor_frequency", value="15")
            #     db.add(camera_conveyor_frequency)
            #     db.commit()
            #     db.refresh(camera_conveyor_frequency)

        splitting_conveyor_duty_cycle = Server.init_config_int(db, "splitting_conveyor_duty_cycle", "50", "LEGO_SORTER_SPLITTING_CONVEYOR_DUTY_CYCLE")
            # splitting_conveyor_duty_cycle = db.query(Models.DBConfiguration).filter(
            #     Models.DBConfiguration.option == "splitting_conveyor_duty_cycle").one_or_none()
            # if splitting_conveyor_duty_cycle is None:
            #     splitting_conveyor_duty_cycle = Models.DBConfiguration(option="splitting_conveyor_duty_cycle", value="50")
            #     db.add(splitting_conveyor_duty_cycle)
            #     db.commit()
            #     db.refresh(splitting_conveyor_duty_cycle)

        splitting_conveyor_frequency = Server.init_config_int(db, "splitting_conveyor_frequency", "50", "LEGO_SORTER_SPLITTING_CONVEYOR_FREQUENCY")
            # splitting_conveyor_frequency = db.query(Models.DBConfiguration).filter(
            #     Models.DBConfiguration.option == "splitting_conveyor_frequency").one_or_none()
            # if splitting_conveyor_frequency is None:
            #     splitting_conveyor_frequency = Models.DBConfiguration(option="splitting_conveyor_frequency", value="50")
            #     db.add(splitting_conveyor_frequency)
            #     db.commit()
            #     db.refresh(splitting_conveyor_frequency)

        camera_conveyor_active_time = Server.init_config_int(db, "camera_conveyor_active_time", "1000", "LEGO_SORTER_CAMERA_CONVEYOR_ACTIVE_TIME")
            # camera_conveyor_active_time = db.query(Models.DBConfiguration).filter(
            #     Models.DBConfiguration.option == "camera_conveyor_active_time").one_or_none()
            # if camera_conveyor_active_time is None:
            #     camera_conveyor_active_time = Models.DBConfiguration(option="camera_conveyor_active_time", value="1000")
            #     db.add(camera_conveyor_active_time)
            #     db.commit()
            #     db.refresh(camera_conveyor_active_time)

        camera_conveyor_wait_time = Server.init_config_int(db, "camera_conveyor_wait_time", "2500", "LEGO_SORTER_CAMERA_CONVEYOR_WAIT_TIME")
            # camera_conveyor_wait_time = db.query(Models.DBConfiguration).filter(
            #     Models.DBConfiguration.option == "camera_conveyor_wait_time").one_or_none()
            # if camera_conveyor_wait_time is None:
            #     camera_conveyor_wait_time = Models.DBConfiguration(option="camera_conveyor_wait_time", value="2500")
            #     db.add(camera_conveyor_wait_time)
            #     db.commit()
            #     db.refresh(camera_conveyor_wait_time)

        sort = Server.init_config_bool_str(db, "sort", "false", "LEGO_SORTER_SORT")
            # sort = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "sort").one_or_none()
            # if "LEGO_SORTER_SORT" in os.environ:
            #     tmp_sort_value = Server.strtoboolstr(os.getenv("LEGO_SORTER_SORT"))
            #     if sort is None:
            #         sort = Models.DBConfiguration(option="sort", value=tmp_sort_value)
            #         db.add(sort)
            #         db.commit()
            #         db.refresh(sort)
            #     else:
            #         sort.value = tmp_sort_value
            #         db.commit()
            # else:
            #     if sort is None:
            #         sort = Models.DBConfiguration(option="sort", value="false")
            #         db.add(sort)
            #         db.commit()
            #         db.refresh(sort)

        crop = Server.init_config_bool_str(db, "crop", "false", "LEGO_SORTER_CROP")
            # crop = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "crop").one_or_none()
            # if "LEGO_SORTER_CROP" in os.environ:
            #     tmp_crop_value = Server.strtoboolstr(os.getenv("LEGO_SORTER_CROP"))
            #     if crop is None:
            #         crop = Models.DBConfiguration(option="crop", value=tmp_crop_value)
            #         db.add(crop)
            #         db.commit()
            #         db.refresh(crop)
            #     else:
            #         crop.value = tmp_crop_value
            #         db.commit()
            # else:
            #     if crop is None:
            #         crop = Models.DBConfiguration(option="crop", value="false")
            #         db.add(crop)
            #         db.commit()
            #         db.refresh(crop)

        # not set in configServer
        store_img_override = Server.init_config_bool_str(db, "store_img_override", "false", "LEGO_SORTER_STORE_IMG_OVERRIDE")
        # not set in configServer
        store_img_session = Server.init_config_str(db, "store_img_session", "default_session", "LEGO_SORTER_STORE_IMG_SESSION")

        server_grpc_max_workers_1 = Server.init_config_int(db, "server_grpc_max_workers_1", "16", "LEGO_SORTER_SERVER_GRPC_MAX_WORKERS_1")
            # server_grpc_max_workers_1 = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "server_grpc_max_workers_1").one_or_none()
            # if server_grpc_max_workers_1 is None:
            #     server_grpc_max_workers_1 = Models.DBConfiguration(option="server_grpc_max_workers_1", value="16")
            #     db.add(server_grpc_max_workers_1)
            #     db.commit()
            #     db.refresh(server_grpc_max_workers_1)

        server_grpc_max_workers_2 = Server.init_config_int(db, "server_grpc_max_workers_2", "4", "LEGO_SORTER_SERVER_GRPC_MAX_WORKERS_2")
            # server_grpc_max_workers_2 = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "server_grpc_max_workers_2").one_or_none()
            # if server_grpc_max_workers_2 is None:
            #     server_grpc_max_workers_2 = Models.DBConfiguration(option="server_grpc_max_workers_2", value="4")
            #     db.add(server_grpc_max_workers_2)
            #     db.commit()
            #     db.refresh(server_grpc_max_workers_2)

        storage_fast_runer_executor_max_workers = Server.init_config_int(db, "storage_fast_runer_executor_max_workers", "1", "LEGO_SORTER_STORAGE_FAST_RUNER_EXECUTOR_MAX_WORKERS")
            # storageFastRunerExecutor_max_workers = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "storage_fast_runer_executor_max_workers").one_or_none()
            # if storageFastRunerExecutor_max_workers is None:
            #     storageFastRunerExecutor_max_workers = Models.DBConfiguration(option="storageFastRunerExecutor_max_workers", value="1")
            #     db.add(storageFastRunerExecutor_max_workers)
            #     db.commit()
            #     db.refresh(storageFastRunerExecutor_max_workers)

        analyzer_fast_runer_executor_max_workers = Server.init_config_int(db, "analyzer_fast_runer_executor_max_workers", "1", "LEGO_SORTER_ANALYZER_FAST_RUNER_EXECUTOR_MAX_WORKERS")
            # analyzerFastRunerExecutor_max_workers = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "analyzerFastRunerExecutor_max_workers").one_or_none()
            # if analyzerFastRunerExecutor_max_workers is None:
            #     analyzerFastRunerExecutor_max_workers = Models.DBConfiguration(option="analyzerFastRunerExecutor_max_workers", value="1")
            #     db.add(analyzerFastRunerExecutor_max_workers)
            #     db.commit()
            #     db.refresh(analyzerFastRunerExecutor_max_workers)

        annotation_fast_runer_executor_max_workers = Server.init_config_int(db, "annotation_fast_runer_executor_max_workers", "1", "LEGO_SORTER_ANNOTATION_FAST_RUNER_EXECUTOR_MAX_WORKERS")
            # annotationFastRunerExecutor_max_workers = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "annotationFastRunerExecutor_max_workers").one_or_none()
            # if annotationFastRunerExecutor_max_workers is None:
            #     annotationFastRunerExecutor_max_workers = Models.DBConfiguration(option="annotationFastRunerExecutor_max_workers", value="1")
            #     db.add(annotationFastRunerExecutor_max_workers)
            #     db.commit()
            #     db.refresh(annotationFastRunerExecutor_max_workers)

        if int(analyzer_fast_runer_executor_max_workers.value) > 2:
            analyzer_fast_runer_executor_max_workers.value = "2"
            db.commit()
            # analyzerFastRunerExecutor_max_workers.save()

        storage_queue_limit = Server.init_config_int(db, "storage_queue_limit", "1000", "LEGO_SORTER_STORAGE_QUEUE_LIMIT", True)
        processing_queue_limit = Server.init_config_int(db, "processing_queue_limit", "1000", "LEGO_SORTER_PROCESSING_QUEUE_LIMIT", True)
        sort_queue_limit = Server.init_config_int(db, "sort_queue_limit", "1000", "LEGO_SORTER_SORT_QUEUE_LIMIT", True)
        annotation_queue_limit = Server.init_config_int(db, "annotation_queue_limit", "1000", "LEGO_SORTER_ANNOTATION_QUEUE_LIMIT", True)
        crops_queue_limit = Server.init_config_int(db, "crops_queue_limit", "1000", "LEGO_SORTER_CROPS_QUEUE_LIMIT", True)
        last_images_limit = Server.init_config_int(db, "last_images_limit", "4", "LEGO_SORTER_LAST_IMAGES_LIMIT")

        # storageFastRunerExecutor_max_workers.value = "3"
        # analyzerFastRunerExecutor_max_workers.value = "2"
        # annotationFastRunerExecutor_max_workers.value = "3"
        # db.commit()


        options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        last_images = deque([], maxlen=int(last_images_limit.value))
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=int(server_grpc_max_workers_1.value)), options=options)
        # server2 = grpc.server(futures.ThreadPoolExecutor(max_workers=int(server_grpc_max_workers_2.value)), options=options)
        storage_fast_runer_executor = futures.ThreadPoolExecutor(max_workers=int(storage_fast_runer_executor_max_workers.value))
        analyzer_fast_runer_executor = futures.ThreadPoolExecutor(max_workers=int(analyzer_fast_runer_executor_max_workers.value))
        annotation_fast_runer_executor = futures.ThreadPoolExecutor(max_workers=int(annotation_fast_runer_executor_max_workers.value))
        sort_fast_runer_executor = futures.ThreadPoolExecutor(max_workers=2)
        queue_info_fast_runer_executor = futures.ThreadPoolExecutor(max_workers=2)
        storage_crops_fast_runer_executor = futures.ThreadPoolExecutor(max_workers=2)

        LegoSorter_pb2_grpc.add_LegoSorterServicer_to_server(LegoSorterService(sorter_config), server)
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
            hub_connection.on_error(lambda data: logger.info(f"[hub_connection] An exception was thrown closed{data.error}"))
            # hub_connection.on("messageReceived", print)
            hub_connection.start()

            # LegoAnalysisFastService use files that require LEGO_SORTER_DETECTOR and LEGO_SORTER_CLASSIFIER env variables that are set earlier in code
            from lego_sorter_server.service.LegoAnalysisFastService import LegoAnalysisFastService
            LegoAnalysisFast_pb2_grpc.add_LegoAnalysisFastServicer_to_server(
                LegoAnalysisFastService(hub_connection, sorter_config, last_images, storage_fast_runer_executor, analyzer_fast_runer_executor, annotation_fast_runer_executor, sort_fast_runer_executor, queue_info_fast_runer_executor, storage_crops_fast_runer_executor, event), server)

        except Exception as exc:
            logger.error(exc)
            logger.warning("Can't connect to web gui and start LegoAnalysisFastService")
        LegoControl_pb2_grpc.add_LegoControlServicer_to_server((LegoControlService(last_images)), server)
        # LegoControl_pb2_grpc.add_LegoControlServicer_to_server((LegoControlService(last_images)), server2)
        server.add_insecure_port(f'[::]:{server_grpc_port_1.value}')
        server.start()
        # server2.add_insecure_port(f'[::]:{server_grpc_port_2.value}')
        # server2.start()
        # uvicorn.run(app, host="0.0.0.0", port=int(server_fastapi_port.value), log_level="info")
        db.close()
        return server, storage_fast_runer_executor, analyzer_fast_runer_executor, annotation_fast_runer_executor, sort_fast_runer_executor, queue_info_fast_runer_executor, storage_crops_fast_runer_executor
        # return server, server2, storageFastRunerExecutor, analyzerFastRunerExecutor, annotationFastRunerExecutor
        # server.wait_for_termination()

    @staticmethod
    # def stop(server: grpc.server, server2: grpc.server, storageFastRunerExecutor: futures.ThreadPoolExecutor, analyzerFastRunerExecutor: futures.ThreadPoolExecutor, annotationFastRunerExecutor: futures.ThreadPoolExecutor, event: Event):
    def stop(server: grpc.server, storage_fast_runer_executor: futures.ThreadPoolExecutor, analyzer_fast_runer_executor: futures.ThreadPoolExecutor, annotation_fast_runer_executor: futures.ThreadPoolExecutor, sort_fast_runer_executor: futures.ThreadPoolExecutor, queue_info_fast_runer_executor: futures.ThreadPoolExecutor, storage_crops_fast_runer_executor: futures.ThreadPoolExecutor, event: Event):
        server.stop(None)
        # server2.stop(None)
        event.set()
        storage_fast_runer_executor.shutdown(wait=False, cancel_futures=True)
        analyzer_fast_runer_executor.shutdown(wait=False, cancel_futures=True)
        annotation_fast_runer_executor.shutdown(wait=False, cancel_futures=True)
        sort_fast_runer_executor.shutdown(wait=False, cancel_futures=True)
        queue_info_fast_runer_executor.shutdown(wait=False, cancel_futures=True)
        storage_crops_fast_runer_executor.shutdown(wait=False, cancel_futures=True)
