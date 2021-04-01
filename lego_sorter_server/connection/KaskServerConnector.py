import getpass
import logging
import zipfile

from deprecated import deprecated
from jumpssh import SSHSession
from pathlib import Path

DETECTION_MODELS_PATH = './lego_sorter_server/analysis/detection/models'
DETECTION_MODELS_REMOTE_PATH = '/backup/LEGO2/SERVER/MODELS/DETECTION/models.zip'
CLASSIFICATION_MODEL_PATH = './lego_sorter_server/analysis/classification/models/saved'
CLASSIFICATION_MODEL_REMOTE_PATH = '/backup/LEGO2/SERVER/MODELS/CLASSIFICATION/saved.zip'


@deprecated
class KaskServerConnector:

    def __init__(self):
        self.login = ""
        self.password = ""
        self.connection_successful = False

    def ask_for_credentials(self):
        self.login = input("Username: ")
        self.password = getpass.getpass()

        return True

    def download_models(self):
        if not self.login:
            success = self.ask_for_credentials()
            if not success:
                raise Exception("Couldn't access credentials")

        gateway_session = SSHSession('kask.eti.pg.gda.pl', self.login, password=self.password, compress=True).open()

        if gateway_session.is_active():
            logging.info("Downloading models. Let's grab a cup of coffee, it can take a while...")

            session = gateway_session.get_remote_session('apl11.eti.pg.gda.pl', username=self.login,
                                                         password=self.password, compress=True)

            extract_detection = False
            extract_classification = False
            if Path(DETECTION_MODELS_PATH).exists() and any(Path(DETECTION_MODELS_PATH).iterdir()):
                logging.info("Detection models exist, skipping...")
            else:
                logging.info("Downloading detection models")
                session.get(DETECTION_MODELS_REMOTE_PATH, DETECTION_MODELS_PATH + ".zip")
                extract_detection = True
                logging.info("detection models downloaded...")

            if Path(CLASSIFICATION_MODEL_PATH).exists():
                logging.info("Classification model exists, skipping...")
            else:
                logging.info("Downloading classification model")
                session.get(CLASSIFICATION_MODEL_REMOTE_PATH, CLASSIFICATION_MODEL_PATH + ".zip")
                logging.info("classification model downloaded...")
                extract_classification = True
                session.close()

            logging.info("All models downloaded! Unzipping")

            if extract_detection:
                with zipfile.ZipFile(DETECTION_MODELS_PATH + ".zip", 'r') as zip_ref:
                    zip_ref.extractall('./lego_sorter_server/detection/')

            if extract_classification:
                with zipfile.ZipFile(CLASSIFICATION_MODEL_PATH + ".zip", 'r') as zip_ref:
                    zip_ref.extractall('./lego_sorter_server/classification/models/')
        else:
            raise Exception("Couldn't connect to the server")
