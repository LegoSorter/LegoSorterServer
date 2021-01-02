import getpass
import logging
import zipfile

from jumpssh import SSHSession


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
            session.get(
                '/backup/LEGO2/SERVER/MODELS/DETECTION/lego_detection_model.zip',
                './lego_sorter_server/detection/models/lego_detection_model.zip')
            session.close()

            logging.info("Model downloaded! Unzipping")

            with zipfile.ZipFile('./lego_sorter_server/detection/models/lego_detection_model.zip', 'r') as zip_ref:
                zip_ref.extractall('./lego_sorter_server/detection/models/')
        else:
            raise Exception("Couldn't connect to the server")
