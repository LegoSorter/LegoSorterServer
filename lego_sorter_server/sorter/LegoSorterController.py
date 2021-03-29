import requests


class LegoSorterController:
    MACHINE_LOCAL_ADDRESS = "http://raspberrypi.local"
    DEFAULT_ADDRESS = 8000

    def run_conveyor(self):
        requests.get(f"{self.MACHINE_LOCAL_ADDRESS}:{self.DEFAULT_ADDRESS}/start")

    def stop_conveyor(self):
        requests.get(f"{self.MACHINE_LOCAL_ADDRESS}:{self.DEFAULT_ADDRESS}/stop")

    def on_brick_recognized(self, brick):
        pass
