import requests


class LegoSorterController:
    MACHINE_LOCAL_ADDRESS = "http://raspberrypi.local"
    DEFAULT_ADDRESS = 8000

    def __init__(self):
        self.speed = 50

    def run_conveyor(self):
        requests.get(f"{self.MACHINE_LOCAL_ADDRESS}:{self.DEFAULT_ADDRESS}/start?duty_cycle={self.speed}")

    def stop_conveyor(self):
        requests.get(f"{self.MACHINE_LOCAL_ADDRESS}:{self.DEFAULT_ADDRESS}/stop")

    def on_brick_recognized(self, brick):
        pass

    def set_machine_speed(self, speed):
        self.speed = speed
