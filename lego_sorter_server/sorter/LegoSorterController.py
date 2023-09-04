import logging

import requests
import config

from lego_sorter_server.service.BrickCategoryConfig import BrickCategoryConfig


class LegoSorterController:
    #TODO: move both of these variables to the separate config file
    CONVEYOR_LOCAL_ADDRESS = config.CONVEYOR_LOCAL_ADDRESS
    SORTER_LOCAL_ADDRESS = config.SORTER_LOCAL_ADDRESS

    def __init__(self, brickCategoryConfig: BrickCategoryConfig):
        self.speed = 50
        self.brickCategoryConfig = brickCategoryConfig

    def run_conveyor(self):
        requests.get(f"{self.CONVEYOR_LOCAL_ADDRESS}/start?duty_cycle={self.speed}")

    def stop_conveyor(self):
        requests.get(f"{self.CONVEYOR_LOCAL_ADDRESS}/stop")

    def on_brick_recognized(self, brick):
        brick_coords, brick_cls, brick_prob = brick
        cat_name, pos = self.brickCategoryConfig[brick_cls]
        logging.info(f"Moving brick with class: {brick_cls} to stack: {cat_name} (pos: {pos})")
        requests.get(f"{self.SORTER_LOCAL_ADDRESS}/sort?action={pos}")

    def set_machine_speed(self, speed):
        self.speed = speed