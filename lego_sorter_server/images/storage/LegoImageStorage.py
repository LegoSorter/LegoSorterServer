import itertools
import json
import os
from pathlib import Path
from time import time
from typing import List

from PIL import Image
import logging


class LegoImageStorage:
    """This class is responsible for storing images of lego bricks"""

    def __init__(self, images_directory='./lego_sorter_server/images/storage/stored'):
        self.images_base_path = Path(images_directory)
        self.full_server_path = os.getcwd()
        self.create_directory(self.images_base_path, parents=True)

        self.jsonSaveData = dict()

        if (os.path.exists(self.images_base_path / 'classificationResults.json') == False):
            f = open(self.images_base_path / 'classificationResults.json', "w")
            f.close()

    @staticmethod
    def create_directory(directory, parents=True):
        if not directory.exists():
            directory.mkdir(parents=parents)
        if not directory.exists():
            raise Exception("Couldn't create an images directory {}", directory.absolute())

        return directory

    @staticmethod
    def generate_file_name(lego_class, img_format="jpg", prefix=''):
        return f'{prefix}{lego_class}_{round(time() * 1000)}.{img_format}'

    @staticmethod
    def extract_lego_class_from_file_name(filename):
        return filename.split('_')[-2]

    def find_image_path(self, filename: str):
        lego_class = self.extract_lego_class_from_file_name(filename)
        image_path = self.images_base_path / lego_class / filename

        if not image_path.exists():
            raise Exception("The image does not exist {}", image_path)

        return image_path

    def get_target_directory_for_lego_class(self, label: str) -> Path:
        target_directory = self.images_base_path / label

        return self.create_directory(target_directory, parents=False)

    def set_json_save_data_final_label(self, brick_id: str, label: str = ''):
        if label == '':
            self.jsonSaveData[brick_id]["final_label"] = self.jsonSaveData[brick_id]["images"][0]["label"]
        else:
            self.jsonSaveData[brick_id]["final_label"] = label

    def save_image(self, image: Image.Image, lego_class: str, prefix: str = '') -> str:
        """Save the image as representation of specified lego_class. Returns a name of the saved image"""
        target_directory = self.get_target_directory_for_lego_class(lego_class)
        filename = self.generate_file_name(lego_class, prefix=prefix)

        image = image.convert("RGB")
        image.save(str(target_directory / filename))

        logging.info(f"Saved the image {filename} of {lego_class} class\n")

        return filename

    def save_image_with_results(self, image: Image.Image, lego_class: str, brick_id: str, label: str, score: str, prefix: str = '') -> str:
        """Save the image as representation of specified lego_class. Returns a name of the saved image"""
        target_directory = self.get_target_directory_for_lego_class(label)
        filename = self.generate_file_name(lego_class, prefix=prefix)

        image = image.convert("RGB")
        image.save(str(target_directory / filename))

        if brick_id not in self.jsonSaveData:
            self.jsonSaveData[brick_id] = {"final_label": "", "images": []}
        self.jsonSaveData[brick_id]['images'].append({"filePath": str(self.full_server_path / target_directory / filename), "label": label, "score": score})

        logging.info(f"Saved the image {filename} of {lego_class} class\n")

        return filename

    def save_images_results_to_json(self):
        with open(self.images_base_path / 'classificationResults.json', mode='r+', encoding='utf-8') as feedsjson:
            try:
                feeds = json.load(feedsjson)
            except:
                feeds = {}
                
            for key, value in self.jsonSaveData.items():
                if key not in feeds:
                    feeds[key] = value
                else:
                    feeds[key]["images"].append(value["images"])
                    feeds[key]["final_label"] = value["final_label"]
            
            feedsjson.truncate(0)
            feedsjson.seek(0)
            json.dump(feeds, feedsjson)

        self.jsonSaveData.clear()

    def get_images(self, lego_class: str, limit: int = 10) -> List[Image.Image]:
        """Returns a list of images for specified lego_class"""

        lego_class_directory = self.images_base_path / lego_class

        if not lego_class_directory.exists():
            return []

        paths_iterator = itertools.islice(lego_class_directory.glob("**/*"), limit)

        return [Image.open(str(image_path)) for image_path in paths_iterator]

    def get_image(self, filename: str) -> Image.Image:
        image_path = self.find_image_path(filename)

        return Image.open(str(image_path))

    def remove_image(self, filename: str):
        image_path = self.find_image_path(filename)
        image_path.unlink()

    def remove_lego_class(self, lego_class: str):
        lego_class_directory = self.images_base_path / lego_class
        lego_class_directory.rmdir()