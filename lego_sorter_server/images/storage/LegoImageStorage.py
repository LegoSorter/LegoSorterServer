import itertools
from pathlib import Path
from time import time
from PIL import Image
import logging


class LegoImageStorage:
    """This class is responsible for storing images of lego bricks"""

    def __init__(self, images_directory='./lego_sorter_server/images/storage/stored'):
        self.images_base_path = Path(images_directory)
        self.create_directory(self.images_base_path, parents=True)

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

    def save_image(self, image: Image.Image, lego_class: str, prefix: str = '') -> str:
        """Save the image as representation of specified lego_class. Returns a name of the saved image"""
        target_directory = self.get_target_directory_for_lego_class(lego_class)
        filename = self.generate_file_name(lego_class, prefix=prefix)

        image.save(str(target_directory / filename))

        logging.info(f"Saved the image {filename} of {lego_class} class\n")

        return filename

    def get_images(self, lego_class: str, limit: int = 10) -> [Image.Image]:
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
