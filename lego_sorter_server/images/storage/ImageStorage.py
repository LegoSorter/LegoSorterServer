from io import BytesIO
from pathlib import Path
from time import time
from PIL import Image


class ImageStorage:
    def __init__(self, images_directory='.'):
        self.images_base_path = Path(images_directory)
        self.create_directory(self.images_base_path, parents=True)

    @staticmethod
    def create_directory(directory, parents=True):
        if not directory.exists():
            directory.mkdir(parents=parents)
        if not directory.exists():
            raise Exception("Couldn't create an images directory {}", directory.absolute())

    @staticmethod
    def rotate_image(image, rotation):
        if rotation == 90:
            image = image.transpose(Image.ROTATE_270)
        if rotation == 180:
            image = image.rotate(180)
        if rotation == 270:
            image = image.transpose(Image.ROTATE_90)

        return image

    @staticmethod
    def generate_file_name(label, img_format=".jpg"):
        return f'{label}_{round(time() * 1000)}.{img_format}'

    def get_target_directory_for_lego_class(self, label):
        target_directory = self.images_base_path / label

        return self.create_directory(target_directory, parents=False)

    def save_image(self, image: BytesIO, label, rotation):
        image = Image.open(image)
        image = self.rotate_image(image, rotation)
        target_directory = self.get_target_directory_for_lego_class(label)
        filename = self.generate_file_name(label)

        image.save(str(target_directory / filename))

        return image



