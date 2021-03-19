from PIL import Image

from lego_sorter_server.analysis.classification.toolkit.transformations.transformation import Transformation


class Simple(Transformation):
    @staticmethod
    def transform(img, desired_size=299):
        old_size = img.size
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = img.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size - new_size[0]) // 2,
                          (desired_size - new_size[1]) // 2))
        return new_im

