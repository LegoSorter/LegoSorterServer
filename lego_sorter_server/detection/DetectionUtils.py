from PIL import Image


def resize(img, target):
    width, height = img.size
    scaling_factor = target / max(width, height)
    im_resized = img.resize((int(width * scaling_factor), int(height * scaling_factor)), Image.BICUBIC)
    new_im = Image.new('RGB', (target, target), color=(0, 0, 0))
    new_im.paste(im_resized, (0, 0))
    return new_im, scaling_factor
