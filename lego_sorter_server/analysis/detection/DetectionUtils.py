import pyvips
from PIL import Image


def resize(img: Image.Image, target):
    width, height = img.size
    scaling_factor = target / max(width, height)
    im_resized = img.resize((int(width * scaling_factor), int(height * scaling_factor)), Image.BICUBIC)
    new_im = Image.new('RGB', (target, target), color=(0, 0, 0))
    new_im.paste(im_resized, (0, 0))
    return new_im, scaling_factor


def resizeVips(img: pyvips.Image, target):
    width = img.width
    height = img.height
    hscaling_factor = target / width
    vscaling_factor = target / height
    # im = Image.fromarray(img.numpy())
    # im.show()
    if(height>width):
        im_resized = img.resize(vscaling_factor, kernel=pyvips.enums.Kernel.CUBIC, vscale=vscaling_factor)  # pyvips.enums.Kernel.CUBIC
    else:
        im_resized = img.resize(hscaling_factor, kernel=pyvips.enums.Kernel.CUBIC, vscale=hscaling_factor)  # pyvips.enums.Kernel.CUBIC
    # im_resized = img.resize((int(width * scaling_factor), int(height * scaling_factor)), Image.BICUBIC)
    # im = Image.fromarray(im_resized.numpy())
    # im.show()
    new_im = pyvips.Image.black(target, target, bands=3)
    new_im = new_im.draw_image(im_resized, 0, 0, mode=pyvips.enums.CombineMode.SET)
    # new_im = new_im.insert(im_resized, (0, 0))
    return new_im, target / max(width, height)


def crop_with_margin_from_bb(image, bounding_box, abs_margin=0, rel_margin=0.10):
    return crop_with_margin(image, bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3], abs_margin,
                            rel_margin)


def crop_with_margin(image, ymin, xmin, ymax, xmax, abs_margin=0, rel_margin=0.10):
    width, height = image.size

    # Apply margins
    avg_length = ((xmax - xmin) + (ymax - ymin)) / 2
    ymin = max(ymin - abs_margin - rel_margin * avg_length, 0)
    xmin = max(xmin - abs_margin - rel_margin * avg_length, 0)
    ymax = min(ymax + abs_margin + rel_margin * avg_length, height)
    xmax = min(xmax + abs_margin + rel_margin * avg_length, width)

    return image.crop([xmin, ymin, xmax, ymax])


def crop_with_margin_from_bb_vips(image, bounding_box, abs_margin=0, rel_margin=0.10):
    return crop_with_margin_vips(image, bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3], abs_margin,
                            rel_margin)


def crop_with_margin_vips(image, ymin, xmin, ymax, xmax, abs_margin=0, rel_margin=0.10):
    width = image.width
    height = image.height

    # Apply margins
    avg_length = ((xmax - xmin) + (ymax - ymin)) / 2
    ymin = max(ymin - abs_margin - rel_margin * avg_length, 0)
    xmin = max(xmin - abs_margin - rel_margin * avg_length, 0)
    ymax = min(ymax + abs_margin + rel_margin * avg_length, height)
    xmax = min(xmax + abs_margin + rel_margin * avg_length, width)

    return image.crop(xmin, ymin, xmax-xmin, ymax-ymin)
