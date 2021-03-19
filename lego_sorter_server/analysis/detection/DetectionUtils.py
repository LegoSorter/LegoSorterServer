from PIL import Image


def resize(img, target):
    width, height = img.size
    scaling_factor = target / max(width, height)
    im_resized = img.resize((int(width * scaling_factor), int(height * scaling_factor)), Image.BICUBIC)
    new_im = Image.new('RGB', (target, target), color=(0, 0, 0))
    new_im.paste(im_resized, (0, 0))
    return new_im, scaling_factor


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
