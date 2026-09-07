from torchvision.io import ImageReadMode, decode_image


def read(path):
    return decode_image(str(path), mode=ImageReadMode.RGB, apply_exif_orientation=False)
