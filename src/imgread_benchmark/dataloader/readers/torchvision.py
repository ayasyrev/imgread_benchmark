import torch
from torchvision.io import ImageReadMode, decode_image


def read(path):
    return decode_image(str(path), mode=ImageReadMode.RGB, apply_exif_orientation=False)


def read_bytes(data):
    # Torch requires a writable buffer; keep this conversion in the measured path.
    encoded = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return decode_image(encoded, mode=ImageReadMode.RGB, apply_exif_orientation=False)
