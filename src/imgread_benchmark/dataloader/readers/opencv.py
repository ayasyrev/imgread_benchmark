import cv2


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR_RGB | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("cv2.imread returned None")
    return image


def read_bgr_cvtcolor(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("cv2.imread returned None")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
