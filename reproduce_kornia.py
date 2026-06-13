from imgread_benchmark.img_libs.kornia import read_img_ndarray as read_kornia
from imgread_benchmark.img_libs.PIL import read_img_ndarray as read_pil
import numpy as np

img_path = "tests/test_imgs/cat.png"
img_k = read_kornia(img_path)
img_p = read_pil(img_path)

diff = np.abs(img_k.astype(float) - img_p.astype(float))
print(f"Max diff: {diff.max()}")
print(f"Mean diff: {diff.mean()}")
