# Image libs installed with pip: Name of lib: name of package
from importlib.util import find_spec


lib_to_package = {
    'PIL': 'pillow',
    'accimage': 'accimage',  # conda
    'jpeg4py': 'jpeg4py',
    'cv2': 'opencv-python-headless',
    'skimage': 'scikit-image',
    'imageio': 'imageio',
    'pyvips': 'pyvips',
    'torchvision': 'torchvision'}


img_lib_available = [lib_name for lib_name in lib_to_package if find_spec(lib_name) is not None]
