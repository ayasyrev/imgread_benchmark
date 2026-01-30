from importlib.util import find_spec

lib_to_package = {
    "PIL": "pillow",
    "accimage": "accimage",  # only conda
    "jpeg4py": "jpeg4py",
    "cv2": "opencv-python-headless",  # conda - opencv
    "skimage": "scikit-image",  # conda
    "imageio": "imageio",  # conda
    "imread": "imread",  # conda
    "kornia": "kornia",
    # "pyvips": "pyvips",  # conda
    "torchvision": "torchvision",
}


img_lib_available = []
for lib_name in lib_to_package:
    if find_spec(lib_name) is not None:
        if lib_name == "jpeg4py":
            try:
                import jpeg4py
                # Attempt to initialize to check for libjpeg-turbo
                # Accessing .JPEG requires the lib, but simple import might not trigger it until usage.
                # However, the previous traceback showed failure during init of JPEG object, which we don't instantiate here.
                # But let's try a dummy check if possible or just rely on finding the spec.
                # Actually, the traceback shows failure in _initialize called by JPEG.__init__.
                # We can try to force initialization if there is a public API for it, or just wrap the usage in tests.
                # Given the constraints, let's just skip it if it fails to import or fundamental check.
                pass 
            except OSError:
                continue
        
        # Determine if we should really exclude it. 
        # For now, let's keep the find_spec check as primary but add specific exclusion for broken jpeg4py environment.
        pass

# Re-implementing the list comprehension with a check
img_lib_available = [
    lib_name for lib_name in lib_to_package if find_spec(lib_name) is not None
]

# Quick fix for the broken environment:
try:
    import jpeg4py
    # Trigger the library loading
    from jpeg4py._cffi import lib
    if lib is None:
         # Try to initialize to see if it raises
        from jpeg4py._cffi import _initialize, backends
        _initialize(backends)
except (ImportError, OSError):
    if "jpeg4py" in img_lib_available:
        img_lib_available.remove("jpeg4py")
except Exception:
    # If anything else goes wrong with jpeg4py, exclude it
    if "jpeg4py" in img_lib_available:
        img_lib_available.remove("jpeg4py")

