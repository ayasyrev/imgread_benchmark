"""Four explicit paths; capability probes run in disposable subprocesses."""

import importlib
import json
import os
import subprocess
import sys
from dataclasses import asdict

from ..models import READER_IDS, ReaderInfo

_DESCRIPTIONS = (
    "Pillow convert RGB → ndarray",
    "torchvision decode_image RGB → Tensor",
    "OpenCV direct RGB → ndarray",
    "OpenCV BGR + cvtColor → ndarray",
)


def get_reader(reader_id):
    paths = {
        "pil-rgb": ("pillow", "read"),
        "torchvision-rgb": ("torchvision", "read"),
        "cv2-rgb": ("opencv", "read_rgb"),
        "cv2-bgr-cvtcolor": ("opencv", "read_bgr_cvtcolor"),
    }
    if reader_id not in paths:
        raise ValueError(f"Unknown reader {reader_id}")
    module, name = paths[reader_id]
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)


def check_runtime(reader_id):
    import torch
    import torchvision

    if (
        torch.__version__.split("+")[0] != "2.10.0"
        or torchvision.__version__.split("+")[0] != "0.25.0"
    ):
        raise ValueError("dataloader requires torch==2.10.0 and torchvision==0.25.0")
    if reader_id.startswith("cv2"):
        import cv2

        flags = (
            ["IMREAD_IGNORE_ORIENTATION", "IMREAD_COLOR_RGB"]
            if reader_id == "cv2-rgb"
            else [
                "IMREAD_IGNORE_ORIENTATION",
                "IMREAD_COLOR",
                "COLOR_BGR2RGB",
                "cvtColor",
            ]
        )
        for flag in flags:
            if not hasattr(cv2, flag):
                raise ValueError(f"cv2 missing required API {flag}")
        version = cv2.__version__
    elif reader_id == "pil-rgb":
        import PIL

        version = PIL.__version__
    else:
        from torchvision.io import decode_image
        import inspect

        if "apply_exif_orientation" not in inspect.signature(decode_image).parameters:
            raise ValueError("torchvision decode_image missing orientation control")
        from torchvision.extension import _HAS_OPS

        if not _HAS_OPS or not hasattr(torch.ops.image, "decode_image"):
            raise ValueError("torchvision CPU image operations unavailable")
        version = torchvision.__version__
    get_reader(reader_id)
    return ReaderInfo(
        reader_id, _DESCRIPTIONS[READER_IDS.index(reader_id)], True, version=version
    )


def probe_reader(reader_id):
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from imgread_benchmark.dataloader.readers import _probe; _probe()",
                reader_id,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        if completed.returncode:
            raise ValueError(completed.stderr.strip() or "probe failed")
        return ReaderInfo(**json.loads(completed.stdout))
    except Exception as exc:
        return ReaderInfo(
            reader_id, _DESCRIPTIONS[READER_IDS.index(reader_id)], False, str(exc)
        )


def _probe():
    reader_id = sys.argv[1]
    try:
        result = check_runtime(reader_id)
    except Exception as exc:
        result = ReaderInfo(
            reader_id,
            _DESCRIPTIONS[READER_IDS.index(reader_id)],
            False,
            f"{type(exc).__name__}: {exc}; install the dataloader extra",
        )
    print(json.dumps(asdict(result)))


def list_readers():
    return tuple(probe_reader(reader) for reader in READER_IDS)
