"""Test that data loading doesn't import image libraries."""


def test_dataloader_help_has_no_heavy_imports():
    import subprocess
    import sys

    script = """
import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'torchvision', 'cv2', 'psutil'):
        raise AssertionError('heavy import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from imgread_benchmark.dataloader import DataLoaderConfig, snapshot_files
from imgread_benchmark.cli import main
main(['--version'])
try:
    main(['dataloader', '--help'])
except SystemExit as error:
    assert error.code == 0
assert not any(x in sys.modules for x in ('torch', 'torchvision', 'cv2', 'psutil'))
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
