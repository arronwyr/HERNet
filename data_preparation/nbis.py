"""Fail-fast MINDTCT invocation for synthetic supervision generation."""
from pathlib import Path
import subprocess
import tempfile
from PIL import Image

def extract_minutiae(label_path, information_path, executable):
    prefix=Path(information_path).with_suffix('')
    prefix.parent.mkdir(parents=True,exist_ok=True)
    executable=Path(executable).resolve()
    if not executable.is_file():raise FileNotFoundError(executable)
    # Use 8-bit grayscale PNG with the same 500-ppi metadata as the experiments.
    with tempfile.TemporaryDirectory(prefix='hernet_mindtct_') as temporary:
        bitmap=Path(temporary)/'label.png'
        with Image.open(label_path) as image:image.convert('L').save(bitmap,dpi=(500,500))
        result=subprocess.run([str(executable),str(bitmap),str(prefix)],capture_output=True,text=True)
        if result.returncode != 0:
            raise RuntimeError(f'MINDTCT failed ({result.returncode}): {result.stderr}')
    if not prefix.with_suffix('.xyt').exists():raise RuntimeError('MINDTCT did not produce an XYT file.')
