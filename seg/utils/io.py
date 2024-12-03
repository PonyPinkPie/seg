import natsort
import os
from .typing import Sequence


IMAGE_POSTFIX: [Sequence[str]] = ["PNG", "JPEG", "JPG", "BMP", "PPM", "TIF", "PGM", "TIFF", "BMP"]

def ls_folder(folder, postfix=None, use_sort=True):
    """
    列出输入folder下面的所有文件目录
    """
    os_sorted = natsort.os_sorted if use_sort else lambda x, *args, **kwargs: x
    if os.path.exists(folder):
        if postfix is None:
            return os_sorted([os.path.join(folder, f) for f in os.listdir(folder)])
        else:
            if isinstance(postfix, str):
                postfix = [postfix.upper()]
            else:
                postfix = [p.upper() for p in postfix]
            return os_sorted([os.path.join(folder, f) for f in os.listdir(folder) if f[f.rfind(".")+1:].upper() in postfix])
    else:
        return []