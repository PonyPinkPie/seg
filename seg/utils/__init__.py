import os
from .config import Config
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        if os.path.isdir(os.path.join(os.path.dirname(__file__), module)):
            exec(f"from .{module} import *")
            del module
        continue

    exec(f"from . {module[:-3]} import *")
    del module
