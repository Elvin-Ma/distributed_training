"""Auto-load local torchrun runtime patches when torchrun starts.

Python imports ``sitecustomize`` automatically during interpreter startup if
this directory is on ``PYTHONPATH``. Keep the import narrowly scoped so ordinary
Python processes are not patched accidentally.
"""

import os
import sys


def _looks_like_torchrun() -> bool:
    argv0 = os.path.basename(sys.argv[0])
    if argv0 == "torchrun":
        return True

    return any(arg == "torch.distributed.run" for arg in sys.argv[:3])


if os.environ.get("QUICK_EXIT") == "1" and _looks_like_torchrun():
    print(f"QUICK_EXIT enabled !!!")
    import torchrun_patch  # noqa: F401
