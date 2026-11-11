"""Make the repository root importable for the test suite.

``gui`` is an application package that is not part of the installed
distribution (only ``spatialrisk`` is installed, via an editable ``.pth``).
Under the bare ``pytest`` console script, ``sys.path[0]`` is the test
directory rather than the repo root, so ``import gui...`` would raise
``ModuleNotFoundError``. Inserting the repo root here fixes that without
changing how the editable-installed ``spatialrisk`` package resolves.
"""

import os
import sys

# Tests must never attempt a real SEPAL login even if the developer's shell
# exported PYSEPAL_DEV_AUTH. solara/settings.py calls dotenv.load_dotenv(),
# which loads the nearest .env walking up from cwd, and load_dotenv does not
# override existing environment variables — this guard ensures the app defaults
# to a safe state for test isolation.
os.environ["PYSEPAL_DEV_AUTH"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
