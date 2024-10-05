"""
 This script is the main submission of the DTU solutions, to run it, you should
 1. Install the package dtu_hsc_solutions from this repository with `pip install -e .` from top-level.
    This installs many dependencies as well.
 2. Have a local directory called `models` containing downloaded model weights. The script assumes
    that this is placed in in the repo root path but you can also customize it with the --models-path argument
 3. Run e.g. `python3 main.py path/to/input/files path/to/output/files T1L3`
 """
from argparser import Argparse
