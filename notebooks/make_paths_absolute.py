import os
from pathlib import Path


def make_paths_relative_to_root():
    """Always use the same, absolute (relative to root) paths

    which makes moving the notebooks around easier.
    """
    top_level = Path(os.getcwd()).parent

    os.chdir(top_level)


make_paths_relative_to_root()