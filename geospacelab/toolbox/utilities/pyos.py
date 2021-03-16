"""
Utilities related to the system's directories and files
"""
import os
from pathlib import Path


def path_join(*args):
    """
    A wrapper of os.path.join() returns the full directory as a str.

            Parameters:
                    args: list  # Any member (str) in args.

            Returns:
                    new_path: str # Joined directory
    """
    new_path = os.path.join(*args)
    return new_path


