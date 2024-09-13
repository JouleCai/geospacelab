# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


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


