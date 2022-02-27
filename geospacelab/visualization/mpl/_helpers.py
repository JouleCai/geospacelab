# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import numpy as np
import datetime

from geospacelab.datahub.__variable_base__ import VariableBase

def check_panel_ax(func):
    def wrapper(*args, **kwargs):
        obj = args[0]
        kwargs.setdefault('ax', None)
        if kwargs['ax'] is None:
            # kwargs['ax'] = obj.axes['major']
            kwargs['ax'] = obj.gca()
        result = func(*args, **kwargs)
        return result
    return wrapper


