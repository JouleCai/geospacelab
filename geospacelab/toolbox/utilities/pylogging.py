# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import logging.config

config = {
    'version': 1,
    'formatters': {
        'simple': {
            'format': '%(message)s',
        },
        'normal': {
            'format': '%(levelname)s: %(message)s',
        },
        'full': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        }
        # ohter formatter
    },
    'handlers': {
        'console-simple': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'normal'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'logging.log',
            'level': 'DEBUG',
            'formatter': 'full'
        },
        # other handler
    },
    'loggers':{
        'StreamLogger-simple': {
            'handlers': ['console-simple'],
            'level': 'DEBUG',
        },
        'StreamLogger': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
        'FileLogger': {
            #  console Handler and file Handler
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        },
        # other Logger
    }
}

logging.config.dictConfig(config)
simpleinfo = logging.getLogger("StreamLogger-simple")
StreamLogger = logging.getLogger("StreamLogger")
FileLogger = logging.getLogger("FileLogger")

