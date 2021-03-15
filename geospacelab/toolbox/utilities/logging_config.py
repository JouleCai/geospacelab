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

