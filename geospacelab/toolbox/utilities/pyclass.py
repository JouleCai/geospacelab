# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import copy
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pybasic as mybasic


def set_object_attributes(obj, *args, dict_deepcopy=True, **kwargs):
    append = kwargs.pop('append', False)
    logging = kwargs.pop('logging', True)

    append_rec = 0
    for ind, arg in enumerate(args):
        if ind % 2 == 0:
            kwargs[arg] = args[ind+1]
        else:
            continue
    for key, value in kwargs.items():
        if not hasattr(obj, key):
            if not append:
                mylog.StreamLogger.warning("Object %s: '%s' is not found in the named attributes!",
                                           mybasic.retrieve_name(obj), key)
                append_rec = 1
                continue
            else:
                if logging:
                    mylog.simpleinfo.info("Object %s: The attribute '%s' is added!", mybasic.retrieve_name(obj), key)

        if type(value) is dict and dict_deepcopy is True:
            value = copy.deepcopy(value)
        setattr(obj, key, value)

    if append_rec:
        mylog.simpleinfo.info("Object %s: To add the new attribute, set append=True", mybasic.retrieve_name(obj))


def get_object_attributes(obj):
    attrs = {}
    for name in vars(obj):
        if name.startswith("__"):
            continue
        if name.startswith("_"):
            continue
        attr = getattr(obj, name)
        if callable(attr):
            continue
        attrs[name] = attr
    return attrs


class StrBase(str):
    """
    Base class inherits from *str*. Useful when adding additional attributes to a string.
    """
    def __new__(cls, str_in):
        if issubclass(str_in.__class__, StrBase):
            obj = str_in
        else:
            obj = str.__new__(cls, str_in)
        return obj

    def config(self, logging=True, **kwargs):
        """
        A standard method used in this package to set attributes to the string.
        :param logging: If ''True'', show logging.
        :type logging: bool, default: True
        :param kwargs: Other optional keyword arguments in :meth:'~geospacelab.toolbox.utilities.pyclass.set_object_attributes'
        """
        set_object_attributes(self, append=False, logging=logging, **kwargs)



if __name__ == "__main__":
    class A(object):
        def __init__(self):
            self.a = 1
            
    a = A()
    b = A()
    set_object_attributes(a, 'b', 2, append=True)
    print(a.b)
    print(b.b)
