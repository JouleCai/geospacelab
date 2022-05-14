# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import numpy as np


def isnumeric(v):
    v_type = type(v)

    if np.issubdtype(v_type, np.integer) or np.issubdtype(v_type, np.floating):
        return True
    else:
        return False


def input_with_default(prompt, default=''):
    ip = input(prompt) or default
    return ip


def dict_set_default(dict1, *args, **kwargs):
    for ind, arg in enumerate(args):
        if ind % 2 == 0:
            kwargs[arg] = args[ind+1]
        else:
            continue
    for key, value in kwargs.items():
        dict1.setdefault(key, value)
    return dict1


def dict_print_tree(dict_in, level=0, max_level=None, value_repr=True, dict_repr=True, full_value=False):
    if level == 0:
        print("\x1b[0;31;40m" + retrieve_name(dict_in) + "\x1b[0m", end="")

    try:
        keys = dict_in.keys()
    except AttributeError:

        if value_repr:
            print("\x1b[2;37;40m" + ": " + "\x1b[0m", end="")
            print("\x1b[0;30;47m" + repr(dict_in) + "\x1b[0m")
        else:
            print("")

        if full_value:
            try:
                print(dict_in[:])
            except:
                print("Not implemented!")
        return None

    if dict_repr:
        print("\x1b[2;37;40m" + ": " + "\x1b[0m", end="")
        print("\x1b[7;34;40m" + repr(dict_in) + "\x1b[0m", end="")
    print("")
    if max_level is not None:
        if level >= max_level:
            return None

    for key in keys:
        print("\x1b[2;37;40m" + "|---" * (level + 1) + "\x1b[0m", end="")
        print("\x1b[1;33;40m" + key + "\x1b[0m", end="")
        new_dict = dict_in[key]
        dict_print_tree(new_dict, level+1, max_level=max_level,
                        value_repr=value_repr, dict_repr=dict_repr, full_value=full_value)


def retrieve_name(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    import inspect
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


def str_join(*args, separator='_', uppercase=False, lowercase=False):
    """
    Join multiple strings into one. The empty string '' will be ignored.
    """
    if uppercase and lowercase:
        raise KeyError("The keywords 'uppercase' and 'lowercase' cannot be set as True at the same time!")

    strList_new = []
    for elem in args:
        if elem == '' or elem is None:
            continue
        if uppercase:
            elem = elem.upper()
        if lowercase:
            elem = elem.lower()
        strList_new.append(elem)
    return separator.join(strList_new)


def list_flatten(l_in, l_out=None):
    if l_out is None:
        l_out = []
    for ind, elem in enumerate(l_in):
        if type(elem) == list:
            list_flatten(l_in, l_out=l_out)
        else:
            l_out.append(elem)
    return l_out
