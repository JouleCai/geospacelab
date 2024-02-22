# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import h5py
import os
import madrigalWeb.madrigalWeb as madrigalweb
import geospacelab.toolbox.utilities.pybasic as pybasic
import geospacelab.toolbox.utilities.pylogging as mylog

default_madrigal_url = "http://cedar.openmadrigal.org/"

"""
Functions for the Madrigal sources
"""


def list_all_instruments(madrigal_url=default_madrigal_url, database=None):
    # list all the instruments from the madrigal sources
    # get sources info
    if database is None:
        database = madrigalweb.MadrigalData(madrigal_url)

    # List all instruments
    inst_list = database.getAllInstruments()
    mylog.simpleinfo.info("List all the instruments from the Madrigal sources:\n")
    for inst in inst_list:
        mylog.simpleinfo.info("%s: %s", str(inst.code), inst.name)


def list_experiments(instrument_code, dt_fr, dt_to, madrigal_url=default_madrigal_url, database=None, relocate=True):
    if database is None:
        database = madrigalweb.MadrigalData(madrigal_url)

    exp_list = database.getExperiments(
        instrument_code,
        dt_fr.year, dt_fr.month, dt_fr.day, dt_fr.hour, dt_fr.minute, dt_fr.second,
        dt_to.year, dt_to.month, dt_to.day, dt_to.hour, dt_to.minute, dt_to.second,
        local=0
    )

    if not list(exp_list):
        raise ValueError('Cannot find the experiments for the database! Check the input values!')

    if exp_list[0].id == -1 and relocate:
        madrigal_url = exp_list[0].madrigalUrl
        mylog.simpleinfo.info("Madrigal sources has been relocated to %s", madrigal_url)
        database = madrigalweb.MadrigalData(madrigal_url)
        exp_list = database.getExperiments(
            instrument_code,
            dt_fr.year, dt_fr.month, dt_fr.day, dt_fr.hour, dt_fr.minute, dt_fr.second,
            dt_to.year, dt_to.month, dt_to.day, dt_to.hour, dt_to.minute, dt_to.second,
            local=0
        )
    return exp_list, madrigal_url, database




"""
Functions for hdf5 files
"""


def show_hdf5_structure(filename, filepath=''):
    """
    Show madrigal hdf5 file structure in console.
    Example:
        fn = "/home/leicai/01_work/00_data/madrigal/DMSP/20151102/dms_20151102_16s1.001.hdf5"
        show_structure(fn)
    """
    with h5py.File(os.path.join(filepath, filename), 'r') as fh5:
        pybasic.dict_print_tree(fh5, value_repr=True, dict_repr=True, max_level=None)


def show_hdf5_metadata(filename, filepath='', fields=None):
    """
    Show madrigal hdf5 file metadata values.
    Example:
        fn = "/home/leicai/01_work/00_data/madrigal/DMSP/20151102/dms_20151102_16s1.001.hdf5"
        show_metadata(fn)
    """
    with h5py.File(os.path.join(filepath, filename), 'r') as fh5:
        if "metadata" in fh5.keys():
            metadata_key = "metadata"
        elif "Metadata" in fh5.keys():
            metadata_key = "Metadata"
        else:
            print("Cannot find the key either metadata or Metadata!")
            return KeyError

        keys_in = fh5[metadata_key].keys()
        if fields is None:
            keys = keys_in
        else:
            keys = fields
        print("\x1b[0;31;40m" + pybasic.retrieve_name(fh5) + "\x1b[0m")
        # for key in keys:
        #    print("\x1b[1;33;40m" + "Metadata<--" + key + ": " + "\x1b[0m")
        #    print(fh5[metadata_key][key][:])
        pybasic.dict_print_tree(fh5[metadata_key], full_value=True)


def show_hdf5_group(filename, filepath='', group_name=""):
    """
    Show madrigal hdf5 file group information.
    Example:
        fn = "/home/leicai/01_work/00_data/madrigal/DMSP/20151102/dms_20151102_16s1.001.hdf5"
        show_metadata(fn, group_name="Metadata")
    """
    with h5py.File(os.path.join(filepath, filename), 'r') as fh5:
        if group_name not in fh5.keys():
            print("Cannot find the key either metadata or Metadata!")
            return KeyError

        keys_in = fh5[group_name].keys()
        print("\x1b[0;31;40m" + pybasic.retrieve_name(fh5) + "\x1b[0m")
        # for key in keys:
        #    print("\x1b[1;33;40m" + "Metadata<--" + key + ": " + "\x1b[0m")
        #    print(fh5[metadata_key][key][:])
        pybasic.dict_print_tree(fh5[group_name], full_value=True)


if __name__ == "__main__":
    fn = "/Users/lcai/Downloads/EISCAT_2021-03-10_beata_ant@uhfa.hdf5"
    #show_hdf5_structure(fn)
    #show_hdf5_metadata(fn)
    #show_hdf5_group(fn, group_name="figures")
    list_all_instruments()


