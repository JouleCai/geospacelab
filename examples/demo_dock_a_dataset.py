# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

"""
This is an example showing how to dock a dataset to the datahub.
"""

import datetime

from geospacelab.datahub import DataHub

# pre-settings
dt_fr = datetime.datetime.strptime('20210309' + '0000', '%Y%m%d%H%M')   # starting time
dt_to = datetime.datetime.strptime('20210309' + '2359', '%Y%m%d%H%M')   # stopping time
database_name = 'madrigal'      # built-in sourced database name
facility_name = 'eiscat'        # facility name

site = 'UHF'                # facility attributes required, check from the eiscat schedule page
antenna = 'UHF'
modulation = 'ant'

# create a datahub instance
dh = DataHub(dt_fr, dt_to)
# dock a dataset
ds_1 = dh.dock(datasource_contents=[database_name, 'isr', facility_name],
                      site=site, antenna=antenna, modulation=modulation, data_file_type='eiscat-hdf5')
# load data
ds_1.load_data()
# assign a variable from its own dataset to the datahub
n_e = dh.assign_variable('n_e')
T_i = dh.assign_variable('T_i')

# get the variables which have been assigned in the datahub
n_e = dh.get_variable('n_e')
T_i = dh.get_variable('T_i')
# if the variable is not assigned in the datahub, but exists in the its own dataset:
comp_O_p = dh.get_variable('comp_O_p', dataset=ds_1)     # O+ ratio
# above line is equivalent to
comp_O_p = dh.datasets[0]['comp_O_p']  # the index of the datasets appended to dh starts from 0.

# The variables, e.g., n_e and T_i, are the class Variable's instances,
# which stores the variable values, errors, and many other attributes, e.g., name, label, unit, depends, ....
# To get the value of the variable, use variable_isntance.value, e.g.,
print(n_e.value)        # return the variable's value, type: numpy.ndarray, axis 0 is always along the time, check n_e.depends.items{}
print(n_e.error)

dh.list_sourced_datasets()