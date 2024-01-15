# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

from geospacelab.datahub import FacilityModel


class DMSPFacility(FacilityModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


dmsp_facility = DMSPFacility('DMSP')
dmsp_facility.url = ''
dmsp_facility.category = 'Madrigal online/DMSP'
dmsp_facility.Notes = ''
