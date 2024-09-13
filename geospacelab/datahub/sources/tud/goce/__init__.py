# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

from geospacelab.datahub import FacilityModel


class GOCEFacility(FacilityModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


goce_facility = GOCEFacility('GOCE')
goce_facility.url = 'https://earth.esa.int/eogateway/missions/goce'
goce_facility.category = 'GOCE online database'
goce_facility.Notes = ''
