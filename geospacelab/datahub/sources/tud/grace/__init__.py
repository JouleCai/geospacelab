# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu


from geospacelab.datahub import FacilityModel


class GraceFacility(FacilityModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


grace_facility = GraceFacility('Grace')
grace_facility.url = 'https://earth.esa.int/eogateway/missions/grace'
grace_facility.category = 'GRACE online database'
grace_facility.Notes = ''
