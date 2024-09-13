# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu


from geospacelab.datahub import FacilityModel


class SWARMFacility(FacilityModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


swarm_facility = SWARMFacility('SWARM')
swarm_facility.url = 'https://earth.esa.int/eogateway/missions/swarm/data'
swarm_facility.category = 'SWARM online database'
swarm_facility.Notes = ''
