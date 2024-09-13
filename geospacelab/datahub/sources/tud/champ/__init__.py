# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu


from geospacelab.datahub import FacilityModel


class CHAMPFacility(FacilityModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


champ_facility = CHAMPFacility('CHAMP')
champ_facility.url = 'https://isdc.gfz-potsdam.de/champ-isdc/access-to-the-champ-data/'
champ_facility.category = 'GOCE online database'
champ_facility.Notes = ''
