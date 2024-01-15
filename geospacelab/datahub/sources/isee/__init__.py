# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu


from geospacelab.datahub import DatabaseModel


class ISEEDatabase(DatabaseModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


isee_database = ISEEDatabase('ISEE')
isee_database.url = 'https://stdb2.isee.nagoya-u.ac.jp/index_e.html'
isee_database.category = 'online database'
isee_database.Notes = """
The global TEC data were derived from GNSS observation data in Receiver INdependent EXchange (RINEX) format obtained from 
a lot of regional GNSS reciever network all over the world. The number of GNSS stations reached more than 8800 in April 
2019. These GNSS observation data were provided by many data providers (data provider list). If you use GNSS-TEC data 
for the purpose of scientific research, you should include the following acknowledgment text.

Global GNSS-TEC data processing has been supported by JSPS KAKENHI Grant Number 16H06286. GNSS RINEX files for the 
GNSS-TEC processing are provided from many organizations listed by the webpage
(http://stdb2.isee.nagoya-u.ac.jp/GPS/GPS-TEC/gnss_provider_list.html).
"""