# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu


from geospacelab.datahub import DatabaseModel


class TUDDatabase(DatabaseModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


tud_database = TUDDatabase('TUDelft/Themosphere')
tud_database.url = 'http://thermosphere.tudelft.nl/'
tud_database.category = 'online database'
tud_database.Notes = """

References

If you use the data on this webpage, please cite our publications. 

Accelerometer-derived density data

G. March, J. van den IJssel, C. Siemes, P. Visser, E. Doornbos, M. Pilinski (2021) Gas-surface interactions modelling influence on satellite aerodynamics and thermosphere density. Journal of Space Weather and Space Climate, 11: 54. https://doi.org/10.1051/swsc/2021035

C. Siemes, J. de Teixeira da Encarnação, E. N. Doornbos, J. van den IJssel, J. Kraus, R. Pereštý, L. Grunwaldt, G. Apelbaum, J. Flury and P. E. Holmdahl Olsen (2016) Swarm accelerometer data processing from raw accelerations to thermospheric neutral densities. Earth, Planets and Space, 68: 92. https://doi.org/10.1186/s40623-016-0474-5

E. N. Doornbos (2012) Thermosphere density and wind determination from satellite dynamics. Dissertation, Springer Nature, Switzerland. ISBN 978-3-642-25129-0 (free PDF version)


Satellite geometry and aerodynamic models

G. March, E. N. Doornbos and P. N. M. A. Visser (2019) High-fidelity geometry models for improving the consistency of CHAMP, GRACE, GOCE and Swarm thermospheric density data sets. Advances in Space Research 63: 213–238. https://doi.org/10.1016/j.asr.2018.07.009

G. March, E. N. Doornbos and P. N. M. A. Visser (2019) CHAMP and GOCE thermospheric wind characterization with improved gas-surface interactions modelling. Advances in Space Research 64: 1225–1242. https://doi.org/10.1016/j.asr.2019.06.023


GNSS-derived density data

J. van den IJssel, E. Doornbos, E. Iorfida, G. March, C. Siemes, O. Montenbruck (2020) Thermosphere densities derived from Swarm GPS observations. Advances in Space Research. https://doi.org/10.1016/j.asr.2020.01.004

P. Visser and J. van den IJssel J (2016) Orbit determination and estimation of non-gravitational accelerations for the GOCE reentry phase. Advances in Space Research 58: 1840–1853. https://doi.org/10.1016/j.asr.2016.07.013


The original data was obtained from the following sources

CHAMP data: GFZ's Information System and Data Center
GOCE and Swarm data: ESA's Earth Online webpage
GRACE data: NASA's Physical Oceanography Distributed Active Archive Center (PO.DAAC)

"""