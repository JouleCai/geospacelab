import utilities.physical_parameters as phy
import scipy.constants as sc

class Constants(object):
    def __init__(self, unit_system="SI"):
        self.pi = sc.pi
        self.k_B = sc.k
        self.e = sc.e
        self.m_e = sc.m_e
        self.m_p = sc.m_p
        self.m_n = sc.m_n
        self.h = sc.h
        self.c = sc.c
        self.mu_0 = sc.mu_0

constants = phy.Constants()
pass