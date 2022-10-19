import numpy as np
from geospacelab.observatory.constants import *


def calc_great_circle_distance(glat_1, glon_1, glat_2, glon_2, r=R_E, formula='Vincenty', unit='degree'):
    factor = np.pi / 180.

    if unit == 'degree':
        glat_1 = glat_1 * factor
        glon_1 = glon_1 * factor
        glat_2 = glat_2 * factor
        glon_2 = glon_2 * factor
    delta_glon = glon_2 - glon_1

    # for a distance more than several kilometers
    if formula == 'classic':
        theta = np.arccos(
            np.sin(glat_1)*np.sin(glat_2) +
            np.cos(glat_1)*np.cos(glat_2)*np.cos(np.abs(delta_glon))
            )
    elif formula == 'Vincenty':
        theta = np.arctan(np.sqrt((np.cos(glat_2)*np.sin(delta_glon))**2
                                  + ((np.cos(glat_1)*np.sin(glat_2)
                                      - np.sin(glat_1)*np.cos(glat_2)*np.cos(delta_glon))**2))
                          / (np.sin(glat_1)*np.sin(glat_2) + np.cos(glat_1)*np.cos(glat_2)*np.cos(delta_glon)))
    else:
        raise NotImplementedError

    d = r * theta
    return d
