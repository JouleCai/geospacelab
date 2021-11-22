# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import numpy


def get_discrete_colors(num, colormap='Set1'):
    if num <= 6 and colormap is None:
        colormap = ['r', 'b', 'k', 'g', 'c', 'm']
        cs = colormap[:num]
    elif colormap == 'Set1' and num <= 9:
        cs = [plt.cm.get_cmap(colormap)(i) for i in numpy.linspace(0, 1, 9)]
        cs = cs[:num]
    elif colormap == 'tab10' and num <= 10:
        cs = [plt.cm.get_cmap(colormap)(i) for i in numpy.linspace(0, 1, 10)]
        cs = cs[:num]
    elif colormap[:-1] == 'tab20' and num <= 20:
        cs = [plt.cm.get_cmap(colormap)(i) for i in numpy.linspace(0, 1, 20)]
        cs = cs[:num]
    else:
        cs = [plt.cm.get_cmap(colormap)(i) for i in numpy.linspace(0.001, 0.999, num)]

    return cs


def get_colormap(colormap=None):
    if colormap is None:
        # return 'nipy_spectral'
        return cmap_jhuapl_ssj_like()
    # self-defined colormaps will be added
    return colormap


def cmap_jhuapl_ssj_like():
    c1 = ['#FFFFFF', '#0000FF', '#00FF00', '#FFFF00', '#FF0000', '#330000']
    c1 = ['#DFDFDF', '#7F7FFF',
          '#0000FF', '#1E90FF',
          '#00FF00', '#7FFF00',
          '#FFFF00', '#FF0000',
          '#7F0000', '#330000']
    mycmap = colors.LinearSegmentedColormap.from_list("jhuapl_ssj_like", c1, N=500)
    return mycmap


def cmap_aurora_green():
    stops = {'red': [(0.00, 0.1725, 0.1725),
                 (0.50, 0.1725, 0.1725),
                 (1.00, 0.8353, 0.8353)],

         'green': [(0.00, 0.9294, 0.9294),
                   (0.50, 0.9294, 0.9294),
                   (1.00, 0.8235, 0.8235)],

         'blue': [(0.00, 0.3843, 0.3843),
                  (0.50, 0.3843, 0.3843),
                  (1.00, 0.6549, 0.6549)],

         'alpha': [(0.00, 0.0, 0.0),
                   (0.50, 1.0, 1.0),
                   (1.00, 1.0, 1.0)]}

    return LinearSegmentedColormap('aurora', stops, N=256)


def cmap_gist_ncar_modified():
    segmentdata = {
       'red': (
          (0.0, 0.0, 0.0),
          (0.3098, 0.0, 0.0),
          (0.3725, 0.3993, 0.3993),
          (0.4235, 0.5003, 0.5003),
          (0.5333, 1.0, 1.0),
          (0.7922, 1.0, 1.0),
          (0.8471, 0.6218, 0.6218),
          (0.898, 0.9235, 0.9235),
          (1.0, 0.9235, 0.9235)
       ),
       'green': (
          (0.0, 0.0, 0.0),
          (0.051, 0.3722, 0.3722),
          (0.1059, 0.0, 0.0),
          (0.1569, 0.7202, 0.7202),
          (0.1608, 0.7537, 0.7537),
          (0.1647, 0.7752, 0.7752),
          (0.2157, 1.0, 1.0),
          (0.2588, 0.9804, 0.9804),
          (0.2706, 0.9804, 0.9804),
          (0.3176, 1.0, 1.0),
          (0.3686, 0.8081, 0.8081),
          (0.4275, 1.0, 1.0),
          (0.5216, 1.0, 1.0),
          (0.6314, 0.7292, 0.7292),
          (0.863, 0.2796, 0.2796),
          (1.0, 0.0, 0.0),
       ),
       'blue': (
          (0.0, 0.502, 0.502),
          (0.051, 0.0222, 0.0222),
          (0.1098, 1.0, 1.0),
          (0.2039, 1.0, 1.0),
          (0.2627, 0.6145, 0.6145),
          (0.3216, 0.0, 0.0),
          (0.4157, 0.0, 0.0),
          (0.4745, 0.2342, 0.2342),
          (0.5333, 0.0, 0.0),
          (0.5804, 0.0, 0.0),
          (0.6314, 0.0549, 0.0549),
          (0.6902, 0.0, 0.0),
          (0.7373, 0.0, 0.0),
          (0.7922, 0.9738, 0.9738),
          (0.8, 1.0, 1.0),
          (1.0, 1.0, 1.0),
       )
    }

    return LinearSegmentedColormap('gist_ncar_modified', segmentdata)
