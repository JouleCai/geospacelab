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
        return cmap_jet_modified()
    # self-defined colormaps will be added
    return colormap


def cmap_jhuapl_ssj_like():
    c1 = ['#FFFFFF', '#0000FF', '#00FF00', '#FFFF00', '#FF0000', '#330000']
    c1 = ['#DFDFDF', '#7F7FFF',
          '#0000FF', '#1E90FF',
          '#00FF00', '#EFFF00',
          '#FFFF00', '#FF0000',
          '#7F0000', '#330000']
    mycmap = colors.LinearSegmentedColormap.from_list("jhuapl_ssj_like", c1, N=500)
    return mycmap


def cmap_aurora():
    c1 = ['#848F9E', '#6A4CA1', '#4035A3', '#395EA6', '#003300', '#006600', '#009900', '#00BB00', '#00DD00', '#00FF00', '#80FF80', '#B3FFB3']
    c1 = ['#AEAAB0', '#78519A', '#51227B', '#310073',
          '#000073', '#004373', '#005B73', '#007365',
          '#00733A', '#007300', '#009900', '#00BB00',
          '#00DD00', '#00FF00', '#80FF80', '#B3FFB3']
    # c1 = ['#7D8094', '#6F5496', '#591D99',
    #       '#33189C', '#192B9E', '#2196A1',
    #       '#1FA380', '#1DA65B', '#1EA81E',
    #       '#15BD1D', '#16C91F', '#18D921',
    #       '#1AEB24', '#1BF726', '#B3FFB3']
    mycmap = colors.LinearSegmentedColormap.from_list("jhuapl_ssj_like", c1, N=500)
    return mycmap


def cmap_for_kp():
    c1 = [
        '#193300', '#009900', '#00CC00', '#00FF00', '#B2FF66',
        '#FFFF66', '#FFFF33', '#FFFF00', '#FF8000',
        '#FF3333', '#FF0000', '#CC0000', '#990000'
    ]
    mycmap = colors.LinearSegmentedColormap.from_list("kp", c1, N=100)
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


def cmap_jet_modified():
    segmentdata = {
        'red': (
            (0/36, 0.0, 0.0),
            (9/36, 0.0, 0.0),
            (10/36, 0.2, 0.2),
            (15/36, 0.2, 0.2),
            (16/36, 0.4, 0.4),
            (17/36, 0.65, 0.65),
            (18/36, 0.85, 0.85),
            (19/36, 0.97, 0.97),
            (20/36, 1.0, 1.0),
            (25/36, 1.0, 1.0),
            (26/36, 0.9, 0.9),
            (27/36, 0.9, 0.9),
            (31/36, 0.9, 0.9),
            (32/36, 1.0, 1.0),
            (36/36, 1.0, 1.0)
        ),
        'green': (
            (0/36, 0.0, 0.0),
            (4/36, 0.0, 0.0),
            (5/36, 0.2, 0.2),
            (6/36, 0.4, 0.4),
            (7/36, 0.6, 0.6),
            (8/36, 0.9, 0.9),
            (9/36, 1.0, 1.0),
            (19/36, 1.0, 1.0),
            (20/36, 0.97, 0.97),
            (21/36, 0.8, 0.8),
            (22/36, 0.6, 0.6),
            (23/36, 0.4, 0.4),
            (24/36, 0.2, 0.2),
            (25/36, 0.0, 0.0),
            (32/36, 0.0, 0.0),
            (33/36, 0.2, 0.2),
            (34/36, 0.4, 0.4),
            (35/36, 0.6, 0.6),
            (36/36, 0.8, 0.8)
        ),
        'blue': (
            (0/36, 0.2, 0.2),
            (1/36, 0.4, 0.4),
            (2/36, 0.6, 0.6),
            (3/36, 0.8, 0.8),
            (4/36, 1.0, 1.0),
            (10/36, 1.0, 1.0),
            (11/36, 0.8, 0.8),
            (12/36, 0.6, 0.6),
            (13/36, 0.4, 0.4),
            (14/36, 0.2, 0.2),
            (15/36, 0.0, 0.0),
            (25/36, 0.0, 0.0),
            (26/36, 0.1, 0.1),
            (27/36, 0.2, 0.2),
            (28/36, 0.4, 0.4),
            (29/36, 0.6, 0.6),
            (30/36, 0.8, 0.8),
            (31/36, 1.0, 1.0),
            (36/36, 1.0, 1.0),
        )
    }
    return LinearSegmentedColormap('gist_ncar_like', segmentdata)


def cmap_jet_modified_r():
    cmap = cmap_jet_modified()
    return cmap.reversed()


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
          (0.6863, 0.2796, 0.2796),
          (0.7451, 0.0, 0.0),
          (0.7922, 0.0, 0.0),
          (0.8431, 0.2, 0.2),
          (0.898, 0.5, 0.5),
          (1.0, 0.7, 0.7)
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
          (0.8431, 1.0, 1.0),
          (0.898, 0.9341, 0.9341),
          (1.0, 0.9341, 0.9341)
       )
    }

    return LinearSegmentedColormap('gist_ncar_modified', segmentdata)


def cmap_gist_ncar_modified_r():
    cmap = cmap_gist_ncar_modified()
    return cmap.reversed()
