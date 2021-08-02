import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
    c1 = ['#FFFFFF', '#7F7FFF',
          '#0000FF', '#1E90FF',
          '#00FF00', '#7FFF00',
          '#FFFF00', '#FF0000',
          '#7F0000', '#330000']
    mycmap = colors.LinearSegmentedColormap.from_list("jhuapl_ssj_like", c1, N=500)
    return mycmap
