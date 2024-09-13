# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import geospacelab.visualization.mpl.panels as panels
import geospacelab.toolbox.utilities.pybasic as basic


class GeoPanelBase(panels.PanelBase):

    def __init__(self, *args, proj_class=None, proj_config: dict = None, **kwargs):

        self.proj_class = proj_class
        proj_config = proj_config if type(proj_config) is dict or proj_config is None else self._raise_error(TypeError)
        self.projection = self.proj_class(**proj_config)
        self._extent = None
        kwargs.update(projection=self.projection)
        super().__init__(*args, **kwargs)

    def add_axes(self, *args, major=False, label=None, **kwargs):
        if major:
            kwargs.setdefault('projection', self.projection)
        ax = super().add_axes(*args, major=major, label=label, **kwargs)
        return ax

    def set_map_extent(self, boundary_latitudes, boundary_longitudes, **kwargs):
        x = boundary_longitudes.flatten()
        y = boundary_latitudes.flatten()

        data = self.projection.transform_points(ccrs.PlateCarree(), x, y)
        ext = [np.nanmin(data[:, 0]), np.nanmax(data[:, 0]), np.nanmin(data[:, 1]), np.nanmax(data[:, 1])]

        self._extent = ext
        self().set_extent(ext, self.projection)

    def set_map_boundary(self, path=None, transform=None, **kwargs):
        self().set_boundary(path, transform=transform)

    def overlay_coastlines(self, *args, **kwargs):
        cl = self().coastlines(**kwargs)
        return cl

    def overlay_gridlines(self, *args, **kwargs):
        gl = self().gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True,
            **kwargs
        )
        return gl

    def overlay_lands(self, *args, **kwargs):

        ll = self().add_feature(cfeature.LAND, **kwargs)
        return ll

    def add_title(self, x=0.5, y=1.1, title=None, **kwargs):
        kwargs.setdefault('ha', 'center')
        kwargs.setdefault('va', 'baseline')
        super().add_title(x=x, y=y, title=title, **kwargs)

    @property
    def proj_class(self):
        return self._proj_class

    @proj_class.setter
    def proj_class(self, proj):
        if proj is None:
            proj = 'Stereographic'
        if isinstance(proj, str):
            proj = getattr(ccrs, proj)
        elif issubclass(proj, ccrs.Projection):
            proj = proj
        else:
            raise TypeError

        self._proj_class = proj
