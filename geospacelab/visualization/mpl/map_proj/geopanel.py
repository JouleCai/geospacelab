# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import numpy as np
import datetime
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (
    LongitudeLocator, LatitudeLocator,
    LongitudeFormatter, LatitudeFormatter)
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib.cm as cm
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import geospacelab.visualization.mpl as mpl
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool


def test():
    import matplotlib.pyplot as plt
    dt = datetime.datetime(2012, 1, 19, 10, 0)
    p = PolarMap(pole='N', lon_c=None, ut=dt, mlt_c=0)
    p.add_subplot(major=True)

    p.set_extent(boundary_style='circle')

    p.add_coastlines()
    p.add_grids()
    plt.show()
    pass


class PolarMap(mpl.Panel):
    def __init__(self, cs='AACGM', style=None, lon_c=None, pole='N', ut=None, lst_c=None, mlt_c=None, mlon_c=None,
                 boundary_lat=30., boundary_style='circle',
                 grid_lat_res=10., grid_lon_res=15., mirror_south=True,
                 proj_type='Stereographic', **kwargs):
        if style is None:
            style = input("Specify the mapping style: lon-fixed, lst-fixed, mlon-fixed, or mlt-fixed? ")

        if style in ['lon-fixed']:
            if lon_c is None:
                raise ValueError
            lst_c = None
            mlon_c = None
            mlt_c = None

        if style in ['lst-fixed']:
            if lst_c is None:
                raise ValueError
            lon_c = None
            mlon_c = None
            mlt_c = None

        if style in ['mlon-fixed']:
            if mlon_c is None:
                raise ValueError
            lst_c = None
            lon_c = mlon_c
            mlt_c = None

        if style in ['mlt-fixed']:
            if mlt_c is None:
                raise ValueError
            lst_c = None
            lon_c = None
            mlon_c = None

        if lon_c is not None and pole == 'S':
            lon_c = lon_c + 180.

        self.lat_c = None
        self.lon_c = lon_c
        self.ut = ut
        self.boundary_lat = boundary_lat
        self.boundary_style = boundary_style
        self.grid_lat_res = grid_lat_res
        self.grid_lon_res = grid_lon_res
        self.pole = pole
        self.lst_c = lst_c
        self.cs = cs
        self.depend_mlt = False
        self.mlt_c = mlt_c
        self.mirror_south = mirror_south
        self._extent = None
        super().__init__(**kwargs)

        proj = getattr(ccrs, proj_type)
        self.proj = proj(central_latitude=self.lat_c, central_longitude=self.lon_c)

        self.set_extent()

    @staticmethod
    def _transform_mlt_to_lon(mlt):
        lon = mlt / 24. * 360.
        lon = np.mod(lon, 360.)
        return lon

    def add_subplot(self, *args, major=False, label=None, **kwargs):
        if major:
            kwargs.setdefault('projection', self.proj)
        super().add_subplot(*args, major=major, label=label, **kwargs)

    def add_axes(self, *args, major=False, label=None, **kwargs):
        if major:
            kwargs.setdefault('projection', self.proj)
        super().add_axes(*args, major=major, label=label, **kwargs)

    # def add_lands(self):
    #     import cartopy.io.shapereader as shpreader
    #
    #     resolution = '110m'
    #     shpfilename = shpreader.natural_earth(resolution=resolution,
    #                                           category='physical',
    #                                           name='coastline')
    #     reader = shpreader.Reader(shpfilename)
    #     lands = list(reader.geometries())
    #
    #     for ind1, land in enumerate(lands):
    #         land_polygons = list(land)
    #         for ind2, polygon in enumerate(land_polygons):
    #             x, y = polygon.exterior.coords.xy
    #
    def cs_transform(self, cs_fr=None, cs_to=None, coords=None, ut=None):
        import geospacelab.cs as geo_cs

        if cs_to is None:
            cs_to = self.cs
        if cs_fr == cs_to:
            return coords
        if ut is None:
            ut = self.ut
        cs_class = getattr(geo_cs, cs_fr.upper())
        cs1 = cs_class(coords=coords, ut=ut)
        cs2 = cs1(cs_to=cs_to, append_mlt=self.depend_mlt)
        if self.depend_mlt:
            lon = self._transform_mlt_to_lon(cs2.coords.mlt)
        else:
            lon = cs2.coords.lon
        cs2.coords.lon = lon
        return cs2

    def add_coastlines(self):
        import cartopy.io.shapereader as shpreader

        resolution = '110m'
        shpfilename = shpreader.natural_earth(resolution=resolution,
                                              category='physical',
                                              name='coastline')
        coastlines = list(shpreader.Reader(shpfilename).geometries())

        x = np.array([])
        y = np.array([])
        for ind, c in enumerate(coastlines[:-1]):
            # print(ind)
            # print(len(c.xy[0]))
            # if ind not in [4013, 4014]:
            #    continue
            x0 = np.array(c.xy[0])
            y0 = np.array(c.xy[1])
            # if len(x0) < 20:  # omit small islands, etc.
            #    continue
            x0 = np.mod(x0[::1], 360)
            y0 = y0[::1]
            x = np.append(np.append(x, x0), np.nan)
            y = np.append(np.append(y, y0), np.nan)

            # csObj = scs.SpaceCS(x, y, CS='GEO', dt=self.dt, coords_labels=['lon', 'lat'])

        coords = {'lat': y, 'lon': x, 'height': 250.}
        coords = self.cs_transform(cs_fr='GEO', cs_to=self.cs, coords=coords)
        x_new = coords['lon']
        y_new = coords['lat']
        # x_new, y_new = x, y
        self.major_ax.plot(x_new, y_new, transform=ccrs.Geodetic(),
                           linestyle='-', linewidth=0.5, color='#778088', zorder=100, alpha=0.6)
        # self.ax.scatter(x_new, y_new, transform=self.default_transform,
        #             marker='.', edgecolors='none', color='#C0C0C0', s=1)

        return

    def add_grids(self, lat_res=None, lon_res=None):
        if lat_res is not None:
            self.grid_lat_res = lat_res
        else:
            lat_res = self.grid_lat_res
        if lon_res is not None:
            self.grid_lon_res = lon_res
        else:
            lon_res = self.grid_lon_res

        if lon_res is None:
            xlocator = LongitudeLocator()
        else:
            num_lons = 360. // lon_res + 1.
            lons = np.linspace(-180., 180., int(num_lons))
            xlocator = mticker.FixedLocator(lons)

        if lat_res is None:
            ylocator = LatitudeLocator()
        else:
            lat_fr = np.abs(self.boundary_lat) + np.mod(90 - np.abs(self.boundary_lat), lat_res)
            lat_to = 85.
            # num_lats = (lat_to - lat_fr) / lat_res + 1.
            lats = np.arange(lat_fr, lat_to, lat_res) * np.sign(self.lat_c)
            ylocator = mticker.FixedLocator(lats)
        gl = self.major_ax.gridlines(crs=ccrs.PlateCarree(), color='b', linewidth=0.3, linestyle=':', draw_labels=False)


        gl.xlocator = xlocator
        gl.ylocator = ylocator

        #gl.xformatter = LONGITUDE_FORMATTER()
        # gl.yformatter = LATITUDE_FORMATTER()
        # for ea in gl.label_artists:
        #     if ea[1]==False:
        #         tx = ea[2]
        #         xy = tx.get_position()
        #         #print(xy)
        #
        #         tx.set_position([30, xy[1]])

    def set_extent(self, boundary_lat=None, boundary_style=None):
        if boundary_lat is not None:
            self.boundary_lat = boundary_lat
        if boundary_style is not None:
            self.boundary_style = boundary_style
        x = np.array([270., 90., 180., 0.])
        y = np.ones(4) * self.boundary_lat
        x = np.arange(0., 360., 5.)
        y = np.empty_like(x)
        y[:] = 1. * self.boundary_lat
        data = self.proj.transform_points(ccrs.PlateCarree(), x, y)
        # self.axes.plot(x, y, '.', transform=ccrs.PlateCarree())
        ext = [np.nanmin(data[:, 0]), np.nanmax(data[:, 0]), np.nanmin(data[:, 1]), np.nanmax(data[:, 1])]
        self._extent = ext
        self.major_ax.set_extent(ext, self.proj)

        self._set_boundary_style()

        self._check_mirror_south()

    def _check_mirror_south(self):
        if self.pole == 'S' and self.mirror_south:
            xlim = self.major_ax.get_xlim()
            self.major_ax.set_xlim([max(xlim), min(xlim)])

    def _set_boundary_style(self):

        style = self.boundary_style
        if style == 'square':
            return
        elif style == 'circle':
            theta = np.linspace(0, 2 * np.pi, 400)
            center = self.proj.transform_point(self.lon_c, self.lat_c, ccrs.PlateCarree())
            radius = (self._extent[1] - self._extent[0]) / 2
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            self.major_ax.set_boundary(circle, transform=self.proj)
        else:
            raise NotImplementedError

    def add_pcolormesh(self, data, coords=None, cs=None, **kwargs):
        cs_new = self.cs_transform(cs_fr=cs, coords=coords)
        self.major_ax.pcolormesh(cs_new['lon'], cs_new['lat'], data, transform=ccrs.PlateCarree(), **kwargs)

    def add_sc_trajectory(self, sc_lat, sc_lon, sc_alt, sc_dt=None, show_trajectory=True,
                          time_tick=False, time_tick_res=600., time_tick_scale=0.02,
                          time_tick_label=True, time_tick_label_format="%M:%H", time_tick_label_fontsize=8,
                          time_minor_tick=False, time_minor_tick_res=60, **kwargs):
        kwargs.setdefault('trajectory_config', {
            'linewidth': 1,
            'linestyle': '-',
            'color':    'k',
        })
        kwargs.setdefault('linewidth', 1)
        kwargs.setdefault('color', 'k')

        if self.pole == 'N':
            ind_lat = np.where(sc_lat > self.boundary_lat)[0]
        else:
            ind_lat = np.where(sc_lat < self.boundary_lat)[0]

        sc_lat = sc_lat.flatten()[ind_lat]
        sc_lon = sc_lon.flatten()[ind_lat]
        sc_alt = sc_alt.flatten()[ind_lat]
        sc_dt = sc_dt.flatten()[ind_lat]

        coords = {
            'lat': sc_lat,
            'lon': sc_lon,
            'height': sc_alt,
        }
        cs_new = self.cs_transform(cs_fr='GEO', coords=coords, ut=sc_dt)
        if show_trajectory:
            self.major_ax.plot(cs_new['lon'], cs_new['lat'], proj=ccrs.Geodetic(), **kwargs['trajectory_config'])

        if time_tick:
            data = self.proj.transform_points(ccrs.PlateCarree(), cs_new['lon'], cs_new['lat'])
            xdata = data[:, 0]
            ydata = data[:, 1]

            sectime, dt0 = dttool.convert_datetime_to_sectime(
                sc_dt, datetime.datetime(self.ut.year, self.ut.month, self.ut.day)
            )

            time_ticks = np.arange(np.floor(sectime[0] / time_tick_res) * time_tick_res,
                                   np.ceil(sectime[-1] / time_tick_res) * time_tick_res, time_tick_res)

            from scipy.interpolate import interp1d

            f = interp1d(sectime, xdata, fill_value='extrapolate')
            x_i = f(time_ticks)
            f = interp1d(sectime, ydata, fill_value='extrapolate')
            y_i = f(time_ticks)

            new_series = np.polynomial.polynomial.Polynomial.fit(xdata, ydata, deg=3)
            p = new_series.convert().coef
            x_i

            tick_length = (self._extent[1] - self._extent[0]) * time_tick_scale
            # self.major_ax.plot(x_time_ticks, y_time_ticks, **kwargs['time_tick_config'])

        if time_minor_tick:
            pass

    def add_sc_coloured_line(self):
        pass

    def add_colorbar(self, ax, im, cscale='linear', clabel=None, cticks=None, cticklabels=None, ticklabelstep=1):
        pos = ax.get_position()
        left = pos.x1 + 0.02
        bottom = pos.y0
        width = 0.02
        height = pos.y1 - pos.y0 - 0.03
        cax = self.figure.add_axes([left, bottom, width, height])
        cb = self.figure.colorbar(im, cax=cax)
        ylim = cax.get_ylim()

        cb.set_label(clabel, rotation=270, va='bottom', size='medium')
        if cticks is not None:
            cb.ax.yaxis.set_ticks(cticks)
            if cticklabels is not None:
                cb.ax.yaxis.set_ticklabels(cticklabels)
        else:
            if cscale == 'log':
                num_major_ticks = int(np.ceil(np.diff(np.log10(ylim)))) * 2
                cax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=num_major_ticks))
                n = ticklabelstep
                [l.set_visible(False) for (i, l) in enumerate(cax.yaxis.get_ticklabels()) if i % n != 0]
                # [l.set_ha('right') for (i, l) in enumerate(cax.yaxis.get_ticklabels()) if i % n != 0]
                minorlocator = mpl.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                                     numticks=12)
                cax.yaxis.set_minor_locator(minorlocator)
                cax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        cax.yaxis.set_tick_params(labelsize='x-small')
        return [cax, cb]

    @property
    def pole(self):
        return self._pole

    @pole.setter
    def pole(self, value):
        if value.upper() in ['N', 'NORTH', 'NORTHERN']:
            self._pole = 'N'
            self.lat_c = 90.
            self.boundary_lat = np.abs(self.boundary_lat)
        elif value.upper() in ['S', 'SOUTH', 'SOUTHERN']:
            self._pole = 'S'
            self.lat_c = -90.
            self.boundary_lat = - np.abs(self.boundary_lat)
        else:
            raise ValueError

    @property
    def lst_c(self):
        return self._lst_c

    @lst_c.setter
    def lst_c(self, lst):
        self._lst_c = lst
        if lst is None:
            return
        if self.pole == 'N':
            self.lon_c = (self.lst_c - (self.ut.hour + self.ut.minute / 60)) * 15.
        elif self.pole == 'S':
            self.lon_c = (self.lst_c - (self.ut.hour + self.ut.minute / 60)) * 15. + 180.

    @property
    def mlt_c(self):
        return self._mlt_c

    @mlt_c.setter
    def mlt_c(self, mlt):
        if mlt is not None:
            self.depend_mlt = True
            self.lon_c = self._transform_mlt_to_lon(mlt)
            if self.pole == 'S':
                self.lon_c = np.mod(self.lon_c+180., 360)
            if self.cs == "GEO":
                raise AttributeError('A magnetic coordinate system must be specified (Set the attribute "cs")!')
        self._mlt_c = mlt


if __name__ == "__main__":
    test()