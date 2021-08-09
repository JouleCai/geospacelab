
import numpy as np
import datetime
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (
    LongitudeLocator, LatitudeLocator,
    LongitudeFormatter, LatitudeFormatter)
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib.cm as cm

import geospacelab.visualization.mpl_toolbox as mpl


def test():
    import matplotlib.pyplot as plt
    dt = datetime.datetime(2012, 1, 19, 10, 0)
    p = PolarView(pole='S', lon_c=0, ut=dt, mlt_c=0)
    p.add_subplot(major=True)

    p.set_extent(boundary_style='circle')

    p.add_coastlines()
    p.add_grids()
    plt.show()
    pass


class PolarView(mpl.Panel):
    def __init__(self, lon_c=None, pole='N', ut=None, lst_c=None, boundary_lat=30., boundary_style='circle',
                 cs='AACGM', mlt_c=None, grid_lat_res=10., grid_lon_res=15., mirror_south=True,
                 proj_style='Stereographic', **kwargs):

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

        proj = getattr(ccrs, proj_style)
        self.proj = proj(central_latitude=self.lat_c, central_longitude=self.lon_c)

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
    def cs_transform(self, cs_fr=None, cs_to=None, coords=None):
        import geospacelab.cs as geo_cs

        if cs_fr == cs_to:
            return coords

        cs_class = getattr(geo_cs, cs_fr.upper())
        cs1 = cs_class(coords=coords, ut=self.ut)
        cs2 = cs1.transform(cs_to=cs_to, append_mlt=self.depend_mlt)
        if self.depend_mlt:
            lon = self._transform_mlt_to_lon(cs2.coords.mlt)
        else:
            lon = cs2.coords.lon

        cs2.coords.lon = lon
        coords = {'lat': cs2.coords.lat, 'lon': cs2.coords.lon, 'h': cs2.coords.h}
        return coords

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
            if len(x0) < 20:  # omit small islands, etc.
                continue
            x0 = np.mod(x0[::1], 360)
            y0 = y0[::1]
            x = np.append(np.append(x, x0), np.nan)
            y = np.append(np.append(y, y0), np.nan)

            # csObj = scs.SpaceCS(x, y, CS='GEO', dt=self.dt, coords_labels=['lon', 'lat'])

        coords = {'lat': y, 'lon': x, 'h': 250.}
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
            lat_fr = np.abs(-1) + np.mod(90. - np.abs(-1), lat_res)
            lat_to = 85.
            num_lats = (lat_to - lat_fr) / lat_res + 1.
            lats = np.linspace(lat_fr, lat_to, int(num_lats)) * np.sign(self.lat_c)
            ylocator = mticker.FixedLocator(lats)
        gl = self.major_ax.gridlines(crs=ccrs.PlateCarree(), color='b', linewidth=0.3, linestyle=':')
        gl.xlocator = xlocator
        gl.ylocator = ylocator

    def set_extent(self, boundary_lat=None, boundary_style=None):
        if boundary_lat is not None:
            self.boundary_lat = boundary_lat
        if boundary_style is not None:
            self.boundary_style = boundary_style
        x = np.array([270., 90., 180., 0.])
        y = np.ones(4) * self.boundary_lat
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