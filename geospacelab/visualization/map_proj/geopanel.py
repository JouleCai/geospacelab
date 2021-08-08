import geospacelab.visualization.mpl_toolbox as mpl
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (
    LongitudeLocator, LatitudeLocator,
    LongitudeFormatter, LatitudeFormatter)
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib.cm as cm


class PolarView(mpl.Panel):
    def __init__(self, pole='N', lon_c=None,  ut=None, lst_c=None, boundary_lat=0., boundary_style='circle',
                 coords='GEO', mlt_c=None, grid_lat_res=10., grid_lon_res=15.,
                 proj_style='Stereographic', **kwargs):

        self.lat_c = None
        self.lon_c = lon_c
        self.boundary_lat = boundary_lat
        self.boundary_style = boundary_style
        self.grid_lat_res = grid_lat_res
        self.grid_lon_res = grid_lon_res
        self.pole = pole
        self.ut = ut
        self.lst_c = lst_c
        self.coords = coords
        self.depend_mlt = False
        self.mlt_c=None
        super().__init__(**kwargs)

        proj = getattr(ccrs, proj_style)
        self.proj = proj(central_latitude=self.lat_c, central_longitude=self.lon_c)

    def __call__(self, *args, cs_fr='GEO', coords_labels=None):

        cs1 = gsl_cs.SpaceCS(args, coords=cs_fr, dt=self.ut, coords_labels=coords_labels)
        cs2 = cs1.transform(coords_to=self.coords, append_mlt=self.depend_mlt)
        if self.depend_mlt:
            lon = self._convert_mlt_to_lon(cs2.coords.mlt)
        else:
            lon = cs2.coords.lon
        return lon, cs2.coords.lat

    @staticmethod
    def _convert_mlt_to_lon(mlt):
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

    def add_coastlines(self):
        import cartopy.io.shapereader as shpreader

        resolution = '10m'
        shpfilename = shpreader.natural_earth(resolution=resolution,
                                              category='physical',
                                              name='coastline')
        coastlines = list(shpreader.Reader(shpfilename).geometries())

        x = np.array([])
        y = np.array([])
        for ind, c in enumerate(coastlines[:-1]):
            # print(len(c.xy[0]))
            # if ind not in [4013, 4014]:
            #    continue
            x0 = np.array(c.xy[0])
            y0 = np.array(c.xy[1])
            if len(x0) < 200:  # omit small islands, etc.
                continue
            x0 = x0[::30]
            y0 = y0[::30]
            x = np.append(np.append(x, x0), np.nan)
            y = np.append(np.append(y, y0), np.nan)

            # csObj = scs.SpaceCS(x, y, CS='GEO', dt=self.dt, coords_labels=['lon', 'lat'])
        x_new, y_new = self.__call__(x, y, 300., cs_fr='GEO', coords_labels=['lon', 'lat', 'alt'])

        self.major_ax.plot(x_new, y_new, transform=ccrs.Geodetic(),
                           linestyle='-', linewidth=0.3, color='#C0C0C0')
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

    def set_extent(self, boundary_lat=None):
        if boundary_lat is not None:
            self.boundary_lat = boundary_lat
        x = np.array([270., 90., 180., 0.])
        y = np.ones(4) * self.boundary_lat
        data = self.proj.transform_points(ccrs.PlateCarree(), x, y)
        # self.axes.plot(x, y, '.', transform=ccrs.PlateCarree())
        ext = [np.min(data[:, 0]), np.max(data[:, 0]), np.min(data[:, 1]), np.max(data[:, 1])]
        self.major_ax.set_extent(ext, self.proj)

    def set_boundary_style(self, style=None):
        if style is not None:
            self.boundary_style = style
        else:
            style = self.boundary_style
        if style == 'square':
            return
        elif style == 'circle':
            theta = np.linspace(0, 2 * np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            self.major_ax.set_boundary(circle, transform=self.major_ax.transAxes)
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
        if self.pole == 'N':
            self.lon_c = (24 - (self.ut.hour + self.ut.minute / 60) + self.lst_c) * 15.
        elif self.pole == 'S':
            self.lon_c = (24 - (self.ut.hour + self.ut.minute / 60) + self.lst_c) * 15. + 180.

    @property
    def mlt_c(self):
        return self._mlt_c

    @mlt_c.setter
    def mlt_c(self, mlt):
        if mlt is not None:
            self.lst_c = mlt
            self._depend_mlt = True
            if self.coords == "GEO":
                raise AttributeError('A magnetic coordinate system must be specified (Set the attribute "coords")!')
        self._mlt_c = mlt



