import cartopy.mpl.geoaxes as geoaxes
import cartopy.crs as ccrs


class GeoPanel(geoaxes.GeoAxes):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PolarStereo(ccrs.Stereographic):

    def __init__(self):
        pass

class PlateCarree(ccrs.PlateCarree):

    def __init__(self, *args, **kwargs):
        super(PlateCarree, self).__init__(*args, **kwargs)

    def _as_mpl_axes(self):
        return GeoPanel, {'map_projection': self}

