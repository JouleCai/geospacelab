import datetime
import numbers
import numpy as np


class LocationMixin:

    def get_lst(self, glon=None, dts=None):
        if glon is None:
            glon = self.site.location['GEO_LON']
        if dts is None:
            dts = self['DATETIME'].flatten()

        lst = np.array(dt + datetime.timedelta(seconds=int(glon*240)) for dt in dts)
        var = self.add_variable(var_name='')
        var.value = lst[:, np.newaxis]
        var.label = 'LST'
