import unittest
import datetime
from geopack import geopack,t89,t96,t01,t04
import collections


def approx_eq(x,y, tolerance=1e-5):
    if isinstance(x, collections.Iterable):
        err = [abs(x0-y0) < tolerance for x0,y0 in zip(x,y)]
        return not (False in err)
    else:
        return abs(x-y) < tolerance


class KnownValues(unittest.TestCase):

    test = {
        'ut': (datetime.datetime(2001,1,1,2,3,4)-datetime.datetime(1970,1,1)).total_seconds(),
        'r_theta_phi': (1.1,0.1,0.2),
        'r_gsm': [-5.1,0.3,2.8],
        'v_gse': [-400,0,10],
        'r_gsw': [-5.1650466,0.31912656,2.6759018],
        'r_gse': (-5.1, 0.9056171572514691, 2.6664316163164146),
        'r_geo': (2.8011944117533565,-2.4048913761357267,4.5066403602406275),
        'r_sm': (-2.9670092644479498,0.3,5.004683409035982),
        'r_mag': (2.298686529948157,1.8997853069109853,5.004683409035982),
        'r_gei': (-0.05919908606124502,3.691434427381819,4.5066403602406275),
        'geod': (110,2),
        'geo': (6470.4815,0.431707561),
        'geod2': (109.546387,1.14329302),
        'geo2': (6470,0.43),
        'recalc': -0.53356312486214,
        'sun': (2.296253490174557,4.899558241818756,4.915948604659666,-0.40152585209539443,0.4090854263337441),
        'igrf': (262.8292494578462,-19.305779063359893,-50.34573331501855),
        'igrf_gsw': (263.86956,-20.599686,-43.992093),
        'dip': (266.04028284777775,-20.204186166677108,-57.492114467356956),
        'dip_gsw': (267.24898,-21.492340,-51.061152),
        't89': (20.77213175686351,-0.6465547428023687,-15.071641970338984),
        't96': (61.178346985193215,-1.4611972499959456,-40.44976904223083),
        't01': (46.35361850144623,1.4399149705997756,-31.998997670712665),
        't04': (10.251787900560483,-2.894661363001944,-9.615717866328614),
        'trace_setting': [-1,10,1.1],   # dir, rlim,r0 (Re).
        'trace_t89_igrf': (-0.7218581171377333,0.03278201781940832 ,0.8293646495541395),
        'trace_t96_igrf': (-0.7289210907095749,0.035354863396548725,0.8230575125223648),
        'trace_t01_igrf': (-0.726408232521053,0.03894789557265691  ,0.8251143626438783),
        'trace_t04_igrf': (-0.7153766644654653,0.024213930362513274,0.8352541020689167),
        'mgnp_shu': (-1.0470115138905702,1.4898498476086839,13.905265244347715),
        'mgnp_t96': (-1.140826561384304 ,1.4730846293697084,13.748789874117278),
        'bzimf': -5,
        'mgnp_t96_par': [10,500],   # density (cm^-3) ,velocity (km/s).
        'ps': -0.533585131
    }
    test['par0'] = [test['ps']]+test['r_gsm']
    test['par1'] = [2] + test['par0']
    test['par2'] = [[2,-87,2,-5, 0,0]+test['par0']] + test['par0']
    test['mgnp_shu_par'] = test['mgnp_t96_par']+ [test['bzimf']]

    def test_to_known_values(self):
        """geopack should give known result with known input"""

        # new recalc, which returns dipole tilt angle in rad.
        self.assertEqual(self.test['recalc'], geopack.recalc(self.test['ut'], *self.test['v_gse']))

        # new sun, which returns 5 angles in rad.
        self.assertEqual(self.test['sun'], geopack.sun(self.test['ut']))

        # new internal models.
        self.assertTrue(approx_eq(self.test['igrf'], geopack.igrf_gsm(*self.test['r_gsm'])))
        self.assertTrue(approx_eq(self.test['dip'], geopack.dip(*self.test['r_gsm'])))
        self.assertTrue(approx_eq(self.test['igrf_gsw'], geopack.igrf_gsw(*self.test['r_gsw']), 1e-3))
        self.assertTrue(approx_eq(self.test['dip_gsw'], geopack.dip_gsw(*self.test['r_gsw']), 1e-3))

        # new external models.
        self.assertTrue(approx_eq(self.test['t89'], t89.t89(*self.test['par1'])))
        self.assertTrue(approx_eq(self.test['t96'], t96.t96(*self.test['par2'])))
        self.assertTrue(approx_eq(self.test['t01'], t01.t01(*self.test['par2'])))
        self.assertTrue(approx_eq(self.test['t04'], t04.t04(*self.test['par2'])))

        # new coord transform, which returns B in nT.
        self.assertTrue(approx_eq(self.test['r_mag'], geopack.geomag(*self.test['r_geo'], 1)))
        self.assertTrue(approx_eq(self.test['r_geo'], geopack.geomag(*self.test['r_mag'], -1)))

        self.assertTrue(approx_eq(self.test['r_geo'], geopack.geigeo(*self.test['r_gei'], 1)))
        self.assertTrue(approx_eq(self.test['r_gei'], geopack.geigeo(*self.test['r_geo'], -1)))

        self.assertTrue(approx_eq(self.test['r_sm'],  geopack.magsm (*self.test['r_mag'], 1)))
        self.assertTrue(approx_eq(self.test['r_mag'], geopack.magsm (*self.test['r_sm'], -1)))

        self.assertTrue(approx_eq(self.test['r_gse'], geopack.gsmgse(*self.test['r_gsm'], 1)))
        self.assertTrue(approx_eq(self.test['r_gsm'], geopack.gsmgse(*self.test['r_gse'], -1)))

        self.assertTrue(approx_eq(self.test['r_gsm'], geopack.smgsm (*self.test['r_sm'], 1)))
        self.assertTrue(approx_eq(self.test['r_sm'],  geopack.smgsm (*self.test['r_gsm'], -1)))

        self.assertTrue(approx_eq(self.test['r_gsm'], geopack.geogsm(*self.test['r_geo'], 1)))
        self.assertTrue(approx_eq(self.test['r_geo'], geopack.geogsm(*self.test['r_gsm'], -1)))

        self.assertTrue(approx_eq(self.test['r_gsw'], geopack.gswgsm(*self.test['r_gsm'], -1)))
        self.assertTrue(approx_eq(self.test['r_gsm'], geopack.gswgsm(*self.test['r_gsw'], 1)))

        self.assertTrue(approx_eq(self.test['geo'], geopack.geodgeo(*self.test['geod'],  1), 1e-3))
        self.assertTrue(approx_eq(self.test['geod2'], geopack.geodgeo(*self.test['geo2'], -1), 1e-3))

        # new trace.
        self.assertTrue(approx_eq(self.test['trace_t89_igrf'],
                                  geopack.trace(*self.test['r_gsm'], *self.test['trace_setting'], self.test['par1'][0], 't89', 'igrf')))
        self.assertTrue(approx_eq(self.test['trace_t96_igrf'],
                                  geopack.trace(*self.test['r_gsm'], *self.test['trace_setting'], self.test['par2'][0], 't96', 'igrf')))
        self.assertTrue(approx_eq(self.test['trace_t01_igrf'],
                                  geopack.trace(*self.test['r_gsm'], *self.test['trace_setting'], self.test['par2'][0], 't01', 'igrf')))
        self.assertTrue(approx_eq(self.test['trace_t04_igrf'],
                                  geopack.trace(*self.test['r_gsm'], *self.test['trace_setting'], self.test['par2'][0], 't04', 'igrf')))

        # new magnetopaus model.
        self.assertTrue(approx_eq(self.test['mgnp_t96'], geopack.t96_mgnp(*self.test['mgnp_t96_par'], *self.test['r_gsm'])))
        self.assertTrue(approx_eq(self.test['mgnp_shu'], geopack.shuetal_mgnp(*self.test['mgnp_shu_par'], *self.test['r_gsm'])))


if __name__ == '__main__':
    unittest.main()
