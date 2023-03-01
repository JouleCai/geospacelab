"""
Ovation Prime model modified from Ovation Pyme by lkilcommons: https://github.com/lkilcommons/OvationPyme
Note: this is a new version. Acknowledgement to the authors will be added after the new.
"""
import os
import datetime
from collections import OrderedDict

import numpy as np
from scipy import interpolate

# from ovationprime import ovation_utilities

# from ovationprime.ovation_utilities import robinson_auroral_conductance
# from ovationprime.ovation_utilities import brekke_moen_solar_conductance

# import geospacepy
# from geospacepy import special_datetime, sun, satplottools
import aacgmv2  # available on pip
# import apexpy
from logbook import Logger

log = Logger('OvationPyme.ovation_prime')

# Determine where this module's source file is located
# to determine where to look for the tables
src_file_dir = os.path.dirname(os.path.realpath(__file__))
ovation_datadir = os.path.join(src_file_dir, 'data')


def _check_for_old_jtype(estimator, type_of_flux):
    """Check the type of flux (2nd constructor argument) of
    a FluxEstimator or SeasonalFluxEstimator class and
    raise an extensive error to inform user that they need
    to modify their calling function, and why"""

    name = estimator.__class__.__name__
    explaination = ('Constructor interface to {} has changed'.format(name)
                    + ' now the only valid second argument values'
                    + ' (for type of flux) are "energy" or "number".\n'
                    + ' Formerly, the second argument could take values'
                    + ' which confused types of flux with types of aurora'
                    + ' (e.g. you could specify ion or electron, which'
                    + ' is a property of the auroral type (choose "ion"'
                    + ' auroral type to get ion fluxes).\n'
                    + ' If you wish to calculate average energy, you'
                    + ' will need to switch from a FluxEstimator class'
                    + ' to an AverageEnergyEstimator class')
    if type_of_flux not in ['energy', 'number']:
        raise RuntimeError('{} is not a valid fluxtype.\n{}'.format(type_of_flux,
                                                                    explaination))


class LatLocaltimeInterpolator(object):
    def __init__(self, mlat_grid, mlt_grid, var):
        self.mlat_orig = mlat_grid
        self.mlt_orig = mlt_grid
        self.zvar = var
        n_north, n_south = np.count_nonzero(self.mlat_orig > 0.), np.count_nonzero(self.mlat_orig < 0.)

        if n_south == 0.:
            self.hemisphere = 'N'
        elif n_north == 0.:
            self.hemisphere = 'S'
        else:
            raise ValueError(
                'Latitude grid contains northern (N={0}) and southern (N={1}) values.'.format(n_north, n_south) + \
                ' Can only interpolate one hemisphere at a time.')

    def interpolate(self, new_mlat_grid, new_mlt_grid, method='nearest'):
        """
        Rectangularize and Interpolate (using Linear 2D interpolation)
        """
        X0, Y0 = satplottools.latlt2cart(self.mlat_orig.flatten(), self.mlt_orig.flatten(), self.hemisphere)
        X, Y = satplottools.latlt2cart(new_mlat_grid.flatten(), new_mlt_grid.flatten(), self.hemisphere)
        interpd_zvar = interpolate.griddata((X0, Y0), self.zvar.flatten(), (X, Y), method=method, fill_value=0.)
        return interpd_zvar.reshape(new_mlat_grid.shape)


class BinCorrector(object):
    """
    We've found that often there are strange outlier bins that show up in
    OvationPyme results. This attempts to identify them by computing a numerical
    derivative around each ring of constant latitude.
    """

    def __init__(self, mlat_grid, mlt_grid):
        self.mlat_grid = mlat_grid
        self.mlats = self.mlat_grid[:, 0].flatten()
        self.mlt_grid = mlt_grid
        self.mlts = self.mlt_grid[0, :].flatten()
        self.dy_thresh = None

    def fix(self, y_grid, min_mlat=49, max_mlat=75, label=''):
        """
        Compute derivatives and attempt to identify bad bins
        Assumes mlat varies along the first dimension of the gridded location
        arrays
        """
        debug = False
        plot = False
        bad_bins = np.zeros_like(y_grid, dtype=bool)
        y_grid_corr = y_grid.copy()
        if self.dy_thresh is None:
            self.dy_thresh = 3. * np.nanstd(np.diff(y_grid.flatten()))
        wraparound = lambda x, nwrap: np.concatenate([x[-1 * (nwrap + 1):-1], x, x[:nwrap]])

        for i_mlat, mlat in enumerate(self.mlats):
            if not (np.abs(mlat) >= min_mlat and np.abs(mlat) <= max_mlat):
                if debug:
                    log.debug('MLAT ring at {0} mlat is not between'.format(mlat)
                              + ' {0} and {1}'.format(min_mlat, max_mlat)
                              + ' skipping')
                continue
            mlts_nowrap = self.mlt_grid[i_mlat, :].copy()
            mlts_nowrap[mlts_nowrap < 0] += 24
            mlts_nowrap[-1] = 23.9
            y = y_grid[i_mlat, :]
            # Wrap around first and last nwarp indicies in MLT
            # this prevents out of bounds errors in the spline/derviative
            nwrap = 4  # Pchip is cubic so order+1
            mlts = wraparound(mlts_nowrap, nwrap)
            mlts[:nwrap] -= 24.  # to keep mlt in increasing order
            mlts[-1 * nwrap:] += 24.
            y = wraparound(y, nwrap)
            # y_i = interpolate.PchipInterpolator(mlts, y)
            dy = np.diff(np.concatenate([y[:1], y]))  # compute 1st derivative of spline
            i_dy = interpolate.interp1d(mlts, dy, kind='nearest')
            mlt_mask = np.ones_like(mlts, dtype=bool)
            for i_mlt, mlt in enumerate(mlts_nowrap.flatten()):
                if np.abs(i_dy(mlt)) > self.dy_thresh:
                    bad_bins[i_mlat, i_mlt] = True
                    mlt_mask[i_mlt + nwrap] = False

            y_corr_i = interpolate.PchipInterpolator(mlts[mlt_mask], y[mlt_mask])
            y_corr = y_corr_i(mlts)
            y_grid_corr[i_mlat, :] = y_corr_i(mlts_nowrap)
            if plot:
                self.plot_single_spline(mlat, mlts, y, dy, mlt_mask, y_corr, label=label)

        return y_grid_corr

    def plot_single_spline(self, mlat, mlts, y, dy, mlt_mask, y_corr, label=''):
        import matplotlib.pyplot as plt
        f = plt.figure(figsize=(8, 6))
        ax = f.add_subplot(111)
        ax.plot(mlts, y, 'bo', label='data')
        ax.plot(mlts, dy, 'r-', label='Deriv')
        bad_bins = np.logical_not(mlt_mask)
        ax.plot(mlts, y_corr, 'g.', label='After Correction')
        ax.plot(mlts[bad_bins], y[bad_bins], 'rx',
                label='Bad@dy>{0:.1f}'.format(self.dy_thresh))
        ax.set_title('Spline fit (mlat={0:.1f})'.format(mlat))
        ax.set_xlabel('MLT')
        ax.legend()
        if not os.path.exists('/tmp/ovationpyme'):
            os.makedirs('/tmp/ovationpyme')
        f.savefig('/tmp/ovationpyme/ovationpyme_spline_{0}_{1}.png'.format(label, np.floor(mlat * 10)))
        plt.close(f)

    def __call__(self, y):
        """

        """
        return self.fix(y)


class ConductanceEstimator(object):
    """
    Implements the 'Robinson Formula'
    for estimating Pedersen and Hall height integrated
    conductivity (conducance) from
    average electron energy and
    total electron energy flux
    (assumes a Maxwellian electron energy distribution)
    """

    def __init__(self, fluxtypes=['diff']):

        # Use diffuse aurora only
        self.numflux_estimator = {}
        self.eavg_estimator = {}
        for fluxtype in fluxtypes:
            self.numflux_estimator[fluxtype] = FluxEstimator(fluxtype, 'number')
            self.eavg_estimator[fluxtype] = AverageEnergyEstimator(fluxtype)

    def get_conductance(self, dt, hemi='N', solar=True, auroral=True, background_p=None, background_h=None,
                        conductance_fluxtypes=['diff'], interp_bad_bins=True,
                        return_dF=False, return_f107=False,
                        dnflux_bad_thresh=1.0e8, deavg_bad_thresh=.3):
        """
        Compute total conductance using Robinson formula and emperical solar conductance model
        """
        log.notice(
            "Getting conductance with solar {0}, aurora {1}, fluxtypes {2}, background_ped: {3}, background_hall {4}".format(
                solar,
                auroral, conductance_fluxtypes, background_p, background_h))

        all_sigp_auroral, all_sigh_auroral = [], []
        # Create a bin interpolation corrector
        for fluxtype in conductance_fluxtypes:
            mlat_grid, mlt_grid, numflux_grid, dF = self.numflux_estimator[fluxtype].get_flux_for_time(dt, hemi=hemi,
                                                                                                       return_dF=True)
            # mlat_grid, mlt_grid, energyflux_grid = self.energyflux_estimator.get_flux_for_time(dt, hemi=hemi)
            mlat_grid, mlt_grid, eavg_grid = self.eavg_estimator[fluxtype].get_eavg_for_time(dt, hemi=hemi)

            if interp_bad_bins:
                # Clean up any extremely large bins
                fixer = BinCorrector(mlat_grid, mlt_grid)

                # Fix numflux
                fixer.dy_thresh = dnflux_bad_thresh
                numflux_grid = fixer.fix(numflux_grid, label='nflux_{0}'.format(fluxtype))

                # Fix avg energy
                fixer.dy_thresh = deavg_bad_thresh
                eavg_grid = fixer.fix(eavg_grid, label='eavg_{0}'.format(fluxtype))

                # zero out lowest latitude numflux row because it makes no sense
                # has some kind of artefact at post midnight
                bad = np.abs(mlat_grid) < 52.0
                numflux_grid[bad] = 0.

            # raise RuntimeError('Debug stop!')

            aur_conds = robinson_auroral_conductance(numflux_grid, eavg_grid)
            this_sigp_auroral, this_sigh_auroral = aur_conds
            all_sigp_auroral.append(this_sigp_auroral)
            all_sigh_auroral.append(this_sigh_auroral)

        sigp_solar, sigh_solar, f107 = self.solar_conductance(dt, mlat_grid, mlt_grid, return_f107=True)
        total_sigp_sqrd = np.zeros_like(sigp_solar)
        total_sigh_sqrd = np.zeros_like(sigh_solar)

        if solar:
            total_sigp_sqrd += sigp_solar ** 2
            total_sigh_sqrd += sigh_solar ** 2

        if auroral:
            # Sum up all contributions (sqrt of summed squares)
            for sigp_auroral, sigh_auroral in zip(all_sigp_auroral, all_sigh_auroral):
                total_sigp_sqrd += sigp_auroral ** 2
                total_sigh_sqrd += sigh_auroral ** 2
            # sigp_auroral *= 1.5
            # sigh_auroral *= 1.5

        # Now take square root to get hall and pedersen conductance
        if solar or auroral:
            sigp = np.sqrt(total_sigp_sqrd)
            sigh = np.sqrt(total_sigh_sqrd)
        else:
            # No conductance except flat background
            sigp = total_sigp_sqrd
            sigh = total_sigh_sqrd

        if background_h is not None and background_p is not None:
            # Cousins et. al. 2015, nightside artificial background of 4 Siemens
            # Ellen found this to be the background nightside conductance level which
            # best optimizes the SuperDARN ElePot AMIE ability to predict AMPERE deltaB data, and
            # the AMPERE MagPot AMIE ability to predict SuperDARN LOS V
            sigp[sigp < background_p] = background_p
            sigh[sigh < background_h] = background_h

        if return_dF and return_f107:
            return mlat_grid, mlt_grid, sigp, sigh, dF, f107
        elif return_dF:
            return mlat_grid, mlt_grid, sigp, sigh, dF
        elif return_f107:
            return mlat_grid, mlt_grid, sigp, sigh, f107
        else:
            return mlat_grid, mlt_grid, sigp, sigh

    def solar_conductance(self, dt, mlats, mlts, return_f107=False):
        """
        Estimate the solar conductance using methods from:
            Cousins, E. D. P., T. Matsuo, and A. D. Richmond (2015), Mapping
            high-latitude ionospheric electrodynamics with SuperDARN and AMPERE

            --which cites--

            Asgeir Brekke, Joran Moen, Observations of high latitude ionospheric conductances

            Maybe is not good for SZA for southern hemisphere? Don't know
            Going to use absolute value of latitude because that's what's done
            in Cousins IDL code.
        """
        # Find the closest hourly f107 value
        # to the current time to specifiy the conductance
        f107 = ovation_utilities.get_daily_f107(dt)
        if hasattr(self, '_f107'):
            log.warning(('Warning: Overriding real F107 {0}'.format(f107)
                         + 'with secret instance property _f107 {0}'.format(self._f107)
                         + 'this is for debugging and will not'
                         + 'produce accurate results for a particular date.'))
            f107 = self._f107

        # print "F10.7 = %f" % (f107)

        # Convert from magnetic to geocentric using the AACGMv2 python library
        flatmlats, flatmlts = mlats.flatten(), mlts.flatten()
        flatmlons = aacgmv2.convert_mlt(flatmlts, dt, m2a=True)
        try:
            glats, glons = aacgmv2.convert(flatmlats, flatmlons, 110. * np.ones_like(flatmlats),
                                           date=dt, a2g=True, geocentric=False)
        except AttributeError:
            # convert method was deprecated
            glats, glons, r = aacgmv2.convert_latlon_arr(flatmlats,
                                                         flatmlons,
                                                         110.,
                                                         dt,
                                                         method_code='A2G')

        sigp, sigh = brekke_moen_solar_conductance(dt, glats, glons, f107)

        sigp_unflat = sigp.reshape(mlats.shape)
        sigh_unflat = sigh.reshape(mlats.shape)

        if return_f107:
            return sigp_unflat, sigh_unflat, f107
        else:
            return sigp_unflat, sigh_unflat


class AverageEnergyEstimator(object):
    """A class which estimates average energy by estimating both
    energy and number flux
    """

    def __init__(self, atype, numflux_threshold=5.0e7):
        self.numflux_threshold = numflux_threshold
        self.numflux_estimator = FluxEstimator(atype, 'number')
        self.energyflux_estimator = FluxEstimator(atype, 'energy')

    def get_eavg_for_time(self, dt, hemi='N', return_dF=False, combine_hemispheres=True):

        kwargs = {
            'hemi': hemi,
            'combine_hemispheres': combine_hemispheres,
            'return_dF': True
        }

        grid_mlats, grid_mlts, gridnumflux, dF = self.numflux_estimator.get_flux_for_time(dt, **kwargs)
        grid_mlats, grid_mlts, gridenergyflux, dF = self.energyflux_estimator.get_flux_for_time(dt, **kwargs)

        grideavg = (gridenergyflux / 1.6e-12) / gridnumflux  # energy flux Joules->eV
        grideavg = grideavg / 1000.  # eV to keV

        # Limit to reasonable number fluxes
        n_pts = len(grideavg.flatten())
        n_low_numflux = np.count_nonzero(gridnumflux < self.numflux_threshold)
        grideavg[gridnumflux < self.numflux_threshold] = 0.
        log.debug(('Zeroed {:d}/{:d} average energies'.format(n_low_numflux, n_pts)
                   + 'with numflux below {:e}'.format(self.numflux_threshold)))

        # Limit to DMSP SSJ channels range
        n_over = np.count_nonzero(grideavg > 30)
        n_under = np.count_nonzero(grideavg < .5)
        log.debug('Zeroed {:d}/{:d} average energies over 30 keV'.format(n_over, n_pts))
        log.debug('Zeroed {:d}/{:d} average energies under .2 keV'.format(n_under, n_pts))
        grideavg[grideavg > 30.] = 30.  # Max of 30keV
        grideavg[grideavg < .2] = 0.  # Min of 1 keV

        if not return_dF:
            return grid_mlats, grid_mlts, grideavg
        else:
            return grid_mlats, grid_mlts, grideavg, dF


class FluxEstimator(object):
    """
    A class which estimates auroral flux
    based on the Ovation Prime regressions,
    at arbitrary locations and times.

    Locations are in magnetic latitude and local
    time, and are interpolated using a B-spline
    representation
    """

    def __init__(self, atype, energy_or_number, seasonal_estimators=None):
        """

        doy - int
            day of year

        atype - str, ['diff','mono','wave','ions']
            type of aurora for which to load regression coeffients

        energy_or_number - str, ['energy','number']

            Type of flux you want to estimate

        seasonal_estimators - dict, optional
            A dictionary of SeasonalFluxEstimators for seasons
            'spring','fall','summer','winter', if you
            don't want to create them
            (for efficiency across multi-day calls)

        """
        self.atype = atype  # Type of aurora

        # Check for legacy values of this argument
        _check_for_old_jtype(self, energy_or_number)

        self.energy_or_number = energy_or_number  # Type of flux

        seasons = ['spring', 'summer', 'fall', 'winter']

        if seasonal_estimators is None:
            # Make a seasonal estimator for each season with nonzero weight
            self.seasonal_flux_estimators = {season: SeasonalFluxEstimator(season, atype, energy_or_number) for season
                                             in seasons}
        else:
            # Ensure the passed seasonal estimators are approriate for this atype and jtype
            for season, estimator in seasonal_estimators.items():
                jtype_atype_ok = jtype_atype_ok and (self.jtype == estimator.jtype and self.atype == estimator.atype)
            if not jtype_atype_ok:
                raise RuntimeError(
                    'Auroral and flux type of SeasonalFluxEstimators do not match {0} and {1}!'.format(self.atype,
                                                                                                       self.jtype))

    def season_weights(self, doy):
        """
        Determines the relative weighting of the
        model coeffecients for the various seasons for a particular
        day of year (doy). Nominally, weights the seasons
        based on the difference between the doy and the peak
        of the season (solstice/equinox)

        Returns:
            a dictionary with a key for each season.
            Each value in the dicionary is a float between 0 and 1
        """
        weight = OrderedDict(winter=0.,
                             spring=0.,
                             summer=0.,
                             fall=0.)

        if doy >= 79. and doy < 171:
            weight['summer'] = 1. - (171. - doy) / 92.
            weight['spring'] = 1. - weight['summer']

        elif doy >= 171. and doy < 263.:
            weight['fall'] = 1. - (263. - doy) / 92.
            weight['summer'] = 1. - weight['fall']

        elif doy >= 263. and doy < 354.:
            weight['winter'] = 1. - (354. - doy) / 91.
            weight['fall'] = 1. - weight['winter']

        elif doy >= 354 or doy < 79:
            # For days of year > 354, subtract 365 to get negative
            # day of year values for computation
            doy0 = doy - 365. if doy >= 354 else doy
            weight['spring'] = 1. - (79. - doy0) / 90.
            weight['winter'] = 1. - weight['spring']

        return weight

    def get_season_fluxes(self, dF, weights):
        """
        Extract the flux for each season and hemisphere and
        store them in a dictionary
        Return positive latitudes, since northern and southern
        latitude/localtime grids are the same
        """
        seasonfluxesN, seasonfluxesS = OrderedDict(), OrderedDict()
        gridmlats, gridmlts = None, None
        for season, estimator in self.seasonal_flux_estimators.items():
            if weights[season] == 0.:
                continue  # Skip calculation for seasons with zero weight

            flux_outs = estimator.get_gridded_flux(dF)
            gridmlatsN, gridmltsN, gridfluxN = flux_outs[:3]
            gridmlatsS, gridmltsS, gridfluxS = flux_outs[3:]
            seasonfluxesN[season] = gridfluxN
            seasonfluxesS[season] = gridfluxS
            gridmlats = gridmlatsN
            gridmlts = gridmltsN
        return gridmlats, gridmlts, seasonfluxesN, seasonfluxesS

    def get_flux_for_time(self, dt,
                          hemi='N', return_dF=False, combine_hemispheres=True):
        """
        The weighting of the seasonal flux for the different hemispheres
        is a bit counterintuitive, but after some investigation of the flux
        patterns produced for Northern and Southern hemispheres using a
        particular SeasonalFluxEstimator (which in turn reads a particular
        season's coefficients file), it seems like the 'summer' coefficients
        file contains the northern hemisphere coefficients for Boreal Summer
        (roughly May-August) and the southern hemisphere coefficients for
        Austral Summer (roughly November-February).

        In earlier versions of this code, the season weighting was wrong,
        because the code operated on the assumption that 'summer'
        meant Boreal summer, and specified a range of dates for the data
        used to construct the coefficients.

        In the IDL version of this model, the flux produced for
        Northern and Southern hemispheres is averaged, with the following
        comment on the IDL keyword argument:

        ;n_or_s=3 for combine north and south.  In effect this is the only
        ;option that works.  The result is appropriate to the northern
        ;hemisphere.  To get a result appropriate to the southern hemisphere,
        ;call with doy = 365 - actual doy

        Combining hemispheres is probably nessecary because
        there are data gaps (particularly in the northern hemisphere dawn)
        so this is the default behavior here as well. This can be overriden
        by passing combine_hemispheres=False
        """
        doy = dt.timetuple().tm_yday

        if not combine_hemispheres:
            log.warning(('Warning: IDL version of OP2010 always combines hemispheres.'
                         + 'know what you are doing before switching this behavior'))

        if hemi == 'N':
            weights = self.season_weights(doy)
        elif hemi == 'S':
            weights = self.season_weights(365. - doy)
        else:
            raise ValueError('Invalid hemisphere {0} (use N or S)'.format(hemi))

        dF = ovation_utilities.calc_dF(dt)
        if hasattr(self, '_dF'):
            log.warning(('Warning: Overriding real Newell Coupling {0}'.format(dF)
                         + 'with secret instance property _dF {0}'.format(self._dF)
                         + 'this is for debugging and will not'
                         + 'produce accurate results for a particular date'))
            dF = self._dF

        season_fluxes_outs = self.get_season_fluxes(dF, weights)
        grid_mlats, grid_mlts, seasonfluxesN, seasonfluxesS = season_fluxes_outs

        gridflux = np.zeros_like(grid_mlats)
        for season, W in weights.items():
            if W == 0.:
                continue

            gridfluxN = seasonfluxesN[season]
            gridfluxS = seasonfluxesS[season]

            if combine_hemispheres:
                gridflux += W * (gridfluxN + gridfluxS) / 2
            elif hemi == 'N':
                gridflux += W * gridfluxN
            elif hemi == 'S':
                gridflux += W * gridfluxS

        if hemi == 'S':
            grid_mlats = -1. * grid_mlats  # by default returns positive latitudes

        if not return_dF:
            return grid_mlats, grid_mlts, gridflux
        else:
            return grid_mlats, grid_mlts, gridflux, dF


class SeasonalFluxEstimator(object):
    """
    A class to hold and caculate predictions from the regression coeffecients
    which are tabulated in the data/premodel/{season}_{atype}_*.txt
    files.

    Given a particular season, type of aurora ( one of ['diff','mono','wave'])
    and type of flux, returns
    """

    _valid_atypes = ['diff', 'mono', 'wave', 'ions']

    def __init__(self, season, atype, energy_or_number):
        """
        season - str,['winter','spring','summer','fall']
            season for which to load regression coeffients

        atype - str, ['diff','mono','wave','ions']
            type of aurora for which to load regression coeffients, ions
            are not implemented

        energy_or_number - str, ['energy','number']
            type of flux you want to estimate
        """

        nmlt = 96  # number of mag local times in arrays (resolution of 15 minutes)
        nmlat = 160  # number of mag latitudes in arrays (resolution of 1/4 of a degree (.25))
        ndF = 12  # number of coupling strength bins

        self.n_mlt_bins, self.n_mlat_bins, self.n_dF_bins = nmlt, nmlat, ndF

        self.atype = atype
        if atype not in self._valid_atypes:
            raise ValueError(('Not a valid aurora type {}.'.format(atype)
                              + 'valid values {}'.format(self._valid_atypes)))

        # Check for legacy values of this argument
        _check_for_old_jtype(self, energy_or_number)

        self.energy_or_number = energy_or_number

        # The mlat bins are orgainized like -50:-dlat:-90, 50:dlat:90
        self.mlats = np.concatenate([np.linspace(-90., -50., self.n_mlat_bins // 2)[::-1],
                                     np.linspace(50., 90., self.n_mlat_bins // 2)])

        self.mlts = np.linspace(0., 24., self.n_mlt_bins)

        # Determine file names
        file_suffix = '_n' if energy_or_number == 'number' else ''
        self.afile = os.path.join(ovation_datadir, 'premodel/{0}_{1}{2}.txt'.format(season, atype, file_suffix))
        self.pfile = os.path.join(ovation_datadir, 'premodel/{0}_prob_b_{1}.txt'.format(season, atype))
        # Defualt values of header (don't know why need yet)
        # b1 = 0.
        # b2 = 0.
        # yend = 1900
        # dend = 1
        # y0 = 1900
        # d0 = 1
        # files_done = 0
        # sf0 = 0

        with open(self.afile, 'r') as f:
            aheader = f.readline()  # y0,d0,yend,dend,files_done,sf0
            # print "Read Auroral Flux Coefficient File %s,\n Header: %s" % (self.afile,aheader)
            # Don't know if it will read from where f pointer is after reading header line
            adata = np.genfromtxt(f, max_rows=nmlat * nmlt)
            # print "First line was %s" % (str(adata[0,:]))

        # These are the coefficients for each bin which are used
        # in the predicted flux calulation for electron auroral types
        # and for ions
        self.b1a, self.b2a = np.zeros((nmlt, nmlat)), np.zeros((nmlt, nmlat))
        self.b1a.fill(np.nan)
        self.b2a.fill(np.nan)
        mlt_bin_inds, mlat_bin_inds = adata[:, 0].astype(int), adata[:, 1].astype(int)
        self.b1a[mlt_bin_inds, mlat_bin_inds] = adata[:, 2]
        self.b2a[mlt_bin_inds, mlat_bin_inds] = adata[:, 3]

        self.b1p = np.full((nmlt, nmlat), np.nan)
        self.b2p = np.full((nmlt, nmlat), np.nan)
        self.prob = np.full((nmlt, nmlat, ndF), np.nan)

        # pdata has 2 columns, b1, b2 for first 15361 rows
        # pdata has nmlat*nmlt rows (one for each positional bin)

        # Electron auroral types also include a probability in their
        # predicted flux calculations (related to the probability of
        # observing one type of aurora versus another)
        if atype in ['diff', 'mono', 'wave']:
            with open(self.pfile, 'r') as f:
                pheader = f.readline()  # y0,d0,yend,dend,files_done,sf0
                # Don't know if it will read from where f pointer is after reading header line
                pdata_b = np.genfromtxt(f, max_rows=nmlt * nmlat)  # 2 columns, b1 and b2
                # print "Shape of b1p,b2p should be nmlt*nmlat=%d, is %s" % (nmlt*nmlat,len(pdata_b[:,0]))
                pdata_p = np.genfromtxt(f, max_rows=nmlt * nmlat * ndF)  # 1 column, pval

            # in the file the probability is stored with coupling strength bin
            # varying fastest (this is Fortran indexing order)
            pdata_p_column_dFbin = pdata_p.reshape((-1, ndF), order='F')

            # mlt is first dimension
            self.b1p[mlt_bin_inds, mlat_bin_inds] = pdata_b[:, 0]
            self.b2p[mlt_bin_inds, mlat_bin_inds] = pdata_b[:, 1]
            for idF in range(ndF):
                self.prob[mlt_bin_inds, mlat_bin_inds, idF] = pdata_p_column_dFbin[:, idF]

        # IDL original read
        # readf,20,i,j,b1,b2,rF
        # ;;   b1a_all(atype, iseason,i,j) = b1
        # ;;   b2a_all(atype, iseason,i,j) = b2
        # adata has 5 columns, mlt bin number, mlat bin number, b1, b2, rF
        # adata has nmlat*nmlt rows (one for each positional bin)

    def which_dF_bin(self, dF):
        """
        Given a coupling strength value, finds the bin it falls into
        """
        dFave = 4421.  # Magic numbers!
        dFstep = dFave / 8.
        i_dFbin = np.round(dF / dFstep)
        # Range check 0 <= i_dFbin <= n_dF_bins-1
        if i_dFbin < 0 or i_dFbin > self.n_dF_bins - 1:
            i_dFbin = 0 if i_dFbin < 0 else self.n_dF_bins - 1
        return int(i_dFbin)

    def prob_estimate(self, dF, i_mlt_bin, i_mlat_bin):
        """
        Estimate probability of <something> by using tabulated
        linear regression coefficients ( from prob_b files )
        WRT coupling strength dF (which are different for each position bin)

        If p doesn't come out sensible by the initial regression,
        (i.e both regression coefficients are zero)
        then tries loading from the probability array. If the value
        in the array is zero, then estimates a value using adjacent
        coupling strength bins in the probability array
        """

        # Look up the regression coefficients
        b1, b2 = self.b1p[i_mlt_bin, i_mlat_bin], self.b2p[i_mlt_bin, i_mlat_bin]

        p = b1 + b2 * dF  # What is this the probability of?

        # range check 0<=p<=1
        if p < 0. or p > 1.:
            p = 1. if p > 1. else 0.

        if b1 == 0. and b2 == 0.:
            i_dFbin = self.which_dF_bin(dF)
            # Get the tabulated probability
            p = self.prob[i_mlt_bin, i_mlat_bin, i_dFbin]

            if p == 0.:
                # If no tabulated probability we must estimate by interpolating
                # between adjacent coupling strength bins
                i_dFbin_1 = i_dFbin - 1 if i_dFbin > 0 else i_dFbin + 2  # one dF bin before by preference, two after in extremis
                i_dFbin_2 = i_dFbin + 1 if i_dFbin < self.n_dF_bins - 1 else i_dFbin - 2  # one dF bin after by preference, two before in extremis
                p = (self.prob[i_mlt_bin, i_mlat_bin, i_dFbin_1] + self.prob[i_mlt_bin, i_mlat_bin, i_dFbin_2]) / 2.

        return p

    def estimate_auroral_flux(self, dF, i_mlt_bin, i_mlat_bin):
        """
        Does what it says on the tin,
        estimates the flux using the regression coeffecients in the 'a' files
        """
        b1, b2 = self.b1a[i_mlt_bin, i_mlat_bin], self.b2a[i_mlt_bin, i_mlat_bin]
        # There are no spectral types for ions, so there is no need
        # to weight the predicted flux by a probability
        if self.atype == 'ions':
            p = 1.
        else:
            p = self.prob_estimate(dF, i_mlt_bin, i_mlat_bin)
        # print(p, b1, b2, dF)
        flux = (b1 + b2 * dF) * p
        return self.correct_flux(flux)

    def correct_flux(self, flux):
        """
        A series of magical (unexplained, unknown) corrections to flux given a particular
        type of flux
        """
        fluxtype = self.energy_or_number

        if flux < 0.:
            flux = 0.

        if self.atype is not 'ions':
            # Electron Energy Flux
            if fluxtype == 'energy':
                if flux > 10.:
                    flux = 0.5
                elif flux > 5.:
                    flux = 5.

            # Electron Number Flux
            elif fluxtype == 'number':
                if flux > 2.0e9:
                    flux = 1.0e9
                elif flux > 2.0e10:
                    flux = 0.
        else:
            # Ion Energy Flux
            if fluxtype == 'energy':
                if flux > 2.:
                    flux = 2.
                elif flux > 4.:
                    flux = 0.25

            # Ion Number Flux
            if fluxtype == 'number':
                if flux > 1.0e8:
                    flux = 1.0e8
                elif flux > 5.0e8:
                    flux = 0.
        return flux

    def get_gridded_flux(self, dF, combined_N_and_S=False, interp_N=True):
        """
        Return the flux interpolated onto arbitary locations
        in mlats and mlts

        combined_N_and_S, bool, optional
            Average the fluxes for northern and southern hemisphere
            and use them for both hemispheres (this is what standard
            ovation prime does always I think, so I've made it default)
            The original code says that this result is appropriate for
            the northern hemisphere, and to use 365 - actual doy to
            get a basic result appropriate for the southern hemisphere

        interp_N, bool, optional
            Interpolate flux linearly for each latitude ring in the wedge
            of low coverage in northern hemisphere dawn/midnight region
        """

        fluxgridN = np.zeros((self.n_mlat_bins // 2, self.n_mlt_bins))
        fluxgridN.fill(np.nan)
        # Make grid coordinates
        mlatgridN, mltgridN = np.meshgrid(self.mlats[self.n_mlat_bins // 2:], self.mlts, indexing='ij')

        fluxgridS = np.zeros((self.n_mlat_bins // 2, self.n_mlt_bins))
        fluxgridS.fill(np.nan)
        # Make grid coordinates
        mlatgridS, mltgridS = np.meshgrid(self.mlats[:self.n_mlat_bins // 2], self.mlts, indexing='ij')
        # print(self.mlats[:self.n_mlat_bins//2])

        for i_mlt in range(self.n_mlt_bins):
            for j_mlat in range(self.n_mlat_bins // 2):
                # The mlat bins are orgainized like -50:-dlat:-90,50:dlat:90
                fluxgridN[j_mlat, i_mlt] = self.estimate_auroral_flux(dF, i_mlt, self.n_mlat_bins // 2 + j_mlat)
                fluxgridS[j_mlat, i_mlt] = self.estimate_auroral_flux(dF, i_mlt, j_mlat)

        if interp_N:
            fluxgridN, inwedge = self.interp_wedge(mlatgridN, mltgridN, fluxgridN)
            self.inwedge = inwedge

        if not combined_N_and_S:
            return mlatgridN, mltgridN, fluxgridN, mlatgridS, mltgridS, fluxgridS
        else:
            return mlatgridN, mltgridN, (fluxgridN + fluxgridS) / 2.

    def interp_wedge(self, mlatgridN, mltgridN, fluxgridN):
        """
        Interpolates across the wedge shaped data gap
        around 50 magnetic latitude and 23-4 MLT.
        Interpolation is performed individually
        across each magnetic latitude ring,
        only missing flux values are filled with the
        using the interpolant
        """
        # Constants copied verbatim from IDL code
        x_mlt_min = -1.0  # minimum MLT for interpolation [hours] --change if desired
        x_mlt_max = 4.0  # maximum MLT for interpolation [hours] --change if desired
        x_mlat_min = 49.0  # minimum MLAT for interpolation [degrees]
        # x_mlat_max=67.0
        x_mlat_max = 75.0  # maximum MLAT for interpolation [degrees] --change if desired (LMK increased this from 67->75)

        valid_interp_mlat_bins = np.logical_and(mlatgridN[:, 0] >= x_mlat_min, mlatgridN[:, 0] <= x_mlat_max).flatten()
        inwedge = np.zeros(fluxgridN.shape, dtype=bool)  # Store where we did interpolation

        for i_mlat_bin in np.flatnonzero(valid_interp_mlat_bins).tolist():
            # Technically any row in the MLT grid would do, but for consistancy use the i_mlat_bin-th one
            this_mlat = mlatgridN[i_mlat_bin, 0].copy()
            this_mlt = mltgridN[i_mlat_bin, :].copy()
            this_flux = fluxgridN[i_mlat_bin, :].copy()

            # Change from 0-24 MLT to -12 to 12 MLT, so that there is no discontiunity at midnight
            # when we interpolate
            this_mlt[this_mlt > 12.] = this_mlt[this_mlt > 12.] - 24.

            valid_interp_mlt_bins = np.logical_and(this_mlt >= x_mlt_min, this_mlt <= x_mlt_max).flatten()
            mlt_bins_missing_flux = np.logical_not(this_flux > 0.).flatten()

            interp_bins_missing_flux = np.logical_and(valid_interp_mlt_bins, mlt_bins_missing_flux)

            inwedge[i_mlat_bin, :] = interp_bins_missing_flux

            if np.count_nonzero(interp_bins_missing_flux) > 0:
                # Bins right next to missing wedge probably have bad statistics, so
                # don't include them
                interp_bins_missing_flux_inds = np.flatnonzero(interp_bins_missing_flux)
                nedge = 6
                for edge_offset in range(1, nedge + 1):
                    lower_edge_ind = interp_bins_missing_flux_inds[0] - edge_offset
                    upper_edge_ind = np.mod(interp_bins_missing_flux_inds[-1] + edge_offset,
                                            len(interp_bins_missing_flux))
                    interp_bins_missing_flux[lower_edge_ind] = interp_bins_missing_flux[
                        interp_bins_missing_flux_inds[0]]
                    interp_bins_missing_flux[upper_edge_ind] = interp_bins_missing_flux[
                        interp_bins_missing_flux_inds[-1]]

                interp_source_bins = np.flatnonzero(np.logical_not(interp_bins_missing_flux))

                # flux_interp = interpolate.PchipInterpolator(this_mlt[interp_source_bins], this_flux[interp_source_bins])
                flux_interp = interpolate.interp1d(this_mlt[interp_source_bins], this_flux[interp_source_bins],
                                                   kind='linear')
                fluxgridN[i_mlat_bin, interp_bins_missing_flux] = flux_interp(this_mlt[interp_bins_missing_flux])

                # print fluxgridN[i_mlat_bin,interp_bins_missing_flux]
                # print "For latitude %.1f, replaced %d flux bins between MLT %.1f and %.1f with interpolated flux..." % (this_mlat,
                #       np.count_nonzero(interp_bins_missing_flux),np.nanmin(this_mlt[interp_bins_missing_flux]),
                #       np.nanmax(this_mlt[interp_bins_missing_flux]))

        # notwedge = np.logical_not(inwedge)
        # crazy = np.logical_and(inwedge,fluxgridN>np.nanpercentile(fluxgridN[notwedge],90.))
        # fluxgridN[crazy]=np.nanpercentile(fluxgridN[notwedge],90.)

        return fluxgridN, inwedge
