"""Calculate solar position

References
----------
- https://gml.noaa.gov/grad/solcalc/solareqns.PDF
"""

import numpy as np
import datetime

from build.lib.geospacelab import wrapper
import geospacelab.toolbox.utilities.pydatetime as dttool


def fractional_year(doy, hour, is_leap_year=False):
    """Calculate the fractional year in radians.

    Parameters
    ----------
    day_of_year : int or np.ndarray
        The day of the year (1-365 or 1-366 for leap years).
    hour : float or np.ndarray
        The hour of the day (0-23.99).

    Returns
    -------
    float or np.ndarray
        The fractional year in radians.
    """
    total_days = 365 if not is_leap_year else 366
    return 2 * np.pi * (doy - 1 + hour / 24) / total_days


def eot(doy, hour, is_leap_year=False):
    """Calculate the equation of time in minutes.

    Parameters
    ----------
    day_of_year : int or np.ndarray
        The day of the year (1-365 or 1-366 for leap years).
    hour : float or np.ndarray
        The hour of the day (0-23.99).

    Returns
    -------
    float or np.ndarray
        The equation of time in minutes.
    """
    gamma = fractional_year(doy, hour, is_leap_year)
    return 229.18 * (
            0.000075
            + 0.001868 * np.cos(gamma)
            - 0.032077 * np.sin(gamma)
            - 0.014615 * np.cos(2 * gamma)
            - 0.040849 * np.sin(2 * gamma)
    )


def solar_declination(doy, hour, is_leap_year=False):
    """Calculate the solar declination in radians.

    Parameters
    ----------
    day_of_year : int or np.ndarray
        The day of the year (1-365 or 1-366 for leap years).
    hour : float or np.ndarray
        The hour of the day (0-23.99).

    Returns
    -------
    float or np.ndarray
        The solar declination in radians.
    """
    gamma = fractional_year(doy, hour, is_leap_year)
    decl = 0.006918 - 0.399912 * np.cos(gamma) \
           + 0.070257 * np.sin(gamma) \
           - 0.006758 * np.cos(2 * gamma) \
           + 0.000907 * np.sin(2 * gamma) \
           - 0.002697 * np.cos(3 * gamma) \
           + 0.00148 * np.sin(3 * gamma)
    return decl


def solar_hour_angle(doy, hour, longitude, is_leap_year=False, in_degrees=False):
    """Calculate the solar hour angle in radians.

    Parameters
    ----------
    day_of_year : int or np.ndarray
        The day of the year (1-365 or 1-366 for leap years).
    hour : float or np.ndarray
        The hour of the day (0-23.99).
    longitude : float or np.ndarray
        The longitude in degrees.

    Returns
    -------
    float or np.ndarray
        The solar hour angle in radians.
    """
    lst = local_solar_time(doy, hour, longitude, is_leap_year)
    if in_degrees:
        return (lst - 12) * 15
    return np.radians((lst - 12) * 15)


def solar_zenith_angle(doy, hour, longitude, latitude, is_leap_year=False, in_degrees=False):
    """Calculate the solar zenith angle in radians.

    Parameters
    ----------
    day_of_year : int or np.ndarray
        The day of the year (1-365 or 1-366 for leap years).
    hour : float or np.ndarray
        The hour of the day (0-23.99).
    longitude : float or np.ndarray
        The longitude in degrees.
    latitude : float or np.ndarray
        The latitude in degrees.

    Returns
    -------
    float or np.ndarray
        The solar zenith angle in radians.
    """
    decl = solar_declination(doy, hour, is_leap_year)
    ha = solar_hour_angle(doy, hour, longitude, is_leap_year)
    lat_rad = np.radians(latitude)

    cos_zenith = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(ha)
    cos_zenith = np.clip(cos_zenith, -1, 1)  # Ensure the value is within the valid range for arccos
    zenith = np.arccos(cos_zenith)
    if in_degrees:
        return np.degrees(zenith)
    return zenith


def solar_azimuth_angle(doy, hour, longitude, latitude, is_leap_year=False, in_degrees=False):
    """Calculate the solar azimuth angle in radians.

    Parameters
    ----------
    day_of_year : int or np.ndarray
        The day of the year (1-365 or 1-366 for leap years).
    hour : float or np.ndarray
        The hour of the day (0-23.99).
    longitude : float or np.ndarray
        The longitude in degrees.
    latitude : float or np.ndarray
        The latitude in degrees.

    Returns
    -------
    float or np.ndarray
        The solar azimuth angle in radians.
    """
    decl = solar_declination(doy, hour, is_leap_year)
    ha = solar_hour_angle(doy, hour, longitude, is_leap_year)
    lat_rad = np.radians(latitude)

    sin_azimuth = -np.cos(decl) * np.sin(ha) / np.cos(solar_zenith_angle(doy, hour, longitude, latitude, is_leap_year))
    cos_azimuth = (np.sin(decl) - np.sin(lat_rad) * np.cos(
        solar_zenith_angle(doy, hour, longitude, latitude, is_leap_year))) / (np.cos(lat_rad) * np.sin(
        solar_zenith_angle(doy, hour, longitude, latitude, is_leap_year)))

    azimuth = np.arctan2(sin_azimuth, cos_azimuth)

    if in_degrees:
        return (np.degrees(azimuth) + 360) % 360  # Ensure azimuth is between 0 and 360 degrees
    return (azimuth + 2 * np.pi) % (2 * np.pi)  # Ensure azimuth is between 0 and 2*pi radians


def sunrise_solarnoon_sunset(doy, longitude, latitude, is_leap_year=False, ):
    """Calculate the sunrise, solar noon, and sunset times in local solar time.

    Parameters
    ----------
    day_of_year : int or np.ndarray
        The day of the year (1-365 or 1-366 for leap years).
    longitude : float or np.ndarray
        The longitude in degrees.
    latitude : float or np.ndarray
        The latitude in degrees.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing the sunrise, solar noon, and sunset times in local solar time (hours).
    """
    decl = solar_declination(doy, 12, is_leap_year)  # Use solar declination at solar noon for calculation
    lat_rad = np.radians(latitude)

    cos_ha = -np.tan(lat_rad) * np.tan(decl)

    if cos_ha < -1:
        # Sun is above the horizon all day (polar day)
        return (0.0, 12.0, 24.0)
    elif cos_ha > 1:
        # Sun is below the horizon all day (polar night)
        return (np.nan, np.nan, np.nan)

    ha = np.arccos(cos_ha)

    sunrise = 12 - np.degrees(ha) / 15 - (longitude / 15) - (eot(doy, 12, is_leap_year) / 60)
    sunset = 12 + np.degrees(ha) / 15 - (longitude / 15) - (eot(doy, 12, is_leap_year) / 60)
    solarnoon = 12 - (longitude / 15) - (eot(doy, 12, is_leap_year) / 60)

    return sunrise % 24, solarnoon % 24, sunset % 24


def local_solar_time(doy, hour, longitude, is_leap_year=False):
    """Calculate the local solar time (LST) in hours.

    Parameters
    ----------
    day_of_year : int or np.ndarray
        The day of the year (1-365 or 1-366 for leap years).
    hour : float or np.ndarray
        The hour of the day (0-23.99).
    longitude : float or np.ndarray
        The longitude in degrees.

    Returns
    -------
    float or np.ndarray
        The local solar time in hours.
    """
    eot_minutes = eot(doy, hour, is_leap_year)
    lst = hour + (longitude / 15) + (eot_minutes / 60)
    return lst % 24  # Ensure LST is between 0 and 24 hours


def _validate_inputs_type(input_names=['dts', 'lons']):
    """Decorator to validate input types for functions that take datetime, longitude, and latitude arrays."""

    def decorator(func):
        def wrapper(**kwargs):
            inputs = {}
            input_types = []
            for name in input_names:
                data = kwargs.get(name, None)
                input_types.append(type(data))
                if data is None:
                    raise ValueError(f"Input '{name}' is required.")
                if type(data) in [int, float, datetime.datetime]:
                    inputs[name] = np.array([data])
                elif isinstance(data, np.ndarray):
                    inputs[name] = data
                elif isinstance(data, (list, tuple)):
                    inputs[name] = np.array(data)
                else:
                    raise TypeError(f"Input '{name}' must be a numpy array, list, tuple, or a single value.")
            kwargs.update(**inputs)
            results = func(**kwargs)
            if type(results) is tuple:
                return tuple(
                    res[0] if all(t in [int, float, datetime.datetime] for t in input_types) else res for res in
                    results)
            if all(t in [int, float, datetime.datetime] for t in input_types):
                return results[0]
            return results

        return wrapper

    return decorator


def _validate_inputs_shape(input_names=['dts', 'lons']):
    """Decorator to validate input shapes for functions that take datetime, longitude, and latitude arrays."""

    def decorator(func):
        def wrapper(**kwargs):
            inputs = [kwargs.get(name, None) for name in input_names]
            input_ndims = [data.ndim for data in inputs]
            inds_dims = np.argsort(input_ndims, )[::-1]  # Sort by number of dimensions descending

            if inputs[inds_dims[0]].ndim == 1:
                input_sizes = [data.size for data in inputs]
                inds_sizes = np.argsort(input_sizes)[::-1]  # Sort by size descending
                input_names_sorted = [input_names[i] for i in inds_sizes]
                inputs = [inputs[i] for i in inds_sizes]
            else:
                inputs = [inputs[i] for i in inds_dims]
                input_names_sorted = [input_names[i] for i in inds_dims]
            input_ndims = [data.ndim for data in inputs]

            mode = kwargs.pop('input_mode', None)
            if mode is None:
                if len(set(input_ndims)) == 1 and len(set([data.shape for data in inputs])) == 1:
                    mode = 'strict'
                else:
                    mode = 'broadcast'

            if mode == 'strict':
                if len(set(input_ndims)) != 1 and len(set([data.shape for data in inputs])) != 1:
                    raise ValueError(
                        "All input arrays must have the same shape and number of dimensions for 'strict' mode.")
            elif mode == 'broadcast':
                for i in range(len(inputs) - 1):
                    data_0 = inputs[i]
                    data_1 = inputs[i + 1]
                    if data_0.ndim > 1:
                        if data_0.ndim == data_1.ndim:
                            if data_0.shape != data_1.shape:
                                raise ValueError(
                                    f"{input_names_sorted[i]} and {input_names_sorted[i + 1]} must have the same shape.")
                        elif data_1.ndim == 1:
                            if data_1.size == 1:
                                data_1 = np.full_like(data_0, data_1[0], dtype=data_1.dtype)
                                inputs[i + 1] = data_1
                            elif data_1.shape[0] == data_0.shape[0]:
                                data_1 = np.tile(data_1[:, np.newaxis], (1, data_0.shape[1]))
                                inputs[i + 1] = data_1
                            elif data_1.shape[0] == data_0.shape[1]:
                                data_1 = np.tile(data_1[np.newaxis, :], (data_0.shape[0], 1))
                                inputs[i + 1] = data_1
                            else:
                                raise ValueError(
                                    f"The dimension of {input_names_sorted[i + 1]} must match one of the dimenssions of {input_names_sorted[i]}.")
                        elif data_1.ndim == data_0.ndim - 1:
                            if data_1.shape == data_0.shape[1:]:
                                data_1 = np.tile(data_1[np.newaxis, :], (data_0.shape[0], 1))
                                inputs[i + 1] = data_1
                            elif data_1.shape == data_0.shape[:-1]:
                                data_1 = np.tile(data_1[:, np.newaxis], (1, data_0.shape[1]))
                                inputs[i + 1] = data_1
                            elif data_1.ndim == 1 and data_1.size == 1:
                                data_1 = np.full_like(data_0, data_1[0], dtype=data_1.dtype)
                                inputs[i + 1] = data_1
                            else:
                                raise ValueError(
                                    f"The shape of {input_names_sorted[i + 1]} must match the trailing dimensions of {input_names_sorted[i]}.")
                        elif data_1.ndim == data_0.ndim - 2:
                            if data_1.shape == data_0.shape[2:]:
                                data_1 = np.tile(data_1[np.newaxis, np.newaxis, :],
                                                 (data_0.shape[0], data_0.shape[1], 1))
                                inputs[i + 1] = data_1
                            elif data_1.shape == data_0.shape[:-2]:
                                data_1 = np.tile(data_1[:, :, np.newaxis], (1, 1, data_0.shape[2]))
                                inputs[i + 1] = data_1
                            elif data_1.shape == data_0.shape[1:-1]:
                                data_1 = np.tile(data_1[np.newaxis, :, np.newaxis],
                                                 (data_0.shape[0], 1, data_0.shape[2]))
                                inputs[i + 1] = data_1
                            elif data_1.ndim == 1 and data_1.size == 1:
                                data_1 = np.full_like(data_0, data_1[0], dtype=data_1.dtype)
                                inputs[i + 1] = data_1
                            else:
                                raise ValueError(
                                    f"The shape of {input_names_sorted[i + 1]} must match the trailing dimensions of {input_names_sorted[i]}.")
                        else:
                            raise ValueError(
                                f"The number of dimensions of {input_names_sorted[i + 1]} must be less than or equal to that of {input_names_sorted[i]}.")
                    elif data_0.ndim == 1:
                        if data_0.size == 1:
                            data_0 = np.full_like(data_1, data_0[0], dtype=data_0.dtype)
                            inputs[i] = data_0
                        elif data_1.size == 1:
                            data_1 = np.full_like(data_0, data_1[0], dtype=data_1.dtype)
                            inputs[i + 1] = data_1
                        else:
                            if data_0.shape[0] != data_1.shape[0]:
                                raise ValueError(
                                    f"{input_names_sorted[i]} and {input_names_sorted[i + 1]} must have the same size or one of them must be size 1.")

            shape = inputs[0].shape
            kwargs.update({name: inputs[i].flatten() for i, name in enumerate(input_names_sorted)})
            results = func(**kwargs)
            if type(results) is tuple:
                return tuple(res.reshape(shape) for res in results)
            return results.reshape(shape)

        return wrapper

    return decorator


@_validate_inputs_type(input_names=['dts', 'lons'])
@_validate_inputs_shape(input_names=['dts', 'lons'])
def convert_datetime_longitude_to_local_solar_time(dts=None, lons=None):
    """Calculate the local solar time (LST) in hours.

    Parameters
    ----------
    dts : np.ndarray of datetime
        The array of datetime values.
    lons : np.ndarray
        The array of longitudes in degrees.

    Returns
    -------
    np.ndarray
        The local solar time in hours.
    """

    # Convert datetime to day of year and hour

    doy = dttool.get_doy(dts)

    hour = np.array([dt.hour + dt.minute / 60 + dt.second / 3600 for dt in dts])

    # Calculate the equation of time
    is_leap = is_leap_year(dts[0].year)

    # Calculate the local solar time
    lst = local_solar_time(doy, hour, lons % 360., is_leap_year=is_leap)

    return lst


@_validate_inputs_type(input_names=['dts', 'lons', 'lats'])
@_validate_inputs_shape(input_names=['dts', 'lons', 'lats'])
def convert_datetime_longitude_latitude_to_solar_position(dts=None, lons=None, lats=None):
    """Calculate the solar zenith and azimuth angles.

    Parameters
    ----------
    dts : np.ndarray of datetime
        The array of datetime values.
    lons : np.ndarray
        The array of longitudes in degrees.
    lats : np.ndarray
        The array of latitudes in degrees.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing the solar zenith and azimuth angles in degrees.
    """

    # Convert datetime to day of year and hour

    doy = dttool.get_doy(dts)

    hour = np.array([dt.hour + dt.minute / 60 + dt.second / 3600 for dt in dts])

    # Calculate the equation of time
    is_leap = is_leap_year(dts[0].year)

    # Calculate the solar zenith and azimuth angles
    zenith = solar_zenith_angle(doy, hour, lons % 360., lats, is_leap_year=is_leap, in_degrees=True)
    azimuth = solar_azimuth_angle(doy, hour, lons % 360., lats, is_leap_year=is_leap, in_degrees=True)

    return zenith, azimuth


def is_leap_year(year):
    """Determine if a given year is a leap year.

    Parameters
    ----------
    year : int
        The year to check.

    Returns
    -------
    bool
        True if the year is a leap year, False otherwise.
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


if __name__ == "__main__":
    # Example usage
    dts = [datetime.datetime(2024, 6, 21, 12, 0), datetime.datetime(2024, 12, 21, 12, 0)]
    lons = 0
    lats = 0

    lst = convert_datetime_longitude_to_local_solar_time(dts=dts, lons=lons)
    zenith, azimuth = convert_datetime_longitude_latitude_to_solar_position(dts=dts, lons=lons, lats=lats)

    print("Local Solar Time (hours):", lst)
    print("Solar Zenith Angle (degrees):", zenith)
    print("Solar Azimuth Angle (degrees):", azimuth)