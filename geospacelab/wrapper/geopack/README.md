# The geopack and Tsyganenko models in Python
**Author: Sheng Tian, Univ. of Minnesota, tianx138@umn.edu**

This python `geopack` has integrated two modules originally written in Fortran: the `geopack` and the Tsyganenko models (T89, T96, T01, and T04). The Fortran `geopack05` is available at https://ccmc.gsfc.nasa.gov/modelweb/magnetos/data-based/Geopack_2005.html and `geopack08` is available at http://geo.phys.spbu.ru/~tsyganenko/Geopack-2008.html. Their DLM in IDL is available at http://ampere.jhuapl.edu/code/idl_geopack.html. As a crucial complement to `geopack05` and `geopack08`, the Tsyganenko models are available in Fortran at https://ccmc.gsfc.nasa.gov/models/modelinfo.php?model=Tsyganenko%20Magnetic%20Field.

Test results are attached in `./test_geopack1.md` to demonstrate that the Python `geopack` returns the same outputs as the Fortran and IDL counterparts. However, invisible at the user-level, several improvements have been internally implemented:
1. The latest IGRF coefficients are used, which cover the time range from 1900 to 2025. Years beyond this range are valid inputs and the corresponding IGRF coefficients will be extrapolated, whereas the Fortran and IDL versions do not extrapolate well if at all.

2. The IGRF coefficients in the Python `geopack` are time series at a milli-second cadance, whereas the coefficients are daily in the Fortran `geopack`.

3. `igrf_gsm` is changed to a wrapper of `igrf_geo` plus the proper coordinate transforms. There are many places in the Fortran version where pages of codes are copy-pasted. Though not aesthetically pleasing, I let them live in the python version, because it requires tremendous efforts to fix them all. However, the igrf_geo is the one place that is obvious and easy to fix, so I did it.

4. All `goto` statements in the Fortran `geopack` and Tsyganenko models are eliminated.

5. A `gswgsm` is added to support the new GSW coordinate introduced in `geopack08`.


## Installation
The package requires Python pre-installed and depends on the `numpy` and `scipy` packages. I've only tested the Python `geopack` on Mac OS in Python 3.6. Performance on other platform and other versions of Python is unclear.

To install the Python `geopack` through `pip`, type `> pip3 install geopack` in the terminal.

To install the **latest** version, manually install on a Mac (and hopefully Linux):

1. Download the latest package at https://github.com/tsssss/geopack/. 
2. Unzip it, open a terminal, and `cd` to the unzipped directory
3. Install the package to Python by typing `python3 setup.py install` in the terminal


## Notes on `geopack08` and `T07d`
The Python version of `geopack` tries to be compatible with both Fortran `geopack05`  and `geopack08`. The major change of `geopack08` is a new coordinate called `GSW`, which is similar to the widely used `GSM` but more suitable to study the tail physics. To be backward compatible with `geopack05`, the Python version still uses `GSM` as the major coordinate for vectors. However, to keep updated with `geopack08`, the Python version provides a new coordinate transform function `GSWGSM`, so that users can easily switch to their favorite coordinate. A new Tsyganenko `T07d` model has been released with a new algorithm. Support for T07d is under development.


## Notes on the G and W parameters
There are two G parameters used as optional inputs to the T01 model. There definitions are in Tsyganenko (2001). Similarly, there are six W parameters used as optional inputs to the T04 model, defined in Tsyganenko (2005). The python version does not support the calculations of the G and W parameters. For users interested, here is the link for the Qin-Denton W and G parameters at https://rbsp-ect.newmexicoconsortium.org/data_pub/QinDenton/. Thanks for Dr Shawn Young for providing the references and relevant information.

Back in my mind, there are some potential ways to implement the G and W parameter. But please do understand that the package does not have any funding support. I usually do major updates during summer or winter break, when it's easier to find spare time. For users that are familiar with the G and W parameters, let me know if you have any suggestions or ideas on solutions to implement them in the package!


## Example of getting the time tag
The model needs to be updated for each new time step. Time used is the unix timestamp, which is the seconds from 1970-01-01/00:00. Here are some examples in Python to get the time from intuitive inputs.

```python
# Test for 2001-01-02/03:04:05 UT
import datetime
from dateutil import parser

# From date and time
t1 = datetime.datetime(2001,1,2,3,4,5)
t0 = datetime.datetime(1970,1,1)
ut = (t1-t0).total_seconds()
print(ut)
978404645.0

# From string, need the package dateutil
t1 = parser.parse('2001-01-02/03:04:05')
ut = (t1-t0).total_seconds()
print(ut)
978404645.0
```


## Usage

Here is a short example on how to import the package and call functions. A detailed explanation of all functions is listed in the next section.

```python
from geopack import geopack, t89

ut = 100    # 1970-01-01/00:01:40 UT.
xgsm,ygsm,zgsm = [1,2,3]
ps = geopack.recalc(ut)
b0xgsm,b0ygsm,b0zgsm = geopack.dip(xgsm,ygsm,zgsm)    		# calc dipole B in GSM.
dbxgsm,dbygsm,dbzgsm = t89.t89(2, ps, xgsm,ygsm,zgsm)       # calc T89 dB in GSM.
bxgsm,bygsm,bzgsm = [b0xgsm+dbxgsm,b0ygsm+dbygsm,b0zgsm+dbzgsm]
print(bxgsm,bygsm,bzgsm)
-539.5083883330017 -569.5906371610358 -338.8680547453352
```

And here is another way to import the package and refer to the functions.

```python
import geopack

ut = 100    # 1970-01-01/00:01:40 UT.
xgsm,ygsm,zgsm = [1,2,3]
ps = geopack.geopack.recalc(ut)
b0xgsm,b0ygsm,b0zgsm = geopack.geopack.dip(xgsm,ygsm,zgsm)
dbxgsm,dbygsm,dbzgsm = geopack.t89.t89(2, ps, xgsm,ygsm,zgsm)
print(b0xgsm,b0ygsm,b0zgsm)
-544.425907831383 -565.7731166717405 -321.43413443108597
```

Another way to import the package.

```python
import geopack.geopack as gp

ut = 100    # 1970-01-01/00:01:40 UT.
xgsm,ygsm,zgsm = [2,1,100]
ps = gp.recalc(ut)
xgsm,ygsm,zgsm = gp.geogsm(2,1,100, 1)
print(xgsm,ygsm,zgsm)
(-41.00700906453125, -19.962123759781406, 89.0221254665413)
```

To use the feature in `geopack08`, users can supply the solar wind magnetic field in GSE and express vectors in GSW

```python
from geopack import geopack

ut = 100    # 1970-01-01/00:01:40 UT.
xgsm,ygsm,zgsm = [1,2,3]
vgse = [-400,0,10]                                       # solar wind velocity in GSE.
ps = geopack.recalc(ut, *vgse)                           # init with time & SW velocity.
# or use ps = geopack.recalc(ut, vgse[0],vgse[1],vgse[2])
xgsw,ygsw,zgsw = gswgsm(xgsm,ygsm,zgsm, -1)              # convert position to GSW.
b0xgsw,b0ygsw,b0zgsw = geopack.dip_gsw(xgsw,ygsw,zgsw)   # calc dipole B in GSW.
print(b0xgsw,b0ygsw,b0zgsw)
-540.5392569443875 -560.7296994901754 -336.47913346240205
print((geopack.gswgsm(b0xgsw,b0ygsw,b0zgsw, 1)))         # dipole B in GSM.
(-544.4259078313833, -565.7731166717405, -321.4341344310859)
```



## Package Interface
The Python `geopack` follows the Python way: function parameters are all input parameters and the outputs are returned. (This is very different from the Fortran and IDL `geopack`.)

* When changing to a new time of interest

  * `recalc`. Re-calculate the dipole tilt angle (and other internal parameters) for a given time.

    ```
    Example
    ps = recalc(ut)
    ps = recalc(ut, vxgse,vygse,vzgse)
    
    Input
    ut: The given time in the universal time in second.
    vxgse,vygse,vzgse: The solar wind velocity in GSE. If they are omitted, a default value of [-400,0,0] is used so that GSM and GSW are the same.
    
    Return
    ps: Dipole tilt angle in radian (defined in GSM, not GSW).
    ```

* Get the internal model magnetic fields

  * `dip`. Calculate the internal magnetic field from the dipole model for a given position and time (The time dependence is taken care of by `recalc`), in the GSM coordinate.

    ```
    Example
    bxgsm,bygsm,bzgsm = dip(xgsm,ygsm,zgsm)
    
    Input
    xgsm,ygsm,zgsm: The given position in cartesian GSM coordinate in Re (earth radii, 1 Re = 6371.2 km).
    
    Return
    bxgsm,bygsm,bzgsm: Cartesian GSM components of the internal magnetic field in nT.
    ```

  * `dip_gsw`. Calculate the internal magnetic field from the dipole model for a given position and time (The time dependence is taken care of by `recalc`), in the GSW coordinate.

    ```
    Example
    bxgsw,bygsw,bzgsw = dip_gsw(xgsw,ygsw,zgsw)
    
    Input
    xgsw,ygsw,zgsw: The given position in cartesian GSW coordinate in Re (earth radii, 1 Re = 6371.2 km).
    
    Return
    bxgsw,bygsw,bzgsw: Cartesian GSW components of the internal magnetic field in nT.
    ```

  * `igrf_gsm`. Calculate the internal magnetic field from the IGRF model (http://www.ngdc.noaa.gov/iaga/vmod/igrf.html) for a given position and time, in the GSM coordinate.

    ```
    Example
    bxgsm,bygsm,bzgsm = igrf_gsm(xgsm,ygsm,zgsm)
    
    Input
    xgsm,ygsm,zgsm: The given position in cartesian GSM coordinate in Re (earth radii, 1 Re = 6371.2 km).
    
    Return
    bxgsm,bygsm,bzgsm: Cartesian GSM components of the internal magnetic field in nT.
    ```

  * `igrf_gsw`. Calculate the internal magnetic field from the IGRF model (http://www.ngdc.noaa.gov/iaga/vmod/igrf.html) for a given position and time, in the GSW coordinate.

    ```
    Example
    bxgsw,bygsw,bzgsw = igrf_gsw(xgsw,ygsw,zgsw)
    
    Input
    xgsw,ygsw,zgsw: The given position in cartesian GSW coordinate in Re (earth radii, 1 Re = 6371.2 km).
    
    Return
    bxgsw,bygsw,bzgsw: Cartesian GSW components of the internal magnetic field in nT.
    ```

  * `igrf_geo`. Calculate the internal magnetic field from the IGRF model (http://www.ngdc.noaa.gov/iaga/vmod/igrf.html) for a given position and time, in the GEO coordinate.

    ```
    Example
    br,btheta,bphi = igrf_gsm(r,theta,phi)
    
    Input
    r,theta,phi: The given position in spherical GEO coordinate. r is the radia distance in Re; theta is the co-latitude in radian; phi is the longitude in radian.
    
    Return
    br,btheta,bphi: Spherical GSM components of the internal magnetic field in nT. br is outward; btheta is southward; bphi is eastward.
    ```

* Get the external model magntic fields

  Four models (T89, T96, T01, and T04) developed by Dr. Tsyganenko are implemented in the package. 

  * `t89`. Calculate the external magnetic field from the T89 model for a given position and time, in the GSM coordinate.

    ```
    Example
    bxgsm,bygsm,bzgsm = t89(par, ps, xgsm,ygsm,zgsm)
    
    Input
    par: A model parameter. It is an integer (1-7) maps to the Kp index
    | par |  1   |    2    |    3    |    4    |    5    |    6    |  7   |
    | Kp  | 0,0+ | 1-,1,1+ | 2-,2,2+ | 3-,3,3+ | 4-,4,4+ | 5-,5,5+ | > 6- |
    ps: Dipole tilt angle in radian.
    xgsm,ygsm,zgsm: The given position in cartesian GSM coordinate in Re (earth radii, 1 Re = 6371.2 km).
    ```

  * `t96`. Calculate the external magnetic field from the T96 model for a given position and time, in the GSM coordinate.

    ```
    Example
    bxgsm,bygsm,bzgsm = t96(par, ps, xgsm,ygsm,zgsm)
    
    Input
    ps: Dipole tilt angle in radian.
    xgsm,ygsm,zgsm: The given position in cartesian GSM coordinate in Re (earth radii, 1 Re = 6371.2 km).
    par: A model paramter. It is a 10-element array, whose elements are (1-10)
    | par |  1   |  2  |     3-4     |   5-10   |
    | Var | Pdyn | Dst | ByIMF,BzIMF | not used |
    where Pdyn is the solar wind dynamic pressure in nPa; Dst is the Dst index in nT; ByImf,BzImf are the y and z components of the IMF (interplanetary magnetif field) in GSM.
    ```

  * `t01`. Calculate the external magnetic field from the T01 model for a given position and time, in the GSM coordinate.

    ```
    Example
    bxgsm,bygsm,bzgsm = t01(par, ps, xgsm,ygsm,zgsm)
    
    Input
    ps: Dipole tilt angle in radian.
    xgsm,ygsm,zgsm: The given position in cartesian GSM coordinate in Re (earth radii, 1 Re = 6371.2 km).
    par: A model paramter. It is a 10-element array, whose elements are (1-10)
    | par |  1   |  2  |     3-4     |  5-6  |   7-10   |
    | Var | Pdyn | Dst | ByIMF,BzIMF | G1,G2 | not used |
    where Pdyn is the solar wind dynamic pressure in nPa; Dst is the Dst index in nT; ByImf,BzImf are the y and z components of the IMF (interplanetary magnetif field) in GSM; G1,G2 are two indices defined in Tsyganenko (2001).
    
    N. A. Tsyganenko, A new data-based model of the near magnetosphere magnetic field: 1. Mathematical structure. 2. Parameterization and fitting to observations (submitted to JGR, July 2001)
    ```

  * `t04`. Calculate the external magnetic field from the T04 model for a given position and time, in the GSM coordinate.

    ```
    Example
    bxgsm,bygsm,bzgsm = t04(par, ps, xgsm,ygsm,zgsm)
    
    Input
    ps: Dipole tilt angle in radian.
    xgsm,ygsm,zgsm: The given position in cartesian GSM coordinate in Re (earth radii, 1 Re = 6371.2 km).
    par: A model paramter. It is a 10-element array, whose elements are (1-10)
    | par |  1   |  2  |     3-4     |   5-10   |
    | Var | Pdyn | Dst | ByIMF,BzIMF | W1 to W6 |
    where Pdyn is the solar wind dynamic pressure in nPa; Dst is the Dst index in nT; ByImf,BzImf are the y and z components of the IMF (interplanetary magnetif field) in GSM; W1,W2,...,W6 are six indices defined in Tsyganenko (2005).
    
    N. A. Tsyganenko and M. I. Sitnov, Modeling the dynamics of the inner magnetosphere during strong geomagnetic storms, J. Geophys. Res., v. 110 (A3), A03208, doi: 10.1029/2004JA010798, 2005.
    ```

  **Note:** All 4 models share the same interface, but the meanings of `par` are very different.

* Convert a cartesian vector among coordinates

  The supported coordinates are: GEO, GEI, MAG, GSM, GSE, and SM. They are defined in Hapgood (1992). And GSW, defined in Hones+(1986) is added in `geopack_08`. The functions for the coordinate transform are:  `geomag`, `geigeo`, `magsm`, `gsmgse`, `smgsm`, `geogsm`,`gswgsm`. They share the same interface, so they are explained together.

  ```
  Usage
  b1,b2,b3 = geomag(h1,h2,h3, flag)
  
  Example
  xmag,ymag,zmag = geomag(xgeo,ygeo,zgeo,  1)
  xgeo,ygeo,zgeo = geomag(xmag,ymag,zmag, -1)
  ...
  
  Input and Return
  h1,h2,h3: Cartesian components of a vector in "coord1"
  b1,b2,b3: Cartesian components of the vector in "coord2"
  flag: flag > 0 -- coord1 to coord2; flag < 0 -- coord2 to coord1
  ```

  In addition `geodgeo` converts a position between altitude (in km)/geodetic latitude (in rad) and geocentric distance (in km)/colatitude (in rad).

  ```
  Usage
  b1,b2 = geodgeo(h1,h2, flag)
  
  Example
  rgeo,thetageo = geodgeo(hgeod,xmugeod,  1)
  hgeod,xmugeod = geodgeo(rgeo,thetageo, -1)
  
  Input and Return
  h1,h2: Components of a vector in "coord1"
  b1,b2: Components of a vector in "coord2"
  flag: flag > 0 -- coord1 to coord2; flag < 0 -- coord2 to coord1
  ```

* Trace along model magnetic fields: `trace`

  ```
  Example
  x1gsm,y1gsm,z1gsm = trace(x0gsm,y0gsm,z0gsm, dir, rlim, r0, par, exname, inname)
  
  Input
  x0gsm,y0gsm,z0gsm: The given position in cartesian GSM coordinate in Re (earth radii, 1 Re = 6371.2 km).
  dir: Direction of tracing. dir = -1 for parallel; dir = 1 for anti-parallel.
  rlim: Maximum tracing radius in Re. Default value is 10 Re.
  r0: Minimum tracing radius in Re. Default value is 1 Re.
  inname: A string specifies the internal model, one of 'dipole','igrf'. The default value is 'igrf'.
  exname: A string specifies the external model, one of 't89','t96','t01','t04'. The default value is 't89' and its par is default to be 2.
  par: The model parameter. Its dimension and the meaning depend on the external model. Please check the interface of the models for details.
  ```

Functions do not appear in the above list are considered as internal functions. For usages of them, advanced users can check the source code of the Python `geopack`.



## References

Hapgood, M. A. (1992). Space physics coordinate transformations: A user guide. Planetary and Space Science, 40(5), 711–717. http://doi.org/10.1016/0032-0633(92)90012-D

N. A. Tsyganenko, A new data-based model of the near magnetosphere magnetic field: 1. Mathematical structure. 2. Parameterization and fitting to observations (submitted to JGR, July 2001)

N. A. Tsyganenko and M. I. Sitnov, Modeling the dynamics of the inner magnetosphere during strong geomagnetic storms, J. Geophys. Res., v. 110 (A3), A03208, doi: 10.1029/2004JA010798, 2005.
