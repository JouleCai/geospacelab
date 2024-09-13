

# The Python `geopack` 

[TOC]

This python `geopack` has integrated two modules originally written in Fortran: the `geopack` and the Tsyganenko models (T89, T96, T01, and T04). The Fortran `geopack` is available at https://ccmc.gsfc.nasa.gov/modelweb/magnetos/data-based/Geopack_2005.html and its DLM in IDL is available at http://ampere.jhuapl.edu/code/idl_geopack.html. As a crucial complement to `geopack`, the Tsyganenko models are available in Fortran at https://ccmc.gsfc.nasa.gov/models/modelinfo.php?model=Tsyganenko%20Magnetic%20Field.

As the tests below will show, the Python `geopack` returns the same outputs as the Fortran and IDL counterparts. Minor differences exist but the Python results are likely to be the most accurate. This is because Fortran and IDL are using float-point numbers. Small changes are implemented: the Fortran  `recalc` and `sun` are re-written, and that `igrf_geo` and `igrf_gsm` are re-organized. These changes are explained in the next section, followed by a section containing test results.

---

## Changes in the Python `geopack`

### The Python `recalc`
The Fortran `recalc` contains a  long list of IGRF coefficients and many `goto` statements which are both unecessary. In the Python `recalc`, we load the IGRF coefficients from `igrf12coeffs.txt`, available at https://www.ngdc.noaa.gov/IAGA/vmod/igrf12coeffs.txt. By loading the coefficients automatically, `recalc` is greatly simplified. And the limitation on input time in the Fortran version is removed by smart interpolation and extrapolation.

### The Python `sun`
The Fortran `sun` contains many "magic numbers", which are now replaced by equations at known references. The blocks on calculating the solar parameters follow the notation and equations at http://aa.usno.navy.mil/faq/docs/SunApprox.php. And the blocks on calculating the Greenwich mean sidereal time (GMST) follow the notation and equations at http://aa.usno.navy.mil/faq/docs/GAST.php. This procedure should accept an array of input times, enabling a bulk coordinate transformation.

### The Python `igrf_geo` and `igrf_gsm`
The Fortran `igrf_geo` contains complex `goto` structures and is not very intuitive. I've re-written the code and added many comments. For more details, please refer to the exellent note at https://hanspeterschaub.info/Papers/UnderGradStudents/MagneticField.pdf. The Fortran `igrf_gsm` essentially repeats the whole procedure of `igrf_geo`. This is a perfect situation for code reuse. So I changed the `igrf_gsm` as a wrapper of `igrf_geo` with the proper coordinate transforms.

---

## Tests on the Python `geopack`

In this section, We test the results from the Python `geopack` against those from the Fortran `geopack`. In case people want to replicate the tests, the Appendix includes a brief instruction on running `geopack` in Fortran.

Tsyganenko models are tested first, because although they are parts of the Python `geopack`, these models run independently of the Fortran `geopack`.

The core of the Fortran `geopack` is `recalc` and `sun`, which prepare all basic information for later calculations. They should be tested first and the most. The rest of the Fortran `geopack` consists of several groups:
1. Models for the internal magnetic field (`dip`, `igrf_geo`, `igrf_gsm`)
2. Functions for coordinate transforms among (`geo`,`mag`,`sm`,`gsm`,`gse`) and the related math functions (`sphcar`,`bspcar`,`bcarsp`)
3. Functions for tracing (`trace`, `step`, `rhand`), and models for the magnetopause (`shuetal_mgnp`,`t96_mgnp`).

These parts will be tested one by one in this section.



### External magnetic fields: Tsyganenko models

The external magnetic fields are from the Tsyganenko models. These models run on their own.

The test inputs are:
* Time/dipole tilt: 2001-01-01/02:03:04 UT/`ps = -0.533585131 rad`
* Position: `(xgsm,ygsm,zgsm) = (-5.1,0.3,2.8)`
* Model parameter:
    * T89: `iopt = 2`
    * T96, T01, and T04: `par = [2., -87., 2., -5., 0., 0., ps, xgsm,ygsm,zgsm]`.

The model magnetic field (internal field is not included) `(bxgsm,bygsm,bzgsm)` are:

|Model| Version | GSM Bx (nT) | GSM By (nT)  | GSM Bz (nT) |
|---|---|---|---|---|
| T89 | Fortran | 20.7721329 | -0.646554828 | -15.0716438 |
|                      | Python  | 20.77213175686351 | -0.6465547428023687 | -15.071641970338984 |
|                      | IDL DLM | 20.772132  | -0.64655478  | -15.071642  |
| T96 | Fortran | 61.1784935 | -1.46160448  | -40.4486618 |
|                      | Python  | 61.178346985193215 | -1.4611972499959456 | -40.44976904223083  |
|                      | IDL DLM | 61.178075  | -1.4611772   | -40.448863  |
| T01 | Fortran | 46.3536224 | 1.43994296   | -32.0043602 |
|                      | Python  | 46.35361850144623 | 1.4399149705997756  | -31.998997670712665 |
|                      | IDL DLM | 46.972664  | 1.5442350    | -31.354185  |
| T04 | Fortran | 10.252864409458805 | -2.8946251063460302 | -9.6220929668066475 |
|                      | Python  | 12.009626706113062 | -2.634590598591451 | -11.964061109578982 |
|                      | IDL DLM | 12.009174  | -2.6345250   | -11.963656  |

Note 1: T04 is updated to T04c on 2020-02-07.

### `recalc` and `sun`

Since this routine is the core of `geopack` and I've made changes to it. I'll test some internal steps. The routine does the following things that will be tested.

1. Loads IGRF coefficients for a given time and applies normalization coefficients to the coefficients

The IGRF coefficients after being interpolated to the given time are:

|Version | g[1] | g[11]  | g[21] | h[2] | h[12]  | h[22] |
|---|---|---|---|---|---|---|
| Fortran |-29606.8828     | 789.080017   | 72.4200058  | 5164.88037    |-230.680008   |-17.9600010  |
| Python  |-29606.42169927 | 789.03618706 | 72.56048774 | 5164.43743875 |-230.56349752 |-17.98709929 |

The results are slightly different because the interpolation in the Fortran version is not very accurate: the given time is truncated to the start of the day. However, in the Python version, the IGRF coefficients are interpolated up to mill-second resolution.

The IGRF coefficients after normalizations are applied are:

|Version | g[1] | g[11]  | g[21] | h[2] | h[12]  | h[22] |
|---|---|---|---|---|---|---|
| Fortran |-29606.8828     | 4366.75781    | 1045.56384    | 5164.88037    |-902.678345   |-339.500122   |
| Python  |-29606.42169927 | 4366.51513798 | 1047.59204175 | 5164.43743875 |-902.22239376 |-340.01238166 |

2. Calculates parameters related to the Sun
Since I've rewritten `sun`, here is a test on its return values:

|Version | gmst (rad) | slong (rad) | srasn (rad) | sdec (rad) |
|---|---|---|---|---|
| Fortran |2.29623127        | 4.89966297        | 4.91595507        | -0.401530176         |
| Python  |2.296253490174557 | 4.899558241818756 | 4.915948604659666 | -0.40152585209539443 |

3. Prepare rotation angles for coordinate transforms
These angles will be tested when we test coordinate transforms.




### Internal magnetic fields: IGRF and dipole

The internal magnetic fields are from the IGRF and dipole models. They are in the Fortran `geopack` and depend on `recalc`.

The test inputs are:
* time/dipole tilt: 2001-01-01/02:03:04 UT/`ps = -0.533585131 rad`
* position: `(xgsm,ygsm,zgsm) = (-5.1,0.3,2.8)`

The IGRF and dipole magnetic fields are:

| Model |Version | GSM Bx (nT) | GSM By (nT) | GSM Bz (nT) |
|---|---|---|---|---|
| Dipole | Fortran |266.046814         | -20.2048283         | -57.4973526         |
|                         | Python  |266.04028284777775 | -20.204186166677108 | -57.492114467356956 |
|                         | IDL DLM |266.04064          | -20.204224          | -57.492450          |
| IGRF   | Fortran |262.836670         | -19.3072014         | -50.3504410         |
|                         | Python  |262.8292494578462  | -19.305779063359893 | -50.34573331501855  |
|                         | IDL DLM |262.82959          | -19.305938          | -50.346050          |

Note that we are testing the Python `igrf_gsm`, which is a wrapper of `igrf_geo` and uses `sphcar`, `bspcar` and `bcarsp`. Therefore the test includes the test for these functions.



### Coordinate transforms

The coordinate transforms are being tested with the following inputs:
* Time/dipole tilt: 2001-01-01/02:03:04 UT/`ps = -0.533585131 rad`
* Position: `(xgsm,ygsm,zgsm) = (-5.1,0.3,2.8) Re`
* Solar wind: `(vxgse,vygse,vzgse) = (-400,0,10) km/s` 

Here are its transforms in all the supported coordinates:

| Coord |Version | X (Re) | Y (Re) | Z (Re) |
|---|---|---|---|---|
| GSM | Input   |-5.1 | 0.3 | 2.8 |
| GEO | Fortran |2.80123758           | -2.40482044        | 4.50665092         |
|                      | Python  |2.8011944117533565   | -2.4048913761357267 | 4.5066403602406275 |
|                      | IDL DLM |2.8012418            | -2.4048176         | 4.5066502          |
| GSE | Fortran |-5.09999990          | 0.905646503        | 2.66642165         |
|                      | Python  |-5.1                 | 0.9056171572514691 | 2.6664316163164146 |
|                      | IDL DLM |-5.0999999           | 0.90564978         | 2.6664205          |
| SM  | Fortran |-2.96689892          | 0.300000012        | 5.00474834         |
|                      | Python  |-2.9670092644479498  | 0.3                | 5.004683409035982  |
|                      | IDL DLM |-2.9670018           | 0.30000001         | 5.0046877          |
| MAG | Fortran |2.29868531           | 1.89961421         | 5.00474834         |
|                      | Python  |2.298686529948157    | 1.8997853069109853 | 5.004683409035982  |
|                      | IDL DLM |2.2986287            | 1.8998436          | 5.0046877          |
| GEI | Fortran |-0.0591988564        | 3.69142103         | 4.50665092         |
|                      | Python  |-0.05919908606124502 | 3.691434427381819  | 4.5066403602406275 |
|                      | IDL DLM |-0.059239285         | 3.6914216          | 4.5066502          |
| GSW | Fortran |-5.16504717 | 0.319132447 | 2.67590141 |
| | Python |-5.1650469653520155 | 0.3191264636083484 | 2.6759013707408523 |
| | IDL DLM |-5.1650466 | 0.31912656 | 2.6759018 |

A separated test for `GEODGEO` is done for GEOD: `(h,xmu) = (110 km,2 rad)`. We get

| Coord | Version | R (km)            | Theta (rad)         |
| ----- | ------- | ----------------- | ------------------- |
| GEO   | Fortran | 6470.48145        | 0.431707561         |
|       | Python  | 6470.481107075519 | 0.43170762466397006 |

and for GEO: `(r,theta) = (6470 km,0.43 rad)`. We get

| Coord | Version | h (km)             | Xmu (rad)         |
| ----- | ------- | ------------------ | ----------------- |
| GEOD  | Fortran | 109.546387         | 1.14329302        |
|       | Python  | 109.54721490560405 | 1.143335918840163 |



### Tracing routines

A very important part of `geopack` is `trace` and its subroutines. I will only test on `trace`, since if it works, then the subroutines should all work.

The inputs are:
* Time/dipole tilt: 2001-01-01/02:03:04 UT/`ps = -0.533585131 rad`
* Position: `(xgsm,ygsm,zgsm) = (-5.1,0.3,2.8)`
* Model parameter:
    * T89: `iopt = 2`
    * T96, T01, and T04: `par = [2., -87., 2., -5., 0., 0., ps, xgsm,ygsm,zgsm]`.
* Minimum tracing distance: `r0=1.1`
* Maximum tracing distance: `rlim=10`
* Tracing direction: `dir=-1`
* Internal model: `igrf`

Here are the comparisons of the footpoint location in GSM (I didn't get the Fortran `trace` to work, so the Python and IDL results are compared):

|Model| Version | GSM X (Re) | GSM Y (Re) | GSM Z (Re) |
|---|---|---|---|---|
| T89 | Python  | -0.7218581171377333 | 0.03278201781940832  | 0.8293646495541395 |
|                      | IDL DLM | -0.72185779         | 0.032780279          | 0.82936518         |
| T96 | Python  | -0.7289210907095749 | 0.035354863396548725 | 0.8230575125223648 |
|                      | IDL DLM | -0.72891962         | 0.035352710          | 0.82305917         |
| T01 | Python  | -0.726408232521053  | 0.03894789557265691  | 0.8251143626438783 |
|                      | IDL DLM | -0.72618079         | 0.039247587          | 0.82530059         |
| T04 | Python  | -0.7153766644654653 | 0.024213930362513274 | 0.8352541020689167 |
|                      | IDL DLM | -0.71700103         | 0.025224728          | 0.83383019         |



### Magnetopause models

As the last part, we test `t96_mgnp` and `shuetal_mgnp`. The inputs are:
* Time/dipole tilt: 2001-01-01/02:03:04 UT/`ps = -0.533585131 rad`
* Position: `(xgsm,ygsm,zgsm) = (-5.1,0.3,2.8)`
* Model parameter:
    * T89: `iopt = 2`
    * T96, T01, and T04: `par = [2., -87., 2., -5., 0., 0., ps, xgsm,ygsm,zgsm]`.
* Density: `den = 10`
* Velocity: `vel = 500`
* Bz: `bz = -5`

The model magnetopause locations in GSM are:

|Model| Version | GSM X (Re) | GSM Y (Re) | GSM Z (Re) |
|---|---|---|---|---|
| T96 | Fortran | -1.14082575         | 1.47308505         | 13.7487936         |
|                      | Python  | -1.140826561384304  | 1.4730846293697084 | 13.748789874117278 |
|                      | IDL DLM | -1.1408265          | 1.4730847          | 13.748790          |
| Shu | Fortran | -1.04701376         | 1.48984993         | 13.9052658         |
|                      | Python  | -1.0470115138905702 | 1.4898498476086839 | 13.905265244347715 |
|                      | IDL DLM | -1.0470114          | 1.4898499          | 13.905265          |



---

## Summary

As a summary, the above tests demonstrate that the Python `geopack` returns same answers as the Fortran and IDL versions for given inputs. As a powerful language, Python is adopted by the science community and the Python `geopack` is created following the trend. Please contact me (tianx138@umn.edu) for bugs, comments, suggestions, etc. Thansk!



---

## Appendix

### A Fortran `hello world`

Create a file named `hello.f90` in your home directory, with the following lines:
```
print *, "Hello World!"
end
```
Then in terminal, compile it
```
> cd ~
> gfortran -o hello hello.f90
```
Then run it
```
> ./hello
Hello World!
```
The last line should be what you see.



### How to use the Fortran `geopack`

I have downloaded the source code and saved it as `geopack_2005.for`. Now in the terminal, I `cd` to the folder it is saved and compile the code by:
```
$ gfortran -c geopack_2005.for -ffixed-form
```
Note that file extension **matters**: `*.for` is treated as in the fixed form, while `*.f90` is treated as in the free form. Since the Fortran `geopack` and Tsyganenko models are all written in the fixed form, to be save, just use `-ffixed-form` when compiling the codes.

Now I create the archive containing the code
```
$ ar -rcs geopack.a geopack_2005.o
```

Now there should be a file `geopack.a` in the folder. I've created a test code `test_geopack.f90` with the following lines in it:
```
program test_geopack

    implicit none
    real b1,b2,b3
    real r,theta,phi
    
    r = 1.1
    theta = 0.1
    phi = 0.2
    
    ! 2001-01-01/02:03:04, doy = 1
    call recalc(2001, 1, 2, 3, 4)
    call igrf_geo(r,theta,phi, b1,b2,b3)
    
    print *, b1,b2,b3
end
```

To run the test code, in terminal, I type:
```
$ gfortran -o test_geopack test_geopack.f90 geopack.a
$ ./test_geopack
  -42611.9023      -3373.67896      -531.660034    
```
