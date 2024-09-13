#
#
# def sun_position(doy, time, glat, glon, height, **kwargs):
#
#
#
# subroutine soco (ld,t,flat,Elon,height,
#      &          DECLIN, ZENITH, SUNRSE, SUNSET)
# c--------------------------------------------------------------------
# c       s/r to calculate the solar declination, zenith angle, and
# c       sunrise & sunset times  - based on Newbern Smith's algorithm
# c       [leo mcnamara, 1-sep-86, last modified 16-jun-87]
# c       {dieter bilitza, 30-oct-89, modified for IRI application}
# c
# c in:   ld      local day of year
# c       t       local hour (decimal)
# c       flat    northern latitude in degrees
# c       elon    east longitude in degrees
# c		height	height in km
# c
# c out:  declin      declination of the sun in degrees
# c       zenith      zenith angle of the sun in degrees
# c       sunrse      local time of sunrise in hours
# c       sunset      local time of sunset in hours
# c-------------------------------------------------------------------
# c
#         common/const/dtr,pi     /const1/humr,dumr
# c amplitudes of Fourier coefficients  --  1955 epoch.................
#         data    p1,p2,p3,p4,p6 /
#      &  0.017203534,0.034407068,0.051610602,0.068814136,0.103221204 /
# c
# c s/r is formulated in terms of WEST longitude.......................
#         wlon = 360. - Elon
# c
# c time of equinox for 1980...........................................
#         td = ld + (t + Wlon/15.) / 24.
#         te = td + 0.9369
# c
# c declination of the sun..............................................
#         dcl = 23.256 * sin(p1*(te-82.242)) + 0.381 * sin(p2*(te-44.855))
#      &      + 0.167 * sin(p3*(te-23.355)) - 0.013 * sin(p4*(te+11.97))
#      &      + 0.011 * sin(p6*(te-10.41)) + 0.339137
#         DECLIN = dcl
#         dc = dcl * dtr
# c
# c the equation of time................................................
#         tf = te - 0.5
#         eqt = -7.38*sin(p1*(tf-4.)) - 9.87*sin(p2*(tf+9.))
#      &      + 0.27*sin(p3*(tf-53.)) - 0.2*cos(p4*(tf-17.))
#         et = eqt * dtr / 4.
# c
#         fa = flat * dtr
#         phi = humr * ( t - 12.) + et
# c
#         a = sin(fa) * sin(dc)
#         b = cos(fa) * cos(dc)
#         	cosx = a + b * cos(phi)
#         	if(abs(cosx).gt.1.) cosx=sign(1.,cosx)
#         zenith = acos(cosx) / dtr
# c
# c calculate sunrise and sunset times --  at the ground...........
# c see Explanatory Supplement to the Ephemeris (1961) pg 401......
# c sunrise at height h metres is at...............................
# 		h=height*1000.
#         chih = 90.83 + 0.0347 * sqrt(h)
# c this includes corrections for horizontal refraction and........
# c semi-diameter of the solar disk................................
#         ch = cos(chih * dtr)
#         cosphi = (ch -a ) / b
# c if abs(secphi) > 1., sun does not rise/set.....................
# c allow for sun never setting - high latitude summer.............
#         secphi = 999999.
#         if(cosphi.ne.0.) secphi = 1./cosphi
#         sunset = 99.
#         sunrse = 99.
#         if(secphi.gt.-1.0.and.secphi.le.0.) return
# c allow for sun never rising - high latitude winter..............
#         sunset = -99.
#         sunrse = -99.
#         if(secphi.gt.0.0.and.secphi.lt.1.) return
# c
#         	cosx = cosphi
#         	if(abs(cosx).gt.1.) cosx=sign(1.,cosx)
#         phi = acos(cosx)
#         et = et / humr
#         phi = phi / humr
#         sunrse = 12. - phi - et
#         sunset = 12. + phi - et
#         if(sunrse.lt.0.) sunrse = sunrse + 24.
#         if(sunset.ge.24.) sunset = sunset - 24.
# c special case sunrse > sunset
#         if(sunrse.gt.sunset) then
#         	sunx=sign(99.0,flat)
#         	if(ld.gt.91.and.ld.lt.273) then
#         		sunset = sunx
#         		sunrse = sunx
#         	else
#         		sunset = -sunx
#         		sunrse = -sunx
#         	endif
#         	endif
# c
#         return
#         end
# c