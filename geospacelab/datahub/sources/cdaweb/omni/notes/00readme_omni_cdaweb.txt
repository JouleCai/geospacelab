https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/00readme_omni_cdaweb

---------------
1. Directory "hourly"  contains the hourly mean values of 
   the interplanetary magnetic  field (IMF) and solar wind plasma parameters 
   measured by various spacecraft near  the  Earth's  orbit,  as  well  as  
   geomagnetic and solar activity indices, and energetic proton fluxes.
--------------------------------

2. Directory "coho1hr_magplasma" contains the hourly mean values of OMNI
   the interplanetary magnetic  field (IMF) and solar wind plasma parameters
   converted into RTN coordinate system 
   For more details, see https://omniweb.sci.gsfc.nasa.gov/coho/
---------

3. Directories  "hro_1min", "hro_5min" "hro2_1min", "hro2_5min" 
   provide access to high resolution OMNI (HRO)
   data at 1-min and 5-min resolution.  

   The content and building of high resolution OMNI (HRO) are extensively described at 
    https://omniweb.sci.gsfc.nasa.gov/html/HROdocum.html.
    HRO and related spacecraft-specific data are accessible with multiple functionalities
    from  https://omniweb.sci.gsfc.nasa.gov/ow_min.html
--

3a. New high res OMNI data at dirs "hro2_1min", "hro2_5min" 
    provides access to the Modified (new ) High Resolution OMNI (HRO) data set 
    (at 1-min and 5-min resolution) is based on the definitive Wind plasma data set. 
    The main difference between the new and old high res. OMNI data is the following - 
    to create the new OMNI data set we used the definitive Wind/SWE 
    plasma data (shifted to the Bow shock nose ) rather than the Wind/SWE shifted
    "KP de-spiked" plasma data - they are at "hro_1min", "hro_5min" directories.
    Using these new hro2 data users can get more accurate plasma parameters and 
    the "Alpha/Proton Density Ratio" parameter (the latter parameter is not available
    in the SWE_KP data).
    However, users should keep in mind that the time coverage in the new HRO OMNI data set
    is decreased by 2-10% (depending on a year), and the latest date in the new data set
    is usually lags behind of the OMNI data that based on SWE_KP data.
 
____________________________________________________________________________

Related data and directories:
SPDF Data and Orbits Services <https://spdf.gsfc.nasa.gov/>

-------------------------------------------
  Acknowledgement:

 Use of these data in publications should be accompanied by
 acknowledgements of the  Space Physics Data Facility(SPDF) and

---------------------------------------------------------------------------
Authorizing NASA Official:  Robert Candey, Head, SPDF, NASA Goddard Space Flight Center
                        e-mail:  Robert.M.Candey@nasa.gov
-----------------------------------------------------------------

 
