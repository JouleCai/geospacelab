# Wrapper of the geopack package

This is a wrapper of the [geopack](https://github.com/tsssss/geopack/tree/master) package

GeospaceLAB uses geopack for several coordinate transformations. However, sometime geopack has a connection issue to [](http://www.ngdc.noaa.gov/IAGA/vmod/coeffs/) to download the igrf coefficients. This wrapper copy the source scripts from geospack and mutes the update function, which is used to download the coefficients. The latest version is 1.0.13 and the coefficients have been updated to IGRF14 in this wrapper.