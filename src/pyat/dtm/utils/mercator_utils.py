from osgeo import osr


def is_same_mercator(srs1: osr.SpatialReference, srs2: osr.SpatialReference) -> bool:
    """
    Check if two Mercator_2SP spatial references are equivalent.

    For example, the following projections are equivalent:
    +proj=merc +lon_0=0 +lat_ts=-45 +k_0=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs
    +proj=merc +lat_ts=-45 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    """
    return bool(clean_mercator(srs1).IsSame(clean_mercator(srs2)))


def clean_mercator(srs: osr.SpatialReference) -> osr.SpatialReference:
    """
    If the provided spatial reference is a Mercator_2SP projection with WGS 84 spheroid, return a new spatial reference built with GDAL.
    (Allows to get a SpatialReference correctly set with the Mercator_2SP projection)
    """
    if srs.GetAttrValue("PROJECTION") == "Mercator_2SP" and srs.GetAttrValue("SPHEROID") == "WGS 84":
        new_srs = osr.SpatialReference()
        new_srs.SetWellKnownGeogCS("WGS84")
        new_srs.SetMercator2SP(
            stdp1=srs.GetProjParm("standard_parallel_1"),
            clat=0.0,
            clong=srs.GetProjParm("central_meridian"),
            fe=srs.GetProjParm("false_easting"),
            fn=srs.GetProjParm("false_northing"),
        )
        return new_srs
    return srs
