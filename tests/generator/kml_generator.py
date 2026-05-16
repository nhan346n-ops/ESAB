#! /usr/bin/env python3
# coding: utf-8

import tempfile as tmp
from osgeo import ogr


def create_kml(dir: str, coords: dict) -> str:

    filepath = tmp.mktemp(".kml", dir=dir)
    # Create the output Driver
    outDriver = ogr.GetDriverByName("KML")
    outDataSource = outDriver.CreateDataSource(filepath)

    for k in coords.keys():
        geom_poly = ogr.Geometry(ogr.wkbPolygon)
        ring = ogr.Geometry(ogr.wkbLinearRing)

        pointlist = coords[k]
        for point in pointlist:
            ring.AddPoint(point[0], point[1])
        geom_poly.AddGeometry(ring)
        outLayer = outDataSource.CreateLayer(k, geom_type=ogr.wkbPolygon)
        # Get the output Layer's Feature Definition
        featureDefn = outLayer.GetLayerDefn()
        # create a new feature
        outFeature = ogr.Feature(featureDefn)

        # Set new geometry
        outFeature.SetGeometry(geom_poly)
        outFeature.SetStyleString("BRUSH(fc:#FF0000FF);PEN(c:#00FF00FF)")
        # Add new feature to output Layer
        outLayer.CreateFeature(outFeature)

    # dereference the feature
    outFeature = None
    # Save and close DataSources
    outDataSource = None
    return filepath
