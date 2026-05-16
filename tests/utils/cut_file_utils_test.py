import unittest
from datetime import datetime

import geopandas as gpd
import pytz
from shapely.geometry import Point, Polygon

from pyat.utils.cut_file_utils import cut_with_geo_mask
from sonar_netcdf.utils.nc_merger import Timeline


class TestCutWithGeoMask(unittest.TestCase):
    def setUp(self):
        """
        Set up test data:
        - A GeoDataFrame representing navigation profiles with latitude, longitude, and timestamps.
        - A GeoDataFrame representing a geographical area (mask) defined by polygons.
        """
        # Create a GeoDataFrame for navigation profiles
        self.data = {
            "geometry": [
                Point(18.0, 10.0),  # Out
                Point(20.0, 10.0),  # In
                Point(20.5, 12.5),  # In
                Point(22.5, 13.5),  # Out
                Point(23.5, 14.5),  # Out
                Point(21.0, 12.5),  # In
                Point(22.0, 12.0),  # In
            ],
            "times": [
                pytz.utc.localize(datetime(2023, 9, 1, 9, 30)),
                pytz.utc.localize(datetime(2023, 9, 1, 10, 0)),
                pytz.utc.localize(datetime(2023, 9, 1, 10, 30)),
                pytz.utc.localize(datetime(2023, 9, 1, 11, 0)),
                pytz.utc.localize(datetime(2023, 9, 1, 11, 30)),
                pytz.utc.localize(datetime(2023, 9, 1, 12, 0)),
                pytz.utc.localize(datetime(2023, 9, 1, 12, 30)),
            ],
        }
        self.nav_gdf = gpd.GeoDataFrame(self.data, crs="EPSG:4326")

        # Create a GeoDataFrame representing a geographical zone (mask)
        polygon = Polygon([(19.0, 9.0), (23.0, 9.0), (23.0, 13.0), (19.0, 13.0)])
        self.geo_mask_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")

    def test_cut_with_geo_mask_basic(self):
        """
        Basic test for cutting navigation profiles with a geographical zone (mask).
        Verifies if the function returns the correct timelines when the profile is inside the mask.
        """
        # Call the function
        line_prefix = "Line_test"
        result = cut_with_geo_mask(self.nav_gdf, self.geo_mask_gdf, line_prefix=line_prefix)

        # Expected result: time segments for points inside the zone (2nd and 3rd rows)
        expected = [
            Timeline(name=f"{line_prefix}_1", start=self.data["times"][1], stop=self.data["times"][2]),
            Timeline(name=f"{line_prefix}_2", start=self.data["times"][5], stop=self.data["times"][6]),
        ]

        # Check result
        self.assertEqual(len(expected), len(result))
        for timelineExpected, timelineResult in zip(expected, result):
            self.assertEqual(timelineExpected.name, timelineResult.name)
            self.assertEqual(timelineExpected.start, timelineResult.start)
            self.assertEqual(timelineExpected.stop, timelineResult.stop)

    def test_cut_with_geo_mask_reverse(self):
        """
        Basic test for cutting navigation profiles with a geographical zone (mask).
        Verifies if the function returns the correct timelines when the profile is inside the mask.
        """
        # Call the function
        line_prefix = "Line_test_revere"
        result = cut_with_geo_mask(self.nav_gdf, self.geo_mask_gdf, line_prefix=line_prefix, reverse_geo_mask=True)

        # Expected result: time segments for points inside the zone (2nd and 3rd rows)
        expected = [
            Timeline(name=f"{line_prefix}", start=self.data["times"][3], stop=self.data["times"][4]),
        ]

        # Check result
        self.assertEqual(len(expected), len(result))
        for timelineExpected, timelineResult in zip(expected, result):
            self.assertEqual(timelineExpected.name, timelineResult.name)
            self.assertEqual(timelineExpected.start, timelineResult.start)
            self.assertEqual(timelineExpected.stop, timelineResult.stop)


if __name__ == "__main__":
    unittest.main()
