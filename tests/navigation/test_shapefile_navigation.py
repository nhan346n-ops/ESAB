#!/usr/bin/env python3
"""
Unit tests for ShapefileNavigation class.
"""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely import LineString, MultiLineString, Polygon
from shapely.geometry import Point

from pyat.navigation.abstract_navigation import AbstractNavigation
from pyat.navigation.shapefile_navigation import ShapefileNavigation


@pytest.fixture
def linestring_shapefile(temp_dir):
    """
    Create a shapefile with LineString geometry.

    Returns:
        Path to the created shapefile
    """
    # Create a LineString with multiple points
    line_coords = [
        (-46.1811198, 24.5467195),
        (-46.1812000, 24.5468000),
        (-46.1813000, 24.5469000),
        (-46.1814000, 24.5470000),
    ]

    data = {
        "DATE": ["07/08/2022"],
        "HEURE": ["18:27:08"],
        "HEADING": [45.0],
        "IMMERSION": [10.0],
        "Campagne": ["HERMINE2"],
    }

    geometry = [LineString(line_coords)]
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

    shapefile_path = Path(temp_dir) / "linestring_navigation.shp"
    gdf.to_file(shapefile_path)

    return str(shapefile_path)


@pytest.fixture
def multilinestring_shapefile(temp_dir):
    """
    Create a shapefile with MultiLineString geometry.

    Returns:
        Path to the created shapefile
    """
    # Create multiple LineStrings
    line1 = LineString([(-46.18, 24.54), (-46.19, 24.55)])
    line2 = LineString([(-46.19, 24.55), (-46.20, 24.56)])

    data = {"DATE": ["07/08/2022"], "HEURE": ["18:27:08"], "Campagne": ["HERMINE2"]}

    geometry = [MultiLineString([line1, line2])]
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

    shapefile_path = Path(temp_dir) / "multiline_navigation.shp"
    gdf.to_file(shapefile_path)

    return str(shapefile_path)


@pytest.fixture
def linestring_shapefile(temp_dir):
    """
    Create a shapefile with LineString geometry.

    Returns:
        Path to the created shapefile
    """
    from shapely.geometry import LineString

    # Create a LineString with multiple points
    line_coords = [
        (-46.1811198, 24.5467195),
        (-46.1812000, 24.5468000),
        (-46.1813000, 24.5469000),
        (-46.1814000, 24.5470000),
    ]

    data = {"DATE": ["07/08/2022"], "HEURE": ["18:27:08"], "CAP": [45.0], "IMMERSION": [10.0], "Campagne": ["HERMINE2"]}

    geometry = [LineString(line_coords)]
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

    shapefile_path = Path(temp_dir) / "linestring_navigation.shp"
    gdf.to_file(shapefile_path)

    return str(shapefile_path)


@pytest.fixture
def multilinestring_shapefile(temp_dir):
    """
    Create a shapefile with MultiLineString geometry.

    Returns:
        Path to the created shapefile
    """
    # Create multiple LineStrings
    line1 = LineString([(-46.18, 24.54), (-46.19, 24.55)])
    line2 = LineString([(-46.19, 24.55), (-46.20, 24.56)])

    data = {"DATE": ["07/08/2022"], "HEURE": ["18:27:08"], "Campagne": ["HERMINE2"]}

    geometry = [MultiLineString([line1, line2])]
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

    shapefile_path = Path(temp_dir) / "multiline_navigation.shp"
    gdf.to_file(shapefile_path)

    return str(shapefile_path)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_shapefile(temp_dir):
    """
    Create a sample shapefile with navigation data.

    Returns:
        Path to the created shapefile
    """
    # Create sample data
    data = {
        "DATE": ["07/08/2022", "07/08/2022", "07/08/2022"],
        "HEURE": ["18:27:08", "18:27:18", "18:27:28"],
        "LATITUDE": [24.5467195, 24.5468000, 24.5469000],
        "LONGITUDE": [-46.1811198, -46.1812000, -46.1813000],
        "CAP": [45.0, 50.0, 55.0],
        "IMMERSION": [10.0, 11.0, 12.0],
        "SOURCE": ["Camera", "Camera", "Camera"],
        "NumCamp": ["18001851", "18001851", "18001851"],
        "Campagne": ["HERMINE2", "HERMINE2", "HERMINE2"],
        "NumPlongee": ["PL2054_12", "PL2054_12", "PL2054_12"],
        "Engin": ["Nautile", "Nautile", "Nautile"],
    }

    # Create geometries from lat/lon
    geometry = [Point(lon, lat) for lat, lon in zip(data["LATITUDE"], data["LONGITUDE"])]

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

    # Save to shapefile
    shapefile_path = Path(temp_dir) / "test_navigation.shp"
    gdf.to_file(shapefile_path)

    return str(shapefile_path)


@pytest.fixture
def minimal_shapefile(temp_dir):
    """
    Create a minimal shapefile with only required fields.

    Returns:
        Path to the created shapefile
    """
    # Create minimal data (no DATE/HEURE, no optional fields)
    geometry = [Point(-46.1811198, 24.5467195), Point(-46.1812000, 24.5468000), Point(-46.1813000, 24.5469000)]

    gdf = gpd.GeoDataFrame({"id": [1, 2, 3]}, geometry=geometry, crs="EPSG:4326")

    shapefile_path = Path(temp_dir) / "minimal_navigation.shp"
    gdf.to_file(shapefile_path)

    return str(shapefile_path)


class TestShapefileNavigationLineStringGeometry:
    """Tests for LineString geometry handling."""

    def test_linestring_converted_to_points(self, linestring_shapefile):
        """Test that LineString is converted to individual points."""

        nav = ShapefileNavigation(linestring_shapefile)

        # Should have 4 points from the LineString
        assert len(nav.get_latitudes()) == 4
        assert len(nav.get_longitudes()) == 4
        assert len(nav.get_times()) == 4

        # Verify coordinates
        lats = nav.get_latitudes()
        lons = nav.get_longitudes()

        assert lats[0] == pytest.approx(24.5467195, rel=1e-6)
        assert lats[1] == pytest.approx(24.5468000, rel=1e-6)
        assert lats[2] == pytest.approx(24.5469000, rel=1e-6)
        assert lats[3] == pytest.approx(24.5470000, rel=1e-6)

        assert lons[0] == pytest.approx(-46.1811198, rel=1e-6)
        assert lons[1] == pytest.approx(-46.1812000, rel=1e-6)
        assert lons[2] == pytest.approx(-46.1813000, rel=1e-6)
        assert lons[3] == pytest.approx(-46.1814000, rel=1e-6)

    def test_linestring_attributes_duplicated(self, linestring_shapefile):
        """Test that attributes are duplicated for each point from LineString."""
        nav = ShapefileNavigation(linestring_shapefile)

        # All points should have the same heading (duplicated from original row)
        headings = nav.get_headings()
        assert headings is not None
        assert len(headings) == 4
        assert all(h == 45.0 for h in headings)

        # All points should have the same vertical_offsets
        vertical_offsets = nav.get_vertical_offsets()
        assert vertical_offsets is not None
        assert len(vertical_offsets) == 4
        assert all(a == 10.0 for a in vertical_offsets)

    def test_multilinestring_converted_to_points(self, multilinestring_shapefile):
        """Test that MultiLineString is converted to individual points."""
        nav = ShapefileNavigation(multilinestring_shapefile)

        # Should have 4 points total (2 from each LineString)
        assert len(nav.get_latitudes()) == 4
        assert len(nav.get_longitudes()) == 4

    def test_linestring_with_minimal_data(self, temp_dir):
        """Test LineString with minimal data (no attributes)."""
        line = LineString([(-46.18, 24.54), (-46.19, 24.55), (-46.20, 24.56)])
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[line], crs="EPSG:4326")

        shapefile_path = Path(temp_dir) / "minimal_line.shp"
        gdf.to_file(shapefile_path)

        nav = ShapefileNavigation(str(shapefile_path))

        assert len(nav.get_latitudes()) == 3
        assert len(nav.get_longitudes()) == 3

        # Times should be sequential indices since no DATE/HEURE
        times = nav.get_times()
        np.testing.assert_array_equal(times, np.array([0, 1, 2]))


class TestShapefileNavigationLineStringGeometry:
    """Tests for LineString geometry handling."""

    def test_linestring_converted_to_points(self, linestring_shapefile):
        """Test that LineString is converted to individual points."""
        nav = ShapefileNavigation(linestring_shapefile)

        # Should have 4 points from the LineString
        assert len(nav.get_latitudes()) == 4
        assert len(nav.get_longitudes()) == 4
        assert len(nav.get_times()) == 4

        # Verify coordinates
        lats = nav.get_latitudes()
        lons = nav.get_longitudes()

        assert lats[0] == pytest.approx(24.5467195, rel=1e-6)
        assert lats[1] == pytest.approx(24.5468000, rel=1e-6)
        assert lats[2] == pytest.approx(24.5469000, rel=1e-6)
        assert lats[3] == pytest.approx(24.5470000, rel=1e-6)

        assert lons[0] == pytest.approx(-46.1811198, rel=1e-6)
        assert lons[1] == pytest.approx(-46.1812000, rel=1e-6)
        assert lons[2] == pytest.approx(-46.1813000, rel=1e-6)
        assert lons[3] == pytest.approx(-46.1814000, rel=1e-6)

    def test_linestring_attributes_duplicated(self, linestring_shapefile):
        """Test that attributes are duplicated for each point from LineString."""
        nav = ShapefileNavigation(linestring_shapefile)

        # All points should have the same heading (duplicated from original row)
        headings = nav.get_headings()
        assert headings is not None
        assert len(headings) == 4
        assert all(h == 45.0 for h in headings)

        # All points should have the same vertical_offsets
        vertical_offsets = nav.get_vertical_offsets()
        assert vertical_offsets is not None
        assert len(vertical_offsets) == 4
        assert all(a == -10.0 for a in vertical_offsets)

    def test_multilinestring_converted_to_points(self, multilinestring_shapefile):
        """Test that MultiLineString is converted to individual points."""
        nav = ShapefileNavigation(multilinestring_shapefile)

        # Should have 4 points total (2 from each LineString)
        assert len(nav.get_latitudes()) == 4
        assert len(nav.get_longitudes()) == 4

    def test_linestring_with_minimal_data(self, temp_dir):
        """Test LineString with minimal data (no attributes)."""
        line = LineString([(-46.18, 24.54), (-46.19, 24.55), (-46.20, 24.56)])
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[line], crs="EPSG:4326")

        shapefile_path = Path(temp_dir) / "minimal_line.shp"
        gdf.to_file(shapefile_path)

        nav = ShapefileNavigation(str(shapefile_path))

        assert len(nav.get_latitudes()) == 3
        assert len(nav.get_longitudes()) == 3

        # Times should be sequential indices since no DATE/HEURE
        times = nav.get_times()
        np.testing.assert_array_equal(times, np.array([0, 1, 2]))

    def test_multiple_linestrings(self, temp_dir):
        """Test shapefile with multiple separate LineStrings."""
        line1 = LineString([(-46.18, 24.54), (-46.19, 24.55)])
        line2 = LineString([(-46.20, 24.56), (-46.21, 24.57)])

        data = {"DATE": ["07/08/2022", "07/08/2022"], "HEURE": ["18:27:08", "18:27:18"], "CAP": [45.0, 50.0]}

        gdf = gpd.GeoDataFrame(data, geometry=[line1, line2], crs="EPSG:4326")

        shapefile_path = Path(temp_dir) / "multiple_lines.shp"
        gdf.to_file(shapefile_path)

        nav = ShapefileNavigation(str(shapefile_path))

        # Should have 4 points total (2 from each LineString)
        assert len(nav.get_latitudes()) == 4
        assert len(nav.get_longitudes()) == 4

        # First 2 points should have CAP=45.0, last 2 should have CAP=50.0
        headings = nav.get_headings()
        assert headings[0] == 45.0
        assert headings[1] == 45.0
        assert headings[2] == 50.0
        assert headings[3] == 50.0


class TestShapefileNavigationInit:
    """Tests for ShapefileNavigation initialization."""

    def test_init_with_valid_shapefile(self, sample_shapefile):
        """Test initialization with a valid shapefile."""
        nav = ShapefileNavigation(sample_shapefile)

        assert nav is not None
        assert nav.shapefile_path.exists()

    def test_init_with_nonexistent_file(self):
        """Test initialization with a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ShapefileNavigation("/path/to/nonexistent/file.shp")

    def test_init_with_non_point_geometry(self, temp_dir):
        """Test initialization with unsupported geometries raises ValueError."""
        # Create shapefile with Polygon geometries (not supported)
        polygon = Polygon([(-46.18, 24.54), (-46.19, 24.55), (-46.19, 24.54), (-46.18, 24.54)])
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[polygon], crs="EPSG:4326")

        shapefile_path = Path(temp_dir) / "polygon.shp"
        gdf.to_file(shapefile_path)

        with pytest.raises(ValueError, match="Point or LineString"):
            ShapefileNavigation(str(shapefile_path))


class TestShapefileNavigationBasicMethods:
    """Tests for basic navigation methods."""

    def test_get_name(self, sample_shapefile):
        """Test get_name returns the shapefile stem."""
        nav = ShapefileNavigation(sample_shapefile)
        name = nav.get_name()

        assert name == "test_navigation"

    def test_get_times_with_date_time_fields(self, sample_shapefile):
        """Test get_times returns datetime objects when DATE/HEURE fields exist."""
        nav = ShapefileNavigation(sample_shapefile)
        times = nav.get_times()

        assert isinstance(times, np.ndarray)
        assert len(times) == 3
        assert all(isinstance(t, datetime) for t in times)

        # Verify chronological order
        assert times[0] < times[1] < times[2]

    def test_get_times_without_date_time_fields(self, minimal_shapefile):
        """Test get_times returns sequential indices when no date/time fields."""
        nav = ShapefileNavigation(minimal_shapefile)
        times = nav.get_times()

        assert isinstance(times, np.ndarray)
        assert len(times) == 3
        np.testing.assert_array_equal(times, np.array([0, 1, 2]))

    def test_get_latitudes(self, sample_shapefile):
        """Test get_latitudes extracts Y coordinates from geometry."""
        nav = ShapefileNavigation(sample_shapefile)
        latitudes = nav.get_latitudes()

        assert isinstance(latitudes, np.ndarray)
        assert len(latitudes) == 3
        assert latitudes[0] == pytest.approx(24.5467195, rel=1e-6)
        assert latitudes[1] == pytest.approx(24.5468000, rel=1e-6)
        assert latitudes[2] == pytest.approx(24.5469000, rel=1e-6)

    def test_get_longitudes(self, sample_shapefile):
        """Test get_longitudes extracts X coordinates from geometry."""
        nav = ShapefileNavigation(sample_shapefile)
        longitudes = nav.get_longitudes()

        assert isinstance(longitudes, np.ndarray)
        assert len(longitudes) == 3
        assert longitudes[0] == pytest.approx(-46.1811198, rel=1e-6)
        assert longitudes[1] == pytest.approx(-46.1812000, rel=1e-6)
        assert longitudes[2] == pytest.approx(-46.1813000, rel=1e-6)


class TestShapefileNavigationOptionalMethods:
    """Tests for optional navigation methods."""

    def test_get_headings_when_available(self, sample_shapefile):
        """Test get_headings returns CAP field data when available."""
        nav = ShapefileNavigation(sample_shapefile)
        headings = nav.get_headings()

        assert headings is not None
        assert isinstance(headings, np.ndarray)
        assert len(headings) == 3
        np.testing.assert_array_equal(headings, np.array([45.0, 50.0, 55.0]))

    def test_get_headings_when_unavailable(self, minimal_shapefile):
        """Test get_headings returns None when CAP field is missing."""
        nav = ShapefileNavigation(minimal_shapefile)
        headings = nav.get_headings()

        assert headings is None

    def test_get_vertical_offsets_when_available(self, sample_shapefile):
        """Test get_vertical_offsets returns IMMERSION field data when available."""
        nav = ShapefileNavigation(sample_shapefile)
        vertical_offsets = nav.get_vertical_offsets()

        assert vertical_offsets is not None
        assert isinstance(vertical_offsets, np.ndarray)
        assert len(vertical_offsets) == 3
        np.testing.assert_array_equal(vertical_offsets, np.array([-10.0, -11.0, -12.0]))

    def test_get_vertical_offsets_when_unavailable(self, minimal_shapefile):
        """Test get_vertical_offsets returns None when IMMERSION field is missing."""
        nav = ShapefileNavigation(minimal_shapefile)
        vertical_offsets = nav.get_vertical_offsets()

        assert vertical_offsets is None

    def test_get_altitudes_returns_none(self, minimal_shapefile):
        """Test get_altitudes returns None (not implemented)."""
        nav = ShapefileNavigation(minimal_shapefile)
        altitudes = nav.get_altitudes()

        assert altitudes is None

    def test_get_speeds_returns_none(self, sample_shapefile):
        """Test get_speeds returns None (not implemented)."""
        nav = ShapefileNavigation(sample_shapefile)
        speeds = nav.get_speeds()

        assert speeds is None

    def test_get_courses_over_ground_returns_none(self, sample_shapefile):
        """Test get_courses_over_ground returns None (not implemented)."""
        nav = ShapefileNavigation(sample_shapefile)
        cog = nav.get_courses_over_ground()

        assert cog is None

    def test_get_sensor_quality_indicators_returns_none(self, sample_shapefile):
        """Test get_sensor_quality_indicators returns None (not implemented)."""
        nav = ShapefileNavigation(sample_shapefile)
        sqi = nav.get_sensor_quality_indicators()

        assert sqi is None


class TestShapefileNavigationDateTimeParsing:
    """Tests for datetime parsing functionality."""

    def test_parse_datetime_valid(self):
        """Test _parse_datetime with valid date and time strings."""
        # We need an instance to test the method
        # Create a dummy instance (we'll use a fixture in actual test)
        nav = ShapefileNavigation.__new__(ShapefileNavigation)

        dt = nav._parse_datetime("07/08/2022", "18:27:08")

        assert dt.year == 2022
        assert dt.month == 8
        assert dt.day == 7
        assert dt.hour == 18
        assert dt.minute == 27
        assert dt.second == 8

    def test_parse_datetime_invalid(self):
        """Test _parse_datetime with invalid strings returns datetime.min."""
        nav = ShapefileNavigation.__new__(ShapefileNavigation)

        dt = nav._parse_datetime("invalid", "date")

        assert dt == datetime.min


class TestShapefileNavigationAlternativeTimeColumns:
    """Tests for alternative time column names (TIME, HOUR)."""

    def test_with_time_column(self, temp_dir):
        """Test initialization with TIME column instead of HEURE."""
        # Create shapefile with TIME column
        data = {"DATE": ["07/08/2022", "07/08/2022"], "TIME": ["18:27:08", "18:27:18"]}
        geometry = [Point(-46.18, 24.54), Point(-46.19, 24.55)]
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

        shapefile_path = Path(temp_dir) / "time_column.shp"
        gdf.to_file(shapefile_path)

        nav = ShapefileNavigation(str(shapefile_path))
        times = nav.get_times()

        assert len(times) == 2
        assert all(isinstance(t, datetime) for t in times)

    def test_with_hour_column(self, temp_dir):
        """Test initialization with HOUR column."""
        # Create shapefile with HOUR column
        data = {"DATE": ["07/08/2022", "07/08/2022"], "HOUR": ["18:27:08", "18:27:18"]}
        geometry = [Point(-46.18, 24.54), Point(-46.19, 24.55)]
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

        shapefile_path = Path(temp_dir) / "hour_column.shp"
        gdf.to_file(shapefile_path)

        nav = ShapefileNavigation(str(shapefile_path))
        times = nav.get_times()

        assert len(times) == 2
        assert all(isinstance(t, datetime) for t in times)


class TestShapefileNavigationProtocolCompliance:
    """Tests to verify compliance with AbstractNavigation protocol."""

    def test_implements_all_required_methods(self, sample_shapefile):
        """Test that ShapefileNavigation implements all required protocol methods."""
        from pyat.navigation.shapefile_navigation import ShapefileNavigation

        nav = ShapefileNavigation(sample_shapefile)

        # Check if instance implements the protocol
        assert isinstance(nav, AbstractNavigation)

    def test_required_methods_return_correct_types(self, sample_shapefile):
        """Test that required methods return correct types."""
        nav = ShapefileNavigation(sample_shapefile)

        # get_name should return Optional[str]
        assert nav.get_name() is None or isinstance(nav.get_name(), str)

        # get_times should return np.ndarray
        assert isinstance(nav.get_times(), np.ndarray)

        # get_latitudes should return np.ndarray
        assert isinstance(nav.get_latitudes(), np.ndarray)

        # get_longitudes should return np.ndarray
        assert isinstance(nav.get_longitudes(), np.ndarray)

    def test_optional_methods_return_correct_types(self, sample_shapefile):
        """Test that optional methods return correct types."""
        nav = ShapefileNavigation(sample_shapefile)

        # All optional methods should return Optional[np.ndarray]
        headings = nav.get_headings()
        assert headings is None or isinstance(headings, np.ndarray)

        altitudes = nav.get_altitudes()
        assert altitudes is None or isinstance(altitudes, np.ndarray)

        vertical_offsets = nav.get_vertical_offsets()
        assert vertical_offsets is None or isinstance(vertical_offsets, np.ndarray)

        speeds = nav.get_speeds()
        assert speeds is None or isinstance(speeds, np.ndarray)

        cog = nav.get_courses_over_ground()
        assert cog is None or isinstance(cog, np.ndarray)

        sqi = nav.get_sensor_quality_indicators()
        assert sqi is None or isinstance(sqi, np.ndarray)


class TestShapefileNavigationEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_shapefile(self, temp_dir):
        """Test handling of empty shapefile."""
        # Create empty shapefile
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        shapefile_path = Path(temp_dir) / "empty.shp"
        gdf.to_file(shapefile_path)

        nav = ShapefileNavigation(str(shapefile_path))

        assert len(nav.get_times()) == 0
        assert len(nav.get_latitudes()) == 0
        assert len(nav.get_longitudes()) == 0

    def test_single_point(self, temp_dir):
        """Test handling of shapefile with single point."""
        data = {"DATE": ["07/08/2022"], "HEURE": ["18:27:08"]}
        geometry = [Point(-46.18, 24.54)]
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

        shapefile_path = Path(temp_dir) / "single_point.shp"
        gdf.to_file(shapefile_path)

        nav = ShapefileNavigation(str(shapefile_path))

        assert len(nav.get_times()) == 1
        assert len(nav.get_latitudes()) == 1
        assert len(nav.get_longitudes()) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
