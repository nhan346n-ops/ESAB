#!/usr/bin/env python3
"""
ShapefileNavigation class for reading navigation data from shapefiles.
Implements the AbstractNavigation protocol.
"""

import logging as log
from datetime import datetime
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

logger = log.getLogger("shapefile_navigation")


class ShapefileNavigation:
    """
    Navigation data reader for shapefile format.

    Reads navigation data from a shapefile containing point geometries
    with temporal and spatial information.
    """

    def __init__(self, shapefile_path: str):
        """
        Initialize the shapefile navigation reader.

        Args:
            shapefile_path: Path to the .shp file

        Raises:
            FileNotFoundError: If the shapefile doesn't exist
            ValueError: If the shapefile doesn't contain valid data
        """
        self.shapefile_path = Path(shapefile_path)

        if not self.shapefile_path.exists():
            raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

        # Load the shapefile
        self._gdf = gpd.read_file(shapefile_path)

        # Validate and convert geometry if needed
        self._process_geometry()

        # Extract and cache navigation data
        self._extract_navigation_data()

    def _process_geometry(self):
        """
        Process and validate geometry types.

        Converts LineString geometries to individual points.
        Validates that all geometries are Points or LineStrings.

        Raises:
            ValueError: If geometry types are not supported
        """
        geom_types = self._gdf.geometry.geom_type.unique()

        # Check if we have only Points
        if all(self._gdf.geometry.geom_type == "Point"):
            return  # Nothing to do, already points

        # Check if we have LineStrings
        if all(geom_type in ["LineString", "MultiLineString"] for geom_type in geom_types):
            self._convert_linestrings_to_points()
        elif "LineString" in geom_types or "MultiLineString" in geom_types:
            # Mixed geometry types with LineStrings
            self._convert_linestrings_to_points()
        elif not all(self._gdf.geometry.geom_type == "Point"):
            # Unsupported geometry types
            raise ValueError(f"Shapefile must contain Point or LineString geometries, found: {geom_types}")

    def _convert_linestrings_to_points(self):
        """
        Convert LineString geometries to individual Point rows.

        Each vertex of a LineString becomes a separate point with duplicated attributes.
        """
        new_rows = []

        for idx, row in self._gdf.iterrows():
            geom = row.geometry

            if geom.geom_type == "Point":
                # Keep point as is
                new_rows.append(row)

            elif geom.geom_type == "LineString":
                # Extract all coordinates from the LineString
                coords = list(geom.coords)

                # Create a new row for each coordinate
                for coord in coords:
                    new_row = row.copy()
                    new_row.geometry = Point(coord)
                    new_rows.append(new_row)

            elif geom.geom_type == "MultiLineString":
                # Handle MultiLineString by processing each LineString
                for line in geom.geoms:
                    coords = list(line.coords)
                    for coord in coords:
                        new_row = row.copy()
                        new_row.geometry = Point(coord)
                        new_rows.append(new_row)

        # Create new GeoDataFrame from the expanded rows
        self._gdf = gpd.GeoDataFrame(new_rows, crs=self._gdf.crs).reset_index(drop=True)

    def _parse_datetime(self, date_str: str, time_str: str) -> datetime:
        """
        Parse date and time strings into a datetime object.

        Args:
            date_str: Date string in format DD/MM/YYYY
            time_str: Time string in format HH:MM:SS

        Returns:
            datetime object
        """
        try:
            datetime_str = f"{date_str} {time_str}"
            return datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S")
        except Exception:
            return datetime.min

    def _extract_navigation_data(self):
        """Extract and cache navigation arrays from the geodataframe."""
        # Times (as Unix timestamps)
        date_col = "DATE"
        time_col = next((c for c in ["TIME", "HOUR", "HEURE"] if c in self._gdf.columns), None)
        if date_col in self._gdf.columns and time_col in self._gdf.columns:
            self._times = np.array(
                [self._parse_datetime(d, t) for d, t in zip(self._gdf[date_col], self._gdf[time_col])]
            )
        else:
            self._times = np.arange(len(self._gdf), dtype=float)

        # Latitudes and Longitudes from geometry
        self._latitudes = np.array([geom.y for geom in self._gdf.geometry])
        self._longitudes = np.array([geom.x for geom in self._gdf.geometry])

        # Optional: Headings (CAP field)
        heading_col = next((c for c in ["HEADING", "CAP"] if c in self._gdf.columns), None)
        self._headings = self._gdf[heading_col].to_numpy(dtype=float) if heading_col is not None else None

        # Optional: Altitudes (not in this shapefile format)
        self._altitudes = None

        # Optional: Vertical offsets (IMMERSION field)
        self._vertical_offsets = (
            self._gdf["IMMERSION"].to_numpy(dtype=float) * -1.0 if "IMMERSION" in self._gdf.columns else None
        )

        # Optional: Speeds (not in this shapefile format)
        self._speeds = None

        # Optional: Course over ground (not in this shapefile format)
        self._courses_over_ground = None

        # Optional: Sensor quality indicators (not in this shapefile format)
        self._sensor_quality_indicators = None

        self._name = self.shapefile_path.stem

        logger.info("Extracted %d navigation points from shapefile %s.", len(self._gdf), self._name)

    def get_name(self) -> Optional[str]:
        return self._name

    def get_times(self) -> np.ndarray:
        return self._times

    def get_latitudes(self) -> np.ndarray:
        return self._latitudes

    def get_longitudes(self) -> np.ndarray:
        return self._longitudes

    def get_headings(self) -> Optional[np.ndarray]:
        return self._headings

    def get_altitudes(self) -> Optional[np.ndarray]:
        return self._altitudes

    def get_vertical_offsets(self) -> Optional[np.ndarray]:
        return self._vertical_offsets

    def get_speeds(self) -> Optional[np.ndarray]:
        return self._speeds

    def get_courses_over_ground(self) -> Optional[np.ndarray]:
        return self._courses_over_ground

    def get_sensor_quality_indicators(self) -> Optional[np.ndarray]:
        return self._sensor_quality_indicators
