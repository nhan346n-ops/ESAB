import json
import logging
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString

from pyat.utils.coords import compute_distance

logger = logging.getLogger(__name__)


@dataclass
class SlicePath:
    """
    Geographic path defined by arrays of latitude and longitude coordinates.

    This class represents a path along which sensor data will be sliced,
    providing utilities for path length calculation, resampling, and export.

    Attributes:
        latitudes: array of latitude values in degrees
        longitudes: array of longitude values in degrees
    """

    # Array of latitude values
    latitudes: np.ndarray
    # Array of longitude values
    longitudes: np.ndarray

    @property
    def length(self) -> float:
        """
        Calculate the total length of the path.

        Returns:
            total path length in meters
        """
        return np.sum(self.distances)

    @property
    def distances(self) -> np.ndarray:
        """
        Calculate distances between consecutive points.

        Returns:
            array of distances in meters between consecutive positions
        """
        return compute_distance(latitudes=self.latitudes, longitudes=self.longitudes)

    def resample(self, total_points: int) -> "SlicePath":
        """
        Resample the path to a specified number of points.

        This method preserves all original points and distributes additional points
        proportionally to segment lengths. Longer segments receive more interpolated points.
        The resampling maintains the path's shape while providing uniform point spacing.

        Args:
            total_points: desired total number of points in the resampled path

        Returns:
            new SlicePath with resampled coordinates
        """
        # Create the original LineString
        original_line = LineString(np.column_stack((self.longitudes, self.latitudes)))

        # Initialize arrays for the new points
        new_lons = np.zeros(total_points)
        new_lats = np.zeros(total_points)

        # Generate new points at equal intervals
        for i in range(total_points):
            # Calculate the fraction of the total length
            fraction = i / (total_points - 1)

            # Get the point at this fraction of the line's length
            new_point = original_line.interpolate(fraction, normalized=True)
            new_lons[i] = new_point.x
            new_lats[i] = new_point.y

        # Compute cumulative distances for the original and densified lines
        original_distances = np.cumsum(self.distances)
        densified_distances = np.cumsum(compute_distance(new_lons, new_lats))

        # Replace the closest densified points with the original points
        for i, dist in enumerate(original_distances):
            # Find the index of the closest densified point (by cumulative distance)
            closest_idx = np.argmin(np.abs(densified_distances - dist))
            # Replace the densified point with the original point
            new_lons[closest_idx] = self.longitudes[i]
            new_lats[closest_idx] = self.latitudes[i]

        return SlicePath(latitudes=new_lats, longitudes=new_lons)

    def write_to_geojson(self, output_file: str) -> None:
        """
        Export the path as a GeoJSON LineString.

        Args:
            output_file: path to the output GeoJSON file
        """
        # Build coordinates array (GeoJSON format is [longitude, latitude])
        coordinates = []
        for lat, lon in zip(self.latitudes, self.longitudes):
            coordinates.append([float(lat), float(lon)])

        # Create GeoJSON structure
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coordinates},
                    "properties": {"globe_description": {"name": "Polyline_resampled", "comment": ""}},
                }
            ],
        }

        # Write to file
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(geojson, f, indent=2)
            logger.info(f"Path exported to {output_file}")
        except IOError as e:
            logger.warning(f"Failed to write GeoJSON file {output_file}: {e}")
