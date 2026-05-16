from datetime import datetime

import geopandas as gpd
import pandas as pd
from shapely.affinity import translate

from pyat.xsf.navshift.isobath_registration_tools import ShiftVector


class NavShifter:

    def __init__(
        self, nav: gpd.GeoDataFrame, shift_vectors: list[ShiftVector], start: datetime = None, stop: datetime = None
    ):
        self.nav = nav
        self.start = nav.index[0] if start is None else start
        self.stop = nav.index[-1] if stop is None else stop
        self.shift_vectors = shiftvectors_to_geodataframe(shift_vectors)

    def __call__(self) -> gpd.geodataframe:
        """
        Linearly offsets a navigation between a time period, provided some shift vectors.
        """

        # Merge vectors and nav on time
        nav_vectors_joined = self.nav.merge(self.shift_vectors, left_index=True, right_index=True, how="left")

        # add no offset at start if no vectors at that time
        if nav_vectors_joined.loc[self.start, ["x", "y"]].isna().all():
            nav_vectors_joined.loc[self.start, ["x", "y"]] = [0, 0]
        # same for stop
        if nav_vectors_joined.loc[self.stop, ["x", "y"]].isna().all():
            nav_vectors_joined.loc[self.stop, ["x", "y"]] = [0, 0]

        # Interpolate missing values in x and y in both-direction
        nav_vectors_joined[["x", "y"]] = nav_vectors_joined[["x", "y"]].interpolate(
            method="time", limit_direction="both"
        )
        # populate quality in accordance to the SeaDatanet Measurand qualifier flags vocabulary (L20)
        interpolated_values = nav_vectors_joined["quality"].isna()
        nav_vectors_joined.loc[interpolated_values, "quality"] = 8  # -> interpolated value

        # translate projected points
        nav_vectors_joined["points"] = nav_vectors_joined.apply(
            lambda row: translate(row["points"], xoff=row["x"], yoff=row["y"]), axis=1
        )

        # drop any unwanted columns
        nav_vectors_joined.drop(["x", "y"], axis=1, inplace=True)

        # unproject back to EPSG:4326 and return nav
        return nav_vectors_joined.to_crs(epsg=4326)


def shiftvectors_to_geodataframe(shift_vectors: list[ShiftVector]) -> pd.DataFrame:
    """
    Converts shift vectors to GeoDataFrame.
    """
    # convert shift vectors to time indexed geodataframe
    shift_vectors = pd.DataFrame(shift_vectors).set_index("datetime")
    # add a quality column based on the SeaDatanet Measurand qualifier flags vocabulary (L20)
    shift_vectors["quality"] = 5  # -> Modified value

    return shift_vectors
