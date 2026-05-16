import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.linalg import norm, orthogonal_procrustes
from shapely.geometry import LineString, Point, Polygon


@dataclass
class ShiftVector:
    """
    Defines a shift-vector dataclass.
    """

    datetime: datetime
    x: float
    y: float

    def to_dict(self):
        """
        Converts the shift-vector dataclass to a dictionary, handling non JSON-serializable datetime types.
        """
        shift_vector_dict = asdict(self)
        shift_vector_dict["datetime"] = self.datetime.isoformat()
        return shift_vector_dict


def isobath_to_gdf(isobath: LineString) -> gpd.GeoDataFrame:
    """
    Returns a GeoDataFrame representation of one isobath geometry as points and segments
    with azimuth between pairs of vertices of each segments.
    """

    isobath_segments = split_linestring(isobath)

    data = {
        "point": [Point(coord[0], coord[1]) for coord in isobath.coords[:-1]],
        "segment": isobath_segments[:, 1],
        "azimuth": np.unwrap(isobath_segments[:, 0].astype(float), period=360),
    }

    # define a GeoDataFrame that represents the isobath geometry as points
    isobath_gdf = gpd.GeoDataFrame(data, crs=None).set_geometry("segment")

    # compute point to point distance using length of each segments
    isobath_gdf["length"] = isobath_gdf["segment"].length

    # compute cumulated distance for later rolling window operations
    isobath_gdf["distance"] = isobath_gdf["length"].cumsum()

    return isobath_gdf


def join_isobath_and_nav(
    isobaths: gpd.GeoDataFrame,
    nav: gpd.GeoDataFrame,
    max_dist: float = 50,
    min_overlap: int = 6,
    vertex_distance: float = 4,
) -> gpd.GeoDataFrame:
    """
    Finds the nearest nav point to each source isobath.
    Returns a joined GeoDataframe with closest navpoint to each srce isobath,
    discarding shortest srce isobath, i.e. those whose length < min_overlap * resampled distance.
    """
    # project navigation to match isobath CRS
    nav.to_crs(isobaths.crs, inplace=True)
    # copy projected point geometry so that it will be kept after spatial join : "navpoint"
    nav["navpoint"] = nav.geometry
    # spatial join isobaths and nearest nav points and sort by increasing time
    isobath_nav_merged = isobaths.sjoin_nearest(
        nav,
        max_distance=max_dist,
        lsuffix="isobath",
        rsuffix="nav",
        distance_col="distance",
    )
    isobath_nav_merged.sort_values(by="times", inplace=True)
    # if multiple isobaths are matched to the same nav point, keep only the closest one
    isobath_nav_merged = isobath_nav_merged.loc[isobath_nav_merged.groupby("times")["distance"].idxmin()]
    # drop projected geometry from nav
    nav.drop("navpoint", axis=1, inplace=True)
    # drop any other than projected nav points coming from nav into isobath
    cols_to_drop = nav.columns.intersection(isobath_nav_merged.columns)
    isobath_nav_merged.drop(cols_to_drop, axis=1, inplace=True)

    return isobath_nav_merged[isobath_nav_merged.geometry.length > min_overlap * vertex_distance]


def get_elevation_field(gpd: gpd.GeoDataFrame) -> str:
    """
    Returns elevation field name.
    Designed to cope with gdal code dependent behaviour :
        - isobaths computed with java gdal version contains "elevation" field
        - isobaths computed with python gdal version contains "elev" field
    """
    elevation_field = (col for col in gpd.columns if col.startswith("elev"))
    return next(elevation_field)


def join_srce_trgt(
    srce: gpd.GeoDataFrame, trgt: gpd.GeoDataFrame, min_overlap: int = 6, vertex_distance: float = 4
) -> gpd.GeoDataFrame:
    """
    Finds the nearest trgt isobath to each srce isobath with same elevation.
    Returns a joined GeoDataframe with a set of navpoint, srce and trgt isobath.
    """
    # retrieve elevation column name
    elevation_field = get_elevation_field(srce)
    # depth attribute join srce and trgt isobath
    srce_trgt_joined = srce.merge(trgt, on=elevation_field, suffixes=("_srce", "_trgt"))
    # convert it back to a geodataframe
    srce_trgt_joined = gpd.GeoDataFrame(srce_trgt_joined, geometry="geometry_srce", crs=srce.crs)

    # Calculate distance between srce and trgt isobath with same elevation
    srce_trgt_joined["distance"] = srce_trgt_joined.apply(
        lambda row: row["geometry_srce"].distance(row["geometry_trgt"]), axis=1
    )
    # discard shortest trgt isobath, i.e. those whose length < min_overlap (6) * resampled distance (4)
    srce_trgt_joined = srce_trgt_joined[srce_trgt_joined["geometry_trgt"].length > min_overlap * vertex_distance]

    # Group by srce isobath and retain only the nearest trgt isobath
    nearest_trgt = srce_trgt_joined.loc[srce_trgt_joined.groupby("geometry_srce")["distance"].idxmin()]
    # TODO : allow srce to match multiple trgts when possible, idxmin() only retrieve the first occurence

    return nearest_trgt.sort_values(by="times")


def clip_trgt(srce: gpd.GeoDataFrame, trgt: gpd.GeoDataFrame, max_dist: float = 100) -> gpd.GeoDataFrame:
    """
    Returns a clipped version of trgt isobath within srce isobath bufferd bounding polygon,
    representing maximum search radius for matching isobath.
    """
    clip_polygon = gpd.GeoSeries([srce.union_all().convex_hull.buffer(max_dist)])
    clip_polygon.crs = srce.crs
    clipped_trgt = trgt.clip(clip_polygon)
    return clipped_trgt.explode()


def split_linestring(line: LineString) -> np.ndarray:
    """
    Splits linestring geometry into segments and returns segment azimuth and segment Linestring representation.
    """
    return np.asarray(
        [[segment_azimuth(pt1, pt2), LineString([pt1, pt2])] for pt1, pt2 in zip(line.coords, line.coords[1:])]
    )


def segment_azimuth(pt1: tuple, pt2: tuple) -> float:
    """
    Returns segment azimuth
    """
    angle = math.atan2(pt2[0] - pt1[0], pt2[1] - pt1[1])
    return math.degrees(angle)


def get_nav_buffer(nav_gdf: gpd.GeoDataFrame, distance: float) -> Polygon:
    """
    Returns a buffer of the navigation points.
    """
    # Extract the coordinates from the navigation GeoDataFrame
    coordinates = nav_gdf.geometry.apply(lambda point: (point.x, point.y)).tolist()
    # Create a LineString from the extracted coordinates
    linestring = LineString(coordinates)
    # Return the buffer of the LineString
    return linestring.buffer(distance=distance, cap_style="flat")


def redistribute_vertices(geom: LineString, distance: float) -> LineString:
    """
    Returns a LineString whose vertices are evenly spaced according to distance.
    """
    num_vert = int(round(geom.length / distance))
    if num_vert == 0:
        num_vert = 1
    return LineString([geom.interpolate(float(n) / num_vert, normalized=True) for n in range(num_vert + 1)])


def match_src_trgt(
    src_trgt: gpd.GeoDataFrame,
    max_dist: float = 100,
    max_rot: float = 20,
    min_overlap: int = 6,
    vertex_distance: float = 4,
    chaincode_windows: range = range(10, 100, 20),
    debug=False,
    debug_dir=r"C:\Temp",
) -> List[ShiftVector]:
    """
    Computes several chain codes representation of isobaths (chaincode_windows ranges).
    Returns a list of shift vector dict  that best matches srce isobath portion to corresponding trgt isobath portion
    given several parameters : a maximum allowed amplitude and rotation, a minimum vertices overlap.
    """

    shift_vectors = []

    # Loop through each src isobath and try to match corresponding trgt isobath portion
    for src in src_trgt.itertuples():

        # retrieve evenly sampled srce and corresponding target isobath so that they can be matched
        s_isobath = isobath_to_gdf(redistribute_vertices(src.geometry_srce, distance=vertex_distance))
        t_isobath = isobath_to_gdf(redistribute_vertices(src.geometry_trgt, distance=vertex_distance))

        if len(s_isobath) > min_overlap and len(t_isobath) > min_overlap:

            # compute chain_codes over several rolling windows
            s_isobath = compute_chain_code(s_isobath, chaincode_windows)
            t_isobath = compute_chain_code(t_isobath, chaincode_windows)

            # compute chain-codes pairs mismatch and global misplacement
            matchs = compute_mismatch_and_misplacement(s_isobath, t_isobath)

            # plot if DEBUG
            if debug:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.set_title(f"src_{src.id_srce:04d}-trgt_{src.id_trgt:04d}")
                matchs["mismatch_Schain_code_10_Tchain_code_10"].plot(ax=ax)
                matchs["mismatch_Schain_code_30_Tchain_code_30"].plot(ax=ax)
                matchs["mismatch_Schain_code_50_Tchain_code_50"].plot(ax=ax)
                matchs["mismatch_Schain_code_70_Tchain_code_70"].plot(ax=ax)
                matchs["mismatch_Schain_code_90_Tchain_code_90"].plot(ax=ax)
                matchs["misplacement"].plot(ax=ax)

            # retrieve best match portions
            match_result = match_isobath(matchs, len(s_isobath), len(t_isobath))

            # plot if DEBUG
            if debug:
                out_img = Path(debug_dir).joinpath("navpoint_" + src.times.strftime("%Y_%m_%d_%H_%M_%S_%f") + ".png")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.set_title(f"src_{src.id_srce:04d}-trgt_{src.id_trgt:04d}")
                t_isobath.plot(ax=ax, color="blue")
                s_isobath.plot(ax=ax, color="red")
                t_isobath.iloc[match_result["trgt_slice"].start : match_result["trgt_slice"].stop].plot(
                    ax=ax, color="cyan"
                )
                s_isobath.iloc[match_result["srce_slice"].start : match_result["srce_slice"].stop].plot(
                    ax=ax, color="orange"
                )
                ax.scatter(src.navpoint.x, src.navpoint.y, marker="+", color="red")

            # try to register the corresponding portions of src to trgt isobaths
            # We need at list 6 points to register
            if match_result["overlap"] > min_overlap:
                T, R, d = local_registration(
                    s_isobath.iloc[match_result["srce_slice"].start : match_result["srce_slice"].stop],
                    t_isobath.iloc[match_result["trgt_slice"].start : match_result["trgt_slice"].stop],
                )

                # plot if DEBUG
                if debug:
                    ax.plot(np.add((0, T[0]), src.navpoint.x), np.add((0, T[1]), src.navpoint.y), color="orange")
                # compute shift-vector azimuth and norm
                azimuth = segment_azimuth(pt1=[0, 0], pt2=T)
                amplitude = norm(T)

                if amplitude < max_dist and abs(R) < max_rot:
                    # populate the shiftVectors list
                    # shift_vectors.append({"navpoint": src.navpoint, "datetime": src.times, "translation": T})
                    # {"datetime": "2016-07-22T17:23:34.771Z", "x": 150.0, "y": 30.0},
                    shift_vectors.append(ShiftVector(datetime=src.times, x=T[0], y=T[1]))
                    # plot if DEBUG
                    results_msg = f"[ok]\n{src.times}\naz : {azimuth:.1f}°\namplitude : {amplitude:.1f}m\nrotation : {R:1f}°\nscore : {match_result['score']}\nd_procrustes : {d}"

                else:
                    # plot if DEBUG
                    results_msg = f"[too far away]\n{src.times}\naz : {azimuth:.1f}°\namplitude : {amplitude:.1f}m\nrotation : {R:1f}°\nscore : {match_result['score']} | d_procrustes : {d}"

            else:
                # plot if DEBUG
                results_msg = f"[low overlap]\n{src.times}\nscore : {match_result['score']}"

            # plot if DEBUG
            if debug:
                bbox = {"boxstyle": "round", "fc": "blanchedalmond", "ec": "orange", "alpha": 0.5}
                ax.text(
                    1.1, 0.1, results_msg, fontsize=9, bbox=bbox, transform=ax.transAxes, horizontalalignment="left"
                )
                plt.savefig(out_img, bbox_inches="tight")
                plt.close()

    return shift_vectors


def compute_chain_code(isobath: gpd.GeoDataFrame, win_lengths: range) -> gpd.GeoDataFrame:
    """
    Computes several chain-code like representation of the isobath i.e. azimuth variation rate along different
    arc-length, through a temporary datetime-like index (mandatory for rolling window operations) based on cumulated
    distance, where 1m <-> 1s.
    """
    isobath["dist_idx"] = pd.to_datetime(isobath["distance"], unit="s")

    # perform rolling computation of chain codes over the different win_length window
    for win_length in win_lengths:
        isobath[f"chain_code_{win_length}"] = (
            isobath["azimuth"]
            .set_axis(isobath["dist_idx"])
            .rolling(str(win_length) + "s", center=True, min_periods=1, closed="both")
            .apply(lambda x: x[math.floor(len(x) / 2) :].mean() - x[: math.ceil(len(x) / 2)].mean())
            .set_axis(isobath.index)
        )
    # return the geodataframe without the temporary datetime-like index
    return isobath.drop(columns="dist_idx")


def compute_mismatch_and_misplacement(s_isobath: gpd.GeoDataFrame, t_isobath: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Matches isobath by computing rolling correlation of chain-codes.
    """
    s_chaincode_cols = [col_name for col_name in s_isobath.columns.values if "chain_code" in col_name]
    t_chaincode_cols = [col_name for col_name in t_isobath.columns.values if "chain_code" in col_name]

    # trgt is NaN-padded at the end, so that the rolling window can go all over  trgt, and stops when the first
    # element of 'srce' matches last of 'trgt'
    t_isobath = t_isobath.reindex(range(0, len(s_isobath) + len(t_isobath) - 1), fill_value=np.nan)

    arr = pd.DataFrame()

    # MISPLACEMENT : compute the rolling misplacement between srce and target BEWARE :  rolling.apply() doesn't
    # preserve geometry, we must instead use the rolling window as an iterable to keep geometries
    rolling_window = t_isobath.rolling(window=len(s_isobath) - 1, min_periods=1, center=False, closed="both")
    arr["misplacement"] = [misplacement(trgt, s_isobath) for trgt in rolling_window]
    out_cols = []

    # MISMATCH : Compute the rolling mismatch between srce and target
    for s_col, t_col in zip(s_chaincode_cols, t_chaincode_cols):
        out_col = f"mismatch_S{s_col}_T{t_col}"
        out_cols.append(out_col)
        # the fact that we pass s_col as argument for the lambda fun raises pylint error
        # pylint: disable=cell-var-from-loop
        arr[out_col] = (
            t_isobath[t_col]
            .rolling(window=len(s_isobath) - 1, min_periods=1, center=False, closed="both")
            .apply(lambda trgt: mismatch(trgt=trgt, srce=s_isobath[s_col].values), raw=True)
        )

    t_isobath = t_isobath.join(arr)

    return t_isobath


def match_isobath(matches: gpd.GeoDataFrame, srce_len: int, trgt_len: int, distmax: float = 100) -> dict:
    """
    Returns the best match between source isobath and target isobath
    as a dict with keys "score", "overlap", "srce_slice", "trgt_slice".
    """
    # find out wich columns contain mismatch values
    mismatch_cols = [col_name for col_name in matches.columns if col_name.startswith("mismatch")]
    # distance to the lowest misplacement index
    idxmin_misplacement = matches["misplacement"].idxmin()
    # Find out best match (i.e. the lowest mismatch) for each mismatch set within a distmax distance to the lowest misplacement
    window_size = round(distmax / matches["length"].mean())
    start_index = max(0, idxmin_misplacement - window_size)
    end_index = min(len(matches) - 1, idxmin_misplacement + window_size)

    trgt_idx = matches[mismatch_cols].iloc[start_index : end_index + 1].idxmin()
    scores = pd.DataFrame(
        data={
            "trgt_idx": trgt_idx,
            "score": matches[mismatch_cols].iloc[start_index : end_index + 1].min(),
            "delta_misplacement": abs(idxmin_misplacement - trgt_idx),
        },
        index=mismatch_cols,
    )
    # Get overall best match -> i.e. the lowest mismatch values index, closest to the lowest misplacement value index
    scores.sort_values(by=["delta_misplacement", "score"], ascending=True, inplace=True)
    best_match = scores.head(1)

    # return corresponding slices of srce and trgt isobath
    i = scores["trgt_idx"].values[0]
    srce_slice = slice(max(0, srce_len - 1 - i), min(srce_len, len(matches) - i))
    trgt_slice = slice(max(0, i + 1 - srce_len), min(trgt_len, i + 1))
    overlap = trgt_slice.stop - trgt_slice.start

    return {
        "score": scores["score"].values[:1],
        "overlap": overlap,
        "srce_slice": srce_slice,
        "trgt_slice": trgt_slice,
    }


def mismatch(trgt: np.ndarray, srce: np.ndarray) -> float:
    """
    Returns mismatch between source and target isobath chain-codes portion from current window
    i.e. the weighted point-to-point difference of chain-codes.
    """
    # target subset length
    n = len(trgt)
    # source subset finite element (trgt is NaN-padded at the end, so that the rolling window stops when the first
    # element of 'srce' matches last of 'trgt' )
    m = np.isfinite(trgt)
    # number of overlapping element from srce and trgt for the current window
    overlap = min(n, np.sum(m))

    return np.sum(np.abs(trgt[m] - srce[-n:][m])) / weight(overlap)


def misplacement(trgt: gpd.GeoDataFrame, srce: gpd.GeoDataFrame) -> float:
    """
    Returns the misplacement between source and target isobath portion from current window
    i.e. the weighted sum of the point-to-point distance.
    """
    n = len(trgt)
    m = trgt.geometry.notna() & ~trgt.geometry.is_empty
    overlap = min(n, np.sum(m))

    return trgt[m].distance(srce[len(srce) - n : overlap + len(srce) - n], align=False).sum() / weight(overlap)


def weight(n: int) -> float:
    """
    Returns a long overlap biased weight factor i.e. log(n+1) to avoid dividing by zero.
    """
    return n * math.log(n + 1)


def local_registration(srce: gpd.GeoDataFrame, trgt: gpd.GeoDataFrame) -> tuple[np.ndarray, float, float]:
    """
    Performs local registration between matching source and target isobath portion using Procrustes analysis.
    Returns translation, rotation and procrustes distance.
    """
    xy_srce = extract_coordinates(srce.set_geometry("point"))
    xy_trgt = extract_coordinates(trgt.set_geometry("point"))
    T, R, d, srce_transformed = procrustes_analysis(srce=xy_srce, trgt=xy_trgt)
    return T, R, d


def extract_coordinates(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Extracts coordinates from a GeoDataFrame assuming that it contains Point geometries.
    """
    return np.array(list(gdf.geometry.apply(lambda geom: (geom.x, geom.y))))


def procrustes_analysis(
    trgt: np.ndarray, srce: np.ndarray, use_rotation: bool = False, use_scale: bool = False
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """
    Performs Procrustes analysis to match isobath with translation, and optionally rotation and scale.
    https://en.wikipedia.org/wiki/Procrustes_analysis

    Parameters:
    - srce: matrix of size n*d representing n source points in d dimensions that should match target points
    - trgt: matrix of size n*d representing n target points

    Returns:
    - T: translation applied to srce.
    - R: rotation matrix applied to srce.
    - d: Procrustes distance (sum of squared distances between transformed srce and trgt).
    - srce_transformed : transformed version of srce (only rotation and translation).
    """
    # ensure we are in 2D for correct rotation angle interpretation
    assert trgt.shape[1] == 2 and srce.shape[1] == 2

    centroid_trgt = np.mean(trgt, axis=0)
    centroid_srce = np.mean(srce, axis=0)
    trgt_centered = trgt - centroid_trgt
    srce_centered = srce - centroid_srce

    R, scale = orthogonal_procrustes(trgt_centered, srce_centered)

    # If the matrix is 2x2, interpret the rotation angle
    angle = np.arctan2(R[1, 0], R[0, 0])
    angle_degrees = np.degrees(angle)

    srce_centered_rotated = np.dot(srce_centered, R.T)

    if use_scale:
        srce_centered_rotated = srce_centered_rotated * scale

    T = centroid_trgt - centroid_srce
    if use_rotation:
        srce_transformed = srce_centered_rotated + centroid_srce + T
    else:
        srce_transformed = srce + T

    d = np.mean(np.sqrt(np.sum((trgt - srce_transformed) ** 2, axis=1)))

    return T, angle_degrees, d, srce_transformed
