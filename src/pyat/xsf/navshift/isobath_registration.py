import os.path
import tempfile
from typing import Dict, List

import geopandas as gpd
import pandas as pd
from osgeo import osr
from pygws.service.progress_monitor import DefaultMonitor

import pyat.xsf.navshift.isobath_registration_tools as regbat
import pyat.dtm.dtm_driver as dtm_driver
import pyat.utils.pyat_logger as log
from pyat.xsf.navshift.isobath_registration_tools import ShiftVector
from pyat.xsf.navshift.nav_shifter import NavShifter
from pyat.dtm.export.dtm_to_isobath import Dtm2Isobath
from pyat.navigation import navigation_exporter, navigation_factory
from pyat.sounder.sounder_to_dtm import SounderToDtmExporter
from pyat.utils.path_utils import delete_files


class IsobathRegistrationProcess:
    """
    Computes navigation shift vectors using isobath registration :
    Given source isobath and navigation, and target isobaths, all as GeoDataFrames
        - performs spatial and attributes joins  (navigation points with source isobaths and corresponding target isobath)
        - computes  and returns shift vectors that match source and target isobath through chain-codes correlations
        and procrustes analysis
    Parameters are :
        - max_dist :
            Maximum allowed shift amplitude (default 100m)
        - vertex_distance:
            Source and target isobath will be resampled at this given vertex distance before matching (default 4m)
        - min_overlap :
            Minimum number of corresponding vertices needed to compute a robust matching transformation (default 6)
    """

    def __init__(
        self,
        trgt_isobaths: gpd.GeoDataFrame,
        srce_isobaths: gpd.GeoDataFrame,
        srce_nav: gpd.GeoDataFrame,
        max_dist: float = 100,
        min_overlap: int = 6,
        vertex_distance: float = 4,
    ):
        self.trgt_isobaths = trgt_isobaths
        self.srce_isobaths = srce_isobaths
        self.srce_nav = srce_nav
        self.max_dist = max_dist
        self.min_overlap = min_overlap
        self.vertex_distance = vertex_distance
        self.logger = log.logging.getLogger(self.__class__.__name__)

    def __call__(self) -> List[ShiftVector]:
        """
        Runs an isobath registration process.
        """
        shift_vectors = []

        # check if any of srce or tgrt isobath is empty
        if self.srce_isobaths.empty or self.trgt_isobaths.empty:
            self.logger.error(
                "No source or target isobaths."
                "\n\tPlease check 'isobath interval' parameter and/or try a lower value."
            )
            return shift_vectors

        # spatial join nearest navigation points to source isobath
        srce_nav_isobaths = regbat.join_isobath_and_nav(
            isobaths=self.srce_isobaths,
            nav=self.srce_nav,
            min_overlap=self.min_overlap,
            vertex_distance=self.vertex_distance,
        )
        if srce_nav_isobaths.empty:
            self.logger.warning(
                "No spatial join between time interval and source isobaths."
                "\n\tPlease check and/or modify 'max_dist', 'min_overlap' or 'vertex_distance' parameters."
            )
            return shift_vectors

        # clip target isobath to source isobath distmax-buffered bounding polygon
        trgt_isobaths_clipped = regbat.clip_trgt(
            srce=self.srce_isobaths, trgt=self.trgt_isobaths, max_dist=self.max_dist
        )
        if trgt_isobaths_clipped.empty:
            self.logger.warning(
                "Source isobaths and target isobaths don't overlap."
                "\n\tPlease check that both source and target isobaths bounding boxes ovelap."
            )
            return shift_vectors

        # spatial join source navigation and isobath to nearest target isobath with same elevation
        srce_trgt_joined = regbat.join_srce_trgt(
            srce=srce_nav_isobaths,
            trgt=trgt_isobaths_clipped,
            min_overlap=self.min_overlap,
            vertex_distance=self.vertex_distance,
        )

        # match srce and trgt isobath
        shift_vectors = regbat.match_src_trgt(
            srce_trgt_joined, min_overlap=self.min_overlap, vertex_distance=self.vertex_distance
        )

        return shift_vectors


def apply_on_sounder_files(
    i_paths: List[str],
    i_dtm: str,
    o_path: str,
    isobath_interval: float,
    cell_size: float = 4,
    max_dist: float = 100,
    min_overlap: int = 6,
    vertex_distance: float = 4,
    overwrite: bool = False,
) -> Dict:
    """
    Calls an isobath registration process on a set of XSF files given a reference DTM.
    @param i_paths : input sounder file paths
    @param i_dtm : input reference DTM (must be projected)
    @param o_path : output NVI file path
    @param overwrite : True to overwrite output files if needed
    @param cell_size : Cell size for individual temporary DTM computed from each sounder file
    @param isobath_interval : Elevation interval between isobath (m)
    @param max_dist : maximum allowed shift amplitude (m)
    @param min_overlap : minimum number of matching isobath vertices needed to compute a robust transformation
    @param vertex_distance : isobath resampling distance, affect both source and target
    """

    # progress monitor and logger init
    logger = log.logging.getLogger(__name__)
    monitor = DefaultMonitor
    n = len(i_paths)
    monitor.begin_task("Registration process", n)

    # get spatial reference from target DTM
    with dtm_driver.open_dtm(i_dtm) as trgt_dtm:
        spatial_reference = trgt_dtm.dtm_file.spatial_reference
        if not spatial_reference.IsProjected():
            raise IOError(
                f'Reference DTM must be projected\nProjection not supported : {spatial_reference.GetAttrValue("PROJECTION")} '
            )

    # target dtm.nc -> isobath .shp
    trgt_isobath_path = tempfile.mktemp(suffix=".shp", dir=os.path.dirname(o_path))
    isobath_exporter = Dtm2Isobath(i_paths=[i_dtm], o_paths=[trgt_isobath_path], isobath_interval=isobath_interval)
    isobath_exporter()

    # Convert target isobaths to geodataframe
    trgt_isobaths = gpd.read_file(trgt_isobath_path)

    shift_vectors = []
    nav_to_shift = gpd.GeoDataFrame()

    # process each input sounder file
    for i_path in i_paths:
        logger.info(f"Registering {i_path}")

        # source sounder file (.mbg, .xsf) -> isobath .shp
        srce_isobath_path = tempfile.mktemp(suffix=".shp", dir=os.path.dirname(o_path))
        export_sounderfile_to_isobath(
            i_path=i_path,
            o_path=srce_isobath_path,
            spatial_reference=spatial_reference,
            cell_size=cell_size,
            isobath_interval=isobath_interval,
            overwrite=overwrite,
        )
        # source sounder isobaths file (.shp) -> geodataframe
        srce_isobaths = gpd.read_file(srce_isobath_path)

        # source sounder file navigation (.mbg, .xsf) -> geodataframe
        with navigation_factory.from_file(i_path) as nav:
            srce_nav = navigation_exporter.to_geodataframe(nav)

        # Init nav_shift process
        registration_process = IsobathRegistrationProcess(
            trgt_isobaths=trgt_isobaths,
            srce_isobaths=srce_isobaths,
            srce_nav=srce_nav,
            max_dist=max_dist,
            min_overlap=min_overlap,
            vertex_distance=vertex_distance,
        )
        # Call nav_shift process and retrieve shift vectors
        shift_vectors += registration_process()

        # concat srce navigations to shift afterward
        nav_to_shift = pd.concat([nav_to_shift, srce_nav])

        # everything went well, delete temporary srce shape-relative file
        delete_files(with_pattern=os.path.splitext(srce_isobath_path)[0])
        monitor.worked(1)

    monitor.done()

    # move original navigation using vectors, if applicable
    if len(shift_vectors) > 0:
        nav_shifter_process = NavShifter(nav=nav_to_shift, shift_vectors=shift_vectors)
        shifted_nav = nav_shifter_process()

        # Produce output navigation files
        out_nav = navigation_factory.from_geodataframe(shifted_nav)
        navigation_exporter.to_nvi(nav=out_nav, o_path=o_path, source_filenames=i_paths, overwrite=overwrite)
    else:
        logger.error(f"No shift vector found")

    # everything went well, delete temporary trgt shape-relative file
    delete_files(with_pattern=os.path.splitext(trgt_isobath_path)[0])

    return {"shift_vectors": shift_vectors, "outfile": [o_path]}


def export_sounderfile_to_isobath(
    i_path: str,
    o_path: str,
    spatial_reference: osr.SpatialReference,
    cell_size: float,
    isobath_interval: float,
    overwrite: bool,
    gap_filling: bool = True,
):
    """
    Exports isobaths from MBG/XSF files, through temporary DTM computation.
    """
    # .xsf.nc -> dtm.nc
    o_dtm_path = tempfile.mktemp(suffix=".dtm.nc", dir=os.path.dirname(o_path))

    dtm_exporter = SounderToDtmExporter(
        i_paths=[i_path],
        o_paths=[o_dtm_path],
        target_resolution=cell_size,
        target_spatial_reference=spatial_reference.ExportToProj4(),
        gap_filling=gap_filling,
        overwrite=overwrite,
    )
    dtm_exporter()

    # .dtm.nc -> isobath .shp
    isobath_exporter = Dtm2Isobath(
        i_paths=[o_dtm_path], o_paths=[o_path], isobath_interval=isobath_interval, overwrite=overwrite
    )
    isobath_exporter()

    # everything went well, delete temporary dtm
    os.remove(o_dtm_path)
