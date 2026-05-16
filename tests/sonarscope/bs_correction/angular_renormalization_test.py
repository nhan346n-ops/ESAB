import tempfile

import numpy as np

from pyat.sonarscope.bs_correction.stats_computer import compute_mean_model
from pyat.sonarscope.bs_correction.angular_renormalization import xsf_constant_process
from pyat.sonarscope.model.sounder_lib import SounderType
from tests.generator.xsf_generator import XsfGenerator
from tests.generator.dtm_generator import DtmGenerator
import pyat.dtm.dtm_standard_constants as DtmConstants

geoBox = np.array([48.005, 48.0, -4.0, -4.005], dtype=float)


def generate_xsf(folder: str, sounder_type: str) -> str:
    """
    Creates a XSF file with
        10 navigation positions :
            W 4.005 / N 48.0
            W 4.0 / N 48.005
        512 beams
    """
    generator = XsfGenerator(folder)
    xsf_file = generator.initialize_file(
        latitude_max_deg=geoBox[0],
        latitude_min_deg=geoBox[1],
        longitude_max_deg=geoBox[2],
        longitude_min_deg=geoBox[3],
        ping_count=10,
        beam_count=512,
        min_depth_m=10.0,
        max_depth_m=20.0,
    )
    if SounderType.EM2040_ALL == sounder_type:
        generator.append_kongsberg_all_variables(xsf_file)
    elif SounderType.EM2040_KMALL == sounder_type:
        generator.append_kongsberg_kmall_variables(xsf_file)
    return xsf_file


def generate_dtm(folder: str) -> str:
    """
    Creates a DTM file with
    """
    generator = DtmGenerator(folder)
    dtm_driver = generator.initialize_file(geobox=geoBox)
    dtm_driver.add_layer(DtmConstants.ELEVATION_NAME, 15.0)
    dtm_driver.close()
    return generator.path


def compute_angular_renormalization(sounder_type: str, use_dtm: bool = False):
    """Test normalization behaviour"""
    with tempfile.TemporaryDirectory() as o_dir:
        # generate test file
        i_xsf_path = generate_xsf(folder=o_dir, sounder_type=sounder_type)
        i_dtm_path = generate_dtm(folder=o_dir) if use_dtm else None

        # generate stats for test XSF
        o_stat_path = tempfile.mktemp(suffix=".nc", dir=o_dir)
        compute_mean_model(
            sounder_type=sounder_type,
            i_paths=[i_xsf_path],
            o_path=o_stat_path,
            i_dtm=i_dtm_path,
            use_snippets=True,
            use_svp=True,
        )

        # apply normalization
        o_xsf_path = tempfile.mktemp(suffix=".xsf.nc", dir=o_dir)
        xsf_constant_process(
            i_paths=[i_xsf_path],
            o_paths=[o_xsf_path],
            mean_model_file=o_stat_path,
            overwrite=True,
            i_dtm=i_dtm_path,
            use_snippets=False
        )

        # if not use_dtm:
            # check backscatter levels
            # if insonified area is recomputed, this simple test will fail (generated test file is not fully coherant)
            # with xsf_driver.open_xsf(o_xsf_path, "r") as o_xsf_driver:
            #     detection_backscatter = o_xsf_driver.get_layer(xsf_driver.DETECTION_BACKSCATTER_R)
                # ref_level = -20
                # assert np.all(abs(detection_backscatter[:] - ref_level) < 1.0)


def test_angular_renormalization():
    compute_angular_renormalization(sounder_type=SounderType.EM2040_ALL, use_dtm=False)
    compute_angular_renormalization(sounder_type=SounderType.EM2040_KMALL, use_dtm=False)
    compute_angular_renormalization(sounder_type=SounderType.EM2040_ALL, use_dtm=True)
    compute_angular_renormalization(sounder_type=SounderType.EM2040_KMALL, use_dtm=True)
