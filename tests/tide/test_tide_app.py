import pathlib
import tempfile

from pyat.tide.tide_app import FESTidePredictorForNavigationFiles
from tests.generator.xsf_generator import XsfGenerator

from .tide_datasets_configuration import get_tide_datasets_path


def generate_xsf(folder: str) -> str:
    """
    Creates a XSF file with
        2 navigation positions :
            W 4.005 / N 48.0
            W 4.0 / N 48.005
        4 beams on both sides of the 2 navigation positions :
            BEAM1 -> W 4.006669079479038 / N 48.00075171097127, depth = 10m
            BEAM2 -> W 4.003330969015191 / N 47.99924826467987, depth = 20m
            BEAM3 -> W 4.001669240784506 / N 48.00575171031162, depth = 10m
            BEAM4 -> W 3.998330807722907 / N 48.004248265335306, depth = 20m
    """
    generator = XsfGenerator(folder)
    return generator.initialize_file(
        latitude_min_deg=48.0,
        latitude_max_deg=48.005,
        longitude_min_deg=-4.005,
        longitude_max_deg=-4.0,
        ping_count=2,
        beam_count=2,
        min_depth_m=10.0,
        max_depth_m=20.0,
    )


# Example usage of the FESTidePredictorForNavigationFiles class
def test_fes_tide_predictor():
    with tempfile.TemporaryDirectory() as o_path:
        xsf_path = generate_xsf(o_path)
        tide_output_path = pathlib.Path(o_path, "tide_prediction.tide.nc")
        estimator = FESTidePredictorForNavigationFiles(
            input_files=[xsf_path],
            output_files=[str(tide_output_path)],
            model_dir=str(get_tide_datasets_path().absolute()),
            # tide_server="http://localhost:4400",
            overwrite=True,
        )
        estimator()
