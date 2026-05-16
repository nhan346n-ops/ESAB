import os.path
import tempfile

from pyat.xsf.navshift.isobath_registration import apply_on_sounder_files
from pyat.dtm.experimental.geometric_translation import GeometricTranslationProcess
from pyat.sounder.sounder_to_dtm import SounderToDtmExporter
from tests.file_test_installer import get_test_path

MBG_PATH = get_test_path() / "mbg" / "0136_20120607_083636_ShipName_ref.mbg"
cell_size = 20
isobath_interval = 100
spatial_reference = "proj=utm +zone=33 +datum=WGS84 +units=m +no_defs +type=crs"


def test_isobath_registration():
    """
    Tests isobath registration behaviour.
    """
    with tempfile.TemporaryDirectory() as o_dir:
        i_mbg_path = str(MBG_PATH)
        o_dtm_path = tempfile.mktemp(suffix=".dtm.nc", dir=o_dir)
        dtm_exporter = SounderToDtmExporter(
            i_paths=[i_mbg_path],
            o_paths=[o_dtm_path],
            target_resolution=cell_size,
            target_spatial_reference=spatial_reference,
            gap_filling=True,
            overwrite=True,
        )
        dtm_exporter()

        # translate it by 3 * cell_size (20m) in each direction -> 60 m offset
        suffix = "_translated"
        o_dtm_translated_path = os.path.splitext(o_dtm_path)[0] + suffix + ".dtm.nc"
        dtm_translater = GeometricTranslationProcess(
            i_paths=[o_dtm_path],
            o_paths=[o_dtm_translated_path],
            rows="3",
            columns="3",
        )
        dtm_translater()

        # try to register sounder file to offset DTM
        o_nvi = tempfile.mktemp(suffix=".nvi.nc", dir=o_dir)
        result = apply_on_sounder_files(
            i_paths=[i_mbg_path],
            i_dtm=o_dtm_translated_path,
            o_path=o_nvi,
            isobath_interval=isobath_interval,
            cell_size=cell_size,
        )

        # load shift vectors, and check that offset is 60m in both directions
        for vector in result['shift_vectors']:
            assert vector.x == 60
            assert vector.y == 60
