import glob
import os.path
import tempfile

from osgeo import ogr
from pynvi.legacy.exporter import FILE_FIELD

from pyat.navigation.convert_to_shp_app import Convert2ShpApp
from tests.file_test_installer import get_test_path
from tests.generator.xsf_generator import XsfGenerator

MBG_PATH = get_test_path().absolute() / "mbg" / "0136_20120607_083636_ShipName_ref.mbg"


def test_mbg():
    try:
        output_file_pattern = tempfile.mktemp()
        # remove old generated file
        output_file = output_file_pattern + ".shp"
        assert os.path.exists(MBG_PATH)
        converter = Convert2ShpApp(i_paths=[MBG_PATH], o_path=output_file)
        converter()
        assert os.path.exists(output_file)
        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.Open(output_file, 0)  # 0 means read-only. 1 means writeable
        layer = data_source.GetLayer()
        feature = layer.GetFeature(0)
        geom = feature.GetGeometryRef()
        assert geom.GetPointCount() == 1176

    finally:
        # do some silent cleanup
        data_source = None
        silent_delete(output_file_pattern=output_file_pattern)


def generate_xsf(folder: str) -> str:
    """
    Creates a plain XSF file
    """
    generator = XsfGenerator(folder)
    return generator.initialize_file(
        latitude_min_deg=48.0,
        latitude_max_deg=48.005,
        longitude_min_deg=-4.005,
        longitude_max_deg=-4.0,
        ping_count=20,
        beam_count=20,
        min_depth_m=10.0,
        max_depth_m=20.0,
    )


def test_xsf():
    with tempfile.TemporaryDirectory() as o_dir:
        output_file_pattern = tempfile.TemporaryFile(dir=o_dir).name
        try:
            # generate test file
            i_xsf_path = generate_xsf(folder=o_dir)
            assert os.path.exists(i_xsf_path)
            output_file = output_file_pattern + ".shp"
            converter = Convert2ShpApp(i_paths=[i_xsf_path], o_path=output_file)
            converter()
            assert os.path.exists(output_file)
            driver = ogr.GetDriverByName("ESRI Shapefile")
            data_source = driver.Open(output_file, 0)  # 0 means read-only. 1 means writeable
            layer = data_source.GetLayer()
            feature = layer.GetFeature(0)
            geom = feature.GetGeometryRef()
            assert geom.GetPointCount() == 20
            assert feature.GetField(FILE_FIELD) == os.path.basename(i_xsf_path)

        finally:
            # do some silent cleanup
            data_source = None
            silent_delete(output_file_pattern=output_file_pattern)


def silent_delete(output_file_pattern: str):
    try:
        to_delete = glob.glob(output_file_pattern + ".*")
        for f in to_delete:
            os.remove(f)
    except Exception as e:
        print(e)
