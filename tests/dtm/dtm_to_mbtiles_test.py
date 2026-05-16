import pytest
from osgeo import gdal

from pyat.dtm.export.dtm_to_mbtiles import Dtm2Mbtiles

# Register the dtm_generator fixture so pytest can find it
pytest_plugins = ["tests.dtm.dtm_generator"]

OUTPUT_MBTILES_FILENAME = "output.mbtiles"


@pytest.fixture
def dtm_test_file(dtm_file_factory):
    """
    DTM file 50x50 with all layers and no-data zones.

    Returns:
        str: Path to DTM file
    """
    return dtm_file_factory(grid_size=50, with_nodata=True, layers="full")


def test_export_file_exists_no_overwrite(tmp_path, dtm_test_file):
    """Test export skips existing file when overwrite is False"""
    # generate test DTM file with random values
    i_dtm_path = dtm_test_file
    # Create an existing output file
    o_path = tmp_path / OUTPUT_MBTILES_FILENAME
    o_path.touch()
    exporter = Dtm2Mbtiles(i_paths=[i_dtm_path], o_paths=[str(o_path)], overwrite=False)
    exporter()
    # File should not be in resulting_files since it wasn't processed
    assert str(o_path) not in exporter.resulting_files


def test_export_one_file(tmp_path, dtm_test_file):
    """Test Dtm2Mbtiles creates MBTiles file from DTM"""
    # generate test DTM file with random values
    i_dtm_path = dtm_test_file
    o_mbtiles_path = tmp_path / OUTPUT_MBTILES_FILENAME
    dtm2mbtiles = Dtm2Mbtiles(i_paths=[i_dtm_path], o_paths=[str(o_mbtiles_path)], overwrite=True)
    result = dtm2mbtiles()
    # Verify results structure
    assert "outfile" in result
    assert isinstance(result["outfile"], list)
    # If export succeeded, file should be in results
    assert o_mbtiles_path.exists()
    if o_mbtiles_path.exists():
        assert str(o_mbtiles_path) in result["outfile"]
        assert str(o_mbtiles_path) in dtm2mbtiles.resulting_files
        assert o_mbtiles_path.stat().st_size > 0  # Check file is not empty
        # test that the resulting file is a valid MBTiles file by checking that it is readable by GDAL
        ds = gdal.Open(str(o_mbtiles_path))
        assert ds is not None
        ds = None  # Close dataset


def test_export_multiple_files(tmp_path, dtm_test_file):
    """Test Dtm2Mbtiles can handle multiple input/output file pairs"""
    o_paths = [tmp_path / "output1.mbtiles", tmp_path / "output2.mbtiles"]
    dtm2mbtiles = Dtm2Mbtiles(
        i_paths=[dtm_test_file, dtm_test_file], o_paths=[str(o_path) for o_path in o_paths], overwrite=True
    )
    result = dtm2mbtiles()
    assert "outfile" in result
    assert isinstance(result["outfile"], list) and len(result["outfile"]) == 2
    for o_path in o_paths:
        assert o_path.exists()
        if o_path.exists():
            assert str(o_path) in result["outfile"]
            assert str(o_path) in dtm2mbtiles.resulting_files
            assert o_path.stat().st_size > 0  # Check file is not empty
            # test that the resulting file is a valid MBTiles file by checking that it is readable by GDAL
            ds = gdal.Open(str(o_path))
            assert ds is not None
            ds = None  # Close dataset
