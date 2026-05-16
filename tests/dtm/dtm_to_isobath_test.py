import pytest
from osgeo import ogr

from pyat.dtm.export.dtm_to_isobath import Dtm2Isobath

# Register the dtm_generator fixture so pytest can find it
pytest_plugins = ["tests.dtm.dtm_generator"]

OUTPUT_ISOBATH_FILENAME = "output.gpkg"


@pytest.fixture
def dtm_test_file(dtm_file_factory):
    """
    DTM file 50x50 with all layers and no-data zones.

    Returns:
        str: Path to DTM file
    """
    return dtm_file_factory(grid_size=50, with_nodata=True, layers="full")


def test_export_one_file(tmp_path, dtm_test_file):
    """Test Dtm2Isobath creates Isobath file from DTM"""
    # generate test DTM file with random values
    i_dtm_path = dtm_test_file
    o_isobath_path = tmp_path / OUTPUT_ISOBATH_FILENAME
    exporter = Dtm2Isobath(i_paths=[i_dtm_path], o_paths=[str(o_isobath_path)], overwrite=True)
    result = exporter()
    # Verify results structure
    assert "outfile" in result
    assert isinstance(result["outfile"], list)
    # If export succeeded, file should exist
    assert o_isobath_path.exists()
    if o_isobath_path.exists():
        assert str(o_isobath_path) in result["outfile"]
        assert str(o_isobath_path) in exporter.resulting_files
        # Check file is not empty
        assert o_isobath_path.stat().st_size > 0
        # test that the resulting file is a valid file that it is readable by GDAL/OGR
        ds = ogr.Open(str(o_isobath_path))
        assert ds is not None
        # Check that there is only one layer in the output file
        assert ds.GetLayerCount() == 1
        # Check that the layer name is correctly named "isobath"
        isobath_layer = ds.GetLayerByIndex(0)
        assert isobath_layer.GetName() == "isobath"
        # check that the layer has 2 fields
        isobath_layer_dfn = isobath_layer.GetLayerDefn()
        assert isobath_layer_dfn.GetFieldCount() == 2
        # check that they are correctly named "ID" and "elev"
        for i in range(isobath_layer_dfn.GetFieldCount()):
            assert isobath_layer_dfn.GetFieldDefn(i).GetName() in ["ID", "elev"]
        ds = None  # Close dataset
