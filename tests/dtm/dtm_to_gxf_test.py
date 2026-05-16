import numpy.testing as npt
import pytest

from pyat.dtm import dtm_driver
from pyat.dtm.convert.gdal_raster_to_dtm import export_gdal_raster_file_to_dtm
from pyat.dtm.export.dtm_to_gxf import Dtm2GXF

# Register the dtm_generator fixture so pytest can find it
pytest_plugins = ["tests.dtm.dtm_generator"]


@pytest.fixture
def dtm_test_file_full(dtm_file_factory):
    """
    DTM file 50x50 with all layers and no-data zones.

    Returns:
        str: Path to DTM file
        list : no-data zones
    """
    return dtm_file_factory(grid_size=50, with_nodata=True, layers="full")


def test_wrapper_with_dtm_driver(tmp_path, dtm_test_file_full):
    """Test Dtm2GXF wrapper with DTM driver and generated test file."""
    # generate test DTM file with random values
    i_dtm_path = dtm_test_file_full

    # Create output GXF using wrapper
    o_gxf_path = tmp_path / "output.gxf"
    dtm2gxf = Dtm2GXF(
        i_paths=[i_dtm_path],
        o_paths=[str(o_gxf_path)],
        overwrite=True,
        file_type="kingdom",
    )
    dtm2gxf()

    # Verify GXF file was created
    assert o_gxf_path.exists()
    # Verify GXF structure
    txt = o_gxf_path.read_text(encoding="utf-8")
    assert "#GRID" in txt
    assert "#POINTS" in txt
    assert "#ROWS" in txt
    assert "#ZMAXIMUM" in txt
    assert "#ZMINIMUM" in txt

    # convert GXF back to DTM, using gdal GXF reading driver
    o_dtm_path = tmp_path / "output.dtm.nc"
    export_gdal_raster_file_to_dtm(str(o_gxf_path), str(o_dtm_path), overwrite=True)

    with dtm_driver.open_dtm(i_dtm_path) as dtm:
        i_dtm_x = dtm.get_x_axis()[:]
        i_dtm_y = dtm.get_y_axis()[:]
        i_dtm_elevation = dtm[dtm_driver.DtmConstants.ELEVATION_NAME][:]
        i_dtm_crs = dtm.dtm_file.spatial_reference

        with dtm_driver.open_dtm(str(o_dtm_path)) as dtm_from_gxf:
            o_dtm_x = dtm_from_gxf.get_x_axis()[:]
            o_dtm_y = dtm_from_gxf.get_y_axis()[:]
            o_dtm_elevation = dtm_from_gxf[dtm_driver.DtmConstants.ELEVATION_NAME][:]
            o_dtm_crs = dtm_from_gxf.dtm_file.spatial_reference

            # compare x from GXF vs DTM
            npt.assert_allclose(
                actual=o_dtm_x,
                desired=i_dtm_x,
                err_msg=f"X axis values from GXF differs from DTM up to {dtm2gxf.precision} m",
            )
            # compare y from GXF vs DTM
            npt.assert_allclose(
                actual=o_dtm_y,
                desired=i_dtm_y,
                err_msg=f"Y axis values from GXF differs from DTM up to {dtm2gxf.precision} m",
            )
            # Compare actual elevations vs expected elevations with tolerance
            npt.assert_allclose(
                actual=o_dtm_elevation,
                desired=i_dtm_elevation,
                atol=dtm2gxf.precision,
                err_msg=f"Elevation values from GXF differs from DTM up to {dtm2gxf.precision} m",
                verbose=True,
                equal_nan=True,
            )

            # compare CRS from GXF vs DTM (should be the same)
            assert o_dtm_crs.IsSame(i_dtm_crs)
