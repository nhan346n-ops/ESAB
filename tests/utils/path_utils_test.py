from pyat.utils import path_utils


def test_splitext_of_fname():
    f = path_utils.splitext_of_fname
    assert f('./a/data.xsf') == ('./a/data', 'xsf')
    assert f('./a/data.xsf.nc') == ('./a/data', 'xsf.nc')

def test_basename_of_fname():
    f = path_utils.basename_of_fname
    assert f('./a/data.xsf') == 'data'
    assert f('./a/data.xsf.nc') == 'data'
