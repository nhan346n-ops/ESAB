try:
    import pyat.common.convolve._convolve
    print('convolve: OK')
except Exception as e:
    print('convolve: err:', e)

try:
    import pyat.dtm.export.cython_dtm2ascii_export
    print('cython export: OK')
except Exception as e:
    print('cython export: err:', e)
