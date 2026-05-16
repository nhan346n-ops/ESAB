import numpy as np

# define NC_FILL_CHAR	((char)0)
# define NC_FILL_SHORT	((short)-32767)
# define NC_FILL_INT	(-2147483647)
# define NC_FILL_UBYTE	(255)
# define NC_FILL_USHORT	(65535)
# define NC_FILL_INT64	((long long)-9223372036854775806LL)
# define NC_FILL_UINT64	((unsigned long long)18446744073709551614ULL)
# define NC_FILL_STRING	((char *)"")


def get_default_fillvalue(dtype, raise_exception = True):
    """Compute and return a default netcdf fill_value for a numpy type"""
    if np.issubdtype(dtype, np.floating):
        if dtype == np.float32:
            return np.nan
        if dtype == np.float64:
            return np.nan
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if np.issubdtype(dtype, np.signedinteger):
            return info.min
        if np.issubdtype(dtype, np.unsignedinteger):
            return info.max
    elif np.issubdtype(dtype, np.character):
        # string subtype
        return ""
    if raise_exception:
        raise NotImplementedError(f"Getdefault fill value failed : type {dtype} is not supported ")
    return None

def test_default_values():
    """Validate the expected default values"""
    # define NC_FILL_BYTE	((signed char)-127)
    assert (get_default_fillvalue(np.int8)) == (-128)
    # define NC_FILL_CHAR	((char)0)
    # define NC_FILL_SHORT	((short)-32767)
    assert (get_default_fillvalue(np.int16)) == (-32768)
    # define NC_FILL_INT	(-2147483647)
    assert (get_default_fillvalue(np.int32)) == (-2147483648)
    # define NC_FILL_INT64	((long long)-9223372036854775806LL)
    assert (get_default_fillvalue(np.int64)) == (-9223372036854775808)

    # define NC_FILL_FLOAT	(9.9692099683868690e+36f) /* near 15 * 2^119 */
    assert np.isnan(get_default_fillvalue(np.float32))
    # define NC_FILL_DOUBLE	(9.9692099683868690e+36)
    assert np.isnan(get_default_fillvalue(np.float64))
    # define NC_FILL_UBYTE	(255)
    assert (get_default_fillvalue(np.uint8)) == (255)
    # define NC_FILL_USHORT	(65535)
    assert (get_default_fillvalue(np.uint16)) == (65535)
    # define NC_FILL_UINT	(4294967295U)
    assert (get_default_fillvalue(np.uint32)) == (4294967295)
    # define NC_FILL_UINT64	((unsigned long long)18446744073709551614ULL)
    assert (get_default_fillvalue(np.uint64)) == (18446744073709551615)
    # define NC_FILL_STRING	((char *)"")
    assert (get_default_fillvalue(str)) == ("")
