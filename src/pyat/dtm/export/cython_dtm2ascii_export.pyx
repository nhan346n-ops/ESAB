# cython: infer_types=True
import numpy as np
cimport numpy as np
cimport cython
from libc.stdio cimport FILE, fopen, fprintf,printf, fclose
from libc.math cimport isnan,abs

cdef enum FORMATS :
    XYZ = 0
    XZY = 1
    YXZ = 2
    YZX = 3
    ZXY = 4
    ZYX = 5

cdef dict print_formats_xyz = {
    XYZ : "%.8f;%.8f;%.2f\n",
    XZY : "%.8f;%.2f;%.8f\n",
    YXZ : "%.8f;%.8f;%.2f\n",
    YZX : "%.8f;%.2f;%.8f\n",
    ZXY : "%.2f;%.8f;%.8f\n",
    ZYX : "%.2f;%.8f;%.8f\n"
}

"""
Cython code handling export to xyz and export from emo files
"""
@cython.boundscheck(False)
@cython.wraparound(False)
def export_xyz(output_path,export_missing_values, double[::1] array_longitude, double[::1] array_latitude, float[:,::1] array_depth, 
        str column_separator,
        const int column_order):
    cdef FILE * ptr_fw
    cdef Py_ssize_t  x_max = array_longitude.shape[0]
    cdef Py_ssize_t  y_max = array_latitude.shape[0]
    assert array_depth.shape[1] == x_max
    assert array_depth.shape[0] == y_max
    ptr_fw  = fopen(output_path, "w")

    cdef double latitude
    cdef double longitude
    cdef bint boolean_variable = True
    cdef double elevation
    
    if export_missing_values is False:
        boolean_variable = False
    if (ptr_fw != NULL):
        for y in range(y_max):
            latitude = array_latitude[y]
            for x in range(x_max):
                longitude = array_longitude[x]
                #fout.write("%f;%f;%f\n", array_longitude[x],array_latitude[y],array_depth[y,x]))
                elevation = array_depth[y,x]
                write_in_file(ptr_fw, column_order, column_separator, longitude, latitude, elevation, export_missing_values)
        fclose(ptr_fw)
    else:
        print("Unable to open file!\n")

cdef write_in_file(FILE * ptr_fw, const int column_order, str column_separator, double longitude, double latitude, double elevation, bint export_missing_values):
    if isnan(elevation) and not export_missing_values :
        return
    
    cdef str format = print_formats_xyz[column_order] if not isnan(elevation) else "%.8f;%.8f\n"
    format = format.replace(";", column_separator)

    if isnan(elevation) :
        if column_order == XYZ :
            fprintf(ptr_fw, format.encode(), longitude, latitude)
        elif column_order == XZY :
            fprintf(ptr_fw, format.encode(), longitude, latitude)
        elif column_order == YXZ :
            fprintf(ptr_fw, format.encode(), latitude, longitude )
        elif column_order == YZX :
            fprintf(ptr_fw, format.encode(), latitude, longitude )
        elif column_order == ZXY :
            fprintf(ptr_fw, format.encode(), longitude, latitude )
        elif column_order == ZYX :
            fprintf(ptr_fw, format.encode(), latitude, longitude )
    else :
        if column_order == XYZ :
            fprintf(ptr_fw, format.encode(), longitude, latitude, elevation)
        elif column_order == XZY :
            fprintf(ptr_fw, format.encode(), longitude, elevation, latitude)
        elif column_order == YXZ :
            fprintf(ptr_fw, format.encode(), latitude, longitude, elevation )
        elif column_order == YZX :
            fprintf(ptr_fw, format.encode(), latitude, elevation, longitude )
        elif column_order == ZXY :
            fprintf(ptr_fw, format.encode(), elevation, longitude, latitude )
        elif column_order == ZYX :
            fprintf(ptr_fw, format.encode(), elevation, latitude, longitude)

cdef void print_unicode(FILE * ptr_fw,unicode value):
    py_byte_string = value.encode('UTF-8')
    cdef char * c_string = py_byte_string
    fprintf(ptr_fw,"%s",c_string)

cdef void print_value_count_value(FILE * ptr_fw,int value):
    if value >=0:
        fprintf(ptr_fw,"%d",value)
    return

cdef void print_interpolation_value(FILE * ptr_fw,signed char value):
    if value >= 0:
        fprintf(ptr_fw, "%d", value)

cdef void print_float_value(FILE * ptr_fw,float value):
    # print a formatted value with 2 decimal precision, if nan print nothing
    if not isnan(value):
        fprintf(ptr_fw,"%.2f",value)

@cython.boundscheck(False)
@cython.wraparound(False)
def export_emo(output_path,
                export_missing_values,
                double[::1] array_longitude,
                double[::1] array_latitude,
                float[:,::1] array_depth,
                float[:,::1] mins,
                float[:,::1] max,
                float[:,::1] stdev_variable,
                int[:,::1] value_count,
                signed char[:,::1] interpolation_flag,
                float[:,::1] smoothed_depth,
                int[:,::1] cdi_index,
                cdi_values,
                cprd_values
                ):
    cdef FILE * ptr_fw
    cdef Py_ssize_t  x_max = array_longitude.shape[0]
    cdef Py_ssize_t  y_max = array_latitude.shape[0]
    cdef Py_ssize_t max_cdi_index = len(cdi_values)

    assert array_depth.shape[1] == x_max
    assert array_depth.shape[0] == y_max
    ptr_fw = fopen(output_path, "w")
    cdef double latitude
    cdef double longitude
    cdef bint missing_value_flag = True
    cdef float elevation
    cdef int cdi_index_value
    cdef char * some_c_string
    if export_missing_values is False:
        missing_value_flag = False
    if (ptr_fw != NULL):
        for y in range(y_max):
            latitude = array_latitude[y]
            for x in range(x_max):
                longitude = array_longitude[x]
                # fout.write("%f;%f;%f\n", array_longitude[x],array_latitude[y],array_depth[y,x]))
                elevation = array_depth[y, x]
                if isnan(elevation):
                    if missing_value_flag:
                        fprintf(ptr_fw, "%.8f;%.8f;;;;;;;;;;;\n",longitude,latitude)
                else:
                    fprintf(ptr_fw, "%.8f;%.8f;",  longitude,latitude)
                    #depth are positive up for emo files, so we revert min and max and mult by -1
                    if max != None:
                        print_float_value(ptr_fw, -max[y, x])
                    fprintf(ptr_fw, ";")
                    if mins != None:
                        print_float_value(ptr_fw, -mins[y, x])
                    fprintf(ptr_fw, ";")
                    print_float_value(ptr_fw,-elevation)
                    fprintf(ptr_fw, ";")
                    if stdev_variable != None:
                        print_float_value(ptr_fw, stdev_variable[y, x])
                    fprintf(ptr_fw, ";")
                    if value_count != None:
                        print_value_count_value(ptr_fw,value_count[y,x],)
                    fprintf(ptr_fw, ";")
                    if interpolation_flag != None:
                        interpolation_flag_value = interpolation_flag[y,x]
                        # do not set interpolation flag if missing (127) or disabled (0)
                        if interpolation_flag_value != 0 and interpolation_flag_value != 127:
                            print_interpolation_value(ptr_fw,interpolation_flag[y,x])
                    fprintf(ptr_fw, ";")
                    if smoothed_depth != None:
                        print_float_value(ptr_fw, -smoothed_depth[y, x])
                        fprintf(ptr_fw, ";")
                        print_float_value(ptr_fw, abs(elevation-smoothed_depth[y, x]))
                    else:
                        fprintf(ptr_fw, ";")
                    fprintf(ptr_fw, ";")
                    if cdi_index!= None:
                        #cdi and CPRD
                        cdi_index_value= cdi_index[y,x]
                        if cdi_index_value < 0:
                            fprintf(ptr_fw, ";")  # invalid index
                        elif cdi_index_value > max_cdi_index:  # invalid index
                            fprintf(ptr_fw, ";")  # invalid index
                        else:
                            some_string = cdi_values[cdi_index_value]
                            print_unicode(ptr_fw, some_string)
                            fprintf(ptr_fw, ";")  # invalid index
                            some_string = cprd_values[cdi_index_value]
                            print_unicode(ptr_fw, some_string)
                    else:
                        fprintf(ptr_fw, ";")
                    fprintf(ptr_fw, ";") # add a useless delimiter to be compliant with emodnet

                    fprintf(ptr_fw, "\n")
        fclose(ptr_fw)



    else:
        raise IOError(u"Unable to open file {output_path}\n")
