"""A dictionary of known variables for xsf files"""

from abc import ABCMeta, abstractmethod
from typing import Dict, Protocol

import netCDF4 as nc
import numpy as np
import sonar_netcdf.sonar_groups as sg
from pyproj import Geod
from scipy.interpolate import interp1d, interpn
from scipy.spatial.transform import Rotation as R

import pyat.utils.pyat_logger as log
from pyat.sonarscope.common import xarray_utils as ut
from pyat.sonarscope.model.constants import DEFAULT_BEAM_GROUP_IDENT
from pyat.utils import signal
from pyat.utils.netcdf import get_default_fillvalue
from pyat.utils.netcdf_utils import get_variable

logger = log.logging.getLogger(__file__)


def createXsfVariable(variable_name: str, instance, group, ident=None, fill_value=None):
    """Recurse group and retrieve all variables declared inside it, store result in a dictionary name:variable path"""

    func_descriptor = group.__dict__[variable_name.upper()]
    # pylint:disable = unnecessary-dunder-call
    func = func_descriptor.__get__(instance)
    if ident is not None:
        variable_path = func(ident=ident)
    else:
        variable_path = func()
    return XsfVariable(variable_path=variable_path, fill_value=fill_value)


class VariableInterface(metaclass=ABCMeta):
    @abstractmethod
    def get_fill_value(self, nc_dataset):
        """return fill value"""

    @abstractmethod
    def get_dimensions(self, nc_dataset):
        """return the dimension for the variable"""

    @abstractmethod
    def get_values(self, nc_dataset):
        """read variable values"""

    @abstractmethod
    def get_attributes(self, nc_dataset):
        """read netcdf attributes"""


class XsfVariable(VariableInterface):
    """A variable obtained from XSF, directly point to a variable in a netcdf file"""

    def __init__(self, variable_path: str, fill_value):
        self.variable_path = variable_path
        self.fill_value = fill_value

    def get_fill_value(self, nc_dataset: nc.Dataset):
        if self.fill_value is not None:
            return self.fill_value
        var = nc_dataset[self.variable_path]
        if hasattr(var, "scale_factor") and var.scale_factor != 1:
            return get_default_fillvalue(var.scale_factor.dtype)
        if hasattr(var, "_FillValue"):
            return var._FillValue
        return get_default_fillvalue(var.dtype)

    def get_dimensions(self, nc_dataset: nc.Dataset):
        var = nc_dataset[self.variable_path]
        return ut.get_dimensions(variable=var)

    def get_values(self, nc_dataset: nc.Dataset):
        """read variable values"""
        var = nc_dataset[self.variable_path]
        var.set_auto_mask(True)
        data = var[:]
        data[data.mask] = self.get_fill_value(nc_dataset=nc_dataset)
        data[~np.isfinite(data)] = self.get_fill_value(nc_dataset=nc_dataset)
        return data

    def get_attributes(self, nc_dataset: nc.Dataset):
        """read netcdf attributes"""
        var = nc_dataset[self.variable_path]
        attr = ut.get_nc_attribute(nc_variable_or_group=var)
        # remove offset and scale parameters as it is already applied
        if "scale_factor" in attr.keys():
            attr.pop("scale_factor")
        if "add_offset" in attr.keys():
            attr.pop("add_offset")
        return attr


class InterpingVariable(VariableInterface):
    def get_dimensions(self, nc_dataset: nc.Dataset):
        # we have the same dimension as platform_longitudes
        platform_longitudes = nc_dataset[sg.BeamGroup1Grp.PLATFORM_LONGITUDE(ident=DEFAULT_BEAM_GROUP_IDENT)]
        return ut.get_dimensions(platform_longitudes)

    def get_values(self, nc_dataset: nc.Dataset):
        """read variable values"""
        geod = Geod(ellps="WGS84")
        platform_longitudes = nc_dataset[sg.BeamGroup1Grp.PLATFORM_LONGITUDE(ident=DEFAULT_BEAM_GROUP_IDENT)]
        platform_latitudes = nc_dataset[sg.BeamGroup1Grp.PLATFORM_LATITUDE(ident=DEFAULT_BEAM_GROUP_IDENT)]
        interpings_distance = geod.line_lengths(platform_longitudes[:], platform_latitudes[:])
        # array size is nb_ping-1, we duplicate the last value to have the same dimension
        interpings_distance = np.append(interpings_distance, interpings_distance[-1])
        return interpings_distance

    def get_attributes(self, nc_dataset: nc.Dataset):
        return {"long_name": "Computed interping distance", "units": "m"}

    def get_fill_value(self, nc_dataset: nc.Dataset):
        return np.nan


class DetectionPointingAngleVertical(VariableInterface):
    """Return Detection beam pointing angle referred to vertical (ie with roll and installation offsets added)"""

    def get_dimensions(self, nc_dataset):
        return ut.get_dimensions(
            nc_dataset[sg.BathymetryGrp.DETECTION_BEAM_POINTING_ANGLE(ident=DEFAULT_BEAM_GROUP_IDENT)]
        )

    def get_values(self, nc_dataset):
        nc_dataset.set_auto_mask(True)
        beam_angle_path = sg.BathymetryGrp.DETECTION_BEAM_POINTING_ANGLE(ident=DEFAULT_BEAM_GROUP_IDENT)
        angle_ref_rx = nc_dataset[beam_angle_path][:]
        dim_swath = angle_ref_rx.shape[0]
        dim_beam = angle_ref_rx.shape[1]

        roll = nc_dataset[sg.BeamGroup1Grp.PLATFORM_ROLL(ident=DEFAULT_BEAM_GROUP_IDENT)][:].data
        pitch = nc_dataset[sg.BeamGroup1Grp.PLATFORM_PITCH(ident=DEFAULT_BEAM_GROUP_IDENT)][:].data

        # TODO : the roll to use depends on reception time,
        # should be taken into account : txDelay, and detection time to interpolate (from /Platform) the right roll to apply
        # see in Globe DetectionBeamPointingAnglesLayers.java

        detection_rx_transducer_index = nc_dataset[
            sg.BathymetryGrp.DETECTION_RX_TRANSDUCER_INDEX(ident=DEFAULT_BEAM_GROUP_IDENT)
        ][:]
        # check to ensure that everything invalid is really removed
        mask_invalid = angle_ref_rx.mask | detection_rx_transducer_index.mask
        detection_rx_transducer_index = np.array(detection_rx_transducer_index)  # remove mask for invalid data
        detection_rx_transducer_index[detection_rx_transducer_index < 0] = (
            0  # modify invalid values to point to first rx transducer, does not matter since all values will be flagged as invalid
        )

        rx_installation_angle = np.asarray(nc_dataset[sg.PlatformGrp.TRANSDUCER_ROTATION_X()])
        ry_installation_angle = np.asarray(nc_dataset[sg.PlatformGrp.TRANSDUCER_ROTATION_Y()])
        rz_installation_angle = np.asarray(nc_dataset[sg.PlatformGrp.TRANSDUCER_ROTATION_Z()])

        # estimate detection beam pointing angle ref vertical is approximately angle = angle_ref_transducer + roll + transducer_installation_offset

        # Simplified version
        # rx_installation_sign = np.full_like(rx_installation_angle, fill_value=1)
        # rx_installation_sign[np.fabs(rz_installation_angle) > 90] = -1
        # values = roll.reshape((-1, 1)) + rx_installation_sign[detection_rx_transducer_index] * (angle_ref_rx + rx_installation_angle[detection_rx_transducer_index])

        # Compute full angles rotations from antenna to surface coordinates
        installation_rot = R.from_euler(
            "ZYX", list(zip(rz_installation_angle, ry_installation_angle, rx_installation_angle)), degrees=True
        )
        rotA2V = installation_rot[detection_rx_transducer_index.ravel()]
        angle_ref_rx[mask_invalid] = 0
        repeat_idx = np.arange(dim_swath).repeat(dim_beam)
        rotV2S = R.from_euler("YX", list(zip(pitch, roll)), degrees=True)[repeat_idx]
        rotACS = R.from_euler("X", angle_ref_rx.ravel(), degrees=True)
        rotSCS = rotV2S * rotA2V * rotACS

        # Retrieve Rx angle using plane normal across component
        ref_normal = [0, 0, 1]
        rotated_normal = rotSCS.apply(ref_normal)
        rx_ref_vert = np.degrees(-np.arcsin(rotated_normal[:, 1]))
        values = rx_ref_vert.reshape(dim_swath, dim_beam)

        # if ME70, detection_beam_point_angle is given with reference to the horizontal plane
        beam_stab_path = sg.BathymetryGrp.DETECTION_BEAM_STABILISATION(ident=DEFAULT_BEAM_GROUP_IDENT)
        beam_stab_var = get_variable(i_dataset=nc_dataset, variable_path=beam_stab_path)
        if beam_stab_var is not None:
            beam_stab_mask = beam_stab_var[:] == 1
            values[beam_stab_mask] = angle_ref_rx[beam_stab_mask]

        values[mask_invalid] = np.nan  # ensure invalidity is well set
        return values

    def get_attributes(self, nc_dataset):
        return {"long_name": "Beam pointing angles ref vertical", "units": "arc_degree"}

    def get_fill_value(self, nc_dataset):
        return np.nan


class DetectionPointingAnglePlatform(VariableInterface):
    """Return Detection beam pointing angle referred to platform (ie with installation offsets added)"""

    def get_dimensions(self, nc_dataset):
        return ut.get_dimensions(
            nc_dataset[sg.BathymetryGrp.DETECTION_BEAM_POINTING_ANGLE(ident=DEFAULT_BEAM_GROUP_IDENT)]
        )

    def get_values(self, nc_dataset):
        nc_dataset.set_auto_mask(True)
        beam_angle_path = sg.BathymetryGrp.DETECTION_BEAM_POINTING_ANGLE(ident=DEFAULT_BEAM_GROUP_IDENT)
        angle_ref_rx = nc_dataset[beam_angle_path][:]
        dim_swath = angle_ref_rx.shape[0]
        dim_beam = angle_ref_rx.shape[1]

        detection_rx_transducer_index = nc_dataset[
            sg.BathymetryGrp.DETECTION_RX_TRANSDUCER_INDEX(ident=DEFAULT_BEAM_GROUP_IDENT)
        ][:]
        # check to ensure that everything invalid is really removed
        mask_invalid = angle_ref_rx.mask | detection_rx_transducer_index.mask
        detection_rx_transducer_index = np.array(detection_rx_transducer_index)  # remove mask for invalid data
        detection_rx_transducer_index[detection_rx_transducer_index < 0] = (
            0  # modify invalid values to point to first rx transducer, does not matter since all values will be flagged as invalid
        )

        rx_installation_angle = np.asarray(nc_dataset[sg.PlatformGrp.TRANSDUCER_ROTATION_X()])
        ry_installation_angle = np.asarray(nc_dataset[sg.PlatformGrp.TRANSDUCER_ROTATION_Y()])
        rz_installation_angle = np.asarray(nc_dataset[sg.PlatformGrp.TRANSDUCER_ROTATION_Z()])

        # Compute full angles rotations from antenna to vessel coordinates
        installation_rot = R.from_euler(
            "ZYX", list(zip(rz_installation_angle, ry_installation_angle, rx_installation_angle)), degrees=True
        )
        rotA2V = installation_rot[detection_rx_transducer_index.ravel()]
        angle_ref_rx[mask_invalid] = 0
        rotACS = R.from_euler("X", angle_ref_rx.ravel(), degrees=True)
        rotVCS = rotA2V * rotACS

        # Retrieve Rx angle using plane normal across component
        ref_normal = [0, 0, 1]
        rotated_normal = rotVCS.apply(ref_normal)
        rx_ref_vcs = np.degrees(-np.arcsin(rotated_normal[:, 1]))
        values = rx_ref_vcs.reshape(dim_swath, dim_beam)

        values[mask_invalid] = np.nan  # ensure invalidity is well set
        return values

    def get_attributes(self, nc_dataset):
        return {"long_name": "Beam pointing angles ref platform", "units": "arc_degree"}

    def get_fill_value(self, nc_dataset):
        return np.nan


class DetectionIncidenceAngle(VariableInterface):
    """
    Return Detection beam incidence angle referred to vertical (ie with roll and installation offsets added)
    Beam incidence is considered here as the beam pointing angle corrected from diffraction using embedded sound speed profiles
    """

    def get_dimensions(self, nc_dataset):
        return ut.get_dimensions(
            nc_dataset[sg.BathymetryGrp.DETECTION_BEAM_POINTING_ANGLE(ident=DEFAULT_BEAM_GROUP_IDENT)]
        )

    def get_values(self, nc_dataset):
        surface_angle = DetectionPointingAngleVertical().get_values(nc_dataset)

        # read SSV
        # celerity profile
        sound_speed_profile = nc_dataset[sg.SoundSpeedProfileGrp.SOUND_SPEED()][:]
        svp_depth_values = nc_dataset[sg.SoundSpeedProfileGrp.SAMPLE_DEPTH()][:]
        profile_time = nc_dataset[sg.SoundSpeedProfileGrp.PROFILE_TIME()][:].data
        ping_time = nc_dataset[sg.BeamGroup1Grp.PING_TIME(ident=DEFAULT_BEAM_GROUP_IDENT)][:].data
        detection_z = nc_dataset[sg.BathymetryGrp.DETECTION_Z(ident=DEFAULT_BEAM_GROUP_IDENT)][:].data

        def interpolate_svp(ping_depth_values):
            celerity_output = np.full(shape=ping_depth_values.shape, fill_value=np.nan)

            # build nD celerity matrix with sorted unique values of ping_time and profile_time
            # dimension = profile/ping(/detection)
            sorted_profile_time, sorted_profile_indices = np.unique(profile_time[:], return_index=True)
            sorted_ping_time, sorted_ping_indices, reverse_ping_indices = np.unique(
                ping_time[:], return_index=True, return_inverse=True
            )
            dst_shape = list(profile_time.shape + ping_depth_values.shape)
            dst_shape[0] = sorted_profile_indices.shape[0]
            dst_shape[1] = sorted_ping_indices.shape[0]
            celerity_matrix = np.full(shape=dst_shape, fill_value=np.nan)

            # Compute celerities from each profile
            for dst_index, src_index in enumerate(sorted_profile_indices):
                if len(svp_depth_values[src_index]) > 0:
                    f = interp1d(
                        svp_depth_values[src_index][:],
                        sound_speed_profile[src_index][:],
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    celerity_matrix[dst_index] = f(ping_depth_values[sorted_ping_indices])

            # Interpolate celerities between profiles

            # Fill pings earlier than first profile
            profile_mask_bellow = ping_time <= sorted_profile_time[0]
            celerity_output[profile_mask_bellow] = celerity_matrix[0, reverse_ping_indices][profile_mask_bellow]
            # Fill pings greater than last profile
            profile_mask_above = ping_time >= profile_time[-1]
            celerity_output[profile_mask_above] = celerity_matrix[-1, reverse_ping_indices][profile_mask_above]
            # Interpolate other pings
            profile_mask = ~(profile_mask_bellow | profile_mask_above)
            if np.any(profile_mask):
                celerity_output[profile_mask] = interpn(
                    points=(sorted_profile_time, sorted_ping_time),
                    values=celerity_matrix,
                    xi=(ping_time[profile_mask], ping_time[profile_mask]),
                    bounds_error=False,
                )
            return celerity_output

        bottom_detection_celerity = interpolate_svp(detection_z)

        # surface_celerity = nc_dataset[sg.BeamGroup1Grp.SOUND_SPEED_AT_TRANSDUCER(ident=DEFAULT_BEAM_GROUP_IDENT)][:].data
        # Note that surface celerity should be replaced by the celerity at transducer depth in case of a submarine engine
        transducer_depth = nc_dataset[sg.BeamGroup1Grp.TX_TRANSDUCER_DEPTH(ident=DEFAULT_BEAM_GROUP_IDENT)][:].data
        surface_celerity = interpolate_svp(transducer_depth)

        # compute direct beam incidence angle where incidence angle is angle relative to vertical

        # compute the snell descartes constant
        m = bottom_detection_celerity / surface_celerity[:, None]
        # default celerity to one (straight line)
        m[~np.isfinite(m)] = 1
        sin_refraction_angle = np.sin(np.radians(surface_angle)) * m
        bottom_incidence_angle = np.rad2deg(np.arcsin(sin_refraction_angle))
        return bottom_incidence_angle

    def get_attributes(self, nc_dataset):
        return {"long_name": "Beam incidence angles ref vertical", "units": "arc_degree"}

    def get_fill_value(self, nc_dataset):
        return np.nan


class WCPresence(VariableInterface):
    """Indicate if a ping contains watercolumn data or not"""

    missing_value = -1

    def get_dimensions(self, nc_dataset):
        platform_longitudes = nc_dataset[sg.BeamGroup1Grp.PLATFORM_LONGITUDE(ident=DEFAULT_BEAM_GROUP_IDENT)]
        return ut.get_dimensions(platform_longitudes)

    def get_values(self, nc_dataset):
        values = None
        sample_count_name = sg.BeamGroup1Grp.SAMPLE_COUNT(ident=DEFAULT_BEAM_GROUP_IDENT)
        try:
            values = np.asarray(nc_dataset[sample_count_name])
            if np.size(values) > 0:
                # if we have more than one sample, we consider to have water column
                values = np.sum(values.squeeze(), axis=1)
                values[np.isnan(values)] = 0
                values[values != 0] = 1
        except IndexError:
            # variable is missing from dataset, (sample_count_name in nc_dataset does not work to check for variable presence, we need to use exception)
            pass

        if values is None:
            # Variable is not found : we do not know
            values = np.full_like(
                nc_dataset[sg.BeamGroup1Grp.PLATFORM_LONGITUDE_VNAME],
                fill_value=self.missing_value,
            )
        elif np.size(values) == 0:
            # In this case variable is found but with an empty array (case were no WC beam dimension is defined)
            # Fill like a No WC variable
            values = np.full_like(
                nc_dataset[sg.BeamGroup1Grp.PLATFORM_LONGITUDE(ident=DEFAULT_BEAM_GROUP_IDENT)][:], fill_value=0
            )
        return values

    def get_attributes(self, nc_dataset):
        return {"long_name": "WC Presence", "flag_meaning": "UNK: -1, No WC: 0, WC: 1"}

    def get_fill_value(self, nc_dataset):
        return -1


class DetectionBackscatterSnippetMeanValues(VariableInterface):
    """Return snippet mean values by beam"""

    def get_dimensions(self, nc_dataset):
        return ut.get_dimensions(nc_dataset[sg.BathymetryGrp.DETECTION_BACKSCATTER_R(ident=DEFAULT_BEAM_GROUP_IDENT)])

    def get_values(self, nc_dataset):
        nc_dataset.set_auto_mask(True)
        values = None
        seabed_image_path = sg.BathymetryGrp.SEABED_IMAGE_SAMPLES_R(ident=DEFAULT_BEAM_GROUP_IDENT)
        seabed_image_var = get_variable(i_dataset=nc_dataset, variable_path=seabed_image_path)
        if seabed_image_var is not None:
            seabed_image_samples_db = nc_dataset[seabed_image_path][:]
            values = signal.db_to_db_mean_amplitude(seabed_image_samples_db, axis=2)

        return values

    def get_attributes(self, nc_dataset):
        return {"long_name": "Backscatter snippets mean value", "units": "dB"}

    def get_fill_value(self, nc_dataset):
        return np.nan


class VariablesContainer:
    """A container of variables"""

    def __init__(self):
        """Build a key value dictionary referencing all variables from a given beam group of a xsf"""
        self.variables: Dict[str, VariableInterface] = {}

    def get_fill_values(self, key: str, nc_dataset: nc.Dataset) -> np.ndarray:
        return self.variables[key].get_fill_value(nc_dataset=nc_dataset)

    def get_values(self, key: str, nc_dataset: nc.Dataset) -> np.ndarray:
        var = self.variables[key]
        return var.get_values(nc_dataset=nc_dataset)

    def get_attributes(self, key: str, nc_dataset: nc.Dataset):
        return self.variables[key].get_attributes(nc_dataset=nc_dataset)

    def get_dimensions(self, key: str, nc_dataset: nc.Dataset):
        """retrieve netcdf dimension associated with the given key
        @ return a key value object
        """
        return self.variables[key].get_dimensions(nc_dataset=nc_dataset)

    def __contains__(self, item):
        return self.variables.__contains__(item)


class VariablesDictionary(Protocol):
    class PositionVariables(Protocol):
        def __init__(self, root_dataset: nc.Dataset):
            pass

    class RunTimeVariables(Protocol):
        pass

    class PingTimeVariables(Protocol):
        def __init__(self, beam_group: str):
            pass

    class PingDetectionVariables(Protocol):
        def __init__(self, beam_group: str):
            pass

    class ComputedPingVariables(Protocol):
        pass

    class ComputedPingDetectionVariables(Protocol):
        pass
