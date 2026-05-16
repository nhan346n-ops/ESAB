#! /usr/bin/env python3
# coding: utf-8

import tempfile

import netCDF4 as nc
import numpy as np
import pyproj
import sonar_netcdf.sonar_groups as sg

from pyat.sonarscope.model.constants import (
    SONAR_GROUP_NAME,
)
from pyat.sonarscope.model.sounder_lib import SounderManufacturer
from pyat.xsf.xsf_driver import BEAM_GROUP_NAME


class XsfGenerator:
    """Class generator of XSF file for the processes tests."""

    def __init__(self, folder=tempfile.tempdir):
        self.folder = folder

    def initialize_file(
        self,
        latitude_min_deg: float,
        latitude_max_deg: float,
        longitude_min_deg: float,
        longitude_max_deg: float,
        ping_count: int,
        beam_count: int,
        min_depth_m: float,
        max_depth_m: float,
    ) -> str:
        """
        Generates a XSF file with
            As many navigation positions as "ping_count"
            "beam_count" beams on both sides of the navigation positions
            Configured Layers in such a way that depths are the same in FCS and SCS coordinates system
        """
        result = tempfile.mktemp(suffix=".xsf.nc", dir=self.folder)
        with nc.Dataset(result, "w", format="NETCDF4") as dataset:
            root_structure = sg.RootGrp()
            root = root_structure.create_group(dataset)

            # create /annotation
            ano_structure = sg.AnnotationGrp()
            ano = ano_structure.create_group(root)
            ano_structure.create_dimension(ano, {sg.AnnotationGrp.TIME_DIM_NAME: 2})
            ano_structure.create_time(ano, long_name="ANO_TIME")
            ano_structure.create_annotation_text(ano)

            # create /provenance
            provenance_structure = sg.ProvenanceGrp()
            provenance = provenance_structure.create_group(root)
            provenance_structure.create_dimension(provenance, {sg.ProvenanceGrp.FILENAMES_DIM_NAME: 0})

            # create /Sonar
            sonar_structure = sg.SonarGrp()
            sonar = sonar_structure.create_group(root)

            # create group /Sonar/Beam_group1
            beam_structure = sg.BeamGroup1Grp()
            beam_group1 = beam_structure.create_group(sonar, ident=BEAM_GROUP_NAME)
            beam_structure.create_dimension(beam_group1, {sg.BeamGroup1Grp.PING_TIME_DIM_NAME: ping_count})
            beam_structure.create_dimension(beam_group1, {sg.BeamGroup1Grp.BEAM_DIM_NAME: beam_count})
            tx_beam_count = 2
            beam_structure.create_dimension(beam_group1, {sg.BeamGroup1Grp.TX_BEAM_DIM_NAME: tx_beam_count})
            beam_group1.preferred_position = 0
            beam_structure.create_beam_type(beam_group1)

            platform_latitude = beam_structure.create_platform_latitude(beam_group1)
            platform_latitude[:] = np.linspace(latitude_min_deg, latitude_max_deg, num=ping_count).reshape(
                1, ping_count
            )

            platform_longitude = beam_structure.create_platform_longitude(beam_group1)
            platform_longitude[:] = np.linspace(longitude_min_deg, longitude_max_deg, num=ping_count).reshape(
                1, ping_count
            )

            heading = self.compute_heading(
                latitude_min_deg,
                latitude_max_deg,
                longitude_min_deg,
                longitude_max_deg,
            )
            platform_heading = beam_structure.create_platform_heading(beam_group1)
            platform_heading[:] = np.full((1, ping_count), heading)
            pitch = 0
            platform_pitch = beam_structure.create_platform_pitch(beam_group1)
            platform_pitch[:] = np.full((1, ping_count), pitch)
            roll = 0
            platform_roll = beam_structure.create_platform_roll(beam_group1)
            platform_roll[:] = np.full((1, ping_count), roll)

            ping_time = beam_structure.create_ping_time(beam_group1)
            ping_time[:] = np.linspace(1359898806964000000, 1359898948064000000, num=ping_count, dtype="i8")
            platform_vertical_offset = beam_structure.create_platform_vertical_offset(beam_group1)
            platform_vertical_offset[:] = np.zeros((1, ping_count))
            waterline_to_chart_datum = beam_structure.create_waterline_to_chart_datum(beam_group1)
            waterline_to_chart_datum[:] = np.zeros((1, ping_count))
            tx_transducer_depth = beam_structure.create_tx_transducer_depth(beam_group1)
            tx_transducer_depth[:] = np.random.default_rng().random((1, ping_count)) * 2.0 + 1.0
            sound_speed_at_transducer = beam_structure.create_sound_speed_at_transducer(beam_group1)
            sound_speed_at_transducer[:] = 1498.0
            transmit_duration_nominal = beam_structure.create_transmit_duration_nominal(beam_group1)
            transmit_duration_nominal[:] = 0.00015
            transmit_type = beam_structure.create_transmit_type(beam_group1)
            transmit_type[:] = [0, 1]

            tx_beam_rotation_theta = beam_structure.create_tx_beam_rotation_theta(beam_group1)
            tx_beam_rotation_theta[:] = [-1, 1]

            # create group /Sonar/Beam_group1/Bathymetry
            bathymetry_structure = sg.BathymetryGrp()
            bathymetry = bathymetry_structure.create_group(beam_group1, ident="Bathymetry")
            bathymetry_structure.create_dimension(bathymetry, {sg.BathymetryGrp.DETECTION_DIM_NAME: beam_count})
            detection_x = bathymetry_structure.create_detection_x(bathymetry)
            detection_x[:] = np.linspace(2.0, 5.0, num=beam_count)
            detection_y = bathymetry_structure.create_detection_y(bathymetry)
            detection_y[:] = np.linspace(-150, 150, num=beam_count)
            detection_z = bathymetry_structure.create_detection_z(bathymetry)
            detection_z[:] = np.linspace(min_depth_m, max_depth_m, num=beam_count)

            lons, lats = self.compute_beam_positions(
                beam_count,
                platform_longitude[:],
                platform_latitude[:],
                detection_y[:],
                heading,
            )
            detection_longitude = bathymetry_structure.create_detection_longitude(bathymetry)
            detection_longitude[:] = lons
            detection_latitude = bathymetry_structure.create_detection_latitude(bathymetry)
            detection_latitude[:] = lats
            detection_rx_transducer_index = bathymetry_structure.create_detection_rx_transducer_index(bathymetry)
            detection_rx_transducer_index[:] = np.zeros((ping_count, beam_count))
            detection_tx_transducer_index = bathymetry_structure.create_detection_tx_transducer_index(bathymetry)
            detection_tx_transducer_index[:] = np.zeros((ping_count, beam_count))
            detection_tx_beam = bathymetry_structure.create_detection_tx_beam(bathymetry)
            detection_tx_beam[:] = np.zeros((ping_count, beam_count))
            multiping_sequence = bathymetry_structure.create_multiping_sequence(bathymetry)
            multiping_sequence[:] = 0

            status = bathymetry_structure.create_status(bathymetry)
            status[:] = np.zeros((ping_count, beam_count))
            status_detail = bathymetry_structure.create_status_detail(bathymetry)
            status_detail[:] = np.zeros((ping_count, beam_count))
            detection_backscatter_r = bathymetry_structure.create_detection_backscatter_r(bathymetry)
            detection_backscatter_r[:] = np.linspace(-26.0, -31.0, num=beam_count)
            detection_backscatter_calibration = bathymetry_structure.create_detection_backscatter_calibration(
                bathymetry
            )
            detection_backscatter_calibration[:] = np.linspace(1.0, 2.0, num=beam_count)
            sample_count = 5
            bathymetry_structure.create_dimension(bathymetry, {sg.BathymetryGrp.SEABED_SAMPLE_DIM_NAME: sample_count})
            seabed_image_samples_r = bathymetry_structure.create_seabed_image_samples_r(bathymetry)
            seabed_image_samples_r[:] = np.repeat(
                np.linspace(-26.0, -31.0, num=beam_count), sample_count, axis=0
            ).reshape(beam_count, sample_count)
            detection_beam_pointing_angle = bathymetry_structure.create_detection_beam_pointing_angle(bathymetry)
            detection_beam_pointing_angle[:] = np.linspace(-40.0, 40.0, num=beam_count)
            detection_beam_stabilisation = bathymetry_structure.create_detection_beam_stabilisation(bathymetry)
            detection_beam_stabilisation[:] = 0
            detection_two_way_travel_time = bathymetry_structure.create_detection_two_way_travel_time(bathymetry)
            detection_two_way_travel_time[:] = np.divide(detection_z[:], 1500)

            platform_structure = sg.PlatformGrp()
            platform = platform_structure.create_group(root)
            platform_structure.create_dimension(platform, {sg.PlatformGrp.TRANSDUCER_DIM_NAME: 5})
            transducer_function = platform_structure.create_transducer_function(platform)
            transducer_function[:] = [0, 0, 1, 1, 1]
            transducer_offset_x = platform_structure.create_transducer_offset_x(platform)
            transducer_offset_x[:] = [4.239, 4.24, 4.223, 4.223, 4.223]
            transducer_offset_y = platform_structure.create_transducer_offset_y(platform)
            transducer_offset_y[:] = [-0.372, 0.389, -0.0454, 0.02315, 0.0654]
            transducer_offset_z = platform_structure.create_transducer_offset_z(platform)
            transducer_offset_z[:] = [1.646, 1.644, 1.723, 1.729, 1.723]
            transducer_rotation_x = platform_structure.create_transducer_rotation_x(platform)
            transducer_rotation_x[:] = [39.79, -39.86, -0.08, -0.08, -0.08]
            transducer_rotation_y = platform_structure.create_transducer_rotation_y(platform)
            transducer_rotation_y[:] = [-0.02, -0.26, -0.04, -0.04, -0.04]
            transducer_rotation_z = platform_structure.create_transducer_rotation_z(platform)
            transducer_rotation_z[:] = [0.22, -0.91000366, -1.6099854, -1.6099854, -1.6099854]

            # create group /Platform/Dynamic_draught
            dynamic_draught_structure = sg.DynamicDraughtGrp()
            dynamic_draught = dynamic_draught_structure.create_group(root)
            dynamic_draught_structure.create_dimension(
                dynamic_draught, {sg.DynamicDraughtGrp.TIME_DIM_NAME: ping_count}
            )
            dynamic_draught_time = dynamic_draught_structure.create_time(dynamic_draught)
            dynamic_draught_time[:] = ping_time[:]
            dynamic_draught_structure.create_delta_draught(dynamic_draught)

            # create group /Environment/Tide/
            tide_structure = sg.TideGrp()
            tide = tide_structure.create_group(root)
            tide_structure.create_dimension(tide, {sg.TideGrp.TIME_DIM_NAME: ping_count})
            tide_time = tide_structure.create_time(tide)
            tide_time[:] = ping_time[:]
            tide_structure.create_tide_indicative(tide)

            # create group /Environment/SoundSpeedProfile
            profile_count = 2
            sound_speed_structure = sg.SoundSpeedProfileGrp()
            sound_speed_grp = sound_speed_structure.create_group(root)
            sound_speed_structure.create_dimension(
                sound_speed_grp, {sg.SoundSpeedProfileGrp.PROFILE_TIME_DIM_NAME: profile_count}
            )
            profile_time = sound_speed_structure.create_profile_time(sound_speed_grp)
            profile_time[:] = [ping_time[0], ping_time[0] + 100000000000]
            sound_speed_profile = sound_speed_structure.create_sound_speed(sound_speed_grp)
            sound_speed_profile[0] = np.array([1500, 1500], dtype=np.float32)
            sound_speed_profile[1] = np.array([1500, 1450, 1800], dtype=np.float32)
            svp_depth_values = sound_speed_structure.create_sample_depth(sound_speed_grp)
            svp_depth_values[0] = np.array([0, 15000], dtype=np.float32)
            svp_depth_values[1] = np.array([0, 100, 12000], dtype=np.float32)

            # create /Platform/001/
            platform_structure.create_dimension(platform, {sg.PlatformGrp.POSITION_DIM_NAME: 1})
            position_ids = sg.PlatformGrp().create_position_ids(platform)
            position_ids[0] = "001"
            posgroup_structure = sg.PositionGrp()
            posgroup = posgroup_structure.create_group(root)
            possubgroup_structure = sg.PositionSubGroup()
            possubgroup_structure.create_group(posgroup, "001")

        return result

    def compute_heading(
        self, latitude_min_deg: float, latitude_max_deg: float, longitude_min_deg: float, longitude_max_deg: float
    ):
        geodesic = pyproj.Geod(ellps="WGS84")
        fwd_azimuth, back_azimuth, distance = geodesic.inv(
            longitude_min_deg, latitude_min_deg, longitude_max_deg, latitude_max_deg
        )
        return fwd_azimuth

    def compute_beam_positions(
        self,
        beam_count: int,
        platform_longitude: np.ndarray,
        platform_latitude: np.ndarray,
        detection_y: np.ndarray,
        heading: float,
    ):
        ping_count = platform_longitude.shape[0]
        lons = np.ndarray((ping_count, beam_count), dtype=float)
        lons[:] = platform_longitude.reshape((ping_count, 1))

        lats = np.ndarray((ping_count, beam_count), dtype=float)
        lats[:] = platform_latitude.reshape((ping_count, 1))

        az = np.full((ping_count, beam_count), heading + 90.0)

        geodesic = pyproj.Geod(ellps="WGS84")
        lons, lats, az = geodesic.fwd(lons, lats, az, detection_y)
        return lons, lats

    def append_kongsberg_all_variables(self, xsf_file: str):
        with nc.Dataset(xsf_file, "r+", format="NETCDF4") as dataset:
            # set constructor
            dataset[SONAR_GROUP_NAME].sonar_manufacturer = SounderManufacturer.KONGSBERG

            ping_time = dataset[sg.BeamGroup1Grp.PING_TIME(BEAM_GROUP_NAME)]
            # create /Platform/Vendor_specific
            platform_structure = sg.PlatformVendorSpecificGrp()
            platform = platform_structure.create_group(dataset)
            dataset[platform_structure.get_group_path()].kongsbergModelNumber = 2040

            # create /Platform/Vendor_specific/runtime
            runtime_structure = sg.RuntimeGrp()
            runtime = runtime_structure.create_group(dataset)
            runtime_structure.create_dimension(runtime, {sg.RuntimeGrp.RUNTIME_COUNT_DIM_NAME: 1})
            runtime_time = runtime_structure.create_time(runtime)
            runtime_time[:] = ping_time[0]
            runtime_ping_mode = runtime_structure.create_ping_mode(runtime)
            runtime_ping_mode[:] = 0
            runtime_frequency_mode = runtime_structure.create_frequency_mode(runtime)
            runtime_frequency_mode[:] = 300000
            runtime_pulse_form = runtime_structure.create_tx_pulse_form(runtime)
            runtime_pulse_form[:] = 0  # CW
            runtime_pulse_length_mode = runtime_structure.create_pulse_length_mode(runtime)
            runtime_pulse_length_mode[:] = 2  # MEDIUM_CW
            runtime_dual_swath_mode = runtime_structure.create_dual_swath_mode(runtime)
            runtime_dual_swath_mode[:] = 0
            runtime_receiver_beamwidth = runtime_structure.create_receiver_beamwidth(runtime)
            runtime_receiver_beamwidth[:] = 1.0
            runtime_tx_beamwidth = runtime_structure.create_tx_beamwidth(runtime)
            runtime_tx_beamwidth[:] = 1.0
            runtime_tx_pulse_length = runtime_structure.create_tx_pulse_length(runtime)
            runtime_tx_pulse_length[:] = 0.000118
            # create /Beam_group1/Vendor_specific
            beamGroup1_vendor_structure = sg.BeamGroup1VendorSpecificGrp()
            beamGroup1_vendor = beamGroup1_vendor_structure.create_group(
                dataset, beamGroup1_vendor_structure.get_group_path(BEAM_GROUP_NAME)
            )
            rxantenna_count = 1
            beamGroup1_vendor_structure.create_dimension(
                beamGroup1_vendor, {sg.BeamGroup1VendorSpecificGrp.RXANTENNA_DIM_NAME: rxantenna_count}
            )
            tx_sector_count = beamGroup1_vendor_structure.create_tx_sector_count(beamGroup1_vendor)
            tx_sector_count[:] = np.full((len(ping_time), rxantenna_count), 3)

            tx_beam_count = 2
            center_frequency = beamGroup1_vendor_structure.create_center_frequency(beamGroup1_vendor)
            center_frequency[:] = np.full((len(ping_time), tx_beam_count), 300000)

            seabed_image_sample_rate = beamGroup1_vendor_structure.create_seabed_image_sample_rate(beamGroup1_vendor)
            seabed_image_sample_rate[:] = 15000

            # create /Beam_group1/Bathymetry/Vendor_specific
            bathymetry_vendor_structure = sg.BathymetryVendorSpecificGrp()
            bathymetry_vendor = bathymetry_vendor_structure.create_group(
                dataset, bathymetry_vendor_structure.get_group_path(BEAM_GROUP_NAME)
            )
            bathymetry_vendor_structure.create_dimension(
                bathymetry_vendor, {sg.BathymetryVendorSpecificGrp.RXANTENNA_DIM_NAME: rxantenna_count}
            )
            sampling_freq = bathymetry_vendor_structure.create_detection_sampling_freq(bathymetry_vendor)
            sampling_freq[:] = np.full((len(ping_time), rxantenna_count), 15000)
            BSN = bathymetry_vendor_structure.create_backscatter_normal_incidence_level(bathymetry_vendor)
            BSN[:] = np.full((len(ping_time), rxantenna_count), -26)
            BS0 = bathymetry_vendor_structure.create_backscatter_oblique_incidence_level(bathymetry_vendor)
            BS0[:] = np.full((len(ping_time), rxantenna_count), -28)

            travel_time = dataset[sg.BathymetryGrp.DETECTION_TWO_WAY_TRAVEL_TIME(BEAM_GROUP_NAME)]
            Rn = bathymetry_vendor_structure.create_range_to_normal_incidence(bathymetry_vendor)
            Rn[:] = np.nanmin(travel_time[:] * sampling_freq[:], axis=1)

            tvg_law_crossover_angle = bathymetry_vendor_structure.create_tvg_law_cross_over_angle(bathymetry_vendor)
            tvg_law_crossover_angle[:] = 10

    def append_kongsberg_kmall_variables(self, xsf_file: str):
        with nc.Dataset(xsf_file, "r+", format="NETCDF4") as dataset:
            # set constructor
            dataset[SONAR_GROUP_NAME].sonar_manufacturer = SounderManufacturer.KONGSBERG

            ping_time = dataset[sg.BeamGroup1Grp.PING_TIME(BEAM_GROUP_NAME)]
            # create /Platform/Vendor_specific/runtime
            runtime_structure = sg.RuntimeGrp()
            runtime = runtime_structure.create_group(dataset)
            runtime_structure.create_dimension(runtime, {sg.RuntimeGrp.RUNTIME_COUNT_DIM_NAME: 1})
            runtime_time = runtime_structure.create_time(runtime)
            runtime_time[:] = ping_time[0]
            runtime_tvg_law_crossover_angle = runtime_structure.create_tvg_law_crossover_angle(runtime)
            runtime_tvg_law_crossover_angle[:] = 10

            # create /Beam_group1/Vendor_specific
            beamGroup1_vendor_structure = sg.BeamGroup1VendorSpecificGrp()
            beamGroup1_vendor = beamGroup1_vendor_structure.create_group(
                dataset, beamGroup1_vendor_structure.get_group_path(BEAM_GROUP_NAME)
            )
            rxantenna_count = 1
            beamGroup1_vendor_structure.create_dimension(
                beamGroup1_vendor, {sg.BeamGroup1VendorSpecificGrp.RXANTENNA_DIM_NAME: rxantenna_count}
            )
            tx_sector_count = beamGroup1_vendor_structure.create_tx_sector_count(beamGroup1_vendor)
            tx_sector_count[:] = np.full((len(ping_time), rxantenna_count), 2)
            frequency_mode = beamGroup1_vendor_structure.create_frequency_mode(beamGroup1_vendor)
            frequency_mode[:] = np.float32(300000)
            pulse_form = beamGroup1_vendor_structure.create_pulse_form(beamGroup1_vendor)
            pulse_form[:] = np.ubyte(0)
            swath_per_ping = beamGroup1_vendor_structure.create_swath_per_ping(beamGroup1_vendor)
            swath_per_ping[:] = np.ubyte(2)
            depth_mode = beamGroup1_vendor_structure.create_depth_mode(beamGroup1_vendor)
            depth_mode[:] = np.byte(4)  # MEDIUM_CW
            seabed_image_sample_rate = beamGroup1_vendor_structure.create_seabed_image_sample_rate(beamGroup1_vendor)
            seabed_image_sample_rate[:] = 15000
            receive_array_size_used = beamGroup1_vendor_structure.create_receive_array_size_used(beamGroup1_vendor)
            receive_array_size_used[:] = 1  # degree (opening)
            transmit_array_size_used = beamGroup1_vendor_structure.create_transmit_array_size_used(beamGroup1_vendor)
            transmit_array_size_used[:] = 1  # degree (opening)
            effective_signal_length = beamGroup1_vendor_structure.create_effectivesignallength_sec(beamGroup1_vendor)
            effective_signal_length[:] = 0.0000537  # seconds

            # create /Beam_group1/Bathymetry/Vendor_specific
            bathymetry_vendor_structure = sg.BathymetryVendorSpecificGrp()
            bathymetry_vendor = bathymetry_vendor_structure.create_group(
                dataset, bathymetry_vendor_structure.get_group_path(BEAM_GROUP_NAME)
            )
            bathymetry_vendor_structure.create_dimension(
                bathymetry_vendor, {sg.BathymetryVendorSpecificGrp.RXANTENNA_DIM_NAME: rxantenna_count}
            )
            BSN = bathymetry_vendor_structure.create_backscatter_normal_incidence_level(bathymetry_vendor)
            BSN[:] = np.full((len(ping_time), rxantenna_count), -26)
            BS0 = bathymetry_vendor_structure.create_backscatter_oblique_incidence_level(bathymetry_vendor)
            BS0[:] = np.full((len(ping_time), rxantenna_count), -28)


if __name__ == "__main__":
    generator = XsfGenerator()

    print(
        generator.initialize_file(
            latitude_min_deg=48.0,
            latitude_max_deg=48.005,
            longitude_min_deg=-4.004,
            longitude_max_deg=-4.0,
            ping_count=2,
            beam_count=2,
            min_depth_m=10.0,
            max_depth_m=20.0,
        )
    )
