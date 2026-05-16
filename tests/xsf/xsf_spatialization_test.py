import unittest
import pytest

import sonarnative

from tests.file_test_installer import get_test_path


class TestBuildConfig(unittest.TestCase):
    def test_version(self):
        config = sonarnative.BuildConfiguration
        print(
            f"Start of soundernative test version "
            f"{config.VERSION_MAJOR}.{config.VERSION_MINOR}.{config.VERSION_PATCH}"
        )


class TestSpatialization(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.data_dir = get_test_path()
        self.test_file = f"{self.data_dir}/wc/0078_20130204_115147_Thalia.xsf.nc"

    def test_read_swath1(self):

        # Don't use multithread for this test, the echos order is not guaranteed
        spatializer = sonarnative.open_spatializer(self.test_file, -1, False)

        sonarnative.set_spatialization_method(spatializer, sonarnative.SpatializationMethod.DetectionInterpolation)
        assert spatializer.get_swath_count() == 1800
        assert spatializer.get_beam_count() == 512
        echoes_count = sonarnative.estimate_beam_echo_count(spatializer, 0, 1)
        assert echoes_count == 168410
        mem_echos = sonarnative.MemEchos(echoes_count)
        assert mem_echos.size == 0
        sonarnative.spatialize_in_memory(spatializer, 0, 1, mem_echos)
        sonarnative.close_spatializer(spatializer)
        assert mem_echos.size == 168410

        # Check some values like test_xsf_spatializer.cpp...
        assert mem_echos.longitude[0] == pytest.approx(-4.374026466050585)
        assert mem_echos.latitude[0] == pytest.approx(48.30366698940525)
        assert mem_echos.across[0] == pytest.approx(-0.372)
        assert mem_echos.beam_opening_along[0] == pytest.approx(1.0000495)
        assert mem_echos.elevation[0] == pytest.approx(-2.5820813)
        assert mem_echos.height[0] == pytest.approx(0.09675957)
        assert mem_echos.echo[0] == pytest.approx(-64.0)

        assert mem_echos.longitude[10000] == pytest.approx(-4.374028121504939)
        assert mem_echos.latitude[10000] == pytest.approx(48.303624789169874)
        assert mem_echos.across[10000] == pytest.approx(-5.081, abs=1e-2)
        assert mem_echos.beam_opening_along[10000] == pytest.approx(1.0000495)
        assert mem_echos.elevation[10000] == pytest.approx(-9.208, abs=1e-2)
        assert mem_echos.height[10000] == pytest.approx(0.09678965)
        assert mem_echos.echo[10000] == pytest.approx(-64.0)

    def test_sampling_filtering(self):
        spatializer = sonarnative.open_spatializer(self.test_file, -1, True)
        mem_echos = sonarnative.MemEchos(168410)
        sonarnative.apply_sampling_filter(spatializer, sonarnative.SamplingParameter(10))
        sonarnative.spatialize_in_memory(spatializer, 0, 1, mem_echos)
        sonarnative.close_spatializer(spatializer)
        assert mem_echos.size == 17020

    def test_threshold_filtering(self):
        spatializer = sonarnative.open_spatializer(self.test_file, -1, False)
        mem_echos = sonarnative.MemEchos(168410)
        sonarnative.apply_threshold_filter(spatializer, sonarnative.ThresholdParameter(True, 8.5, 8.5))
        sonarnative.spatialize_in_memory(spatializer, 0, 1, mem_echos)
        sonarnative.close_spatializer(spatializer)
        assert mem_echos.size == 72

    def test_range_percent_filtering(self):
        spatializer = sonarnative.open_spatializer(self.test_file, -1, False)
        mem_echos = sonarnative.MemEchos(4881)
        sonarnative.apply_bottom_filter(
            spatializer, sonarnative.BottomFilterParameter.new_range_percent(1.5, 25.0, True)
        )
        sonarnative.spatialize_in_memory(spatializer, 1200, 1, mem_echos)
        sonarnative.close_spatializer(spatializer)
        assert mem_echos.size == 4096

    def test_sample_filtering(self):
        spatializer = sonarnative.open_spatializer(self.test_file, -1, False)
        mem_echos = sonarnative.MemEchos(7027)
        sonarnative.apply_bottom_filter(spatializer, sonarnative.BottomFilterParameter.new_sample(10.0, 150, True))
        sonarnative.spatialize_in_memory(spatializer, 666, 1, mem_echos)
        sonarnative.close_spatializer(spatializer)
        assert mem_echos.size == 7027

    def test_specular_filtering(self):
        spatializer = sonarnative.open_spatializer(self.test_file, -1, True)
        mem_echos = sonarnative.MemEchos(168410)
        sonarnative.apply_specular_filter(spatializer, sonarnative.SpecularFilterParameter(True, False, 250))
        sonarnative.spatialize_in_memory(spatializer, 0, 1, mem_echos)
        sonarnative.close_spatializer(spatializer)
        assert mem_echos.size == 5632

    def test_specular_below_sphere_filtering(self):
        spatializer = sonarnative.open_spatializer(self.test_file, -1, False)
        mem_echos = sonarnative.MemEchos(168410)
        sonarnative.apply_specular_filter(spatializer, sonarnative.SpecularFilterParameter(True, True, 2))
        sonarnative.spatialize_in_memory(spatializer, 0, 1, mem_echos)
        sonarnative.close_spatializer(spatializer)
        assert mem_echos.size == 132608

    def test_beam_index_filtering(self):
        spatializer = sonarnative.open_spatializer(self.test_file, -1, False)
        mem_echos = sonarnative.MemEchos(168410)
        sonarnative.apply_beam_index_filter(spatializer, sonarnative.BeamIndexParameter(True, 50, 75))
        sonarnative.spatialize_in_memory(spatializer, 0, 1, mem_echos)
        sonarnative.close_spatializer(spatializer)
        assert mem_echos.size == 8628

    def test_depth_filtering(self):
        spatializer = sonarnative.open_spatializer(self.test_file, -1, False)
        mem_echos = sonarnative.MemEchos(168410)
        sonarnative.apply_depth_filter(spatializer, sonarnative.DepthParameter(True, 20.0, 25.0))
        sonarnative.spatialize_in_memory(spatializer, 0, 1, mem_echos)
        sonarnative.close_spatializer(spatializer)
        assert mem_echos.size == 28601

    def test_across_distance_filter_filtering(self):
        spatializer = sonarnative.open_spatializer(self.test_file, -1, False)
        mem_echos = sonarnative.MemEchos(168410)
        sonarnative.apply_across_distance_filter(spatializer, sonarnative.AcrossDistanceParameter(True, -5.0, 5.0))
        sonarnative.spatialize_in_memory(spatializer, 0, 1, mem_echos)
        sonarnative.close_spatializer(spatializer)
        assert mem_echos.size == 96147

    def test_sidelobe_filter_filtering(self):
        spatializer = sonarnative.open_spatializer(self.test_file, -1, True)
        mem_echos = sonarnative.MemEchos(168410)
        sonarnative.apply_side_lobe_filter(spatializer, sonarnative.SideLobeParameter(True, 12.0))
        sonarnative.spatialize_in_memory(spatializer, 0, 1, mem_echos)
        sonarnative.close_spatializer(spatializer)
        assert mem_echos.size == 5072
