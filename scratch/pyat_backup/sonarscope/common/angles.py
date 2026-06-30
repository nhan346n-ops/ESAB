import numpy as np


class AngleBins:
    """
    A set of angle values used for histograms and models
    """

    def __init__(self, min_angle, max_angle, angle_resolution=1.0):
        self.angle_range = [min_angle, max_angle]
        self.bin_width = angle_resolution  # bin width in deg
        self.bin_count = int((self.angle_range[1] - self.angle_range[0]) / self.bin_width)
        self.bin_centers = np.arange(
            self.angle_range[0] + self.bin_width / 2.0, self.angle_range[1] + self.bin_width / 2.0, self.bin_width
        )


class IncidenceAngleBins(AngleBins):
    """
    A set of angle values used for histograms and models
    """

    def __init__(self, max_angle=89.5, angle_resolution=1):
        super().__init__(min_angle=-0.5, max_angle=max_angle, angle_resolution=angle_resolution)


class TransmissionAngleBins(AngleBins):
    """
    A set of angle values used for histograms and models
    """

    def __init__(self, max_angle=80.5, angle_resolution=1):
        super().__init__(min_angle=-max_angle, max_angle=max_angle, angle_resolution=angle_resolution)
