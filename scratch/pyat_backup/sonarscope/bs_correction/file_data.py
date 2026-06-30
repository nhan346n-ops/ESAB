import numpy as np

class FileDataStore:
    """A file container which store per file data used by bs_correction"""

    def __init__(
        self,
        file: str,
        bs_value: np.ndarray = np.ndarray(0),
        incidence_angle: np.ndarray = np.ndarray(0),
        mode_indices: np.ndarray = np.ndarray(0),
    ):
        self.file = file
        self.bs_value = bs_value
        self.incidence_angle = incidence_angle
        self.mode_indices = mode_indices
