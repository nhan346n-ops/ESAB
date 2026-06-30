import numpy as np
import sonar_netcdf.sonar_groups as sg

from pyat.sonarscope.model.constants import DEFAULT_BEAM_GROUP_IDENT


def get_detection_count(nc_dataset):
    """Return detection dimension count"""
    rx_antenna_index = nc_dataset[sg.BathymetryGrp.DETECTION_RX_TRANSDUCER_INDEX(ident=DEFAULT_BEAM_GROUP_IDENT)][:]
    num_of_detection = rx_antenna_index.shape[1]
    return num_of_detection


def get_detection_antenna_coords(nc_dataset):
    """Return (ping,detection) matrix of (ping,antenna indices).
    It allows to convert (ping-antenna variable to ping-detection variable)
    Simple broadcast indices doesn't work with a third dimension
    """
    rx_antenna_index = nc_dataset[sg.BathymetryGrp.DETECTION_RX_TRANSDUCER_INDEX(ident=DEFAULT_BEAM_GROUP_IDENT)][:]
    # keep antenna index in valid range
    rx_antenna_index[rx_antenna_index < 0] = 0

    num_of_pings = rx_antenna_index.shape[0]
    num_of_detection = rx_antenna_index.shape[1]

    # compute indices to transform ping_antenna variable to ping_detection variable
    detection_antenna_coords = (
        np.repeat(np.arange(num_of_pings)[:, None], num_of_detection, axis=1),
        rx_antenna_index[:],
    )
    return detection_antenna_coords


def get_detection_tx_beam_coords(nc_dataset):
    """Return (ping,detection) matrix of (ping,tx_beam indices).
    It allows to convert (ping-tx_beam variable to ping-detection variable)
    Simple broadcast indices doesn't work with a third dimension (like rxAntenna)
    """
    tx_beam_index = nc_dataset[sg.BathymetryGrp.DETECTION_TX_BEAM(ident=DEFAULT_BEAM_GROUP_IDENT)][:]
    # keep antenna index in valid range
    tx_beam_index[tx_beam_index < 0] = 0

    num_of_pings = tx_beam_index.shape[0]
    num_of_detection = tx_beam_index.shape[1]

    # compute indices to transform ping_antenna variable to ping_detection variable
    detection_tx_beam_coords = (
        np.repeat(np.arange(num_of_pings)[:, None], num_of_detection, axis=1),
        tx_beam_index[:],
    )
    return detection_tx_beam_coords
