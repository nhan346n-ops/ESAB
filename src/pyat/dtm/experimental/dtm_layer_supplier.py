import logging

import pygws.service.execution_context as exec_ctx

import pyat.dtm.dtm_standard_constants as dtm_constants
from pyat.dtm import dtm_driver
from pyat.dtm.export.dtm_to_cog import DtmToCog

logger = logging.getLogger(__name__)


def get_dtm_layer(i_path: str, cog_path: str):
    """
    Extracts DTM information (geobox, min, max values...) and creates a COG file.

    This code is a prototype used by "Globe Web Services Extended" to produce DTM display layers.
    """
    logger.info("Opening DTM file : %s", i_path)
    with dtm_driver.open_dtm(i_path) as i_dtm_driver:
        # Extract DTM information
        dtm_file = i_dtm_driver.dtm_file
        elevation_layer = i_dtm_driver.get_layers()[dtm_constants.ELEVATION_NAME]

        file_info = {
            'path': i_path,
            'row': dtm_file.row_count,
            'col': dtm_file.col_count,
            'geobox': {
                'north': dtm_file.north,
                'south': dtm_file.south,
                'east': dtm_file.east,
                'west': dtm_file.west,
            },
            'spatial_reference': dtm_file.spatial_reference.ExportToProj4(),
            'spatial_resolution_x': dtm_file.spatial_resolution_x,
            'spatial_resolution_y': dtm_file.spatial_resolution_y,
            'max_elevation' : float(elevation_layer[:].max()),
            'min_elevation' : float(elevation_layer[:].min()),
        }
        logger.info(f'DTM file : {file_info}')

        # Create COG
        DtmToCog(i_paths=[i_path], o_paths=[cog_path])()
        file_info['cog'] = cog_path

        # Using rsocket (if present) to return the result
        rsocket_msg_emitter = exec_ctx.get_rsocket_msg_emitter()
        if rsocket_msg_emitter is not None:
            rsocket_msg_emitter.emit_dict(file_info)
            return None
        else:
            return file_info
