#! /usr/bin/env python3
# coding: utf-8

import os

import numpy as np
from pygws.service.progress_monitor import DefaultMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.pyat_logger as log


class ModifyCdiProcess:
    """Modify cdi process class. Rename cdis."""

    def __init__(
        self,
        i_paths: list,
        cdis: list,
        monitor=DefaultMonitor,
    ):
        """Rename cdis.

        Arguments:
            i_paths {list} -- List of dtm file input paths.

        Keyword Arguments:
            params {dict} -- List of cdi to change:[{old: "", new: ""},
                                                    {old: "", new: ""},
                                                    ...].
        """
        self.i_paths = i_paths
        self.monitor = monitor

        self.logger = log.logging.getLogger(self.__class__.__name__)

        self.cdis = []
        for cdi in cdis:
            if len(cdi) == 2 and "old" in cdi and "new" in cdi:
                self.cdis.append(cdi)
            else:
                raise ValueError(f"Invalid value for argument cdis : '{cdi}'")

        if not self.cdis:
            raise ValueError(
                "Useless process without parameters.\nStop the program.\nPlease enter the parameter "
                "cdis with the option -c CDIS [CDIS ...] or --cdis CDIS [CDIS ...].\n"
                "Cdis must be like: '$cdi_to_change $new_name'."
            )

    def __change_cdis(self, i_driver: dtm_driver.DtmDriver) -> None:
        """Method for change the cdi. print if the cdi is changed or not.

        Arguments:
            ind {int} -- [description]
        """
        path = os.path.basename(i_driver.dtm_file.file_path)
        if DtmConstants.CDI in i_driver:
            for cdi in self.cdis:
                old = cdi["old"]
                new = cdi["new"]
                ind_cdi_ref = np.where(i_driver[DtmConstants.CDI][:] == old)[0]

                if ind_cdi_ref.size == 1:
                    self.logger.info(f"In the file {path}, change the cdi {old} by {new}.")
                    i_driver[DtmConstants.CDI][int(ind_cdi_ref)] = new
                elif ind_cdi_ref.size > 1:
                    self.logger.info(f"In the file {path}, change the cdi {old} by {new}.")
                    for index in ind_cdi_ref:
                        i_driver[DtmConstants.CDI][int(index)] = new

                else:
                    self.logger.info(f"In the file {path}, the cdi {old} doesn't exist.")
        else:
            self.logger.error(f"The file {path} has no cdi.")

        self.monitor.worked(1)

    def __process_data(self, i_driver: dtm_driver.DtmDriver, monitor) -> None:
        """Perform the CDI modification"""
        sub_monitor = monitor.split(2)
        self.__change_cdis(i_driver)
        sub_monitor.worked(1)
        # clean up all cdi
        cdi_util.clean_cdi(i_driver.dataset)
        sub_monitor.done()

    def __call__(self) -> None:
        process_util.process_each_input_file_in_write_mode(
            self.i_paths,
            self.__class__.__name__,
            self.logger,
            self.monitor,
            self.__process_data,
        )
