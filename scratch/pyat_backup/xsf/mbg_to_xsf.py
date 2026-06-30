#! /usr/bin/env python3
# coding: utf-8
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import List

import pygws.client.http.gws_server_configuration as gws_conf
import pygws.client.http.gws_service_launcher as gws_service
import pygws.service.execution_context as exec_ctx
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
from pyat.xsf.xsf_upgrader_from_mbg import XsfUpgrader


class MbgToXsfExporter:
    """
    Callable used by pyat/app to launch an export of sounder files to XSF format.
    This export is performed by the GWS service "Convert sounding file to XSF".

    Then, the exported XSF files are upgraded from MBG files.
    This upgrade is performed by the XsfUpgrader pyat class.
    """

    def __init__(
        self,
        i_paths: List[str],  # Sounder files
        out_dir: str,
        i_mbg: List[str],
        overwrite: bool = False,
        ignoreWC: bool = False,
        xsfKeywords: str = "",
        xsfLicense: str = "",
        xsfRights: str = "",
        xsfSummary: str = "",
        xsfTitle: str = "",
        gws_http_port: int = 8080,  # GWS port automatically set by GWS
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Constructor
        """
        self.logger = log.logging.getLogger(self.__class__.__name__)
        self.logger.info("Preparing upgrade...")
        self.gws_http_port = gws_http_port
        self.logger.info(f"Using GWS server on port {self.gws_http_port}")

        self.monitor = monitor

        # Parsing parameters
        self.i_paths = arg_util.parse_list_of_files("i_paths", i_paths, True)
        self.i_mbg = arg_util.parse_list_of_files("i_mbg", i_mbg, True)
        self.overwrite = overwrite

        self.out_dir = Path(out_dir)
        if not self.out_dir.exists():
            os.makedirs(self.out_dir)
        if not self.out_dir.is_dir():
            raise ValueError(f"{self.out_dir} : is not a directory. Process aborted")

        self.ignoreWC = ignoreWC
        self.xsfKeywords = xsfKeywords
        self.xsfLicense = xsfLicense
        self.xsfRights = xsfRights
        self.xsfSummary = xsfSummary
        self.xsfTitle = xsfTitle

    def __call__(self) -> None:
        """Run method."""

        self.monitor.set_work_remaining(len(self.i_paths) + 1)
        begin = datetime.now()

        # Set up the GWS configuration
        gws_conf.configure_gws(gws_http_port=self.gws_http_port)

        # Export raw sounder files to xsf
        self.logger.info("Start converting sounder files to XSF ...")
        xsf_files_to_cut = [self.out_dir / f"xsf_{index}.xsf.nc" for index in range(len(self.i_paths))]
        for sounder_file, xsf_file in zip(self.i_paths, xsf_files_to_cut):
            if not asyncio.run(self._convert_sounder_file(sounder_file, outfile=xsf_file)):
                self.logger.error(f"Conversion failed for {sounder_file}. Process abort")
                return

        # Upgrade the exported XSF files with the MBG data
        self.logger.info("Start upgrading XSF with MBG...")
        self._upgrade_xsf(xsf_files_to_cut)

        # remove temporary xsf files from the sounder file conversion step
        for xsf_file in xsf_files_to_cut:
            if xsf_file.exists():
                os.remove(xsf_file)

        self.monitor.done()
        process_util.log_result(self.logger, begin, [])

        # Using rsocket (if present) to send the result
        rsocket_msg_emitter = exec_ctx.get_rsocket_msg_emitter()
        if rsocket_msg_emitter is not None:
            rsocket_msg_emitter.emit_files([])

    def _upgrade_xsf(self, xsf_files_to_cut: List[Path]) -> None:
        """Invoke XSF upgrader with MBG data"""
        updater = XsfUpgrader(
            i_paths=[str(xsf_file) for xsf_file in xsf_files_to_cut],
            i_mbg=self.i_mbg,
            out_dir=str(self.out_dir),
            overwrite=self.overwrite,
        )
        updater()

    async def _convert_sounder_file(self, sounder_file, outfile: os.PathLike) -> bool:
        """Invoke the GWS service to convert the sounder file to XSF"""
        result = await gws_service.run_service_and_return_output_files(
            "Convert Sounder files to XSF (batch)",
            {
                "fmt": "xsf",
                "in": str(sounder_file),
                "out": str(outfile),
                "overwrite": self.overwrite,
                "ignoreWC": self.ignoreWC,
                "xsfKeywords": self.xsfKeywords,
                "xsfLicense": self.xsfLicense,
                "xsfRights": self.xsfRights,
                "xsfSummary": self.xsfSummary,
                "xsfTitle": self.xsfTitle,
            },
        )

        return result is not None and result.is_ok()
