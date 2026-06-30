#! /usr/bin/env python3
# coding: utf-8

from datetime import datetime, timezone
from typing import Dict, List, Tuple

import osgeo.ogr as ogr
import pygws.service.execution_context as exec_ctx
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.utils.pyat_logger as log
from pyat.sounder import sounder_driver_factory


class CutMbg:
    """
    Utility class to apply a mask to a MBG file and produce the cut lines
    """

    def __init__(
        self,
        i_paths: List[str],
        mask: str | None = None,
        reverse_mask: bool = False,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Constructor.
        """
        self.i_paths = i_paths
        self.mask = mask
        self.reverse_mask = reverse_mask

        # Prefer to use RSocket monitor if available
        self.monitor = monitor
        if exec_ctx.get_root_progress_monitor() is not None:
            self.monitor = exec_ctx.get_root_progress_monitor()
        self.logger = log.logging.getLogger(self.__class__.__name__)

        # Resulting cut lines
        self.result = ""

    def _write_line(self, from_date: datetime, to_date: datetime, line: int) -> None:
        if self.result:
            self.result = self.result + "\n"
        line_range = (
            from_date.strftime("%d/%m/%Y  %H:%M:%S.%f")[:-3] + "  " + to_date.strftime("%d/%m/%Y  %H:%M:%S.%f")[:-3]
        )
        self.logger.info(f"One line found : {line_range}")
        self.result = self.result + "> " + line_range + "  " + "Line_" + str(line)

    def _cut_input_files(self) -> List[Tuple[datetime, datetime]]:
        """Run method."""
        lines = self.cut_input_files()

        # Using rsocket (if present) to send the result
        rsocket_msg_emitter = exec_ctx.get_rsocket_msg_emitter()
        if rsocket_msg_emitter is not None:
            rsocket_msg_emitter.emit_strings(self.result.split("\n"))
            return []

        return lines

    def cut_input_files(self) -> List[Tuple[datetime, datetime]]:
        """Compute and return the resulting lines."""
        self.monitor.begin_task("Evalutating the lines", len(self.i_paths) + 1)
        zone: ogr.Geometry | None = self._read_zone_in_mask() if self.mask is not None else None
        input_files = self._sort_input_files()
        self.monitor.worked(1)
        lines = self._cut_on_zone(input_files, zone)
        self.logger.info(f"Number of lines found {len(lines)}")

        return lines

    def __call__(self) -> Dict:
        """Run method."""
        try:
            self._cut_input_files()
            return self._report_result()
        except Exception as error:
            self.logger.error(f"An exception was thrown : {str(error)}", exc_info=True)
            self.monitor.done()
        return {}

    def _sort_input_files(self) -> List[str]:
        """
        Sort all MBG chronologically
        """
        self.logger.info("Sorting input files")

        # List of File with their datetime
        file_date: List[Tuple[str, int]] = []
        for input_path in self.i_paths:
            with sounder_driver_factory.open_sounder(input_path) as i_driver:
                date = i_driver.read_date_time()[:].flat[0]
                file_date.append((input_path, date))

        # sorts by datetime
        file_date.sort(key=lambda tup: tup[1])
        for tup in file_date:
            date_time = datetime.utcfromtimestamp(tup[1]).strftime("%d/%m/%Y  %H:%M:%S.%f")
            self.logger.info(f"{date_time} - {tup[0]}")

        return [file_date[0] for file_date in file_date]

    def _cut_on_zone(self, input_files: List[str], zone: ogr.Geometry | None) -> List[Tuple[datetime, datetime]]:
        result: List[Tuple[datetime, datetime]] = []
        line_count = 0
        from_date = to_date = None
        for input_path in input_files:
            with sounder_driver_factory.open_sounder(input_path) as i_driver:
                lons = i_driver.read_abscissa()
                lats = i_driver.read_ordinate()
                dates = i_driver.read_date_time()
                if i_driver.sounder_file.antenna_count > 1:
                    # [:, 0] : keep only data from the first antenna
                    lons = lons[:, 0]
                    lats = lats[:, 0]
                    dates = dates[:, 0]
                lons = lons.reshape(i_driver.sounder_file.swath_count)
                lats = lats.reshape(i_driver.sounder_file.swath_count)
                dates = dates.reshape(i_driver.sounder_file.swath_count)

                if zone is not None:
                    for lon, lat, date in zip(lons, lats, dates):
                        point = ogr.Geometry(ogr.wkbPoint)
                        point.AddPoint(lon, lat)
                        if zone.Contains(point) != self.reverse_mask:
                            if from_date is None:
                                from_date = datetime.utcfromtimestamp(date)
                            else:
                                to_date = datetime.utcfromtimestamp(date)
                        else:
                            if from_date and to_date:
                                line_count += 1
                                result.append((from_date, to_date))
                                self._write_line(from_date, to_date, line_count)
                            from_date = None
                            to_date = None
                else:
                    # No zone defined. Consider all the line
                    result.append(
                        (
                            datetime.fromtimestamp(dates[0], timezone.utc),
                            datetime.fromtimestamp(dates[-1], timezone.utc),
                        )
                    )

            self.monitor.worked(1)

        # Is there a left line ?
        if from_date and to_date:
            line_count += 1
            result.append((from_date, to_date))
            self._write_line(from_date, to_date, line_count)
        return result

    def _read_zone_in_mask(self) -> ogr.Geometry:
        """
        Open the shapefile/kml and produce a Geometry covering the area of the mask
        """
        vector_dataset: ogr.DataSource = ogr.Open(self.mask)
        layer_count = vector_dataset.GetLayerCount()
        result: ogr.Geometry = None
        try:
            for i in range(layer_count):
                vector_layer: ogr.Layer = vector_dataset.GetLayer(i)
                for feature in vector_layer:
                    geom: ogr.Geometry = feature.GetGeometryRef()
                    geom.CloseRings()
                    if result is None:
                        result = geom.Buffer(0)
                    else:
                        result = result.Union(geom.Buffer(0))
                    feature = None
                vector_layer = None
            if result is None:
                raise IOError("Bad format for the KML file")
        finally:
            vector_dataset = None  # Close file
        return result

    def _report_result(self) -> Dict:
        self.logger.info("Result")
        self.logger.info(str(self.result))
        return {"result": self.result}


if __name__ == "__main__":
    cutMbg = CutMbg(
        i_paths=[
            "E:/temp/MBG/test1.mbg",
            "E:/temp/MBG/test2.mbg",
        ],
        mask="E:/temp/tmpz9e8t7aj.kml",
    )

    cutMbg()
