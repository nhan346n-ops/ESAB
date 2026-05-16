#! /usr/bin/env python3
# coding: utf-8

import tempfile as tmp
from datetime import datetime
from typing import List, Tuple

from pyat.mbg.mbg_to_cut import CutMbg
from tests.generator.kml_generator import create_kml
from tests.generator.mbg_generator import make_mbg_with_data


def test_csv_lat_lon_export() -> None:
    """
    Convert a CSV (Emo format) to Tiff
    """
    with tmp.TemporaryDirectory() as o_dir:
        path_mbg1 = make_mbg_with_data(
            783,
            2457349,
            (75004304, 76609492),
            (-17.554924495831486, -17.51470724221027),
            (38.566683532223536, 38.49664401332269),
            o_dir,
        )
        path_mbg2 = make_mbg_with_data(
            1296,
            2457349,
            (67803129, 69602009),
            (-17.692696033303296, -17.64389017109131),
            (38.28327886547648, 38.201800778531485),
            o_dir,
        )

        coord = [
            [38.21688402862747, -17.5892028081691],
            [38.49456269102331, -17.472862644883868],
            [38.553399544570205, -17.583394166905844],
            [38.21688402862747, -17.5892028081691],
        ]
        path_kml = create_kml(o_dir, {"zone": coord})

        out_mbg_path = o_dir + "cut.json"
        cutMbg = CutMbg(
            i_paths=[path_mbg1, path_mbg2],
            mask=path_kml,
        )
        lines: List[Tuple[datetime, datetime]] = cutMbg._cut_input_files()
        assert len(lines) == 1
        assert lines[0][0] == datetime.fromisoformat("2015-11-22 21:06:30.285")
        assert lines[0][1] == datetime.fromisoformat("2015-11-22 21:18:07.792")
