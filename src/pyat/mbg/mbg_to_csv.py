"""
Export modules of mbg files
"""

import pandas as pd

from pyat.mbg.mbg_driver import MbgDriver


def export_vertical_depth_nmea(input_mbg: MbgDriver, output_file: str):
    """Export mbg vertical depth values to nmea techsas like file format"""
    vertical_depth = input_mbg.read_vertical_depth()
    vertical_depth = vertical_depth.flatten(order="F")  # interleave array if we have two antennas
    latitudes = input_mbg.read_platform_latitudes()
    latitudes = latitudes.flatten(order="F")  # interleave array if we have two antennas
    longitudes = input_mbg.read_platform_longitudes()
    longitudes = longitudes.flatten(order="F")  # interleave array if we have two antennas
    time = input_mbg.read_date_time()
    time = time.flatten(order="F")  # interleave array if we have two antennas

    sounder_name, _ = input_mbg.read_sounder_desc()

    df = pd.DataFrame(
        data={"latitude": latitudes, "longitude": longitudes, "datetime": time, "vertical_depth": vertical_depth}
    )
    # decode datetime
    df = df.astype({"datetime": "datetime64[s]"})
    #remove invalid data
    df = df[~pd.isnull(df['datetime'] )]

    # format correctly datetime
    t = df["datetime"]
    df["date"] = t.apply(lambda x: x.strftime("%d/%m/%Y"))
    df["time"] = t.apply(lambda x: x.strftime("%H:%M:%S"))

    # add specific columns
    df["sounder_name"] = sounder_name
    df["nmea_header"] = "$MDMES"

    df.to_csv(
        output_file,
        sep=",",
        columns=["nmea_header", "date", "time", "sounder_name", "vertical_depth", "latitude", "longitude"],
        index=False,
        #header=False
    )
