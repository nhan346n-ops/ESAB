import os.path

import matplotlib.pyplot as plt
import numpy as np

from pyat import ConstantModel, AngleNormalizer
from pyat.sonarscope.model.signal.ping_detection_signal import PingDetectionSignal
from pyat.sonarscope.model.sounder_lib import SounderType
from pyat import disable_warning
from pyat import scan_dir
from pyat.sonarscope.model.constants import VariableKeys as Key

from pyat.sonarscope.bs_correction.stats_computer import MeanBSComputer, MeanBSModel
from pyat.xsf import xsf_driver

input_dir = r"C:\data\datasets\Backscatter\Compensation\THALIA_ESSDEC2019\EM2040\XSF\reduced"
work_dir = r"C:\data\datasets\Backscatter\Compensation\THALIA_ESSDEC2019\EM2040\XSF\reduced\workdir"


file_list = list(scan_dir(input_dir, [f"00*.xsf.nc"]))
input_files = list(map(str, file_list))
# read all files
sounder_type = SounderType.EM2040_ALL

nc_file = os.path.join(work_dir, "curves.nc")


def compute_stats() -> MeanBSModel:
    computer = MeanBSComputer(sounder_type=sounder_type)
    # compute measured mean values per mode
    mean_model = computer.compute(input_files=input_files)
    # retrieve statistic data and compute all curves per mode
    # disp.plot(mean_model, curve_detail, display_count=False)
    # plt.show(block=True)
    mean_model.save_to_netcdf(nc_file)
    #
    return mean_model


compute_stats()
mean_model = MeanBSModel.read_from_netcdf(nc_file)

# apply it to backscatter data

# just pick one file
file_list = list(scan_dir(input_dir, [f"0099*.xsf.nc"]))


disable_warning()

normalizer = AngleNormalizer(sounder_type=sounder_type, avg_model=ConstantModel(mean_bs=mean_model))
for f in file_list:

    # read bs to retrieve non corrected values
    xsf = xsf_driver.XsfDriver(file_path=f)
    xsf.open()
    model = PingDetectionSignal(xsf_dataset=xsf)
    model.read([Key.DETECTION_BACKSCATTER, Key.DETECTION_INCIDENCE_ANGLE])
    bs_value = model.xr_dataset[Key.DETECTION_BACKSCATTER].to_numpy()
    xsf.close()

    # get corrected backscatter values
    bs_corrected = normalizer.apply_on_file(f)

    # show result
    fig, (x, y) = plt.subplots(2, 1, sharex=True)
    x.imshow(bs_value.transpose(), cmap="Greys")
    x.set_title(f"Input backscatter values {np.nanmin(bs_value):.2f}: {np.nanmax(bs_value):.2f}")
    y.imshow(bs_corrected.transpose(), cmap="Greys")
    y.set_title(f"Corrected backscatter values {np.nanmin(bs_corrected):.2f}: {np.nanmax(bs_corrected):.2f}")
    plt.show(block=True)
