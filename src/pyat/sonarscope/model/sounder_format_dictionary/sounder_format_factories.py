import pyat.sonarscope.model.sounder_format_dictionary.all_kongsberg_dictionary as all_kongsberg
import pyat.sonarscope.model.sounder_format_dictionary.kmall_kongsberg_dictionary as kmall_kongsberg
import pyat.sonarscope.model.sounder_format_dictionary.reson_dictionary as reson
from pyat.sonarscope.model.sounder_format_dictionary.common_dictionary import VariablesDictionary
from pyat.sonarscope.model.sounder_lib import SounderRawFileFormat
from pyat.xsf.xsf_driver import XsfDriver


def get_variables_dictionary(xsf_dataset: XsfDriver) -> VariablesDictionary:
    sounder_format = SounderRawFileFormat.from_dataset(xsf_dataset)
    if SounderRawFileFormat.ALL == sounder_format:
        return all_kongsberg
    elif SounderRawFileFormat.KMALL == sounder_format:
        return kmall_kongsberg
    elif SounderRawFileFormat.S7K == sounder_format:
        return reson

    raise NotImplementedError(f"Sounder format not supported or not recognized")
