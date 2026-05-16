"""Library of sounder definition"""

import sonar_netcdf.sonar_groups as sg

from pyat.sonarscope.model.constants import SONAR_GROUP_NAME
from pyat.xsf.xsf_driver import XsfDriver


class SounderModel:
    """Contains all sounder names"""

    EM1002 = "EM1002"
    EM2040 = "EM2040"
    EM120 = "EM120"
    EM122 = "EM122"
    EM124 = "EM124"
    EM302 = "EM302"
    EM304 = "EM304"
    EM710 = "EM710"
    EM712 = "EM712"
    ME70 = "ME70"
    SEABAT7150 = "7150"
    SEABAT7125 = "7125"
    SEABAT7111 = "7111"


class SounderType:
    """Contains all sounder names"""

    # AUTO
    AUTO = "AUTO"
    # COMMON
    COMMON = "COMMON"
    # ALL
    EM1002_ALL = "EM1002_ALL"
    EM2040_ALL = "EM2040_ALL"
    EM120_ALL = "EM120_ALL"
    EM122_ALL = "EM122_ALL"
    EM302_ALL = "EM302_ALL"
    EM710_ALL = "EM710_ALL"
    ME70_ALL = "ME70_ALL"

    # KMALL
    EM2040_KMALL = "EM2040_KMALL"
    EM124_KMALL = "EM124_KMALL"
    EM304_KMALL = "EM304_KMALL"
    EM712_KMALL = "EM712_KMALL"

    # S7K
    SEABAT7150_S7K = "7150_S7K"
    SEABAT7125_S7K = "7125_S7K"
    SEABAT7111_S7K = "7111_S7K"

    SOUNDER_TYPES = [
        COMMON,
        EM1002_ALL,
        EM2040_ALL,
        EM120_ALL,
        EM122_ALL,
        EM302_ALL,
        EM710_ALL,
        ME70_ALL,
        EM2040_KMALL,
        EM124_KMALL,
        EM304_KMALL,
        EM712_KMALL,
        SEABAT7150_S7K,
        SEABAT7125_S7K,
        SEABAT7111_S7K,
    ]

    KONGSBERG_SOUNDER_TYPES = [
        EM1002_ALL,
        EM2040_ALL,
        EM120_ALL,
        EM122_ALL,
        EM302_ALL,
        EM710_ALL,
        ME70_ALL,
        EM2040_KMALL,
        EM124_KMALL,
        EM304_KMALL,
        EM712_KMALL,
    ]

    RESON_SOUNDER_TYPES = [
        SEABAT7150_S7K,
        SEABAT7125_S7K,
        SEABAT7111_S7K,
    ]

    @staticmethod
    def from_type(sounder_type: str) -> str:
        if sounder_type in SounderType.SOUNDER_TYPES:
            return sounder_type
        else:
            raise NotImplementedError(f"Not implemented for other sounder than {SounderType.SOUNDER_TYPES}")

    @staticmethod
    def from_dataset(xsf_dataset: XsfDriver) -> str:
        sounder_type = None
        rawfileformat = SounderRawFileFormat.from_dataset(xsf_dataset)
        model = xsf_dataset[SONAR_GROUP_NAME].sonar_model
        if SounderRawFileFormat.ALL == rawfileformat:
            if SounderModel.EM1002 in model:
                sounder_type = SounderType.EM1002_ALL
            elif SounderModel.EM2040 in model:
                sounder_type = SounderType.EM2040_ALL
            elif SounderModel.EM120 in model:
                sounder_type = SounderType.EM120_ALL
            elif SounderModel.EM122 in model:
                sounder_type = SounderType.EM122_ALL
            elif SounderModel.EM302 in model:
                sounder_type = SounderType.EM302_ALL
            elif SounderModel.EM710 in model:
                sounder_type = SounderType.EM710_ALL
            elif SounderModel.ME70 in model:
                sounder_type = SounderType.ME70_ALL
        elif SounderRawFileFormat.KMALL == rawfileformat:
            if SounderModel.EM2040 in model:
                sounder_type = SounderType.EM2040_KMALL
            elif SounderModel.EM124 in model:
                sounder_type = SounderType.EM124_KMALL
            elif SounderModel.EM304 in model:
                sounder_type = SounderType.EM304_KMALL
            elif SounderModel.EM712 in model:
                sounder_type = SounderType.EM712_KMALL
        elif SounderRawFileFormat.S7K == rawfileformat:
            if SounderModel.SEABAT7150 in model:
                sounder_type = SounderType.SEABAT7150_S7K
            if SounderModel.SEABAT7125 in model:
                sounder_type = SounderType.SEABAT7125_S7K
            if SounderModel.SEABAT7111 in model:
                sounder_type = SounderType.SEABAT7111_S7K
        if sounder_type is None:
            raise NotImplementedError(f"Not implemented for other sounder than {SounderType.SOUNDER_TYPES}")
        return sounder_type


class SounderRawFileFormat:
    """Contains all sounder file formats"""

    ALL = "all"
    KMALL = "kmall"
    S7K = "s7k"

    @staticmethod
    def from_ext(sounder_ext: str) -> str:
        sounder_ext = sounder_ext.lower()
        if sounder_ext not in [SounderRawFileFormat.ALL, SounderRawFileFormat.KMALL, SounderRawFileFormat.S7K]:
            raise NotImplementedError(f"Raw file extension not supported or not found: {sounder_ext}")
        return sounder_ext

    @staticmethod
    def from_dataset(xsf_dataset: XsfDriver) -> str:
        sounder_ext = xsf_dataset.get_provenance_ext()
        if sounder_ext is not None:
            sounder_format = SounderRawFileFormat.from_ext(xsf_dataset.get_provenance_ext())
            return sounder_format
        else:
            # try too guess provenance
            manufacturer = xsf_dataset[SONAR_GROUP_NAME].sonar_manufacturer
            sounder_manufacturer = SounderManufacturer.from_manufacturer(manufacturer)
            if SounderManufacturer.KONGSBERG == sounder_manufacturer:
                # search for sg.RuntimeGrp.PING_MODE_VNAME available only for .all
                # use netcdf api to ensure that group really exist
                if sg.RuntimeGrp.PING_MODE_VNAME in xsf_dataset[sg.RuntimeGrp.get_group_path()].variables:
                    return SounderRawFileFormat.ALL
                else:
                    return SounderRawFileFormat.KMALL
            elif SounderManufacturer.RESON == sounder_manufacturer:
                return SounderRawFileFormat.S7K
        raise NotImplementedError(f"Sounder format not supported or not recognized")


class SounderManufacturer:
    """Contains all sounder constructors"""

    KONGSBERG = "kongsberg"
    RESON = "reson"

    @staticmethod
    def from_manufacturer(sounder_manufacturer: str) -> str:
        sounder_manufacturer = sounder_manufacturer.lower()
        if sounder_manufacturer not in [
            SounderManufacturer.KONGSBERG,
        ]:
            raise NotImplementedError(f"Manufacturer not supported or not found: {sounder_manufacturer}")
        return sounder_manufacturer

    @staticmethod
    def from_type(sounder_type: str) -> str:
        sounder_type = SounderType.from_type(sounder_type)
        if sounder_type in SounderType.KONGSBERG_SOUNDER_TYPES:
            return SounderManufacturer.KONGSBERG
        elif sounder_type in SounderType.RESON_SOUNDER_TYPES:
            return SounderManufacturer.RESON
        else:
            raise NotImplementedError(f"Not implemented for other sounder than {SounderType.SOUNDER_TYPES}")
