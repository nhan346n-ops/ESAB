#! /usr/bin/env python3
# coding: utf-8

import pyat.utils.application_utils as app_util
from pyat.dtm.export.dtm_to_ascii import Dtm2Ascii


class Dtm2Emo(Dtm2Ascii):
    def is_exporting_to_xyz(self):
        """
        Return False to generate an emo file
        """
        return False


if __name__ == "__main__":
    app_util.launch_application(app_util.get_json_configuration_file(__file__), Dtm2Emo)
