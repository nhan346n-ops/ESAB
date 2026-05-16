#! /usr/bin/env python3
# coding: utf-8

from typing import List

import sonar_netcdf.sonar_groups as sg

from pyat.xsf.xsf_driver import XsfDriver


class HistoryUpdater:
    """Class managing the transfer of history"""

    def update_history_from_0_2(self, ref_xsf: XsfDriver, o_xsf: XsfDriver) -> None:
        """
        transfer history attribute from a reference XSF version 0.2 to the last version
        Since version 0.3, history is stored in attribute. Previously, was a variable
        """
        # Starting history with migration informations
        history: List[str] = []
        if "history" in ref_xsf.dataset.variables:
            history.extend(
                f"{hist_line}" for hist_line in list(ref_xsf.dataset["history"][:])
            )
        self.__insert_history_0_3(o_xsf, history)

    def update_history_from_0_3(self, ref_xsf: XsfDriver, o_xsf: XsfDriver) -> None:
        """
        transfer history attribute for XSF version 0.3
        Since version 0.3, history is stored in attribute
        """
        # Starting history with migration informations
        history: List[str] = []
        if "Provenance" in ref_xsf.dataset.groups:
            ref_provenanceGrp = ref_xsf.dataset.groups["Provenance"]
            if ref_provenanceGrp.history:
                list_history = ref_provenanceGrp.history if isinstance(ref_provenanceGrp.history, list) else [ref_provenanceGrp.history]
                history.extend(list_history)
        self.__insert_history_0_3(o_xsf, history)

    def __insert_history_0_3(self, o_xsf: XsfDriver, history: List[str]) -> None:
        """Insert specified history to Provenance.history attribute"""
        provenanceGrpStub = sg.ProvenanceGrp()
        o_provenanceGrp = (
            provenanceGrpStub.create_group(o_xsf.dataset)
            if "Provenance" not in o_xsf.dataset.groups
            else o_xsf.dataset.groups["Provenance"]
        )
        if o_provenanceGrp.history:
            list_history = o_provenanceGrp.history if isinstance(o_provenanceGrp.history, list) else [o_provenanceGrp.history]
            history.extend(list_history)

        o_provenanceGrp.history = history
