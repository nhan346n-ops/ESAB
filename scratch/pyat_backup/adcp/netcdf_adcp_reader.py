from pyat.xsf.xsf_reader import XSFReader


class ADCPReader(XSFReader):
    """
    Class used for reading a xsf file
    """

    def __retrieve_known_variables__(self):
        # open the file
        self.ignore_variable = [
            "NMEA_datagram",
            "time",
            "annotation_text",
            "annotation_category",
            "beam",
        ]
        self.beamGroup = self.dataset["/Sonar/Beam_group1"]

        swath_count = self.dataset["/Sonar/"].dimensions["ping_time"]
        self.swath_count = swath_count.size

        beam_count = self.beamGroup.dimensions["beam"]
        self.beam_count = beam_count.size
