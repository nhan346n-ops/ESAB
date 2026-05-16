import pytest

from pyat.sensor.nmea_parser import NMEASentence, ParseError, ChecksumError

NMEA_bad_checksum = "$PIXSE,LMNIN_,0.355,0.149,0.086,0.157,0.156,101702.016560,1*78"
NMEA_bad_data = "Non,NMEA,data"
NMEA_not_implemented_sentence = "$HETHS,191.75,A*16"
NMEA_known_sentence = [
    "$HEHDT,41.87,T*25",
    "$PIXSE,ATITUD,-0.729,-15.705*58",
    "$PIXSE,POSITI,39.27301545,14.38044597,-731.923*4D",
    "$PIXSE,SPEED_,0.972,1.082,0.412*61",
    "$PIXSE,UTMWGS,S,33,446558.301,4347257.020,-731.923*3F",
    "$PIXSE,HEAVE_,-0.211,0.031,-6.256*7E",
    "$PIXSE,TIME__,085832.738582*6B",
    "$PIXSE,DEPIN_,732.811,085832.643253*7D",
    "$PIXSE,LOGDVL,1500.00,1520.56,69.94*60",
    "$PIXSE,LOGDVL,0.00,1520.56,0.00*66",
    "$PIXSE,LOGWAT,1.550,-0.001,-0.036,0.039,0.028,0.089,0.089,085832.615653*7E",
    "$PIXSE,LOGIN_,1.468,0.039,0.033,0.000,085832.536304*67",
    "$PIXSE,LMNIN_,1.550,0.039,0.028,0.089,0.089,085832.615653,1*76",
]


def test_raises_checksum_error():
    """
    Verify a ChecksumError exception is raised
    """
    with pytest.raises(ChecksumError):
        NMEASentence.parse(NMEA_bad_checksum)


def test_raises_parse_error():
    """
    Verify a ParseError exception is raised for bad formated NMEA sentence
    """
    with pytest.raises(ParseError):
        NMEASentence.parse(NMEA_bad_data)


def test_raises_not_implemented_error():
    """
    Verify a NotImplementedError exception is raised for unknown NMEA sentence
    """
    with pytest.raises(NotImplementedError):
        NMEASentence.parse(NMEA_not_implemented_sentence)


def test_raises_no_exception():
    """
    Verify that NMEA parsing raises no exception for implemented PHINS NMEA repeater data
    """
    for line in NMEA_known_sentence:
        try:
            # try to parse NMEA data
            msg = NMEASentence.parse(line)
            print(msg)
        except Exception as e:
            assert False, f"'NMEASentence.parse()' raised an exception {e}"
