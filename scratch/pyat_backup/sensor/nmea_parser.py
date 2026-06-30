import operator
import re
from functools import reduce
from typing import List

from pyat.sensor.nmea import TalkerSentenceFormat, ProprietarySentenceFormat, has_subtypes, GPSQualityIndicator


class ParseError(ValueError):
    """
    defines a ParseError
    """


class ChecksumError(ParseError):
    """
    defines a ChecksumError
    """


class NMEASentence:
    """
    A generic NMEA sentence parser based on NMEA 0183 Protocol
    """

    sentence_re = re.compile(
        r"""
        # start of string, optional whitespace, optional '$'
        ^\s*\$?

        # message (from '$' or start to checksum or end, non-inclusve)
        (?P<nmea_str>
            # sentence type identifier
            (?P<sentence_type>

                # proprietary sentence
                (P\w{3,4},)|

                # query sentence, ie: 'CCGPQ,GGA'
                # NOTE: this should have no data
                (\w{2}\w{2}Q,\w{3})|

                # taker sentence, ie: 'GPGGA'
                (\w{2}\w{3},)
            )

            # rest of message
            (?P<data>[^*]*)

        )
        # checksum: *HH
        (?:[*](?P<checksum>[A-F0-9]{2}))?

        # optional trailing whitespace
        \s*[\r\n]*$
        """,
        re.X | re.IGNORECASE,
    )

    talker_re = re.compile(r"^(?P<talker>\w{2})(?P<sentence>\w{3}),$")
    query_re = re.compile(r"^(?P<talker>\w{2})(?P<listener>\w{2})Q,(?P<sentence>\w{3})$")
    proprietary_re = re.compile(r"^P(?P<manufacturer>\w{3,4}),$")

    @staticmethod
    def compute_checksum(nmea_str):
        """
        Computes NMEA sentence checksum
        """
        return reduce(operator.xor, map(ord, nmea_str), 0)

    @staticmethod
    def parse(line: str, check=False):
        """
        Parses a string representing a NMEA 0183 sentence.
        Returns a dict whose fields and values datatypes are defined in nmea.py for each NMEA sentence type
        Raises :
            - ParseError if the string could not be parsed,
            - ChecksumError if the checksum did not match.
        """
        # Match each group of the NMEA regex
        match = NMEASentence.sentence_re.match(line)
        if not match:
            raise ParseError("could not parse data", line)

        nmea_str = match.group("nmea_str")
        data_str = match.group("data")
        checksum = match.group("checksum")
        sentence_type = match.group("sentence_type").upper()
        data = data_str.split(",")

        # Validate checksum
        if checksum:
            cs_declared = int(checksum, 16)
            cs_actual = NMEASentence.compute_checksum(nmea_str)
            if cs_declared != cs_actual:
                raise ChecksumError(f"checksum does not match: {cs_declared:02X} != {cs_actual:02X}", data)
        elif check:
            raise ChecksumError("strict checking requested but checksum missing", data)

        # 1st Parse Proprietary sentence
        proprietary_match = NMEASentence.proprietary_re.match(sentence_type)
        if proprietary_match:
            manufacturer = proprietary_match.group("manufacturer")
            if manufacturer in ProprietarySentenceFormat:
                return read_nmea_data(data, ProprietarySentenceFormat[manufacturer])
            else:
                raise NotImplementedError(f"NMEA proprietary {manufacturer} sentence parsing not yet implemented", line)
        # 2nd Parse Talker sentence
        talker_match = NMEASentence.talker_re.match(sentence_type)
        if talker_match:
            talker = talker_match.group("talker")
            sentence_id = talker_match.group("sentence")
            if sentence_id in TalkerSentenceFormat.__members__:
                return read_nmea_data(data, TalkerSentenceFormat[sentence_id])
            else:
                raise NotImplementedError(f"NMEA {sentence_id} sentence parsing not yet implemented", line)

        # 3rd Parse Query sentence
        query_match = NMEASentence.query_re.match(sentence_type)
        if query_match and not data_str:
            raise NotImplementedError(f"NMEA query sentence parsing not yet implemented : {sentence_type}", line)

        # if None of them could be parsed raise an Error
        raise ParseError(f"could not parse sentence type: {sentence_type}", line)


def read_nmea_data(data: List[str], nmea_format) -> dict:
    """
    Reads NMEA data based on its format description.
    Raises ParseError if field count does not match data format description
    """
    parsed_data = {}
    nmea_format_basename = ""

    # Check if format has subtypes, and retrieve right formatter:
    if has_subtypes(nmea_format):
        nmea_format_basename = nmea_format.__name__
        format_subtype = data[0]
        data.pop(0)

        if format_subtype not in nmea_format.__members__:
            raise NotImplementedError(
                f"NMEA {nmea_format_basename} {format_subtype} proprietary sentence parsing not yet implemented", data
            )

        nmea_format = nmea_format[format_subtype]

    # Check if the number of fields matches the expected number
    if len(data) != len(nmea_format.fieldname):
        raise ParseError(
            f"Field count mismatch for {nmea_format.name}. Expected {len(nmea_format.desc)}, got {len(data)}", data
        )

    # Store NMEA format type
    if nmea_format_basename:
        parsed_data["type"] = f"{nmea_format_basename}_{nmea_format.name}"
    else:
        parsed_data["type"] = nmea_format.name

    # Loop through the fields and convert them based on each read_formatter defined in the format
    for i, (field_name, read_formatter) in enumerate(zip(nmea_format.fieldname, nmea_format.read_formatter)):
        if field_name is None:
            # skip ignored fields
            continue
        raw_value = data[i]
        if read_formatter is None:
            parsed_data[field_name] = raw_value
        else:
            parsed_data[field_name] = read_formatter(raw_value)
            # check GPS data quality -> ignore not available GPS data
            if parsed_data[field_name] is GPSQualityIndicator.NOT_AVAILABLE:
                raise ParseError(f"Ignoring NOT AVAILABLE {field_name} data", data)

    return parsed_data
