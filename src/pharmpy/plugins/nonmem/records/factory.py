import re

from pharmpy.plugins.nonmem.exceptions import NMTranParseError
from .raw_record import RawRecord
from .problem_record import ProblemRecord
from .parsers import ProblemRecordParser


# Dictionary from canonical record name to record class and non_empty rules of parser
known_records = {
    #'DATA': 'DataRecord',
    #'ESTIMATION': 'OptionRecord',
    #'INPUT': 'OptionRecord',
    'PROBLEM': (ProblemRecord, ProblemRecordParser)
    #'SIZES': 'OptionRecord',
    #'THETA': 'ThetaRecord',
    #'OMEGA': 'OmegaRecord',
}


def split_raw_record_name(line):
    """Splits the raw record name of the first line of a record from the rest of the record
    """
    m = re.match(r'(\s*\$[A-za-z]+)(.*)', line, flags=re.MULTILINE|re.DOTALL)
    if m:
        return m.group(1, 2)
    else:
        raise NMTranParseError(f'Bad record name in: {line}')


def get_canonical_record_name(raw_name):
    """Gets the canonical (standardized) record name from a raw_name"""
    bare = raw_name.lstrip()[1:].upper()       # remove initial white space and the '$'
    if len(bare) >= 3:
        for name in known_records:
            if name.startswith(bare):
                return name
    return None


def create_record(chunk):
    raw_name, content = split_raw_record_name(chunk)
    name = get_canonical_record_name(raw_name)
    if name:
        record_class, record_parser_class = known_records[name]
        record = record_class()
        parser = record_parser_class(content)
        record.root = parser.root
    else:
        record = RawRecord(content)

    record.raw_name = raw_name

    return record
