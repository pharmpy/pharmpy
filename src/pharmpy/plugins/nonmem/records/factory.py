import re

from pharmpy.plugins.nonmem.exceptions import NMTranParseError
from .raw_record import RawRecord
from .problem_record import ProblemRecord


known_records = {
    #'DATA': 'DataRecord',
    #'ESTIMATION': 'OptionRecord',
    #'INPUT': 'OptionRecord',
    'PROBLEM': ProblemRecord,
    #'SIZES': 'OptionRecord',
    #'THETA': 'ThetaRecord',
    #'OMEGA': 'OmegaRecord',
}


def split_raw_record_name(line):
    """Splits the raw record name of the first line of a record from the rest of the record
    """
    m = re.match(r'(\s*\$[A-za-z]+)(.*)', line)
    if m:
        return m.group(0, 1)
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
        record_class = records_list.known_records[name]
        record = record_class()
    else:
        record = RawRecord(content)

    #if record_class_name == 'RawRecord':
    #    record = raw_record.RawRecord(content)
    #elif record_class_name == 'OptionRecord':
    #    record = option_record.OptionRecord(content)
    #elif record_class_name == 'ThetaRecord':
    #    record = theta_record.ThetaRecord(content)
    #elif record_class_name == 'OmegaRecord':
    #    record = omega_record.OmegaRecord(content)
    #elif record_class_name == 'ProblemRecord':
    #    record = problem_record.ProblemRecord(content)
    #elif record_class_name == 'DataRecord':
    #    record = data_record.DataRecord(content)
    record.raw_name = raw_name
    #record.name = name
    return record
