import re

from pharmpy.model import ModelSyntaxError

from .abbreviated_record import AbbreviatedRecord
from .code_record import CodeRecord
from .data_record import DataRecord
from .etas_record import EtasRecord
from .model_record import ModelRecord
from .omega_record import OmegaRecord
from .option_record import OptionRecord
from .parsers import (
    AbbreviatedRecordParser,
    CodeRecordParser,
    DataRecordParser,
    OmegaRecordParser,
    OptionRecordParser,
    ProblemRecordParser,
    SimulationRecordParser,
    ThetaRecordParser,
)
from .problem_record import ProblemRecord
from .raw_record import RawRecord
from .simulation_record import SimulationRecord
from .sizes_record import SizesRecord
from .theta_record import ThetaRecord

# Dictionary from canonical record name to record class and non_empty rules of parser
known_records = {
    'ABBREVIATED': (AbbreviatedRecord, AbbreviatedRecordParser),
    'DATA': (DataRecord, DataRecordParser),
    'DES': (CodeRecord, CodeRecordParser),
    'ERROR': (CodeRecord, CodeRecordParser),
    'ESTIMATION': (OptionRecord, OptionRecordParser),
    'ETAS': (EtasRecord, OptionRecordParser),
    'INPUT': (OptionRecord, OptionRecordParser),
    'MODEL': (ModelRecord, OptionRecordParser),
    'OMEGA': (OmegaRecord, OmegaRecordParser),
    'PK': (CodeRecord, CodeRecordParser),
    'PRED': (CodeRecord, CodeRecordParser),
    'PROBLEM': (ProblemRecord, ProblemRecordParser),
    'SIGMA': (OmegaRecord, OmegaRecordParser),
    'SIMULATION': (SimulationRecord, SimulationRecordParser),
    'SIZES': (SizesRecord, OptionRecordParser),
    'SUBROUTINES': (OptionRecord, OptionRecordParser),
    'TABLE': (EtasRecord, OptionRecordParser),  # Etas for now. Has path property
    'THETA': (ThetaRecord, ThetaRecordParser),
}


def split_raw_record_name(line):
    """Splits the raw record name of the first line of a record from the rest of the record"""
    m = re.match(r'(\s*\$[A-za-z]+)(.*)', line, flags=re.MULTILINE | re.DOTALL)
    if m:
        return m.group(1, 2)
    else:
        raise ModelSyntaxError(f'Bad record name in: {line}')


def get_canonical_record_name(raw_name):
    """Gets the canonical (standardized) record name from a raw_name"""
    bare = raw_name.lstrip()[1:].upper()  # remove initial white space and the '$'
    if len(bare) >= 3:
        for name in known_records:
            if name.startswith(bare):
                return name
        # Synonyms
        if 'INFILE'.startswith(bare):
            return 'DATA'
        elif 'SUBS'.startswith(bare):
            return 'SUBROUTINES'
        elif bare == 'SIML' or bare == 'SIMULATE':
            return 'SIMULATION'
    elif bare == 'PK':
        return bare
    return None


def create_record(chunk):
    raw_name, content = split_raw_record_name(chunk)
    name = get_canonical_record_name(raw_name)
    if name:
        record_class, record_parser_class = known_records[name]
        record = record_class(content, record_parser_class)
    else:
        record = RawRecord(content)
        name = raw_name[1:]

    record.name = name
    record.raw_name = raw_name

    return record
