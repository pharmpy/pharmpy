import re
from typing import Dict, Tuple, Type

from pharmpy.model import ModelSyntaxError

from .abbreviated_record import AbbreviatedRecord
from .code_record import ParsedCodeRecord
from .data_record import DataRecord
from .estimation_record import EstimationRecord
from .etas_record import EtasRecord
from .model_record import ModelRecord
from .omega_record import ParsedOmegaRecord
from .option_record import OptionRecord
from .parsers import (
    AbbreviatedRecordParser,
    CodeRecordParser,
    DataRecordParser,
    OmegaRecordParser,
    OptionRecordParser,
    ProblemRecordParser,
    RecordParser,
    SimulationRecordParser,
    ThetaRecordParser,
)
from .problem_record import ProblemRecord
from .raw_record import RawRecord
from .record import ParsedRecord
from .simulation_record import SimulationRecord
from .sizes_record import SizesRecord
from .subroutine_record import SubroutineRecord
from .table_record import TableRecord
from .theta_record import ThetaRecord

# Dictionary from canonical record name to record class and non_empty rules of parser
known_records: Dict[str, Tuple[Type[ParsedRecord], RecordParser]] = {
    'ABBREVIATED': (AbbreviatedRecord.parsed, AbbreviatedRecordParser()),
    'COVARIANCE': (OptionRecord.parsed, OptionRecordParser()),
    'DATA': (DataRecord.parsed, DataRecordParser()),
    'DES': (ParsedCodeRecord, CodeRecordParser()),
    'ERROR': (ParsedCodeRecord, CodeRecordParser()),
    'ESTIMATION': (EstimationRecord.parsed, OptionRecordParser()),
    'ETAS': (EtasRecord.parsed, OptionRecordParser()),
    'INPUT': (OptionRecord.parsed, OptionRecordParser()),
    'MODEL': (ModelRecord.parsed, OptionRecordParser()),
    'OMEGA': (ParsedOmegaRecord, OmegaRecordParser()),
    'PK': (ParsedCodeRecord, CodeRecordParser()),
    'PRED': (ParsedCodeRecord, CodeRecordParser()),
    'PROBLEM': (ProblemRecord.parsed, ProblemRecordParser()),
    'SIGMA': (ParsedOmegaRecord, OmegaRecordParser()),
    'SIMULATION': (SimulationRecord.parsed, SimulationRecordParser()),
    'SIZES': (SizesRecord.parsed, OptionRecordParser()),
    'SUBROUTINES': (SubroutineRecord.parsed, OptionRecordParser()),
    'TABLE': (TableRecord.parsed, OptionRecordParser()),
    'THETA': (ThetaRecord.parsed, ThetaRecordParser()),
}


def split_raw_record_name(line: str):
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
        elif bare == 'COVR':
            return 'COVARIANCE'
        elif bare == 'ESTM':
            return 'ESTIMATION'
    elif bare == 'PK':
        return bare
    return None


def create_record(chunk: str):
    raw_name, content = split_raw_record_name(chunk)
    canonical_name = get_canonical_record_name(raw_name)
    if canonical_name:
        name = canonical_name
        record_class, record_parser = known_records[name]
        return record_class(name=name, raw_name=raw_name, parser=record_parser, buffer=content)
    else:
        name = raw_name[1:]
        return RawRecord(name=name, raw_name=raw_name, buffer=content)
