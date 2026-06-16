# The (legacy) NONMEM FCON Model class
# This module was created to allow reading in FDATA files
# using the format in the FCONS for testing and verification
# purposes.

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Container, Optional

from pharmpy.deps import pandas as pd
from pharmpy.model import Model as BaseModel
from pharmpy.model import ModelSyntaxError

FORTRAN_FORMAT = re.compile(r'(\d+)\w(\d+)')


def detect_model(src, *args, **kwargs):
    """Check if src represents a legacy NONMEM control stream"""
    if isinstance(src, str) and src.startswith("FILE    FSTREAM"):
        return Model
    else:
        return None


@dataclass
class FCONInternals:
    code: str
    path: Path


class Model(BaseModel):
    def __init__(self, **kwargs):
        self._internals = kwargs['internals']
        self._dataset = kwargs['dataset']


def _parse_labels_and_formats(code):
    in_labl = False
    labels = []
    lines = iter(code.split('\n'))
    for line in lines:
        if line.startswith('LABL'):
            in_labl = True
        if in_labl:
            stripped = line[4:].replace(' ', '')
            if stripped.startswith(',') or (not line.startswith('LABL') and line[0].isalpha()):
                in_labl = False
            else:
                a = stripped.split(',')
                labels.extend(a)
        if line.startswith('FORM'):
            next_line = next(lines)
            formats = parse_formats(next_line.strip()[1:-1])
            break
    else:
        raise ModelSyntaxError("Problems parsing the FORM record in FCONS")

    return labels, formats


def parse_formats(line):
    # NOTE: This does not support all FORTRAN formats
    # Some assumptions:
    #   1. No nested repetitions, e.g: 2(2(2F2.0/))
    #   2. Grouping is done left to write, e.g: 3E19.0,2(4E19.0/) is equivalent to 3E19.0,4E19.0,4E19.0/
    #   3. Parenthesis is only used in repetitions

    fmts_split = _split_formats(line)

    fmts_expanded = []
    for fmt in fmts_split:
        outer_match = re.match(r'(\d+)?\((.+)\)', fmt)
        if outer_match:
            no_of_repeats = int(outer_match.group(1))
            inner_fmt = outer_match.group(2)
        else:
            no_of_repeats = 1
            inner_fmt = fmt
        fmts_expanded.extend([inner_fmt] * no_of_repeats)

    fmts_grouped, current_group = [], []
    for fmt in fmts_expanded:
        if '/' in fmt:
            current_group.append(fmt.rstrip('/'))
            fmts_grouped.append(current_group)
            current_group = []
        else:
            current_group.append(fmt)

    if current_group:
        fmts_grouped.append(current_group)

    return fmts_grouped


def _split_formats(line):
    line_new = []
    depth = 0

    for c in line:
        if c == '(':
            depth += 1
            line_new.append(c)
        elif c == ')':
            depth -= 1
            line_new.append(c)
        elif c == '/' and depth == 0:
            line_new.append('/,')  # Not valid FORTRAN format but to simplify splitting
        else:
            line_new.append(c)

    fmts = ''.join(line_new).rstrip(',')  # E.g. 2E19.0/,2F2.0/, -> 2E19.0/,2F2.0/
    return fmts.split(',')


def parse_dataset(code, path):
    labels, formats = _parse_labels_and_formats(code)
    df = read_fdata_file(path.parent / 'FDATA', labels, formats)
    df.index = range(1, len(df) + 1)

    # Copied conversion from pharmpy/model/external/nonmem/dataset/__init__.py
    idcol = _idcol(labels)
    if all((_ids := df[idcol].astype('int32')) == df[idcol]):
        df[idcol] = _ids

    return df


def get_format_args(fmt):
    m = FORTRAN_FORMAT.match(fmt)
    if not m:
        raise ModelSyntaxError(f"Unrecognized Data format in FORM: {fmt}")
    no_of_cols, col_width = int(m.group(1)), int(m.group(2))
    return no_of_cols, col_width


def read_fdata_file(path, labels, formats):
    current_record = []
    parsed_rows = []
    i = 0

    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if not line:
                continue
            parsed_line = parse_line(line, formats[i])
            current_record.extend(parsed_line)
            i += 1

            if len(current_record) >= len(labels):
                assert len(current_record) == len(labels)
                parsed_rows.append(current_record)
                current_record = []
                i = 0

    df = pd.DataFrame(parsed_rows, columns=labels)
    return df


def parse_line(line, fmts):
    offset, end = 0, 0
    values = []
    for fmt in fmts:
        no_of_cols, col_width = get_format_args(fmt)
        for i in range(no_of_cols):
            start = i * col_width + offset
            end = start + col_width
            value = line[start:end].strip()
            if '.' in value:
                value = float(value)
            else:  # NONMEM only supports numbers
                value = int(value)
            values.append(value)
        offset = end
    return values


def _idcol(columns: Container[str | None]) -> str | None:
    if 'ID' in columns:
        return 'ID'
    elif 'L1' in columns:
        return 'L1'
    else:
        return None


def parse_model(code: str, path: Path, missing_data_token: Optional[str] = None):
    internals = FCONInternals(code=code, path=path)
    dataset = parse_dataset(code, path)
    return Model(internals=internals, dataset=dataset)
