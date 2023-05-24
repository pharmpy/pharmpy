# The (legacy) NONMEM FCON Model class
# This module was created to allow reading in FDATA files
# using the format in the FCONS for testing and verification
# purposes.

import re
from dataclasses import dataclass
from pathlib import Path

from pharmpy.deps import pandas as pd
from pharmpy.model import Model as BaseModel
from pharmpy.model import ModelSyntaxError


def detect_model(src, *args, **kwargs):
    """Check if src represents a legacy NONMEM control stream"""
    if isinstance(src, str) and src.startswith("FILE    FSTREAM"):
        return Model
    else:
        return None


@dataclass
class FCONInternals:
    code: str = None
    path: Path = None


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
            if stripped.startswith(','):
                in_labl = False
            else:
                a = stripped.split(',')
                labels.extend(a)
        if line.startswith('FORM'):
            next_line = next(lines)
            formats = next_line.strip()[1:-1].split(',')
            break
    else:
        raise ModelSyntaxError("Problems parsing the FORM record in FCONS")

    return labels, formats


def parse_dataset(code, path):
    labels, formats = _parse_labels_and_formats(code)
    widths = []
    for fmt in formats:
        m = re.match(r'(\d+)\w(\d+)', fmt)
        if not m:
            raise ModelSyntaxError(f"Unrecognized Data format in FORM: {fmt}")
        widths += [int(m.group(2))] * int(m.group(1))
    df = pd.read_fwf(path.parent / 'FDATA', widths=widths, header=None, names=labels)
    return df


def parse_model(code: str, path: Path):
    internals = FCONInternals(code=code, path=path)
    dataset = parse_dataset(code, path)
    return Model(internals=internals, dataset=dataset)
