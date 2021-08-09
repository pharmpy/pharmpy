# The (legacy) NONMEM FCON Model class
# This module was created to allow reading in FDATA files
# using the format in the FCONS for testing and verification
# purposes.

import re

import pandas as pd

import pharmpy.model
from pharmpy.model import ModelSyntaxError


def detect_model(src, *args, **kwargs):
    """Check if src represents a legacy NONMEM control stream"""
    if src.code.startswith("FILE    FSTREAM"):
        return Model
    else:
        return None


class Model(pharmpy.model.Model):
    def __init__(self, src, **kwargs):
        super().__init__()
        self.source = src

    @property
    def dataset(self):
        found = False
        in_labl = False
        labels = []
        for line in self.source.code.split('\n'):
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
                found = True
            elif found:
                formats = line.strip()[1:-1].split(',')
                break
        if not found:
            raise ModelSyntaxError("Problems parsing the FORM record in FCONS")
        widths = []
        for fmt in formats:
            m = re.match(r'(\d+)\w(\d+)', fmt)
            if not m:
                raise ModelSyntaxError(f"Unrecognized Data format in FORM: {fmt}")
            widths += [int(m.group(2))] * int(m.group(1))
        df = pd.read_fwf(
            self.source.path.parent / 'FDATA', widths=widths, header=None, names=labels
        )
        return df
