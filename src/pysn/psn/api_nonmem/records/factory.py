#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import re

from . import option_record as option_record
from . import problem_record as problem_record
from . import raw_record as raw_record
from . import records_list as records_list
from . import theta_record as theta_record

def get_raw_record_name(line):
    """ Get raw record name. It might be of short or alternative form and contain & breaks
    """
    m = re.match(r'(([A-Za-z]|&\n)+)', line)
    if m:
        record_string = m.group(1)
        return record_string
    else:
        return None

def get_canonical_record_name(raw_name):
    """ Get the canonical (standardized) record name from a raw_name
    """
    if not raw_name:
        return None
    short_form = raw_name.replace("&\n", "").upper()
    if len(short_form) >= 3:
        for name in records_list.known_records:
            if name.startswith(short_form):
                return name
    return None

def get_record_content(line):
    """ Strip the raw name from a record string
    """
    m = re.match(r'([A-Za-z]|&\n)+(.*)', line, re.DOTALL)
    return m.group(2)


def create_record(line):
    raw_name = get_raw_record_name(line)
    name = get_canonical_record_name(raw_name)
    content = get_record_content(line)
    if name:
        record_class_name = records_list.known_records[name]
    else:
        record_class_name = 'RawRecord'
    if record_class_name == 'RawRecord':
        record = raw_record.RawRecord(content)
    elif record_class_name == 'OptionRecord':
        record = option_record.OptionRecord(content)
    elif record_class_name == 'ThetaRecord':
        record = theta_record.ThetaRecord(content)
    elif record_class_name == 'ProblemRecord':
        record = problem_record.ProblemRecord(content)
    record.raw_name = raw_name
    record.name = name
    return record
