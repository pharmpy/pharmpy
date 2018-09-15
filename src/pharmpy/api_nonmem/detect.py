# -*- encoding: utf-8 -*-

"""Filetype detection for NONMEM."""


def detect(lines):
    """Returns true if code lines seem to be a (supported) NONMEM model."""
    for line in lines:
        if line.startswith('$PRO'):
            return True
    return False
