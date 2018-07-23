"""NONMEM 7.x API module"""
from . import records
from .detect import detect
from .model import Model


name = 'nonmem'
description = 'NONMEM 7.x'


__all__ = ['name', 'description', 'records', 'detect', 'Model']
