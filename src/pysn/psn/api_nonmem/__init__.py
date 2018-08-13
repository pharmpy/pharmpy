# -*- encoding: utf-8 -*-

from .. import GenericParser
from .. import api_generic as generic
from .detect import detect
from .model import Model

__all__ = ['detect', 'Model', 'generic', 'GenericParser']
