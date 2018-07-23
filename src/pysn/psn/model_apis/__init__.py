"""Container module for the APIs"""
from . import manager


if 'api_list' not in dir():
    api_list = manager.ModelAPIList(__name__)


__all__ = ['api_list']
