from pathlib import Path

from . import generic
from .records.factory import create_record


class ParameterModel(generic.ParameterModel):
    """A NONMEM 7.x ParameterModel implementation"""

    def initial_estimates(self, problem=0):
        params = self.thetas()

    def thetas(self, problem=0):
        records = self.model.get_records('THETA', problem=problem)
