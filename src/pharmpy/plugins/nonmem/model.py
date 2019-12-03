# The NONMEM Model class
import re
from pathlib import Path

import pharmpy.model
from .nmtran_parser import NMTranParser
from pharmpy.parameter import ParameterSet
import pharmpy.plugins.nonmem.input


class Model(pharmpy.model.Model):
    def __init__(self, src, **kwargs):
        parser = NMTranParser()
        self.source = src
        self.control_stream = parser.parse(src.code)
        self.input = pharmpy.plugins.nonmem.input.ModelInput(self)

    @staticmethod
    def detect(src, *args, **kwargs):
        """ Check if src represents a NONMEM control stream
        i.e. check if it is a file that contain $PRO
        """
        return bool(re.search(r'^\$PRO', src.code, re.MULTILINE))

    def validate(self):
        """Validates NONMEM model (records) syntactically."""
        self.control_stream.validate()

    @property
    def parameters(self):
        """Get the ParameterSet of all parameters
        """
        next_theta = 1
        params = ParameterSet()
        for theta_record in self.control_stream.get_records('THETA'):
            thetas = theta_record.parameters(next_theta)
            params.update(thetas)
            next_theta += len(thetas)
        return params
