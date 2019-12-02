# The NONMEM Model class
import re
from pathlib import Path

from .nmtran_parser import NMTranParser
from pharmpy.parameter import ParameterSet


#class Model(pharmpy.Model):
class Model:
    def __init__(self, src, **kwargs):
        parser = NMTranParser()
        self.source = src
        self.control_stream = parser.parse(src.code)

    @staticmethod
    def detect(src, *args, **kwargs):
        """ Check if src represents a NONMEM control stream
        i.e. check if it is a file that contain $PRO
        """
        return bool(re.search(r'^\$PRO', src.code, re.MULTILINE))

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
