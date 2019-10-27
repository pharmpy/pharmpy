# The NONMEM Model class
from pathlib import Path

from .nmtran_parser import NMTranParser
from pharmpy.parameter import ParameterSet


#class Model(pharmpy.Model):
class Model:
    def __init__(self, obj, from_string=False, **kwargs):
        if from_string:         # FIXME: Now duplication with detect. Centralize the IO to antoher class.
                                # FIXME Also __new__ might be appropriate in base class here. Could have factory there.
            fh = StringIO(obj)
        else:
            fh = open(obj, 'r')
        content = fh.read()
        parser = NMTranParser()
        self.control_stream = parser.parse(content)

    @staticmethod
    def detect(obj, from_string=False, *args, **kwargs):
        """ Check if obj is the path to a NONMEM control stream.
        i.e. check if it is a file that contain $PRO
        """
        if from_string:                 # FIXME: Again centralize the IO of models to another class
            fh = StringIO(obj)
        else:
            if isinstance(obj, str):
                path = Path(obj)
            elif isinstance(obj, Path):
                path = obj
            else:
                return False
            fh = open(path, 'r')
        for line in fh:
            if line.startswith('$PRO'):
                return True

        return False

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
