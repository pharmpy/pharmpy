# The NONMEM Model class

#import pharmpy
from pathlib import Path


#class Model(pharmpy.Model):
class Model:
    def __init__(self, obj, **kwargs):
        pass

    @staticmethod
    def detect(obj, *args, **kwargs):
        """ Check if obj is the path to a NONMEM control stream.
        i.e. check if it is a file that contain $PRO
        """
        if isinstance(obj, str):
            path = Path(obj)
        elif isinstance(obj, Path):
            path = obj
        else:
            return False

        with open(path, 'r') as fh:
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
        for theta_record in self.get_records('THETA'):
            thetas = theta_record.parameters(next_theta)
            params.update(thetas)
            next_theta += len(thetas) 
        return params
