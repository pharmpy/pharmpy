# The NONMEM Model class
import re

import pharmpy.model
import pharmpy.plugins.nonmem.input
from pharmpy.parameter import ParameterSet
from pharmpy.plugins.nonmem.results import NONMEMChainedModelfitResults

from .nmtran_parser import NMTranParser


class Model(pharmpy.model.Model):
    def __init__(self, src, **kwargs):
        super().__init__()
        parser = NMTranParser()
        self.source = src
        if not self.source.filename_extension:
            self.source.filename_extension = '.ctl'
        self.name = self.source.path.stem
        self.control_stream = parser.parse(src.code)
        if self.source.path.is_file():
            lst_path = self.source.path.with_suffix('.lst')
            if lst_path.exists():
                num_est = len(self.control_stream.get_records('ESTIMATION'))
                self.modelfit_results = NONMEMChainedModelfitResults(lst_path, num_est)
        self.input = pharmpy.plugins.nonmem.input.ModelInput(self)
        self._parameters_updated = False

    @staticmethod
    def detect(src, *args, **kwargs):
        """ Check if src represents a NONMEM control stream
        i.e. check if it is a file that contain $PRO
        """
        return bool(re.search(r'^\$PRO', src.code, re.MULTILINE))

    def update_source(self, force=False):
        """Update the source"""
        if self.input._dataset_updated:
            # FIXME: If no name set use the model name. Set that when setting dataset to input!
            datapath = self.input.dataset.pharmpy.write_csv(force=force)
            self.input.path = datapath

            data_record = self.control_stream.get_records('DATA')[0]

            label = self.input.dataset.columns[0]
            data_record.ignore_character_from_header(label)

            # Remove IGNORE/ACCEPT. Could do diff between old dataset and find simple
            # IGNOREs to add i.e. for filter out certain ID.
            del(data_record.ignore)
            del(data_record.accept)
            self._dataset_updated = False

        self._update_parameters()

        super().update_source()

    def _update_parameters(self):
        """ Update parameters
        """
        if not self._parameters_updated:
            return

        params = self.parameters
        next_theta = 1
        for theta_record in self.control_stream.get_records('THETA'):
            theta_record.update(params, next_theta)
            next_theta += len(theta_record)
        next_omega = 1
        for omega_record in self.control_stream.get_records('OMEGA'):
            next_omega = omega_record.update(params, next_omega)
        next_sigma = 1
        for sigma_record in self.control_stream.get_records('SIGMA'):
            next_sigma = sigma_record.update(params, next_sigma)

        self._parameters_updated = False

    def validate(self):
        """Validates NONMEM model (records) syntactically."""
        self.control_stream.validate()

    @property
    def parameters(self):
        """Get the ParameterSet of all parameters
        """
        try:
            return self._parameters
        except AttributeError:
            pass

        next_theta = 1
        params = ParameterSet()
        for theta_record in self.control_stream.get_records('THETA'):
            thetas = theta_record.parameters(next_theta)
            params.update(thetas)
            next_theta += len(thetas)
        next_omega = 1
        for omega_record in self.control_stream.get_records('OMEGA'):
            omegas, next_omega = omega_record.parameters(next_omega)
            params.update(omegas)
        next_sigma = 1
        for sigma_record in self.control_stream.get_records('SIGMA'):
            sigmas, next_sigma = sigma_record.parameters(next_sigma)
            params.update(sigmas)
        self._parameters = params
        return params

    @parameters.setter
    def parameters(self, params):
        """params can be a ParameterSet or a dict-like with name: value

           Current restricions:
            * ParameterSet not supported
            * Only set the exact same parameters. No additions and no removing of parameters
        """
        if isinstance(params, ParameterSet):
            raise NotImplementedError("Can not yet update parameters from ParameterSet")
        else:
            if len(self.parameters) != len(params):
                raise NotImplementedError("Current number of parameters in model and parameters to "
                                          "set are not the same. Not yet supported")
            for p in self.parameters:
                name = p.symbol.name
                if name not in params:
                    raise ValueError(f"Parameter '{name}' not found in input. Currently "
                                     f"not supported")
                if p.lower >= params[name]:
                    raise ValueError(f"Cannot set a lower parameter initial estimate "
                                     f"{params[name]} for parameter '{name}' than the "
                                     f"current lower bound {p.lower}")
                if p.upper <= params[name]:
                    raise ValueError(f"Cannot set a higher parameter initial estimate "
                                     f"{params[name]} for parameter '{name}' than the "
                                     f"current upper bound {p.upper}")
                p.init = params[name]
        self._parameters_updated = True

    def __str__(self):
        return str(self.control_stream)
