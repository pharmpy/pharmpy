"""
===================
Generic Model class
===================

**Base class of all implementations.**

Inherit to *implement*, i.e. to define support for a specific model type. Duck typing is utilized,
but an implementation is expected to implement **all** methods/attributes.

Definitions
-----------
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import sympy


class ModelException(Exception):
    pass


class ModelSyntaxError(ModelException):
    def __init__(self, msg='model syntax error'):
        super().__init__(msg)


class Model:
    """
     Property: name
    """
    @property
    def modelfit_results(self):
        return None

    def update_source(self):
        """Update the source"""
        self.source.code = str(self)

    def write(self, path='', force=False):
        """Write model to file using its source format
           If no path is supplied or does not contain a filename a name is created
           from the name property of the model
           Will not overwrite in case force is True.
           return path written to
        """
        path = Path(path)
        if not path or path.is_dir():
            try:
                filename = f'{self.name}{self.source.filename_extension}'
            except AttributeError:
                raise ValueError('Cannot name model file as no path argument was supplied and the'
                                 'model has no name.')
            path = path / filename
        if not force and path.exists():
            raise FileExistsError(f'File {path} already exists.')
        self.update_source(path=path, force=force)
        self.source.write(path, force=force)
        return path

    def update_inits(self):
        """Update inital estimates of model from its own ModelfitResults
        """
        if self.modelfit_results:
            self.parameters = self.modelfit_results.parameter_estimates
        else:
            # FIXME: Other exception here. ModelfitError?
            raise ModelException("Cannot update initial parameter estimates "
                                 "since parameters were not estimated")

    def copy(self):
        """Create a deepcopy of the model object"""
        return copy.deepcopy(self)

    def update_individual_estimates(self, source):
        self.initial_individual_estimates = self.modelfit_results.individual_estimates

    @property
    def dataset(self):
        raise NotImplementedError()

    def read_raw_dataset(self, parse_columns=tuple()):
        raise NotImplementedError()

    def create_symbol(self, stem):
        """Create a new unique variable symbol

           stem - First part of the new variable name
        """
        # TODO: Also check parameter and rv names and dataset columns
        symbols = self.statements.free_symbols
        i = 1
        while True:
            candidate = sympy.Symbol(f'{stem}{i}')
            if candidate not in symbols:
                return candidate
            i += 1

    def _observation(self):
        stats = self.statements
        for i, s in enumerate(stats):
            if s.symbol.name == 'Y':
                y = s.expression
                break

        for j in range(i, -1, -1):
            y = y.subs({stats[j].symbol: stats[j].expression})

        return y

    def symbolic_population_prediction(self):
        """Symbolic model population prediction
        """
        y = self._observation()

        for eps in self.random_variables.ruv_rvs:
            # FIXME: The rv symbol and the code symbol are different.
            y = y.subs({sympy.Symbol(eps.name): 0})

        for eta in self.random_variables.etas:
            y = y.subs({sympy.Symbol(eta.name): 0})

        return y

    def population_prediction(self, parameters=None, dataset=None):
        """Numeric population prediction

            The prediction is evaluated at the current model parameter values
            or optionally at the given parameter values.
            The evaluation is done for each data record in the model dataset
            or optionally using the dataset argument.

            Return population prediction series
        """
        y = self.symbolic_population_prediction()
        if parameters is not None:
            y = y.subs(parameters)
        else:
            y = y.subs(self.parameters.inits)

        if dataset is not None:
            df = dataset
        else:
            df = self.dataset

        pred = df.apply(lambda row: y.subs(row.to_dict()), axis=1)

        return pred

    def symbolic_eta_gradient(self):
        y = self._observation()
        for eps in self.random_variables.ruv_rvs:
            y = y.subs({sympy.Symbol(eps.name): 0})
        d = [y.diff(sympy.Symbol(x.name)) for x in self.random_variables.etas]
        return d

    def symbolic_eps_gradient(self):
        y = self._observation()
        d = [y.diff(sympy.Symbol(x.name)) for x in self.random_variables.ruv_rvs]
        return d

    def _replace_parameters(self, y, etas, parameters):
        if parameters is not None:
            y = [x.subs(parameters) for x in y]
        else:
            y = [x.subs(self.parameters.inits) for x in y]
        if etas is not None:
            y = [x.subs(etas) for x in y]
        elif self.initial_individual_estimates is not None:
            y = [x.subs(self.initial_individual_estimates) for x in y]
        else:
            repl = {sympy.Symbol(eta.name): 0 for eta in self.random_variables.etas}
            y = [x.subs(repl) for x in y]
        return y

    def eta_gradient(self, etas=None, parameters=None, dataset=None):
        """Numeric eta gradient

           The gradient is evaluated given initial etas, parameters and the model dataset.
           The arguments etas, parameters and dataset can optionally override those
           of the model. Return a DataFrame of gradients.
        """
        y = self.symbolic_eta_gradient()
        y = self._replace_parameters(y, etas, parameters)

        if dataset is not None:
            df = dataset
        else:
            df = self.dataset

        def fn(row):
            row = row.to_dict()
            a = [np.float64(x.subs(row)) for x in y]
            return a

        derivative_names = [f'dF/d{eta.name}' for eta in self.random_variables.etas]
        grad = df.apply(fn, axis=1, result_type='expand')
        grad = pd.DataFrame(grad)
        grad.columns = derivative_names
        return grad

    def eps_gradient(self, etas=None, parameters=None, dataset=None):
        """Numeric epsilon gradient
        """
        y = self.symbolic_eps_gradient()
        y = self._replace_parameters(y, etas, parameters)
        eps_names = [eps.name for eps in self.random_variables.ruv_rvs]

        repl = {sympy.Symbol(eps): 0 for eps in eps_names}
        y = [x.subs(repl) for x in y]

        if dataset is not None:
            df = dataset
        else:
            df = self.dataset

        def fn(row):
            row = row.to_dict()
            a = [np.float64(x.subs(row)) for x in y]
            return a

        grad = df.apply(fn, axis=1, result_type='expand')
        derivative_names = [f'dY/d{eps}' for eps in eps_names]
        grad = pd.DataFrame(grad)
        grad.columns = derivative_names
        return grad
