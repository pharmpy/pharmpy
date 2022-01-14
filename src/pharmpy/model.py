"""
===================
Generic Model class
===================

**Base class of all implementations.**

Inherit to *implement*, i.e. to define support for a specific model type.

Definitions
-----------
"""

import copy
import io
import pathlib
from pathlib import Path

import sympy
from model.estimation import EstimationSteps

from pharmpy.datainfo import ColumnInfo, DataInfo
from pharmpy.parameter import Parameters
from pharmpy.plugins.utils import detect_model
from pharmpy.random_variables import RandomVariables
from pharmpy.statements import ModelStatements
from pharmpy.workflows import default_model_database


class ModelError(Exception):
    """Exception for errors in model object"""

    pass


class ModelSyntaxError(ModelError):
    """Exception for Syntax errors in model code"""

    def __init__(self, msg='model syntax error'):
        super().__init__(msg)


class Model:
    """Model"""

    def __init__(self):
        self.parameters = Parameters([])
        self.random_variables = RandomVariables([])
        self.statements = ModelStatements([])
        self.dependent_variable = sympy.Symbol('y')
        self.observation_transformation = self.dependent_variable
        self.modelfit_results = None

    def __repr__(self):
        return f'<Pharmpy model object {self.name}>'

    def _repr_html_(self):
        stat = self.statements._repr_html_()
        rvs = self.random_variables._repr_latex_()
        return f'<hr>{stat}<hr>${rvs}$<hr>{self.parameters._repr_html_()}<hr>'

    @property
    def name(self):
        """Name of the model"""
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError("Name of a model has to be of string type")
        self._name = value

    @property
    def dependent_variable(self):
        """The model dependent variable, i.e. y"""
        return self._dependent_variable

    @dependent_variable.setter
    def dependent_variable(self, value):
        self._dependent_variable = value

    @property
    def observation_transformation(self):
        """Transformation to be applied to the observation data"""
        return self._observation_transformation

    @observation_transformation.setter
    def observation_transformation(self, value):
        self._observation_transformation = value

    @property
    def parameters(self):
        """Definitions of population parameters

        See :class:`pharmpy.Parameters`
        """
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        if not isinstance(value, Parameters):
            raise TypeError("model.parameters must be of Parameters type")
        self._parameters = value

    @property
    def random_variables(self):
        """Definitions of random variables

        See :class:`pharmpy.RandomVariables`
        """
        return self._random_variables

    @random_variables.setter
    def random_variables(self, value):
        if not isinstance(value, RandomVariables):
            raise TypeError("model.random_variables must be of RandomVariables type")
        self._random_variables = value

    @property
    def statements(self):
        """Definitions of model statements

        See :class:`pharmpy.ModelStatements`
        """
        return self._statements

    @statements.setter
    def statements(self, value):
        if not isinstance(value, ModelStatements):
            raise TypeError("model.statements must be of ModelStatements type")
        self._statements = value

    @property
    def estimation_steps(self):
        """Definitions of estimation steps

        See :class:`pharmpy.EstimationSteps`
        """
        return self._estimation_steps

    @estimation_steps.setter
    def estimation_steps(self, value):
        if not isinstance(value, EstimationSteps):
            raise TypeError("model.estimation_steps must be of EstimationSteps type")
        self._estimation_steps = value

    @property
    def datainfo(self):
        """Definitions of model statements

        See :class:`pharmpy.ModelStatements`
        """
        return self._datainfo

    @datainfo.setter
    def datainfo(self, value):
        if not isinstance(value, DataInfo):
            raise TypeError("model.datainfo must be of DataInfo type")
        self._datainfo = value

    @property
    def dataset(self):
        """Dataset connected to model"""
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value
        self.update_datainfo()

    @property
    def initial_individual_estimates(self):
        """Initial estimates for individual parameters"""
        return self._initial_individual_estimates

    @initial_individual_estimates.setter
    def initial_individual_estimates(self, value):
        self._initial_individual_estimates = value

    @property
    def modelfit_results(self):
        """Modelfit results for this model"""
        return self._modelfit_results

    @modelfit_results.setter
    def modelfit_results(self, value):
        self._modelfit_results = value

    @property
    def model_code(self):
        """Model type specific code"""
        raise NotImplementedError("Generic model does not implement the model_code property")

    def read_modelfit_results(self):
        """Read in modelfit results"""
        raise NotImplementedError("Read modelfit results not implemented for generic models")

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
                filename = f'{self.name}{self.filename_extension}'
            except AttributeError:
                raise ValueError(
                    'Cannot name model file as no path argument was supplied and the'
                    'model has no name.'
                )
            path = path / filename
            new_name = None
        else:
            # Set new name given filename, but after we've checked for existence
            new_name = path.stem
        if not force and path.exists():
            raise FileExistsError(f'File {path} already exists.')
        if new_name:
            self.name = new_name
        self.update_source(path=path, force=force)
        if not force and path.exists():
            raise FileExistsError(f'Cannot overwrite model at {path} with "force" not set')
        with open(path, 'w', encoding='latin-1') as fp:
            fp.write(self.model_code)
        self.database = default_model_database(path=path.parent)
        return path

    def update_datainfo(self):
        """Update model.datainfo for a new dataset"""
        colnames = self.dataset.columns
        try:
            curdi = self.datainfo
        except AttributeError:
            curdi = DataInfo()
        newdi = DataInfo()
        for colname in colnames:
            try:
                col = curdi[colname]
            except IndexError:
                col = ColumnInfo(colname)
            newdi.append(col)
        if curdi != newdi:
            # Remove path if dataset has been updated
            newdi.path = None
        self.datainfo = newdi

    def copy(self):
        """Create a deepcopy of the model object"""
        return copy.deepcopy(self)

    def update_individual_estimates(self, source):
        self.initial_individual_estimates = source.modelfit_results.individual_estimates

    def read_raw_dataset(self, parse_columns=tuple()):
        raise NotImplementedError()

    @staticmethod
    def create_model(obj=None, **kwargs):
        """Factory for creating a :class:`pharmpy.model` object from an object representing the model
        (i.e. path).

        .. _path-like object: https://docs.python.org/3/glossary.html#term-path-like-object

        Parameters
        ----------
        obj
            Currently a `path-like object`_ pointing to the model file.

        Returns
        -------
        - Generic :class:`~pharmpy.generic.Model` if path is None, otherwise appropriate
          implementation is invoked (e.g. NONMEM7 :class:`~pharmpy.api_nonmem.model.Model`).
        """
        if isinstance(obj, str):
            path = Path(obj)
        elif isinstance(obj, pathlib.Path):  # DO NOT change to Path! Will fail fakefs tests
            path = obj
        elif isinstance(obj, io.IOBase):
            path = None
        elif obj is None:
            return Model()
        else:
            raise ValueError("Unknown input type to Model constructor")
        if path is not None:
            with open(path, 'r', encoding='latin-1') as fp:
                code = fp.read()
        else:
            code = obj.read()
        model_class = detect_model(code)
        model = model_class(code, path, **kwargs)
        # Setup model database here
        # Read in model results here?
        # Set filename extension?
        return model
