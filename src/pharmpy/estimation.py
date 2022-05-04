import copy
from collections.abc import MutableSequence

import pandas as pd


class EstimationSteps(MutableSequence):
    """A sequence of estimation steps

    Parameters
    ----------
    steps : EstimationSteps, iterable or None
        Used for initialization
    """

    def __init__(self, steps=None):
        if isinstance(steps, EstimationSteps):
            self._steps = copy.deepcopy(steps._steps)
        elif steps is None:
            self._steps = []
        else:
            self._steps = list(steps)

    def __getitem__(self, i):
        return self._steps[i]

    def __setitem__(self, i, value):
        self._steps[i] = value

    def __delitem__(self, i):
        del self._steps[i]

    def __len__(self):
        return len(self._steps)

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for s1, s2 in zip(self, other):
            if s1 != s2:
                return False
        return True

    def insert(self, i, value):
        self._steps.insert(i, value)

    def copy(self):
        """Create a deepcopy"""
        return copy.deepcopy(self)

    def to_dataframe(self):
        """Convert to DataFrame

        Use this to create an overview of all estimation steps

        Returns
        -------
        pd.DataFrame
            DataFrame overview

        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.estimation_steps.to_dataframe()   # doctest: +ELLIPSIS
          method  interaction   cov  ...  auto keep_every_nth_iter  tool_options
        0   FOCE         True  True  ...  None                None            {}
        """
        method = [s.method for s in self._steps]
        interaction = [s.interaction for s in self._steps]
        cov = [s.cov for s in self._steps]
        evaluation = [s.evaluation for s in self._steps]
        maximum_evaluations = [s.maximum_evaluations for s in self._steps]
        laplace = [s.laplace for s in self._steps]
        isample = [s.isample for s in self._steps]
        niter = [s.niter for s in self._steps]
        auto = [s.auto for s in self._steps]
        keep_every_nth_iter = [s.keep_every_nth_iter for s in self._steps]
        tool_options = [s.tool_options for s in self._steps]
        df = pd.DataFrame(
            {
                'method': method,
                'interaction': interaction,
                'cov': cov,
                'evaluation': evaluation,
                'maximum_evaluations': maximum_evaluations,
                'laplace': laplace,
                'isample': isample,
                'niter': niter,
                'auto': auto,
                'keep_every_nth_iter': keep_every_nth_iter,
                'tool_options': tool_options,
            }
        )
        return df

    def __repr__(self):
        if len(self) == 0:
            return "EstimationSteps()"
        return self.to_dataframe().to_string()

    def _repr_html_(self):
        if len(self) == 0:
            return "EstimationSteps()"
        else:
            return self.to_dataframe().to_html()


class EstimationStep:
    """Definition of one estimation operation"""

    """Supported estimation methods
    """
    supported_methods = ['FO', 'FOCE', 'ITS', 'IMPMAP', 'IMP', 'SAEM', 'BAYES']

    def __init__(
        self,
        method,
        interaction=False,
        cov=False,
        evaluation=False,
        maximum_evaluations=None,
        laplace=False,
        isample=None,
        niter=None,
        auto=None,
        keep_every_nth_iter=None,
        residuals=None,
        predictions=None,
        solver=None,
        solver_rtol=None,
        solver_atol=None,
        tool_options=None,
    ):
        method = self._canonicalize_and_check_method(method)
        self.method = method
        self.interaction = interaction
        self.cov = cov
        self.evaluation = evaluation
        self.maximum_evaluations = maximum_evaluations
        self.laplace = laplace
        self.isample = isample
        self.niter = niter
        self.auto = auto
        self.keep_every_nth_iter = keep_every_nth_iter
        if residuals is None:
            self.residuals = []
        else:
            self.residuals = residuals
        if predictions is None:
            self.predictions = []
        else:
            self.predictions = predictions
        self.solver = solver
        self.solver_rtol = solver_rtol
        self.solver_atol = solver_atol
        if tool_options is None:
            self.tool_options = dict()
        else:
            self.tool_options = tool_options

    def _canonicalize_and_check_method(self, method):
        method = method.upper()
        if method not in self.supported_methods:
            raise ValueError(
                f'EstimationStep: {method} not recognized. Use any of {self.supported_methods}.'
            )
        return method

    @property
    def method(self):
        """Name of the estimation method"""
        return self._method

    @method.setter
    def method(self, value):
        method = self._canonicalize_and_check_method(value)
        self._method = method

    @property
    def maximum_evaluations(self):
        """Maximum allowable number of evaluations of the objective function"""
        return self._maximum_evaluations

    @maximum_evaluations.setter
    def maximum_evaluations(self, value):
        if value is not None and value < 1:
            raise ValueError(
                'Number of maximum evaluations must be more than one, use '
                'evaluation=True or tool_options for special cases (e.g. 0 and -1'
                'in NONMEM)'
            )
        self._maximum_evaluations = value

    @property
    def interaction(self):
        """Preserve eta-epsilon interaction in the computation of the objective function"""
        return self._interaction

    @interaction.setter
    def interaction(self, value):
        self._interaction = bool(value)

    @property
    def evaluation(self):
        """Only perform model evaluation"""
        return self._evaluation

    @evaluation.setter
    def evaluation(self, value):
        self._evaluation = bool(value)

    @property
    def cov(self):
        """Should the parameter uncertainty be estimated?"""
        return self._cov

    @cov.setter
    def cov(self, value):
        self._cov = bool(value)

    @property
    def laplace(self):
        """Use the laplacian method"""
        return self._laplace

    @laplace.setter
    def laplace(self, value):
        self._laplace = bool(value)

    @property
    def isample(self):
        """Number of samples per subject (or similar) for EM methods"""
        return self._isample

    @isample.setter
    def isample(self, value):
        self._isample = value

    @property
    def niter(self):
        """Number of iterations for EM methods"""
        return self._niter

    @niter.setter
    def niter(self, value):
        self._niter = value

    @property
    def auto(self):
        """Let estimation tool automatically add settings"""
        return self._auto

    @auto.setter
    def auto(self, value):
        self._auto = value

    @property
    def keep_every_nth_iter(self):
        """Keep results for every nth iteration"""
        return self._keep_every_nth_iter

    @keep_every_nth_iter.setter
    def keep_every_nth_iter(self, value):
        self._keep_every_nth_iter = value

    @property
    def residuals(self):
        """List of residuals to calculate"""
        return self._residuals

    @residuals.setter
    def residuals(self, value):
        self._residuals = value

    @property
    def predictions(self):
        """List of predictions to estimate"""
        return self._predictions

    @predictions.setter
    def predictions(self, value):
        self._predictions = value

    @property
    def solver(self):
        """Numerical solver to use when numerically solving the ODE system
        Supported solvers and their corresponding NONMEM ADVAN

        +----------------------------+------------------+
        | Solver                     | NONMEM ADVAN     |
        +============================+==================+
        | CVODES                     | ADVAN14          |
        +----------------------------+------------------+
        | DGEAR                      | ADVAN8           |
        +----------------------------+------------------+
        | DVERK                      | ADVAN6           |
        +----------------------------+------------------+
        | IDA                        | ADVAN15          |
        +----------------------------+------------------+
        | LSODA                      | ADVAN13          |
        +----------------------------+------------------+
        | LSODI                      | ADVAN9           |
        +----------------------------+------------------+
        """
        return self._solver

    @solver.setter
    def solver(self, value):
        supported = ['CVODES', 'DGEAR', 'DVERK', 'IDA', 'LSODA', 'LSODI']
        if value is not None:
            value = value.upper()
        if not (value is None or value in supported):
            raise ValueError(f"Unknown solver {value}. Recognized solvers are {supported}.")
        self._solver = value

    @property
    def solver_rtol(self):
        """Relative tolerance for numerical ODE system solver"""
        return self._solver_rtol

    @solver_rtol.setter
    def solver_rtol(self, value):
        self._solver_rtol = value

    @property
    def solver_atol(self):
        """Absolute tolerance for numerical ODE system solver"""
        return self._solver_atol

    @solver_atol.setter
    def solver_atol(self, value):
        self._solver_atol = value

    @property
    def tool_options(self):
        """Dictionary of tool specific options"""
        return self._tool_options

    @tool_options.setter
    def tool_options(self, value):
        self._tool_options = value

    def __eq__(self, other):
        return (
            self.method == other.method
            and self.interaction == other.interaction
            and self.cov == other.cov
            and self.evaluation == other.evaluation
            and self.maximum_evaluations == other.maximum_evaluations
            and self.laplace == other.laplace
            and self.isample == other.isample
            and self.niter == other.niter
            and self.auto == other.auto
            and self.keep_every_nth_iter == other.keep_every_nth_iter
            and self.solver == other.solver
            and self.solver_rtol == other.solver_rtol
            and self.solver_atol == other.solver_atol
            and self.tool_options == other.tool_options
        )

    def __repr__(self):
        return (
            f'EstimationStep("{self.method}", interaction={self.interaction}, '
            f'cov={self.cov}, evaluation={self.evaluation}, '
            f'maximum_evaluations={self.maximum_evaluations}, laplace={self.laplace}, '
            f'isample={self.isample}, niter={self.niter}, auto={self.auto}, '
            f'keep_every_nth_iter={self.keep_every_nth_iter}, solver={self.solver}, '
            f'solver_rtol={self.solver_rtol}, solver_atol={self.solver_atol}, '
            f'tool_options={self.tool_options})'
        )

    def copy(self):
        """Create a deep copy"""
        return copy.deepcopy(self)
