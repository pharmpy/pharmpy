from __future__ import annotations

from collections.abc import Sequence
from typing import overload

from pharmpy.deps import pandas as pd
from pharmpy.internals.immutable import Immutable, frozenmapping


class EstimationStep(Immutable):
    """Definition of one estimation operation"""

    """Supported estimation methods
    """
    supported_methods = frozenset(('FO', 'FOCE', 'ITS', 'IMPMAP', 'IMP', 'SAEM', 'BAYES'))
    supported_solvers = frozenset(('CVODES', 'DGEAR', 'DVERK', 'IDA', 'LSODA', 'LSODI'))
    supported_covs = frozenset(('SANDWICH', 'CPG', 'OFIM'))

    def __init__(
        self,
        method,
        interaction=False,
        cov=None,
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
        eta_derivatives=None,
        epsilon_derivatives=None,
    ):
        self._method = method
        self._interaction = interaction
        self._cov = cov
        self._evaluation = evaluation
        self._maximum_evaluations = maximum_evaluations
        self._laplace = laplace
        self._isample = isample
        self._niter = niter
        self._auto = auto
        self._keep_every_nth_iter = keep_every_nth_iter
        self._residuals = residuals
        self._predictions = predictions
        self._solver = solver
        self._solver_rtol = solver_rtol
        self._solver_atol = solver_atol
        self._tool_options = tool_options
        self._eta_derivatives = eta_derivatives
        self._epsilon_derivatives = epsilon_derivatives

    @classmethod
    def create(
        cls,
        method,
        interaction=False,
        cov=None,
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
        eta_derivatives=None,
        epsilon_derivatives=None,
    ):
        method = EstimationStep._canonicalize_and_check_method(method)
        if maximum_evaluations is not None and maximum_evaluations < 1:
            raise ValueError(
                'Number of maximum evaluations must be more than one, use '
                'evaluation=True or tool_options for special cases (e.g. 0 and -1'
                'in NONMEM)'
            )
        if residuals is None:
            residuals = ()
        else:
            residuals = tuple(residuals)
        if predictions is None:
            predictions = ()
        else:
            predictions = tuple(predictions)
        if cov is not None:
            cov = cov.upper()
        if not (cov is None or cov in EstimationStep.supported_covs):
            raise ValueError(
                f"Unknown cov {cov}. Recognized covs are {sorted(EstimationStep.supported_covs)}."
            )
        if solver is not None:
            solver = solver.upper()
        if not (solver is None or solver in EstimationStep.supported_solvers):
            raise ValueError(
                f"Unknown solver {solver}. Recognized solvers are {sorted(EstimationStep.supported_solvers)}."
            )
        if tool_options is None:
            tool_options = frozenmapping({})
        else:
            tool_options = frozenmapping(tool_options)
        if eta_derivatives is None:
            eta_derivatives = ()
        else:
            eta_derivatives = tuple(eta_derivatives)
        if epsilon_derivatives is None:
            epsilon_derivatives = ()
        else:
            epsilon_derivatives = tuple(epsilon_derivatives)
        return cls(
            method=method,
            interaction=interaction,
            cov=cov,
            evaluation=evaluation,
            maximum_evaluations=maximum_evaluations,
            laplace=laplace,
            isample=isample,
            niter=niter,
            auto=auto,
            keep_every_nth_iter=keep_every_nth_iter,
            residuals=residuals,
            predictions=predictions,
            solver=solver,
            solver_rtol=solver_rtol,
            solver_atol=solver_atol,
            tool_options=tool_options,
            eta_derivatives=eta_derivatives,
            epsilon_derivatives=epsilon_derivatives,
        )

    def replace(self, **kwargs):
        """Derive a new EstimationStep with new properties"""
        d = {key[1:]: value for key, value in self.__dict__.items()}
        d.update(kwargs)
        new = EstimationStep.create(**d)
        return new

    @staticmethod
    def _canonicalize_and_check_method(method):
        method = method.upper()
        if method not in EstimationStep.supported_methods:
            raise ValueError(
                f'EstimationStep: {method} not recognized. Use any of {sorted(EstimationStep.supported_methods)}.'
            )
        return method

    @property
    def method(self):
        """Name of the estimation method"""
        return self._method

    @property
    def maximum_evaluations(self):
        """Maximum allowable number of evaluations of the objective function"""
        return self._maximum_evaluations

    @property
    def interaction(self):
        """Preserve eta-epsilon interaction in the computation of the objective function"""
        return self._interaction

    @property
    def evaluation(self):
        """Only perform model evaluation"""
        return self._evaluation

    @property
    def cov(self):
        """Method to use when estimating parameter uncertainty
        Supported methods and their corresponding NMTRAN code:

        +----------------------------+------------------+
        | Method                     | NMTRAN           |
        +============================+==================+
        | Sandwich                   | $COV             |
        +----------------------------+------------------+
        | Cross-product gradient     | $COV MATRIX=S    |
        +----------------------------+------------------+
        | Observed FIM               | $COV MATRIX=R    |
        +----------------------------+------------------+
        """
        return self._cov

    @property
    def laplace(self):
        """Use the laplacian method"""
        return self._laplace

    @property
    def isample(self):
        """Number of samples per subject (or similar) for EM methods"""
        return self._isample

    @property
    def niter(self):
        """Number of iterations for EM methods"""
        return self._niter

    @property
    def auto(self):
        """Let estimation tool automatically add settings"""
        return self._auto

    @property
    def keep_every_nth_iter(self):
        """Keep results for every nth iteration"""
        return self._keep_every_nth_iter

    @property
    def residuals(self):
        """List of residuals to calculate"""
        return self._residuals

    @property
    def predictions(self):
        """List of predictions to estimate"""
        return self._predictions

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

    @property
    def solver_rtol(self):
        """Relative tolerance for numerical ODE system solver"""
        return self._solver_rtol

    @property
    def solver_atol(self):
        """Absolute tolerance for numerical ODE system solver"""
        return self._solver_atol

    @property
    def eta_derivatives(self):
        """List of names of etas for which to calculate derivatives"""
        return self._eta_derivatives

    @property
    def epsilon_derivatives(self):
        """List of names of epsilons for which to calculate derivatives"""
        return self._epsilon_derivatives

    @property
    def tool_options(self):
        """Dictionary of tool specific options"""
        return self._tool_options

    def __eq__(self, other):
        return (
            isinstance(other, EstimationStep)
            and self.method == other.method
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

    def __hash__(self):
        return hash(
            (
                self._method,
                self._interaction,
                self._cov,
                self._evaluation,
                self._maximum_evaluations,
                self._laplace,
                self._isample,
                self._niter,
                self._auto,
                self._keep_every_nth_iter,
                self._solver,
                self._solver_rtol,
                self._solver_atol,
                self._tool_options,
            )
        )

    def to_dict(self):
        return {
            'method': self._method,
            'interaction': self._interaction,
            'cov': self._cov,
            'evaluation': self._evaluation,
            'maximum_evaluations': self._maximum_evaluations,
            'laplace': self._laplace,
            'isample': self._isample,
            'niter': self._niter,
            'auto': self._auto,
            'keep_every_nth_iter': self._keep_every_nth_iter,
            'solver': self._solver,
            'solver_rtol': self._solver_rtol,
            'solver_atol': self._solver_atol,
            'tool_options': dict(self._tool_options),
        }

    @classmethod
    def from_dict(cls, d):
        d['tool_options'] = frozenmapping(d['tool_options'])
        return cls(**d)

    def __repr__(self):
        cov = f"'{self.cov}'" if self.cov is not None else self.cov
        solver = f"'{self.solver}'" if self.solver is not None else self.solver
        return (
            f"EstimationStep('{self.method}', interaction={self.interaction}, "
            f"cov={cov}, evaluation={self.evaluation}, "
            f"maximum_evaluations={self.maximum_evaluations}, laplace={self.laplace}, "
            f"isample={self.isample}, niter={self.niter}, auto={self.auto}, "
            f"keep_every_nth_iter={self.keep_every_nth_iter}, solver={solver}, "
            f"solver_rtol={self.solver_rtol}, solver_atol={self.solver_atol}, "
            f"tool_options={self.tool_options})"
        )


class EstimationSteps(Sequence, Immutable):
    """A sequence of estimation steps

    Parameters
    ----------
    steps : iterable or None
        Used for initialization
    """

    def __init__(self, steps=()):
        self._steps = steps

    @classmethod
    def create(cls, steps=None):
        if steps is None:
            steps = ()
        else:
            steps = tuple(steps)
        return EstimationSteps(steps=steps)

    def replace(self, **kwargs):
        steps = kwargs.get('steps', self._steps)
        return EstimationSteps.create(steps)

    @overload
    def __getitem__(self, i: int) -> EstimationStep:
        ...

    @overload
    def __getitem__(self, i: slice) -> EstimationSteps:
        ...

    def __getitem__(self, i):
        if isinstance(i, slice):
            return EstimationSteps(self._steps[i])
        return self._steps[i]

    def __add__(self, other):
        if isinstance(other, EstimationSteps):
            return EstimationSteps(self._steps + other._steps)
        elif isinstance(other, EstimationStep):
            return EstimationSteps(self._steps + (other,))
        else:
            return EstimationSteps(self._steps + tuple(other))

    def __radd__(self, other):
        if isinstance(other, EstimationStep):
            return EstimationSteps((other,) + self._steps)
        else:
            return EstimationSteps(tuple(other) + self._steps)

    def __len__(self):
        return len(self._steps)

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for s1, s2 in zip(self, other):
            if s1 != s2:
                return False
        return True

    def __hash__(self):
        return hash(self._steps)

    def to_dict(self):
        return {'steps': tuple(step.to_dict() for step in self._steps)}

    @classmethod
    def from_dict(cls, d):
        return cls(steps=tuple(EstimationStep.from_dict(s) for s in d['steps']))

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
          method  interaction       cov  ...  auto keep_every_nth_iter  tool_options
        0   FOCE         True  SANDWICH  ...  None                None            {}
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
        tool_options = [dict(s.tool_options) for s in self._steps]
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
