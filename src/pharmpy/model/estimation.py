from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union, overload

from pharmpy.internals.immutable import Immutable, frozenmapping

if TYPE_CHECKING:
    import pandas as pd
else:
    from pharmpy.deps import pandas as pd


SUPPORTED_SOLVERS = frozenset(('CVODES', 'DGEAR', 'DVERK', 'IDA', 'LSODA', 'LSODI'))


class Step(Immutable):
    def __init__(
        self,
        solver: Optional[str] = None,
        solver_rtol: Optional[int] = None,
        solver_atol: Optional[int] = None,
        tool_options: Optional[frozenmapping[str, Any]] = None,
    ):
        self._solver = solver
        self._solver_rtol = solver_rtol
        self._solver_atol = solver_atol
        self._tool_options = tool_options

    @staticmethod
    def _canonicalize_solver(solver):
        if solver is not None:
            solver = solver.upper()
        if not (solver is None or solver in SUPPORTED_SOLVERS):
            raise ValueError(
                f"Unknown solver {solver}. Recognized solvers are {sorted(SUPPORTED_SOLVERS)}."
            )
        return solver

    @staticmethod
    def _canonicalize_tool_options(tool_options):
        if tool_options is None:
            tool_options = frozenmapping({})
        else:
            tool_options = frozenmapping(tool_options)
        return tool_options

    @property
    def solver(self) -> Union[str, None]:
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
    def solver_rtol(self) -> Union[int, None]:
        """Relative tolerance for numerical ODE system solver"""
        return self._solver_rtol

    @property
    def solver_atol(self) -> Union[int, None]:
        """Absolute tolerance for numerical ODE system solver"""
        return self._solver_atol

    @property
    def tool_options(self) -> Union[frozenmapping[str, Any], None]:
        """Dictionary of tool specific options"""
        return self._tool_options

    def _add_to_dict(self, d):
        if self._tool_options is not None:
            tool_options = dict(self._tool_options)
        else:
            tool_options = self._tool_options  # self._tool_options is None
        d['solver'] = self._solver
        d['solver_rtol'] = self._solver_rtol
        d['solver_atol'] = self._solver_atol
        d['tool_options'] = tool_options

    @staticmethod
    def _adjust_dict(d):
        del d['class']
        if isinstance(d['tool_options'], dict):
            d['tool_options'] = frozenmapping(d['tool_options'])

    def _partial_repr(self):
        solver = f"'{self.solver}'" if self.solver is not None else self.solver
        return (
            f"solver={solver}, "
            f"solver_rtol={self.solver_rtol}, solver_atol={self.solver_atol}, "
            f"tool_options={self.tool_options}"
        )

    def __eq__(self, other):
        # NOTE: No need to test class here
        return (
            self.solver == other.solver
            and self.solver_rtol == other.solver_rtol
            and self.solver_atol == other.solver_atol
            and self.tool_options == other.tool_options
        )

    def __hash__(self):
        return hash(
            (
                self._solver,
                self._solver_rtol,
                self._solver_atol,
                self._tool_options,
            )
        )


class EstimationStep(Step):
    """Definition of one estimation operation"""

    """Supported estimation methods
    """
    supported_methods = frozenset(('FO', 'FOCE', 'ITS', 'IMPMAP', 'IMP', 'SAEM', 'BAYES'))
    supported_parameter_uncertainty_methods = frozenset(('SANDWICH', 'CPG', 'OFIM', 'EFIM'))

    def __init__(
        self,
        method: str,
        interaction: bool = False,
        parameter_uncertainty_method: Optional[str] = None,
        evaluation: bool = False,
        maximum_evaluations: Optional[int] = None,
        laplace: bool = False,
        isample: Optional[int] = None,
        niter: Optional[int] = None,
        auto: Optional[bool] = None,
        keep_every_nth_iter: Optional[int] = None,
        residuals: Optional[tuple[str, ...]] = None,
        predictions: Optional[tuple[str, ...]] = None,
        solver: Optional[str] = None,
        solver_rtol: Optional[int] = None,
        solver_atol: Optional[int] = None,
        tool_options: Optional[frozenmapping[str, Any]] = None,
        eta_derivatives: Optional[tuple[str, ...]] = None,
        epsilon_derivatives: Optional[tuple[str, ...]] = None,
    ):
        self._method = method
        self._interaction = interaction
        self._parameter_uncertainty_method = parameter_uncertainty_method
        self._evaluation = evaluation
        self._maximum_evaluations = maximum_evaluations
        self._laplace = laplace
        self._isample = isample
        self._niter = niter
        self._auto = auto
        self._keep_every_nth_iter = keep_every_nth_iter
        self._residuals = residuals
        self._predictions = predictions
        self._eta_derivatives = eta_derivatives
        self._epsilon_derivatives = epsilon_derivatives
        super().__init__(
            solver=solver,
            solver_rtol=solver_rtol,
            solver_atol=solver_atol,
            tool_options=tool_options,
        )

    @classmethod
    def create(
        cls,
        method: str,
        interaction: bool = False,
        parameter_uncertainty_method: Optional[str] = None,
        evaluation: bool = False,
        maximum_evaluations: Optional[int] = None,
        laplace: bool = False,
        isample: Optional[int] = None,
        niter: Optional[int] = None,
        auto: Optional[bool] = None,
        keep_every_nth_iter: Optional[int] = None,
        residuals: Optional[Sequence[str]] = None,
        predictions: Optional[Sequence[str]] = None,
        solver: Optional[str] = None,
        solver_rtol: Optional[int] = None,
        solver_atol: Optional[int] = None,
        tool_options: Optional[Mapping[str, Any]] = None,
        eta_derivatives: Optional[Sequence[str]] = None,
        epsilon_derivatives: Optional[Sequence[str]] = None,
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
        if parameter_uncertainty_method is not None:
            parameter_uncertainty_method = parameter_uncertainty_method.upper()
        if not (
            parameter_uncertainty_method is None
            or parameter_uncertainty_method
            in EstimationStep.supported_parameter_uncertainty_methods
        ):
            raise ValueError(
                f"Unknown parameter uncertainty method {parameter_uncertainty_method}. "
                f"Recognized methods are {sorted(EstimationStep.supported_parameter_uncertainty_methods)}."
            )
        solver = Step._canonicalize_solver(solver)
        tool_options = Step._canonicalize_tool_options(tool_options)
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
            parameter_uncertainty_method=parameter_uncertainty_method,
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

    def replace(self, **kwargs) -> EstimationStep:
        """Derive a new EstimationStep with new properties"""
        d = {key[1:]: value for key, value in self.__dict__.items()}
        d.update(kwargs)
        new = EstimationStep.create(**d)
        return new

    @staticmethod
    def _canonicalize_and_check_method(method: str) -> str:
        method = method.upper()
        if method not in EstimationStep.supported_methods:
            raise ValueError(
                f'EstimationStep: {method} not recognized. Use any of {sorted(EstimationStep.supported_methods)}.'
            )
        return method

    @property
    def method(self) -> str:
        """Name of the estimation method"""
        return self._method

    @property
    def maximum_evaluations(self) -> Union[int, None]:
        """Maximum allowable number of evaluations of the objective function"""
        return self._maximum_evaluations

    @property
    def interaction(self) -> bool:
        """Preserve eta-epsilon interaction in the computation of the objective function"""
        return self._interaction

    @property
    def evaluation(self) -> bool:
        """Only perform model evaluation"""
        return self._evaluation

    @property
    def parameter_uncertainty_method(self) -> Union[str, None]:
        """Method to use when estimating parameter uncertainty.
        Supported methods and their corresponding NMTRAN code:

        +----------------------------+-----------------------+
        | Method                     | NMTRAN                |
        +============================+=======================+
        | Sandwich                   | $COVARIANCE           |
        +----------------------------+-----------------------+
        | Cross-product gradient     | $COVARIANCE MATRIX=S  |
        +----------------------------+-----------------------+
        | Observed FIM               | $COVARIANCE MATRIX=R  |
        +----------------------------+-----------------------+
        | Expected FIM               | $DESIGN               |
        +----------------------------+-----------------------+

        By default the following options are appended:
        UNCONDITIONAL: The uncertainty step is implemented regardless of how the estimation step terminates.
        PRINT=E: Print the eigenvalues of the correlation matrix.
        PRECOND=1: Perform up to 1 preconditioning cycle on the R matrix.
        """
        return self._parameter_uncertainty_method

    @property
    def laplace(self) -> bool:
        """Use the laplacian method"""
        return self._laplace

    @property
    def isample(self) -> Union[int, None]:
        """Number of samples per subject (or similar) for EM methods"""
        return self._isample

    @property
    def niter(self) -> Union[int, None]:
        """Number of iterations for EM methods"""
        return self._niter

    @property
    def auto(self) -> Union[bool, None]:
        """Let estimation tool automatically add settings"""
        return self._auto

    @property
    def keep_every_nth_iter(self) -> Union[int, None]:
        """Keep results for every nth iteration"""
        return self._keep_every_nth_iter

    @property
    def residuals(self) -> Union[tuple[str, ...], None]:
        """List of residuals to calculate"""
        return self._residuals

    @property
    def predictions(self) -> Union[tuple[str, ...], None]:
        """List of predictions to estimate"""
        return self._predictions

    @property
    def eta_derivatives(self) -> Union[tuple[str, ...], None]:
        """List of names of etas for which to calculate derivatives"""
        return self._eta_derivatives

    @property
    def epsilon_derivatives(self) -> Union[tuple[str, ...], None]:
        """List of names of epsilons for which to calculate derivatives"""
        return self._epsilon_derivatives

    @property
    def tool_options(self) -> Union[frozenmapping[str, Any], None]:
        """Dictionary of tool specific options"""
        return self._tool_options

    def __eq__(self, other):
        return (
            isinstance(other, EstimationStep)
            and self.method == other.method
            and self.interaction == other.interaction
            and self.parameter_uncertainty_method == other.parameter_uncertainty_method
            and self.evaluation == other.evaluation
            and self.maximum_evaluations == other.maximum_evaluations
            and self.laplace == other.laplace
            and self.isample == other.isample
            and self.niter == other.niter
            and self.auto == other.auto
            and self.keep_every_nth_iter == other.keep_every_nth_iter
            and super().__eq__(other)
        )

    def __hash__(self):
        return hash(
            (
                self._method,
                self._interaction,
                self._parameter_uncertainty_method,
                self._evaluation,
                self._maximum_evaluations,
                self._laplace,
                self._isample,
                self._niter,
                self._auto,
                self._keep_every_nth_iter,
                super().__hash__(),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        d = {
            'class': 'EstimationStep',
            'method': self._method,
            'interaction': self._interaction,
            'parameter_uncertainty_method': self._parameter_uncertainty_method,
            'evaluation': self._evaluation,
            'maximum_evaluations': self._maximum_evaluations,
            'laplace': self._laplace,
            'isample': self._isample,
            'niter': self._niter,
            'auto': self._auto,
            'keep_every_nth_iter': self._keep_every_nth_iter,
        }
        super()._add_to_dict(d)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EstimationStep:
        d = dict(d)
        Step._adjust_dict(d)
        return cls(**d)

    def __repr__(self):
        parameter_uncertainty_method = (
            f"'{self.parameter_uncertainty_method}'"
            if self.parameter_uncertainty_method is not None
            else self.parameter_uncertainty_method
        )
        return (
            f"EstimationStep('{self.method}', interaction={self.interaction}, "
            f"parameter_uncertainty_method={parameter_uncertainty_method}, evaluation={self.evaluation}, "
            f"maximum_evaluations={self.maximum_evaluations}, laplace={self.laplace}, "
            f"isample={self.isample}, niter={self.niter}, auto={self.auto}, "
            f"keep_every_nth_iter={self.keep_every_nth_iter}, {super()._partial_repr()})"
        )


class SimulationStep(Step):
    """Definition of one simulation operation"""

    def __init__(
        self,
        n: int = 1,
        seed: int = 64206,
        solver: Optional[str] = None,
        solver_rtol: Optional[int] = None,
        solver_atol: Optional[int] = None,
        tool_options: Optional[frozenmapping[str, Any]] = None,
    ):
        self._n = n
        self._seed = seed
        super().__init__(
            solver=solver,
            solver_rtol=solver_rtol,
            solver_atol=solver_atol,
            tool_options=tool_options,
        )

    @classmethod
    def create(
        cls,
        n: int = 1,
        seed: int = 64206,
        solver: Optional[str] = None,
        solver_rtol: Optional[int] = None,
        solver_atol: Optional[int] = None,
        tool_options: Optional[Mapping[str, Any]] = None,
    ):
        if n < 1:
            raise ValueError("Need at least one replicate in SimulationStep")
        return cls(n=n, seed=seed)

    def replace(self, **kwargs) -> SimulationStep:
        """Derive a new SimulationStep with new properties"""
        d = {key[1:]: value for key, value in self.__dict__.items()}
        d.update(kwargs)
        new = SimulationStep.create(**d)
        return new

    @property
    def n(self) -> int:
        """Number of simulation replicates"""
        return self._n

    @property
    def seed(self) -> int:
        """Random seed"""
        return self._seed

    def __eq__(self, other):
        return (
            isinstance(other, SimulationStep)
            and self.n == other.n
            and self.seed == other.seed
            and super().__eq__(other)
        )

    def __hash__(self):
        return hash((self._n, self._seed, super().__hash__()))

    def to_dict(self) -> dict[str, Any]:
        d = {
            'class': 'SimulationStep',
            'n': self._n,
            'seed': self._seed,
        }
        super()._add_to_dict(d)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimulationStep:
        d = dict(d)
        Step._adjust_dict(d)
        return cls(**d)

    def __repr__(self):
        return f"SimulationStep(n={self._n}, seed={self._seed}, {super()._partial_repr()})"


class EstimationSteps(Sequence, Immutable):
    """A sequence of estimation steps

    Parameters
    ----------
    steps : iterable or None
        Used for initialization
    """

    def __init__(self, steps: tuple[Union[EstimationStep, SimulationStep], ...] = ()):
        self._steps = steps

    @classmethod
    def create(cls, steps: Optional[Sequence[Union[EstimationStep, SimulationStep]]] = None):
        if steps is None:
            steps = ()
        else:
            steps = tuple(steps)
        return EstimationSteps(steps=steps)

    def replace(self, **kwargs) -> EstimationSteps:
        steps = kwargs.get('steps', self._steps)
        return EstimationSteps.create(steps)

    @overload
    def __getitem__(self, i: int) -> EstimationStep:
        ...

    @overload
    def __getitem__(self, i: slice) -> EstimationSteps:
        ...

    def __getitem__(
        self, i: Union[int, slice]
    ) -> Union[EstimationStep, SimulationStep, EstimationSteps]:
        if isinstance(i, slice):
            return EstimationSteps(self._steps[i])
        return self._steps[i]

    def __add__(self, other) -> EstimationSteps:
        if isinstance(other, EstimationSteps):
            return EstimationSteps(self._steps + other._steps)
        elif isinstance(other, EstimationStep) or isinstance(other, SimulationStep):
            return EstimationSteps(self._steps + (other,))
        else:
            return EstimationSteps(self._steps + tuple(other))

    def __radd__(self, other) -> EstimationSteps:
        if isinstance(other, EstimationStep) or isinstance(other, SimulationStep):
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

    def to_dict(self) -> dict[str, Any]:
        return {'steps': tuple(step.to_dict() for step in self._steps)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EstimationSteps:
        steps = []
        for sdict in d['steps']:
            if sdict['class'] == 'EstimationStep':
                s = EstimationStep.from_dict(sdict)
            else:
                s = SimulationStep.from_dict(sdict)
            steps.append(s)
        return cls(steps=tuple(steps))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame

        Use this to create an overview of all estimation steps

        Returns
        -------
        pd.DataFrame
            DataFrame overview

        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.estimation_steps.to_dataframe()   # doctest: +ELLIPSIS
        method  interaction       parameter_uncertainty_method  ...  auto keep_every_nth_iter  tool_options
        0   FOCE         True  SANDWICH  ...  None                None            {}
        """
        steps = [s for s in self._steps if isinstance(s, EstimationStep)]
        method = [s.method for s in steps]
        interaction = [s.interaction for s in steps]
        parameter_uncertainty_method = [s.parameter_uncertainty_method for s in steps]
        evaluation = [s.evaluation for s in steps]
        maximum_evaluations = [s.maximum_evaluations for s in steps]
        laplace = [s.laplace for s in steps]
        isample = [s.isample for s in steps]
        niter = [s.niter for s in steps]
        auto = [s.auto for s in steps]
        keep_every_nth_iter = [s.keep_every_nth_iter for s in steps]
        tool_options = [dict(s.tool_options) if s.tool_options else dict() for s in steps]
        df = pd.DataFrame(
            {
                'method': method,
                'interaction': interaction,
                'parameter_uncertainty_method': parameter_uncertainty_method,
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
