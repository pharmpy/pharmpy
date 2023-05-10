from __future__ import annotations

from abc import abstractmethod
from collections.abc import Collection, Hashable, Sized
from math import sqrt
from typing import Dict, List, Set, Tuple

import pharmpy.internals.unicode as unicode
from pharmpy.deps import numpy as np
from pharmpy.deps import symengine, sympy
from pharmpy.internals.expr.parse import parse as parse_expr
from pharmpy.internals.expr.subs import subs
from pharmpy.internals.immutable import Immutable

from .numeric import MultivariateNormalDistribution as NumericMultivariateNormalDistribution
from .numeric import NormalDistribution as NumericNormalDistribution
from .numeric import NumericDistribution


class Distribution(Sized, Hashable, Immutable):
    @abstractmethod
    def replace(self, **kwargs):
        pass

    @property
    @abstractmethod
    def names(self) -> Tuple[str, ...]:
        """Names of random variables of distribution"""
        pass

    @property
    @abstractmethod
    def level(self) -> str:
        """Name of VariabilityLevel of the random variables"""
        pass

    @property
    @abstractmethod
    def mean(self) -> sympy.Expr:
        pass

    @property
    @abstractmethod
    def variance(self) -> sympy.Expr:
        pass

    @abstractmethod
    def get_variance(self, name: str) -> sympy.Expr:
        pass

    @abstractmethod
    def get_covariance(self, name1: str, name2: str) -> sympy.Expr:
        pass

    @abstractmethod
    def evalf(self, parameters: Dict[sympy.Symbol, float]) -> NumericDistribution:
        pass

    @abstractmethod
    def __getitem__(self, index) -> Distribution:
        pass

    @property
    @abstractmethod
    def free_symbols(self) -> Set[sympy.Symbol]:
        pass

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        """List of names of all parameters used in definition"""
        params = self.mean.free_symbols.union(self.variance.free_symbols)
        return tuple(sorted(map(str, params)))

    @abstractmethod
    def subs(self, d: Dict[sympy.Expr, sympy.Expr]) -> Distribution:
        pass

    @abstractmethod
    def latex_string(self, aligned: bool = False) -> str:
        pass

    def _repr_latex_(self) -> str:
        return self.latex_string()


class NormalDistribution(Distribution):
    """Normal distribution for one random variable

    Parameters
    ----------
    name : str
        Name of the random variable
    level : str
        Name of the variability level
    mean : expression or number
        Mean of the random variable
    variance : expression or number
        Variance of the random variable

    Example
    -------
    >>> from pharmpy.model import NormalDistribution, Parameter
    >>> omega = Parameter.create('OMEGA_CL', 0.1)
    >>> dist = NormalDistribution.create("IIV_CL", "IIV", 0, omega.symbol)
    >>> dist
    IIV_CL ~ N(0, OMEGA_CL)
    """

    def __init__(self, name: str, level: str, mean: sympy.Expr, variance: sympy.Expr):
        self._name = name
        self._level = level
        self._mean = mean
        self._variance = variance

    @classmethod
    def create(cls, name, level, mean, variance):
        level = level.upper()
        mean = parse_expr(mean)
        variance = parse_expr(variance)
        if sympy.ask(sympy.Q.nonnegative(variance)) is False:
            raise ValueError("Variance of normal distribution must be non-negative")
        return cls(name, level, mean, variance)

    def replace(self, **kwargs):
        """Replace properties and create a new NormalDistribution"""
        name = kwargs.get('name', self._name)
        level = kwargs.get('level', self._level)
        mean = kwargs.get('mean', self._mean)
        variance = kwargs.get('variance', self._variance)
        new = NormalDistribution.create(name, level, mean, variance)
        return new

    @property
    def names(self):
        return (self._name,)

    @property
    def level(self):
        return self._level

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def free_symbols(self):
        """Free symbols including random variable itself"""
        fs = self._mean.free_symbols.union(self._variance.free_symbols)
        fs.add(sympy.Symbol(self._name))
        return fs

    def subs(self, d):
        """Substitute expressions

        Parameters
        ----------
        d : dict
            Dictionary of from: to pairs for substitution

        Examples
        --------
        >>> import sympy
        >>> from pharmpy.model import NormalDistribution, Parameter
        >>> omega = Parameter.create("OMEGA_CL", 0.1)
        >>> dist = NormalDistribution.create("IIV_CL", "IIV", 0, omega.symbol)
        >>> dist = dist.subs({omega.symbol: sympy.Symbol("OMEGA_NEW")})
        >>> dist
        IIV_CL ~ N(0, OMEGA_NEW)

        """
        mean = subs(self._mean, d)
        variance = subs(self._variance, d)
        name = _subs_name(self._name, d)
        return NormalDistribution(name, self._level, mean, variance)

    def evalf(self, parameters: Dict[sympy.Symbol, float]):
        # mu = float(symengine.sympify(rv._mean[0]).xreplace(parameters))
        # sigma = float(symengine.sympify(sympy.sqrt(rv._variance[0,0])).xreplace(parameters))
        mean = self.mean
        variance = self.variance
        try:
            mu = 0 if mean == 0 else float(parameters[mean])
            sigma = 0 if variance == 0 else sqrt(float(parameters[variance]))
            return NumericNormalDistribution(mu, sigma)
        except KeyError as e:
            # NOTE This handles missing parameter substitutions
            raise ValueError(e)

    def __getitem__(self, index):
        if isinstance(index, int):
            if index != 0:
                raise IndexError(index)

        elif isinstance(index, str):
            if index != self._name:
                raise KeyError(index)

        else:
            if isinstance(index, slice):
                index = list(
                    range(index.start, index.stop, index.step if index.step is not None else 1)
                )
                if index != [0]:
                    raise IndexError(index)

            if isinstance(index, Collection):
                if len(index) != 1 or (self._name not in index and 0 not in index):
                    raise KeyError(index)

            else:
                raise KeyError(index)

        return self

    def get_variance(self, name):
        if name != self._name:
            raise KeyError(name)
        return self._variance

    def get_covariance(self, name1, name2):
        if name1 == name2 == self._name:
            return self._variance
        else:
            raise KeyError((name1, name2))

    def __eq__(self, other):
        return (
            isinstance(other, NormalDistribution)
            and self._name == other._name
            and self._level == other._level
            and self._mean == other._mean
            and self._variance == other._variance
        )

    def __len__(self):
        return 1

    def __hash__(self):
        return hash((self._name, self._level, self._mean, self._variance))

    def to_dict(self):
        return {
            'class': self.__class__.__name__,
            'name': self._name,
            'level': self._level,
            'mean': sympy.srepr(self._mean),
            'variance': sympy.srepr(self._variance),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d['name'],
            level=d['level'],
            mean=sympy.parse_expr(d['mean']),
            variance=sympy.parse_expr(d['variance']),
        )

    def __repr__(self):
        return (
            f'{sympy.pretty(sympy.Symbol(self._name), wrap_line=False, use_unicode=True)}'
            f' ~ {unicode.mathematical_script_capital_n}'
            f'({sympy.pretty(self._mean, wrap_line=False, use_unicode=True)}, '
            f'{sympy.pretty(self._variance, wrap_line=False, use_unicode=True)})'
        )

    def latex_string(self, aligned=False):
        if aligned:
            align_str = ' & '
        else:
            align_str = ''
        rv = sympy.Symbol(self._name)._repr_latex_()[1:-1]
        mean = self._mean._repr_latex_()[1:-1]
        sigma = (self._variance)._repr_latex_()[1:-1]
        latex = rv + align_str + r'\sim  \mathcal{N} \left(' + mean + ',' + sigma + r'\right)'
        if not aligned:
            latex = '$' + latex + '$'
        return latex


class JointNormalDistribution(Distribution):
    """Joint distribution of random variables

    Parameters
    ----------
    names : list
        Names of the random variables
    level : str
        Variability level
    mean : matrix or list
        Vector of the means of the random variables
    variance : matrix or list of lists
        Covariance matrix of the random variables

    Example
    -------
    >>> from pharmpy.model import JointNormalDistribution, Parameter
    >>> omega_cl = Parameter("OMEGA_CL", 0.1)
    >>> omega_v = Parameter("OMEGA_V", 0.1)
    >>> corr_cl_v = Parameter("OMEGA_CL_V", 0.01)
    >>> dist = JointNormalDistribution.create(["IIV_CL", "IIV_V"], "IIV", [0, 0],
    ...     [[omega_cl.symbol, corr_cl_v.symbol], [corr_cl_v.symbol, omega_v.symbol]])
    >>> dist
    ⎡IIV_CL⎤    ⎧⎡0⎤  ⎡ OMEGA_CL   OMEGA_CL_V⎤⎫
    ⎢      ⎥ ~ N⎪⎢ ⎥, ⎢                      ⎥⎪
    ⎣IIV_V ⎦    ⎩⎣0⎦  ⎣OMEGA_CL_V   OMEGA_V  ⎦⎭

    """

    def __init__(
        self, names: Tuple[str, ...], level: str, mean: sympy.Matrix, variance: sympy.Matrix
    ):
        self._names = names
        self._level = level
        self._mean = mean
        self._variance = variance
        self._symengine_variance = symengine.Matrix(variance)

    @classmethod
    def create(cls, names, level, mean, variance):
        names = tuple(names)
        level = level.upper()
        mean = sympy.ImmutableMatrix(mean)
        variance = sympy.ImmutableMatrix(variance)
        if variance.is_positive_semidefinite is False:
            raise ValueError(
                'Covariance matrix of joint normal distribution is not positive semidefinite'
            )
        return cls(names, level, mean, variance)

    def replace(self, **kwargs):
        """Replace properties and create a new JointNormalDistribution"""
        names = kwargs.get('names', self._names)
        level = kwargs.get('level', self._level)
        mean = kwargs.get('mean', self._mean)
        variance = kwargs.get('variance', self._variance)
        new = JointNormalDistribution.create(names, level, mean, variance)
        return new

    @property
    def names(self):
        return self._names

    @property
    def level(self):
        return self._level

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def free_symbols(self):
        """Free symbols including random variable itself"""
        return self._mean.free_symbols.union(
            self._variance.free_symbols, (sympy.Symbol(name) for name in self._names)
        )

    def subs(self, d):
        """Substitute expressions

        Parameters
        ----------
        d : dict
            Dictionary of from: to pairs for substitution

        Examples
        --------
        >>> import sympy
        >>> from pharmpy.model import JointNormalDistribution, Parameter
        >>> omega1 = Parameter("OMEGA_CL", 0.1)
        >>> omega2 = Parameter("OMEGA_V", 0.1)
        >>> A = [[omega1.symbol, 0], [0, omega2.symbol]]
        >>> dist = JointNormalDistribution.create(["IIV_CL", "IIV_V"], "IIV", [0, 0], A)
        >>> dist = dist.subs({omega1.symbol: sympy.Symbol("OMEGA_NEW")})
        >>> dist
                ⎡IIV_CL⎤    ⎧⎡0⎤  ⎡OMEGA_NEW     0   ⎤⎫
                ⎢      ⎥ ~ N⎪⎢ ⎥, ⎢                  ⎥⎪
                ⎣IIV_V ⎦    ⎩⎣0⎦  ⎣    0      OMEGA_V⎦⎭

        """
        mean = self._mean.subs(d)
        variance = self._variance.subs(d)
        new_names = tuple(_subs_name(name, d) for name in self._names)
        return JointNormalDistribution(new_names, self._level, mean, variance)

    def evalf(self, parameters: Dict[sympy.Symbol, float]):
        try:
            mu = np.array(symengine.sympify(self.mean).xreplace(parameters)).astype(np.float64)[
                :, 0
            ]
            sigma = np.array(self._symengine_variance.xreplace(parameters)).astype(np.float64)
            return NumericMultivariateNormalDistribution(mu, sigma)
        except RuntimeError as e:
            # NOTE This handles missing parameter substitutions
            raise ValueError(e)

    def __getitem__(self, index):
        if isinstance(index, int):
            if -len(self) <= index < len(self):
                names = (self._names[index],)
            else:
                raise IndexError(index)

        elif isinstance(index, str):
            names = (index,)
            try:
                index = self._names.index(index)
            except ValueError:
                raise KeyError(names[0])

        else:
            if isinstance(index, slice):
                index = range(index.start, index.stop, index.step if index.step is not None else 1)
                index = [self._names[i] for i in index]

            if isinstance(index, Collection):
                if len(index) == 0 or len(index) > len(self._names):
                    raise KeyError(index)

                collection = set(index)
                if not collection.issubset(self._names):
                    raise KeyError(index)

                if len(collection) == len(self._names):
                    return self

                index_list: List[int] = []
                names_list: List[str] = []

                for i, name in enumerate(self._names):
                    if name in collection:
                        index_list.append(i)
                        names_list.append(name)

                index = tuple(index_list)
                names = tuple(names_list)

                if len(index) == 1:
                    index = index[0]

            else:
                raise KeyError(index)

        mean = self._mean[index, [0]] if isinstance(index, int) else self._mean[index, :]
        variance = self._variance[index, index]

        if len(names) == 1:
            return NormalDistribution(names[0], self._level, mean, variance)
        else:
            return JointNormalDistribution(names, self._level, mean, variance)

    def get_variance(self, name):
        i = self._names.index(name)
        return self._variance[i, i]

    def get_covariance(self, name1, name2):
        i1 = self._names.index(name1)
        i2 = self._names.index(name2)
        return self._variance[i1, i2]

    def __eq__(self, other):
        return (
            isinstance(other, JointNormalDistribution)
            and self._names == other._names
            and self._level == other._level
            and self._mean == other._mean
            and self._variance == other._variance
        )

    def __len__(self):
        return len(self._names)

    def __hash__(self):
        return hash((self._names, self._level, self._mean, self._variance))

    def to_dict(self):
        return {
            'class': self.__class__.__name__,
            'names': self._names,
            'level': self._level,
            'mean': sympy.srepr(self._mean),
            'variance': sympy.srepr(self._variance),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            names=d['names'],
            level=d['level'],
            mean=sympy.parse_expr(d['mean']),
            variance=sympy.parse_expr(d['variance']),
        )

    def __repr__(self):
        name_vector = sympy.Matrix(self._names)
        name_strings = sympy.pretty(name_vector, wrap_line=False, use_unicode=True).split('\n')
        mu_strings = sympy.pretty(self._mean, wrap_line=False, use_unicode=True).split('\n')
        sigma_strings = sympy.pretty(self._variance, wrap_line=False, use_unicode=True).split('\n')
        mu_height = len(mu_strings)
        sigma_height = len(sigma_strings)
        max_height = max(mu_height, sigma_height)

        left_parens = unicode.left_parens(max_height)
        right_parens = unicode.right_parens(max_height)

        # Pad the smaller of the matrices
        if mu_height != sigma_height:
            to_pad = mu_strings if mu_height < sigma_height else sigma_strings
            num_lines = abs(mu_height - sigma_height)
            padding = ' ' * len(to_pad[0])
            for i in range(0, num_lines):
                if i % 2 == 0:
                    to_pad.append(padding)
                else:
                    to_pad.insert(0, padding)

        # Pad names
        if len(name_strings) < max_height:
            num_lines = abs(max_height - len(name_strings))
            padding = ' ' * len(name_strings[0])
            for i in range(0, num_lines):
                if i % 2 == 0:
                    name_strings.append(padding)
                else:
                    name_strings.insert(0, padding)

        central_index = max_height // 2
        res = []
        enumerator = enumerate(
            zip(name_strings, left_parens, mu_strings, sigma_strings, right_parens)
        )
        for i, (name_line, lpar, mu_line, sigma_line, rpar) in enumerator:
            if i == central_index:
                res.append(
                    name_line
                    + f' ~ {unicode.mathematical_script_capital_n}'
                    + lpar
                    + mu_line
                    + ', '
                    + sigma_line
                    + rpar
                )
            else:
                res.append(name_line + '    ' + lpar + mu_line + '  ' + sigma_line + rpar)
        return '\n'.join(res)

    def latex_string(self, aligned=False):
        if aligned:
            align_str = ' & '
        else:
            align_str = ''
        names = [sympy.Symbol(name) for name in self._names]
        rv_vec = sympy.Matrix(names)._repr_latex_()[1:-1]
        mean_vec = self._mean._repr_latex_()[1:-1]
        sigma = self._variance._repr_latex_()[1:-1]
        latex = (
            rv_vec + align_str + r'\sim \mathcal{N} \left(' + mean_vec + ',' + sigma + r'\right)'
        )
        if not aligned:
            latex = '$' + latex + '$'
        return latex

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_symengine_variance']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._symengine_variance = symengine.sympify(self._variance)


def _subs_name(name: str, d: Dict[sympy.Expr, sympy.Expr]) -> str:
    if name in d:
        new_name = d[name]
    elif (name_symbol := sympy.Symbol(name)) in d:
        new_name = d[name_symbol]
    else:
        new_name = name
    return new_name if isinstance(new_name, str) else str(new_name)
