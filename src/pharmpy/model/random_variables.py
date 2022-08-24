from __future__ import annotations

import itertools
from collections.abc import Sequence
from functools import lru_cache
from math import sqrt
from typing import Dict, Iterable, Set, Tuple

import pharmpy.math
import pharmpy.unicode as unicode
from pharmpy.deps import numpy as np
from pharmpy.deps import symengine, sympy
from pharmpy.expressions import subs, sympify, xreplace_dict

from .distributions.numeric import (
    MultivariateNormalDistribution as NumericMultivariateNormalDistribution,
)
from .distributions.numeric import NormalDistribution as NumericNormalDistribution
from .distributions.numeric import NumericDistribution


def _create_rng(seed=None):
    """Create a new random number generator"""
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)
    return rng


class Distribution:
    @property
    def names(self):
        """Names of random variables of distribution"""
        return self._names

    @property
    def level(self):
        """Name of VariabilityLevel of the random variables"""
        return self._level

    def __hash__(self):
        return hash(self._names)

    def __len__(self):
        return len(self._names)


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
    >>> omega = Parameter('OMEGA_CL', 0.1)
    >>> dist = NormalDistribution.create("IIV_CL", "IIV", 0, omega.symbol)
    >>> dist
    IIV_CL ~ N(0, OMEGA_CL)
    """

    def __init__(self, names, level, mean, variance):
        self._names = names
        self._level = level
        self._mean = mean
        self._variance = variance

    @classmethod
    def create(cls, name, level, mean, variance):
        name = (name,)
        level = level.upper()
        mean = sympify(mean)
        variance = sympify(variance)
        if sympy.ask(sympy.Q.nonnegative(variance)) is False:
            raise ValueError("Variance of normal distribution must be non-negative")
        return cls(name, level, mean, variance)

    def derive(self, name=None, level=None, mean=None, variance=None):
        if name is None:
            names = self._names
        else:
            names = (name,)
        if level is None:
            level = self._level
        else:
            level = level.upper()
        if mean is None:
            mean = self._mean
        else:
            mean = sympify(mean)
        if variance is None:
            variance = self._variance
        else:
            variance = sympify(variance)
            if sympy.ask(sympy.Q.nonnegative(variance)) is False:
                raise ValueError("Variance of normal distribution must be non-negative")
        return NormalDistribution(names, level, mean, variance)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def free_symbols(self):
        """Free symbols including random variable itself"""
        return (
            {sympy.Symbol(self._names[0])} | self._mean.free_symbols | self._variance.free_symbols
        )

    @property
    def parameter_names(self):
        """List of names of all parameters used in definition"""
        params = self._mean.free_symbols | self._variance.free_symbols
        return sorted([p.name for p in params])

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
        >>> omega = Parameter("OMEGA_CL", 0.1)
        >>> dist = NormalDistribution.create("IIV_CL", "IIV", 0, omega.symbol)
        >>> dist = dist.subs({omega.symbol: sympy.Symbol("OMEGA_NEW")})
        >>> dist
        IIV_CL ~ N(0, OMEGA_NEW)

        """
        mean = subs(self._mean, d)
        variance = subs(self._variance, d)
        name = self._names[0]
        if name in d or sympy.Symbol(name) in d:
            name = d.get(name, d.get(sympy.Symbol(name)))
            if isinstance(name, sympy.Symbol):
                name = name.name
        return NormalDistribution((name,), self._level, mean, variance)

    def get_variance(self, name):
        return self._variance

    def get_covariance(self, name1, name2):
        return sympy.Integer(0)

    def __eq__(self, other):
        return (
            isinstance(other, NormalDistribution)
            and self._names == other._names
            and self._level == other._level
            and self._mean == other._mean
            and self._variance == other._variance
        )

    def __hash__(self):
        return hash((self._names[0], self._level, self._mean, self._variance))

    def __repr__(self):
        return (
            f'{sympy.pretty(sympy.Symbol(self._names[0]), wrap_line=False, use_unicode=True)}'
            f' ~ {unicode.mathematical_script_capital_n}'
            f'({sympy.pretty(self._mean, wrap_line=False, use_unicode=True)}, '
            f'{sympy.pretty(self._variance, wrap_line=False, use_unicode=True)})'
        )

    def _latex_string(self, aligned=False):
        if aligned:
            align_str = ' & '
        else:
            align_str = ''
        rv = sympy.Symbol(self.names[0])._repr_latex_()[1:-1]
        mean = self._mean._repr_latex_()[1:-1]
        sigma = (self._variance)._repr_latex_()[1:-1]
        latex = rv + align_str + r'\sim  \mathcal{N} \left(' + mean + ',' + sigma + r'\right)'
        if not aligned:
            latex = '$' + latex + '$'
        return latex

    def _repr_latex_(self):
        return self._latex_string()


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

    def __init__(self, names, level, mean, variance):
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

    def derive(self, names=None, level=None, mean=None, variance=None):
        if names is None:
            names = self._names
        else:
            names = tuple(names)
        if level is None:
            level = self._level
        else:
            level = level.upper()
        if mean is None:
            mean = self._mean
        else:
            mean = sympy.ImmutableMatrix(mean)
        if variance is None:
            variance = self._variance
        else:
            variance = sympy.ImmutableMatrix(variance)
            if variance.is_positive_semidefinite is False:
                raise ValueError(
                    'Covariance matrix of joint normal distribution is not positive semidefinite'
                )
        return JointNormalDistribution(names, level, mean, variance)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def free_symbols(self):
        """Free symbols including random variable itself"""
        return (
            {sympy.Symbol(name) for name in self._names}
            | self._mean.free_symbols
            | self._variance.free_symbols
        )

    @property
    def parameter_names(self):
        """List of names of all parameters used in definition"""
        params = self._mean.free_symbols | self._variance.free_symbols
        return sorted([p.name for p in params])

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
        names = self._names
        new_names = []
        for name in names:
            if name in d or sympy.Symbol(name) in d:
                name = d.get(name, d.get(sympy.Symbol(name)))
                if name.is_Symbol:
                    name = name.name
            new_names.append(name)

        return JointNormalDistribution(new_names, self._level, mean, variance)

    def get_variance(self, name):
        i = self.names.index(name)
        return self._variance[i, i]

    def get_covariance(self, name1, name2):
        i1 = self.names.index(name1)
        i2 = self.names.index(name2)
        return self._variance[i1, i2]

    def __eq__(self, other):
        return (
            isinstance(other, JointNormalDistribution)
            and self._names == other._names
            and self._level == other._level
            and self._mean == other._mean
            and self._variance == other._variance
        )

    def __hash__(self):
        return hash((self._names, self._level, self._mean, self._variance))

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

    def _latex_string(self, aligned=False):
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

    def _repr_latex_(self):
        return self._latex_string()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_symengine_variance']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._symengine_variance = symengine.sympify(self._variance)


class VariabilityLevel:
    """A variability level

    Parameters
    ----------
    name : str
        A unique identifying name
    reference : bool
        Is this the reference level. Normally IIV would be the reference level
    group : str
        Name of data column to group this level. None for no grouping (default)
    """

    def __init__(self, name, reference=False, group=None):
        self._name = name
        self._reference = reference
        self._group = group

    def __eq__(self, other):
        return (
            isinstance(other, VariabilityLevel)
            and self._name == other._name
            and self._reference == other._reference
            and self._group == other._group
        )

    def __add__(self, other):
        if isinstance(other, VariabilityHierarchy):
            return VariabilityHierarchy([self] + other._levels)

    @property
    def name(self):
        """Name of the variability level"""
        return self._name

    @property
    def reference(self):
        """Is this the reference level"""
        return self._reference

    @property
    def group(self):
        """Group variable for variability level"""
        return self._group


class VariabilityHierarchy:
    """Description of a variability hierarchy"""

    def __init__(self, levels=None):
        if levels is None:
            self._levels = []
        elif isinstance(levels, VariabilityHierarchy):
            self._levels = levels._levels
        else:
            found_ref = False
            for level in levels:
                if not isinstance(level, VariabilityLevel):
                    raise ValueError("Can only add VariabilityLevel to VariabilityHierarchy")
                if level.reference:
                    if found_ref:
                        raise ValueError("A VariabilityHierarchy can only have one reference level")
                    else:
                        found_ref = True
            if not found_ref:
                raise ValueError("A VariabilityHierarchy must have a reference level")
            self._levels = list(levels)

    def __eq__(self, other):
        if not isinstance(other, VariabilityHierarchy):
            return False

        if len(self._levels) != len(other._levels):
            return False
        else:
            for l1, l2 in zip(self._levels, other._levels):
                if l1 != l2:
                    return False
            return True

    def _lookup(self, ind):
        # Lookup one index
        if isinstance(ind, int):
            # Index on numeric level for ints
            i = self._find_reference()
            return self._levels[ind - i]
        elif isinstance(ind, str):
            for varlev in self._levels:
                if varlev.name == ind:
                    return varlev
        elif isinstance(ind, VariabilityLevel):
            for varlev in self._levels:
                if varlev.name == ind.name:
                    return varlev
        raise KeyError(f'Could not find level {ind} in VariabilityHierarchy')

    def __getitem__(self, ind):
        if isinstance(ind, VariabilityHierarchy):
            levels = [level.name for level in ind._levels]
        elif not isinstance(ind, str) and isinstance(ind, Sequence):
            levels = ind
        else:
            return self._lookup(ind)
        new = [self._lookup(level) for level in levels]
        return VariabilityHierarchy(new)

    def __add__(self, other):
        if isinstance(other, VariabilityLevel):
            levels = [other]
        else:
            raise ValueError(f"Cannot add {other} to VariabilityLevel")
        new = VariabilityHierarchy(self._levels + levels)
        return new

    @property
    def names(self):
        """Names of all variability levels"""
        return [varlev.name for varlev in self._levels]

    def _find_reference(self):
        # Find numerical level of first level
        # No error checking since having a reference level is an invariant
        for i, level in enumerate(self._levels):
            if level.reference:
                return -i

    @property
    def levels(self):
        """Dictionary of variability level name to numerical level"""
        ind = self._find_reference()
        d = dict()
        for level in self._levels:
            d[level.name] = ind
            ind += 1
        return d

    def __len__(self):
        return len(self._levels)

    def __contains__(self, value):
        return value in self.names


class RandomVariables(Sequence):
    """A collection of distributions of random variables

    This class provides a container for random variables that preserves their order
    and acts list-like while also allowing for indexing on names.

    Each RandomVariables object has two VariabilityHierarchies that describes the
    allowed variability levels for the contined random variables. One hierarchy
    for residual error variability (epsilons) and one for parameter variability (etas).
    By default the eta hierarchy has the two levels IIV and IOV and the epsilon
    hierarchy has one single level.

    Parameters
    ----------
    rvs : list
        A list of RandomVariable to add. Default is to create an empty RandomVariabels.

    Examples
    --------
    >>> from pharmpy.model import RandomVariables, NormalDistribution, Parameter
    >>> omega = Parameter("OMEGA_CL", 0.1)
    >>> dist = NormalDistribution.create("IIV_CL", "iiv", 0, omega.symbol)
    >>> rvs = RandomVariables.create([dist])
    """

    def __init__(self, dists, eta_levels, epsilon_levels):
        self._dists = dists
        self._eta_levels = eta_levels
        self._epsilon_levels = epsilon_levels

    @classmethod
    def create(cls, dists=None, eta_levels=None, epsilon_levels=None):
        if dists is None:
            dists = ()
        elif isinstance(dists, Distribution):
            dists = (dists,)
        else:
            dists = tuple(dists)
            names = set()
            for dist in dists:
                if not isinstance(dist, Distribution):
                    raise TypeError(f'Can not add variable of type {type(dist)} to RandomVariables')
                for name in dist.names:
                    if name in names:
                        raise ValueError(
                            f'Names of random variables must be unique. Random Variable "{name}" '
                            'was added more than once to RandomVariables'
                        )
                    names.add(name)

        if eta_levels is None:
            iiv_level = VariabilityLevel('IIV', reference=True, group='ID')
            iov_level = VariabilityLevel('IOV', reference=False, group='OCC')
            eta_levels = VariabilityHierarchy([iiv_level, iov_level])
        else:
            if not isinstance(eta_levels, VariabilityHierarchy):
                raise TypeError(
                    f'Type of eta_levels must be a VariabilityHierarchy not a {type(eta_levels)}'
                )

        if epsilon_levels is None:
            ruv_level = VariabilityLevel('RUV', reference=True)
            epsilon_levels = VariabilityHierarchy([ruv_level])
        else:
            if not isinstance(epsilon_levels, VariabilityHierarchy):
                raise TypeError(
                    f'Type of epsilon_levels must be a VariabilityHierarchy not a {type(epsilon_levels)}'
                )

        return cls(dists, eta_levels, epsilon_levels)

    def derive(self, dists=None, eta_levels=None, epsilon_levels=None):
        if dists is None:
            dists = self._dists
        if eta_levels is None:
            eta_levels = self._eta_levels
        if epsilon_levels is None:
            epsilon_levels = self._epsilon_levels
        return RandomVariables(dists, eta_levels, epsilon_levels)

    @property
    def eta_levels(self):
        """VariabilityHierarchy for all etas"""
        return self._eta_levels

    @property
    def epsilon_levels(self):
        """VariabilityHierarchy for all epsilons"""
        return self._epsilon_levels

    def __add__(self, other):
        if isinstance(other, Distribution):
            if other.level not in self._eta_levels and other.level not in self._epsilon_levels:
                raise ValueError(
                    "Level of added distribution is not available in any variability hierarchy"
                )
            return RandomVariables(self._dists + (other,), self._eta_levels, self._epsilon_levels)
        elif isinstance(other, RandomVariables):
            if (
                self._eta_levels != other._eta_levels
                or self._epsilon_levels != other._epsilon_levels
            ):
                raise ValueError("RandomVariables must have same variability hierarchies")
            return RandomVariables(
                self._dists + other._dists, self._eta_levels, self._epsilon_levels
            )
        else:
            try:
                dists = tuple(other)
            except TypeError:
                raise TypeError(f'Type {type(other)} cannot be added to RandomVariables')
            else:
                return RandomVariables(self._dists + dists, self._eta_levels, self._epsilon_levels)

    def __radd__(self, other):
        if isinstance(other, Distribution):
            if other.level not in self._eta_levels and other.level not in self._epsilon_levels:
                raise ValueError(
                    f"Level {other.level} of added distribution is not available in any variability hierarchy"
                )
            return RandomVariables((other,) + self._dists, self._eta_levels, self._epsilon_levels)
        else:
            try:
                dists = tuple(other)
            except TypeError:
                raise TypeError(f'Type {type(other)} cannot be added to RandomVariables')
            else:
                return RandomVariables(dists + self._dists, self._eta_levels, self._epsilon_levels)

    def __len__(self):
        return len(self._dists)

    @property
    def nrvs(self):
        n = 0
        for dist in self._dists:
            n += len(dist)
        return n

    def __eq__(self, other):
        if len(self) == len(other):
            for s, o in zip(self._dists, other._dists):
                if s != o:
                    return False
            return (
                self._eta_levels == other._eta_levels
                and self._epsilon_levels == other._epsilon_levels
            )
        return False

    def _lookup_rv(self, ind):
        if isinstance(ind, sympy.Symbol):
            ind = ind.name
        if isinstance(ind, str):
            for i, dist in enumerate(self._dists):
                if ind in dist.names:
                    return i, dist
        raise KeyError(f'Could not find {ind} in RandomVariables')

    def __getitem__(self, ind):
        if isinstance(ind, int):
            return self._dists[ind]
        elif isinstance(ind, slice):
            return RandomVariables(
                self._dists[ind.start : ind.stop : ind.step], self._eta_levels, self._epsilon_levels
            )
        elif isinstance(ind, list) or isinstance(ind, tuple):
            remove = [name for name in self.names if name not in ind]
            split = self.unjoin(remove)
            keep = []
            for dist in split._dists:
                if dist.names[0] in ind:
                    keep.append(dist)
            return RandomVariables(keep, self._eta_levels, self._epsilon_levels)
        else:
            _, rv = self._lookup_rv(ind)
            return rv

    def __contains__(self, ind):
        try:
            _, _ = self._lookup_rv(ind)
        except KeyError:
            return False
        return True

    @property
    def names(self):
        """List of the names of all random variables"""
        names = []
        for dist in self._dists:
            names.extend(dist.names)
        return names

    @property
    def epsilons(self):
        """Get only the epsilons"""
        return RandomVariables(
            tuple([dist for dist in self._dists if dist.level in self._epsilon_levels.names]),
            self._eta_levels,
            self._epsilon_levels,
        )

    @property
    def etas(self):
        """Get only the etas"""
        return RandomVariables(
            tuple([dist for dist in self._dists if dist.level in self._eta_levels.names]),
            self._eta_levels,
            self._epsilon_levels,
        )

    @property
    def iiv(self):
        """Get only the iiv etas, i.e. etas with variability level 0"""
        return RandomVariables(
            tuple([dist for dist in self._dists if dist.level == self._eta_levels[0].name]),
            self._eta_levels,
            self._epsilon_levels,
        )

    @property
    def iov(self):
        """Get only the iov etas, i.e. etas with variability level 1"""
        return RandomVariables(
            tuple([dist for dist in self._dists if dist.level == self._eta_levels[1].name]),
            self._eta_levels,
            self._epsilon_levels,
        )

    @property
    def free_symbols(self):
        """Set of free symbols for all random variables"""
        symbs = set()
        for dist in self._dists:
            symbs |= dist.free_symbols
        return symbs

    @property
    def parameter_names(self):
        """List of parameter names for all random variables"""
        params = set()
        for dist in self._dists:
            params |= set(dist.parameter_names)
        return sorted([str(p) for p in params])

    @property
    def variance_parameters(self):
        """List of all parameters representing variance for all random variables"""
        parameters = []
        for dist in self._dists:
            if isinstance(dist, NormalDistribution):
                p = dist.variance
                if p not in parameters:
                    parameters.append(p)
            else:
                for p in dist.variance.diagonal():
                    if p not in parameters:
                        parameters.append(p)
        return [p.name for p in parameters]

    def get_covariance(self, rv1, rv2):
        """Get covariance between two random variables"""
        _, dist1 = self._lookup_rv(rv1)
        _, dist2 = self._lookup_rv(rv2)
        if dist1 is not dist2:
            return sympy.Integer(0)
        else:
            return dist1.get_covariance(rv1, rv2)

    def subs(self, d):
        """Substitute expressions

        Parameters
        ----------
        d : dict
            Dictionary of from: to pairs for substitution

        Examples
        --------
        >>> import sympy
        >>> from pharmpy.model import RandomVariables, Parameter
        >>> omega = Parameter("OMEGA_CL", 0.1)
        >>> dist = NormalDistribution.create("IIV_CL", "IIV", 0, omega.symbol)
        >>> rvs = RandomVariables.create([dist])
        >>> rvs.subs({omega.symbol: sympy.Symbol("OMEGA_NEW")})
        IIV_CL ~ N(0, OMEGA_NEW)

        """
        new_dists = [dist.subs(d) for dist in self._dists]
        return self.derive(dists=tuple(new_dists))

    def unjoin(self, inds):
        """Remove all covariances the random variables have with other random variables

        Parameters
        ----------
        inds
            One or multiple names or symbols to unjoin

        Examples
        --------
        >>> from pharmpy.model import RandomVariables, JointNormalDistribution, Parameter
        >>> omega_cl = Parameter("OMEGA_CL", 0.1)
        >>> omega_v = Parameter("OMEGA_V", 0.1)
        >>> corr_cl_v = Parameter("OMEGA_CL_V", 0.01)
        >>> dist1 = JointNormalDistribution.create(["IIV_CL", "IIV_V"], 'IIV', [0, 0],
        ...     [[omega_cl.symbol, corr_cl_v.symbol], [corr_cl_v.symbol, omega_v.symbol]])
        >>> rvs = RandomVariables.create([dist1])
        >>> rvs.unjoin('IIV_CL')
        IIV_CL ~ N(0, OMEGA_CL)
        IIV_V ~ N(0, OMEGA_V)

        See Also
        --------
        join
        """
        if not isinstance(inds, list):
            inds = [inds]
        inds = [ind.name if not isinstance(ind, str) else ind for ind in inds]

        newdists = []
        for dist in self._dists:
            first = True
            keep = None
            if isinstance(dist, JointNormalDistribution) and any(
                item in dist.names for item in inds
            ):
                for i, name in enumerate(dist.names):
                    if name in inds:  # unjoin  this
                        new = NormalDistribution(
                            (name,), dist.level, dist.mean[i], dist.variance[i, i]
                        )
                        newdists.append(new)
                    elif first:  # first of the ones to keep
                        first = False
                        remove = [i for i, n in enumerate(dist.names) if n in inds]
                        if len(dist) - len(remove) == 1:
                            keep = NormalDistribution(
                                (name,), dist.level, dist.mean[i], dist.variance[i, i]
                            )
                        else:
                            names = list(dist.names)
                            mean = sympy.Matrix(dist.mean)
                            variance = sympy.Matrix(dist.variance)
                            for i in reversed(remove):
                                del names[i]
                                mean.row_del(i)
                                variance.row_del(i)
                                variance.col_del(i)
                            keep = JointNormalDistribution(tuple(names), dist.level, mean, variance)
                if keep is not None:
                    newdists.append(keep)
            else:
                newdists.append(dist)
        new_rvs = RandomVariables(tuple(newdists), self._eta_levels, self._epsilon_levels)
        return new_rvs

    def join(self, inds, fill=0, name_template=None, param_names=None):
        """Join random variables together into one joint distribution

        Set new covariances (and previous 0 covs) to 'fill'.
        All joined random variables will form a new joint normal distribution and
        if they were part of previous joint normal distributions they will be taken out
        from these and the remaining variables will be stay.

        Parameters
        ----------
        inds
            Indices of variables to join
        fill : value
            Value to use for new covariances. Default is 0
        name_template : str
            A string template to use for new covariance symbols.
            Using this option will override fill.
        param_names : list
            List of parameter names to be used together with
            name_template.

        Returns
        -------
        A tuple of a the new RandomVariables and a dictionary from newly created covariance parameter names to
        tuple of parameter names. Empty dictionary if no parameter symbols were created

        Examples
        --------
        >>> from pharmpy.model import RandomVariables, NormalDistribution, Parameter
        >>> omega_cl = Parameter("OMEGA_CL", 0.1)
        >>> omega_v = Parameter("OMEGA_V", 0.1)
        >>> dist1 = NormalDistribution.create("IIV_CL", "IIV", 0, omega_cl.symbol)
        >>> dist2 = NormalDistribution.create("IIV_V", "IIV", 0, omega_v.symbol)
        >>> rvs = RandomVariables.create([dist1, dist2])
        >>> rvs, _ = rvs.join(['IIV_CL', 'IIV_V'])
        >>> rvs
        ⎡IIV_CL⎤    ⎧⎡0⎤  ⎡OMEGA_CL     0   ⎤⎫
        ⎢      ⎥ ~ N⎪⎢ ⎥, ⎢                 ⎥⎪
        ⎣IIV_V ⎦    ⎩⎣0⎦  ⎣   0      OMEGA_V⎦⎭

        See Also
        --------
        unjoin
        """
        if any(item not in self.names for item in inds):
            raise KeyError("Cannot join non-existing random variable")
        joined_rvs = self[inds]
        means, M, names, _ = joined_rvs._calc_covariance_matrix()
        cov_to_params = dict()
        if fill != 0:
            for row, col in itertools.product(range(M.rows), range(M.cols)):
                if M[row, col] == 0:
                    M[row, col] = fill
        elif name_template:
            for row, col in itertools.product(range(M.rows), range(M.cols)):
                if M[row, col] == 0 and row > col:
                    param_1, param_2 = M[row, row], M[col, col]
                    cov_name = name_template.format(param_names[col], param_names[row])
                    cov_to_params[cov_name] = (str(param_1), str(param_2))
                    M[row, col], M[col, row] = sympy.Symbol(cov_name), sympy.Symbol(cov_name)
        joined_dist = JointNormalDistribution(
            tuple(names), joined_rvs[0].level, sympy.Matrix(means), M
        )

        unjoined_rvs = self.unjoin(inds)
        newdists = []
        first = True
        for dist in unjoined_rvs._dists:
            if any(item in dist.names for item in inds):
                if first:
                    first = False
                    newdists.append(joined_dist)
            else:
                newdists.append(dist)

        new_rvs = RandomVariables(tuple(newdists), self._eta_levels, self._epsilon_levels)
        return new_rvs, cov_to_params

    def nearest_valid_parameters(self, parameter_values):
        """Force parameter values into being valid

        As small changes as possible

        returns a dict with the valid parameter values
        """
        nearest = parameter_values.copy()
        for dist in self._dists:
            if len(dist) > 1:
                symb_sigma = dist._variance
                sigma = symb_sigma.subs(dict(parameter_values))
                A = np.array(sigma).astype(np.float64)
                B = pharmpy.math.nearest_postive_semidefinite(A)
                if B is not A:
                    for row in range(len(A)):
                        for col in range(row + 1):
                            nearest[symb_sigma[row, col].name] = B[row, col]
        return nearest

    def validate_parameters(self, parameter_values):
        """Validate a dict or Series of parameter values

        Currently checks that all covariance matrices are positive semidefinite
        use_cache for using symengine cached matrices
        """
        for dist in self._dists:
            if len(dist) > 1:
                sigma = dist._symengine_variance
                replacement = {}
                for param in dict(parameter_values):
                    replacement[symengine.Symbol(param)] = parameter_values[param]
                sigma = sigma.subs(replacement)
                if not sigma.free_symbols:  # Cannot validate since missing params
                    a = np.array(sigma).astype(np.float64)
                    if not pharmpy.math.is_positive_semidefinite(a):
                        return False
                else:
                    raise TypeError("Cannot validate parameters since all are not numeric")
        return True

    def sample(self, expr, parameters=None, samples=1, rng=None):
        """Sample from the distribution of expr

        parameters in the distributions will first be replaced"""

        sympified_expr = sympify(expr)
        xreplace_parameters = {} if parameters is None else xreplace_dict(parameters)

        return _sample_from_distributions(
            self,
            sympified_expr,
            xreplace_parameters,
            samples,
            _create_rng(rng),
        )

    def _calc_covariance_matrix(self):
        non_altered = []
        means = []
        blocks = []
        names = []
        for dist in self._dists:
            names.extend(dist.names)
            if isinstance(dist, NormalDistribution):
                means.append(dist.mean)
                blocks.append(sympy.Matrix([dist.variance]))
            else:  # isinstance(dist, JointNormalDistribution):
                means.extend(dist.mean)
                blocks.append(dist.variance)
        if names:
            M = sympy.BlockDiagMatrix(*blocks)
            M = sympy.Matrix(M)
        else:
            M = sympy.Matrix()
        return means, M, names, non_altered

    @property
    def covariance_matrix(self):
        """Covariance matrix of all random variables"""
        _, M, _, _ = self._calc_covariance_matrix()
        return M

    def __repr__(self):
        strings = []
        for dist in self._dists:
            strings.append(repr(dist))
        return '\n'.join(strings)

    def _repr_latex_(self):
        lines = []
        for dist in self._dists:
            latex = dist._latex_string(aligned=True)
            lines.append(latex)
        return '\\begin{align*}\n' + r' \\ '.join(lines) + '\\end{align*}'

    def parameters_sdcorr(self, values):
        """Convert parameter values to sd/corr form

        All parameter values will be converted to sd/corr assuming
        they are given in var/cov form.

        Parameters
        ----------
        values : dict
            Dict of parameter names to values
        """
        newdict = values.copy()
        for dist in self._dists:
            if len(dist) > 1:
                sigma_sym = dist.variance
                sigma = np.array(sigma_sym.subs(values)).astype(np.float64)
                corr = pharmpy.math.cov2corr(sigma)
                for i in range(sigma_sym.rows):
                    for j in range(sigma_sym.cols):
                        name = sigma_sym[i, j].name
                        if i != j:
                            newdict[name] = corr[i, j]
                        else:
                            newdict[name] = np.sqrt(sigma[i, j])
            else:
                name = dist.variance.name
                if name in newdict:
                    newdict[name] = np.sqrt(
                        np.array(subs(dist.variance, values)).astype(np.float64)
                    )
        return newdict


def _sample_from_distributions(distributions, expr, parameters, nsamples, rng):
    random_variable_symbols = expr.free_symbols.difference(parameters.keys())
    filtered_distributions = filter_distributions(distributions, random_variable_symbols)
    sampling_rvs = subs_distributions(filtered_distributions, parameters)
    return sample_expr_from_rvs(sampling_rvs, expr, parameters, nsamples, rng)


def filter_distributions(
    distributions: Iterable[Distribution], symbols: Set[sympy.Symbol]
) -> Iterable[Distribution]:
    covered_symbols = set()

    for dist in distributions:
        symbols_covered_by_dist = symbols.intersection(sympy.Symbol(rv) for rv in dist.names)
        if symbols_covered_by_dist:
            yield dist
            covered_symbols |= symbols_covered_by_dist

    if covered_symbols != symbols:
        raise ValueError('Could not cover all requested symbols with given distributions')


def subs_distributions(
    distributions: Iterable[Distribution], parameters: Dict[sympy.Symbol, float]
) -> Iterable[Tuple[Tuple[sympy.Symbol, ...], NumericDistribution]]:

    for dist in distributions:
        rvs_symbols = tuple(map(sympy.Symbol, dist.names))
        if len(rvs_symbols) > 1:
            try:
                mu = np.array(symengine.sympify(dist.mean).xreplace(parameters)).astype(np.float64)[
                    :, 0
                ]
                sigma = np.array(dist._symengine_variance.xreplace(parameters)).astype(np.float64)
                distribution = NumericMultivariateNormalDistribution(mu, sigma)
            except RuntimeError as e:
                # NOTE This handles missing parameter substitutions
                raise ValueError(e)
        else:
            # mu = float(symengine.sympify(rv._mean[0]).xreplace(parameters))
            # sigma = float(symengine.sympify(sympy.sqrt(rv._variance[0,0])).xreplace(parameters))
            mean = dist.mean
            variance = dist.variance
            try:
                mu = 0 if mean == 0 else float(parameters[mean])
                sigma = 0 if variance == 0 else sqrt(float(parameters[variance]))
                distribution = NumericNormalDistribution(mu, sigma)
            except KeyError as e:
                # NOTE This handles missing parameter substitutions
                raise ValueError(e)

        yield (rvs_symbols, distribution)


def sample_expr_from_rvs(
    sampling_rvs: Iterable[Tuple[Tuple[sympy.Symbol, ...], NumericDistribution]],
    expr: sympy.Expr,
    parameters: Dict[sympy.Symbol, float],
    nsamples: int,
    rng,
):
    expr = expr.xreplace(parameters)

    rvs = list(sampling_rvs)

    if not rvs:
        return np.full(nsamples, float(expr.evalf()))

    samples = sample_rvs(rvs, nsamples, rng)

    return eval_expr(expr, samples)


def eval_expr(
    expr: sympy.Expr,
    samples: Dict[sympy.Symbol, np.ndarray],
) -> np.ndarray:
    ordered_symbols, fn = _lambdify_canonical(expr)
    data = [samples[rv] for rv in ordered_symbols]
    return fn(*data)


def sample_rvs(
    sampling_rvs: Iterable[Tuple[Tuple[sympy.Symbol, ...], NumericDistribution]],
    nsamples: int,
    rng,
) -> Dict[sympy.Symbol, np.ndarray]:
    data = {}
    for symbols, distribution in sampling_rvs:
        cursample = distribution.sample(rng, nsamples)
        if len(symbols) > 1:
            # NOTE this makes column iteration faster
            cursample = np.array(cursample, order='F')
            for j, s in enumerate(symbols):
                data[s] = cursample[:, j]
        else:
            data[symbols[0]] = cursample

    return data


@lru_cache(maxsize=256)
def _lambdify_canonical(expr):
    ordered_symbols = sorted(expr.free_symbols, key=str)
    # NOTE Substitution allows to use cse. Otherwise weird things happen with
    # symbols that look like function eval (e.g. ETA(1), THETA(3), OMEGA(1,1)).
    ordered_substitutes = [sympy.Symbol(f'__tmp{i}') for i in range(len(ordered_symbols))]
    substituted_expr = expr.xreplace(
        {key: value for key, value in zip(ordered_symbols, ordered_substitutes)}
    )
    fn = sympy.lambdify(ordered_substitutes, substituted_expr, modules='numpy', cse=True)
    return ordered_symbols, fn
