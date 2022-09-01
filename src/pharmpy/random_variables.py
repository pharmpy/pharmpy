import copy
import itertools
import warnings
from collections.abc import MutableSequence, Sequence

import numpy as np
import pandas as pd
import symengine
import sympy
import sympy.stats as stats
from sympy import Symbol as symbol
from sympy.stats.crv_types import ExponentialDistribution, NormalDistribution
from sympy.stats.joint_rv_types import MultivariateNormalDistribution

import pharmpy.math
import pharmpy.unicode as unicode
from pharmpy.statements import sympify


def _create_rng(seed=None):
    """Create a new random number generator"""
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)
    return rng


class RandomVariable:
    """A single random variable

    Parameters
    ----------
    name : str
        Name of the random variable
    level : str
        Name of the variability level. The default levels are IIV, IOV and RUV
    sympy_rv : sympy.RandomSymbol
        RandomSymbol to use for this random variable. See also the normal
        and joint_normal classmethods.

    Examples
    --------
    >>> import sympy
    >>> import sympy.stats
    >>> from pharmpy import RandomVariable
    >>> name = "ETA(1)"
    >>> sd = sympy.sqrt(sympy.Symbol('OMEGA(1,1)'))
    >>> rv = RandomVariable(name, "IIV", sympy.stats.Normal(name, 0, sd))
    >>> rv
    ETA(1) ~ N(0, OMEGA(1,1))

    See Also
    --------
    normal, joint_normal
    """

    def __init__(self, name, level, sympy_rv=None):
        self._name = name
        self.level = level
        self.symbol = symbol(name)
        self._sympy_rv = sympy_rv
        if sympy_rv is not None:
            if isinstance(sympy_rv.pspace.distribution, NormalDistribution):
                self._mean = sympy.Matrix([sympy_rv.pspace.distribution.mean])
                self._variance = sympy.Matrix([sympy_rv.pspace.distribution.std**2])
            elif isinstance(sympy_rv.pspace.distribution, MultivariateNormalDistribution):
                raise ValueError(
                    "Cannot create multivariate random variables using constructor. "
                    "Use the joint_normal classmethod instead."
                )
            else:
                self._mean = None
                self._variance = None
        else:
            self._mean = None
            self._variance = None
        self._symengine_variance = None
        self._joint_names = None

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.level == other.level
            and self._mean == other._mean
            and self._variance == other._variance
            and self._sympy_rv == other._sympy_rv
        )

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        level = RandomVariable._canonicalize_level(value)
        self._level = level

    @staticmethod
    def _canonicalize_level(level):
        supported = ('IIV', 'IOV', 'RUV')
        ulevel = level.upper()
        if ulevel not in supported:
            raise ValueError(f'Unknown variability level {level}. Must be one of {supported}.')
        return ulevel

    @classmethod
    def normal(cls, name, level, mean, variance):
        """Create a normally distributed random variable

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
        >>> from pharmpy import RandomVariable, Parameter
        >>> omega = Parameter('OMEGA_CL', 0.1)
        >>> rv = RandomVariable.normal("IIV_CL", "IIV", 0, omega.symbol)
        >>> rv
        IIV_CL ~ N(0, OMEGA_CL)

        """
        rv = cls(name, level)
        rv._mean = sympy.Matrix([sympify(mean)])
        rv._variance = sympy.Matrix([sympify(variance)])
        if rv._variance.is_positive_semidefinite is False:
            raise ValueError(f"Mean cannot be {mean} must be positive")
        rv._symengine_variance = symengine.sympify(rv._variance)
        return rv

    @classmethod
    def joint_normal(cls, names, level, mu, sigma):
        """Create joint normally distributed random variables

        Parameters
        ----------
        names : list
            Names of the random variables
        level : str
            Variability level
        mu : matrix or list
            Vector of the means of the random variables
        sigma : matrix or list of lists
            Covariance matrix of the random variables

        Example
        -------
        >>> from pharmpy import RandomVariable, Parameter
        >>> omega_cl = Parameter("OMEGA_CL", 0.1)
        >>> omega_v = Parameter("OMEGA_V", 0.1)
        >>> corr_cl_v = Parameter("OMEGA_CL_V", 0.01)
        >>> rv1, rv2 = RandomVariable.joint_normal(["IIV_CL", "IIV_V"], 'IIV', [0, 0],
        ...     [[omega_cl.symbol, corr_cl_v.symbol], [corr_cl_v.symbol, omega_v.symbol]])
        >>> rv1
        ⎡IIV_CL⎤    ⎧⎡0⎤  ⎡ OMEGA_CL   OMEGA_CL_V⎤⎫
        ⎢      ⎥ ~ N⎪⎢ ⎥, ⎢                      ⎥⎪
        ⎣IIV_V ⎦    ⎩⎣0⎦  ⎣OMEGA_CL_V   OMEGA_V  ⎦⎭

        """

        mean = sympy.Matrix(mu)
        variance = sympy.Matrix(sigma)
        if variance.is_positive_semidefinite is False:
            raise ValueError('Sigma matrix is not positive semidefinite')
        rvs = []
        for name in names:
            rv = cls(name, level)
            rv._mean = mean.copy()
            rv._variance = variance.copy()
            rv._symengine_variance = symengine.Matrix(variance.rows, variance.cols, sigma)
            rv._joint_names = names.copy()
            rvs.append(rv)
        return rvs

    @property
    def name(self):
        """Name of the random variable"""
        return self._name

    @name.setter
    def name(self, name):
        if self._joint_names:
            index = self._joint_names.index(self._name)
            self._joint_names[index] = name
        self._name = name

    @property
    def joint_names(self):
        """Names of all (including this) jointly varying rvs in a list"""
        return [] if not self._joint_names else self._joint_names

    @property
    def sympy_rv(self):
        """Corresponding sympy random variable"""
        if self._sympy_rv is None:
            # Normal distribution that might have 0 variance
            if len(self._variance) == 1 and self._variance[0].is_zero:
                return sympy.Integer(0)
            elif self._mean.rows > 1:
                return sympy.stats.Normal('X', self._mean, self._variance)
            else:
                return sympy.stats.Normal(self.name, self._mean[0], sympy.sqrt(self._variance[0]))
        else:
            return self._sympy_rv

    @property
    def free_symbols(self):
        """Free symbols including random variable itself"""
        if self._mean is not None:
            return {self.symbol} | self._mean.free_symbols | self._variance.free_symbols
        else:
            free = {s for s in self.sympy_rv.pspace.free_symbols if s.name != self.name}
            return free | {self.symbol}

    @property
    def parameter_names(self):
        """List of names of all parameters used in definition"""
        if self._mean is not None:
            params = self._mean.free_symbols | self._variance.free_symbols
        else:
            params = {s for s in self.sympy_rv.pspace.free_symbols if s.name != self.name}
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
        >>> from pharmpy import RandomVariable, Parameter
        >>> omega = Parameter("OMEGA_CL", 0.1)
        >>> rv = RandomVariable.normal("IIV_CL", "IIV", 0, omega.symbol)
        >>> rv.subs({omega.symbol: sympy.Symbol("OMEGA_NEW")})
        >>> rv
        IIV_CL ~ N(0, OMEGA_NEW)

        """
        if self._mean is not None:
            self._mean = self._mean.subs(d)
            self._variance = self._variance.subs(d)
            self._symengine_variance = symengine.Matrix(
                self._variance.rows, self._variance.cols, self._variance
            )
        if self._sympy_rv is not None:
            self._sympy_rv = self._sympy_rv.subs(d)

    def copy(self):
        """Make copy of RandomVariable"""
        return copy.deepcopy(self)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        # Custom copier because symengine objects cannot be copied
        new = RandomVariable(self.name, self.level)
        new._mean = self._mean.copy()
        new._variance = self._variance.copy()
        new._symengine_variance = symengine.sympify(self._variance)
        new._sympy_rv = self._sympy_rv
        if self._joint_names is None:
            new._joint_names = None
        else:
            new._joint_names = self._joint_names.copy()
        return new

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_symengine_variance']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._symengine_variance = symengine.sympify(self._variance)

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        if self._mean is not None:  # Normal distribution
            if self._mean.rows > 1:
                name_vector = sympy.Matrix(self._joint_names)
                name_strings = sympy.pretty(name_vector, wrap_line=False, use_unicode=True).split(
                    '\n'
                )
                mu_strings = sympy.pretty(self._mean, wrap_line=False, use_unicode=True).split('\n')
                sigma_strings = sympy.pretty(
                    self._variance, wrap_line=False, use_unicode=True
                ).split('\n')
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
            else:
                return (
                    f'{sympy.pretty(self.symbol, wrap_line=False, use_unicode=True)}'
                    f' ~ {unicode.mathematical_script_capital_n}'
                    f'({sympy.pretty(self._mean[0], wrap_line=False, use_unicode=True)}, '
                    f'{sympy.pretty(self._variance[0], wrap_line=False, use_unicode=True)})'
                )
        else:
            if isinstance(self.sympy_rv.pspace.distribution, ExponentialDistribution):
                return (
                    f'{sympy.pretty(self.symbol, use_unicode=True)} ~ '
                    f'Exp({self.sympy_rv.pspace.distribution.rate})'
                )
            else:
                return f'{sympy.pretty(self.symbol, use_unicode=True)} ~ UnknownDistribution'

    def _latex_string(self, aligned=False):
        if aligned:
            align_str = ' & '
        else:
            align_str = ''
        if self._mean.rows > 1:
            rv_vec = sympy.Matrix(self._joint_names)._repr_latex_()[1:-1]
            mean_vec = self._mean._repr_latex_()[1:-1]
            sigma = self._variance._repr_latex_()[1:-1]
            latex = (
                rv_vec
                + align_str
                + r'\sim \mathcal{N} \left('
                + mean_vec
                + ','
                + sigma
                + r'\right)'
            )
        else:
            rv = self.symbol._repr_latex_()[1:-1]
            mean = self._mean[0]._repr_latex_()[1:-1]
            sigma = (self._variance[0])._repr_latex_()[1:-1]
            latex = rv + align_str + r'\sim  \mathcal{N} \left(' + mean + ',' + sigma + r'\right)'
        if not aligned:
            latex = '$' + latex + '$'
        return latex

    def _repr_latex_(self):
        return self._latex_string()


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


class RandomVariables(MutableSequence):
    """A collection of random variables

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
    >>> from pharmpy import RandomVariables, RandomVariable, Parameter
    >>> omega = Parameter("OMEGA_CL", 0.1)
    >>> rv = RandomVariable.normal("IIV_CL", "iiv", 0, omega.symbol)
    >>> rvs = RandomVariables([rv])
    """

    def __init__(self, rvs=None):
        if isinstance(rvs, RandomVariables):
            self._rvs = copy.deepcopy(rvs._rvs)
        elif rvs is None:
            self._rvs = []
        else:
            self._rvs = list(rvs)
            names = set()
            for rv in self._rvs:
                if not isinstance(rv, RandomVariable):
                    raise ValueError(f'Can not add variable of type {type(rv)} to RandomVariables')
                if rv.name in names:
                    raise ValueError(
                        f'Names of random variables must be unique. Random Variable "{rv.name}" '
                        'was added more than once to RandomVariables'
                    )
                names.add(rv.name)

        iiv_level = VariabilityLevel('IIV', reference=True, group='ID')
        iov_level = VariabilityLevel('IOV', reference=False, group='OCC')
        eta_levels = VariabilityHierarchy([iiv_level, iov_level])
        ruv_level = VariabilityLevel('RUV', reference=True)
        epsilon_levels = VariabilityHierarchy([ruv_level])
        self._eta_levels = eta_levels
        self._epsilon_levels = epsilon_levels

    def __len__(self):
        return len(self._rvs)

    def __eq__(self, other):
        if len(self) == len(other):
            for s, o in zip(self, other):
                if s != o:
                    return False
            return True
        return False

    def _lookup_rv(self, ind, insert=False):
        if isinstance(ind, sympy.Symbol):
            ind = ind.name
        if isinstance(ind, str):
            for i, rv in enumerate(self._rvs):
                if ind == rv.name:
                    return i, rv
            raise KeyError(f'Could not find {ind} in RandomVariables')
        elif isinstance(ind, RandomVariable):
            try:
                i = self._rvs.index(ind)
            except ValueError:
                raise KeyError(f'Could not find {ind.name} in RandomVariables')
            return i, ind
        if insert:
            # Must allow for inserting after last element.
            return ind, None
        else:
            return ind, self._rvs[ind]

    def __getitem__(self, ind):
        if isinstance(ind, list):
            rvs = []
            for i in ind:
                index, rv = self._lookup_rv(i)
                rvs.append(self[index])
            return RandomVariables(rvs)
        else:
            _, rv = self._lookup_rv(ind)
            return rv

    def __contains__(self, ind):
        try:
            _, _ = self._lookup_rv(ind)
        except KeyError:
            return False
        return True

    def _remove_joint_normal(self, rv):
        # Remove rv from all other rvs
        for other in self:
            if other.name == rv.name:
                continue
            joint_names = other._joint_names
            if joint_names is None or rv.name not in joint_names:
                continue
            joint_index = joint_names.index(rv.name)
            del other._joint_names[joint_index]
            if len(other._joint_names) == 1:
                other._joint_names = None
            other._mean.row_del(joint_index)
            other._variance.row_del(joint_index)
            other._variance.col_del(joint_index)
            other._symengine_variance = symengine.sympify(other._variance)

    def _remove_joint_normal_not_in_self(self):
        # Remove rv from all joint normals not in self
        names = self.names
        for rv in self._rvs:
            if rv._joint_names is not None:
                indices = [i for i, joint_name in enumerate(rv._joint_names) if joint_name in names]
                new_joint = [rv._joint_names[i] for i in indices]
                if len(new_joint) == 1:
                    new_joint = None
                rv._joint_names = new_joint
                means = [rv._mean[i] for i in indices]
                rv._mean = sympy.Matrix(means)
                rv._variance = rv._variance[indices, indices]
                rv._symengine_variance = symengine.sympify(rv._variance)

    def __setitem__(self, ind, value):
        if isinstance(ind, slice):
            if ind.step is None:
                step = 1
            else:
                step = ind.step
            indices = list(range(ind.start, ind.stop, step))
            if len(value) != len(indices):
                raise ValueError('Bad number of rvs to set using slice')
            for i, val in zip(indices, value):
                self[i] = val
            return
        if not isinstance(value, RandomVariable):
            raise ValueError(
                f'Trying to set {type(value)} to RandomVariables. Must be of type RandomVariable.'
            )
        i, rv = self._lookup_rv(ind)
        self.unjoin(rv)
        i, _ = self._lookup_rv(ind)  # Might have moved
        self._rvs[i] = value

    def __delitem__(self, ind):
        i, rv = self._lookup_rv(ind)
        joint_names = rv._joint_names
        if joint_names is not None:
            joint_names = joint_names.copy()
            joint_index = joint_names.index(rv.name)
            for name in joint_names:
                other = self[name]
                del other._joint_names[joint_index]
                if len(other._joint_names) == 1:
                    other._joint_names = None
                other._mean.row_del(joint_index)
                other._variance.row_del(joint_index)
                other._variance.col_del(joint_index)
                other._symengine_variance = symengine.sympify(other._variance)
        del self._rvs[i]

    def __sub__(self, other):
        new = RandomVariables(self._rvs)
        for rv in other:
            if rv in new:
                del new[rv]
        return new

    def insert(self, ind, value):
        if not isinstance(value, RandomVariable):
            raise ValueError(
                f'Trying to insert {type(value)} into RandomVariables. '
                'Must be of type RandomVariable.'
            )
        i, _ = self._lookup_rv(ind, insert=True)
        self._rvs.insert(i, value)

    @property
    def names(self):
        """List of the names of all random variables"""
        return [rv.name for rv in self._rvs]

    @property
    def epsilons(self):
        """Get only the epsilons"""
        return RandomVariables([rv for rv in self._rvs if rv.level in self._epsilon_levels.names])

    @property
    def etas(self):
        """Get only the etas"""
        return RandomVariables([rv for rv in self._rvs if rv.level in self._eta_levels.names])

    @property
    def iiv(self):
        """Get only the iiv etas, i.e. etas with variability level 0"""
        return RandomVariables([rv for rv in self._rvs if rv.level == self._eta_levels[0].name])

    @property
    def iov(self):
        """Get only the iov etas, i.e. etas with variability level 1"""
        return RandomVariables([rv for rv in self._rvs if rv.level == self._eta_levels[1].name])

    @property
    def free_symbols(self):
        """Set of free symbols for all random variables"""
        symbs = set()
        for rv in self._rvs:
            symbs |= rv.free_symbols
        return symbs

    def copy(self):
        """Make copy of RandomVariables"""
        new = RandomVariables()
        for rv in self._rvs:
            new._rvs.append(rv.copy())
        return new

    @property
    def parameter_names(self):
        """List of parameter names for all random variables"""
        params = set()
        for rv in self:
            params |= set(rv.parameter_names)
        return sorted([str(p) for p in params])

    @property
    def variance_parameters(self):
        """List of all parameters representing variance for all random variables"""
        parameters = []
        for rvs, dist in self.distributions():
            if len(rvs) == 1:
                p = dist.std**2
                if p not in parameters:
                    parameters.append(dist.std**2)
            else:
                for p in dist.sigma.diagonal():
                    if p not in parameters:
                        parameters.append(p)
        return [p.name for p in parameters]

    def get_variance(self, rv):
        """Get variance for a random variable"""
        _, rv = self._lookup_rv(rv)
        if rv._joint_names is None:
            return rv._variance[0]
        else:
            i = rv._joint_names.index(rv.name)
            return rv._variance[i, i]

    def get_covariance(self, rv1, rv2):
        """Get covariance between two random variables"""
        _, rv1 = self._lookup_rv(rv1)
        _, rv2 = self._lookup_rv(rv2)
        if rv1._joint_names is None or rv2.name not in rv1._joint_names:
            return sympy.Integer(0)
        else:
            i1 = rv1._joint_names.index(rv1.name)
            i2 = rv1._joint_names.index(rv2.name)
            return rv1._variance[i1, i2]

    def _rename_rv(self, current, new):
        for rv in self._rvs:
            if rv.name == current:
                rv.name = new
            if rv._joint_names and current in rv._joint_names:
                i = rv._joint_names.index(current)
                rv._joint_names[i] = new

    def subs(self, d):
        """Substitute expressions

        Parameters
        ----------
        d : dict
            Dictionary of from: to pairs for substitution

        Examples
        --------
        >>> import sympy
        >>> from pharmpy import RandomVariables, Parameter
        >>> omega = Parameter("OMEGA_CL", 0.1)
        >>> rv = RandomVariable.normal("IIV_CL", "IIV", 0, omega.symbol)
        >>> rvs = RandomVariables([rv])
        >>> rvs.subs({omega.symbol: sympy.Symbol("OMEGA_NEW")})
        >>> rvs
        IIV_CL ~ N(0, OMEGA_NEW)

        """
        s = dict()
        for key, value in d.items():
            key = sympify(key)
            value = sympify(value)
            if key.name in self.names:
                self._rename_rv(key.name, value.name)
            else:
                s[key] = value
        for rv in self._rvs:
            rv.subs(s)

    def unjoin(self, inds):
        """Remove all covariances the random variables have with other random variables

        Parameters
        ----------
        inds
            One or multiple indices to unjoin

        Examples
        --------
        >>> from pharmpy import RandomVariables, RandomVariable, Parameter
        >>> omega_cl = Parameter("OMEGA_CL", 0.1)
        >>> omega_v = Parameter("OMEGA_V", 0.1)
        >>> corr_cl_v = Parameter("OMEGA_CL_V", 0.01)
        >>> rv1, rv2 = RandomVariable.joint_normal(["IIV_CL", "IIV_V"], 'IIV', [0, 0],
        ...     [[omega_cl.symbol, corr_cl_v.symbol], [corr_cl_v.symbol, omega_v.symbol]])
        >>> rvs = RandomVariables([rv1, rv2])
        >>> rvs.unjoin('IIV_CL')
        >>> rvs
        IIV_CL ~ N(0, OMEGA_CL)
        IIV_V ~ N(0, OMEGA_V)

        See Also
        --------
        join
        """
        if not isinstance(inds, list):
            inds = [inds]
        for ind in inds:
            i, rv = self._lookup_rv(ind)
            if rv._joint_names is None:
                return
            index = rv._joint_names.index(rv.name)
            self._remove_joint_normal(rv)
            del self._rvs[i]
            rv._mean = sympy.Matrix([rv._mean[index]])
            rv._variance = sympy.Matrix([rv._variance[index, index]])
            rv._symengine_variance = symengine.sympify(rv._variance)
            rv._joint_names = None
            self._rvs.insert(i - index, rv)

    def join(self, inds, fill=0, name_template=None, param_names=None):
        """Join random variables together into one joint distribution

        Set new covariances (and previous 0 covs) to 'fill'

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
        A dictionary from newly created covariance parameter names to
        tuple of parameter names. Empty dictionary if no parameter
        symbols were created

        Examples
        --------
        >>> from pharmpy import RandomVariables, RandomVariable, Parameter
        >>> omega_cl = Parameter("OMEGA_CL", 0.1)
        >>> omega_v = Parameter("OMEGA_V", 0.1)
        >>> rv1 = RandomVariable.normal("IIV_CL", 'IIV', 0, omega_cl.symbol)
        >>> rv2 = RandomVariable.normal("IIV_V", 'IIV', 0, omega_v.symbol)
        >>> rvs = RandomVariables([rv1, rv2])
        >>> rvs.join(['IIV_CL', 'IIV_V'])
        {}
        >>> rvs
        ⎡IIV_CL⎤    ⎧⎡0⎤  ⎡OMEGA_CL     0   ⎤⎫
        ⎢      ⎥ ~ N⎪⎢ ⎥, ⎢                 ⎥⎪
        ⎣IIV_V ⎦    ⎩⎣0⎦  ⎣   0      OMEGA_V⎦⎭

        See Also
        --------
        unjoin
        """
        cov_to_params = dict()
        selection = self[inds]
        selection._remove_joint_normal_not_in_self()
        means, M, names, others = selection._calc_covariance_matrix()
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
                    M[row, col], M[col, row] = symbol(cov_name), symbol(cov_name)

        for i in inds:
            self._remove_joint_normal(self[i])

        new = []
        first = True
        for rv in self._rvs:
            if rv.name in selection:
                if first:
                    new.extend(selection._rvs)
                    first = False
            else:
                if rv not in new:
                    new.append(rv)
                    joint_etas = [
                        self[eta_name] for eta_name in rv.joint_names if eta_name != rv.name
                    ]
                    new += joint_etas

        new_rvs = RandomVariable.joint_normal(names, 'iiv', means, M)
        for rv, new_rv in zip(selection, new_rvs):
            rv._sympy_rv = new_rv._sympy_rv
            rv._mean = sympy.Matrix(means)
            rv._variance = M.copy()
            rv._symengine_variance = symengine.Matrix(M.rows, M.cols, M)
            rv._joint_names = [rv.name for rv in new_rvs]
        self._rvs = new
        return cov_to_params

    def distributions(self):
        """List with one entry per distribution instead of per random variable.

        Returned is a list of tuples of a list of random variables that are jointly
        distributed and the distribution.

        Example
        -------
        >>> from pharmpy import RandomVariables, RandomVariable, Parameter
        >>> omega_cl = Parameter("OMEGA_CL", 0.1)
        >>> omega_v = Parameter("OMEGA_V", 0.1)
        >>> omega_ka = Parameter("OMEGA_KA", 0.1)
        >>> corr_cl_v = Parameter("OMEGA_CL_V", 0.01)
        >>> rv1, rv2 = RandomVariable.joint_normal(["IIV_CL", "IIV_V"], 'IIV', [0, 0],
        ...     [[omega_cl.symbol, corr_cl_v.symbol], [corr_cl_v.symbol, omega_v.symbol]])
        >>> rv3 = RandomVariable.normal("IIV_KA", 'IIV', 0, omega_ka.symbol)
        >>> rvs = RandomVariables([rv1, rv2, rv3])
        >>> dists = rvs.distributions()

        """
        distributions = []
        i = 0
        while i < len(self):
            rv = self[i]
            symrv = rv.sympy_rv
            n = 1 if rv._joint_names is None else len(rv._joint_names)
            dist = symrv.pspace.distribution
            if isinstance(dist, stats.crv_types.NormalDistribution):
                i += 1
                distributions.append(([rv], dist))
            else:  # Joint Normal
                rvs = [self[k] for k in range(i, i + n)]
                i += n
                distributions.append((rvs, dist))
        return distributions

    def nearest_valid_parameters(self, parameter_values):
        """Force parameter values into being valid

        As small changes as possible

        returns a dict with the valid parameter values
        """
        nearest = parameter_values.copy()
        for rvs, dist in self.distributions():
            if len(rvs) > 1:
                symb_sigma = rvs[0]._variance
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
        for rvs, dist in self.distributions():
            if len(rvs) > 1:
                sigma = rvs[0]._symengine_variance
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

        parameters in the distriutions will first be replaced"""
        rng = _create_rng(rng)
        if not parameters:
            parameters = dict()
        expr = sympify(expr).subs(parameters)
        symbols = expr.free_symbols
        expr_names = [symb.name for symb in symbols]
        i = 0
        sampling_rvs = []
        for rvs, dist in self.distributions():
            names = [rv.name for rv in rvs]
            if set(names) & set(expr_names):
                new_name = f'__J{i}'
                if len(rvs) > 1:
                    mu = dist.mu.subs(parameters)
                    sigma = dist.sigma.subs(parameters)
                else:
                    mu = dist.mean.subs(parameters)
                    sigma = dist.std.subs(parameters)
                new_rv = sympy.stats.Normal(new_name, mu, sigma)
                sampling_rvs.append((names, new_rv))
        if sampling_rvs:
            # FIXME: Unnecessary to go via DataFrame
            df = pd.DataFrame(index=range(samples))
            for names, new_rv in sampling_rvs:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    if sympy.__version__ == '1.8':
                        cursample = next(
                            sympy.stats.sample(new_rv, library='numpy', size=samples, seed=rng)
                        )
                    else:
                        cursample = sympy.stats.sample(
                            new_rv, library='numpy', size=samples, seed=rng
                        )
                    if len(names) > 1:
                        df[names] = cursample
                    else:
                        df[names[0]] = cursample
            ordered_symbols = list(symbols)
            input_list = [df[symb.name].values for symb in ordered_symbols]
            fn = sympy.lambdify(ordered_symbols, expr, 'numpy')
            a = fn(*input_list)
        else:
            a = np.full(samples, float(expr.evalf()))
        return a

    def _calc_covariance_matrix(self):
        non_altered = []
        means = []
        blocks = []
        names = []
        for rvs, dist in self.distributions():
            names.extend([rv.name for rv in rvs])
            if isinstance(dist, stats.crv_types.NormalDistribution):
                means.append(dist.mean)
                blocks.append(sympy.Matrix([dist.std**2]))
            elif isinstance(dist, stats.joint_rv_types.MultivariateNormalDistribution):
                means.extend(dist.mu)
                blocks.append(dist.sigma)
            else:
                non_altered.extend(rvs)
        if names:
            M = sympy.BlockDiagMatrix(*blocks)
            M = sympy.Matrix(M)
        else:
            M = sympy.Matrix()
        return means, M, names, non_altered

    @property
    def covariance_matrix(self):
        """Covariance matrix of all random variables"""
        _, M, _, others = self._calc_covariance_matrix()
        if others:
            raise ValueError('Only normal distributions are supported')
        return M

    def __repr__(self):
        strings = []
        for rvs, dist in self.distributions():
            strings.append(repr(rvs[0]))
        return '\n'.join(strings)

    def _repr_latex_(self):
        lines = []
        for rvs, dist in self.distributions():
            rv = rvs[0]
            latex = rv._latex_string(aligned=True)
            lines.append(latex)
        return '\\begin{align*}\n' + r' \\ '.join(lines) + '\\end{align*}'

    def parameters_sdcorr(self, values):
        """Convert parameter values to sd/corr form

        All parameter values will be converted to sd/corr assuming
        they are given in var/cov form. Only parameters for normal
        distributions will be affected.

        Parameters
        ----------
        values : dict
            Dict of parameter names to values
        """
        newdict = values.copy()
        for rvs, dist in self.distributions():
            if len(rvs) > 1:
                sigma_sym = dist.sigma
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
                name = (dist.std**2).name
                if name in newdict:
                    newdict[name] = dist.std.subs(values)
        return newdict
