from __future__ import annotations

from collections.abc import Container as CollectionsContainer
from collections.abc import Sequence as CollectionsSequence
from itertools import chain, product
from typing import Container, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, overload

from pharmpy.deps import numpy as np
from pharmpy.deps import symengine, sympy, sympy_stats
from pharmpy.internals.expr.eval import eval_expr
from pharmpy.internals.expr.parse import parse as parse_expr
from pharmpy.internals.expr.subs import subs, xreplace_dict
from pharmpy.internals.immutable import Immutable
from pharmpy.internals.math import cov2corr, is_positive_semidefinite, nearest_postive_semidefinite

from .distributions.numeric import NumericDistribution
from .distributions.symbolic import Distribution, JointNormalDistribution, NormalDistribution


def _create_rng(seed=None) -> np.random.Generator:
    """Create a new random number generator"""
    if isinstance(seed, np.random.Generator):
        return seed
    else:
        return np.random.default_rng(seed)


class VariabilityLevel(Immutable):
    """A variability level

    Parameters
    ----------
    name : str
        A unique identifying name
    reference : bool
        Is this the reference level? Normally IIV would be the reference level
    group : str
        Name of data column to group this level. None for no grouping (default)
    """

    def __init__(self, name: str, reference: bool = False, group: Optional[str] = None):
        self._name = name
        self._reference = reference
        self._group = group

    @classmethod
    def create(cls, name: str, reference=False, group: Optional[str] = None):
        return VariabilityLevel(name, bool(reference), group)

    def replace(self, **kwargs):
        name = kwargs.get('name', self._name)
        reference = kwargs.get('reference', self._reference)
        group = kwargs.get('group', self._group)
        return VariabilityLevel(name, bool(reference), group)

    def __eq__(self, other):
        return (
            isinstance(other, VariabilityLevel)
            and self._name == other._name
            and self._reference == other._reference
            and self._group == other._group
        )

    def __hash__(self):
        return hash((self._name, self._reference, self._group))

    def to_dict(self):
        return {'name': self._name, 'reference': self._reference, 'group': self._group}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @property
    def name(self) -> str:
        """Name of the variability level"""
        return self._name

    @property
    def reference(self) -> bool:
        """Is this the reference level"""
        return self._reference

    @property
    def group(self):
        """Group variable for variability level"""
        return self._group

    def __repr__(self):
        return f"VariabilityLevel({self._name}, reference={self._reference}, group={self._group})"


class VariabilityHierarchy(Immutable):
    """Description of a variability hierarchy"""

    def __init__(self, levels=()):
        self._levels = levels

    @classmethod
    def create(cls, levels=None):
        if levels is None:
            levels = ()
        elif isinstance(levels, VariabilityHierarchy):
            levels = levels._levels
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
            levels = tuple(levels)
        return VariabilityHierarchy(levels)

    def replace(self, **kwargs):
        """Replace properties and create a new VariabilityHierarchy object"""
        levels = kwargs.get('levels', self._levels)
        new = VariabilityHierarchy.create(levels)
        return new

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

    def __hash__(self):
        return hash(self._levels)

    def to_dict(self):
        levels = tuple(level.to_dict() for level in self._levels)
        return {'levels': levels}

    @classmethod
    def from_dict(cls, d):
        levels = tuple(VariabilityLevel.from_dict(vl) for vl in d['levels'])
        return cls(levels=levels)

    def _lookup(self, ind: Union[int, str, VariabilityLevel]) -> VariabilityLevel:
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

    @overload
    def __getitem__(self, ind: Union[Sequence, VariabilityHierarchy]) -> VariabilityHierarchy:
        ...

    @overload
    def __getitem__(self, ind: Union[int, str, VariabilityLevel]) -> VariabilityLevel:
        ...

    def __getitem__(self, ind):
        if isinstance(ind, VariabilityHierarchy):
            levels = [level.name for level in ind._levels]
        elif not isinstance(ind, str) and isinstance(ind, CollectionsSequence):
            levels = ind
        else:
            return self._lookup(ind)
        new = [self._lookup(level) for level in levels]
        return VariabilityHierarchy.create(new)

    def __add__(self, other):
        if isinstance(other, VariabilityLevel):
            levels = (other,)
        else:
            raise ValueError(f"Cannot add {other} to VariabilityHierarchy")
        new = VariabilityHierarchy.create(self._levels + levels)
        return new

    def __radd__(self, other):
        if isinstance(other, VariabilityLevel):
            return VariabilityHierarchy.create((other,) + self._levels)
        else:
            raise ValueError(f"Cannot add {other} to VariabilityLevel")

    @property
    def names(self) -> List[str]:
        """Names of all variability levels"""
        return [varlev.name for varlev in self._levels]

    def _find_reference(self) -> int:
        # Find numerical level of first level
        # No error checking since having a reference level is an invariant
        return next(  # pragma: no cover
            (-i for i, level in enumerate(self._levels) if level.reference)
        )

    @property
    def levels(self):
        """Dictionary of variability level name to numerical level"""
        ind = self._find_reference()
        d = {}
        for level in self._levels:
            d[level.name] = ind
            ind += 1
        return d

    def __len__(self):
        return len(self._levels)

    def __contains__(self, value):
        return value in self.names


class RandomVariables(CollectionsSequence, Immutable):
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
        A list of RandomVariable to add. Default is to create an empty RandomVariables.

    Examples
    --------
    >>> from pharmpy.model import RandomVariables, NormalDistribution, Parameter
    >>> omega = Parameter("OMEGA_CL", 0.1)
    >>> dist = NormalDistribution.create("IIV_CL", "iiv", 0, omega.symbol)
    >>> rvs = RandomVariables.create([dist])
    """

    def __init__(
        self,
        dists: Tuple[Distribution, ...],
        eta_levels: VariabilityHierarchy,
        epsilon_levels: VariabilityHierarchy,
    ):
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
            eta_levels = VariabilityHierarchy((iiv_level, iov_level))
        else:
            if not isinstance(eta_levels, VariabilityHierarchy):
                raise TypeError(
                    f'Type of eta_levels must be a VariabilityHierarchy not a {type(eta_levels)}'
                )

        if epsilon_levels is None:
            ruv_level = VariabilityLevel('RUV', reference=True)
            epsilon_levels = VariabilityHierarchy((ruv_level,))
        else:
            if not isinstance(epsilon_levels, VariabilityHierarchy):
                raise TypeError(
                    f'Type of epsilon_levels must be a VariabilityHierarchy not a {type(epsilon_levels)}'
                )

        return cls(dists, eta_levels, epsilon_levels)

    def replace(self, **kwargs):
        dists = kwargs.get('dists', self._dists)
        eta_levels = kwargs.get('eta_levels', self._eta_levels)
        epsilon_levels = kwargs.get('epsilon_levels', self._epsilon_levels)
        return RandomVariables.create(dists, eta_levels, epsilon_levels)

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

    def __hash__(self):
        return hash((self._dists, self._eta_levels, self._epsilon_levels))

    def to_dict(self):
        dists = tuple(d.to_dict() for d in self._dists)
        return {
            'dists': dists,
            'eta_levels': self._eta_levels.to_dict(),
            'epsilon_levels': self._epsilon_levels.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        eta_levels = VariabilityHierarchy.from_dict(d['eta_levels'])
        epsilon_levels = VariabilityHierarchy.from_dict(d['epsilon_levels'])
        dists = []
        for dist_dict in d['dists']:
            if dist_dict['class'] == 'NormalDistribution':
                dist = NormalDistribution.from_dict(dist_dict)
            else:
                dist = JointNormalDistribution.from_dict(dist_dict)
            dists.append(dist)
        return cls(dists=tuple(dists), eta_levels=eta_levels, epsilon_levels=epsilon_levels)

    def _lookup_rv(self, ind):
        if isinstance(ind, sympy.Symbol):
            ind = ind.name
        if isinstance(ind, str):
            for i, dist in enumerate(self._dists):
                if ind in dist.names:
                    return i, dist
        raise KeyError(f'Could not find {ind} in RandomVariables')

    @overload
    def __getitem__(self, ind: Union[int, str, sympy.Symbol]) -> Distribution:
        ...

    @overload
    def __getitem__(self, ind: Union[slice, Container[str]]) -> RandomVariables:
        ...

    def __getitem__(self, ind):
        if isinstance(ind, int):
            return self._dists[ind]
        elif isinstance(ind, slice):
            return RandomVariables(
                self._dists[ind.start : ind.stop : ind.step], self._eta_levels, self._epsilon_levels
            )
        elif not isinstance(ind, str) and isinstance(ind, CollectionsContainer):
            remove = [name for name in self.names if name not in ind]
            split = self.unjoin(remove)
            keep = tuple(dist for dist in split._dists if dist.names[0] in ind)
            return RandomVariables(keep, self._eta_levels, self._epsilon_levels)
        else:
            _, rv = self._lookup_rv(ind)
            return rv

    def __contains__(self, ind):
        try:
            self._lookup_rv(ind)
            return True
        except KeyError:
            return False

    @property
    def names(self):
        """List of the names of all random variables"""
        return list(chain.from_iterable(dist.names for dist in self._dists))

    @property
    def epsilons(self):
        """Get only the epsilons"""
        return RandomVariables(
            tuple(dist for dist in self._dists if dist.level in self._epsilon_levels.names),
            self._eta_levels,
            self._epsilon_levels,
        )

    @property
    def etas(self):
        """Get only the etas"""
        return RandomVariables(
            tuple(dist for dist in self._dists if dist.level in self._eta_levels.names),
            self._eta_levels,
            self._epsilon_levels,
        )

    @property
    def iiv(self):
        """Get only the iiv etas, i.e. etas with variability level 0"""
        return RandomVariables(
            tuple(dist for dist in self._dists if dist.level == self._eta_levels[0].name),
            self._eta_levels,
            self._epsilon_levels,
        )

    @property
    def iov(self):
        """Get only the iov etas, i.e. etas with variability level 1"""
        return RandomVariables(
            tuple(dist for dist in self._dists if dist.level == self._eta_levels[1].name),
            self._eta_levels,
            self._epsilon_levels,
        )

    @property
    def free_symbols(self):
        """Set of free symbols for all random variables"""
        return set().union(*(dist.free_symbols for dist in self._dists))

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        """List of parameter names for all random variables"""
        params = set().union(*(dist.parameter_names for dist in self._dists))
        return tuple(sorted(map(str, params)))

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
        new_dists = tuple(dist.subs(d) for dist in self._dists)
        return self.replace(dists=new_dists)

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
                            name, dist.level, dist.mean[i], dist.variance[i, i]
                        )
                        newdists.append(new)
                    elif first:  # first of the ones to keep
                        first = False
                        remove = [i for i, n in enumerate(dist.names) if n in inds]
                        if len(dist) - len(remove) == 1:
                            keep = NormalDistribution(
                                name, dist.level, dist.mean[i], dist.variance[i, i]
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
        assert isinstance(joined_rvs, RandomVariables)
        means, M, names = joined_rvs._calc_covariance_matrix()
        cov_to_params = {}
        if fill != 0:
            for row, col in product(range(M.rows), range(M.cols)):
                if M[row, col] == 0:
                    M[row, col] = fill
        elif name_template:
            for row, col in product(range(M.rows), range(M.cols)):
                if M[row, col] == 0 and row > col:
                    param_1, param_2 = M[row, row], M[col, col]
                    assert isinstance(param_names, list)
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
                symb_sigma = dist.variance
                sigma = symb_sigma.subs(dict(parameter_values))
                A = np.array(sigma).astype(np.float64)
                B = nearest_postive_semidefinite(A)
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
            if isinstance(dist, JointNormalDistribution):
                sigma = dist._symengine_variance
                replacement = {}
                for param in dict(parameter_values):
                    replacement[symengine.Symbol(param)] = parameter_values[param]
                sigma = sigma.subs(replacement)
                if not sigma.free_symbols:  # Cannot validate since missing params
                    a = np.array(sigma).astype(np.float64)
                    if not is_positive_semidefinite(a):
                        return False
                else:
                    raise TypeError("Cannot validate parameters since all are not numeric")
        return True

    def sample(self, expr, parameters=None, samples=1, rng=None):
        """Sample from the distribution of expr

        parameters in the distributions will first be replaced"""

        sympified_expr = parse_expr(expr)
        xreplace_parameters = {} if parameters is None else xreplace_dict(parameters)

        return _sample_from_distributions(
            self,
            sympified_expr,
            xreplace_parameters,
            samples,
            _create_rng(rng),
        )

    def _calc_covariance_matrix(self):
        means = []
        names = []
        n = 0
        for dist in self._dists:
            names.extend(dist.names)
            n += len(dist)

        M = sympy.zeros(n)

        if not names:
            return means, M, names

        row = 0
        col = 0
        for dist in self._dists:
            if isinstance(dist, NormalDistribution):
                means.append(dist.mean)
                M[row, col] = dist.variance
                row += 1
                col += 1
            else:  # isinstance(dist, JointNormalDistribution):
                means.extend(dist.mean)
                var = dist.variance
                for i in range(var.rows):
                    for j in range(var.cols):
                        M[row + i, col + j] = var[i, j]
                row += var.rows
                col += var.cols
        return means, M, names

    @property
    def covariance_matrix(self):
        """Covariance matrix of all random variables"""
        _, M, _ = self._calc_covariance_matrix()
        return M

    def __repr__(self):
        return '\n'.join(map(repr, self._dists))

    def _repr_latex_(self):
        lines = (dist.latex_string(aligned=True) for dist in self._dists)
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
                corr = cov2corr(sigma)
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

    def get_rvs_with_same_dist(self, rv):
        """Gets random variables with the same distribution as input random variable

        The resulting RandomVariables objects includes the input random variable.

        Parameters
        ----------
        rv : str
            Name of random variable

        Returns
        -------
        RandomVariables
            RandomVariables object with all distributions as input random variable (including input)
        """
        _, dist_input = self._lookup_rv(rv)

        rvs = [dist for dist in self if dist.variance == dist_input.variance]

        return RandomVariables.create(rvs)

    def replace_with_sympy_rvs(self, expr):
        """Replaces Pharmpy RVs in a Sympy expression with Sympy RVs

        Takes a Sympy expression and replaces all RVs with Sympy RVs, resulting expression
        can be used in different Sympy functions (e.g. sympy.stats.std())

        Parameters
        ----------
        expr : sympy.Expr
            Expression which will get RVs replaced

        Returns
        -------
        sympy.Expr
            Expression with replaced RVs
        """
        rvs_in_expr = {self[rv] for rv in expr.free_symbols.intersection(self.free_symbols)}
        subs_dict = {}
        for i, rv in enumerate(rvs_in_expr):
            if isinstance(rv, JointNormalDistribution):
                # sympy.stats.MultivariateNormal uses variance, sympy.stats.Normal takes std
                dist = sympy.stats.MultivariateNormal(f'__rv{i}', rv.mean, rv.variance)
                for j in range(0, len(rv.names)):
                    subs_dict[rv.names[j]] = dist[j]
            else:
                subs_dict[rv.names[0]] = sympy_stats.Normal(
                    f'__rv{i}', rv.mean, sympy.sqrt(rv.variance)
                )
        return expr.subs(subs_dict)


def _sample_from_distributions(distributions, expr, parameters, nsamples, rng):
    random_variable_symbols = expr.free_symbols.difference(parameters.keys())
    filtered_distributions = filter_distributions(distributions, random_variable_symbols)
    sampling_rvs = subs_distributions(filtered_distributions, parameters)
    sampled_expr = subs(expr, parameters, simultaneous=True)
    return sample_expr_from_rvs(sampling_rvs, sampled_expr, nsamples, rng)


def filter_distributions(
    distributions: Iterable[Distribution], symbols: Set[sympy.Symbol]
) -> Iterable[Distribution]:
    covered_symbols = set()

    for dist in distributions:
        rvs_covered_by_dist = tuple(rv for rv in dist.names if sympy.Symbol(rv) in symbols)
        if rvs_covered_by_dist:
            yield dist[rvs_covered_by_dist]
            covered_symbols.update(sympy.Symbol(rv) for rv in rvs_covered_by_dist)

    if covered_symbols != symbols:
        raise ValueError('Could not cover all requested symbols with given distributions')


def subs_distributions(
    distributions: Iterable[Distribution], parameters: Dict[sympy.Symbol, float]
) -> Iterable[Tuple[Tuple[sympy.Symbol, ...], NumericDistribution]]:
    for dist in distributions:
        rvs_symbols = tuple(map(sympy.Symbol, dist.names))
        numeric_distribution = dist.evalf(parameters)
        yield (rvs_symbols, numeric_distribution)


def sample_expr_from_rvs(
    sampling_rvs: Iterable[Tuple[Tuple[sympy.Symbol, ...], NumericDistribution]],
    expr: sympy.Expr,
    nsamples: int,
    rng,
):
    samples = sample_rvs(sampling_rvs, nsamples, rng)
    return eval_expr(expr, nsamples, samples)


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
