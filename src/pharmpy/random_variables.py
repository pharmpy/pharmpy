import copy
import enum
import itertools
import warnings
from collections.abc import MutableSequence, Iterable

import numpy as np
import pandas as pd
import symengine
import sympy
import sympy.stats as stats
from sympy.stats.rv import RandomSymbol

import pharmpy.math
import pharmpy.unicode as unicode
from pharmpy.symbols import symbol

from .data_structures import OrderedSet


class RandomVariable:
    """A single random variable

    Example
    -------

    Parameters
    ----------
    name : str
        Name of the random variable
    level : str
        Name of the variability level. The default levels are IIV, IOV and RUV
    sympy_rv : sympy.RandomSymbol
        RandomSymbol to use for this random variable. See also the normal
        and joint_normal classmethods.
    """
    def __init__(self, name, level, sympy_rv=None):
        level = RandomVariable._canonicalize_level(level)
        self._name = name
        self.level = level
        self.symbol = symbol(name)
        self._sympy_rv = sympy_rv
        self._mean = None
        self._variance = None
        self._symengine_variance = None
        self._joint_names = None

    def __eq__(self, other):
        return self.name == other.name and self.level == other.level and self._mean == other._mean and self._variance == other._variance and self._sympy_rv == other._sympy_rv

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
        >>> rv = RandomVariable.normal("IIV_CL", 0, omega.symbol)
        """
        rv = cls(name, level)
        rv._mean = sympy.Matrix([sympy.sympify(mean)])
        rv._variance = sympy.Matrix([sympy.sympify(variance)])
        rv._symengine_variance = symengine.sympify(rv._variance)
        return rv

    @classmethod
    def joint_normal(cls, names, level, mu, sigma):
        """Create joint normally distributed random variables

        Parameters
        ----------
        names : list
            Names of the random variables
        mu : matrix or list
            Vector of the means of the random variables
        sigma : matrix or list of lists
            Covariance matrix of the random variables

        Example
        -------
        >>> from pharmpy import RandomVariables, Parameter
        >>> omega_cl = Parameter("OMEGA_CL", 0.1)
        >>> omega_v = Parameter("OMEGA_V", 0.1)
        >>> corr_cl_v = Parameter("OMEGA_CL_V", 0.01)
        >>> RandomVariable.joint_normal(["IIV_CL", "IIV_V"], [0, 0], [[omega_cl.symbol, corr_cl_v], [corr_cl_v, omega_v]])
        """

        mean = sympy.Matrix(mu)
        variance = sympy.Matrix(sigma)
        if variance.is_positive_semidefinite is False:
            raise ValueError(f'Sigma matrix is not positive semidefinite')
        rvs = []
        for name in names:
            rv = cls(name, level)
            rv._mean = mean
            rv._variance = variance
            rv._symengine_variance = symengine.Matrix(variance.rows, variance.cols, sigma)
            rv._joint_names = names
            rvs.append(rv)
        return rvs

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if self._joint_names:
            index = self._joint_names.index(value)
            self._joint_names[index] = name
        self._name = name

    @property
    def joint_names(self):
        """Get the names of all (including this) jointly varying rvs
        """
        return [] if not self._joint_names else self._joint_names

    @property
    def sympy_rv(self):
        """Get the corresponding sympy random variable
        """
        if self._sympy_rv is None:
            # Normal distribution that might have 0 variance
            if self._variance.is_zero:
                return sympy.Integer(0)
            elif self._mean.rows > 1:
                return sympy.stats.Normal('X', self._mean, self._variance)
            else:
                return sympy.stats.Normal(self.name, self._mean[0], sympy.sqrt(self._variance[0]))
        else:
            return self._sympy_rv

    @property
    def free_symbols(self):
        if self._mean is not None:
            return {self.symbol} | self._mean.free_symbols | self._variance.free_symbols
        else:
            free = {s for s in rv.pspace.free_symbols if s.name != rv.name}
            return free | {self.symbol}

    @property
    def parameter_names(self):
        if self._mean is not None:
            params = self._mean.free_symbols | self._variance.free_symbols
        else:
            params = {s for s in rv.pspace.free_symbols if s.name != rv.name}
        return [p.name for p in params]

    def subs(self, d):
        if self._mean is not None:
            self._mean = self._mean.subs(d)
            self._variance = self._variance.subs(d)
            self._symengine_variance = symengine.Matrix(self._variance.rows, self._variance.cols, self._variance)
        if self._sympy_rv is not None:
            self._sympy_rv = self._sympy_rv.subs(d)

    def copy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        symengine_variance = self._symengine_variance
        self._symengine_variance = None
        method = self.__deepcopy__
        # Trick to use default deepcopy
        self.__deepcopy__ = None
        new = copy.deepcopy(self)
        self.__deepcopy__ = method
        new._symengine_variance = symengine.sympify(self._variance)
        self._symengine_variance = symengine_variance
        return new

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        if self._mean is not None:    # Normal distribution
            if self._mean.rows > 1:
                name_vector = sympy.Matrix(self._joint_names)
                name_strings = sympy.pretty(name_vector, wrap_line=False).split('\n')
                mu_strings = sympy.pretty(self._mean, wrap_line=False).split('\n')
                sigma_strings = sympy.pretty(self._variance, wrap_line=False).split('\n')
                mu_height = len(mu_strings)
                sigma_height = len(sigma_strings)
                max_height = max(mu_height, sigma_height)

                left_parens = unicode.left_parens(len(name_strings))
                right_parens = unicode.right_parens(len(name_strings))

                # Pad the smaller of the matrices
                if mu_height != sigma_height:
                    to_pad = mu_strings if mu_strings < sigma_strings else sigma_strings
                    num_lines = abs(mu_height - sigma_height)
                    padding = ' ' * len(to_pad[0])
                    for i in range(0, num_lines):
                        if i // 2 == 0:
                            to_pad.append(padding)
                        else:
                            to_pad.insert(0, padding)

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
                        res.append(name_line + '     ' + lpar + mu_line + '  ' + sigma_line + rpar)
                return '\n'.join(res) + '\n'
            else:
                return f'{sympy.pretty(self.symbol, wrap_line=False)}' \
                f' ~ {unicode.mathematical_script_capital_n}({sympy.pretty(self._mean[0], wrap_line=False)}, ' \
                f'{sympy.pretty(self._variance[0], wrap_line=False)})\n'

    def _latex_string(self, aligned=False):
        lines = []
        if aligned:
            align_str = ' & '
        else:
            align_str = ''
        if self._mean.rows > 1:
            rv_vec = sympy.Matrix(self._joint_names)._repr_latex_()[1:-1]
            mean_vec = self._mean._repr_latex_()[1:-1]
            sigma = self._variance._repr_latex_()[1:-1]
            latex = rv_vec + align_str + r'\sim \mathcal{N} \left(' + mean_vec + ',' + sigma + r'\right)'
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
    def __init__(self, name, level, group):
        self.name = name
        self.level = level
        self.group = group


class VariabilityHierarchy:
    def __init__(self):
        self._levels = []

    @property
    def names(self):
        return [varlev.name for varlev in self._levels]

    @property
    def levels(self):
        return [varlev.level for varlev in self._levels]

    def get_name(self, i):
        for varlev in self._levels:
            if varlev.level == i:
                return varlev.name
        raise KeyError(f'No variability level {i}')

    def add_variability_level(self, name, level, group):
        nums = self.levels
        new = VariabilityLevel(name, level, group)
        if nums:
            if not (level == min(nums) - 1 or level == max(nums) + 1):
                raise ValueError(f'Cannot set variability level {self.level}. '
                    'New variability level must be one level higher or one level lower than any current level')
            if level == min(nums) - 1:
                self._levels.insert(0, new)
            else:
                self._levels.append(new)
        else:
            self._levels.append(new)

    def add_higher_level(self, name, group):
        nums = self.levels
        level = max(nums) + 1
        self.add_variability_level(name, level, group)

    def add_lower_level(self, name, group):
        nums = [varlev.level for varlev in self._levels]
        level = min(nums) + 1
        self.add_variability_level(name, level, group)

    def set_variability_level(self, level, name, group):
        """Change the name and group of variability level
        """
        for varlev in self._levels:
            if varlev.level == level:
                varlev.name = name
                varlev.group = group
                break
        else:
            raise KeyError(f'No variability level {level}')

    def remove_variability_level(self, ind):
        """Remove a variability level

        Parameters
        ----------
        ind : str or int
            name or number of variability level
        """
        for i, varlev in enumerate(self._levels):
            if isinstance(ind, str) and varlev.name == ind or isinstance(ind, int) and varlev.level == ind:
                index = i
                break
        else:
            raise KeyError(f'No variability level {ind}')
        if index == 0:
            raise ValueError(f'Cannot remove the base variability level (0)')
        del self._levels[index]
        for varlev in self._levels:
            if index < 0 and varlevel.level < index:
                varlev.level += 1
            elif index > 0 and varlevel.level > index:
                varlev.level -= 1

    def __len__(self):
        return len(self._levels)


class RandomVariables(MutableSequence):
    """A collection of random variables

        Describe default levels here
    """

    def __init__(self, rvs=None):
        if isinstance(rvs, RandomVariables):
            self._rvs = copy.deepcopy(rvs._rvs)
        elif rvs is None:
            self._rvs = []
        else:
            self._rvs = list(rvs)
        eta_levels = VariabilityHierarchy()
        eta_levels.add_variability_level('IIV', 0, 'ID')
        eta_levels.add_higher_level('IOV', 'OCC')
        epsilon_levels = VariabilityHierarchy()
        epsilon_levels.add_variability_level('RUV', 0, None)
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
            i = self._rvs.index(ind)
            return i, ind
        if insert:
            # Must allow for inserting after last element.
            return ind, None
        else:
            return ind, self._rvs[ind]

    def __getitem__(self, ind):
        _, rv = self._lookup_rv(ind)
        return rv

    def _remove_joint_normal(self, rv):
        joint_names = rv._joint_names
        if joint_names is not None:
            joint_index = joint_names.index(rv.name)
            for name in joint_names:
                if name == rv.name:
                    continue
                other = self[name]
                del other._joint_names[joint_index]
                other._mean.row_del(joint_index)
                other._variance.row_del(joint_index)
                other._variance.col_del(joint_index)
                other._symengine_variance = symengine.sympify(other._variance)

    def __setitem__(self, ind, value):
        if isinstance(ind, slice):
            # FIXME: This is too crude
            self._rvs[ind] = value
            return
        if not isinstance(value, RandomVariable):
            raise ValueError(f'Trying to set {type(value)} to RandomVariables. Must be of type RandomVariable.')
        i, rv = self._lookup_rv(ind)
        if rv._joint_names is not None:
            self._remove_joint_normal(rv)
        self._rvs[i] = rv

    def __delitem__(self, ind):
        i, rv = self._lookup_rv(ind)
        joint_names = rv._joint_names
        if joint_names is not None:
            joint_index = joint_names.index(rv.name)
            for name in joint_names:
                other = self[name]
                del other._joint_names[joint_index]
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
            raise ValueError(f'Trying to insert {type(value)} into RandomVariables. Must be of type RandomVariable.')
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
        return RandomVariables([rv for rv in self._rvs if rv.level == self._eta_levels.get_name(0)])

    @property
    def iov(self):
        """Get only the iov etas, i.e. etas with variability level 1"""
        return RandomVariables([rv for rv in self._rvs if rv.level == self._eta_levels.get_name(1)])

    @property
    def free_symbols(self):
        """Set of free symbols for all random variables"""
        symbs = set()
        for rv in self._rvs:
            symbs |= rv.free_symbols
        return symbs

    def copy(self):
        new = RandomVariables()
        for rv in self._rvs:
            new._rvs.append(rv.copy())
        return new

    @property
    def parameter_names(self):
        params = set()
        for rv in self:
            params |= set(rv.parameter_names)
        return sorted([str(p) for p in params])

    @property
    def variance_parameters(self):
        parameters = []
        for rvs, dist in self.distributions():
            if len(rvs) == 1:
                parameters.append(dist.std ** 2)
            else:
                parameters += list(dist.sigma.diagonal())
        return [p.name for p in parameters]

    def subs(self, d):
        s = dict()
        for key, value in d.items():
            if key in self.names:
                self[key].name = value
            else:
                s[key] = value
        for rv in self._rvs:
            rv.subs(s)

    def remove_covariance(self, ind):
        # FIXME: Call disjoin? dejoin? unjoin?
        """Remove all covariances the random variable has with other random variables

        """
        i, rv = self._lookup_rv(ind)
        index = rv._joint_names.index(rv.name)
        del self[i]
        rv._mean = sympy.Matrix([rv._mean[index, index]])
        rv._variance = sympy.Matrix([rv._variance[index, index]])
        rv._symengine_variance = symengine.sympify(rv._variance)
        rv._joint_names = None
        rv.insert(i - index, rv)

    def join(self, inds, fill=0, name_template=None, param_names=None):
        """Join random variables together into one joint distribution

        Set new covariances (and previous 0 covs) to 'fill'
        """
        cov_to_params = dict()
        selection = RandomVariables([self[ind] for ind in inds])
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

        new = []
        first = True
        for rv in self._rvs:
            if rv in selection:
                if first:
                    new.extend(selection._rvs)
                    first = False
            else:
                new.append(rv)

        new_rvs = RandomVariable.joint_normal(names, 'iiv', means, M)
        for rv, new_rv in zip(selection, new_rvs):
            rv._sympy_rv = new_rv._sympy_rv
            rv._mean = sympy.Matrix(means)
            rv._variance = M
            rv._symengine_variance = symengine.Matrix(M.rows, M.cols, M)
            rv._joint_names = [rv.name for rv in new_rvs]
        self._rvs = new
        return cov_to_params

    def distributions(self):
        """List with one entry per distribution instead of per random variable.
        """
        distributions = []
        i = 0
        while i < len(self):
            rv = self[i]
            symrv = rv.sympy_rv
            n = 0 if rv._joint_names is None else len(rv._joint_names)
            if symrv == 0:
                if n == 1:
                    # Workaround beause sympy disallows 0 std and sigma
                    symrv = sympy.stats.Normal(rv.name, rv._mean, 9999)
                    symrv.pspace.distribution.std = 0
                    symrv.pspace.distribution.args = (rv._mean, 0)
                else:
                    symrv = sympy.stats.Normal('X', rv._mean, sympy.eyes(n))
                    symrv.pspace.distribution.sigma = sympy.zeroes(n)
                    symrv.pspace.distribution.args = (rv._mean, sympy.zeroes(n))

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

        returns an updated parameter_values
        """
        nearest = parameter_values.copy()
        for rvs, dist in self.distributions():
            if len(rvs) > 1:
                symb_sigma = rvs[0]._variance
                sigma = symb_sigma.subs(dict(parameter_values))
                A = np.array(sigma).astype(np.float64)
                B = pharmpy.math.nearest_posdef(A)
                if B is not A:
                    for row in range(len(A)):
                        for col in range(row + 1):
                            nearest[symb_sigma[row, col].name] = B[row, col]
        return nearest

    def validate_parameters(self, parameter_values):
        """Validate a dict or Series of parameter values

        Currently checks that all covariance matrices are posdef
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
                    if not pharmpy.math.is_posdef(a):
                        return False
        return True

    def sample(self, expr, parameters=None, samples=1):
        """Sample from the distribution of expr

        parameters in the distriutions will first be replaced"""
        expr = sympy.sympify(expr)
        if not parameters:
            parameters = dict()
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
        df = pd.DataFrame(index=range(samples))
        # FIXME: Unnecessary to go via DataFrame
        for names, new_rv in sampling_rvs:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                cursample = next(sympy.stats.sample(new_rv, library='numpy', size=samples))
                if len(names) > 1:
                    df[names] = cursample
                else:
                    df[names[0]] = cursample
        ordered_symbols = list(symbols)
        input_list = [df[symb.name].values for symb in ordered_symbols]
        fn = sympy.lambdify(ordered_symbols, expr, 'numpy')
        a = fn(*input_list)
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
                blocks.append(sympy.Matrix([dist.std ** 2]))
            elif isinstance(dist, stats.joint_rv_types.MultivariateNormalDistribution):
                means.extend(dist.mu)
                blocks.append(dist.sigma)
            else:
                non_altered.extend(rvs)
        if names:
            M = sympy.BlockDiagMatrix(*blocks)
            M = sympy.Matrix(M)
        return means, M, names, non_altered

    @property
    def covariance_matrix(self):
        """Covariance matrix of all random variables

        currently only supports normal distribution
        """
        _, M, _, others = self._calc_covariance_matrix()
        if others:
            raise ValueError('Only normal distributions are supported')
        return M

    def __repr__(self):
        res = ''
        for rvs, dist in self.distributions():
            res += repr(rvs[0])
        return res

    def _repr_latex_(self):
        lines = []
        for rvs, dist in self.distributions():
            rv = rvs[0]
            latex = rv._latex_string(aligned=True)
            lines.append(latex)
        return '\\begin{align*}\n' + r' \\ '.join(lines) + '\\end{align*}'

#    def get_rvs_from_same_dist(self, rv):
#        """Get all RVs from same distribution as input rv.
#
#        Parameters
#        ----------
#        rv : RandomSymbol
#            Random symbol to find associated rvs for (i.e. rvs with same distribution)."""
#        joined_rvs = []

#        for rvs, _ in self.distributions():
#            if rv.name in [rv_dist.name for rv_dist in rvs]:
#                joined_rvs += rvs
#
#        return RandomVariables(joined_rvs)
#
#    def are_consecutive(self, subset):
#        """Determines if subset has same order as full set (self)."""
#        rvs_self = sum([rvs[0] for rvs in self.distributions()], [])
#        rvs_subset = sum([rvs[0] for rvs in subset.distributions()], [])
#
#        i = 0
#        for rv in rvs_self:
#            if rv.name == rvs_subset[i].name:
#                if i == len(rvs_subset) - 1:
#                    return True
#                i += 1
#            elif i > 0:
#                return False
#        return False
#


#
#    def get_connected_iovs(self, iov):
#        iovs = []
#        connected = False
#        for rv in self:
#            if rv == iov:
#                iovs.append(rv)
#                connected = True
#            elif (
#                rv.variability_level == VariabilityLevel.IOV
#                and rv.pspace.distribution == iov.pspace.distribution
#            ):
#                iovs.append(rv)
#            elif connected:
#                break
#        return iovs
