import enum
import itertools
import warnings

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


class VariabilityLevel(enum.Enum):
    """Representation of a variability level

    currently supports only IIV and RUV.
    Can be changed into something not-enum
    in future keeping the IIV and RUV as
    class singletons
    """

    IIV = enum.auto()
    IOV = enum.auto()
    RUV = enum.auto()


class JointDistributionSeparate(RandomSymbol):
    """One random variable in a joint distribution

    sympy can currently only represent the random variables in a joint distribution
    as one single indexed variable that cannot be separately named. This class
    makes separation possible.

    This class can probably not solve all issues with joint rvs, but it can at least
    handle separate symbols, pass as a random variable for random_variables and lead back
    to its pspace.
    """

    def __new__(cls, name, joint_symbol):
        return super().__new__(cls, symbol(name), joint_symbol.pspace)


def JointNormalSeparate(names, mean, cov):
    """Conveniently create a joint normal distribution and create separate random variables"""
    x = stats.Normal('__DUMMY__', mean, cov)
    rvs = [JointDistributionSeparate(name, x) for name in names]
    return rvs


class RandomVariables(OrderedSet):
    """An ordered set of random variables

    currently separately named jointrvs are not supported in sympy
    (i.e. it is not possible to do [eta1, eta2] = Normal(...))
    Use JointDistributionSeparate as a workaround
    Joints must come in the correct order
    """

    nletter = '𝒩 '

    @staticmethod
    def _normal_definition_string(rv):
        """Provide a array of pretty strings for the definition of a Normal random variable
        This should ideally be available from sympy.
        """
        dist = rv.pspace.distribution
        return [
            f'{sympy.pretty(rv, wrap_line=False)}'
            f'~ {RandomVariables.nletter}({sympy.pretty(dist.mean, wrap_line=False)}, '
            f'{sympy.pretty(dist.std**2, wrap_line=False)})'
        ]

    @staticmethod
    def _joint_normal_definition_string(rvs):
        """Create an array of pretty strings for the definition of a Joint Normal random variable"""
        dist = rvs[0].pspace.distribution
        name_vector = sympy.Matrix(rvs)
        name_strings = sympy.pretty(name_vector, wrap_line=False).split('\n')
        mu_strings = sympy.pretty(dist.mu, wrap_line=False).split('\n')
        sigma_strings = sympy.pretty(dist.sigma, wrap_line=False).split('\n')
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
                    + f' ~ {RandomVariables.nletter}'
                    + lpar
                    + mu_line
                    + ', '
                    + sigma_line
                    + rpar
                )
            else:
                res.append(name_line + '     ' + lpar + mu_line + '  ' + sigma_line + rpar)

        return res

    def __getitem__(self, index):
        if isinstance(index, int):
            for i, e in enumerate(self):
                if i == index:
                    return e
        else:
            for e in self:
                if (
                    isinstance(index, str)
                    and e.name == index
                    or not isinstance(index, str)
                    and e.name == index.name
                ):
                    return e
        raise KeyError(f'Random variable "{index}" does not exist')

    def __repr__(self):
        """Give a nicely formatted view of the definitions of all
        random variables.
        """
        res = ''
        for rvs, dist in self.distributions():
            if isinstance(dist, stats.crv_types.NormalDistribution):
                lines = RandomVariables._normal_definition_string(rvs[0])
            elif isinstance(dist, stats.joint_rv_types.MultivariateNormalDistribution):
                lines = RandomVariables._joint_normal_definition_string(rvs)
            res += '\n'.join(lines) + '\n'
        return res

    def _repr_latex_(self):
        lines = []
        for rvs, dist in self.distributions():
            if isinstance(dist, stats.joint_rv_types.MultivariateNormalDistribution):
                rv_vec = sympy.Matrix(rvs)._repr_latex_()[1:-1]
                mean_vec = dist.mu._repr_latex_()[1:-1]
                sigma = dist.sigma._repr_latex_()[1:-1]
                latex = rv_vec + r' & \sim \mathcal{N} \left(' + mean_vec + ',' + sigma + r'\right)'
            else:
                rv = rvs[0]._repr_latex_()[1:-1]
                mean = dist.mean._repr_latex_()[1:-1]
                sigma = (dist.std ** 2)._repr_latex_()[1:-1]
                latex = rv + r' & \sim  \mathcal{N} \left(' + mean + ',' + sigma + r'\right)'
            lines.append(latex)
        return '\\begin{align*}\n' + r' \\ '.join(lines) + '\\end{align*}'

    @property
    def free_symbols(self):
        symbs = set()
        for rv in self:
            free = {s for s in rv.pspace.free_symbols if s.name != rv.name}
            symbs |= free
            symbs.add(symbol(rv.name))
        return symbs

    def all_parameters(self):
        params = set()
        dists = self.distributions()
        for _, dist in dists:
            params |= dist.free_symbols
        return sorted([str(p) for p in params])

    def get_eta_params(self, eta_name):
        params = set()
        for rvs, dist in self.distributions():
            if eta_name in [rv for rv in rvs]:
                params |= dist.free_symbols
        return sorted([str(p) for p in params])

    def distributions(self, level=None, exclude_level=None):
        """List with one entry per distribution instead of per random variable.

        Parameters
        ----------
        level
            Only iterate over random variables of this variability level
        exclude_level
            Iterate over random variables of all other variability levels
        """
        distributions = []
        i = 0
        while i < len(self):
            rv = self[i]
            dist = rv.pspace.distribution
            if isinstance(dist, stats.crv_types.NormalDistribution):
                i += 1
                if (level is None or level == rv.variability_level) and (
                    exclude_level is None or exclude_level != rv.variability_level
                ):
                    distributions.append(([rv], dist))
            else:  # Joint Normal
                n = self[i].pspace.distribution.sigma.rows
                rvs = [self[k] for k in range(i, i + n)]
                i += n
                if (level is None or level == rv.variability_level) and (
                    exclude_level is None or exclude_level != rv.variability_level
                ):
                    distributions.append((rvs, dist))
        return distributions

    def get_rvs_from_same_dist(self, rv):
        """Get all RVs from same distribution as input rv.

        Parameters
        ----------
        rv : RandomSymbol
            Random symbol to find associated rvs for (i.e. rvs with same distribution)."""
        joined_rvs = []

        for rvs, dist in self.distributions():
            if dist == rv.pspace.distribution:
                joined_rvs += rvs

        return RandomVariables(joined_rvs)

    def extract_from_block(self, rv_to_extract):
        """
        Extracts single random variable from joint distribution and creates new distribution.
        A new distribution will be created for remaining RVs, single normal distribution if one
        remains, joint normal otherwise.

        Parameters
        ----------
        rv_to_extract : RandomSymbol
            Random symbol to create new single distribution for."""
        associated_rvs = self.get_rvs_from_same_dist(rv_to_extract)

        cov = associated_rvs.covariance_matrix()
        rv_extracted = None
        index_to_remove = None
        names = []

        for i, rv in enumerate(associated_rvs):
            if rv.name == rv_to_extract.name:
                rv_extracted = stats.Normal(rv.name, 0, sympy.sqrt(cov[i, i]))
                rv_extracted.variability_level = VariabilityLevel.IIV
                index_to_remove = i
            else:
                names.append(rv.name)

        cov.row_del(index_to_remove)
        cov.col_del(index_to_remove)

        if len(cov) == 1:
            rv_remaining = stats.Normal(names[0], 0, sympy.sqrt(cov[0]))
            rv_remaining.variability_level = VariabilityLevel.IIV
        else:
            means = sympy.zeros(cov.shape[0] - 1)
            rv_remaining = JointNormalSeparate(names, means, cov)

            for rv in rv_remaining:
                rv.variability_level = VariabilityLevel.IIV

        split_block = [rv_extracted, rv_remaining]
        split_block = split_block.reverse() if index_to_remove != 0 else split_block

        rvs_new = RandomVariables()

        has_added_changed_block = False

        for rv in self:
            if rv in associated_rvs:
                if not has_added_changed_block:
                    {rvs_new.add(rv_block) for rv_block in split_block}
                    has_added_changed_block = True
            else:
                rvs_new.add(rv)
            self.discard(rv)

        self.update(rvs_new)

        return rv_extracted

    def are_consecutive(self, subset):
        """Determines if subset has same order as full set (self). If totally different,
        False will be returned."""
        rvs_self = sum([rvs[0] for rvs in self.distributions()], [])
        rvs_subset = sum([rvs[0] for rvs in subset.distributions()], [])

        i = 0
        for rv in rvs_self:
            if rv.name == rvs_subset[i].name:
                if i == len(rvs_subset) - 1:
                    return True
                i += 1
            elif i > 0:
                return False
        return False

    @property
    def ruv_rvs(self):
        """Get list of all ruv random variables (epsilons)"""
        ruv = []
        for rv in self:
            if rv.variability_level == VariabilityLevel.RUV:
                ruv.append(rv)
        return ruv

    @property
    def etas(self):
        """Get list of all eta random variables"""
        etas = []
        for rv in self:
            if rv.variability_level != VariabilityLevel.RUV:
                etas.append(rv)
        return etas

    def variance_parameters(self, unique=True, level=None, exclude_level=None):
        parameters = []
        for rvs, dist in self.distributions(level=level, exclude_level=exclude_level):
            if len(rvs) == 1:
                parameters.append(dist.std ** 2)
            else:
                parameters += list(dist.sigma.diagonal())
        if unique:
            parameters = list(OrderedSet(parameters))  # Only unique in order
        return parameters

    def _calc_covariance_matrix(self, ruv=False):
        non_altered = []
        means = []
        blocks = []
        names = []
        if ruv:
            dists = self.distributions(level=VariabilityLevel.RUV)
        else:
            dists = self.distributions(exclude_level=VariabilityLevel.RUV)
        for rvs, dist in dists:
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

    def covariance_matrix(self, ruv=False):
        """Covariance matrix of all random variables

        currently only supports normal distribution
        """
        _, M, _, others = self._calc_covariance_matrix(ruv=ruv)
        if others:
            raise ValueError('Only normal distributions are supported')
        return M

    def merge_normal_distributions(self, fill=0, create_cov_params=False):
        """Merge all normal distributed rvs together into one joint normal

        Set new covariances (and previous 0 covs) to 'fill'
        """
        cov_to_params = dict()
        cov_number = 1
        means, M, names, others = self._calc_covariance_matrix()
        if fill != 0:
            for row, col in itertools.product(range(M.rows), range(M.cols)):
                if M[row, col] == 0:
                    M[row, col] = fill
        elif create_cov_params:
            for row, col in itertools.product(range(M.rows), range(M.cols)):
                if M[row, col] == 0 and row > col:
                    param_1 = M[row, row]
                    param_2 = M[col, col]
                    cov_name = f'COV{cov_number}'
                    cov_number += 1
                    cov_to_params[cov_name] = (str(param_1), str(param_2))

                    M[row, col] = symbol(cov_name)
                    M[col, row] = symbol(cov_name)

        new_rvs = JointNormalSeparate(names, means, M)
        self.__init__(new_rvs + others)
        return cov_to_params

    def __getstate__(self):
        """Serialization methods needed to handle variability_level on random variables"""
        var_levels = [rv.variability_level for rv in self]
        return {'self': list(self), 'var_levels': var_levels}

    def __setstate__(self, d):
        OrderedSet.__init__(self)
        for rv, var_level in zip(d['self'], d['var_levels']):
            rv.variability_level = var_level
            self.add(rv)

    def __deepcopy__(self, memo):
        return self.copy()

    def copy(self):
        # Special copy because separated joints need special treatment
        new_rvs = RandomVariables()
        for rvs, dist in self.distributions():
            if len(rvs) == 1:
                rv = rvs[0].copy()
                rv.variability_level = rvs[0].variability_level
                new_rvs.add(rv)
            else:
                cp = JointNormalSeparate([rv.name for rv in rvs], dist.mu, dist.sigma)
                for new, old in zip(cp, rvs):
                    new.variability_level = old.variability_level
                new_rvs.update(cp)
        return new_rvs

    def validate_parameters(self, parameter_values, use_cache=False):
        """Validate a dict or Series of parameter values

        Currently checks that all covariance matrices are posdef
        use_cache for using symengine cached matrices
        """
        if use_cache and not hasattr(self, '_cached_sigmas'):
            self._cached_sigmas = {}
            for rvs, dist in self.distributions():
                if len(rvs) > 1:
                    sigma = dist.sigma
                    a = [
                        [symengine.Symbol(e.name) for e in sigma.row(i)] for i in range(sigma.rows)
                    ]
                    A = symengine.Matrix(a)
                    self._cached_sigmas[rvs[0]] = A

        for rvs, dist in self.distributions():
            if len(rvs) > 1:
                if not use_cache:
                    sigma = dist.sigma.subs(dict(parameter_values))
                    # Switch to numpy here. Sympy posdef check is problematic
                    # see https://github.com/sympy/sympy/issues/18955
                    if not sigma.free_symbols:
                        a = np.array(sigma).astype(np.float64)
                        if not pharmpy.math.is_posdef(a):
                            return False
                else:
                    sigma = self._cached_sigmas[rvs[0]]
                    replacement = {}
                    # Following because https://github.com/symengine/symengine/issues/1660
                    for param in dict(parameter_values):
                        replacement[symengine.Symbol(param)] = parameter_values[param]
                    sigma = sigma.subs(replacement)
                    if not sigma.free_symbols:  # Cannot validate since missing params
                        a = np.array(sigma).astype(np.float64)
                        if not pharmpy.math.is_posdef(a):
                            return False
        return True

    def nearest_valid_parameters(self, parameter_values):
        """Force parameter values into being valid

        As small changes as possible

        returns an updated parameter_values
        """
        nearest = parameter_values.copy()
        for rvs, dist in self.distributions():
            if len(rvs) > 1:
                sigma = dist.sigma.subs(dict(parameter_values))
                A = np.array(sigma).astype(np.float64)
                B = pharmpy.math.nearest_posdef(A)
                if B is not A:
                    for row in range(len(A)):
                        for col in range(row + 1):
                            nearest[dist.sigma[row, col].name] = B[row, col]
        return nearest

    def expression(self, expr, parameters):
        """Replace all symbols with same names as rvs with the corresponding rvs
        or indexed variables for joint distributions and replace parameter values
        """
        d = dict()
        i = 1
        for rvs, dist in self.distributions():
            if len(rvs) > 1:
                joint_name = f'__J{i}'
                mu = dist.mu.subs(parameters)
                sigma = dist.sigma.subs(parameters)
                x = stats.Normal(joint_name, mu, sigma)
                d.update({symbol(rv.name): x[n] for n, rv in enumerate(rvs)})
                i += 1
            else:
                mean = dist.mean.subs(parameters)
                std = dist.std.subs(parameters)
                d[symbol(rvs[0].name)] = stats.Normal(rvs[0].name, mean, std)
        return expr.subs(d)

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

    def get_connected_iovs(self, iov):
        iovs = []
        connected = False
        for rv in self:
            if rv == iov:
                iovs.append(rv)
                connected = True
            elif (
                rv.variability_level == VariabilityLevel.IOV
                and rv.pspace.distribution == iov.pspace.distribution
            ):
                iovs.append(rv)
            elif connected:
                break
        return iovs


# pharmpy sets a parametrization attribute to the sympy distributions
# It will currently not affect the sympy distribution itself just convey the information
# The distribution class itself need to change to accomodate pdf etc.
# Could have different distributions for different parametrization, but would like to be able
# to convert between them. Ideally also support arbitrary parametrizations.
# Classes will be used by ModelfitResults to be able to reparametrize results without
# reparametrizing the whole model.

# For now simply use these for the results object to set proper parametrization of parameter.


class NormalParametrizationVariance:
    name = 'variance'
    distribution = stats.crv_types.NormalDistribution

    def __init__(self, mean, variance):
        pass


class NormalParametrizationSd:
    name = 'sd'
    distribution = stats.crv_types.NormalDistribution

    def __init__(self, mean, sd):
        pass


class MultivariateNormalParametrizationCovariance:
    name = 'covariance'
    distribution = stats.joint_rv_types.MultivariateNormalDistribution

    def __init__(self, mean, variance):
        pass


class MultivariateNormalParametrizationSdCorr:
    name = 'sdcorr'
    distribution = stats.joint_rv_types.MultivariateNormalDistribution

    def __init__(self, mean, sd, corr):
        pass
