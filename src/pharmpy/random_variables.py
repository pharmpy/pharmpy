import enum
import itertools

import numpy as np
import symengine
import sympy
import sympy.stats as stats
from sympy.stats.rv import RandomSymbol

import pharmpy.math

from .data_structures import OrderedSet


class VariabilityLevel(enum.Enum):
    """ Representation of a variability level

       currently supports only IIV and RUV.
       Can be changed into something not-enum
       in future keeping the IIV and RUV as
       class singletons
    """
    IIV = enum.auto()
    RUV = enum.auto()


class JointDistributionSeparate(RandomSymbol):
    """One random variable in a joint distribution

       sympy can currently only represent the random variables in a joint distribution
       as one single indexed variable that cannot be separately named. This class
       makes separation possible.

       This class can probably not solve all issues with joint rvs, but it can at least
       handle seprate symbols, pass as a random variable for random_variables and lead back
       to its pspace.
    """
    def __new__(cls, name, joint_symbol):
        return super().__new__(cls, sympy.Symbol(name), joint_symbol.pspace)


def JointNormalSeparate(names, mean, cov):
    """Conveniently create a joint normal distribution and create separate random variables
    """
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
    def _left_parens(height):
        """Return an array containing each row of a large parenthesis
           used for pretty printing
        """
        a = ['⎧']
        for _ in range(height - 2):
            a.append('⎪')
        a.append('⎩')
        return a

    @staticmethod
    def _right_parens(height):
        """Return an array containing each row of a large parenthesis
           used for pretty printing
        """
        a = ['⎫']
        for _ in range(height - 2):
            a.append('⎪')
        a.append('⎭')
        return a

    @staticmethod
    def _normal_definition_string(rv):
        """Provide a array of pretty strings for the definition of a Normal random variable
           This should ideally be available from sympy.
        """
        dist = rv.pspace.distribution
        return [f'{sympy.pretty(rv, wrap_line=False)}'
                f'~ {RandomVariables.nletter}({sympy.pretty(dist.mean, wrap_line=False)}, '
                f'{sympy.pretty(dist.std**2, wrap_line=False)})']

    @staticmethod
    def _joint_normal_definition_string(rvs):
        """Provide an array of pretty strings for the definition of a Joint Normal random variable
        """
        dist = rvs[0].pspace.distribution
        name_vector = sympy.Matrix(rvs)
        name_strings = sympy.pretty(name_vector, wrap_line=False).split('\n')
        mu_strings = sympy.pretty(dist.mu, wrap_line=False).split('\n')
        sigma_strings = sympy.pretty(dist.sigma, wrap_line=False).split('\n')
        mu_height = len(mu_strings)
        sigma_height = len(sigma_strings)
        max_height = max(mu_height, sigma_height)

        left_parens = RandomVariables._left_parens(len(name_strings))
        right_parens = RandomVariables._right_parens(len(name_strings))

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
        enumerator = enumerate(zip(name_strings, left_parens, mu_strings, sigma_strings,
                                   right_parens))
        for i, (name_line, lpar, mu_line, sigma_line, rpar) in enumerator:
            if i == central_index:
                res.append(name_line + f' ~ {RandomVariables.nletter}' + lpar +
                           mu_line + ', ' + sigma_line + rpar)
            else:
                res.append(name_line + '     ' + lpar + mu_line +
                           '  ' + sigma_line + rpar)

        return res

    def __getitem__(self, index):
        if isinstance(index, int):
            for i, e in enumerate(self):
                if i == index:
                    return e
        else:
            for e in self:
                if e == index or e.name == index:
                    return e
        raise KeyError(f'Random variable "{index}" does not exist')

    def __repr__(self):
        """ Give a nicely formatted view of the definitions of all
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

    def all_parameters(self):
        params = set()
        for _, dist in self.distributions():
            params |= dist.free_symbols
        return sorted([str(p) for p in params])

    def distributions(self, level=None):
        """Iterate with one entry per distribution instead of per random variable.

           level - only iterate over random variables of this variability level
        """
        i = 0
        while i < len(self):
            rv = self[i]
            dist = rv.pspace.distribution
            if isinstance(dist, stats.crv_types.NormalDistribution):
                i += 1
                if level is None or level == rv.variability_level:
                    yield [rv], dist
            else:       # Joint Normal
                n = self[i].pspace.distribution.sigma.rows
                rvs = [self[k] for k in range(i, i + n)]
                i += n
                if level is None or level == rv.variability_level:
                    yield rvs, dist

    def iiv_variance_parameters(self):
        parameters = []
        for rvs, dist in self.distributions(level=VariabilityLevel.IIV):
            if len(rvs) == 1:
                parameters.append(dist.std ** 2)
            else:
                parameters += list(dist.sigma.diagonal())
        return parameters

    def merge_normal_distributions(self, fill=0):
        """Merge all normal distributed rvs together into one joint normal

           Set new covariances (and previous 0 covs) to 'fill'
        """
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
            if len(blocks) > 1:
                # Need special case for len(blocks) == 1 because of sympy 1.5.1 bug #18618
                M = sympy.BlockDiagMatrix(*blocks)
                M = sympy.Matrix(M)
            else:
                M = sympy.Matrix(blocks[0])
            if fill != 0:
                for row, col in itertools.product(range(M.rows), range(M.cols)):
                    if M[row, col] == 0:
                        M[row, col] = fill
            new_rvs = JointNormalSeparate(names, means, M)
            self.__init__(new_rvs + non_altered)

    def copy(self):
        # Special copy because separated joints need special treatment
        new_rvs = RandomVariables()
        for rvs, dist in self.distributions():
            if len(rvs) == 1:
                new_rvs.add(rvs[0].copy())
            else:
                new_rvs.update(JointNormalSeparate([rv.name for rv in rvs], dist.mu, dist.sigma))
        return rvs

    def validate_parameters(self, parameter_values, use_cache=False):
        """ Validate a dict or Series of parameter values

            Currently checks that all covariance matrices are posdef
            use_cache for using symengine cached matrices
        """
        if use_cache and not hasattr(self, '_cached_sigmas'):
            self._cached_sigmas = {}
            for rvs, dist in self.distributions():
                if len(rvs) > 1:
                    sigma = dist.sigma
                    a = [[symengine.Symbol(e.name) for e in sigma.row(i)]
                         for i in range(sigma.rows)]
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
                    if not sigma.free_symbols:      # Cannot validate since missing params
                        a = np.array(sigma).astype(np.float64)
                        if not pharmpy.math.is_posdef(a):
                            return False
        return True

    def nearest_valid_parameters(self, parameter_values):
        """ Force parameter values into being valid

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
