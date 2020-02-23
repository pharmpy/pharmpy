import itertools

import sympy
import sympy.stats as stats
from sympy.matrices import MatrixBase, MatrixExpr
from sympy.stats.rv import RandomSymbol

from .data_structures import OrderedSet


class MultivariateNormalDistribution(stats.joint_rv_types.MultivariateNormalDistribution):
    """A joint normal distribution that does not check for posdef.
       Needed because of sympy issue #18625
    """
    @staticmethod
    def check(mu, sigma):
        if mu.shape[0] != sigma.shape[0]:
            raise ValueError("Size of the mean vector and covariance matrix are incorrect.")


def Normal(name, mean, std):
    """sympy function patched to use our MultivariateNormalDistribution
    """
    if isinstance(mean, (list, MatrixBase, MatrixExpr)) and isinstance(std, (list, MatrixBase,
                                                                             MatrixExpr)):
        return stats.joint_rv_types.multivariate_rv(MultivariateNormalDistribution,
                                                    name, mean, std)
    return stats.crv_types.rv(name, stats.crv_types.NormalDistribution, (mean, std))


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
    x = Normal('__DUMMY__', mean, cov)
    rvs = [JointDistributionSeparate(name, x) for name in names]
    return rvs


class RandomVariables(OrderedSet):
    """An ordered set of random variables

       currently separately named jointrvs are not supported in sympy
       (i.e. it is not possible to do [eta1, eta2] = Normal(...))
       Use JointDistributionSeparate as a workaround
       Joints must come in the correct order
    """
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
        return [f'{sympy.pretty(rv)} ~ 𝓝({sympy.pretty(dist.mean)}, '
                f'{sympy.pretty(dist.std**2)})']

    @staticmethod
    def _joint_normal_definition_string(rvs):
        """Provide an array of pretty strings for the definition of a Joint Normal random variable
        """
        dist = rvs[0].pspace.distribution
        name_vector = sympy.Matrix(rvs)
        name_strings = sympy.pretty(name_vector).split('\n')
        mu_strings = sympy.pretty(dist.mu).split('\n')
        sigma_strings = sympy.pretty(dist.sigma).split('\n')
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
                res.append(name_line + f' ~ 𝓝\N{SIX-PER-EM SPACE}' + lpar + '\N{SIX-PER-EM SPACE}' +
                           mu_line + ', ' + sigma_line + rpar)
            else:
                res.append(name_line + '     ' + lpar + '\N{SIX-PER-EM SPACE}' + mu_line +
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

    def distributions(self):
        """Iterate with one entry per distribution instead of per random variable.
        """
        i = 0
        while i < len(self):
            rv = self[i]
            dist = rv.pspace.distribution
            if isinstance(dist, stats.crv_types.NormalDistribution):
                i += 1
                yield [rv], dist
            else:       # Joint Normal
                rvs = [x for x in self if x.pspace.distribution == dist]
                i += len(rvs)
                yield rvs, dist

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
