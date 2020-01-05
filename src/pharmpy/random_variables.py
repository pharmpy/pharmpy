import sympy
import sympy.stats as stats

from .data_structures import OrderedSet


class RandomVariables(OrderedSet):
    """A set of random variables
    """
    @staticmethod
    def _rv_definition_string(rv):
        """ Return a pretty string of the definition of a random variable
        This should ideally be available from sympy.
        Currently supports Normal and JointNormal
        """
        dist = rv.pspace.distribution
        if isinstance(dist, stats.crv_types.NormalDistribution):
            return [f'{sympy.pretty(rv)} ~ 𝓝({sympy.pretty(dist.mean)}, {sympy.pretty(dist.std)})']
        elif isinstance(dist, stats.joint_rv_types.MultivariateNormalDistribution):
            mu_strings = sympy.pretty(dist.mu).split('\n')
            sigma_strings = sympy.pretty(dist.sigma).split('\n')
            mu_height = len(mu_strings)
            sigma_height = len(sigma_strings)
            max_height = max(mu_height, sigma_height)

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
            symbol_padding = ' ' * len(f'{sympy.pretty(rv)} ~ ')
            for i, (mu_line, sigma_line) in enumerate(zip(mu_strings, sigma_strings)):
                if i == central_index:
                    res.append(f'{sympy.pretty(rv)} ~ 𝓝\N{SIX-PER-EM SPACE}(' +
                               mu_line + ', ' + sigma_line + ')')
                else:
                    res.append(symbol_padding + '    ' + mu_line + '  ' + sigma_line + ' ')

            return res

    def __getitem__(self, index):
        for e in self:
            if e == index or e.name == index:
                return e
        raise KeyError(f'Random variable "{index}" does not exist')

    def __str__(self):
        """ Give a nicely formatted view of the definitions of all
            random variables.
        """
        res = ''
        for rv in self:
            lines = RandomVariables._rv_definition_string(rv)
            res += '\n'.join(lines) + '\n'
        return res
