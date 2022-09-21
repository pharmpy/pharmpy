from abc import abstractmethod
from collections.abc import Collection

import pharmpy.unicode as unicode
from pharmpy.deps import symengine, sympy
from pharmpy.expressions import subs, sympify


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

    @abstractmethod
    def __getitem__(self, index):
        pass


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

    def __getitem__(self, index):
        if isinstance(index, int):
            if index == 0:
                return self
            raise IndexError(index)

        elif isinstance(index, str):
            if index == self._names[0]:
                return self

        else:
            if isinstance(index, slice):
                index = range(index.start, index.stop, index.step)

            if isinstance(index, Collection):
                if len(index) == 1 and (self._names[0] in index or 0 in index):
                    return self

        raise KeyError(index)

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

    def __getitem__(self, index):
        if isinstance(index, int):
            if -len(self) <= index < len(self):
                cls = NormalDistribution
                names = (self._names[index],)
            else:
                raise IndexError(index)

        elif isinstance(index, str):
            cls = NormalDistribution
            names = (index,)
            try:
                index = self._names.index(index)
            except ValueError:
                raise KeyError(names[0])

        else:
            if isinstance(index, slice):
                index = range(index.start, index.stop, index.step)

            if isinstance(index, Collection):
                if len(index) == 0 or len(index) > len(self._names):
                    raise KeyError(index)

                collection = set(index)
                if not collection.issubset(self._names):
                    raise KeyError(index)

                if len(collection) == len(self._names):
                    return self

                index = []
                names = []

                for i, name in enumerate(self._names):
                    if name in collection:
                        index.append(i)
                        names.append(name)

                if len(index) == 1:
                    cls = NormalDistribution
                    index = index[0]

                else:
                    cls = JointNormalDistribution

            else:
                raise KeyError(index)

        level = self._level
        mean = self._mean[index, [0]] if isinstance(index, int) else self._mean[index]
        variance = self._variance[index][index]
        return cls(names, level, mean, variance)

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
