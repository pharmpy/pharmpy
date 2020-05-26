import math

import numpy as np
import sympy.stats

import pharmpy.math
from pharmpy.model import ModelSyntaxError
from pharmpy.parameter import Parameter, ParameterSet
from pharmpy.parse_utils.generic import (AttrToken, AttrTree, insert_after, insert_before_or_at_end,
                                         remove_token_and_space)
from pharmpy.random_variables import JointNormalSeparate, RandomVariables, VariabilityLevel

from .record import Record


class OmegaRecord(Record):
    def parameters(self, start_omega, previous_size):
        """Get a ParameterSet for this omega record
        """
        row = start_omega
        block = self.root.find('block')
        bare_block = self.root.find('bare_block')
        same = bool(self.root.find('same'))
        parameters = ParameterSet()
        if not (block or bare_block):
            for node in self.root.all('diag_item'):
                init = node.init.NUMERIC
                fixed = bool(node.find('FIX'))
                sd = bool(node.find('SD'))
                var = bool(node.find('VAR'))
                n = node.n.INT if node.find('n') else 1
                if sd and var:
                    raise ModelSyntaxError(f'Initial estimate for {self.name.upper} cannot be both'
                                           f' on SD and VAR scale\n{self.root}')
                if init == 0 and not fixed:
                    raise ModelSyntaxError(f'If initial estimate for {self.name.upper} is 0 it'
                                           f' must be set to FIX')
                if sd:
                    init = init ** 2
                for _ in range(n):
                    name = f'{self.name}({row},{row})'
                    param = Parameter(name, init, lower=0, fix=fixed)
                    parameters.add(param)
                    row += 1
            size = 1
            next_omega = row
        else:
            inits = []
            if bare_block:
                size = previous_size
            else:
                size = self.root.block.size.INT
            fix, sd, corr, cholesky = self._block_flags()
            for node in self.root.all('omega'):
                init = node.init.NUMERIC
                n = node.n.INT if node.find('n') else 1
                inits += [init] * n
            if not same:
                if size != pharmpy.math.triangular_root(len(inits)):
                    raise ModelSyntaxError('Wrong number of inits in BLOCK')
                if not cholesky:
                    A = pharmpy.math.flattened_to_symmetric(inits)
                    if corr:
                        for i in range(size):
                            for j in range(size):
                                if i != j:
                                    if sd:
                                        A[i, j] = A[i, i] * A[j, j] * A[i, j]
                                    else:
                                        A[i, j] = math.sqrt(A[i, i]) * math.sqrt(A[j, j]) * A[i, j]
                    if sd:
                        np.fill_diagonal(A, A.diagonal()**2)
                else:
                    L = np.zeros((size, size))
                    inds = np.tril_indices_from(L)
                    L[inds] = inits
                    A = L @ L.T
                for i in range(size):
                    for j in range(0, i + 1):
                        name = f'{self.name}({i + start_omega},{j + start_omega})'
                        init = A[i, j]
                        lower = None if i != j else 0
                        param = Parameter(name, init, lower=lower, fix=fix)
                        parameters.add(param)
            next_omega = start_omega + size
        return parameters, next_omega, size

    def _block_flags(self):
        """Get a tuple of all interesting flags for block
        """
        fix = bool(self.root.find('FIX'))
        var = bool(self.root.find('VAR'))
        sd = bool(self.root.find('SD'))
        cov = bool(self.root.find('COV'))
        corr = bool(self.root.find('CORR'))
        cholesky = bool(self.root.find('CHOLESKY'))
        for node in self.root.all('omega'):
            if node.find('FIX'):
                if fix:
                    raise ModelSyntaxError('Cannot specify option FIX more than once')
                else:
                    fix = True
            if node.find('VAR'):
                if var or sd or cholesky:
                    raise ModelSyntaxError('Cannot specify either option VARIANCE, SD or '
                                           'CHOLESKY more than once')
                else:
                    var = True
            if node.find('SD'):
                if sd or var or cholesky:
                    raise ModelSyntaxError('Cannot specify either option VARIANCE, SD or '
                                           'CHOLESKY more than once')
                else:
                    sd = True
            if node.find('COV'):
                if cov or corr:
                    raise ModelSyntaxError('Cannot specify either option COVARIANCE or '
                                           'CORRELATION more than once')
                else:
                    cov = True
            if node.find('CORR'):
                if corr or cov:
                    raise ModelSyntaxError('Cannot specify either option COVARIANCE or '
                                           'CORRELATION more than once')
                else:
                    corr = True
            if node.find('CHOLESKY'):
                if cholesky or var or sd:
                    raise ModelSyntaxError('Cannot specify either option VARIANCE, SD or '
                                           'CHOLESKY more than once')
                else:
                    cholesky = True
        return fix, sd, corr, cholesky

    def _rv_name(self, num):
        if self.name == 'OMEGA':
            rv_name = 'ETA'
        else:
            rv_name = 'EPS'
        return f'{rv_name}({num})'

    def _rv_vector_name(self, omega_numbers):
        rv_strs = []
        for om in omega_numbers:
            name = self._rv_name(om)
            rv_strs.append(name)
        return '(' + ', '.join(rv_strs) + ')'

    def update(self, parameters, first_omega, previous_size):
        """From a ParameterSet update the OMEGAs in this record
           returns the next omega number
        """
        block = self.root.find('block')
        bare_block = self.root.find('bare_block')
        if not (block or bare_block):
            size = 1
            i = first_omega
            new_nodes = []
            for node in self.root.children:
                if node.rule != 'diag_item':
                    new_nodes.append(node)
                else:
                    sd = bool(node.find('SD'))
                    fix = bool(node.find('FIX'))
                    n = node.n.INT if node.find('n') else 1
                    new_inits = []
                    new_fix = []
                    for j in range(i, i + n):
                        name = f'{self.name}({j},{j})'
                        if not sd:
                            value = float(parameters[name].init)
                        else:
                            value = parameters[name].init ** 0.5
                        new_inits.append(value)
                        new_fix.append(parameters[name].fix)
                    if n == 1 or (new_inits.count(new_inits[0]) == len(new_inits) and
                                  new_fix.count(new_fix[0]) == len(new_fix)):  # All equal?
                        if float(str(node.init)) != new_inits[0]:
                            node.init.tokens[0].value = str(new_inits[0])
                        if new_fix[0] != fix:
                            if new_fix[0]:
                                insert_before_or_at_end(node, 'RPAR', [AttrToken('WS', ' '),
                                                                       AttrToken('FIX', 'FIX')])
                            else:
                                remove_token_and_space(node, 'FIX')
                        new_nodes.append(node)
                    else:
                        # Need to split xn
                        new_children = []
                        for child in node.children:
                            if child.rule != 'n':
                                new_children.append(child)
                        node.children = new_children
                        for j, init in enumerate(new_inits):
                            new_node = AttrTree.transform(node)
                            if float(str(new_node.init)) != init:
                                new_node.init.tokens[0].value = str(init)
                            if new_fix[j] != fix:
                                insert_before_or_at_end(node, 'RPAR', [AttrToken('WS', ' '),
                                                                       AttrToken('FIX', 'FIX')])
                            else:
                                remove_token_and_space(node, 'FIX')
                            new_nodes.append(new_node)
                            if j != len(new_inits) - 1:     # Not the last
                                new_nodes.append(AttrTree.create('ws', {'WS': ' '}))
                    i += n
            self.root.children = new_nodes
            next_omega = i
        else:
            same = bool(self.root.find('same'))
            if same:
                return first_omega + previous_size, previous_size
            size = self.root.block.size.INT
            fix, sd, corr, cholesky = self._block_flags()
            row = first_omega
            col = first_omega
            inits = []
            new_fix = []
            for row in range(first_omega, first_omega + size):
                for col in range(first_omega, row + 1):
                    name = f'{self.name}({row},{col})'
                    inits.append(parameters[name].init)
                    new_fix.append(parameters[name].fix)
            if len(set(new_fix)) != 1:      # Not all true or all false
                raise ValueError('Cannot only fix some parameters in block')

            A = pharmpy.math.flattened_to_symmetric(inits)

            if corr:
                for i in range(size):
                    for j in range(size):
                        if i != j:
                            A[i, j] = A[i, j] / (math.sqrt(A[i, i]) * math.sqrt(A[j, j]))
            if sd:
                np.fill_diagonal(A, A.diagonal()**0.5)

            if cholesky:
                A = np.linalg.cholesky(A)

            inds = np.tril_indices_from(A)
            array = list(A[inds])
            i = 0
            new_nodes = []
            for node in self.root.children:
                if node.rule != 'omega':
                    new_nodes.append(node)
                else:
                    n = node.n.INT if node.find('n') else 1
                    if array[i:i+n].count(array[i]) == n:  # All equal?
                        if float(str(node.init)) != array[i]:
                            node.init.tokens[0].value = str(array[i])
                        new_nodes.append(node)
                    else:
                        # Need to split xn
                        new_children = []
                        for child in node.children:
                            if child.rule not in ['n', 'LPAR', 'RPAR']:
                                new_children.append(child)
                        node.children = new_children
                        for j, init in enumerate(array[i:i+n]):
                            new_node = AttrTree.transform(node)
                            if float(str(new_node.init)) != init:
                                new_node.init.tokens[0].value = str(init)
                            new_nodes.append(new_node)
                            if j != n - 1:     # Not the last
                                new_nodes.append(AttrTree.create('ws', {'WS': ' '}))
                    i += n
            self.root.children = new_nodes
            if new_fix[0] != fix:
                if new_fix[0]:
                    insert_after(self.root, 'block', [AttrToken('WS', ' '),
                                                      AttrToken('FIX', 'FIX')])
                else:
                    remove_token_and_space(self.root, 'FIX', recursive=True)
            next_omega = first_omega + size
        return next_omega, size

    def random_variables(self, start_omega, previous_cov=None):
        """Get a RandomVariableSet for this omega record

           start_omega - the first omega in this record
           previous_sigma - the matrix of the previous omega block
        """
        next_cov = None        # The cov matrix if a block
        block = self.root.find('block')
        bare_block = self.root.find('bare_block')
        zero_fix = []
        if not (block or bare_block):
            rvs = RandomVariables()
            i = start_omega
            numetas = len(self.root.all('diag_item'))
            for node in self.root.all('diag_item'):
                init = node.init.NUMERIC
                fixed = bool(node.find('FIX'))
                name = self._rv_name(i)
                if not (init == 0 and fixed):       # 0 FIX are not RVs
                    eta = sympy.stats.Normal(name, 0, sympy.sqrt(
                        sympy.Symbol(f'{self.name}({i},{i})')))
                    rvs.add(eta)
                else:
                    zero_fix.append(name)
                i += 1
        else:
            if bare_block:
                numetas = previous_cov.rows
            else:
                numetas = self.root.block.size.INT
            same = bool(self.root.find('same'))
            params, _, _ = self.parameters(start_omega, previous_cov.rows if
                                           hasattr(previous_cov, 'rows') else None)
            all_zero_fix = True
            for param in params:
                if not (param.init == 0 and param.fix):
                    all_zero_fix = False
            if all_zero_fix and len(params) > 0 or (previous_cov == 'ZERO' and same):
                names = [self._rv_name(i) for i in range(start_omega, start_omega + numetas)]
                return RandomVariables(), start_omega + numetas, 'ZERO', names
            if numetas > 1:
                names = [self._rv_name(i) for i in range(start_omega, start_omega + numetas)]
                means = [0] * numetas
                if same:
                    rvs = JointNormalSeparate(names, means, previous_cov)
                    next_cov = previous_cov
                else:
                    cov = sympy.zeros(numetas)
                    for row in range(numetas):
                        for col in range(row + 1):
                            cov[row, col] = sympy.Symbol(
                                f'{self.name}({start_omega + row},{start_omega + col})')
                            if row != col:
                                cov[col, row] = cov[row, col]
                    next_cov = cov
                    rvs = JointNormalSeparate(names, means, cov)
            else:
                rvs = RandomVariables()
                name = self._rv_name(start_omega)
                if same:
                    symbol = previous_cov
                else:
                    symbol = sympy.Symbol(f'{self.name}({start_omega},{start_omega})')
                eta = sympy.stats.Normal(name, 0, sympy.sqrt(symbol))
                next_cov = symbol
                rvs.add(eta)

        if self.name == 'OMEGA':
            level = VariabilityLevel.IIV
        else:
            level = VariabilityLevel.RUV
        for rv in rvs:
            rv.variability_level = level

        return rvs, start_omega + numetas, next_cov, zero_fix
