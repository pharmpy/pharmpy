import math
import re
import warnings

import numpy as np
import sympy
import sympy.stats

import pharmpy.math
from pharmpy.model import ModelSyntaxError
from pharmpy.parameter import Parameter, Parameters
from pharmpy.parse_utils.generic import (
    AttrToken,
    AttrTree,
    insert_after,
    insert_before_or_at_end,
    remove_token_and_space,
)
from pharmpy.random_variables import RandomVariable, RandomVariables
from pharmpy.symbols import symbol

from .parsers import OmegaRecordParser
from .record import Record


class OmegaRecord(Record):
    def parameters(self, start_omega, previous_size, seen_labels=None):
        """Get a Parameters for this omega record"""
        row = start_omega
        block = self.root.find('block')
        bare_block = self.root.find('bare_block')
        same = bool(self.root.find('same'))
        parameters = Parameters()
        coords = []

        try:
            self.comment_map
        except AttributeError:
            self.comment_map = dict()

        if seen_labels is None:
            seen_labels = set()
        if not (block or bare_block):
            for node in self.root.all('diag_item'):
                init = node.init.NUMERIC
                fixed = bool(node.find('FIX'))
                sd = bool(node.find('SD'))
                var = bool(node.find('VAR'))
                n = node.n.INT if node.find('n') else 1
                if sd and var:
                    raise ModelSyntaxError(
                        f'Initial estimate for {self.name.upper()} cannot be both'
                        f' on SD and VAR scale\n{self.root}'
                    )
                if init == 0 and not fixed:
                    raise ModelSyntaxError(
                        f'If initial estimate for {self.name.upper()} is 0 it'
                        f' must be set to FIX'
                    )
                if sd:
                    init = init ** 2
                for _ in range(n):
                    name = self._find_label(node, seen_labels)
                    comment = self._get_name(node)
                    if not name:
                        name = f'{self.name}({row},{row})'
                    if comment:
                        self.comment_map[name] = comment
                    seen_labels.add(name)
                    coords.append((row, row))
                    param = Parameter(name, init, lower=0, fix=fixed)
                    parameters.append(param)
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
            labels = []
            for node in self.root.all('omega'):
                init = node.init.NUMERIC
                n = node.n.INT if node.find('n') else 1
                inits += [init] * n
                name = self._find_label(node, seen_labels)
                if name is not None:
                    seen_labels.add(name)
                labels.append(name)
                if n > 1:
                    labels.extend([None] * (n - 1))
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
                        np.fill_diagonal(A, A.diagonal() ** 2)
                else:
                    L = np.zeros((size, size))
                    inds = np.tril_indices_from(L)
                    L[inds] = inits
                    A = L @ L.T
                label_index = 0
                for i in range(size):
                    for j in range(0, i + 1):
                        name = labels[label_index]
                        if name is None:
                            name = f'{self.name}({i + start_omega},{j + start_omega})'
                        coords.append((i + start_omega, j + start_omega))
                        init = A[i, j]
                        lower = None if i != j else 0
                        param = Parameter(name, init, lower=lower, fix=fix)
                        parameters.append(param)
                        label_index += 1
            next_omega = start_omega + size
        try:
            self.name_map
        except AttributeError:
            self.name_map = {name: c for i, (name, c) in enumerate(zip(parameters.names, coords))}
        return parameters, next_omega, size

    def _find_label(self, node, seen_labels):
        """Find label from comment of omega parameter"""
        # needed to avoid circular import with Python 3.6
        import pharmpy.plugins.nonmem as nonmem

        name = None
        if 'comment' in nonmem.conf.parameter_names:
            name = self._get_name(node)
            if name in seen_labels:
                warnings.warn(
                    f'The parameter name {name} is duplicated. Falling back to basic NONMEM names.'
                )
                name = None
        return name

    def _get_name(self, node):
        name = None
        found = False
        for subnode in self.root.tree_walk():
            if id(subnode) == id(node):
                found = True
                continue
            if found and (subnode.rule == 'omega' or subnode.rule == 'diag_item'):
                break
            if found and (subnode.rule == 'NEWLINE' or subnode.rule == 'COMMENT'):
                m = re.search(r';\s*([a-zA-Z_]\w*)', str(subnode))
                if m:
                    name = m.group(1)
                    break
        return name

    def _block_flags(self):
        """Get a tuple of all interesting flags for block"""
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
                    raise ModelSyntaxError(
                        'Cannot specify either option VARIANCE, SD or ' 'CHOLESKY more than once'
                    )
                else:
                    var = True
            if node.find('SD'):
                if sd or var or cholesky:
                    raise ModelSyntaxError(
                        'Cannot specify either option VARIANCE, SD or ' 'CHOLESKY more than once'
                    )
                else:
                    sd = True
            if node.find('COV'):
                if cov or corr:
                    raise ModelSyntaxError(
                        'Cannot specify either option COVARIANCE or ' 'CORRELATION more than once'
                    )
                else:
                    cov = True
            if node.find('CORR'):
                if corr or cov:
                    raise ModelSyntaxError(
                        'Cannot specify either option COVARIANCE or ' 'CORRELATION more than once'
                    )
                else:
                    corr = True
            if node.find('CHOLESKY'):
                if cholesky or var or sd:
                    raise ModelSyntaxError(
                        'Cannot specify either option VARIANCE, SD or ' 'CHOLESKY more than once'
                    )
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

    def _get_param_name(self, row, col):
        """Get the name of the current OMEGA/SIGMA at (row, col)"""
        invmap = {value: key for key, value in self.name_map.items()}
        name = invmap[(row, col)]
        return name

    def update(self, parameters, first_omega, previous_size):
        """From a Parameters update the OMEGAs in this record
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
                        name = self._get_param_name(j, j)
                        if not sd:
                            value = float(parameters[name].init)
                        else:
                            value = parameters[name].init ** 0.5
                        value = int(value) if value.is_integer() else value
                        new_inits.append(value)
                        new_fix.append(parameters[name].fix)
                    if n == 1 or (
                        new_inits.count(new_inits[0]) == len(new_inits)
                        and new_fix.count(new_fix[0]) == len(new_fix)
                    ):  # All equal?
                        if float(str(node.init)) != new_inits[0]:
                            node.init.tokens[0].value = str(new_inits[0])
                        if new_fix[0] != fix:
                            if new_fix[0]:
                                insert_before_or_at_end(
                                    node, 'RPAR', [AttrToken('WS', ' '), AttrToken('FIX', 'FIX')]
                                )
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
                                insert_before_or_at_end(
                                    node, 'RPAR', [AttrToken('WS', ' '), AttrToken('FIX', 'FIX')]
                                )
                            else:
                                remove_token_and_space(node, 'FIX')
                            new_nodes.append(new_node)
                            if j != len(new_inits) - 1:  # Not the last
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
            inits = []
            new_fix = []
            for row in range(first_omega, first_omega + size):
                for col in range(first_omega, row + 1):
                    name = self._get_param_name(row, col)
                    inits.append(parameters[name].init)
                    new_fix.append(parameters[name].fix)
            if len(set(new_fix)) != 1:  # Not all true or all false
                raise ValueError('Cannot only fix some parameters in block')

            A = pharmpy.math.flattened_to_symmetric(inits)

            if corr:
                for i in range(size):
                    for j in range(size):
                        if i != j:
                            A[i, j] = A[i, j] / (math.sqrt(A[i, i]) * math.sqrt(A[j, j]))
            if sd:
                np.fill_diagonal(A, A.diagonal() ** 0.5)

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
                    if array[i : i + n].count(array[i]) == n:  # All equal?
                        if float(str(node.init)) != array[i]:
                            value = int(array[i]) if float(array[i]).is_integer() else array[i]
                            node.init.tokens[0].value = str(value)
                        new_nodes.append(node)
                    else:
                        # Need to split xn
                        new_children = []
                        for child in node.children:
                            if child.rule not in ['n', 'LPAR', 'RPAR']:
                                new_children.append(child)
                        node.children = new_children
                        for j, init in enumerate(array[i : i + n]):
                            new_node = AttrTree.transform(node)
                            if float(str(new_node.init)) != init:
                                init = int(init) if float(init).is_integer() else init
                                new_node.init.tokens[0].value = str(init)
                            new_nodes.append(new_node)
                            if j != n - 1:  # Not the last
                                new_nodes.append(AttrTree.create('ws', {'WS': ' '}))
                    i += n
            self.root.children = new_nodes
            if new_fix[0] != fix:
                if new_fix[0]:
                    insert_after(
                        self.root, 'block', [AttrToken('WS', ' '), AttrToken('FIX', 'FIX')]
                    )
                else:
                    remove_token_and_space(self.root, 'FIX', recursive=True)
            next_omega = first_omega + size
        return next_omega, size

    def random_variables(self, start_omega, previous_cov=None):
        """Get a RandomVariableSet for this omega record

        start_omega - the first omega in this record
        previous_cov - the matrix of the previous omega block
        """
        same = bool(self.root.find('same'))
        if not hasattr(self, 'name_map') and not same:
            if isinstance(previous_cov, sympy.Symbol):
                prev_size = 1
            elif previous_cov is not None:
                prev_size = len(previous_cov)
            else:
                prev_size = None
            self.parameters(start_omega, prev_size)
        if hasattr(self, 'name_map'):
            rev_map = {value: key for key, value in self.name_map.items()}
        next_cov = None  # The cov matrix if a block
        block = self.root.find('block')
        bare_block = self.root.find('bare_block')
        zero_fix = []
        etas = []
        if not (block or bare_block):
            rvs = RandomVariables()
            i = start_omega
            numetas = len(self.root.all('diag_item'))
            for node in self.root.all('diag_item'):
                init = node.init.NUMERIC
                fixed = bool(node.find('FIX'))
                name = self._rv_name(i)
                if not (init == 0 and fixed):  # 0 FIX are not RVs
                    eta = RandomVariable.normal(name, 'iiv', 0, symbol(rev_map[(i, i)]))
                    rvs.append(eta)
                    etas.append(eta.name)
                else:
                    zero_fix.append(name)
                    etas.append(name)
                i += 1
        else:
            if bare_block:
                numetas = previous_cov.rows
            else:
                numetas = self.root.block.size.INT
            params, _, _ = self.parameters(
                start_omega, previous_cov.rows if hasattr(previous_cov, 'rows') else None
            )
            all_zero_fix = True
            for param in params:
                if not (param.init == 0 and param.fix):
                    all_zero_fix = False
            if all_zero_fix and len(params) > 0 or (previous_cov == 'ZERO' and same):
                names = [self._rv_name(i) for i in range(start_omega, start_omega + numetas)]
                self.eta_map = {eta: start_omega + i for i, eta in enumerate(names)}
                return RandomVariables(), start_omega + numetas, 'ZERO', names
            if numetas > 1:
                names = [self._rv_name(i) for i in range(start_omega, start_omega + numetas)]
                means = [0] * numetas
                if same:
                    rvs = RandomVariable.joint_normal(names, 'iiv', means, previous_cov)
                    etas = [rv.name for rv in rvs]
                    next_cov = previous_cov
                else:
                    cov = sympy.zeros(numetas)
                    for row in range(numetas):
                        for col in range(row + 1):
                            cov[row, col] = symbol(rev_map[(start_omega + row, start_omega + col)])
                            if row != col:
                                cov[col, row] = cov[row, col]
                    next_cov = cov
                    rvs = RandomVariable.joint_normal(names, 'iiv', means, cov)
                    etas = [rv.name for rv in rvs]
            else:
                rvs = RandomVariables()
                name = self._rv_name(start_omega)
                if same:
                    sym = previous_cov
                else:
                    sym = symbol(rev_map[(start_omega, start_omega)])
                eta = RandomVariable.normal(name, 'iiv', 0, sym)
                next_cov = sym
                rvs.append(eta)
                etas.append(eta.name)

        if self.name == 'OMEGA':
            if same:
                level = 'IOV'
            else:
                level = 'IIV'
        else:
            level = 'RUV'
        for rv in rvs:
            rv.level = level
        self.eta_map = {eta: start_omega + i for i, eta in enumerate(etas)}
        return rvs, start_omega + numetas, next_cov, zero_fix

    def renumber(self, new_start):
        old_start = min(self.eta_map.values())
        if new_start != old_start:
            for name in self.eta_map:
                self.eta_map[name] += new_start - old_start
            for name in self.name_map:
                old_row, old_col = self.name_map[name]
                self.name_map[name] = (
                    old_row + new_start - old_start,
                    old_col + new_start - old_start,
                )

    def remove(self, names):
        """Remove some etas from block given eta names"""
        first_omega = min(self.eta_map.values())
        indices = {self.eta_map[name] - first_omega for name in names}
        for name in names:
            del self.eta_map[name]

        block = self.root.find('block')
        bare_block = self.root.find('bare_block')
        same = bool(self.root.find('same'))
        if not (block or bare_block):
            keep = []
            i = 0
            for node in self.root.children:
                if node.rule == 'diag_item':
                    if i not in indices:
                        keep.append(node)
                    i += 1
                else:
                    keep.append(node)
            self.root.children = keep
        elif same and not bare_block:
            self.root.block.size.INT = len(self) - len(indices)
        elif block:
            fix, sd, corr, cholesky = self._block_flags()
            inits = []
            for node in self.root.all('omega'):
                init = node.init.NUMERIC
                n = node.n.INT if node.find('n') else 1
                inits += [init] * n
            A = pharmpy.math.flattened_to_symmetric(inits)
            A = np.delete(A, list(indices), axis=0)
            A = np.delete(A, list(indices), axis=1)
            s = f' BLOCK({len(A)})'
            if fix:
                s += ' FIX'
            if sd:
                s += ' SD'
            if corr:
                s += ' CORR'
            if cholesky:
                s += ' CHOLESKY'
            s += '\n'
            for row in range(0, len(A)):
                s += ' '.join(np.atleast_1d(A[row, 0 : (row + 1)]).astype(str)) + '\n'
            parser = OmegaRecordParser(s)
            self.root = parser.root

    def update_name_map(self, trans):
        """Update name_map given dict from -> to"""
        for key, value in trans.items():
            if key in self.name_map:
                n = self.name_map[key]
                del self.name_map[key]
                self.name_map[value] = n

    def __len__(self):
        return len(self.eta_map)
