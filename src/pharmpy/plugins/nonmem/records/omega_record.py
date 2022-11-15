import math
import re
import warnings
from typing import List, Literal, Optional, Tuple, Union, cast

from pharmpy.deps import numpy as np
from pharmpy.deps import sympy
from pharmpy.internals.math import flattened_to_symmetric, triangular_root
from pharmpy.internals.parse import AttrToken, AttrTree
from pharmpy.internals.parse.generic import (
    eval_token,
    insert_after,
    insert_before_or_at_end,
    remove_token_and_space,
)
from pharmpy.model import (
    JointNormalDistribution,
    ModelSyntaxError,
    NormalDistribution,
    Parameter,
    RandomVariables,
)

from .parsers import OmegaRecordParser
from .record import Record


class OmegaRecord(Record):
    def parameters(self, start_omega, previous_size, seen_labels=None):
        """Get a Parameters for this omega record"""
        row = start_omega
        block = self.root.find('block')
        bare_block = self.root.find('bare_block')
        same = bool(self.root.find('same'))
        parameters = []
        coords = []

        try:
            self.comment_map
        except AttributeError:
            self.comment_map = {}

        if seen_labels is None:
            seen_labels = set()
        if not (block or bare_block):
            for node in self.root.subtrees('diag_item'):
                init = cast(float, eval_token(node.subtree('init').leaf('NUMERIC')))
                fixed = bool(node.find('FIX'))
                sd = bool(node.find('SD'))
                var = bool(node.find('VAR'))
                n = cast(int, eval_token(node.subtree('n').leaf('INT'))) if node.find('n') else 1
                if sd and var:
                    name = '(anonymous)' if self.name is None else self.name.upper()
                    raise ModelSyntaxError(
                        f'Initial estimate for {name} cannot be both'
                        f' on SD and VAR scale\n{self.root}'
                    )
                if init == 0 and not fixed:
                    name = '(anonymous)' if self.name is None else self.name.upper()
                    raise ModelSyntaxError(
                        f'If initial estimate for {name} is 0 it must be set to FIX'
                    )
                if sd:
                    init = init**2
                for _ in range(n):
                    name = self._find_label(node, seen_labels)
                    comment = self._get_name(node)
                    if not name:
                        name = f'{self.name}({row},{row})'
                    if comment:
                        self.comment_map[name] = comment
                    seen_labels.add(name)
                    coords.append((row, row))
                    param = Parameter.create(name, init, lower=0, fix=fixed)
                    parameters.append(param)
                    row += 1
            size = 1
            next_omega = row
        else:
            inits = []
            if bare_block:
                size = previous_size
            else:
                size = cast(int, eval_token(self.root.subtree('block').subtree('size').leaf('INT')))
            fix, sd, corr, cholesky = self._block_flags()
            labels, comments = [], []
            for node in self.root.subtrees('omega'):
                init = cast(float, eval_token(node.subtree('init').leaf('NUMERIC')))
                n = cast(int, eval_token(node.subtree('n').leaf('INT'))) if node.find('n') else 1
                inits += [init] * n
                name = self._find_label(node, seen_labels)
                comment = self._get_name(node)
                if name is not None:
                    seen_labels.add(name)
                labels.append(name)
                comments.append(comment)
                if n > 1:
                    labels.extend([None] * (n - 1))
                    comments.extend([None] * (n - 1))
            if not same:
                if size != triangular_root(len(inits)):
                    raise ModelSyntaxError('Wrong number of inits in BLOCK')
                if not cholesky:
                    A = flattened_to_symmetric(inits)
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
                        comment = comments[label_index]
                        if name is None:
                            name = f'{self.name}({i + start_omega},{j + start_omega})'
                        if comment is not None:
                            self.comment_map[name] = comment
                        coords.append((i + start_omega, j + start_omega))
                        init = A[i, j]
                        lower = None if i != j else 0
                        param = Parameter.create(name, init, lower=lower, fix=fix)
                        parameters.append(param)
                        label_index += 1
            next_omega = start_omega + size
        try:
            self.name_map
        except AttributeError:
            self.name_map = dict(zip((p.name for p in parameters), coords))
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
        for node in self.root.subtrees('omega'):
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

    def _get_param_name(self, row, col):
        """Get the name of the current OMEGA/SIGMA at (row, col)"""
        return next(key for key, value in self.name_map.items() if value == (row, col))

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
                if not isinstance(node, AttrTree) or node.rule != 'diag_item':
                    new_nodes.append(node)
                else:
                    sd = bool(node.find('SD'))
                    fix = bool(node.find('FIX'))
                    n = (
                        cast(int, eval_token(node.subtree('n').leaf('INT')))
                        if node.find('n')
                        else 1
                    )
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
                    new_init = new_inits[0]
                    if n == 1 or (
                        new_inits.count(new_init) == len(new_inits)
                        and new_fix.count(new_fix[0]) == len(new_fix)
                    ):  # All equal?
                        init = node.subtree('init')
                        if eval_token(init.leaf('NUMERIC')) != new_init:
                            new_init = int(new_init) if float(new_init).is_integer() else new_init
                            init = init.replace_first(AttrToken('NUMERIC', str(new_init)))
                        node = node.replace_first(init)
                        if new_fix[0] != fix:
                            if new_fix[0]:
                                node = insert_before_or_at_end(
                                    node, 'RPAR', [AttrToken('WS', ' '), AttrToken('FIX', 'FIX')]
                                )
                            else:
                                node = remove_token_and_space(node, 'FIX')
                        new_nodes.append(node)
                    else:
                        # Need to split xn
                        node = node.remove('n')
                        for j, new_init in enumerate(new_inits):
                            init = node.subtree('init')
                            if eval_token(init.leaf('NUMERIC')) != new_init:
                                new_init = (
                                    int(new_init) if float(new_init).is_integer() else new_init
                                )
                                init = init.replace_first(AttrToken('NUMERIC', str(new_init)))
                            node = node.replace_first(init)
                            if new_fix[j] != fix:
                                node = insert_before_or_at_end(
                                    node, 'RPAR', [AttrToken('WS', ' '), AttrToken('FIX', 'FIX')]
                                )
                            else:
                                node = remove_token_and_space(node, 'FIX')
                            new_nodes.append(node)
                            if j != len(new_inits) - 1:  # Not the last
                                new_nodes.append(AttrTree.create('ws', {'WS': ' '}))
                    i += n
            self.root = AttrTree(self.root.rule, tuple(new_nodes))
            next_omega = i
        else:
            same = bool(self.root.find('same'))
            if same:
                return first_omega + previous_size, previous_size
            size = cast(int, eval_token(self.root.subtree('block').subtree('size').leaf('INT')))
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

            A = flattened_to_symmetric(inits)

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
                if not isinstance(node, AttrTree) or node.rule != 'omega':
                    new_nodes.append(node)
                else:
                    n = (
                        cast(int, eval_token(node.subtree('n').leaf('INT')))
                        if node.find('n')
                        else 1
                    )
                    if array[i : i + n].count(array[i]) == n:  # All equal?
                        init = node.subtree('init')
                        new_init = array[i]
                        if eval_token(init.leaf('NUMERIC')) != new_init:
                            new_init = int(new_init) if float(new_init).is_integer() else new_init
                            init = init.replace_first(AttrToken('NUMERIC', str(new_init)))
                        node = node.replace_first(init)
                        new_nodes.append(node)
                    else:
                        # NOTE Split xn
                        new_children = []
                        for child in node.children:
                            if child.rule not in ('n', 'LPAR', 'RPAR'):
                                new_children.append(child)
                        node = AttrTree(node.rule, tuple(new_children))
                        for j, new_init in enumerate(array[i : i + n]):
                            init = node.subtree('init')
                            if eval_token(init.leaf('NUMERIC')) != new_init:
                                new_init = (
                                    int(new_init) if float(new_init).is_integer() else new_init
                                )
                                init = init.replace_first(AttrToken('NUMERIC', str(new_init)))
                            node = node.replace_first(init)
                            new_nodes.append(node)
                            if j != n - 1:  # NOTE Not the last
                                new_nodes.append(AttrTree.create('ws', {'WS': ' '}))
                    i += n
            self.root = AttrTree(self.root.rule, tuple(new_nodes))
            if new_fix[0] != fix:
                if new_fix[0]:
                    self.root = insert_after(
                        self.root, 'block', [AttrToken('WS', ' '), AttrToken('FIX', 'FIX')]
                    )
                else:
                    self.root = remove_token_and_space(self.root, 'FIX', recursive=True)
            next_omega = first_omega + size
        return next_omega, size

    def random_variables(
        self,
        start_omega: int,
        previous_start_omega: int,
        previous_cov: Optional[Union[sympy.Symbol, sympy.Matrix]],
    ) -> Tuple[
        RandomVariables,
        int,
        int,
        Union[str, Literal['ZERO'], sympy.Symbol, sympy.Matrix],
        List[str],
    ]:
        """Get RandomVariables for this omega record

        start_omega - the index of the first omega in this record
        previous_start_omega - the index of the first omega in the previous omega block
        previous_cov - the matrix of the previous omega block
        """
        all_zero_fix = False
        same = bool(self.root.find('same'))

        if self.name == 'OMEGA':
            if same:
                level = 'IOV'
            else:
                level = 'IIV'
        else:
            level = 'RUV'

        name_map = {}
        if not hasattr(self, 'name_map') and not same:
            if isinstance(previous_cov, sympy.Symbol):
                prev_size = 1
            elif previous_cov is not None:
                prev_size = len(previous_cov)
            else:
                prev_size = None
            self.parameters(start_omega, prev_size)
        if hasattr(self, 'name_map'):
            name_map = self.name_map
        rev_map = {value: key for key, value in name_map.items()}
        next_cov = None  # The cov matrix if a block
        next_start = previous_start_omega if same else start_omega
        block = self.root.find('block')
        bare_block = self.root.find('bare_block')
        zero_fix = []
        if not (block or bare_block):
            dists = []
            etas = []
            i = start_omega
            numetas = len(list(self.root.subtrees('diag_item')))
            for _ in range(numetas):
                omega_name = rev_map[(i, i)]
                name = self._rv_name(i)
                dist = NormalDistribution.create(name, level, 0, sympy.Symbol(omega_name))
                dists.append(dist)
                etas.append(name)
                i += 1
            rvs = RandomVariables.create(dists)
        else:
            if bare_block:
                assert previous_cov is not None
                numetas = previous_cov.rows
            else:
                numetas = cast(
                    int, eval_token(self.root.subtree('block').subtree('size').leaf('INT'))
                )
            params, _, _ = self.parameters(start_omega, getattr(previous_cov, 'rows', None))
            all_zero_fix = (
                all(param.init == 0 and param.fix for param in params)
                and len(params) > 0
                or (previous_cov == 'ZERO' and same)
            )
            if numetas >= 2:
                names = [self._rv_name(i) for i in range(start_omega, start_omega + numetas)]
                if all_zero_fix:
                    zero_fix = names
                means = [0] * numetas
                if same:
                    next_cov = previous_cov
                    dist = JointNormalDistribution.create(names, level, means, previous_cov)
                    etas = dist.names
                    rvs = RandomVariables.create((dist,))
                else:
                    cov = sympy.zeros(numetas)
                    for row in range(numetas):
                        for col in range(row + 1):
                            cov[row, col] = sympy.Symbol(
                                rev_map[(start_omega + row, start_omega + col)]
                            )
                            if row != col:
                                cov[col, row] = cov[row, col]
                    next_cov = cov
                    dist = JointNormalDistribution.create(names, level, means, cov)
                    etas = dist.names
                    rvs = RandomVariables.create((dist,))
            else:
                sym = previous_cov if same else sympy.Symbol(rev_map[(start_omega, start_omega)])
                name = self._rv_name(start_omega)
                if all_zero_fix:
                    zero_fix = [name]
                dist = NormalDistribution.create(name, level, 0, sym)
                next_cov = sym
                etas = [name]
                rvs = RandomVariables.create((dist,))

        self.eta_map = {eta: start_omega + i for i, eta in enumerate(etas)}
        if all_zero_fix:
            next_cov = 'ZERO'
        return rvs, start_omega + numetas, next_start, next_cov, zero_fix

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
            self.root = AttrTree(self.root.rule, tuple(keep))
        elif same and not bare_block:
            self.root = self.root.replace_first(
                (block := self.root.subtree('block')).replace_first(
                    block.subtree('size').replace_first(
                        AttrToken('INT', str(len(self) - len(indices)))
                    )
                )
            )
        elif block:
            fix, sd, corr, cholesky = self._block_flags()
            inits = []
            for node in self.root.subtrees('omega'):
                init = cast(float, eval_token(node.subtree('init').leaf('NUMERIC')))
                n = cast(int, eval_token(node.subtree('n').leaf('INT'))) if node.find('n') else 1
                inits += [init] * n
            A = flattened_to_symmetric(inits)
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
