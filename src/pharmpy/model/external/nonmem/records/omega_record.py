import math
import re
from typing import cast

from pharmpy.deps import numpy as np
from pharmpy.internals.math import flattened_to_symmetric, triangular_root
from pharmpy.internals.parse import AttrToken, AttrTree
from pharmpy.internals.parse.generic import (
    eval_token,
    insert_after,
    insert_before_or_at_end,
    remove_token_and_space,
)
from pharmpy.model import ModelSyntaxError

from .record import Record


class OmegaRecord(Record):
    def parse(self):
        """Parse the omega record

        Return a list with one tuple for each block. Each block will have a list of names,
        a list of inits, fixedness and same.
        """
        block = self.root.find('block')
        bare_block = self.root.find('bare_block')
        same = bool(self.root.find('same'))
        blocks = []

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
                name = self._get_name(node)
                for _ in range(n):
                    block = ([name], [init], fixed, False)
                    blocks.append(block)
        else:
            inits = []
            fix, sd, corr, cholesky = self._block_flags()
            comments = []
            for node in self.root.subtrees('omega'):
                init = cast(float, eval_token(node.subtree('init').leaf('NUMERIC')))
                n = cast(int, eval_token(node.subtree('n').leaf('INT'))) if node.find('n') else 1
                inits += [init] * n
                comment = self._get_name(node)
                comments.append(comment)
                if n > 1:
                    comments.extend([None] * (n - 1))
            if not same:
                size = cast(int, eval_token(self.root.subtree('block').subtree('size').leaf('INT')))
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
                names = []
                inits = []
                for i in range(size):
                    for j in range(0, i + 1):
                        comment = comments[label_index]
                        init = A[i, j]
                        names.append(comment)
                        inits.append(init)
                        label_index += 1
                block = (names, inits, fix, False)
            else:
                block = (None, None, None, True)
            blocks.append(block)
        return blocks

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

    def update(self, parameters):
        """From a Parameters update the OMEGAs in this record"""
        block = self.root.find('block')
        bare_block = self.root.find('bare_block')
        if not (block or bare_block):
            i = 0
            size = 1
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
                        if not sd:
                            value = parameters[j].init
                        else:
                            value = parameters[j].init ** 0.5
                        new_inits.append(value)
                        new_fix.append(parameters[j].fix)
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
            tree = AttrTree(self.root.rule, tuple(new_nodes))
            return OmegaRecord(self.name, self.raw_name, tree)
        else:
            same = bool(self.root.find('same'))
            if same:
                return self
            size = cast(int, eval_token(self.root.subtree('block').subtree('size').leaf('INT')))
            fix, sd, corr, cholesky = self._block_flags()
            inits = [param.init for param in parameters]
            new_fix = [param.fix for param in parameters]
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
            tree = AttrTree(self.root.rule, tuple(new_nodes))
            if new_fix[0] != fix:
                if new_fix[0]:
                    tree = insert_after(
                        tree, 'block', [AttrToken('WS', ' '), AttrToken('FIX', 'FIX')]
                    )
                else:
                    tree = remove_token_and_space(tree, 'FIX', recursive=True)
            return OmegaRecord(self.name, self.raw_name, tree)

    def remove(self, inds):
        """Remove some etas from block given eta names
        inds - list of tuples of block index and eta index within block
        """
        if len(inds) == 0:
            return self

        block = self.root.find('block')
        bare_block = self.root.find('bare_block')
        same = bool(self.root.find('same'))
        if not (block or bare_block):
            keep = []
            i = 0
            eta_inds = {ind for ind, _ in inds}
            in_keep = True
            for node in self.root.children:
                if node.rule == 'diagonal':
                    in_keep = False
                if node.rule == 'diag_item':
                    in_keep = i not in eta_inds
                    i += 1
                if in_keep:
                    keep.append(node)
            tree = AttrTree(self.root.rule, tuple(keep))
            return OmegaRecord(self.name, self.raw_name, tree)
        elif same and not bare_block:
            indices = {ind for _, ind in inds}
            tree = self.root.replace_first(
                (block := self.root.subtree('block')).replace_first(
                    block.subtree('size').replace_first(
                        AttrToken('INT', str(len(self) - len(indices)))
                    )
                )
            )
            return OmegaRecord(self.name, self.raw_name, tree)
        elif block:
            fix, sd, corr, cholesky = self._block_flags()
            inits = []
            indices = {ind for _, ind in inds}
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

            temp = self.raw_name + s
            from .factory import create_record

            return create_record(temp)  # FIXME: No need to reparse

    def __len__(self):
        # Number of blocks in record
        block = self.root.find('block')
        bare_block = self.root.find('bare_block')
        if not (block or bare_block):
            size = 0
            for node in self.root.subtrees('diag_item'):
                n = cast(int, eval_token(node.subtree('n').leaf('INT'))) if node.find('n') else 1
                size += n
        else:
            size = 1
        return size
