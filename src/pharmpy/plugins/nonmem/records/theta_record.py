import re
from typing import Union, cast

from pharmpy.internals.parse import AttrToken, AttrTree
from pharmpy.internals.parse.generic import eval_token, remove_token_and_space

from .record import Record

max_upper_bound = 1000000
min_lower_bound = -1000000


class ThetaRecord(Record):
    def _multiple(self, theta: AttrTree) -> int:
        """Return the multiple (xn) of a theta or 1 if no multiple"""
        if theta.find('n'):
            return cast(int, eval_token(theta.subtree('n').leaf('INT')))
        else:
            return 1

    @property
    def inits(self):
        # List of initial estimates for all thetas
        inits = []
        for theta in self.root.subtrees('theta'):
            init = eval_token(theta.subtree('init').leaf('NUMERIC'))
            n = self._multiple(theta)
            inits.extend([init] * n)
        return inits

    @property
    def fixs(self):
        # List of fixedness for all thetas
        fixs = []
        for theta in self.root.subtrees('theta'):
            fix = bool(theta.find('FIX'))
            n = self._multiple(theta)
            fixs.extend([fix] * n)
        return fixs

    @property
    def bounds(self):
        # List of tuples of lower, upper bounds for all thetas
        bounds = []
        for theta in self.root.subtrees('theta'):
            if theta.find('low'):
                low = theta.subtree('low')
                if low.find('NEG_INF'):
                    lower = min_lower_bound
                else:
                    lower = eval_token(next(iter(low.tokens)))
            else:
                lower = min_lower_bound
            if theta.find('up'):
                up = theta.subtree('up')
                if up.find('POS_INF'):
                    upper = max_upper_bound
                else:
                    upper = eval_token(next(iter(up.tokens)))
            else:
                upper = max_upper_bound
            n = self._multiple(theta)
            bounds.extend([(lower, upper)] * n)
        return bounds

    @property
    def comment_names(self):
        # A list of all comment names in the record
        # None if missing
        names = []
        intheta = False
        n = 0
        for node in self.root.tree_walk():
            if node.rule == 'theta':
                intheta = True
                if n != 0:
                    names.extend([None] * n)
                n = self._multiple(node)
            if intheta and node.rule == 'COMMENT':
                m = re.search(r';\s*([a-zA-Z_]\w*)', str(node))
                if m:
                    names.append(m.group(1))
                else:
                    names.append(None)
                n -= 1
                if n == 0:
                    intheta = False

        if n != 0:
            names.extend([None] * n)

        return names

    def update(self, parameters):
        """From a Parameters update the THETAs in this record

        parameters - Only parameters defined by this $THETA record
        """
        i = 0

        def _update_theta(child: Union[AttrTree, AttrToken]):
            nonlocal i

            if not isinstance(child, AttrTree) or not child.rule == 'theta':
                return child

            theta = child

            param = parameters[i]
            new_init = param.init
            init = theta.subtree('init')
            if eval_token(init.leaf('NUMERIC')) != new_init:
                init = init.replace_first(AttrToken('NUMERIC', str(new_init)))
            theta = theta.replace_first(init)
            fix = bool(theta.find('FIX'))
            if fix != param.fix:
                if param.fix:
                    space = AttrToken('WS', ' ')
                    fix_token = AttrToken('FIX', 'FIX')
                    theta = AttrTree(theta.rule, theta.children + (space, fix_token))
                else:
                    theta = remove_token_and_space(theta, 'FIX')

            up = theta.find('up')
            if up != param.upper:
                if up is None and param.upper < 1000000:
                    comma = AttrToken('COMMA', ',')
                    upper = AttrToken('NUMERIC', param.upper)
                    up_node = AttrTree('up', (upper,))
                    keep = []
                    for node in theta.children:
                        keep.append(node)
                        if node.rule == 'init':
                            keep.extend((comma, up_node))
                    theta = AttrTree(theta.rule, tuple(keep))

            i += self._multiple(theta)

            return theta

        new_tree = self.root.map(_update_theta)
        return ThetaRecord(self.name, self.raw_name, new_tree)

    def remove(self, inds):
        if not inds:
            return self
        keep = []
        i = 0
        for node in self.root.children:
            if node.rule == 'theta':
                if i not in inds:
                    keep.append(node)
                i += 1
            else:
                keep.append(node)

        new_tree = AttrTree(self.root.rule, tuple(keep))
        return ThetaRecord(self.name, self.raw_name, new_tree)

    def __len__(self):
        """Number of thetas in this record"""
        tot = 0
        for theta in self.root.subtrees('theta'):
            tot += self._multiple(theta)
        return tot
