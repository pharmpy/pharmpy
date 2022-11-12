import re
import warnings
from itertools import count
from typing import Union, cast

from pharmpy.internals.parse import AttrToken, AttrTree
from pharmpy.internals.parse.generic import eval_token, remove_token_and_space
from pharmpy.model import Parameter

from .record import Record

max_upper_bound = 1000000
min_lower_bound = -1000000


class ThetaRecord(Record):
    def __init__(self, content, parser_class):
        super().__init__(content, parser_class)
        self.name_map = None

    def add_nonmem_name(self, name_original, theta_number):
        if not re.match(r'THETA\(\d+\)', name_original):
            self.root = self.root.add_comment_node(name_original)
            self.root = self.root.add_newline_node()
        self.name_map = {name_original: theta_number}

    def parameters(self, first_theta, seen_labels=None):
        """Get a parameter set for this theta record.
        first_theta is the number of the first theta in this record
        """
        if seen_labels is None:
            seen_labels = set()
        pset = []
        current_theta = first_theta
        for theta in self.root.subtrees('theta'):
            init = eval_token(theta.subtree('init').leaf('NUMERIC'))
            fix = bool(theta.find('FIX'))
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

            for _ in range(0, n):
                name = None
                import pharmpy.plugins.nonmem as nonmem

                if 'comment' in nonmem.conf.parameter_names:
                    # needed to avoid circular import with Python 3.6
                    found = False
                    for subnode in self.root.tree_walk():
                        if id(subnode) == id(theta):
                            if found:
                                break
                            else:
                                found = True
                                continue
                        if found and subnode.rule == 'COMMENT':
                            m = re.search(r';\s*([a-zA-Z_]\w*)', str(subnode))
                            if m:
                                name = m.group(1)
                                break
                    if name in seen_labels:
                        warnings.warn(
                            f'The parameter name {name} is duplicated. Falling back to basic '
                            f'NONMEM names.'
                        )
                        name = None

                if not name:
                    name = f'THETA({current_theta})'
                seen_labels.add(name)
                new_par = Parameter.create(name, init, lower, upper, fix)
                current_theta += 1
                pset.append(new_par)

        if not self.name_map:
            self.name_map = {name: first_theta + i for i, name in enumerate([p.name for p in pset])}
        return pset

    def _multiple(self, theta: AttrTree) -> int:
        """Return the multiple (xn) of a theta or 1 if no multiple"""
        if theta.find('n'):
            return cast(int, eval_token(theta.subtree('n').leaf('INT')))
        else:
            return 1

    def update(self, parameters, first_theta):
        """From a Parameters update the THETAs in this record

        Currently only updating initial estimates
        """
        assert self.name_map is not None
        i = first_theta
        index = {v: k for k, v in self.name_map.items()}

        def _update_theta(child: Union[AttrTree, AttrToken]):
            nonlocal i

            if not isinstance(child, AttrTree) or not child.rule == 'theta':
                return child

            theta = child

            name = index[i]
            param = parameters[name]
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

            i += self._multiple(theta)

            return theta

        self.root = self.root.map(_update_theta)

    def update_name_map(self, trans):
        """Update name_map given dict from -> to"""
        assert self.name_map is not None
        for key, value in trans.items():
            if key in self.name_map:
                n = self.name_map[key]
                del self.name_map[key]
                self.name_map[value] = n

    def renumber(self, new_start):
        assert self.name_map is not None
        for new_index, name in zip(
            count(new_start), map(lambda t: t[0], sorted(self.name_map.items(), key=lambda t: t[1]))
        ):
            self.name_map[name] = new_index

    def remove(self, names):
        assert self.name_map is not None

        first_theta = min(self.name_map.values())
        indices = {self.name_map[name] - first_theta for name in names}

        for name in names:
            del self.name_map[name]

        keep = []
        i = 0
        for node in self.root.children:
            if node.rule == 'theta':
                if i not in indices:
                    keep.append(node)
                i += 1
            else:
                keep.append(node)

        self.root = AttrTree(self.root.rule, tuple(keep))

    def __len__(self):
        """Number of thetas in this record"""
        tot = 0
        for theta in self.root.subtrees('theta'):
            tot += self._multiple(theta)
        return tot
