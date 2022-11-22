import re
import warnings
from dataclasses import dataclass, replace
from typing import List, Set, Union, cast

from pharmpy.internals.parse import AttrToken, AttrTree
from pharmpy.internals.parse.generic import eval_token, remove_token_and_space
from pharmpy.model import Parameter

from .record import Record

# from .names import nonmem_to_pharmpy

max_upper_bound = 1000000
min_lower_bound = -1000000


@dataclass(frozen=True)
class ThetaRecord(Record):
    def add_nonmem_name(self, name_original):
        if re.match(r'THETA\(\d+\)', name_original):
            return self
        return replace(self, root=self.root.add_comment_node(name_original).add_newline_node())

    def parameters(self, first_theta, seen_labels=None):
        """Get a parameter set for this theta record.
        first_theta is the number of the first theta in this record
        """
        if seen_labels is None:
            seen_labels = set()
        pset: List[Parameter] = []
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

        return pset

    def _multiple(self, theta: AttrTree) -> int:
        """Return the multiple (xn) of a theta or 1 if no multiple"""
        if theta.find('n'):
            return cast(int, eval_token(theta.subtree('n').leaf('INT')))
        else:
            return 1

    def update(self, parameters):
        """From a Parameters update the THETAs in this record

        Currently only updating initial estimates
        """
        assert len(self) == len(parameters)

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

            i += self._multiple(theta)

            return theta

        return replace(self, root=self.root.map(_update_theta))

    def remove(self, indices: Set[int]):
        keep = []
        i = 0
        for node in self.root.children:
            if node.rule == 'theta':
                if i not in indices:
                    keep.append(node)
                i += 1
            else:
                keep.append(node)

        return replace(self, root=replace(self.root, children=tuple(keep)))

    def __len__(self):
        """Number of thetas in this record"""
        tot = 0
        for theta in self.root.subtrees('theta'):
            tot += self._multiple(theta)
        return tot
