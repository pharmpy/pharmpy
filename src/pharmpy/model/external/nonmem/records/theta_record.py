import math
import re
from typing import Union, cast

from pharmpy.internals.parse import AttrToken, AttrTree
from pharmpy.internals.parse.generic import eval_token, remove_token_and_space
from pharmpy.model import ModelSyntaxError

from .record import Record

MAX_UPPER_BOUND = 1000000
MIN_LOWER_BOUND = -1000000
INF = float("inf")


def lower_token(theta):
    # Get raw lower bound for a theta subtree
    # Either a float, None for non-existing or 'neginf' for -INF
    if theta.find('low'):
        low = theta.subtree('low')
        if low.find('NEG_INF'):
            lower = 'neginf'
        else:
            lower = eval_token(next(iter(low.tokens)))
    else:
        lower = None
    return lower


def upper_token(theta):
    # Get raw upper bound for a theta subtree
    # Either a float, None for non-existing or 'inf' for INF
    if theta.find('up'):
        up = theta.subtree('up')
        if up.find('POS_INF'):
            upper = 'inf'
        else:
            upper = eval_token(next(iter(up.tokens)))
    else:
        upper = None
    return upper


def add_upper_bound(theta, bound):
    comma = AttrToken('COMMA', ',')
    upper = AttrToken('NUMERIC', format_number(bound))
    up_node = AttrTree('up', (upper,))
    keep = []
    for node in theta.children:
        keep.append(node)
        if node.rule == 'init':
            keep.extend((comma, up_node))
    theta = AttrTree(theta.rule, tuple(keep))
    return theta


def add_lower_bound(theta, bound):
    comma = AttrToken('COMMA', ',')
    lower = AttrToken('NUMERIC', format_number(bound))
    low_node = AttrTree('low', (lower,))
    keep = []
    for node in theta.children:
        if node.rule == 'init':
            keep.extend((low_node, comma))
        keep.append(node)
    theta = AttrTree(theta.rule, tuple(keep))
    return theta


def remove_upper_bound(theta):
    keep = []
    in_upper = False
    for node in theta.children:
        if not in_upper:
            keep.append(node)
        if node.rule == 'init':
            in_upper = True
        elif node.rule == 'up':
            in_upper = False
    theta = AttrTree(theta.rule, tuple(keep))
    return theta


def remove_lower_bound(theta):
    keep = []
    in_lower = False
    for node in theta.children:
        if node.rule == 'low':
            in_lower = True
        elif node.rule == 'init':
            in_lower = False
        if not in_lower:
            keep.append(node)
    theta = AttrTree(theta.rule, tuple(keep))
    return theta


def format_number(x):
    if not math.isinf(x) and int(x) == x:
        return str(int(x))
    else:
        return str(x)


def replace_bound(theta, which, bound):
    # which is either low or up
    val = AttrToken('NUMERIC', format_number(bound))
    new_node = AttrTree(which, (val,))
    keep = []
    for node in theta.children:
        if node.rule == which:
            keep.append(new_node)
        else:
            keep.append(node)
    theta = AttrTree(theta.rule, tuple(keep))
    return theta


def replace_lower_bound(theta, bound):
    theta = replace_bound(theta, 'low', bound)
    return theta


def replace_upper_bound(theta, bound):
    theta = replace_bound(theta, 'up', bound)
    return theta


def remove_parentheses(theta):
    keep = [node for node in theta.children if node.rule not in ("LPAR", "RPAR")]
    theta = AttrTree(theta.rule, tuple(keep))
    return theta


def add_parentheses(theta):
    keep = []
    up = theta.find('up')
    for node in theta.children:
        if node.rule == "LPAR":
            return theta
        elif node.rule == "low":
            lpar = AttrToken('LPAR', '(')
            keep.append(lpar)
            keep.append(node)
        elif up is None and node.rule == "init" or up is not None and node.rule == "up":
            rpar = AttrToken('RPAR', ')')
            keep.append(node)
            keep.append(rpar)
        else:
            keep.append(node)
    theta = AttrTree(theta.rule, tuple(keep))
    return theta


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
            if init == MAX_UPPER_BOUND or init == MIN_LOWER_BOUND:
                raise ModelSyntaxError(
                    f"Initial estimate of THETA cannot be {MAX_UPPER_BOUND} or {MIN_LOWER_BOUND}"
                )
            n = self._multiple(theta)
            inits.extend([init] * n)
        return inits

    @property
    def fixs(self):
        # List of fixedness for all thetas
        fixs = []
        for theta in self.root.subtrees('theta'):
            init = eval_token(theta.subtree('init').leaf('NUMERIC'))
            lowtok = lower_token(theta)
            uptok = upper_token(theta)
            # Raise if FIX is within the parentheses and explicit bounds
            # are used that are not the same as the init
            inparens = False
            fix = False
            for child in theta.children:
                if child.rule == 'LPAR':
                    inparens = True
                elif child.rule == 'RPAR':
                    inparens = False
                elif child.rule == 'FIX':
                    if inparens:
                        if lowtok is not None:
                            message = (
                                "FIX inside parentheses of $THETA requires all bounds to be"
                                " the same as the initial value. "
                            )
                            if uptok is not None:
                                if not (lowtok == uptok == init):
                                    raise ModelSyntaxError(
                                        f"{message}init={init}, lower={lowtok}, upper={uptok}"
                                    )
                            else:
                                if not (lowtok == init):
                                    raise ModelSyntaxError(f"{message}init={init}, lower={lowtok}")
                    fix = True
                    break

            if not fix and uptok is None and lowtok == init:
                raise ModelSyntaxError(
                    "Lower bound cannot be equal to initial estimate of THETA unless FIX"
                )
            if not fix and init == 0:
                raise ModelSyntaxError("Initial estimate of THETA cannot be 0 unless fixed")
            n = self._multiple(theta)
            fixs.extend([fix] * n)
        return fixs

    @property
    def bounds(self):
        # List of tuples of lower, upper bounds for all thetas
        bounds = []
        for theta in self.root.subtrees('theta'):
            lowtok = lower_token(theta)
            if isinstance(lowtok, float):
                if lowtok == MIN_LOWER_BOUND:
                    lower = -INF
                elif lowtok < MIN_LOWER_BOUND:
                    raise ModelSyntaxError(f"Too low lower bound: {lowtok} < {MIN_LOWER_BOUND}")
                else:
                    lower = lowtok
            else:
                lower = -INF

            uptok = upper_token(theta)
            if isinstance(uptok, float):
                if uptok == MAX_UPPER_BOUND:
                    upper = INF
                elif uptok > MAX_UPPER_BOUND:
                    raise ModelSyntaxError(f"Too high upper bound: {uptok} > {MAX_UPPER_BOUND}")
                else:
                    upper = uptok
            else:
                upper = INF

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
            low = theta.find('low')
            n = self._multiple(theta)

            need_upper_bound = param.upper < 1000000

            if up != param.upper:
                have_upper_bound = up is not None
                if not have_upper_bound and need_upper_bound:
                    theta = add_upper_bound(theta, param.upper)
                elif have_upper_bound and not need_upper_bound:
                    theta = remove_upper_bound(theta)
                else:
                    theta = replace_upper_bound(theta, param.upper)

            if low != param.lower:
                have_lower_bound = low is not None
                need_lower_bound = param.lower > -1000000 or need_upper_bound
                if not have_lower_bound and need_lower_bound:
                    theta = add_lower_bound(theta, param.lower)
                    theta = add_parentheses(theta)
                elif have_lower_bound and not need_lower_bound:
                    theta = remove_lower_bound(theta)
                    if n == 1:
                        theta = remove_parentheses(theta)
                else:
                    theta = replace_lower_bound(theta, param.lower)

            i += n

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
