import random
from collections import namedtuple

import pytest

import pharmpy.plugins.nonmem.nmtran_parser
import pharmpy.plugins.nonmem.records


@pytest.fixture
def records():
    return pharmpy.plugins.nonmem.records


@pytest.fixture
def parser():
    return pharmpy.plugins.nonmem.nmtran_parser.NMTranParser()


@pytest.fixture
def create_record(nonmem):
    """Record creating method with some (general) logging/asserts"""

    def func(buf, fail=False):
        record = nonmem.records.factory.create_record(buf)
        print(str(record))
        if fail:
            assert record.name is None
        else:
            assert record.name.startswith(record.raw_name)
        if buf.startswith('$'):
            assert str(record) == buf
        else:
            assert str(record) == '$' + buf
        return record

    return func


# -- ONLY ThetaRecord ------------------------------------------------------------------------------


class _GeneratedTheta:
    """
    A generated theta (see :class:`RandomThetas` for random generation).

    Can "messy" print and validate against tree parse, a.k.a. post-processing feature creep...
    """

    _WS = '\t' + (' '*9)
    Values = namedtuple('Values', ['low', 'init', 'up'])

    __slots__ = ('num', 'num_str', 'fixed', 'n_thetas')

    def __init__(self, num, num_str, fixed, n_thetas):
        self.num = self.Values(*num)
        self.num_str = self.Values(*num_str)
        self.fixed = fixed
        self.n_thetas = n_thetas

    @classmethod
    def new(cls, **kwargs):
        """Alternative constructor; Creates GeneratedTheta object from (partial) kwargs."""
        args = dict(num=(None, None, None), num_str=(None, None, None), fixed=None, n_thetas=None)
        if 'num' not in kwargs:
            num = tuple(kwargs.pop(x, None) for x in ['lower', 'init', 'upper'])
            kwargs.update(num=num)
        args.update(**kwargs)
        return cls(**args)

    @classmethod
    def from_tree(cls, tree):
        """
        Alternative constructor; Generate from AttrTree.

        This is a 'basic' non-API implementation of postprocessing, for grammar-close lexer-parser
        testing.
        """
        def find_tokens(tree, rule, rule_token):
            nodes = filter(lambda x: x.rule == rule, tree.tree_walk())
            return list(map(lambda x: getattr(x, rule_token), nodes))

        params = []
        for param in tree.all('param'):
            single = param.all('single')
            multi = param.all('multi')
            assert (len(single) + len(multi)) == 1
            params += single + multi

        thetas = []
        for param in params:
            kw = dict(num=[], fixed=None, n_thetas=None)
            for rule in ['low', 'init', 'up']:
                tok = find_tokens(param, rule, 'NUMERIC')
                assert len(tok) <= 1
                kw['num'] += tok if tok else [None]

            fixed = param.all('fix')
            assert len(fixed) <= 1
            kw['fixed'] = bool(fixed)

            n_thetas = find_tokens(param, 'n_thetas', 'INT')
            assert len(n_thetas) <= 1
            if n_thetas:
                kw['n_thetas'] = int(n_thetas[0])

            thetas += [cls.new(**kw)]
        return thetas

    def __eq__(self, other):
        """
        Equivalence check.

        Attributes which are None on self won't count. Thus partial equivalence, when __str__
        randomizing drops some info, works. No, this whole system isn't the best but it is a clean
        test of lexer-parser (without API magic).
        """
        for key, val in self.num._asdict().items():
            if val and val != getattr(other.num, key):
                return False
        for attr in ['fixed', 'n_thetas']:
            val = getattr(self, attr, None)
            if val and val != getattr(other, attr, None):
                return False
        return True

    def __str__(self):
        low = self._lr_pad(self.num_str.low)
        init = self._lr_pad(self.num_str.init)
        high = self._lr_pad(self.num_str.up)
        fix = ''
        if self.fixed:
            fix = self._lr_pad(random.choice(['FIX', 'FIXED']))
        form = random.randrange(6)
        if form == 0:
            out = init + fix
        elif form == 1:
            out = '(%s,%s %s)' % (low, init, fix)
        elif form == 2:
            out = '(%s,%s,%s %s)' % (low, init, high, fix)
        elif form == 3:
            out = '(%s,%s,%s) %s' % (low, init, high, fix)
        elif form == 4:
            init_missing = random.choice(self._WS)*random.randrange(3)
            out = '(%s,%s,%s %s)' % (low, init_missing, high, fix)
        else:
            n_thetas = self._l_pad('x') + self._lr_pad(self.n_thetas)
            out = '(%s)%s' % (init, n_thetas)
        return out

    def __repr__(self):
        """Pretty format on some 'standard form' (mostly for good pytest diffs)."""
        fix = ' FIX' if self.fixed else ''
        vals = [getattr(self.num, a) or '' for a in ['low', 'init', 'up']]
        out = '(%s, %s, %s%s)' % (*vals, fix)
        if self.n_thetas is not None:
            out += ('x%s' % (self.n_thetas,))
        return out

    def _l_pad(self, obj):
        """Format obj via wrapping with (left) random whitespace padding."""
        if obj is None:
            return ''
        lpad = random.choice(self._WS)*random.randrange(5)
        return lpad + str(obj)

    def _lr_pad(self, obj):
        """Format obj via wrapping with (left + right) random whitespace padding."""
        if obj is None:
            return ''
        lpad = random.choice(self._WS)*random.randrange(5)
        rpad = random.choice(self._WS)*random.randrange(5)
        return lpad + str(obj) + rpad


@pytest.fixture
def GeneratedTheta():
    """Provide GeneratedTheta (helper) class to tests."""
    return _GeneratedTheta


@pytest.fixture
def RandomThetas(RandomData, GeneratedTheta):
    """Provide RandomTheta (helper) class to tests."""

    class _RandomThetas(RandomData):
        """Extension of :class:`RandomData` with new generator, for producing thetas."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._gen_str_num = self.str_num(N=3)

        # -- methods producing random generators ---------------------------
        def theta(self):
            """Returns (generator for) random thetas."""

            def f():
                # TODO: This is backwards. Why should GeneratedTheta keep string-representation?
                # Causes issues whereas RandomData.str_num needs to do float(s) to return 'correct'
                # value for equivalence check...
                num, _str = zip(*next(self._gen_str_num))
                fixed = random.getrandbits(1)
                n_thetas = random.randrange(1, 5)
                return GeneratedTheta(num, _str, fixed, n_thetas)

            return self._gen(f)

    return _RandomThetas
