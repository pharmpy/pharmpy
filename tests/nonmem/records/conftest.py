# -*- encoding: utf-8 -*-

import random
from collections import namedtuple

import pytest


# -- general test helpers --------------------------------------------------------------------------
@pytest.fixture
def create_record(api, request):
    """(Inject) record creating method, with some (general) logging/asserts"""

    def func(cls, buf, fail=False):
        record = api.records.factory.create_record(buf)
        print(str(record))
        if fail:
            assert record.name is None
        else:
            assert record.name.startswith(record.raw_name)
            assert record.name == cls.canonical_name
        if buf.startswith('$'):
            assert str(record) == buf
        else:
            assert str(record) == '$' + buf
        return record

    request.cls.create_record = func


# -- ThetaRecord test helpers ----------------------------------------------------------------------
class _GeneratedTheta:
    """
    A generated theta (see :class:`RandomThetas` for random generation).

    Can "messy" print and validate to tree parse.
    """

    _WS = '\t' + (' '*9)
    Values = namedtuple('Values', ['lower_bound', 'init', 'upper_bound'])

    __slots__ = ('num', 'num_str', 'fixed', 'n_thetas')

    def __init__(self, num=None, num_str=None, fixed=None, n_thetas=None):
        if num:
            self.num = self.Values(*num)
        if num_str:
            self.num_str = self.Values(*num_str)
        self.fixed = fixed
        self.n_thetas = n_thetas

    def from_tree(self, tree):
        """Generate from tree. WIP. Idea is to base validate here instead of deeper in the tests."""
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
            kw = dict(num=[], num_str=None, fixed=None, n_thetas=None)
            for rule in ['lower_bound', 'init', 'upper_bound']:
                tok = self.find_tokens(param, rule, 'NUMERIC')
                assert len(tok) <= 1
                kw['num'] += [tok] if tok else None

            fixed = param.all('fix')
            assert len(fixed) <= 1
            kw['fixed'] = bool(fixed)

            n_thetas = self.find_tokens(param, 'n_thetas', 'INT')
            assert len(n_thetas) <= 1
            if n_thetas:
                kw['n_thetas'] = int(n_thetas[0])

            thetas += [GeneratedTheta(**kw)]
            pass

    def __eq__(self, other):
        """Compare to other object. WIP. Idea is that tests can validate against reference."""
        pass

    def __str__(self):
        if self.fixed:
            fix = self._lr_pad(random.choice(['FIX', 'FIXED']))
        else:
            fix = self._lr_pad('')
        low = self._lr_pad(self.num_str.lower_bound)
        init = self._lr_pad(self.num_str.init)
        high = self._lr_pad(self.num_str.upper_bound)
        form = random.randrange(6)
        if form == 0:
            s = init + fix
        elif form == 1:
            s = '(%s,%s %s)' % (low, init, fix)
        elif form == 2:
            s = '(%s,%s,%s %s)' % (low, init, high, fix)
        elif form == 3:
            s = '(%s,%s,%s) %s' % (low, init, high, fix)
        elif form == 4:
            init_missing = random.choice(self._WS)*random.randrange(3)
            s = '(%s,%s,%s %s)' % (low, init_missing, high, fix)
        else:
            n_thetas = self._l_pad('x') + self._lr_pad(self.n_thetas)
            s = '(%s)%s' % (init, n_thetas)
        return s

    def _l_pad(self, obj):
        """Format obj via wrapping in (left) random whitespace padding."""
        lpad = random.choice(self._WS)*random.randrange(5)
        return lpad + str(obj)

    def _lr_pad(self, obj):
        """Format obj via wrapping in (left + right) random whitespace padding."""
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
                num, _str = zip(*next(self._gen_str_num))
                fixed = random.getrandbits(1)
                n_thetas = random.randrange(1, 5)
                return GeneratedTheta(num, _str, fixed, n_thetas)

            return self._gen(f)

    return _RandomThetas
