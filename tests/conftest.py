import csv
import random
import re
import string
from collections import namedtuple
from pathlib import Path

import pytest
from pyfakefs.fake_filesystem_unittest import Patcher

# Do not let pyfakefs patch mpmath since it emits warnings
# and doesn't do anything with the filesystem anyway
Patcher.SKIPNAMES.update(['mpmath'])


@pytest.fixture(scope='session')
def testdata():
    """Test data (root) folder."""
    return Path(__file__).resolve().parent / 'testdata'


@pytest.fixture(scope='session')
def csv_read():
    """Load test data from CSV. Supports tuples."""

    _tuple_matcher = re.compile(r'^\((.*)\)$')

    def func(root, file, names=None):
        TestData = tuple
        if names:
            TestData = namedtuple('TestData', names)

        def descape(item):
            return item.replace('\\n', '\n')

        def tupleize(item):
            m = _tuple_matcher.match(item)
            if not m:
                return descape(item)
            if not m.group(1):
                return ()
            return tuple(descape(x.strip(' "')) for x in m.group(1).split(','))

        with open(Path(root, file), 'r') as f:
            dialect = csv.Sniffer().sniff(f.read(1024))
            f.seek(0)
            reader = csv.reader(f, dialect)
            return tuple(TestData(*tuple(map(tupleize, row))) for row in reader)

    return func


@pytest.fixture(scope='session')
def str_repr():
    """Debug print (string representations)."""

    def func(string):
        if not string:
            return '-- EMPTY --'
        return '"' + repr(string)[1:-1] + '"'
    return func


class _RandomData:
    """
    A collection of random data generators.

    Use to procedurally generate data for testing.

    Args:
        N (int): Number of values to cap generators on. Infinite if None.
        comment_char (str): Char to begin comments (and exclude from comment content).
    """

    comment_char = ';'
    comment_charset = (' '*25 + string.ascii_letters*5 + string.digits*3 + '\t'*3 +
                       string.punctuation).replace(comment_char, '')

    def __init__(self, N=None, comment_char=';'):
        self.history = []
        if comment_char != self.comment_char:
            self.comment_charset = self.comment_charset.replace(self.comment_char, comment_char)
            self.comment_char = comment_char
        self.N = N

    # -- methods producing random generators ---------------------------
    def pos_int(self, size=1E4):
        """Returns (generator for) non-negative integer x, where x <= size."""
        return self._gen(random.randint, 0, size)

    def int(self, size=1E4):
        """Returns (generator for) integer x, where -size < x < size."""
        return self._gen(random.randint, -size, size)

    def float(self, size=1E9):
        """Returns (generator for) floating point num x, where x ~ normal_dist(0, sd=size)."""
        return self._gen(random.normalvariate, 0, size)

    def str_num(self, N=0, sci_fmt=('%E', '%.0E')):
        """Returns (generator for) tuple (str, num) from mixed int/float (convenience function).

        Args:
            N (int): Generate 1 tuple (str, num) if 0. Else, sorted (nested) tuple of 'N' tuples.
            sci_fmt (tuple): Format to randomly apply when formatting floats.
        """

        def f(sci_fmt, scalar, n):
            def g():
                if random.getrandbits(1):
                    if random.getrandbits(1):
                        val = random.randint(0, 9)
                    else:
                        val = random.randint(-1E5, 1E5)
                    s = str(val)
                else:
                    fmt = random.choice(sci_fmt)
                    if random.getrandbits(1):
                        val = random.normalvariate(0, 1)
                    else:
                        val = random.normalvariate(0, 10**(random.randint(-50, 50)))
                    s = fmt % (val,)
                    val = float(s)
                return val, s

            if scalar:
                return g()
            else:
                sort = sorted((g() for _ in range(n)), key=lambda x: x[0])
                return tuple(sort)

        return self._gen(f, sci_fmt, not N, N)

    def bool(self):
        """Returns (generator for) bool."""
        return self._gen(lambda: bool(random.getrandbits(1)))

    def choice(self, *choices):
        """Returns (generator for) choice (in set from args)."""
        return self._gen(lambda x: random.choice(x), choices)

    def biased_choice(self, *bias_choice):
        """
        Returns (generator for) biased choice (in set from args).

        Each arg is a tuple (bias, choice) where bias is the relative frequency (float).
        """

        biases, choices = zip(*bias_choice)
        base = min(biases)*100
        rep = tuple(round((b*100)/base) for b in biases)
        choices = [[ch]*rep for ch, rep in zip(choices, rep)]
        choices = [x for lst in choices for x in lst]
        return self._gen(lambda x: random.choice(x), choices)

    def comment(self, maxlen=30):
        """
        Returns (generator for) comment (uses :attr:`comment_char` and :attr:`comment_charset`).

        Args:
            maxlen (int): Maximum length of comment.
        """

        def f(st, ch, _max):
            comment = st + random.choice(['', ' '])
            if not bool(random.getrandbits(2)):
                return comment
            return comment + ''.join(random.choice(ch) for _ in range(1, random.randrange(_max)))
        return self._gen(f, self.comment_char, self.comment_charset, maxlen)

    def pad(self, maxlen=5, nl=False):
        """
        Returns (generator for) whitespace padding/empty str 50/50.

        Args:
            maxlen (int): Maximum length of str.
            nl (bool): Allow line endings (half as likely as ' ').
        """

        def f(x):
            return random.choice(chars)*random.choice([0]*x + list(range(1, x)))
        if nl:
            chars = ' \n '
        else:
            chars = ' '
        return self._gen(f, maxlen)

    # -- private methods -----------------------------------------------
    def _gen(self, f, *args, **kwargs):
        """Make generator from function and args."""

        n_gen = 0
        while (self.N is None) or (n_gen < self.N):
            obj = f(*args, **kwargs)
            self.history += [obj]
            n_gen += 1
            yield obj

    def __str__(self):
        rval = ', '.join(repr(x) for x in self.history)
        return '%s(N=%s): %s' % (self.__class__.__name__, self.N, rval)


@pytest.fixture(scope='session')
def RandomData():
    """Provide RandomData (helper) class to tests."""
    return _RandomData


@pytest.fixture(scope='session')
def datadir(testdata):
    return testdata / 'nonmem'


@pytest.fixture(scope='session')
def pheno_path(datadir):
    return datadir / 'pheno_real.mod'
