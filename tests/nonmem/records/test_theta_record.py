# -*- encoding: utf-8 -*-

import pytest


@pytest.fixture(autouse=True)
def canonical_name(request):
    """Inject canonical name into all classes"""
    request.cls.canonical_name = 'THETA'


@pytest.fixture
def parse_assert(api, GeneratedTheta):
    """Provide parse_assert (helper) function to tests."""

    def func(buf: str, reference: list):
        """
        Parses buf into GeneratedTheta(s) and assserts eq. to reference.

        Args:
            buf: Buffer to parse.
            reference: Assert equivalence with these GeneratedTheta objects.
        """

        for i, line in enumerate(buf.splitlines()):
            print('%d: %s' % (i, repr(line)))
        tree = api.records.parser.ThetaRecordParser(buf)
        print(str(tree) + '\n')
        root = tree.root

        assert str(root) == buf
        assert GeneratedTheta.from_tree(root) == reference
        return root

    return func


class TestParser:
    """
    Logical (non-semantical) grouping of pure parser tests (no API post-processing).

    Motivation: Experiment with sandboxing testing where grammar changes can flag before unit tests
    on API needs to step in. Might be unnecessary? Not removing, but next record won't get this
    luxury, I'm sure...

    Lookup :func:`parse_assert`, :class:`GeneratedTheta` and :class:`RandomThetas` for how it fits
    together.
    """
    def test_single_inits(self, RandomData, GeneratedTheta, parse_assert):
        for val in RandomData(5).int():
            parse_assert(str(val), [GeneratedTheta.new(init=val)])

        for val in RandomData(5).float():
            parse_assert(str(val), [GeneratedTheta.new(init=val)])

    def test_padded_inits(self, RandomData, GeneratedTheta, parse_assert):
        data = RandomData(5)
        for lpad, val, rpad in zip(data.pad(), data.float(), data.pad()):
            parse_assert(str(lpad) + str(val) + str(rpad), [GeneratedTheta.new(init=val)])

        data = RandomData(5)
        for lpad, val, rpad in zip(data.pad(nl=True), data.float(), data.pad(nl=True)):
            parse_assert(str(lpad) + str(val) + str(rpad), [GeneratedTheta.new(init=val)])

        data = RandomData(5)
        for val in data.pad(nl=True):
            parse_assert(str(val), [])

    def test_comments(self, RandomData, parse_assert):
        bufs, comments = [], []
        data = RandomData(10)
        for lpad, comment in zip(data.pad(nl=True), data.comment()):
            bufs += [lpad + comment]
            comments += [comment.strip().lstrip(';').strip()]
        buf = '\n'.join(bufs)
        root = parse_assert(buf, [])
        nodes = filter(lambda x: x.rule == 'comment', root.tree_walk())
        assert comments == list(map(lambda x: getattr(x, 'TEXT'), nodes))

    def test_messy_randoms(self, RandomThetas, parse_assert):
        bufs, thetas = [], []
        for i, theta in enumerate(RandomThetas(20).theta()):
            thetas += [theta]
            bufs += [str(theta)]
            print(bufs[-1])
        buf = '\n'.join(bufs)
        parse_assert(buf, thetas)


@pytest.mark.usefixtures('create_record')
class TestRecordCreate:
    def test_init(self):
        self.create_record('THET')
        self.create_record('THETA 0')
        self.create_record('THETA   12.3 \n\n')
        self.create_record('THETAA 123', fail=True)
        self.create_record('THEETA', fail=True)


@pytest.mark.usefixtures('create_record')
class TestRecordMutability:
    pass
