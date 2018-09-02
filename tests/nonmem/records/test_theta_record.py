# -*- encoding: utf-8 -*-

from collections import namedtuple

import pytest


@pytest.fixture
def parse_assert(nonmem, GeneratedTheta):
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
        tree = nonmem.records.parser.ThetaRecordParser(buf)
        print(str(tree) + '\n')
        root = tree.root

        assert str(root) == buf
        assert GeneratedTheta.from_tree(root) == reference
        return root

    return func


# -- ONLY PARSER ---------------------------------------------------------

# Experiment with sandboxing testing where grammar changes can flag before unit tests on API needs
# to step in. Might be unnecessary? Not removing, but next record won't get this luxury, I'm sure...
# See :func:`parse_assert`, :class:`GeneratedTheta` and
# :class:`RandomThetas` for how it works.


# def test_single_inits(parse_assert, RandomData, GeneratedTheta):
#     for val in RandomData(5).int():
#         parse_assert(str(val), [GeneratedTheta.new(init=val)])
#
#     for val in RandomData(5).float():
#         parse_assert(str(val), [GeneratedTheta.new(init=val)])
#
#
# def test_padded_inits(parse_assert, RandomData, GeneratedTheta):
#     data = RandomData(5)
#     for lpad, val, rpad in zip(data.pad(), data.float(), data.pad()):
#         parse_assert(str(lpad) + str(val) + str(rpad),
#                      [GeneratedTheta.new(init=val)])
#
#     data = RandomData(5)
#     for lpad, val, rpad in zip(
#         data.pad(
#             nl=True), data.float(), data.pad(
#             nl=True)):
#         parse_assert(str(lpad) + str(val) + str(rpad),
#                      [GeneratedTheta.new(init=val)])
#
#     data = RandomData(5)
#     for val in data.pad(nl=True):
#         parse_assert(str(val), [])
#
#
# def test_comments(parse_assert, RandomData):
#     bufs, comments = [], []
#     data = RandomData(10)
#     for lpad, comment in zip(data.pad(nl=True), data.comment()):
#         bufs += [lpad + comment]
#         comments += [comment.strip().lstrip(';').strip()]
#     buf = '\n'.join(bufs)
#     root = parse_assert(buf, [])
#     nodes = filter(lambda x: x.rule == 'comment', root.tree_walk())
#     assert comments == list(map(lambda x: getattr(x, 'TEXT'), nodes))
#
#
# def test_messy_random(parse_assert, RandomThetas):
#     bufs, thetas = [], []
#     for i, theta in enumerate(RandomThetas(20).theta()):
#         thetas += [theta]
#         bufs += [str(theta)]
#         print(bufs[-1])
#     buf = '\n'.join(bufs)
#     parse_assert(buf, thetas)


# -- RECORD CLASS --------------------------------------------------------


@pytest.mark.usefixtures('create_record')
@pytest.mark.parametrize('buf,theta_dicts', [
    ('THETA 0', [dict(init=0)]),
    ('THETA   12.3 \n\n', [dict(init=12.3)]),
    ('THETA  (0,0.00469) ; CL', [dict(low=0, init=0.00469)]),
    ('THETA  (0,3) 2 FIXED (0,.6,1) 10 (-INF,-2.7,0) (37 FIXED)', [
        dict(low=0, init=3), dict(init=2, fix=True), dict(low=0, init=0.6, up=1), dict(init=10),
        dict(low=float('-INF'), init=-2.7, up=0), dict(init=37, fix=True)]
     )
])
def test_create(create_record, buf, theta_dicts):
    print('buf =', repr(buf))
    rec = create_record(buf)
    assert rec.name == 'THETA'
    for i, ref in enumerate(theta_dicts):
        rec_dict = {k: getattr(rec.thetas[i], k) for k in ref.keys()}
        assert rec_dict == ref


def test_create_replicate(create_record):
    single = create_record('THETA 2 2 2 2 (0.001,0.1,1000) (0.001,0.1,1000) (0.001,0.1,1000)'
                           '       (0.5 FIXED) (0.5 FIXED)')
    multi = create_record('THETA (2)x4 (0.001,0.1,1000)x3 (0.5 FIXED)x2')
    for theta_1, theta_2 in zip(single.thetas, multi.thetas):
        assert theta_1.low == theta_2.low
        assert theta_1.init == theta_2.init
        assert theta_1.up == theta_2.up
        assert theta_1.fix == theta_2.fix


@pytest.mark.usefixtures('create_record')
@pytest.mark.parametrize('buf,n,theta_dicts', [
    ('THETA 0', 1, [dict(init=1)]),
    ('THETA 0', 1, [dict(low=float('-inf'), init=1.23, up=100, fix=True)]),
    ('THETA 1 2', 2, [dict(init=1)]),
    ('THETA 1 2', 2, [dict(init=1), dict(init=0, fix=True),
                      dict(low=9, init=1.2383289E2, fix=True)]),
])
def test_replace(create_record, buf, n, theta_dicts):
    rec = create_record(buf)
    thetas = rec.thetas
    assert len(thetas) == n

    args = ('low', 'init', 'up', 'fix', 'node')
    theta = namedtuple('ThetaInit', args)._make([None]*len(args))

    thetas = [theta._replace(**dict_) for dict_ in theta_dicts]
    rec.thetas = thetas

    for theta, ref in zip(rec.thetas, theta_dicts):
        rec_dict = {k: getattr(theta, k) for k in ref.keys()}
        assert rec_dict == ref
