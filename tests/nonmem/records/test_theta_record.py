# -*- encoding: utf-8 -*-

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pysn.parameters import Scalar


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
@pytest.mark.parametrize('buf,thetas', [
    ('THETA 0', np.array((
        Scalar(0),
    ))),
    ('THETA   12.3 \n\n', np.array((
        Scalar(12.3)
    ))),
    ('THETA  (0,0.00469) ; CL', np.array((
        Scalar(0.00469, lower=0),
    ))),
    ('THETA  (0,3) 2 FIXED (0,.6,1) 10 (-INF,-2.7,0) (37 FIXED)', np.array((
        Scalar(3, lower=0), Scalar(2, fix=True), Scalar(0.6, lower=0, upper=1),
        Scalar(10), Scalar(-2.7, upper=0), Scalar(37, fix=True),
    ))),
])
def test_create(create_record, buf, thetas):
    rec = create_record(buf)
    assert rec.name == 'THETA'
    assert_array_equal(rec.thetas, thetas)


def test_create_replicate(create_record):
    single = create_record('THETA 2 2 2 2 (0.001,0.1,1000) (0.001,0.1,1000) (0.001,0.1,1000)'
                           '       (0.5 FIXED) (0.5 FIXED)')
    multi = create_record('THETA (2)x4 (0.001,0.1,1000)x3 (0.5 FIXED)x2')
    assert_array_equal(single.thetas, multi.thetas)


@pytest.mark.usefixtures('create_record')
@pytest.mark.parametrize('buf,n,new_thetas', [
    ('THETA 0', 1, np.array((
        Scalar(1),
    ))),
    ('THETA 0', 1, np.array((
        Scalar(1.23, fix=True, upper=100),
    ))),
    ('THETA 1 2', 2, np.array((
        Scalar(1),
    ))),
    ('THETA 1 2', 2, np.array((
        Scalar(1), Scalar(0, fix=True),
        Scalar(1.2383289E2, lower=9, fix=True),
    ))),
])
def test_replace(create_record, buf, n, new_thetas):
    rec = create_record(buf)
    thetas = rec.thetas
    assert len(thetas) == n

    rec.thetas = new_thetas
    assert_array_equal(rec.thetas, new_thetas)
