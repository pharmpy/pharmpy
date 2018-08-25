# -*- encoding: utf-8 -*-

# from collections import OrderedDict

import pytest


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


# -- ONLY PARSER -----------------------------------------------------------------------------------

# Experiment with sandboxing testing where grammar changes can flag before unit tests on API needs
# to step in. Might be unnecessary? Not removing, but next record won't get this luxury, I'm sure...
# See :func:`parse_assert`, :class:`GeneratedTheta` and :class:`RandomThetas` for how it works.


def test_single_inits(parse_assert, RandomData, GeneratedTheta):
    for val in RandomData(5).int():
        parse_assert(str(val), [GeneratedTheta.new(init=val)])

    for val in RandomData(5).float():
        parse_assert(str(val), [GeneratedTheta.new(init=val)])


def test_padded_inits(parse_assert, RandomData, GeneratedTheta):
    data = RandomData(5)
    for lpad, val, rpad in zip(data.pad(), data.float(), data.pad()):
        parse_assert(str(lpad) + str(val) + str(rpad), [GeneratedTheta.new(init=val)])

    data = RandomData(5)
    for lpad, val, rpad in zip(data.pad(nl=True), data.float(), data.pad(nl=True)):
        parse_assert(str(lpad) + str(val) + str(rpad), [GeneratedTheta.new(init=val)])

    data = RandomData(5)
    for val in data.pad(nl=True):
        parse_assert(str(val), [])


def test_comments(parse_assert, RandomData):
    bufs, comments = [], []
    data = RandomData(10)
    for lpad, comment in zip(data.pad(nl=True), data.comment()):
        bufs += [lpad + comment]
        comments += [comment.strip().lstrip(';').strip()]
    buf = '\n'.join(bufs)
    root = parse_assert(buf, [])
    nodes = filter(lambda x: x.rule == 'comment', root.tree_walk())
    assert comments == list(map(lambda x: getattr(x, 'TEXT'), nodes))


def test_messy_random(parse_assert, RandomThetas):
    bufs, thetas = [], []
    for i, theta in enumerate(RandomThetas(20).theta()):
        thetas += [theta]
        bufs += [str(theta)]
        print(bufs[-1])
    buf = '\n'.join(bufs)
    parse_assert(buf, thetas)


# -- RECORD CLASS ----------------------------------------------------------------------------------


@pytest.mark.usefixtures('create_record')
@pytest.mark.parametrize('buf,theta_dicts', [
    ('THET', []),
    ('THETA 0', [dict(init=0)]),
    ('THETA   12.3 \n\n', [dict(init=12.3)]),
    ('THETA  (0,0.00469) ; CL', [dict(lower_bound=0, init=0.00469)]),
])
def test_create(create_record, buf, theta_dicts):
    rec = create_record(buf)
    assert rec.name == 'THETA'
    for i, ref in enumerate(theta_dicts):
        rec_dict = {k: getattr(rec.thetas[i], k) for k in ref.keys()}
        assert rec_dict == ref


# @pytest.mark.usefixtures('create_record')
# @pytest.mark.parametrize('buf,theta_dicts', [
#     ('THETA 0', [dict(init=1)]),
# ])
# def test_modify(create_record, buf, theta_dicts):
#     rec = create_record(buf)
#     thetas = rec.thetas
#     for i, key_val in enumerate(theta_dicts):
#         thetas[i] = thetas[i]._replace(**key_val)
#     rec.thetas = thetas
#
#     for theta, ref in zip(rec.thetas, theta_dicts):
#         rec_dict = {k: getattr(theta, k) for k in ref.keys()}
#         assert rec_dict == ref
