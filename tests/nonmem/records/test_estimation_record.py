import pytest


@pytest.mark.parametrize(
    "buf,like,llike",
    [
        ('$ESTIMATION METHOD=1 LIKELIHOOD', True, False),
        ('$ESTIMATION LIKE METHOD=1', True, False),
        ('$ESTIMATION METHOD=1', False, False),
        ('$ESTIMATION METHOD=1 -2LOGLIKELIHOOD', False, True),
        ('$ESTIMATION METHOD=1 -2LL', False, True),
        ('$ESTIMATION METHOD=1 -2LLIKELIHOOD', False, True),
    ],
)
def test_likelihood(parser, buf, like, llike):
    recs = parser.parse(buf)
    rec = recs.records[0]
    assert rec.likelihood == like
    assert rec.loglikelihood == llike
