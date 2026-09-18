import pytest


@pytest.mark.parametrize(
    "buf,value",
    [
        ('$SIMULATION (1212) NSUBPROBS=0', 1),
        ('$SIMULATION (1212) NSUBPROBLEMS=1', 1),
        ('$SIMULATION (1212) SUBPROB=60', 60),
        ('$SIMULATION (1212 NORMAL) SUBPROB=60', 60),
        ('$SIMULATION (1212) (898 UNIFORM) SUBPROB=60', 60),
    ],
)
def test_nsubs(parser, buf, value):
    recs = parser.parse(buf)
    rec = recs.records[0]
    assert rec.nsubs == value
