import pytest

from pharmpy.model.external.nonmem.records.factory import create_record


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


@pytest.mark.parametrize(
    "code",
    [
        "$SIMULATION (1413) (123421 UNIFORM) ONLYSIM NSUB=200\n",
        "$SIMULATION (1137034) (6222994 UNIFORM) ONLYSIMULATION NOPREDICTION NSUB=200\n",
        "$SIMULATION (1525618458) (11111 UNIFORM) ONLYSIMULATION NOPREDICTION NSUBPROBLEMS=200 PARAFILE=ON\n",
    ],
)
def test_simulation_record_round_trips(code):
    rec = create_record(code)
    assert str(rec) == code
