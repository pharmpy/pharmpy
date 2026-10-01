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
        "$SIM (2424) ONLYSIM\n",
        "$SIM NSUB=2 (12345) ONLYSIM\n",
        "$SIMULATION (193081) SUBPROB=1\n",
        "$SIMULATION (1413) ONLYSIM NSUB=50\n",
        "$SIMULATION (1231) ONLYSIM NSUB=100\n",
        "$SIMULATION (1231) ONLYSIM NSUB = 100\n",
        "$SIMULATION (1413) (123421 UNIFORM) ONLYSIM NSUB=200\n",
        "$SIMULATION (1137034) (6222994 UNIFORM) ONLYSIMULATION NOPREDICTION NSUB=200\n",
        "$SIMULATION (1525618458) (11111 UNIFORM) ONLYSIMULATION NOPREDICTION NSUBPROBLEMS=200 PARAFILE=ON\n",
        "$SIMULATION ONLYSIM (1413) NSUB=200\n",
        "$SIMULATION (1234 NORMAL) NSUB=10\n",
        "$SIMULATION (1234 UNIFORM) NSUB=10\n",
        "$SIMULATION (1234 NONPARAMETRIC) NSUB=10\n",
        "$SIMULATION (1234 NORM) NSUB=10\n",
        "$SIMULATION (1234 UNIF) NSUB=10\n",
        "$SIMULATION (1234) (5678 NORMAL) NSUB=10\n",
        "$SIMULATION (1234) (5678 UNIFORM) NSUB=10\n",
        "$SIMULATION (1234 NEW) NSUB=10\n",
        "$SIMULATION (1234 UNIFORM NEW) NSUB=10\n",
        "$SIMULATION (1234) (5678 UNIFORM NEW) NSUB=10\n",
        "$SIMULATION (1234) RANMETHOD=3\n",
        "$SIMULATION (1234) RANMETHOD=3S2P\n",
        "$SIMULATION (1234) RANMETHOD=P\n",
        "$SIMULATION (1234) PARAFILE=ON\n",
        "$SIMULATION (1234) PARAFILE=OFF\n",
        "$SIMULATION (1234) PARAFILE=myparafile.txt\n",
        "$SIMULATION (1234) TRUE=INITIAL\n",
        "$SIMULATION (1234) TRUE=FINAL\n",
        "$SIMULATION (1234) TRUE=PRIOR\n",
        "$SIMULATION (1234) OMITTED\n",
        "$SIMULATION (1234) PREDICTION\n",
        "$SIMULATION (1234) NOPREDICTION\n",
        "$SIMULATION (1234) REQUESTFIRST\n",
        "$SIMULATION (1234) REQUESTSECOND\n",
        "$SIMULATION (1234) BOOTSTRAP=100\n",
        "$SIMULATION (1234) BOOTSTRAP=100 REPLACE\n",
        "$SIMULATION (1234) BOOTSTRAP=100 NOREPLACE\n",
        "$SIMULATION (1234) BOOTSTRAP=100 STRAT=SEX\n",
        "$SIMULATION (1234) BOOTSTRAP=100 STRATF=GROUP\n",
        "$SIMULATION (1234) REWIND\n",
        "$SIMULATION (1234) NOREWIND\n",
        "$SIMULATION (1234) SUPRESET\n",
        "$SIMULATION (1234) NOSUPRESET\n",
        "$SIMULATION (1234 UNIFORM NEW) RANMETHOD=3S2P PARAFILE=ON TRUE=FINAL\n",
        "$SIMULATION (1234 NORMAL) RANMETHOD=P PARAFILE=OFF TRUE=INITIAL NSUB=50\n",
        "$SIMULATION (1234) (5678 UNIFORM NEW) RANMETHOD=1 PARAFILE=run1.par TRUE=PRIOR ONLYSIM NSUB=200\n",
        "$SIMULATION NSUB=200 (1234 UNIFORM) TRUE=FINAL PARAFILE=ON\n",
        "$SIMULATION TRUE=FINAL PARAFILE=ON (1234 UNIFORM) NSUB=200\n",
        "$SIMULATION PARAFILE=ON TRUE=FINAL RANMETHOD=3S2P (1234 NEW) NSUB=200\n",
        "$SIMULATION RANMETHOD=3S2P (1234 NEW) PARAFILE=ON NSUB=200 TRUE=FINAL\n",
    ],
)
def test_simulation_record_round_trips(code):
    rec = create_record(code)
    assert str(rec) == code


@pytest.mark.parametrize(
    "code,expected",
    [
        ("$SIMULATION (1413) ONLYSIM\n", 1),
        ("$SIMULATION (1413) NSUB=0\n", 1),
        ("$SIM NSUB=2 (12345) ONLYSIM\n", 2),
        ("$SIMULATION (193081) SUBPROB=1\n", 1),
        ("$SIMULATION (1231) ONLYSIM NSUB=100\n", 100),
        ("$SIMULATION (1231) ONLYSIM NSUB = 100\n", 100),
        (
            "$SIMULATION (1525618458) (11111 UNIFORM) ONLYSIMULATION NOPREDICTION NSUBPROBLEMS=200 PARAFILE=ON\n",
            200,
        ),
        (
            "$SIMULATION (1234) (5678 UNIFORM NEW) RANMETHOD=1 PARAFILE=run1.par TRUE=PRIOR ONLYSIM NSUB=200\n",
            200,
        ),
        ("$SIMULATION NSUB=200 (1234 UNIFORM) TRUE=FINAL PARAFILE=ON\n", 200),
    ],
)
def test_combination_nsubs(code, expected):
    rec = create_record(code)
    assert rec.nsubs == expected
