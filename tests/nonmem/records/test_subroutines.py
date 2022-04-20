import pytest


@pytest.mark.parametrize(
    "buf,advan,trans",
    [
        ('$SUBROUTINES ADVAN1 TRANS2', 'ADVAN1', 'TRANS2'),
        ('$SUBROUTINES ADVAN15', 'ADVAN15', 'TRANS1'),
        ('$SUBROUTINES ADVAN=ADVAN2', 'ADVAN2', 'TRANS1'),
        ('$SUBROUTINES ADVAN=ADVAN5 TRANS=TRANS1', 'ADVAN5', 'TRANS1'),
    ],
)
def test_advan_trans(parser, buf, advan, trans):
    recs = parser.parse(buf)
    rec = recs.records[0]
    assert rec.advan == advan
    assert rec.trans == trans
