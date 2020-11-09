import pytest


@pytest.mark.parametrize(
    "buf,option,value,expected",
    [
        ('$SIZES PC=23', 'PC', 40, '$SIZES PC=40'),
        ('$SIZES LIM1=5', 'PC', 23, '$SIZES LIM1=5'),
        ('$SIZES LIM1=5', 'PC', 31, '$SIZES LIM1=5 PC=31'),
        ('$SIZES LIM1=5 PC=23', 'PC', 29, '$SIZES LIM1=5'),
    ],
)
def test_set_option(parser, buf, option, value, expected):
    recs = parser.parse(buf)
    rec = recs.records[0]
    setattr(rec, option, value)
    assert str(rec) == expected
