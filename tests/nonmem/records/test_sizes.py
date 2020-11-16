import pytest


@pytest.mark.parametrize(
    "buf,option,old_value,value,expected",
    [
        ('$SIZES PC=23', 'PC', 23, 40, '$SIZES PC=40'),
        ('$SIZES LIM1=5', 'PC', 30, 23, '$SIZES LIM1=5'),
        ('$SIZES LIM1=5', 'PC', 30, 31, '$SIZES LIM1=5 PC=31'),
        ('$SIZES LIM1=5 PC=23', 'PC', 23, 29, '$SIZES LIM1=5'),
    ],
)
def test_set_option(parser, buf, option, old_value, value, expected):
    recs = parser.parse(buf)
    rec = recs.records[0]
    assert getattr(rec, option) == old_value
    setattr(rec, option, value)
    assert str(rec) == expected


@pytest.mark.parametrize(
    "buf,option,value,ex",
    [
        ('$SIZES PC=23', 'PC', 100, ValueError),
    ],
)
def test_exceptions(parser, buf, option, value, ex):
    recs = parser.parse(buf)
    rec = recs.records[0]
    with pytest.raises(ex):
        setattr(rec, option, value)
