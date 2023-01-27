import pytest


@pytest.mark.parametrize(
    "buf,option,old_value,value,expected",
    [
        ('$SIZES PC=23', 'PC', 23, 40, '$SIZES PC=40'),
        ('$SIZES LIM1=5', 'PC', 30, 23, '$SIZES LIM1=5'),
        ('$SIZES LIM1=5', 'PC', 30, 31, '$SIZES LIM1=5 PC=31'),
        ('$SIZES LIM1=5 PC=23', 'PC', 23, 29, '$SIZES LIM1=5'),
        ('$SIZES LTH=120', 'LTH', 120, 121, '$SIZES LTH=121'),
        ('$SIZES LIM1=5', 'LTH', 100, 120, '$SIZES LIM1=5 LTH=120'),
    ],
)
def test_set_option(parser, buf, option, old_value, value, expected):
    recs = parser.parse(buf)
    rec = recs.records[0]
    assert getattr(rec, option) == old_value
    if option == 'PC':
        newrec = rec.set_PC(value)
    elif option == 'LTH':
        newrec = rec.set_LTH(value)
    assert str(newrec) == expected


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
        rec.set_PC(value)
