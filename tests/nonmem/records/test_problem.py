import pytest


# FIXME: Change parsing of $PROBLEM. First row is free form. Rest can be comments or empty lines
@pytest.mark.parametrize(
    'buf,title,new_title,new_str',
    [
        ('$PROB\n', '', 'PHENO  MODEL', '$PROB PHENO  MODEL\n'),
        ('$PROBLEM ABC\n', 'ABC', None, None),
        ('$PROBLEM   A;BC \n\n', 'A;BC ', "New", "$PROBLEM New\n\n"),
        ('$PROBLEM A\n', 'A', None, None),
        ('$PROBLEM ABC \n', 'ABC ', None, None),
        ('$PROBL A ; B ; C \n', 'A ; B ; C ', None, None),
        ('$PROBL A ; B \n', 'A ; B ', None, None),
        ('$PROBLEM \n ', '', None, None),
        ('$PROBLEM \n \n', '', None, None),
    ],
)
def test_modify_string(parser, buf, title, new_title, new_str):
    recs = parser.parse(buf)
    rec = recs.records[0]
    assert str(rec) == buf
    assert rec.title == title
    if new_title:
        rec.title = new_title
        assert rec.title == new_title
        assert str(rec) == new_str
