import pytest

from pharmpy.model.external.nonmem.nmtran_parser import NMTranParser
from pharmpy.model.external.nonmem.parsing import parse_table_columns


def test_simple_parse():
    parser = NMTranParser()

    model = parser.parse('$PROBLEM MYPROB\n')

    assert len(model.records) == 1
    assert type(model.records[0]).__name__ == 'ProblemRecord'
    assert str(model) == '$PROBLEM MYPROB\n'

    model2_str = ';Comment\n   $PROBLEM     TW2\n'
    model2 = parser.parse(model2_str)

    assert len(model2.records) == 2
    assert type(model2.records[0]).__name__ == 'RawRecord'
    assert type(model2.records[1]).__name__ == 'ProblemRecord'

    assert str(model2) == model2_str


def test_round_trip(pheno_path):
    parser = NMTranParser()

    with open(pheno_path, 'r') as fh:
        content = fh.read()
    model = parser.parse(content)
    assert str(model) == content


@pytest.mark.parametrize(
    'buf,columns',
    [
        ('$TABLE ID TIME', [['ID', 'TIME', 'DV', 'PRED', 'RES', 'WRES']]),
        ('$TABLE ID TIME NOAPPEND', [['ID', 'TIME']]),
        (
            '$TABLE ID TIME CIPREDI=CONC NOAPPEND',
            [['ID', 'TIME', 'CIPREDI']],
        ),
        (
            '$TABLE ID TIME CONC=CIPREDI NOAPPEND',
            [['ID', 'TIME', 'CIPREDI']],
        ),
        (
            '$TABLE ID TIME CONC=CIPREDI NOAPPEND\n$TABLE ID TIME CONC',
            [['ID', 'TIME', 'CIPREDI'], ['ID', 'TIME', 'CIPREDI', 'DV', 'PRED', 'RES', 'WRES']],
        ),
    ],
)
def test_table_columns(buf, columns):
    parser = NMTranParser()

    cs = parser.parse(buf)
    cs._active_problem = -1  # To trick cs.get_records
    parsed_columns = parse_table_columns(cs, netas=2)
    assert parsed_columns == columns
