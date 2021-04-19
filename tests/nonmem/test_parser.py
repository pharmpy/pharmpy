from pharmpy.plugins.nonmem.nmtran_parser import NMTranParser


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
