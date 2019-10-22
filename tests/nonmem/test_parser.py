from pharmpy.plugins.nonmem.nmtran_parser import NMTranParser


def test_simple_parse():
    parser = NMTranParser()

    model = parser.parse('$PROBLEM MYPROB')

    assert len(model.records) == 1
    assert type(model.records[0]).__name__ == 'ProblemRecord'
    assert str(model) == '$PROBLEM MYPROB'
