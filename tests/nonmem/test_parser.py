from pharmpy.plugins.nonmem.nmtran_parser import NMTranParser


def test_simple_parse():
    parser = NMTranParser()

    model = parser.parse('$PROBLEM MYPROB')

    assert len(model.records) == 1
