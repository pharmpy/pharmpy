from collections import OrderedDict

from pysn.input import InputFilterOperator


def test_data_path(nonmem):
    record = nonmem.records.create_record('DATA "pheno.dta"')
    assert record.path == 'pheno.dta'

    record = nonmem.records.create_record('DATA /home/full/pheno.dta')
    assert record.path == '/home/full/pheno.dta'

    record = nonmem.records.create_record("DATA 'pheno.dta'")
    assert str(record.root.filename) == "'pheno.dta'"
    assert record.path == "pheno.dta"

    record = nonmem.records.create_record('DATA "C:\windowspath\with space in.csv"')
    assert record.path == 'C:\windowspath\with space in.csv'

    record = nonmem.records.create_record("DATA \n pheno.dta \n; comment\n")
    print(record.parser)
    assert record.path == 'pheno.dta'


def test_filter(nonmem):
    record = nonmem.records.create_record('DATA pheno.dta NOWIDE')
    print(record.parser)
    assert str(record.root.filename) == 'pheno.dta'
    assert str(record.root.find('option')) == 'NOWIDE'
    assert record.ignore_character is None

    record = nonmem.records.create_record("DATA  'pheno.dta'  IGNORE=(ID.EQ.1,MDV.NEN.0) ")
    print(record.parser)
    assert str(record.root.filename) == "'pheno.dta'"
    assert str(record.root.ignore) == 'IGNORE=(ID.EQ.1,MDV.NEN.0)'
    assert record.ignore_character is None
    filters = record.filters
    assert filters[0].symbol == "ID"
    assert filters[0].value == "1"
    assert filters[0].operator == InputFilterOperator.STRING_EQUAL
    assert filters[1].symbol == "MDV"
    assert filters[1].value == "0"
    assert filters[1].operator == InputFilterOperator.NOT_EQUAL

    record = nonmem.records.create_record('DATA "pheno.dta" NOWIDE IGNORE=(ID==1)')
    print(record.parser)
    assert str(record.root.filename) == '"pheno.dta"'
    assert str(record.root.find('option')) == 'NOWIDE'
    assert str(record.root.ignore) == 'IGNORE=(ID==1)'
    assert record.ignore_character is None

    record = nonmem.records.create_record("DATA      pheno.dta IGNORE=@\n")
    print(record.parser)
    assert str(record.root.filename) == 'pheno.dta'
    assert record.ignore_character == '@'

def test_option_record(nonmem):
    record = nonmem.records.create_record('DATA pheno.dta NOWIDE')
    assert record.option_pairs == OrderedDict([('NOWIDE', None)])
