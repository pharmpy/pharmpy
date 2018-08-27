def test_data_path(api):
    record = api.records.create_record('DATA "pheno.dta"')
    assert record.path == 'pheno.dta'

    record = api.records.create_record('DATA /home/full/pheno.dta')
    assert record.path == '/home/full/pheno.dta'

    record = api.records.create_record("DATA 'pheno.dta'")
    assert str(record.root.filename) == "'pheno.dta'"
    assert record.path == "pheno.dta"

    record = api.records.create_record('DATA "C:\windowspath\with space in.csv"')
    assert record.path == 'C:\windowspath\with space in.csv'

    record = api.records.create_record("DATA \n pheno.dta \n; comment\n")
    print(record.parser)
    assert record.path == 'pheno.dta'


def test_filter(api):
    record = api.records.create_record('DATA pheno.dta NOWIDE')
    print(record.parser)
    assert str(record.root.filename) == 'pheno.dta'
    assert str(record.root.find('option')) == 'NOWIDE'

    record = api.records.create_record("DATA  'pheno.dta'  IGNORE=(ID.EQ.1,MDV.NEN.0) ")
    print(record.parser)
    assert str(record.root.filename) == "'pheno.dta'"
    assert str(record.root.ignore) == 'IGNORE=(ID.EQ.1,MDV.NEN.0)'

    record = api.records.create_record('DATA "pheno.dta" NOWIDE IGNORE=(ID==1)')
    print(record.parser)
    assert str(record.root.filename) == '"pheno.dta"'
    assert str(record.root.find('option')) == 'NOWIDE'
    assert str(record.root.ignore) == 'IGNORE=(ID==1)'

    record = api.records.create_record("DATA      pheno.dta IGNORE=@\n")
    print(record.parser)
    assert str(record.root.filename) == 'pheno.dta'
    assert str(record.root.ignore.char) == '@'
