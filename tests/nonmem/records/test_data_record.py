def test_data_path(api):
    record = api.records.create_record("DATA 'pheno.dta'")
    assert record.path == 'pheno.dta'
    record = api.records.create_record('DATA "pheno.dta"')
    assert record.path == 'pheno.dta'
    record = api.records.create_record('DATA /home/full/pheno.dta')
    assert record.path == '/home/full/pheno.dta'
    record = api.records.create_record('DATA "C:\windowspath\with space in.csv"')
    assert record.path == 'C:\windowspath\with space in.csv'


def test_filter(api):
    record = api.records.create_record("DATA pheno.dta IGNORE=(ID.EQ.1) NOWIDE")    # NOWIDE before IGNORE works but not after.
    #record = api.records.create_record("DATA pheno.dta IGNORE=(ID.EQ.1,MDV.NEN.0) NOWIDE")
