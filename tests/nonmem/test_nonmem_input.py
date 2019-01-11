from io import StringIO


def test_data_io(nonmem, pheno_data):
    data_io = nonmem.input.NMTRANDataIO(pheno_data, '@')
    assert data_io.read(7) == '      1'
    data_io = nonmem.input.NMTRANDataIO(pheno_data, 'I')
    assert data_io.read(13) == '      1    0.'
    data_io = nonmem.input.NMTRANDataIO(pheno_data, 'Q')
    assert data_io.read(5) == 'ID TI'


def test_data_read_data_frame(pheno):
    abc = ['A', 'B', 'C']
    inp = pheno.input.__class__
    df = inp.read_dataset(StringIO("1,2,3"), abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    assert list(df.columns) == ['A', 'B', 'C']
    df = inp.read_dataset(StringIO("1, 2   , 3"), abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    df = inp.read_dataset(StringIO("1,,3"), abc)
    assert list(df.iloc[0]) == [1, 0, 3]
    df = inp.read_dataset(StringIO("1,,"), abc)
    assert list(df.iloc[0]) == [1, 0, 0]
    df = inp.read_dataset(StringIO(",2,4"), abc)
    assert list(df.iloc[0]) == [0, 2, 4]
    df = inp.read_dataset(StringIO("1\t2\t3"), abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    df = inp.read_dataset(StringIO("1\t2\t"), abc)
    assert list(df.iloc[0]) == [1, 2, 0]
    df = inp.read_dataset(StringIO("3 4 6"), abc)
    assert list(df.iloc[0]) == [3, 4, 6]
    df = inp.read_dataset(StringIO("3   28   , 341"), abc)
    assert list(df.iloc[0]) == [3, 28, 341]

    # Mismatch length of column_names and data frame
    df = inp.read_dataset(StringIO("1,2,3"), abc + ['D'])
    assert list(df.iloc[0]) == [1, 2, 3, 0]
    assert list(df.columns) == ['A', 'B', 'C', 'D']
    df = inp.read_dataset(StringIO("1,2,3,6"), abc)
    assert list(df.iloc[0]) == [1, 2, 3, 6]
    assert list(df.columns) == ['A', 'B', 'C', None]


def test_data_read(nonmem, pheno):
    df = pheno.input.data_frame
    assert list(df.iloc[1]) == [1.0, 2.0, 0.0, 1.4, 7.0, 17.3, 0.0, 0.0]
    assert list(df.columns) == ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV', 'FA1', 'FA2']
    assert pheno.input.id_column == 'ID'


def test_input_filter(nonmem, pheno):
    filters = pheno.input.filters
    assert len(filters) == 0
