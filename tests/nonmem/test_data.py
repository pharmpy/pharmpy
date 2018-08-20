import pysn.psn.api_nonmem.data

def test_data_io(pheno_data):
    data_io = pysn.psn.api_nonmem.data.NMTRANDataIO(pheno_data, '@')
    assert data_io.read(7) == '      1' 
    data_io = pysn.psn.api_nonmem.data.NMTRANDataIO(pheno_data, 'I')
    assert data_io.read(13) == '      1    0.'
    data_io = pysn.psn.api_nonmem.data.NMTRANDataIO(pheno_data, 'Q')
    assert data_io.read(5) == 'ID TI'
