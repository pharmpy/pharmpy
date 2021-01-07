from pharmpy import Model


def test_data_filename_set(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_abbr.mod')
    assert model.control_stream.abbreviated.replace == {
        'THETA(CL)': 'THETA(1)',
        'THETA(V)': 'THETA(2)',
        'ETA(CL)': 'ETA(1)',
        'ETA(V)': 'ETA(2)',
    }
