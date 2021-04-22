from pharmpy import Model


def test_data_filename_set(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_abbr.mod')
    assert model.control_stream.abbreviated.replace == {
        'THETA(CL)': 'THETA(1)',
        'THETA(V)': 'THETA(2)',
        'ETA(CL)': 'ETA(1)',
        'ETA(V)': 'ETA(2)',
    }


def test_replace(parser):
    rec = parser.parse('$ABBR REPLACE K34="3,4"').records[0]
    assert rec.replace == {'K34': '3,4'}


def test_translate_to_pharmpy_names(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_abbr.mod')
    new_dict = model.control_stream.abbreviated.translate_to_pharmpy_names()
    assert new_dict == {
        'THETA(1)': 'THETA_CL',
        'THETA(2)': 'THETA_V',
        'ETA(1)': 'ETA_CL',
        'ETA(2)': 'ETA_V',
    }
