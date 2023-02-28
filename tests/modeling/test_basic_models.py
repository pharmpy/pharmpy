from pharmpy.modeling import create_basic_pk_model


def test_create_basic_pk_model():
    model = create_basic_pk_model('iv')
    assert len(model.parameters) == 6

    model = create_basic_pk_model('oral')
    print(model.parameters)
    assert len(model.parameters) == 8
