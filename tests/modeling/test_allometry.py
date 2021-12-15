import pytest

from pharmpy.modeling import add_allometry, read_model
from pharmpy.statements import Assignment


def test_allometry(testdata):
    model = read_model(testdata / 'nonmem' / 'pheno.mod')
    ref_model = model.copy()
    add_allometry(
        model,
        allometric_variable='WGT',
        reference_value=70,
        parameters=['CL'],
        initials=[0.7],
        lower_bounds=[0],
        upper_bounds=[2],
        fixed=True,
    )
    assert model.statements[1] == Assignment('CL', 'CL*(WGT/70)**ALLO_CL')
    assert model.parameters['ALLO_CL'].init == 0.7
    assert model.parameters['ALLO_CL'].lower == 0
    assert model.parameters['ALLO_CL'].upper == 2

    with pytest.raises(ValueError):
        add_allometry(
            model,
            allometric_variable='WGT',
            reference_value=70,
            parameters=['CL'],
            initials=[0.7, 23],
            lower_bounds=[0],
            upper_bounds=[2],
            fixed=True,
        )

    with pytest.raises(ValueError):
        add_allometry(
            model,
            allometric_variable='WGT',
            reference_value=70,
            parameters=['CL'],
            initials=[0.7],
            lower_bounds=[1, 2],
            upper_bounds=[2],
            fixed=True,
        )

    with pytest.raises(ValueError):
        add_allometry(
            model,
            allometric_variable='WGT',
            reference_value=70,
            parameters=['CL'],
            initials=[0.7],
            lower_bounds=[1],
            upper_bounds=[1, 2],
            fixed=True,
        )

    with pytest.raises(ValueError):
        add_allometry(model, allometric_variable='WGT', reference_value=70, parameters=[])

    model = ref_model.copy()

    add_allometry(
        model,
        allometric_variable='WGT',
        reference_value=70,
        parameters=['CL', 'V'],
        initials=[0.7, 0.8],
        lower_bounds=[0, 0.1],
        upper_bounds=[2, 3],
        fixed=True,
    )
    assert model.statements[1] == Assignment('CL', 'CL*(WGT/70)**ALLO_CL')
    assert model.statements[3] == Assignment('V', 'V*(WGT/70)**ALLO_V')
    assert model.parameters['ALLO_V'].init == 0.8

    model = ref_model.copy()
    add_allometry(model, allometric_variable='WGT', reference_value=70, parameters=['CL', 'V'])
    assert model.parameters['ALLO_CL'].init == 0.75
    assert model.parameters['ALLO_V'].init == 1

    model = ref_model.copy()
    add_allometry(model, allometric_variable='WGT', reference_value=70)
    assert model.parameters['ALLO_CL'].init == 0.75
    assert model.parameters['ALLO_V'].init == 1

    model = read_model(testdata / 'nonmem' / 'models' / 'pheno_trans1.mod')
    with pytest.raises(ValueError):
        add_allometry(model, allometric_variable='WGT', reference_value=70)

    model = read_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan3.mod')
    add_allometry(model, allometric_variable='WGT', reference_value=70)
    assert model.parameters['ALLO_Q'].init == 0.75

    model = read_model(testdata / 'nonmem' / 'models' / 'pheno_advan3_trans1.mod')
    with pytest.raises(ValueError):
        add_allometry(model, allometric_variable='WGT', reference_value=70)

    add_allometry(
        model,
        allometric_variable='WGT',
        reference_value=70,
        parameters=['K'],
        initials=[1],
        lower_bounds=[0],
        upper_bounds=[1],
    )
