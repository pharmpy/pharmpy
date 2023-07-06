import pytest

from pharmpy.model import Assignment
from pharmpy.modeling import add_allometry, add_peripheral_compartment, set_tmdd


def test_allometry(load_model_for_test, testdata):
    ref_model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')

    model = add_allometry(
        ref_model,
        allometric_variable='WGT',
        reference_value=70,
        parameters=['CL'],
        initials=[0.7],
        lower_bounds=[0],
        upper_bounds=[2],
        fixed=True,
    )
    assert model.statements[1] == Assignment.create('CL', 'CL*(WGT/70)**ALLO_CL')
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

    model = add_allometry(
        ref_model,
        allometric_variable='WGT',
        reference_value=70,
        parameters=['CL', 'V'],
        initials=[0.7, 0.8],
        lower_bounds=[0, 0.1],
        upper_bounds=[2, 3],
        fixed=True,
    )
    assert model.statements[1] == Assignment.create('CL', 'CL*(WGT/70)**ALLO_CL')
    assert model.statements[3] == Assignment.create('V', 'V*(WGT/70)**ALLO_V')
    assert model.parameters['ALLO_V'].init == 0.8

    model = add_allometry(
        ref_model, allometric_variable='WGT', reference_value=70, parameters=['CL', 'V']
    )
    assert model.parameters['ALLO_CL'].init == 0.75
    assert model.parameters['ALLO_V'].init == 1

    model = add_allometry(ref_model, allometric_variable='WGT', reference_value=70)
    assert model.parameters['ALLO_CL'].init == 0.75
    assert model.parameters['ALLO_V'].init == 1

    model = add_peripheral_compartment(ref_model)
    model = add_allometry(model, allometric_variable='WGT', reference_value=70)
    assert model.statements[1] == Assignment.create('VP1', 'VP1*(WGT/70)**ALLO_VP1')
    assert model.statements[3] == Assignment.create('QP1', 'QP1*(WGT/70)**ALLO_QP1')
    assert model.statements[5] == Assignment.create('CL', 'CL*(WGT/70)**ALLO_CL')
    assert model.statements[7] == Assignment.create('V1', 'V1*(WGT/70)**ALLO_V1')
    assert model.parameters['ALLO_VP1'].init == 1.0
    assert model.parameters['ALLO_QP1'].init == 0.75
    model = add_peripheral_compartment(model)
    model = add_allometry(model, allometric_variable='WGT', reference_value=70)
    assert model.statements[1] == Assignment.create('VP2', 'VP2*(WGT/70)**ALLO_VP2')
    assert model.statements[3] == Assignment.create('QP2', 'QP2*(WGT/70)**ALLO_QP2')
    assert model.parameters['ALLO_VP2'].init == 1.0
    assert model.parameters['ALLO_QP2'].init == 0.75

    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pheno_trans1.mod')
    with pytest.raises(ValueError):
        add_allometry(model, allometric_variable='WGT', reference_value=70)

    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan3.mod')
    model = add_allometry(model, allometric_variable='WGT', reference_value=70)
    assert model.parameters['ALLO_Q'].init == 0.75
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pheno_advan3_trans1.mod')
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


def test_add_allometry_tmdd(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = set_tmdd(model, type="full")
    add_allometry(
        model,
        allometric_variable='WGT',
        reference_value=70,
    )

    # FIXME: currently adds to CL and LAFREE, is this correct?
    model = load_model_for_test(pheno_path)
    model = set_tmdd(model, type="qss")
    add_allometry(
        model,
        allometric_variable='WGT',
        reference_value=70,
    )
