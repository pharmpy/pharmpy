from pharmpy.modeling import (
    add_metabolite,
    add_peripheral_compartment,
    has_presystemic_metabolite,
    remove_peripheral_compartment,
)


def test_add_metabolite(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pheno_conc.mod')
    model = add_metabolite(model)
    odes = model.statements.ode_system
    assert odes.compartment_names == ['CENTRAL', 'METABOLITE']
    assert not has_presystemic_metabolite(model)
    a = model.code.split('\n')
    assert a[20] == 'IF (DVID.EQ.1) Y = Y'
    assert a[21] == 'IF (DVID.EQ.2) Y = Y_M'

    assert odes.central_compartment.name == 'CENTRAL'

    model = add_peripheral_compartment(model, "METABOLITE")
    odes = model.statements.ode_system

    assert odes.find_peripheral_compartments("METABOLITE")
    assert not odes.find_peripheral_compartments()

    model = add_peripheral_compartment(model)
    odes = model.statements.ode_system

    assert odes.find_peripheral_compartments("METABOLITE")
    assert odes.find_peripheral_compartments()

    model = remove_peripheral_compartment(model, "METABOLITE")
    odes = model.statements.ode_system

    assert not odes.find_peripheral_compartments("METABOLITE")
    assert odes.find_peripheral_compartments()

    model = remove_peripheral_compartment(model)
    odes = model.statements.ode_system

    assert not odes.find_peripheral_compartments("METABOLITE")
    assert not odes.find_peripheral_compartments()


def test_presystemic_metabolite(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pheno_conc.mod')
    model = add_metabolite(model, presystemic=True)
    odes = model.statements.ode_system

    assert odes.compartment_names == ['DEPOT', 'CENTRAL', 'METABOLITE']

    depot = odes.find_depot(model.statements)
    central = odes.central_compartment
    metabolite = odes.find_compartment("METABOLITE")

    assert has_presystemic_metabolite(model)
    assert odes.get_flow(depot, central)
    assert odes.get_flow(depot, metabolite)
    assert odes.get_flow(central, metabolite)

    assert odes.central_compartment.name == 'CENTRAL'

    model = add_peripheral_compartment(model, "METABOLITE")
    odes = model.statements.ode_system

    assert odes.find_peripheral_compartments("METABOLITE")
    assert not odes.find_peripheral_compartments()

    model = add_peripheral_compartment(model)
    odes = model.statements.ode_system

    assert odes.find_peripheral_compartments("METABOLITE")
    assert odes.find_peripheral_compartments()

    model = remove_peripheral_compartment(model, "METABOLITE")
    odes = model.statements.ode_system

    assert not odes.find_peripheral_compartments("METABOLITE")
    assert odes.find_peripheral_compartments()

    model = remove_peripheral_compartment(model)
    odes = model.statements.ode_system

    assert not odes.find_peripheral_compartments("METABOLITE")
    assert not odes.find_peripheral_compartments()
