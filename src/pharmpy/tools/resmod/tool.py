import sympy

import pharmpy.model
import pharmpy.tools
from pharmpy import Parameter, Parameters, RandomVariable, RandomVariables
from pharmpy.modeling import set_iiv_on_ruv, set_power_on_ruv
from pharmpy.statements import Assignment, ModelStatements
from pharmpy.tools.workflows import Task, Workflow


class Resmod(pharmpy.tools.Tool):
    def __init__(self, model):
        self.model = model
        super().__init__()
        self.model.database = self.database.model_database

    def run(self):
        wf = self.create_workflow()
        res = self.dispatcher.run(wf, self.database)
        # res.to_json(path=self.database.path / 'results.json')
        # res.to_csv(path=self.database.path / 'results.csv')
        return res

    def create_workflow(self):
        self.model.database = (
            self.database.model_database
        )  # FIXME: Not right! Changes the user model object
        wf = Workflow()
        task_base_model = Task('create_base_model', _create_base_model, self.model)
        task_iiv = Task('create_iiv_on_ruv_model', _create_iiv_on_ruv_model)
        wf.add_tasks(task_base_model)
        wf.add_tasks(task_iiv, connect=True)
        # task_create_models = Task('create_models', create_models, self.model, final_task=False)
        # wf.add_tasks(task_create_models)
        wf_fit = self.workflow_creator()
        wf.add_tasks(wf_fit, connect=True)
        task_results = Task('results', post_process, final_task=True)
        wf.add_tasks(task_results, connect=True)
        return wf


def post_process(models):
    return models


def _create_base_model(input_model):
    base_model = pharmpy.model.Model()
    theta = Parameter('theta', 0.1)
    omega = Parameter('omega', 0.01, lower=0)
    sigma = Parameter('sigma', 1, lower=0)
    params = Parameters([theta, omega, sigma])
    base_model.parameters = params

    eta = RandomVariable.normal('eta', 'iiv', 0, omega.symbol)
    sigma = RandomVariable.normal('epsilon', 'ruv', 0, sigma.symbol)
    rvs = RandomVariables([eta, sigma])
    base_model.random_variables = rvs

    y = Assignment('Y', theta.symbol + eta.symbol + sigma.symbol)
    stats = ModelStatements([y])
    base_model.statements = stats

    base_model.dependent_variable = y.symbol
    base_model.name = "base"
    base_model.dataset = _create_dataset(input_model)
    base_model.database = input_model.database
    return base_model


def _create_iiv_on_ruv_model(base_model):
    model = base_model.copy()
    model.database = base_model.database  # FIXME: Should be unnecessary
    set_iiv_on_ruv(model)
    model.name = 'iiv_on_ruv'
    return model


def _create_power_model(base_model):
    model = base_model.copy()
    model.individual_prediction_symbol = sympy.Symbol('IPRED')
    set_power_on_ruv(model)
    return model


def _create_dataset(input_model):
    residuals = input_model.modelfit_results.residuals
    df = residuals['CWRES'].reset_index().rename(columns={'CWRES': 'DV'})
    return df


# def create_models(input_model):
#    dataset = _create_dataset(input_model)
#    base_model = _create_base_model(dataset)
#    iiv_on_ruv = _create_iiv_on_ruv_model(base_model)
#    power = _create_power_model(base_model)
#    return (base_model, iiv_on_ruv, power)
