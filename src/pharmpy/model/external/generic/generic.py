import json
import re

from pharmpy.model import Model


def detect_model(src):
    is_generic = re.search(r'"__magic__"\s*:\s*"Pharmpy Model"', src, re.MULTILINE)
    return is_generic


def convert_model(model):
    new = Model(
        dataset=model.dataset,
        datainfo=model.datainfo,
        name=model.name,
        parameters=model.parameters,
        statements=model.statements,
        random_variables=model.random_variables,
        estimation_steps=model.estimation_steps,
        dependent_variables=model.dependent_variables,
        observation_transformation=model.observation_transformation,
        description=model.description,
        parent_model=model.name,
        filename_extension=model.filename_extension,
        initial_individual_estimates=model.initial_individual_estimates,
    )
    return new


def parse_model(code, path=None):
    d = json.loads(code)
    model = Model.from_dict(d)
    return model
