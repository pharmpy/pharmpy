# Functional interface to extract dataset information


def get_number_of_individuals(model):
    """Retrieve the number of individuals in the model dataset"""
    return model.dataset.pharmpy.ninds


def get_number_of_observations(model):
    """Retrieve the total number of observations in the model dataset"""
    return model.dataset.pharmpy.nobs


def get_number_of_observations_per_individual(model):
    """Number of observations for each individual"""
    return model.dataset.pharmpy.nobsi
