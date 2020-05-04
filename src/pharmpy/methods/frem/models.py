import itertools

import numpy as np

import pharmpy.math
from pharmpy.random_variables import VariabilityLevel


def calculate_parcov_inits(model, ncovs):
    """Get dict of new updated inits for parcov block

       model already has a FREM block
       Initial estimates for the parcov block is calculated given correlations of individual etas
    """
    eta_corr = model.modelfit_results.individual_estimates.corr()
    rvs, dist = list(model.random_variables.distributions(level=VariabilityLevel.IIV))[-1]

    sigma = dist.sigma
    inits = sigma.subs(model.parameters.inits)
    inits = np.array(inits).astype(np.float64)
    sd = np.sqrt(inits.diagonal())
    npars = len(sd) - ncovs

    cov = pharmpy.math.corr2cov(eta_corr.to_numpy(), sd)
    cov[npars:, :npars][cov[npars:, :npars] == 0] = 0.0001
    cov[:npars, npars:][cov[:npars, npars:] == 0] = 0.0001
    cov = pharmpy.math.nearest_posdef(cov)

    parcov_inits = cov[npars:, :npars]
    parcov_symb = sigma[npars:, :npars]

    param_inits = {parcov_symb[i, j].name: parcov_inits[i, j] for i, j in
                   itertools.product(range(ncovs), range(npars))}
    return param_inits


def create_model3b(model3, ncovs):
    """Create model 3b from model 3

       currently only updates the parcov omega block
    """
    model3b = model3.copy()
    parcov_inits = calculate_parcov_inits(model3, ncovs)
    parameters = model3.parameters
    parameters.inits = parcov_inits
    model3b.parameters = parameters
    model3b.name = 'model_3b'
    return model3b
