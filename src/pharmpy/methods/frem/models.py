import itertools

import numpy as np

import pharmpy.math
from pharmpy.random_variables import VariabilityLevel


def calculate_parcov_inits(model, ncovs):
    """Get dict of new updated inits for parcov block

    model already has a FREM block
    Initial estimates for the parcov block is calculated given correlations of individual etas
    """
    rvs, dist = model.random_variables.distributions(level=VariabilityLevel.IIV)[-1]
    rvs = [rv.name for rv in rvs]
    ie = model.modelfit_results.individual_estimates
    eta_corr = ie[rvs].corr()
    eta_corr.fillna(value=1.0, inplace=True)  # Identical etas will get NaN as both diag and corr

    sigma = dist.sigma
    inits = sigma.subs(model.parameters.inits)
    inits = np.array(inits).astype(np.float64)
    sd = np.sqrt(inits.diagonal())
    npars = len(sd) - ncovs

    cov = pharmpy.math.corr2cov(eta_corr.to_numpy(), sd)
    cov[cov == 0] = 0.0001
    cov = pharmpy.math.nearest_posdef(cov)

    parcov_inits = cov[npars:, :npars]
    parcov_symb = sigma[npars:, :npars]

    param_inits = {
        parcov_symb[i, j].name: parcov_inits[i, j]
        for i, j in itertools.product(range(ncovs), range(npars))
    }
    return param_inits


def create_model3b(model1b, model3, ncovs):
    """Create model 3b from model 3

    * Update parcov omega block
    * Set FIX pattern back from model1b
    * Use initial etas from model3
    """
    parameters = model3.parameters

    parcov_inits = calculate_parcov_inits(model3, ncovs)
    parameters.inits = parcov_inits
    parameters.inits = model3.random_variables.nearest_valid_parameters(parameters.inits)
    parameters.fix = model1b.parameters.fix

    model3b = model3.copy()
    model3b.parameters = parameters
    model3b.name = 'model_3b'
    model3b.update_individual_estimates(model3)
    return model3b
