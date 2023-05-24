import itertools

from pharmpy.deps import numpy as np
from pharmpy.internals.math import corr2cov, nearest_postive_semidefinite
from pharmpy.modeling import fix_or_unfix_parameters, set_initial_estimates


def calculate_parcov_inits(model, ie, ncovs):
    """Get dict of new updated inits for parcov block

    model already has a FREM block
    Initial estimates for the parcov block is calculated given correlations of individual etas
    """
    dist = model.random_variables.iiv[-1]
    rvs = list(dist.names)
    eta_corr = ie[rvs].corr()
    eta_corr.fillna(value=1.0, inplace=True)  # Identical etas will get NaN as both diag and corr

    sigma = dist.variance
    inits = sigma.subs(model.parameters.inits)
    inits = np.array(inits).astype(np.float64)
    sd = np.sqrt(inits.diagonal())
    npars = len(sd) - ncovs

    cov = corr2cov(eta_corr.to_numpy(), sd)
    cov[cov == 0] = 0.0001
    cov = nearest_postive_semidefinite(cov)

    parcov_inits = cov[npars:, :npars]
    parcov_symb = sigma[npars:, :npars]

    param_inits = {
        parcov_symb[i, j].name: parcov_inits[i, j]
        for i, j in itertools.product(range(ncovs), range(npars))
    }
    return param_inits


def create_model3b(model1b, model3, model3_res, ncovs):
    """Create model 3b from model 3

    * Update parcov omega block
    * Set FIX pattern back from model1b
    * Use initial etas from model3
    """
    model3b = model3.replace(name='model_3b')
    ie = model3_res.individual_estimates
    parcov_inits = calculate_parcov_inits(model3, ie, ncovs)
    model3b = set_initial_estimates(model3b, parcov_inits)
    model3b = set_initial_estimates(
        model3b, model3b.random_variables.nearest_valid_parameters(model3b.parameters.inits)
    )
    model3b = fix_or_unfix_parameters(model3b, model1b.parameters.fix)

    model3b = model3b.replace(initial_individual_estimates=ie)
    return model3b
