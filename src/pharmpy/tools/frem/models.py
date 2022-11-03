import itertools

from pharmpy.deps import numpy as np
from pharmpy.internals.math import corr2cov, nearest_postive_semidefinite
from pharmpy.modeling import fix_or_unfix_parameters, set_initial_estimates


def calculate_parcov_inits(model, ncovs):
    """Get dict of new updated inits for parcov block

    model already has a FREM block
    Initial estimates for the parcov block is calculated given correlations of individual etas
    """
    dist = model.random_variables.iiv[-1]
    rvs = list(dist.names)
    ie = model.modelfit_results.individual_estimates
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


def create_model3b(model1b, model3, ncovs):
    """Create model 3b from model 3

    * Update parcov omega block
    * Set FIX pattern back from model1b
    * Use initial etas from model3
    """
    model3b = model3.copy()
    model3b.name = 'model_3b'
    parcov_inits = calculate_parcov_inits(model3, ncovs)
    set_initial_estimates(model3b, parcov_inits)
    set_initial_estimates(
        model3b, model3b.random_variables.nearest_valid_parameters(model3b.parameters.inits)
    )
    fix_or_unfix_parameters(model3b, model1b.parameters.fix)

    model3b.initial_individual_estimates = model3.modelfit_results.individual_estimates
    return model3b
