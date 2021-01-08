====
FREM
====

Pharmpy currently handles the postprocessing, plotting and creation of model_3b of the PsN FREM tool.

.. math::

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The FREM postprocessing and results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The postprocessing starts after estimating the parameters :math:`P` of the FREM model together with their uncertainty covariance matrix :math:`Cov(P)`. Let us denote the random variables representing the model parameters :math:`\eta_i` for :math:`1 \leq i \leq n_{par}` and the random variables for the covariates
:math:`\eta_k` for :math:`n_{par} + 1 \leq k \leq n_{cov} + n_{par}`. Then

.. math::

        \begin{bmatrix}
            \eta_1 \\
            \vdots \\
	        \eta_{n_{par}} \\
	        \eta_{n_{par} + 1} \\
	        \vdots \\
	        \eta_{n_{par} + n_{cov}} \\
         \end{bmatrix}
	\sim \mathcal{N}(\mu, \Omega)

where 

.. math::


	\mu = 
    \begin{bmatrix}
        0 \\
        \vdots \\
        0 \\	
        \overline{C}_{1} \\
        \vdots \\ 
	    \overline{C}_{n_{cov}} \\
    \end{bmatrix}
    =
    \begin{bmatrix}
        \mu_1 \\
        \mu_2 \\
    \end{bmatrix}

and

.. math::

    \Omega =
    \begin{bmatrix}
        \omega_{11} & \omega_{21} & \cdots & \omega_{n1} \\
        \omega_{21} & \omega_{22} & \cdots & \omega_{n2} \\
        \vdots & \vdots & \ddots & \vdots \\
        \omega_{n1} & \omega_{n2} & \cdots & \omega_{nn} \\
    \end{bmatrix} =
    \begin{bmatrix}
        \Omega_{11} & \Omega_{12} \\
        \Omega_{21} & \Omega_{22} \\
   \end{bmatrix}

:math:`\overline{C}_k` is the covariate reference. For continuous covariates the reference is the mean of the baselines and for categoricals it is the non-mode value of the baselines.
The latter partition being for parameters (index 1) and covariates (index 2), i.e.
:math:`\Omega_{11}` is the original parameter matrix, :math:`\Omega_{22}` is the covariate matrix and :math:`\Omega_{21}` and :math:`\Omega_{12}^T` is the parameter-covariate covariance block. 

Covariate effects
~~~~~~~~~~~~~~~~~

The effects of each covariate on each parameter is calculated with uncertainty and summarized in the `covariate_effects` table.

.. jupyter-execute::
    :hide-code:

    from pharmpy.results import read_results
    res = read_results('tests/testdata/frem/results.json')
    res.covariate_effects

The effects are given as fractions where 1 means no effect and are calculated conditioned on the 5th and 95th percentile of the covariate values respectively.

Assume that the estimated parameter vector is joint normally distributed with mean vector :math:`P` and covariance matrix :math:`Cov(P)`. Then the marginal distribution of the :math:`\omega_{ij}` of :math:`\Omega` will also be joint normally distributed. Sample 1000 times from this marginal distribution to get :math:`\omega_{ijk}` and :math:`\Omega_k` for :math:`1\leq k \leq 1000`.

If the covariate etas were scaled in the FREM model the scaling needs to be applied to all :math:`\Omega_k` by first creating the scaling matrix

.. math::

	S=
    \begin{bmatrix}
        1 & & & & \\
        & 1 & & & \\
        & & \ddots & &\\
        & & & \sigma_1 &\\
        & & & & \sigma_2 \\
    \end{bmatrix}

where :math:`\sigma_i` is the standard deviation of the i:th covariate in the data, and then get each scaled matrix as :math:`S \Omega_k S`.

Do for each sample:
For each covariate :math:`k` create the marginal distribution of all parameters and that single covariate. Calculate the means of the parameters given the covariate values in the 5th and 95th percentile of the dataset in turn. The vector of the means would be given by the conditional joint normal distribution as:

.. math::

	\bar{\mu} = \mu_1 + \Omega_{12}\Omega_{22}^{-1}(a - \mu_2)

where :math:`a` is the given value of the covariate.

For each parameter and covariate calculate the mean, 5:th and 95:th percentile over all conditional parameter means. These are the covariate effects and their uncertainties. I.e. the conditional mean of the parameter given in turn the 5th and the 95th percentile of the covariate data. Since we currently assume log-normally distributed individual parameters each mean is exponentiated.

The covariate effect plots give the covariate effects in percent with uncertainty for each parameter and covariate in turn. The red figures are the 5th and 95th percentile covariate values.

.. jupyter-execute::
    :hide-code:

    res.plot_covariate_effects()


Parameter covariate coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter covariate coefficients for each covariate separately and for all taken together is available in `coefficients`. The definition for one coefficient is 
`Cov(Par, Covariate) / Var(Covariate)` and generalized for all together the matrix :math:`\Sigma_{12}\Sigma_{22}^{-1}`

.. jupyter-execute::
    :hide-code:

    res.coefficients

Individual covariate effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The combined effects of all covariates on the parameters of each individual is calculated with uncertainty and summarized in the `individual_effects` table.

.. jupyter-execute::
    :hide-code:

    res.individual_effects

The conditional distribution as above is calculated for the estimated parameters (observed in the table) and the samples (that gives p5 and p95). The same :math:`\mu` and :math:`\Omega` are used, but the given condition is instead the covariate baseline as estimated from the model for each individual.

The plot shows the individuals with the lowest and the highest percentual covariate effect and the uncertainty.

.. jupyter-execute::
    :hide-code:

    res.plot_individual_effects()


Unexplained variability
~~~~~~~~~~~~~~~~~~~~~~~

The unexplained variability is calculated and summarized in the `unexplained_variability` table.

.. jupyter-execute::
    :hide-code:

    res.unexplained_variability

For each sample the conditional distribution is calculated given no covariates, each covariate in turn and all covariates at the same time. The variability will be given by the conditional covariance matrix that can be calculated as:

.. math::

	\bar{\Omega} = \Omega_{11} - \Omega_{12} \Omega_{22}^{-1} \Omega_{21} =  \Omega_{11} - \Omega_{21}^T \Omega_{22}^{-1} \Omega_{21}

The presented results are the 5th and 95th percetiles of the standard deviations of the parameters from :math:`\bar{\Omega}`. The observed standard deviation is the conditional 

The plot display the original unexplained variability with the uncertainty for all parameter and covariate combinations.

.. jupyter-execute::
    :hide-code:

    res.plot_unexplained_variability()

All variability parameters given the estimated parameters conditioned on each covariate in turn can be found in `parameter_variability`.

.. jupyter-execute::
    :hide-code:

    res.parameter_variability


Parameter estimates
~~~~~~~~~~~~~~~~~~~

The parameter initial estimates and final estimates of the base model, all intermediate models and the FREM model are tabled in `parameter_inits_and_estimates`.

.. jupyter-execute::
    :hide-code:

    res.parameter_inits_and_estimates

Relative difference between of the base model parameters estimates and the final model parameter estimates are calculated in `base_parameter_change`.

.. jupyter-execute::
    :hide-code:

    res.base_parameter_change


OFV
~~~

OFV of the base model, all intermediate models and the final FREM model are collected into `ofv`.

.. jupyter-execute::
    :hide-code:

    res.ofv

Estimated covariate values
~~~~~~~~~~~~~~~~~~~~~~~~~~

The FREM model also gives an estimate of the covariate values themselves. Ideally these values should be close to the ones in the dataset. Summary statistics for the estimated
covariate values are put into `estimated_covariates`.

.. jupyter-execute::
    :hide-code:

    res.estimated_covariates
