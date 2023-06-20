"""
This module contain functions for checking the format of an nlmixr model conversion
in order to inform the users of any errors or mistakes that can could be made.

It serves purpose in catching known errors that are not yet solved, or limitations
that are found in the conversion software
"""

import pharmpy.model
from pharmpy.deps import sympy
from pharmpy.modeling import (
    has_additive_error_model,
    has_combined_error_model,
    has_proportional_error_model,
)


def check_model(
    model: pharmpy.model.Model, skip_error_model_check: bool = False
) -> pharmpy.model.Model:
    """
    Perform all neccessary checks to see if there are any issues with the input
    model. Such as if the error model is unknown, or if there are other limitations
    in the handling of the model.
    Skipping checking the error model has no effect on the translation.

    Parameters
    ----------
    model : pharmpy.model.Model
        pharmpy model object
    skip_error_model_check : bool
        Can choose to skip checking which error model the model has since this
        can improve runtime.

    Returns
    -------
    pharmpy.model.Model
        Issues will be printed to the terminal and model is returned.

    """

    # Checks for the dataset
    if model.dataset is not None:
        if "TIME" in model.dataset.columns:
            # Forcefully change so that doses and observation dont happen
            # at the same time
            if same_time(model):
                print_warning(
                    "Observation and bolus dose at the same time in the data. Modified for nlmixr model"
                )
                model = change_same_time(model)
        else:
            # Add placeholder time to be able to run model
            print_warning(
                "TIME column required to run model. Added time with zero value for all events"
            )
            model = add_time(model)

    # Checks regarding error model
    if not skip_error_model_check:
        if not known_error_model(model):
            print_warning("Format of error model cannot be determined.")

    # Checks regarding random variables
    if rvs_same(model, sigma=True):
        print_warning("Sigma with value same not supported. Parameters are updated")
        model = change_rvs_same(model, sigma=True)
    if rvs_same(model, omega=True):
        print_warning("Omega with value same not supported. Parameters are updated")
        model = change_rvs_same(model, omega=True)

    # Checks regarding esimation method
    method = model.estimation_steps[0].method
    if not known_estimation_method(method):
        print_warning(
            f"Estimation method {method} unknown to nlmixr2. Using 'FOCEI' as placeholder"
        )

    return model


def add_time(model):
    dataset = model.dataset
    dataset["TIME"] = 0
    model = model.replace(dataset=dataset)
    return model


def known_estimation_method(method):
    nonmem_method_to_nlmixr = {"FOCE": "foce", "FO": "fo", "SAEM": "saem"}
    if method in nonmem_method_to_nlmixr.keys():
        return True
    else:
        return False


def known_error_model(model: pharmpy.model.Model) -> bool:
    """
    Check if the associated error model is known to pharmpy. Currently check if
    model hase
    - additive error model
    - proportional error model
    - combined error model

    Parameters
    ----------
    model : pharmpy.model.Model
        pharmpy model object

    Returns
    -------
    bool
        True if error model is defined. False if unknown.

    """
    return (
        has_additive_error_model(model)
        or has_combined_error_model(model)
        or has_proportional_error_model(model)
    )


def same_time(model: pharmpy.model.Model) -> bool:
    """
    Check if a dataset to connected model has bolus doses and observations at the
    exact same time. This causes issues in nlmixr2 as the dose always comes before
    the observation, increasing its value´

    Parameters
    ----------
    model : pharmpy.model.Model
        A pharmpy model objects.

    Returns
    -------
    bool
        True if bolus and observation at the exact same time, for any datapoints.

    """
    temp_model = model
    temp_model = temp_model.replace(dataset=temp_model.dataset.reset_index())
    dataset = temp_model.dataset

    if "RATE" in dataset.columns:
        rate = True
    else:
        rate = False

    evid_ignore = [0, 3, 4]

    for index, row in dataset.iterrows():
        if index != 0:
            if row["ID"] == dataset.loc[index - 1]["ID"]:
                if row["TIME"] == dataset.loc[index - 1]["TIME"]:
                    ID = row["ID"]
                    TIME = row["TIME"]
                    subset = dataset[(dataset["ID"] == ID) & (dataset["TIME"] == TIME)]
                    if any([x not in evid_ignore for x in subset["EVID"].unique()]) and any(
                        [x in evid_ignore for x in subset["EVID"].unique()]
                    ):
                        if rate:
                            if any([x != 0 for x in subset["RATE"].unique()]) and any(
                                [x == 0 for x in subset["RATE"].unique()]
                            ):
                                return True
                        else:
                            return True

    return False


def change_same_time(model: pharmpy.model.Model) -> pharmpy.model.Model:
    """
    Force dosing to happen after observation, if bolus dose is given at the
    exact same time. Done by adding 0.000001 to the time of the bolus dose

    Parameters
    ----------
    model : pharmpy.model.Model
        A pharmpy model object

    Returns
    -------
    model : pharmpy.model.Model
        The same model with a changed dataset.

    """
    dataset = model.dataset.copy()
    dataset = dataset.reset_index(drop=True)

    if "RATE" in dataset.columns:
        rate = True
    else:
        rate = False

    evid_ignore = [0, 3, 4]

    for index, row in dataset.iterrows():
        if index != 0:
            if row["ID"] == dataset.loc[index - 1]["ID"]:
                if row["TIME"] == dataset.loc[index - 1]["TIME"]:
                    ID = row["ID"]
                    TIME = row["TIME"]
                    subset = dataset[(dataset["ID"] == ID) & (dataset["TIME"] == TIME)]
                    if any([x not in evid_ignore for x in subset["EVID"].unique()]) and any(
                        [x in evid_ignore for x in subset["EVID"].unique()]
                    ):
                        if rate:
                            dataset.loc[
                                (dataset["ID"] == ID)
                                & (dataset["TIME"] == TIME)
                                & (dataset["RATE"] == 0)
                                & (~dataset["EVID"].isin(evid_ignore)),
                                "TIME",
                            ] += 0.000001
                        else:
                            dataset.loc[
                                (dataset["ID"] == ID)
                                & (dataset["TIME"] == TIME)
                                & (~dataset["EVID"].isin(evid_ignore)),
                                "TIME",
                            ] += 0.000001

    model = model.replace(dataset=dataset)
    return model


def rvs_same(model: pharmpy.model.Model, sigma: bool = False, omega: bool = False) -> bool:
    """
    Check if there are random variables that are referencing the same
    distribution value.
    Comes from NONMEM format

    Parameters
    ----------
    model : pharmpy.model.Model
        A pharmpy model object.
    sigma : bool, optional
        Check for same sigma values. The default is False.
    omega : bool, optional
        Check for same omega values. The default is False.

    Returns
    -------
    bool
        True if there are random variables referenceing the same distribution
        value. Otherwise False.

    """
    if sigma:
        rvs = model.random_variables.epsilons
    elif omega:
        rvs = model.random_variables.etas

    checked_variance = []
    for rv in rvs:
        var = rv.variance
        if var in checked_variance:
            return True
        else:
            checked_variance.append(var)
    return False


def change_rvs_same(
    model: pharmpy.model.Model, sigma: bool = False, omega: bool = False
) -> pharmpy.model.Model:
    """
    Add more distribution parameters if mutiple random variables are referencing
    the same distribution. Done for sigma and omega values.
    Prints conversion to console.

    Parameters
    ----------
    model : pharmpy.model.Model
        A pharmpy model object.
    sigma : bool, optional
        Check for same sigma values. The default is False.
    omega : bool, optional
        Check for same omega values. The default is False.

    Returns
    -------
    model : TYPE
        Return model with added distribution values.

    """
    if sigma:
        rvs = model.random_variables.epsilons
    elif omega:
        rvs = model.random_variables.etas

    checked_variance = []
    var_to_add = {}
    rvs_and_var = {}
    for rv in rvs:
        current_var = []

        if isinstance(rv.variance, sympy.Symbol):
            variance = [rv.variance]
        else:
            variance = rv.variance

        for var in variance:
            if var in checked_variance:
                n = 1
                new_var = sympy.Symbol(var.name + "_" + f'{n}')
                while new_var in checked_variance:
                    n += 1
                    new_var = sympy.Symbol(var.name + "_" + f'{n}')

                var_to_add[new_var] = var

                current_var.append(new_var)

                rvs_and_var[(rv.names, var)] = new_var
                # print(rv, " : ", new_var)
            else:
                current_var.append(var)

        checked_variance += current_var

    params = model.parameters
    for s in var_to_add:
        param = model.parameters[var_to_add[s]].replace(name=s.name)
        params = params + param
    model = model.replace(parameters=params)

    etas = [e[0] for e in rvs_and_var.keys()]
    for rv in rvs:
        if rv.names in etas:
            old_to_new = [name_var for name_var in rvs_and_var if name_var[0] == rv.names]
            new_rv = rv
            for el in old_to_new:
                old = el[1]
                new = rvs_and_var[el]
                new_variance = new_rv.variance.replace(old, new)
                new_rv = new_rv.replace(variance=new_variance)
            all_rvs = model.random_variables

            keep = [name for name in all_rvs.names if name not in rv.names]
            model = model.replace(random_variables=all_rvs[keep])
            model = model.replace(random_variables=model.random_variables + new_rv)

    # Add newline after all updated sigma values have been printed
    print()
    return model


def print_warning(warning: str) -> None:
    """
    Help function for printing warning messages to the console

    Parameters
    ----------
    warning : str
        warning description to be printed

    Returns
    -------
    None
        Prints warning to console

    """
    print(f'-------\nWARNING : \n{warning}\n-------')
