"""
This module contain functions for checking the format of an nlmixr model conversion 
in order to inform the users of any errors or mistakes that can could be made.

It serves purpose in catching known errors that are not yet solved, or limitations 
that are found in the conversion software
"""

import pharmpy.model
import warnings
from pharmpy.deps import sympy
from pharmpy.modeling import (
    has_additive_error_model,
    has_proportional_error_model,
    has_combined_error_model,
    remove_iiv
    )

def check_model(model: pharmpy.model) -> pharmpy.model:
    """
    Perform all neccessary checks to see if there are any issues with the input 
    model. Such as if the error model is unknown, or if there are other limitations 
    in the handling of the model.

    Parameters
    ----------
    model : pharmpy.model
        pharmpy model object

    Returns
    -------
    pharmpy.model
        Issues will be printed to the terminal and model is returned.

    """
    if not mixed_dose_types(model):
        print_warning("The connected model data contains mixed dosage types. Nlmixr cannot handle this currently \nConverted model will not run on associated data")
    if not known_error_model(model):
        print_warning("Format of error model cannot be determined. Will try to translate either way")
    if same_time(model):
        print_warning("Observation and bolus dose at the same time in the data. Modified for nlmixr model")
        model = change_same_time(model)
    if same_sigma(model):
        print_warning("Sigma with value same not supported. Updated as follows.")
        model = change_same_sigma(model)
        
    return model

def mixed_dose_types(model: pharmpy.model.Model) -> bool:
    """
    Check if there are both infusions and bolus doses in the dataset. If this 
    is the case, nlmixr might have issues running associated model on specified
    data.

    Parameters
    ----------
    model : pharmpy.model.Model
        pharmpy model object

    Returns
    -------
    bool
        True if contains mixed doses. False otherwise

    """
    dataset = model.dataset
    if "RATE" in dataset.columns:
        no_bolus = len(dataset[(dataset["RATE"] == 0) & (dataset["EVID"] != 0)])
        if no_bolus != 0:
            return False
        else:
            return True
    else:
        return True

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
    return (has_additive_error_model(model) or 
            has_combined_error_model(model) or 
            has_proportional_error_model(model))

def same_time(model: pharmpy.model) -> bool:
    temp_model = model
    temp_model = temp_model.replace(dataset = temp_model.dataset.reset_index())
    dataset = temp_model.dataset
    
    if "RATE" in dataset.columns:
        rate = True 
    else:
        rate = False
        
    for index, row in dataset.iterrows():
        if index != 0:
            if row["ID"] == dataset.loc[index-1]["ID"]:
                if row["TIME"] == dataset.loc[index-1]["TIME"]:
                    ID = row["ID"]
                    TIME = row["TIME"]
                    subset = dataset[(dataset["ID"] == ID) & (dataset["TIME"] == TIME)]
                    if any([x not in [0,3] for x in subset["EVID"].unique()]) and any([x in [0,3] for x in subset["EVID"].unique()]):
                        if rate:
                            if any([x != 0 for x in subset["RATE"].unique()]) and any([x == 0 for x in subset["RATE"].unique()]):
                                return True
                        else:
                            return True

    return False

def change_same_time(model: pharmpy.model) -> pharmpy.model:
    """
    Force dosing to happen after observation, if bolus dose is given at the
    exact same time.

    Parameters
    ----------
    model : pharmpy.model
        A pharmpy.model object

    Returns
    -------
    model : TYPE
        The same model with a changed dataset.

    """
    dataset = model.dataset.copy()
    dataset = dataset.reset_index()
    time = dataset["TIME"]
    
    if "RATE" in dataset.columns:
        rate = True 
    else:
        rate = False
    with warnings.catch_warnings():
        # Supress a numpy deprecation warning
        warnings.simplefilter("ignore")
        for index, row in dataset.iterrows():
            if index != 0:
                if row["ID"] == dataset.loc[index-1]["ID"]:
                    if row["TIME"] == dataset.loc[index-1]["TIME"]:
                        temp = index-1
                        while dataset.loc[temp]["TIME"] == row["TIME"]:
                            if dataset.loc[temp]["EVID"] not in [0,3]:
                                if rate:
                                    if dataset.loc[temp]["RATE"] == 0:
                                        time[temp] = time[temp] + 10**-6
                                else:
                                    time[temp] = time[temp] + 10**-6
                            temp += 1
    model.dataset["TIME"] = time
    return model

def same_sigma(model):
    sigmas = []
    for eps in model.random_variables.epsilons:
        sigma = eps.variance
        if sigma in sigmas:
            return True
        else:
            sigmas.append(sigma)
    return False

def change_same_sigma(model):
    
    sigmas = []
    sigmas_to_add = {}
    eps_and_sigma = {}
    for eps in model.random_variables.epsilons:
        sigma = eps.variance
        if sigma in sigmas:
            n = 1
            new_sigma = sympy.Symbol(sigma.name + "_" + f'{n}')
            while new_sigma in sigmas:
                n += 1
                new_sigma = sympy.Symbol(sigma.name + "_" + f'{n}')
            
            sigmas_to_add[new_sigma] = sigma
            
            sigmas.append(new_sigma)
            
            eps_and_sigma[eps.names] = new_sigma
            print(eps, " : ", new_sigma)
        else:
            sigmas.append(sigma)
    
    for rv in model.random_variables.epsilons:
        if rv.names in eps_and_sigma:
            new_eps = rv.replace(variance = eps_and_sigma[rv.names])
            
            rvs = model.random_variables
            keep = [name for name in model.random_variables.names if name not in [rv.names[0]]]
            
            model = model.replace(random_variables = rvs[keep])
            model = model.replace(random_variables = model.random_variables + new_eps)
            
            
    params = model.parameters
    for s in sigmas_to_add:
        param = model.parameters[sigmas_to_add[s]].replace(name = s.name)
        params = params + param
    model = model.replace(parameters = params)

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
