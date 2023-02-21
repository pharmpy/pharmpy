"""
This module contain functions for checking the format of an nlmixr model conversion 
in order to inform the users of any errors or mistakes that can could be made.

It serves purpose in catching known errors that are not yet solved, or limitations 
that are found in the conversion software
"""

import pharmpy.model
from pharmpy.modeling import (
    has_additive_error_model,
    has_proportional_error_model,
    has_combined_error_model,
    )
from .modify_code import change_same_time

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
    temp_model = model.copy()
    temp_model.dataset = temp_model.dataset.reset_index()
    dataset = temp_model.dataset
    for index, row in dataset.iterrows():
        if index != 0:
            if (row["ID"] == dataset.loc[index-1]["ID"] and
                row["TIME"] == dataset.loc[index-1]["TIME"] and
                row["EVID"] not in [0,3] and 
                dataset.loc[index-1]["EVID"] == 0):
                return True
    return False

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
