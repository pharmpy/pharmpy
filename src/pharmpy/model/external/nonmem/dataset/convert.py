import re
from functools import lru_cache
from typing import Union

from pharmpy.deps import numpy as np
from pharmpy.model.data import DatasetError


def convert_in_place(df, columns: list[str], null_value: str, missing_data_token: str):
    df[columns] = convert(df[columns], null_value, missing_data_token)


def convert(df, null_value: str, missing_data_token: str):
    return df.apply(_convert_data_item(convert_fortran_number(null_value), missing_data_token))


@lru_cache(32)
def _convert_data_item(
    null_value: Union[float, np.float64],
    missing_data_token: str,
):
    @lru_cache(4096)
    def _convert(x: Union[str, None]):
        if x in (None, ".", ""):
            return null_value

        if len(x) > 24:
            raise DatasetError("The dataset contains an item that is longer than 24 characters")
        if x == missing_data_token:
            return np.nan
        try:
            return convert_fortran_number(x)
        except ValueError as e:
            raise DatasetError(str(e)) from e

    return np.vectorize(_convert)


_fortran_number = re.compile(r"([+\-]?)([^+\-dD]*)([+-])([^+\-dD]*)")
_dD = re.compile(r"[dD]")


@lru_cache(maxsize=4096)
def convert_fortran_number(number_string: str):
    """This function will try to convert the number_string from the general fortran exponential format
    into an np.float64. It covers "1d1", "1D1", "a+b", "a-b", "+" and "-". All other cases will
    return None to signal that the number_string is not of the special form.

    Move somewhere else. Will be used in output parsing as well
    """
    try:
        return np.float64(number_string)
    except (TypeError, ValueError):
        pass

    if number_string in ("+", "-"):
        return float(number_string + "0.0")  # Converts "-" into -0.0

    # Handles formats like "1+1" = 1.0e1
    m = _fortran_number.match(number_string)
    if m:
        mantissa_sign = "-" if m.group(1) == "-" else ""
        mantissa = m.group(2)
        exponent_sign = m.group(3)
        exponent = m.group(4)
        return np.float64(f"{mantissa_sign}{mantissa}E{exponent_sign}{exponent}")

    # Handles normal cases of using D or d instead of E or e
    clean_number = _dD.sub("e", number_string)
    if clean_number != number_string:
        try:
            return np.float64(clean_number)
        except (TypeError, ValueError):
            pass

    raise ValueError(f"Could not convert the fortran number {number_string} to float")
