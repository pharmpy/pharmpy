def name_mangle(s: str) -> str:
    """
    Changes the format of parameter name to avoid using parenthesis

    Parameters
    ----------
    s : str
        Parameter name to be changed

    Returns
    -------
    str
        Parameter name with parenthesis removed

    Example
    -------
    name_mangle("ETA(1)")
    -> "ETA1"

    """
    return s.replace('(', '').replace(')', '').replace(',', '_')
