from pathlib import Path


def write_csv(model, path=None, force=False):
    """Write dataset to a csv file

    Parameters
    ----------
    model : Model
        Model whose dataset to write to file
    path : Path
        Destination path. Default is to use original path with .csv suffix.
    force : bool
        Overwrite file with same path. Default is False.

    Returns
    -------
    Path
       path to the written file.

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, write_csv
    >>> model = load_example_model("pheno")
    >>> write_csv(model, path="newdataset.csv")    # doctest: +SKIP
    newdataset.csv

    """
    import pharmpy.data

    if path is None:
        path = Path("")
    else:
        path = Path(path)
    if not path or path.is_dir():
        try:
            filename = f"{model.datainfo.path.name.with_suffix('.csv')}"
        except AttributeError:
            filename = f"{model.name + '.csv'}"
        path /= filename
    if not force and path.exists():
        raise FileExistsError(f'File at {path} already exists.')

    model.dataset.to_csv(path, na_rep=pharmpy.data.conf.na_rep, index=False)
    model.datainfo.path = path
    return path
