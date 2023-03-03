from pathlib import Path
from typing import Optional, Union

from pharmpy.internals.fs.path import path_absolute
from pharmpy.model import Model


def create_dataset_path(model: Model, path: Optional[Union[str, Path]] = None) -> Path:
    path = path_absolute(Path("" if path is None else path))

    if path and not path.is_dir():
        return path

    if (di_path := model.datainfo.path) is None:
        filename = f"{model.name}.csv"
    else:
        filename = di_path.with_suffix('.csv').name

    return path / filename


def write_csv(model: Model, path: Optional[Union[str, Path]] = None, force: bool = False) -> Model:
    """Write dataset to a csv file and updates the datainfo path

    Parameters
    ----------
    model : Model
        Model whose dataset to write to file
    path : None or str or Path
        Destination path. Default is to use original path with .csv suffix.
    force : bool
        Overwrite file with same path. Default is False.

    Returns
    -------
    Model
       Updated model object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, write_csv
    >>> model = load_example_model("pheno")
    >>> model = write_csv(model, path="newdataset.csv")    # doctest: +SKIP

    """
    from pharmpy.model import data

    path = create_dataset_path(model, path)
    if not force and path.exists():
        raise FileExistsError(f'File at {path} already exists.')

    path = path_absolute(path)
    model.dataset.to_csv(path, na_rep=data.conf.na_rep, index=False)
    model = model.replace(datainfo=model.datainfo.replace(path=path))
    return model
