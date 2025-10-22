import importlib


def parse_modelfit_results(model, path, esttool=None, strict=False):
    module_name = model.__class__.__module__
    a = module_name.split(".")
    name = a[3]
    if esttool is not None:
        name = esttool
    tool_module_name = f'pharmpy.tools.external.{name}'
    module = importlib.import_module(tool_module_name)

    res = module.parse_modelfit_results(model, path, strict=strict)
    return res
