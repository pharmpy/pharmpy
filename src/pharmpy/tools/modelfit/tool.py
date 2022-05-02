from pharmpy.model import Model
from pharmpy.workflows import Task, Workflow


def create_workflow(models=None, n=None, tool=None):
    """
    If models is None and n is None: fit one unknown model
    If models is model object (n must be None): fit that one model
    If models is list of models (n must be None): fit these models
    If models is None and n is an int: fit n unknown models
    """
    wf = create_fit_workflow(models, n, tool)
    wf.name = "modelfit"
    task_result = Task('results', post_process_results)
    wf.add_task(task_result, predecessors=wf.output_tasks)
    return wf


def create_fit_workflow(models=None, n=None, tool=None):
    execute_model = retrieve_from_database_or_execute_model_with_tool(tool)

    wf = Workflow()
    if models is None:
        if n is None:
            task = Task('run', execute_model)
            wf.add_task(task)
        else:
            for i in range(n):
                task = Task(f'run{i}', execute_model)
                wf.add_task(task)
    elif isinstance(models, Model):
        task = Task('run', execute_model, models)
        wf.add_task(task)
    else:
        for i, model in enumerate(models):
            task = Task(f'run{i}', execute_model, model)
            wf.add_task(task)
    return wf


def post_process_results(*models):
    if len(models) > 1:
        return models
    else:
        return models[0]


def retrieve_from_database_or_execute_model_with_tool(tool):
    def task(model):
        try:
            db_results = model.database.retrieve_modelfit_results(model.name)
        except (KeyError, AttributeError, FileNotFoundError):
            db_results = None

        if db_results is not None:
            # NOTE We have the results
            try:
                db_model = model.database.retrieve_model(model.name)
            except (KeyError, AttributeError, FileNotFoundError):
                db_model = None

            # NOTE Here we could invalidate cached results if certain errors
            # happened such as a missing or outdated license. We do not do that
            # at the moment.

            # if db_model == model and model.has_same_dataset_as(db_model):
            if db_model and model.has_same_dataset_as(db_model):
                # NOTE Inputs are identical so we can reuse the results
                model.modelfit_results = db_results
                return model

        # NOTE Fallback to executing the model
        execute_model = get_execute_model(tool)
        return execute_model(model)

    return task


def get_execute_model(tool):
    from pharmpy.tools.modelfit import conf

    if tool is None:
        tool = conf.default_tool

    if tool == 'nonmem':
        from pharmpy.plugins.nonmem.run import execute_model
    elif tool == 'nlmixr':
        from pharmpy.plugins.nlmixr.run import execute_model
    else:
        raise ValueError(f"Unknown estimation tool {tool}")

    return execute_model
