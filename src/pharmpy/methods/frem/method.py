from pathlib import Path

from pharmpy import Model

from .models import create_model3b


def update_model3b_for_psn(rundir, ncovs):
    """Function to update model3b from psn

       NOTE: This function lets pharmpy tie in to the PsN workflow
             and is a temporary solution
    """
    model_path = Path(rundir) / 'm1'
    model3 = Model(model_path / 'model_3.mod')
    model3b_old = Model(model_path / 'model_3b.mod')
    model3b = create_model3b(model3, int(ncovs))
    # Since we need phi from model3 run, but model 3b
    model3b_old.parameters = model3b.parameters
    model3b_old.name = model3b.name
    model3b_old.write(model_path, force=True)
