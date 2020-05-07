from pathlib import Path

from pharmpy import Model

from .models import create_model3b


def update_model3b_for_psn(rundir, ncovs):
    """Function to update model3b from psn

       NOTE: This function lets pharmpy tie in to the PsN workflow
             and is a temporary solution
    """
    model_path = Path(rundir) / 'm1'
    model1b = Model(model_path / 'model_1b.mod')
    model3 = Model(model_path / 'model_3.mod')
    model3b = create_model3b(model1b, model3, int(ncovs))
    model3b.write(model_path, force=True)
