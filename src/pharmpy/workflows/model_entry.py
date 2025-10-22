from __future__ import annotations

from typing import Optional

from pharmpy.internals.immutable import Immutable
from pharmpy.model import Model

from .log import Log
from .results import ModelfitResults, SimulationResults


class ModelEntry(Immutable):
    """Model with different modelfit related attributes.

    Object that can connect a model object to results from a modelfit. Attributes that can be connected to the
    model object are the modelfit results, the parent model, and the log.

    A ModelEntry object can be created with only the model object, and results can be attached as a separate step.

    Parameters
    ----------
    model : Model
        Pharmpy model
    parent : Model
        Pharmpy model that is parent to `model` argument (optional)
    modelfit_results : ModelfitResults
        Modelfit results connected to `model` argument (optional)
    log : Log
        Log connected to `model` argument (optional)
    """

    def __init__(
        self,
        model: Model,
        parent: Optional[Model] = None,
        modelfit_results: Optional[ModelfitResults] = None,
        simulation_results: Optional[SimulationResults] = None,
        log: Optional[Log] = None,
    ):
        self._model = model
        self._parent = parent
        self._modelfit_results = modelfit_results
        self._simulation_results = simulation_results
        self._log = log

    @classmethod
    def create(
        cls,
        model: Model,
        parent: Optional[Model] = None,
        modelfit_results: Optional[ModelfitResults] = None,
        simulation_results: Optional[SimulationResults] = None,
        log: Optional[Log] = None,
    ) -> ModelEntry:
        if parent:
            ModelEntry._canonicalize_parent(model, parent)
        return cls(
            model=model,
            parent=parent,
            modelfit_results=modelfit_results,
            simulation_results=simulation_results,
            log=log,
        )

    @staticmethod
    def _canonicalize_parent(model: Model, parent: Model) -> None:
        if model.name == parent.name:
            raise ValueError(
                f'Name of `model` and `parent` are the same: `{model.name}`, `{parent.name}`'
            )

    def attach_results(
        self,
        modelfit_results: ModelfitResults,
        simulation_results: Optional[SimulationResults] = None,
        log: Optional[Log] = None,
    ) -> ModelEntry:
        """Attaches modelfit results and possible log to ModelEntry objects"""
        if self._modelfit_results:
            raise ValueError('ModelEntry `modelfit_results` attribute already set')
        if self._simulation_results:
            raise ValueError('ModelEntry `simulation_results` attribute already set')
        if self._log:
            raise ValueError('ModelEntry `log` attribute already set')

        return ModelEntry.create(
            self._model,
            parent=self._parent,
            modelfit_results=modelfit_results,
            simulation_results=simulation_results,
            log=log,
        )

    @property
    def model(self) -> Model:
        """Model object"""
        return self._model

    @property
    def parent(self) -> Optional[Model]:
        """Parent model of main model"""
        return self._parent

    @property
    def modelfit_results(self) -> Optional[ModelfitResults]:
        """Modelfit results of main model"""
        return self._modelfit_results

    @property
    def simulation_results(self) -> Optional[SimulationResults]:
        """Simulation results of main model"""
        return self._simulation_results

    @property
    def log(self):
        """Log of main model"""
        return self._log

    def __repr__(self) -> str:
        return f'<Pharmpy model entry object {self.model.name}>'
