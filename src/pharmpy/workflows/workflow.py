from __future__ import annotations

import copy
import uuid
from typing import Generic, Iterable, List, Optional, TypeVar, Union

from pharmpy.deps import networkx as nx

from .task import Task

T = TypeVar('T')


class Workflow(Generic[T]):
    """Workflow class

    Representation of a directed acyclic graph with Tasks as nodes and
    the flow of parameters as edges.

    Parameters
    ----------
    tasks : list
        List of tasks for initialization
    """

    def __init__(self, tasks: Optional[Iterable[Task]] = None, name: Optional[str] = None):
        self._g = nx.DiGraph()
        self.name = name
        if tasks:
            for task in tasks:
                self.add_task(task)

    def add_task(self, task: Task, predecessors: Optional[Union[Task, List[Task]]] = None):
        """Add a task to the workflow

        Predecessors will be connected if given.

        Parameters
        ----------
        task : Task
            Task to add
        predecessors : list or Task
            One or multiple predecessor tasks to connect to the added task
        """
        self._g.add_node(task)
        if predecessors is not None:
            if not isinstance(predecessors, list):
                self._g.add_edge(predecessors, task)
            else:
                for pred in predecessors:
                    self._g.add_edge(pred, task)

    def insert_workflow(
        self, other: Workflow, predecessors: Optional[Union[Task, List[Task]]] = None
    ):
        """Insert other workflow

        Parameters
        ----------
        other : Workflow
            Workflow to insert
        predecessors : list or Task
            One or multiple predecessor tasks to connect to the inputs
            of the inserted workflow. If None all output tasks will be found
            and used as predecessors.
        """
        if predecessors is None:
            output_tasks = self.output_tasks
        else:
            if isinstance(predecessors, list):
                output_tasks = predecessors
            else:
                output_tasks = [predecessors]
        input_tasks = other.input_tasks
        self._g = nx.compose(self._g, other._g)
        if len(input_tasks) == len(output_tasks):
            for inp, outp in zip(input_tasks, output_tasks):
                self._g.add_edge(outp, inp)
        elif len(input_tasks) == 1:
            for outp in output_tasks:
                self._g.add_edge(outp, input_tasks[0])
        elif len(output_tasks) == 1:
            for inp in input_tasks:
                self._g.add_edge(output_tasks[0], inp)
        else:
            raise ValueError('Having N:M connections between workflows is currently not supported')

    def as_dask_dict(self):
        """Create a dask workflow dictionary

        Returns
        -------
        dict
            Dask workflow dictionary
        """
        ids = {}
        for task in self._g.nodes():
            ids[task] = f'{task.name}-{uuid.uuid4()}'

        if len(self.output_tasks) == 1:
            ids[self.output_tasks[0]] = 'results'
        else:
            raise ValueError("Workflow can only have one output task")

        as_dict = {}
        for task in nx.dfs_tree(self._g):
            key = ids[task]
            input_list = list(task.task_input)
            input_list.extend(ids[t] for t in self._g.predecessors(task))
            value = (task.function, *input_list)
            as_dict[key] = value
        return as_dict

    @property
    def input_tasks(self) -> List[Task]:
        """All input (source) tasks of entire workflow"""
        return [node for node in self._g.nodes if self._g.in_degree(node) == 0]

    @property
    def output_tasks(self) -> List[Task]:
        """All output tasks (sink) tasks of entire workflow"""
        return [node for node in self._g.nodes if self._g.out_degree(node) == 0]

    def get_upstream_tasks(self, task: Task) -> List[Task]:
        """Get all tasks upstream of a certain task

        Parameters
        ----------
        task : Task
            Downstream task

        Returns
        -------
        list
            Upstream tasks
        """
        edges = nx.edge_dfs(self._g, task, orientation='reverse')
        return [node for node, _, _ in edges]

    def get_predecessors(self, task: Task) -> List[Task]:
        """Get all predecessors of task

        Parameters
        ----------
        task : Task
            Downstream task

        Returns
        -------
        list
            Predecessors of task
        """
        return list(self._g.predecessors(task))

    def get_successors(self, task: Task) -> List[Task]:
        """Get all successors of task

        Parameters
        ----------
        task : Task
            Upstream task

        Returns
        -------
        list
            Successors of task
        """
        return list(self._g.successors(task))

    def copy(self) -> Workflow[T]:
        """Deepcopy of workflow

        Returns
        -------
        Workflow
            Deepcopy
        """
        wf_copy = copy.deepcopy(self)
        return wf_copy

    def plot_dask(self, filename: str):
        """Save a visualization of workflow to file

        Parameters
        ----------
        filename : str
            Path
        """
        from dask import visualize  # pyright: ignore [reportPrivateImportUsage]

        visualize(self.as_dask_dict(), filename=filename, collapse_outputs=True)

    @property
    def tasks(self) -> List[Task]:
        """All tasks in workflow"""
        return list(self._g.nodes())

    def __len__(self):
        return len(self._g.nodes)

    def __str__(self):
        return '\n'.join([str(task) for task in self._g])

    def __add__(self, other: Workflow):
        wf_new = Workflow()
        wf_new._g = nx.compose(self._g, other._g)
        return wf_new
