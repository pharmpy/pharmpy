from __future__ import annotations

import inspect
import uuid
from typing import Generic, Optional, TypeVar, Union

from pharmpy.deps import networkx as nx
from pharmpy.internals.immutable import Immutable

from .task import Task

T = TypeVar('T')


class WorkflowBase:
    """Common base class for Workflow and WorkflowBuilder"""

    _g: nx.DiGraph

    @property
    def tasks(self) -> list[Task]:
        """All tasks in workflow"""
        return list(self._g.nodes())

    @property
    def input_tasks(self) -> list[Task]:
        """All input (source) tasks of entire workflow"""
        return [node for node in self._g.nodes if self._g.in_degree(node) == 0]

    @property
    def output_tasks(self) -> list[Task]:
        """All output tasks (sink) tasks of entire workflow"""
        return [node for node in self._g.nodes if self._g.out_degree(node) == 0]

    def get_upstream_tasks(self, task: Task) -> list[Task]:
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

    def get_predecessors(self, task: Task) -> list[Task]:
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

    def get_successors(self, task: Task) -> list[Task]:
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

    def traverse(self, algorithm, source=None):
        supported_algorithms = ['dfs', 'bfs']
        if algorithm == 'dfs':
            return nx.dfs_tree(self._g, source)
        elif algorithm == 'bfs':
            if not source:
                raise ValueError('Source node needed for bfs traversal')
            return nx.bfs_tree(self._g, source)
        else:
            raise ValueError(f'Unknown algorithm `{algorithm}`: must be in {supported_algorithms}')

    def sort(self, algorithm='topological'):
        supported_algorithms = ['topological']
        if algorithm == 'topological':
            return nx.topological_sort(self._g)
        else:
            raise ValueError(f'Unknown algorithm `{algorithm}`: must be in {supported_algorithms}')

    def __len__(self):
        return len(self._g.nodes)


class WorkflowBuilder(WorkflowBase):
    """Builder class for Workflow"""

    def __init__(self, workflow=None, tasks=None, name=None):
        if workflow:
            self._g = workflow._g.copy()
            self.name = workflow.name
        else:
            self._g = nx.DiGraph()
            self.name = name
        if tasks:
            for task in tasks:
                self.add_task(task)

    def add_task(self, task: Task, predecessors: Optional[Union[Task, list[Task]]] = None):
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

    def replace_task(self, task: Task, new_task: Task):
        """Replace a task with a new task

        Parameters
        ----------
        task : Task
            Task to replace
        new_task : Task
            New task
        """
        mapping = {task: new_task}
        nx.relabel_nodes(self._g, mapping, copy=False)

    def insert_workflow(
        self, other: Workflow, predecessors: Optional[Union[Task, list[Task]]] = None
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

    def __add__(self, other: Workflow):
        wb_new = WorkflowBuilder()
        wb_new._g = nx.compose(self._g, other._g)
        wb_new.name = self.name
        return wb_new


class Workflow(WorkflowBase, Generic[T], Immutable):
    """Workflow class

    Representation of a directed acyclic graph with Tasks as nodes and
    the flow of parameters as edges.

    Parameters
    ----------
    tasks : list
        List of tasks for initialization
    """

    def __init__(self, builder=None, graph=None, name: Optional[str] = None):
        if builder is not None:
            self._g = nx.freeze(builder._g.copy())
            self._name = builder.name
        elif graph is not None:
            self._g = graph
            self._name = name

    @classmethod
    def create(cls, builder=None, graph=None, name=None):
        return cls(builder=builder, graph=graph, name=name)

    def replace(self, **kwargs):
        builder = kwargs.get('builder', None)
        if builder is None:
            g = kwargs.get('graph', self._g)
            name = kwargs.get('name', self._name)
        else:
            g = None
            name = None
        return Workflow.create(builder=builder, graph=g, name=name)

    def as_dask_dict(self):
        """Create a dask workflow dictionary

        Returns
        -------
        dict
            Dask workflow dictionary
        """
        ids = self.get_id_mapping()

        if len(self.output_tasks) == 1:
            ids[self.output_tasks[0]] = 'results'
        else:
            raise ValueError("Workflow can only have one output task")

        as_dict = {}
        for task in self.traverse(algorithm='dfs'):
            key = ids[task]
            input_list = list(task.task_input)
            input_list.extend(ids[t] for t in self.get_predecessors(task))
            value = (task.function, *input_list)
            as_dict[key] = value
        return as_dict

    def get_id_mapping(self) -> dict[Task, str]:
        """Create a dictionary of task and a unique name"""
        ids = {}
        for task in self._g.nodes():
            assert isinstance(task, Task)
            ids[task] = f'{task.name}-{uuid.uuid4()}'
        return ids

    @property
    def name(self):
        """Name of Workflow"""
        return self._name

    def plot_dask(self, filename: str):
        """Save a visualization of workflow to file

        Parameters
        ----------
        filename : str
            Path
        """
        from dask import visualize  # pyright: ignore [reportPrivateImportUsage]

        visualize(self.as_dask_dict(), filename=filename, collapse_outputs=True)

    def __str__(self):
        return '\n'.join([str(task) for task in self._g])

    def __add__(self, other: Workflow):
        g = nx.compose(self._g, other._g)
        wf_new = Workflow(graph=g)
        return wf_new


def insert_context(wb: WorkflowBuilder, context):
    """Insert context for all tasks in a workflow needing it

    having context as first argument of function
    """
    for task in wb.tasks:
        parameters = tuple(inspect.signature(task.function).parameters)
        if parameters and parameters[0] == 'context':
            new_task = task.replace(task_input=(context, *task.task_input))
            wb.replace_task(task, new_task)
